// dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
// Phase 2 Attempt 32 (Opt 3+4 bundled) — float4 lane layout for STG.E.128 scratch +
// LDG.E.64 SMEM Kc unpack. Lane l now owns 4 contiguous fp32 per stripe (was 2 per
// stripe in A24's interleaved float2 layout). All six load/store sites updated
// CONSISTENTLY: Q load (qn_warp), Kc unpack in Pass 1, scratch_o STG, reduce LDG of
// scratch_o, final bf16 output write. Expected: -3 to -7% mean wall (Opt 3 attacks
// 25/46 of all lg_throttle samples by halving STG count; Opt 4 halves SMEM LDG count
// on the inner-loop Kc unpack). Coupled per CLAUDE.md §10 — neither alone is
// implementable without SHFL repack overhead that dilutes the gain.
// Phase 2 Attempt 24 — whole-split empty-sentinel hoist on top of A23.
//   If a CTA's slice of sparse_indices is all -1, skip the entire TMA+compute
//   loop. Running state (row_max=-inf, row_sum=0, o_acc=0) already produces
//   the correct scratch defaults consumed by the reduce phase.
// Phase 2 Attempt 23 (P3) — L2 evict_first cache hint on KV cp.async.bulk on top of A22.
//
// Two-pass design:
//   Pass 1 (sparse_attention_split_kernel): grid (S, N), one CTA per (split, token).
//     Each CTA processes a contiguous slice of the 2048 sparse indices, computes
//     online softmax in registers, and writes (m_partial, l_partial, o_partial)
//     to torch-allocated scratch tensors.
//   Pass 2 (sparse_attention_reduce_kernel): grid (N), one CTA per token.
//     Combines S partials per head via the standard FlashAttention reduction
//     (m_global = max m_s; l_global = Σ l_s · exp(m_s − m_global);
//      o_global = Σ o_s · exp(m_s − m_global) / l_global) and writes the final
//     bf16 output and fp32 2-base lse.
//
// Sentinel rule (CLAUDE.md §1): a token whose 2048 indices are all -1 must end
// with output=0, lse=-inf. Preserved here because every split sees only -1, so
// every l_partial = 0 and every m_partial = -inf → l_global = 0 → reduction
// short-circuits to (0, -inf).

#include <cstdint>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

constexpr int NUM_QO_HEADS       = 16;
constexpr int HEAD_DIM_CKV       = 512;
constexpr int HEAD_DIM_KPE       = 64;
constexpr int PAGE_SIZE          = 64;
constexpr int TOPK               = 2048;

constexpr int THREADS_PER_BLOCK  = 512;
constexpr int WARPS_PER_BLOCK    = THREADS_PER_BLOCK / 32;
static_assert(WARPS_PER_BLOCK == NUM_QO_HEADS, "one warp per head");

constexpr int CKV_PER_LANE       = HEAD_DIM_CKV / 32;     // 16
constexpr int KPE_PER_LANE       = HEAD_DIM_KPE / 32;     // 2

constexpr int MAX_SPLITS         = 64;                    // launch-time cap

constexpr float LOG2_E           = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// ---- TMA bulk + mbarrier helpers (raw PTX; CLAUDE.md GT-7). ----
__device__ __forceinline__ uint32_t s2int(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(p)));
}
__device__ __forceinline__ void mbarrier_init(uint64_t* m, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :: "r"(s2int(m)), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* m, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                 :: "r"(s2int(m)), "r"(tx_bytes));
}
__device__ __forceinline__ void cp_async_bulk_g2s(
    void* smem_dst, const void* global_src, uint32_t size_bytes, uint64_t* mbar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :: "r"(s2int(smem_dst)), "l"(global_src), "r"(size_bytes), "r"(s2int(mbar)));
}
// Variant with .L2::cache_hint — pass an L2 policy (created via createpolicy.fractional).
__device__ __forceinline__ void cp_async_bulk_g2s_evict_first(
    void* smem_dst, const void* global_src, uint32_t size_bytes,
    uint64_t* mbar, uint64_t cache_policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;\n"
        :: "r"(s2int(smem_dst)), "l"(global_src), "r"(size_bytes),
           "r"(s2int(mbar)), "l"(cache_policy));
}
// Build an "all 100% evict-first" L2 cache policy for streaming KV reads.
__device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t pol;
    asm volatile(
        "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"
        : "=l"(pol));
    return pol;
}
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* m, uint32_t parity) {
    asm volatile(
        "{\n"
        "    .reg .pred P;\n"
        "L_WAIT%=:\n"
        "    mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "    @P bra L_DONE%=;\n"
        "    bra L_WAIT%=;\n"
        "L_DONE%=:\n"
        "}\n"
        :: "r"(s2int(m)), "r"(parity));
}

// Inner-loop tile geometry.
constexpr int K_PER_ITER         = 4;
constexpr int CP_BUFFERS         = 3;                              // 3-stage prefetch
constexpr int CP_PREFETCH_AHEAD  = CP_BUFFERS - 1;                 // = 2
constexpr int CP_KC_BF16_PER_BUF = K_PER_ITER * HEAD_DIM_CKV;
constexpr int CP_KP_BF16_PER_BUF = K_PER_ITER * HEAD_DIM_KPE;
constexpr int CP_TOTAL_BF16      = CP_BUFFERS * (CP_KC_BF16_PER_BUF + CP_KP_BF16_PER_BUF);
constexpr int CP_SMEM_BYTES      = CP_TOTAL_BF16 * 2;             // ~14 KB
constexpr int CP_BATCH_BYTES     = K_PER_ITER * (HEAD_DIM_CKV + HEAD_DIM_KPE) * 2;  // 4608 B/expect_tx
constexpr int SMEM_IDX_CAP       = 128;                            // covers max kv_per_split for our N≤8 workloads

// Issue all KV loads for one batch via cp.async.bulk; called by ALL threads
// but only thread 0 executes the issuance + mbarrier arrive.
// Reads sparse indices from a CTA-local SMEM cache (smem_idx) populated at CTA entry.
// smem_idx is sentinel-padded with -1 for slots beyond the actual slice; this lets
// us drop bound checks here.
__device__ __forceinline__ void issue_load_batch_tma(
    int                                k_batch_local_start,    // local offset into smem_idx
    int                                tid,
    const int32_t*                     smem_idx,               // [SMEM_IDX_CAP], SMEM-resident
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    __nv_bfloat16*                     smem_kc_buf,
    __nv_bfloat16*                     smem_kp_buf,
    uint64_t*                          mbar,
    uint64_t                           kv_l2_policy)
{
    if (tid == 0) {
        mbarrier_arrive_expect_tx(mbar, CP_BATCH_BYTES);
        #pragma unroll
        for (int i = 0; i < K_PER_ITER; ++i) {
            int local = k_batch_local_start + i;
            int32_t kv_idx = smem_idx[local];      // -1 sentinel for invalid/tail
            int32_t safe = (kv_idx >= 0) ? kv_idx : 0;
            cp_async_bulk_g2s_evict_first(
                smem_kc_buf + i * HEAD_DIM_CKV,
                ckv_cache + (int64_t)safe * HEAD_DIM_CKV,
                HEAD_DIM_CKV * sizeof(__nv_bfloat16),
                mbar, kv_l2_policy);
            cp_async_bulk_g2s_evict_first(
                smem_kp_buf + i * HEAD_DIM_KPE,
                kpe_cache + (int64_t)safe * HEAD_DIM_KPE,
                HEAD_DIM_KPE * sizeof(__nv_bfloat16),
                mbar, kv_l2_policy);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1 — per-(split, token) CTA: online softmax over its slice of 2048 KV
// indices, emit (m, l, o) partials to scratch.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void sparse_attention_split_kernel(
    const __nv_bfloat16* __restrict__ q_nope,         // [N, 16, 512]
    const __nv_bfloat16* __restrict__ q_pe,           // [N, 16, 64]
    const __nv_bfloat16* __restrict__ ckv_cache,      // flat [P*64, 512]
    const __nv_bfloat16* __restrict__ kpe_cache,      // flat [P*64, 64]
    const int32_t*       __restrict__ sparse_indices, // [N, 2048]
    float                              sm_scale,
    float*               __restrict__ scratch_m,      // [N, S]
    float*               __restrict__ scratch_l,      // [N, S]
    float*               __restrict__ scratch_o,      // [N, S, 16, 512]
    int*                 __restrict__ completion_counters, // [N], pre-zeroed (O4 fused reduce)
    __nv_bfloat16*       __restrict__ output,         // [N, 16, 512] (O4 fused reduce)
    float*               __restrict__ lse,            // [N, 16]      (O4 fused reduce)
    int                                num_tokens,
    int                                splits_per_token,
    int                                kv_per_split)
{
    const int s = blockIdx.x;        // split index   ∈ [0, S)
    const int t = blockIdx.y;        // token index   ∈ [0, N)
    if (t >= num_tokens || s >= splits_per_token) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;    // 0..15 — query head index
    const int lane_id = tid & 31;

    const int k_start = s * kv_per_split;
    const int k_end   = min(k_start + kv_per_split, TOPK);

    const __nv_bfloat16* qn_t  = q_nope + t * NUM_QO_HEADS * HEAD_DIM_CKV;
    const __nv_bfloat16* qp_t  = q_pe   + t * NUM_QO_HEADS * HEAD_DIM_KPE;
    const int32_t*       idx_t = sparse_indices + t * TOPK;

    // SMEM layout for cp.async.bulk 3-buffer prefetch KV tile.
    extern __shared__ __nv_bfloat16 smem_raw[];
    __nv_bfloat16* smem_kc[CP_BUFFERS];
    __nv_bfloat16* smem_kp[CP_BUFFERS];
    #pragma unroll
    for (int b = 0; b < CP_BUFFERS; ++b) {
        smem_kc[b] = smem_raw + b * CP_KC_BF16_PER_BUF;
        smem_kp[b] = smem_raw + CP_BUFFERS * CP_KC_BF16_PER_BUF + b * CP_KP_BF16_PER_BUF;
    }
    __shared__ alignas(8) uint64_t mbar[CP_BUFFERS];
    __shared__ int32_t smem_idx[SMEM_IDX_CAP];
    if (tid == 0) {
        #pragma unroll
        for (int b = 0; b < CP_BUFFERS; ++b) mbarrier_init(&mbar[b], 1);
    }
    // L2 cache policy for KV reads: 100% evict_first (KV rows are read once per CTA,
    // never reused — keeping L2 capacity available for scratch_o + Q in reduce phase).
    const uint64_t kv_l2_policy = make_l2_evict_first_policy();
    // Cooperatively pre-load this CTA's idx slice into SMEM. Sentinel-pad with -1
    // for slots beyond the actual slice (last-split case where actual < kv_per_split).
    // Also compute any_valid flag (are any slots non-sentinel?) for whole-split skip.
    __shared__ int smem_any_valid;
    if (tid == 0) smem_any_valid = 0;
    __syncthreads();
    {
        const int actual = k_end - k_start;
        int local_any = 0;
        for (int i = tid; i < SMEM_IDX_CAP; i += THREADS_PER_BLOCK) {
            int32_t v = (i < actual) ? idx_t[k_start + i] : -1;
            smem_idx[i] = v;
            if (v >= 0) local_any = 1;
        }
        if (local_any) atomicOr(&smem_any_valid, 1);
    }
    __syncthreads();

    // Per-warp running state in registers.
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[CKV_PER_LANE];
    #pragma unroll
    for (int i = 0; i < CKV_PER_LANE; ++i) o_acc[i] = 0.0f;

    // A32: float4-friendly lane layout. Lane l owns Q elements at
    // {4*l + b + 128*j' : j'∈[0,4), b∈{0,1,2,3}}, packed as 4 LDG.64 (= 2 bf162) per lane.
    float qn_local[CKV_PER_LANE];
    {
        const __nv_bfloat16* qn_warp = qn_t + warp_id * HEAD_DIM_CKV;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            // 8 bytes per lane = 2 bf162 = 4 bf16 widened to 4 fp32.
            int2 packed = *reinterpret_cast<const int2*>(qn_warp + 128 * j + 4 * lane_id);
            __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
            __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
            qn_local[4 * j + 0] = __bfloat162float(__low2bfloat16(v0));
            qn_local[4 * j + 1] = __bfloat162float(__high2bfloat16(v0));
            qn_local[4 * j + 2] = __bfloat162float(__low2bfloat16(v1));
            qn_local[4 * j + 3] = __bfloat162float(__high2bfloat16(v1));
        }
    }
    // Q_pe layout unchanged (per-lane stride-2 already gives 32 distinct banks via LDG.32).
    float qp_local[KPE_PER_LANE];
    {
        const __nv_bfloat16* qp_lane =
            qp_t + warp_id * HEAD_DIM_KPE + lane_id * KPE_PER_LANE;
        __nv_bfloat162 qp_v = *reinterpret_cast<const __nv_bfloat162*>(qp_lane);
        qp_local[0] = __bfloat162float(__low2bfloat16(qp_v));
        qp_local[1] = __bfloat162float(__high2bfloat16(qp_v));
    }

    // ---- Inner loop: cp.async.bulk + mbarrier 3-buffer prefetch ----
    // Prefetch CP_PREFETCH_AHEAD = 2 batches ahead in the prologue, then
    // issue 1 batch per loop iteration. Each buffer's parity toggles each
    // time it is consumed.
    // Whole-split skip: if no slot in this CTA's slice is valid, bypass the
    // entire TMA+compute loop. Running state already initialized to sentinel
    // (row_max=-inf, row_sum=0, o_acc=0) — downstream writes produce the
    // correct scratch defaults for the reduce phase.
    int total_iters = (k_end - k_start + K_PER_ITER - 1) / K_PER_ITER;
    if (total_iters > 0 && smem_any_valid) {
        uint32_t phase[CP_BUFFERS] = {0, 0, 0};

        // Prologue: prime CP_PREFETCH_AHEAD batches if available. Bound only by
        // actual slice length — the SMEM_IDX_CAP check is redundant given that
        // the launcher guarantees kv_per_split <= SMEM_IDX_CAP.
        #pragma unroll
        for (int p = 0; p < CP_PREFETCH_AHEAD; ++p) {
            int local_p = p * K_PER_ITER;
            if (local_p < (k_end - k_start)) {
                issue_load_batch_tma(local_p, tid, smem_idx,
                                     ckv_cache, kpe_cache,
                                     smem_kc[p], smem_kp[p], &mbar[p],
                                     kv_l2_policy);
            }
        }

        int k_curr = k_start;

        for (int it = 0; it < total_iters; ++it) {
            int consume_buf = it % CP_BUFFERS;
            int issue_buf   = (it + CP_PREFETCH_AHEAD) % CP_BUFFERS;
            int local_issue = (it + CP_PREFETCH_AHEAD) * K_PER_ITER;
            bool has_issue  = (local_issue < (k_end - k_start));

            // 1) Prefetch the batch CP_PREFETCH_AHEAD iters ahead.
            if (has_issue) {
                issue_load_batch_tma(local_issue, tid, smem_idx,
                                     ckv_cache, kpe_cache,
                                     smem_kc[issue_buf], smem_kp[issue_buf], &mbar[issue_buf],
                                     kv_l2_policy);
            }

            // 2) Wait for current batch's TMA to complete.
            mbarrier_wait_parity(&mbar[consume_buf], phase[consume_buf]);
            phase[consume_buf] ^= 1;
            int buf = consume_buf;   // alias for the rest of the loop body

            // 3) Per-lane SMEM → register unpack (A32 float4-friendly layout) + K logits.
            float kc_local[K_PER_ITER][CKV_PER_LANE];
            float logits[K_PER_ITER];
            #pragma unroll
            for (int i = 0; i < K_PER_ITER; ++i) {
                const __nv_bfloat16* kc_row = smem_kc[buf] + i * HEAD_DIM_CKV;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int2 packed = *reinterpret_cast<const int2*>(kc_row + 128 * j + 4 * lane_id);
                    __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&packed.x);
                    __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&packed.y);
                    kc_local[i][4 * j + 0] = __bfloat162float(__low2bfloat16(v0));
                    kc_local[i][4 * j + 1] = __bfloat162float(__high2bfloat16(v0));
                    kc_local[i][4 * j + 2] = __bfloat162float(__low2bfloat16(v1));
                    kc_local[i][4 * j + 3] = __bfloat162float(__high2bfloat16(v1));
                }
                const __nv_bfloat16* kp_lane =
                    smem_kp[buf] + i * HEAD_DIM_KPE + lane_id * KPE_PER_LANE;
                __nv_bfloat162 kp_v = *reinterpret_cast<const __nv_bfloat162*>(kp_lane);
                float kp0 = __bfloat162float(__low2bfloat16(kp_v));
                float kp1 = __bfloat162float(__high2bfloat16(kp_v));

                float partial = 0.0f;
                #pragma unroll
                for (int j = 0; j < CKV_PER_LANE; ++j) {
                    partial += qn_local[j] * kc_local[i][j];
                }
                partial += qp_local[0] * kp0;
                partial += qp_local[1] * kp1;
                float l = warp_reduce_sum(partial) * sm_scale;

                // Validity from sentinel-padded SMEM cache (no L2 trip).
                // local_idx ≤ kv_per_split + K_PER_ITER - 1 ≤ SMEM_IDX_CAP (launcher invariant).
                int local_idx = (k_curr - k_start) + i;
                bool valid = (smem_idx[local_idx] >= 0);
                logits[i] = valid ? l : -INFINITY;
            }

            // 4) Batched online-softmax rescale (same as A12).
            float batch_max = logits[0];
            #pragma unroll
            for (int i = 1; i < K_PER_ITER; ++i) batch_max = fmaxf(batch_max, logits[i]);

            float m_new     = fmaxf(row_max, batch_max);
            float scale_old = (row_max == -INFINITY) ? 0.0f : __expf(row_max - m_new);
            row_sum *= scale_old;
            #pragma unroll
            for (int j = 0; j < CKV_PER_LANE; ++j) o_acc[j] *= scale_old;

            if (m_new != -INFINITY) {
                float p_arr[K_PER_ITER];
                #pragma unroll
                for (int i = 0; i < K_PER_ITER; ++i) {
                    p_arr[i] = (logits[i] == -INFINITY) ? 0.0f : __expf(logits[i] - m_new);
                    row_sum += p_arr[i];
                }
                #pragma unroll
                for (int i = 0; i < K_PER_ITER; ++i) {
                    if (p_arr[i] == 0.0f) continue;
                    #pragma unroll
                    for (int j = 0; j < CKV_PER_LANE; ++j) {
                        o_acc[j] += p_arr[i] * kc_local[i][j];
                    }
                }
            }

            row_max = m_new;
            k_curr += K_PER_ITER;
        }
    }

    // A32: 4 STG.E.128 per warp via float4. Lane l owns o_acc[4*j+b] for output
    // position 128*j + 4*lane + b. Halves STG instruction count vs A24.
    float* scratch_warp = scratch_o
        + ((int64_t)t * splits_per_token + s) * NUM_QO_HEADS * HEAD_DIM_CKV
        + warp_id * HEAD_DIM_CKV;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4 v;
        v.x = o_acc[4 * j + 0];
        v.y = o_acc[4 * j + 1];
        v.z = o_acc[4 * j + 2];
        v.w = o_acc[4 * j + 3];
        *reinterpret_cast<float4*>(scratch_warp + 128 * j + 4 * lane_id) = v;
    }

    if (lane_id == 0) {
        int64_t ml_off = (int64_t)t * splits_per_token * NUM_QO_HEADS
                       + s * NUM_QO_HEADS
                       + warp_id;
        scratch_m[ml_off] = row_max;
        scratch_l[ml_off] = row_sum;
    }

    // ====================================================================
    // O4 fused reduce: last-arrival CTA per token does the reduction.
    // ====================================================================
    // Ensure all our scratch writes (scratch_m, scratch_l, scratch_o) are
    // device-visible before the atomic counter increment.
    __threadfence();

    __shared__ bool smem_i_am_last;
    if (tid == 0) {
        int prev = atomicAdd(&completion_counters[t], 1);
        smem_i_am_last = (prev == splits_per_token - 1);
    }
    __syncthreads();
    if (!smem_i_am_last) return;

    // ----- Reduce phase (only the last-arriving CTA per token) -----
    // (1 warp per head, 32 lanes per warp, same layout as the prior reduce kernel.)

    // Step 1: per-head global max across S splits.
    const float* mh_ptr = scratch_m + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + warp_id;
    const float* lh_ptr = scratch_l + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + warp_id;

    float m_local = -INFINITY;
    if (lane_id < splits_per_token) {
        m_local = mh_ptr[lane_id * NUM_QO_HEADS];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, m_local, off);
        m_local = fmaxf(m_local, v);
    }
    float m_global = m_local;

    // Step 2: per-split scale = exp(m_partial - m_global), and l_global.
    float scale_s = 0.0f;
    float l_local = 0.0f;
    if (lane_id < splits_per_token) {
        float m_s = mh_ptr[lane_id * NUM_QO_HEADS];
        float l_s = lh_ptr[lane_id * NUM_QO_HEADS];
        scale_s = (m_global == -INFINITY) ? 0.0f : __expf(m_s - m_global);
        l_local = l_s * scale_s;
    }
    float l_warp = l_local;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        l_warp += __shfl_xor_sync(0xffffffff, l_warp, off);
    }
    float l_global = l_warp;
    bool empty   = (l_global == 0.0f);
    float inv_l  = empty ? 0.0f : 1.0f / l_global;

    // Step 3: combine partial outputs into final o_global.
    float o_acc2[CKV_PER_LANE];
    #pragma unroll
    for (int i = 0; i < CKV_PER_LANE; ++i) o_acc2[i] = 0.0f;

    for (int s_iter = 0; s_iter < splits_per_token; ++s_iter) {
        float scale_b = __shfl_sync(0xffffffff, scale_s, s_iter);
        if (scale_b == 0.0f) continue;
        const float* sw = scratch_o
            + ((int64_t)t * splits_per_token + s_iter) * NUM_QO_HEADS * HEAD_DIM_CKV
            + warp_id * HEAD_DIM_CKV;
        float o_partial[CKV_PER_LANE];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 v = *reinterpret_cast<const float4*>(sw + 128 * j + 4 * lane_id);
            o_partial[4 * j + 0] = v.x;
            o_partial[4 * j + 1] = v.y;
            o_partial[4 * j + 2] = v.z;
            o_partial[4 * j + 3] = v.w;
        }
        #pragma unroll
        for (int i = 0; i < CKV_PER_LANE; ++i) {
            o_acc2[i] += scale_b * o_partial[i];
        }
    }

    // A32: final bf16 write — 4 int2 (= 2 bf162 packed) STGs per warp at the
    // matching float4-friendly layout.
    __nv_bfloat16* out_warp = output + t * NUM_QO_HEADS * HEAD_DIM_CKV
                            + warp_id * HEAD_DIM_CKV;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        __nv_bfloat162 p0 = __floats2bfloat162_rn(
            o_acc2[4 * j + 0] * inv_l, o_acc2[4 * j + 1] * inv_l);
        __nv_bfloat162 p1 = __floats2bfloat162_rn(
            o_acc2[4 * j + 2] * inv_l, o_acc2[4 * j + 3] * inv_l);
        int2 packed;
        packed.x = *reinterpret_cast<const int*>(&p0);
        packed.y = *reinterpret_cast<const int*>(&p1);
        *reinterpret_cast<int2*>(out_warp + 128 * j + 4 * lane_id) = packed;
    }
    if (lane_id == 0) {
        float lse_val = empty ? -INFINITY : (m_global * LOG2_E + log2f(l_global));
        lse[t * NUM_QO_HEADS + warp_id] = lse_val;
    }

    // P1: reset counter for next kernel launch (single tid suffices; CUDA
    // stream serialization makes this write visible to the next launch).
    if (tid == 0) {
        completion_counters[t] = 0;
    }
}

// ---------------------------------------------------------------------------
// Pass 2 — per-token CTA (1 warp per head): combine S partials, write final
// output (bf16) and lse (fp32 2-base). [O4: now unused; kept for reference.]
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void sparse_attention_reduce_kernel(
    const float*         __restrict__ scratch_m,      // [N, S, 16]
    const float*         __restrict__ scratch_l,      // [N, S, 16]
    const float*         __restrict__ scratch_o,      // [N, S, 16, 512]
    __nv_bfloat16*       __restrict__ output,         // [N, 16, 512]
    float*               __restrict__ lse,            // [N, 16]
    int                                num_tokens,
    int                                splits_per_token)
{
    const int t = blockIdx.x;
    if (t >= num_tokens) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;       // head
    const int lane_id = tid & 31;

    // Step 1: per-head global max (each warp owns one head).
    // Read S m_partial values for this head.
    const float* mh_ptr = scratch_m + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + warp_id;
    const float* lh_ptr = scratch_l + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + warp_id;

    float m_local = -INFINITY;
    // S ≤ 16; spread across the 32 lanes (most lanes inactive on the max scan).
    if (lane_id < splits_per_token) {
        m_local = mh_ptr[lane_id * NUM_QO_HEADS];
    }
    // Warp reduce max.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float v = __shfl_xor_sync(0xffffffff, m_local, off);
        m_local = fmaxf(m_local, v);
    }
    float m_global = m_local;        // broadcast across warp via shfl_xor

    // Step 2: per-split scale = exp(m_partial - m_global). Compute all S scales
    // (each lane handles one split if lane < S; later we broadcast as needed).
    // Also compute l_global = Σ l_s · scale_s.
    float scale_s = 0.0f;
    float l_local = 0.0f;
    if (lane_id < splits_per_token) {
        float m_s = mh_ptr[lane_id * NUM_QO_HEADS];
        float l_s = lh_ptr[lane_id * NUM_QO_HEADS];
        // If a split was empty (m_s = -inf, l_s = 0), exp(-inf - m_global) is 0
        // when m_global > -inf and the contribution vanishes. If m_global is
        // also -inf (all splits empty), we explicitly emit 0.
        scale_s = (m_global == -INFINITY) ? 0.0f : __expf(m_s - m_global);
        l_local = l_s * scale_s;
    }
    float l_warp = l_local;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        l_warp += __shfl_xor_sync(0xffffffff, l_warp, off);
    }
    float l_global = l_warp;
    bool  empty    = (l_global == 0.0f);
    float inv_l    = empty ? 0.0f : 1.0f / l_global;

    // Step 3: combine outputs.
    // Per-lane slice: 16 contiguous output features.
    // o_global[d] = (Σ_s o_partial[s, d] · scale_s) / l_global
    float o_acc[CKV_PER_LANE];
    #pragma unroll
    for (int i = 0; i < CKV_PER_LANE; ++i) o_acc[i] = 0.0f;

    // We need scale_s for s = 0..S-1 on every lane. Use broadcast via shfl.
    // scale_s currently lives on lane=s for s<S; gather to per-lane regs.
    for (int s = 0; s < splits_per_token; ++s) {
        float scale_b = __shfl_sync(0xffffffff, scale_s, s);
        if (scale_b == 0.0f) continue;        // empty split contributes nothing

        const float* scratch_warp = scratch_o
            + ((int64_t)t * splits_per_token + s) * NUM_QO_HEADS * HEAD_DIM_CKV
            + warp_id * HEAD_DIM_CKV;
        // INTERLEAVED layout matching the split kernel write: 8 LDG.64 per warp.
        float o_partial[CKV_PER_LANE];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 v = *reinterpret_cast<const float2*>(scratch_warp + 64 * j + 2 * lane_id);
            o_partial[2 * j + 0] = v.x;
            o_partial[2 * j + 1] = v.y;
        }
        #pragma unroll
        for (int i = 0; i < CKV_PER_LANE; ++i) {
            o_acc[i] += scale_b * o_partial[i];
        }
    }

    // Final write: output[t, head, 64*j + 2*lane + b] in bf16, INTERLEAVED.
    // 8 STG.32 per warp (32 lanes × 4 bytes = 128 contiguous bytes per STG).
    __nv_bfloat16* out_warp = output + t * NUM_QO_HEADS * HEAD_DIM_CKV
                            + warp_id * HEAD_DIM_CKV;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat162 packed = __floats2bfloat162_rn(
            o_acc[2 * j + 0] * inv_l,
            o_acc[2 * j + 1] * inv_l);
        *reinterpret_cast<__nv_bfloat162*>(out_warp + 64 * j + 2 * lane_id) = packed;
    }

    // lse: per head, lane 0 writes.
    if (lane_id == 0) {
        float lse_val = empty
            ? -INFINITY
            : (m_global * LOG2_E + log2f(l_global));
        lse[t * NUM_QO_HEADS + warp_id] = lse_val;
    }
}

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------
static int choose_splits(int num_tokens) {
    // Target ~132 active CTAs (B200 SM count); cap at MAX_SPLITS to keep
    // reduction cheap. For N=1: S=16; N=8: S=16; N=16: S=8; N=32: S=4.
    int s = 132 / (num_tokens > 0 ? num_tokens : 1);
    if (s < 1) s = 1;
    if (s > MAX_SPLITS) s = MAX_SPLITS;
    return s;
}

// ---- P1: persistent counter buffer (CLAUDE.md GT-16 inter-kernel scratch). ----
// Lazy-grown via cudaMalloc; kernel-side resets it to 0 so subsequent calls
// don't pay a cudaMemsetAsync. First-time allocation is zero-initialized once.
static int*    g_counters          = nullptr;
static int     g_counters_capacity = 0;

static void ensure_counters(int n, cudaStream_t stream) {
    if (n > g_counters_capacity) {
        if (g_counters) cudaFree(g_counters);
        int new_cap = (n > g_counters_capacity * 2) ? n : g_counters_capacity * 2;
        if (new_cap < 64) new_cap = 64;
        cudaMalloc(&g_counters, new_cap * sizeof(int));
        cudaMemsetAsync(g_counters, 0, new_cap * sizeof(int), stream);
        g_counters_capacity = new_cap;
    }
}

// ---- P2: consolidated scratch buffer (m + l + o in one allocation). ----
static float*   g_scratch          = nullptr;
static int64_t  g_scratch_capacity = 0;

static void ensure_scratch(int64_t fp32_count, cudaStream_t /*stream*/) {
    if (fp32_count > g_scratch_capacity) {
        if (g_scratch) cudaFree(g_scratch);
        int64_t new_cap = (fp32_count > g_scratch_capacity * 2) ? fp32_count : g_scratch_capacity * 2;
        cudaMalloc(&g_scratch, new_cap * sizeof(float));
        g_scratch_capacity = new_cap;
    }
}

void launch_sparse_attention_c(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse)
{
    const at::cuda::CUDAGuard device_guard(q_nope.device());
    const int num_tokens = static_cast<int>(q_nope.size(0));
    if (num_tokens == 0) return;

    const int splits_per_token = choose_splits(num_tokens);
    const int kv_per_split     = (TOPK + splits_per_token - 1) / splits_per_token;
    // Kernel invariant: per-CTA slice fits in smem_idx[SMEM_IDX_CAP].
    TORCH_CHECK(kv_per_split <= SMEM_IDX_CAP,
                "kv_per_split (", kv_per_split, ") exceeds SMEM_IDX_CAP (", SMEM_IDX_CAP, ")");

    // P1+P2: persistent device buffers (per CLAUDE.md GT-16 inter-kernel scratch).
    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_counters(num_tokens, stream);

    const int64_t ml_count = (int64_t)num_tokens * splits_per_token * NUM_QO_HEADS;
    const int64_t o_count  = ml_count * HEAD_DIM_CKV;
    const int64_t total_scratch = ml_count * 2 + o_count;
    ensure_scratch(total_scratch, stream);

    float* scratch_m = g_scratch;
    float* scratch_l = scratch_m + ml_count;
    float* scratch_o = scratch_l + ml_count;

    dim3 grid_split(splits_per_token, num_tokens);
    sparse_attention_split_kernel<<<grid_split, THREADS_PER_BLOCK, CP_SMEM_BYTES, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
        sparse_indices.data_ptr<int32_t>(),
        static_cast<float>(sm_scale),
        scratch_m,
        scratch_l,
        scratch_o,
        g_counters,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        lse.data_ptr<float>(),
        num_tokens,
        splits_per_token,
        kv_per_split);
    // O4: no separate reduce kernel — it's fused into the split kernel above.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sparse_attention_c", &launch_sparse_attention_c,
          "DSA Sparse Attention (Phase 2 A2: 2-pass split-K + reduction)");
}
