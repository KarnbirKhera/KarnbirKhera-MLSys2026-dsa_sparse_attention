// dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
// Phase 2 Attempt T3-568 (C1a) — tcgen05 MMA + TMA-staged KV with 8×T rewrite.
// Builds on T3-567d (16-warp tcgen05). Adds per-row cp.async.bulk for KV into
// a linear staging region, double-buffered, with SMEM→SMEM rewrite into 8×T
// before each MMA. Prefetch 1 iter ahead.
//
// Fixes the insufficient-output-coverage bug of 4-warp design: now 16 warps
// (one per head) hold the full per-head 512-element output in registers,
// matching A23's pattern. MMA is done by warps 0-3 only (cta_group::1 uses
// one warpgroup); the result is broadcast via SMEM to warps 0-15 for softmax+PV.
//
// GTs used:
//   GT-11: 8×T core-tile, SBO=256, LBO=128 per K-step (byte-level).
//   GT-12: warp 0 lane L<16 → M-row L for kind::f8f6f4 M=64 cta_group::1
//          (we apply the same pattern for kind::f16, confirmed empirically).
//   GT-14: don't unroll K-slab loop.
//   GT-15: transpose_A=0, transpose_B=0 for K-major 8×T.
//   GT-17: all 32 lanes issue tcgen05.ld.
//   GT-18: tcgen05.commit uses .mbarrier::arrive::one.shared::cluster.b64.

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
constexpr int HEAD_DIM_TOT       = HEAD_DIM_CKV + HEAD_DIM_KPE;  // 576
constexpr int TOPK               = 2048;

constexpr int MMA_M              = 64;
constexpr int MMA_N              = 16;
constexpr int MMA_K              = 16;
constexpr int K_TILES            = HEAD_DIM_TOT / MMA_K;  // 36
constexpr int KV_PER_ITER_MMA    = MMA_N;  // 16

constexpr int THREADS_PER_BLOCK  = 512;    // 16 warps — one per head
constexpr int WARPS_PER_BLOCK_MMA_FILL = 16;
constexpr int MAX_SPLITS         = 64;
constexpr int SMEM_IDX_CAP       = 128;
constexpr float LOG2_E           = 1.4426950408889634f;

__device__ __forceinline__ uint32_t s2int(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(p)));
}

// SMEM descriptor per ISA Table 42.
__device__ __forceinline__ uint64_t make_smem_desc(
    uint32_t smem_addr_bytes, uint32_t lbo_bytes, uint32_t sbo_bytes)
{
    uint64_t d = 0;
    d |= ((uint64_t)((smem_addr_bytes & 0x3FFFF) >> 4));
    d |= (((uint64_t)((lbo_bytes & 0x3FFFF) >> 4))) << 16;
    d |= (((uint64_t)((sbo_bytes & 0x3FFFF) >> 4))) << 32;
    d |= ((uint64_t)0b001) << 46;
    d |= ((uint64_t)((smem_addr_bytes >> 7) & 0x7)) << 49;
    return d;
}

// IDESC per ISA Table 44. kind::f16 w/ BF16 operands, F32 accumulator.
__device__ __forceinline__ uint32_t make_idesc(int n, int m) {
    uint32_t d = 0;
    d |= (uint32_t)1 << 4;    // dtype D = F32
    d |= (uint32_t)1 << 7;    // atype = BF16
    d |= (uint32_t)1 << 10;   // btype = BF16
    // transpose_A=0, transpose_B=0 per GT-15
    d |= ((uint32_t)(n >> 3) & 0x3F) << 17;
    d |= ((uint32_t)(m >> 4) & 0x1F) << 24;
    return d;
}

// ---- tcgen05 ----
__device__ __forceinline__ void tcgen05_alloc_1(uint32_t smem_dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_dealloc_1(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tcgen05_mma_f16_1(
    uint32_t taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int en_d)
{
    asm volatile(
        "{ .reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(en_d));
}
__device__ __forceinline__ void tcgen05_commit_1(uint32_t mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::);
}
__device__ __forceinline__ void tcgen05_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::);
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::);
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x16(uint32_t taddr, float* d) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3]),"=f"(d[4]),"=f"(d[5]),
          "=f"(d[6]),"=f"(d[7]),"=f"(d[8]),"=f"(d[9]),"=f"(d[10]),"=f"(d[11]),
          "=f"(d[12]),"=f"(d[13]),"=f"(d[14]),"=f"(d[15])
        : "r"(taddr));
}

// ---- mbarrier / TMA ----
__device__ __forceinline__ void mbarrier_init(uint64_t* m, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(s2int(m)), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* m, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(s2int(m)), "r"(tx));
}
__device__ __forceinline__ void cp_async_bulk_g2s(
    void* dst, const void* src, uint32_t nb, uint64_t* mbar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(s2int(dst)), "l"(src), "r"(nb), "r"(s2int(mbar)));
}
// Variant with L2 evict_first hint (GT-38 pattern from A23).
__device__ __forceinline__ void cp_async_bulk_g2s_evict_first(
    void* dst, const void* src, uint32_t nb, uint64_t* mbar, uint64_t cache_policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;\n"
        :: "r"(s2int(dst)), "l"(src), "r"(nb), "r"(s2int(mbar)), "l"(cache_policy));
}
__device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* m, uint32_t parity) {
    asm volatile(
        "{ .reg .pred P;\n"
        "L_WAIT%=:\n"
        "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "  @P bra L_DONE%=;\n"
        "  bra L_WAIT%=;\n"
        "L_DONE%=: }\n" :: "r"(s2int(m)), "r"(parity));
}

constexpr int SMEM_Q_BYTES         = MMA_M * HEAD_DIM_TOT * 2;   // 73728
constexpr int SMEM_KV_BYTES        = MMA_N * HEAD_DIM_TOT * 2;   // 18432 (8×T, one buffer)
constexpr int SMEM_KV_LINEAR_BYTES = MMA_N * HEAD_DIM_TOT * 2;   // 18432 (linear, per buffer)
constexpr int KV_BUFFERS           = 3;                          // triple-buffer linear staging
constexpr int SMEM_LOGITS_BYTES    = NUM_QO_HEADS * MMA_N * 4;   // 1024

// 8×T layout addresses, parametric on K-step stride.
// q_addr (M=64): K-step stride 2048.
// kv_addr (N=16): K-step stride 512.
__device__ __forceinline__ int q_addr(int m, int k_byte) {
    int k_step     = k_byte >> 5;
    int kb_in_step = k_byte & 31;
    return k_step * 2048
         + (m >> 3) * 256
         + (kb_in_step >> 4) * 128
         + (m & 7) * 16
         + (kb_in_step & 15);
}
__device__ __forceinline__ int kv_addr(int n, int k_byte) {
    int k_step     = k_byte >> 5;
    int kb_in_step = k_byte & 31;
    return k_step * 512
         + (n >> 3) * 256
         + (kb_in_step >> 4) * 128
         + (n & 7) * 16
         + (kb_in_step & 15);
}

constexpr int CKV_PER_LANE = HEAD_DIM_CKV / 32;  // 16, stride-32 per lane

__global__ __launch_bounds__(THREADS_PER_BLOCK, 1)
void sparse_attention_mma_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_cache,
    const __nv_bfloat16* __restrict__ kpe_cache,
    const int32_t*       __restrict__ sparse_indices,
    float                              sm_scale,
    float*               __restrict__ scratch_m,
    float*               __restrict__ scratch_l,
    float*               __restrict__ scratch_o,
    int*                 __restrict__ completion_counters,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ lse,
    int                                num_tokens,
    int                                splits_per_token,
    int                                kv_per_split)
{
    const int s = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= num_tokens || s >= splits_per_token) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;       // 0..15, head index
    const int lane_id = tid & 31;

    const int k_start = s * kv_per_split;
    const int k_end   = min(k_start + kv_per_split, TOPK);

    const __nv_bfloat16* qn_t  = q_nope + t * NUM_QO_HEADS * HEAD_DIM_CKV;
    const __nv_bfloat16* qp_t  = q_pe   + t * NUM_QO_HEADS * HEAD_DIM_KPE;
    const int32_t*       idx_t = sparse_indices + t * TOPK;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* pr = smem_raw;
    __nv_bfloat16* smem_q      = reinterpret_cast<__nv_bfloat16*>(pr); pr += SMEM_Q_BYTES;
    __nv_bfloat16* smem_kv_mma = reinterpret_cast<__nv_bfloat16*>(pr); pr += SMEM_KV_BYTES;
    __nv_bfloat16* smem_kv_lin[KV_BUFFERS];
    for (int b = 0; b < KV_BUFFERS; ++b) {
        smem_kv_lin[b] = reinterpret_cast<__nv_bfloat16*>(pr);
        pr += SMEM_KV_LINEAR_BYTES;
    }
    float*         smem_logits = reinterpret_cast<float*>(pr); pr += SMEM_LOGITS_BYTES;
    pr = (uint8_t*)(((uintptr_t)pr + 7) & ~7ULL);
    uint64_t* mbar_mma    = reinterpret_cast<uint64_t*>(pr); pr += 8;
    uint64_t* mbar_kv     = reinterpret_cast<uint64_t*>(pr); pr += 8 * KV_BUFFERS;
    pr = (uint8_t*)(((uintptr_t)pr + 3) & ~3ULL);
    int32_t*  smem_idx    = reinterpret_cast<int32_t*>(pr); pr += SMEM_IDX_CAP * 4;
    uint32_t* tmem_slot   = reinterpret_cast<uint32_t*>(pr); pr += 4;

    if (tid == 0) {
        mbarrier_init(mbar_mma, 1);
        for (int b = 0; b < KV_BUFFERS; ++b) mbarrier_init(&mbar_kv[b], 1);
    }

    // Sparse indices with sentinel padding.
    {
        const int actual = k_end - k_start;
        for (int i = tid; i < SMEM_IDX_CAP; i += THREADS_PER_BLOCK) {
            smem_idx[i] = (i < actual) ? idx_t[k_start + i] : -1;
        }
    }

    // Zero-fill Q SMEM.
    #pragma unroll 4
    for (int i = tid; i < SMEM_Q_BYTES / 16; i += THREADS_PER_BLOCK) {
        reinterpret_cast<int4*>(smem_q)[i] = make_int4(0, 0, 0, 0);
    }
    __syncthreads();

    // Fill real Q heads into 8×T layout.
    {
        const int total = NUM_QO_HEADS * (HEAD_DIM_TOT / 8);  // 16 * 72
        for (int pair = tid; pair < total; pair += THREADS_PER_BLOCK) {
            int h = pair / (HEAD_DIM_TOT / 8);
            int chunk = pair % (HEAD_DIM_TOT / 8);
            int kb_src = chunk * 16;
            int4 val;
            if (kb_src < HEAD_DIM_CKV * 2) {
                val = *reinterpret_cast<const int4*>(
                    reinterpret_cast<const char*>(qn_t + h * HEAD_DIM_CKV) + kb_src);
            } else {
                val = *reinterpret_cast<const int4*>(
                    reinterpret_cast<const char*>(qp_t + h * HEAD_DIM_KPE) + (kb_src - HEAD_DIM_CKV * 2));
            }
            int dst_byte = q_addr(h, kb_src);
            *reinterpret_cast<int4*>(reinterpret_cast<char*>(smem_q) + dst_byte) = val;
        }
    }

    // TMEM alloc (32 cols).
    if (warp_id == 0) tcgen05_alloc_1(s2int(tmem_slot), 32);
    __syncthreads();
    uint32_t tmem_acc = *tmem_slot;

    // A23-style per-head running state. Warp h owns head h.
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[CKV_PER_LANE];
    #pragma unroll
    for (int i = 0; i < CKV_PER_LANE; ++i) o_acc[i] = 0.0f;

    const int n_iters = (k_end - k_start + KV_PER_ITER_MMA - 1) / KV_PER_ITER_MMA;
    uint32_t phase_mma = 0;
    uint32_t phase_kv[KV_BUFFERS] = {0, 0};

    // ---- TMA issue helper ----
    // L2 evict_first policy: KV rows are single-use per CTA — don't pollute L2 (GT-38).
    const uint64_t kv_l2_policy = make_l2_evict_first_policy();
    auto issue_tma_batch = [&](int batch_base_local, int buf_idx) {
        if (tid == 0) {
            const uint32_t tx = MMA_N * HEAD_DIM_TOT * 2;
            mbarrier_arrive_expect_tx(&mbar_kv[buf_idx], tx);
            __nv_bfloat16* dst = smem_kv_lin[buf_idx];
            #pragma unroll 1
            for (int n = 0; n < MMA_N; ++n) {
                int local = batch_base_local + n;
                int32_t kv_idx = (local < (k_end - k_start)) ? smem_idx[local] : -1;
                int32_t safe = (kv_idx >= 0) ? kv_idx : 0;
                cp_async_bulk_g2s_evict_first(
                    dst + n * HEAD_DIM_TOT,
                    ckv_cache + (int64_t)safe * HEAD_DIM_CKV,
                    HEAD_DIM_CKV * 2, &mbar_kv[buf_idx], kv_l2_policy);
                cp_async_bulk_g2s_evict_first(
                    dst + n * HEAD_DIM_TOT + HEAD_DIM_CKV,
                    kpe_cache + (int64_t)safe * HEAD_DIM_KPE,
                    HEAD_DIM_KPE * 2, &mbar_kv[buf_idx], kv_l2_policy);
            }
        }
    };

    // Helper lambda for the 8×T rewrite from a linear buffer.
    // Uses all 16 warps (one of 3 rounds per warp).
    auto rewrite_lin_to_mma = [&](int buf_idx) {
        const __nv_bfloat16* lin = smem_kv_lin[buf_idx];
        int lane_hi = lane_id >> 3;
        int lane_lo = lane_id & 7;
        int n_local = (lane_hi & 2) * 4 + lane_lo;
        int k_tile_local = lane_hi & 1;
        #pragma unroll
        for (int round = 0; round < 3; ++round) {
            int k_step = warp_id + round * WARPS_PER_BLOCK_MMA_FILL;
            if (k_step >= K_TILES) break;
            int src_col_byte = k_step * 32 + k_tile_local * 16;
            int4 val = *reinterpret_cast<const int4*>(
                reinterpret_cast<const char*>(lin + n_local * HEAD_DIM_TOT) + src_col_byte);
            int dst_byte = k_step * 512 + lane_id * 16;
            *reinterpret_cast<int4*>(reinterpret_cast<char*>(smem_kv_mma) + dst_byte) = val;
        }
    };

    // Warp-specialized rewrite: only warps 4-15 do the rewrite (12 warps × 3 rounds = 36 slots).
    // Map: warp (L in 4..15, effective id = L-4 in 0..11) covers k_step = (L-4) + round*12.
    auto rewrite_lin_to_mma_warps_4_15 = [&](int buf_idx) {
        if (warp_id < 4) return;
        const __nv_bfloat16* lin = smem_kv_lin[buf_idx];
        int eff_wid = warp_id - 4;    // 0..11
        int lane_hi = lane_id >> 3;
        int lane_lo = lane_id & 7;
        int n_local = (lane_hi & 2) * 4 + lane_lo;
        int k_tile_local = lane_hi & 1;
        #pragma unroll
        for (int round = 0; round < 3; ++round) {
            int k_step = eff_wid + round * 12;
            if (k_step >= K_TILES) break;
            int src_col_byte = k_step * 32 + k_tile_local * 16;
            int4 val = *reinterpret_cast<const int4*>(
                reinterpret_cast<const char*>(lin + n_local * HEAD_DIM_TOT) + src_col_byte);
            int dst_byte = k_step * 512 + lane_id * 16;
            *reinterpret_cast<int4*>(reinterpret_cast<char*>(smem_kv_mma) + dst_byte) = val;
        }
    };

    // Helper for MMA issue (tid==0 only).
    auto issue_mma = [&]() {
        if (tid == 0) {
            uint64_t q_desc_base = make_smem_desc(s2int(smem_q),  128, 256);
            uint64_t k_desc_base = make_smem_desc(s2int(smem_kv_mma), 128, 256);
            uint32_t idesc = make_idesc(MMA_N, MMA_M);
            for (int kt = 0; kt < K_TILES; ++kt) {
                uint64_t q_desc = q_desc_base + (uint64_t)kt * 128;
                uint64_t k_desc = k_desc_base + (uint64_t)kt * 32;
                tcgen05_mma_f16_1(tmem_acc, q_desc, k_desc, idesc, kt == 0 ? 0 : 1);
            }
            tcgen05_commit_1(s2int(mbar_mma));
        }
    };

    // ---- TMA prologue: prefetch iter 0 and iter 1 ----
    for (int p = 0; p < KV_BUFFERS && p < n_iters; ++p) {
        issue_tma_batch(p * KV_PER_ITER_MMA, p);
    }

    // ---- MMA prologue: wait for iter 0's TMA, rewrite, issue MMA 0 ----
    if (n_iters > 0) {
        mbarrier_wait_parity(&mbar_kv[0], phase_kv[0]);
        phase_kv[0] ^= 1;
        rewrite_lin_to_mma(0);
        __syncthreads();
        issue_mma();
    }

    // ---- Main loop: C9 pipelined MMA. ----
    // Per iter k: wait MMA k → .ld → broadcast → [setup MMA k+1] → softmax+PV iter k
    // MMA k+1 runs async during iter k's softmax+PV.
    for (int it = 0; it < n_iters; ++it) {
        int batch_base   = it * KV_PER_ITER_MMA;
        int consume_buf  = it % KV_BUFFERS;

        mbarrier_wait_parity(mbar_mma, phase_mma);
        phase_mma ^= 1;
        tcgen05_fence_after();

        if (warp_id < 4) {
            float logits_reg[16];
            tcgen05_ld_32x32b_x16(tmem_acc, logits_reg);
            tcgen05_wait_ld();
            if (warp_id == 0 && lane_id < NUM_QO_HEADS) {
                int h = lane_id;
                #pragma unroll
                for (int i = 0; i < MMA_N; ++i) {
                    smem_logits[h * MMA_N + i] = logits_reg[i];
                }
            }
        }
        __syncthreads();

        int next_it = it + 1;
        if (next_it < n_iters) {
            int next_buf = next_it % KV_BUFFERS;
            mbarrier_wait_parity(&mbar_kv[next_buf], phase_kv[next_buf]);
            phase_kv[next_buf] ^= 1;
            rewrite_lin_to_mma(next_buf);
            __syncthreads();
            issue_mma();
        }

        // ---- Softmax + PV: warp h handles head h (same as A23 per-head-per-warp). ----
        // warp_id ∈ [0, 16) = head index.
        int h = warp_id;
        // Read 16 logits for this head from SMEM.
        float logits[MMA_N];
        #pragma unroll
        for (int i = 0; i < MMA_N; ++i) logits[i] = smem_logits[h * MMA_N + i];

        // Scale + sentinel mask.
        #pragma unroll
        for (int i = 0; i < MMA_N; ++i) logits[i] *= sm_scale;
        #pragma unroll
        for (int i = 0; i < MMA_N; ++i) {
            int local = batch_base + i;
            int32_t kv_idx = (local < (k_end - k_start)) ? smem_idx[local] : -1;
            if (kv_idx < 0) logits[i] = -INFINITY;
        }
        float row_m_local = -INFINITY;
        #pragma unroll
        for (int i = 0; i < MMA_N; ++i) row_m_local = fmaxf(row_m_local, logits[i]);

        float m_new     = fmaxf(row_max, row_m_local);
        float scale_old = (row_max == -INFINITY) ? 0.0f : __expf(row_max - m_new);
        row_sum *= scale_old;
        #pragma unroll
        for (int i = 0; i < CKV_PER_LANE; ++i) o_acc[i] *= scale_old;

        if (m_new != -INFINITY) {
            float pp[MMA_N];
            #pragma unroll
            for (int i = 0; i < MMA_N; ++i) {
                pp[i] = (logits[i] == -INFINITY) ? 0.0f : __expf(logits[i] - m_new);
                row_sum += pp[i];
            }
            // PV: 32 lanes per warp. Lane L owns pairs {j*64+2*L, j*64+2*L+1}
            // for j ∈ [0, 8). 8 pairs × 2 = 16 positions per lane, 32 lanes × 16 = 512 ✓.
            // Read __nv_bfloat162 (2 bf16 per LDG.32).
            const __nv_bfloat16* kv_lin_iter = smem_kv_lin[consume_buf];
            #pragma unroll 4
            for (int i = 0; i < MMA_N; ++i) {
                if (pp[i] == 0.0f) continue;
                const __nv_bfloat16* v_row = kv_lin_iter + i * HEAD_DIM_TOT;
                #pragma unroll
                for (int j = 0; j < CKV_PER_LANE / 2; ++j) {
                    int d_pair = j * 64 + 2 * lane_id;
                    __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(v_row + d_pair);
                    float2 vf = __bfloat1622float2(v2);
                    o_acc[2*j + 0] += pp[i] * vf.x;
                    o_acc[2*j + 1] += pp[i] * vf.y;
                }
            }
        }
        row_max = m_new;
        __syncthreads();  // before we overwrite linear[consume_buf] via new TMA

        // Issue next-cycle's TMA into consume_buf (now free — PV done).
        {
            int issue_local = (it + KV_BUFFERS) * KV_PER_ITER_MMA;
            if (issue_local < (k_end - k_start)) {
                issue_tma_batch(issue_local, consume_buf);
            }
        }
    }

    // Write partials: warp h owns head h. Same pattern as A23.
    {
        int h = warp_id;
        int64_t ml_off = (int64_t)t * splits_per_token * NUM_QO_HEADS
                       + s * NUM_QO_HEADS + h;
        if (lane_id == 0) {
            scratch_m[ml_off] = row_max;
            scratch_l[ml_off] = row_sum;
        }
        int64_t o_off = ((int64_t)t * splits_per_token + s) * NUM_QO_HEADS * HEAD_DIM_CKV
                      + h * HEAD_DIM_CKV;
        // New layout matching PV: lane L, j → positions {j*64+2*L, j*64+2*L+1}.
        // Store as float2 for vectorized writes.
        #pragma unroll
        for (int j = 0; j < CKV_PER_LANE / 2; ++j) {
            float2 v = {o_acc[2*j + 0], o_acc[2*j + 1]};
            *reinterpret_cast<float2*>(scratch_o + o_off + j * 64 + 2 * lane_id) = v;
        }
    }
    __syncthreads();
    if (warp_id == 0) tcgen05_dealloc_1(tmem_acc, 32);
    __syncthreads();

    // Last-arrival atomic reduce.
    __threadfence();
    __shared__ bool smem_last;
    if (tid == 0) {
        int prev = atomicAdd(&completion_counters[t], 1);
        smem_last = (prev == splits_per_token - 1);
    }
    __syncthreads();
    if (!smem_last) return;

    // Reduce (same as A23): warp_id = head.
    {
        int h = warp_id;
        const float* mh = scratch_m + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + h;
        const float* lh = scratch_l + ((int64_t)t * splits_per_token) * NUM_QO_HEADS + h;

        float m_local = -INFINITY;
        if (lane_id < splits_per_token) m_local = mh[lane_id * NUM_QO_HEADS];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float v = __shfl_xor_sync(0xffffffff, m_local, off);
            m_local = fmaxf(m_local, v);
        }
        float m_global = m_local;

        float scale_s = 0.0f, l_local = 0.0f;
        if (lane_id < splits_per_token) {
            float m_s = mh[lane_id * NUM_QO_HEADS];
            float l_s = lh[lane_id * NUM_QO_HEADS];
            scale_s = (m_global == -INFINITY) ? 0.0f : __expf(m_s - m_global);
            l_local = l_s * scale_s;
        }
        float l_warp = l_local;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) l_warp += __shfl_xor_sync(0xffffffff, l_warp, off);
        float l_global = l_warp;
        bool empty   = (l_global == 0.0f);
        float inv_l  = empty ? 0.0f : 1.0f / l_global;

        float oacc2[CKV_PER_LANE];
        #pragma unroll
        for (int i = 0; i < CKV_PER_LANE; ++i) oacc2[i] = 0.0f;
        for (int si = 0; si < splits_per_token; ++si) {
            float scale_b = __shfl_sync(0xffffffff, scale_s, si);
            if (scale_b == 0.0f) continue;
            const float* sw = scratch_o
                + ((int64_t)t * splits_per_token + si) * NUM_QO_HEADS * HEAD_DIM_CKV
                + h * HEAD_DIM_CKV;
            // New layout: lane L at pair (j*64 + 2*L).
            #pragma unroll
            for (int j = 0; j < CKV_PER_LANE / 2; ++j) {
                float2 v = *reinterpret_cast<const float2*>(sw + j * 64 + 2 * lane_id);
                oacc2[2*j + 0] += scale_b * v.x;
                oacc2[2*j + 1] += scale_b * v.y;
            }
        }

        __nv_bfloat16* ow = output + t * NUM_QO_HEADS * HEAD_DIM_CKV + h * HEAD_DIM_CKV;
        #pragma unroll
        for (int j = 0; j < CKV_PER_LANE / 2; ++j) {
            __nv_bfloat162 packed = __floats2bfloat162_rn(
                oacc2[2*j + 0] * inv_l,
                oacc2[2*j + 1] * inv_l);
            *reinterpret_cast<__nv_bfloat162*>(ow + j * 64 + 2 * lane_id) = packed;
        }
        if (lane_id == 0) {
            float lse_v = empty ? -INFINITY : (m_global * LOG2_E + log2f(l_global));
            lse[t * NUM_QO_HEADS + h] = lse_v;
        }
    }

    if (tid == 0) completion_counters[t] = 0;
}

// ---- Launcher ----
static int choose_splits(int n) {
    int s = 132 / (n > 0 ? n : 1);
    if (s < 1) s = 1;
    if (s > MAX_SPLITS) s = MAX_SPLITS;
    return s;
}

static int*    g_counters = nullptr;
static int     g_counters_cap = 0;
static void ensure_counters(int n, cudaStream_t stream) {
    if (n > g_counters_cap) {
        if (g_counters) cudaFree(g_counters);
        int c = (n > g_counters_cap * 2) ? n : g_counters_cap * 2;
        if (c < 64) c = 64;
        cudaMalloc(&g_counters, c * sizeof(int));
        cudaMemsetAsync(g_counters, 0, c * sizeof(int), stream);
        g_counters_cap = c;
    }
}
static float*  g_scratch = nullptr;
static int64_t g_scratch_cap = 0;
static void ensure_scratch(int64_t n, cudaStream_t) {
    if (n > g_scratch_cap) {
        if (g_scratch) cudaFree(g_scratch);
        int64_t c = (n > g_scratch_cap * 2) ? n : g_scratch_cap * 2;
        cudaMalloc(&g_scratch, c * sizeof(float));
        g_scratch_cap = c;
    }
}

void launch_sparse_attention_c(
    torch::Tensor q_nope, torch::Tensor q_pe,
    torch::Tensor ckv_cache, torch::Tensor kpe_cache,
    torch::Tensor sparse_indices, double sm_scale,
    torch::Tensor output, torch::Tensor lse)
{
    const at::cuda::CUDAGuard device_guard(q_nope.device());
    const int N = static_cast<int>(q_nope.size(0));
    if (N == 0) return;

    const int S = choose_splits(N);
    const int kv_per_split = (TOPK + S - 1) / S;
    TORCH_CHECK(kv_per_split <= SMEM_IDX_CAP, "kv_per_split > SMEM_IDX_CAP");

    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_counters(N, stream);
    const int64_t ml_count = (int64_t)N * S * NUM_QO_HEADS;
    const int64_t o_count  = ml_count * HEAD_DIM_CKV;
    ensure_scratch(ml_count * 2 + o_count, stream);

    float* scratch_m = g_scratch;
    float* scratch_l = scratch_m + ml_count;
    float* scratch_o = scratch_l + ml_count;

    int smem_bytes = SMEM_Q_BYTES + SMEM_KV_BYTES
                   + SMEM_KV_LINEAR_BYTES * KV_BUFFERS
                   + SMEM_LOGITS_BYTES
                   + 8 + 8 * KV_BUFFERS + SMEM_IDX_CAP * 4 + 4 + 64;
    cudaFuncSetAttribute(sparse_attention_mma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 grid(S, N);
    sparse_attention_mma_kernel<<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
        sparse_indices.data_ptr<int32_t>(),
        static_cast<float>(sm_scale),
        scratch_m, scratch_l, scratch_o, g_counters,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        lse.data_ptr<float>(), N, S, kv_per_split);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sparse_attention_c", &launch_sparse_attention_c,
          "DSA Sparse Attention (T3-567d: 16-warp tcgen05 MMA for QK^T)");
}
