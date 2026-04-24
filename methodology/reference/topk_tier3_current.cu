// dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
// Tier 3 + Option C: fused tcgen05 FP8 MMA + SMEM-transpose H-reduction.
// Per-page: FP8 K HBM -> SMEM 8xT -> tcgen05.mma.kind::f8f6f4 -> TMEM -> tcgen05.ld
// -> scale multiply -> scores_smem[64×64] -> 64 threads each read DOWN a column with
// a 64-FMA weighted-relu-sum -> final_scores[B, N_max] GMEM.
// Eliminates the scores_all[B,64,N_max] round trip without the shuffle-tree critical
// path that caused GT-29's Tier 4(b) regression. Downstream (at::topk, translate) unchanged.

#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int PAGE_DATA_BYTES = 8192;
constexpr int PAGE_BYTES      = 8448;
constexpr int TOPK            = 2048;
constexpr int H               = 64;       // num query heads

// Tier 3 constants
constexpr int PAGES_PER_CTA   = 4;       // GT-13 min=2
constexpr int MMA_M           = 64;      // matches H exactly
constexpr int MMA_N           = 64;      // one page per N-tile (no straddling)
constexpr int MMA_K           = 32;      // f8f6f4 fixed per Table 41
constexpr int NUM_SLABS       = INDEX_HEAD_DIM / MMA_K;  // 4
// Opt-#5: padded stride for scores_smem ([64 × SCORES_STRIDE]) to break 16-way
// SMEM bank conflict on writes/reads. 65 is co-prime with 32 banks.
constexpr int SCORES_STRIDE   = 65;
// IDESC: M=64, N=64, K=32, dtype=F32(=1), atype=E4M3(=0), btype=E4M3(=0),
// no transpose/negate/sparsity/.ws. See d3.md Decision 4.
constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);  // 0x04100010

// SMEM layout (all 16-B aligned; see d4.md address composition)
// Opt-2: double-buffered K_smem and scale_smem for cross-page load/MMA overlap.
constexpr int Q_SMEM_OFFSET       = 0;                   // 8192 B
constexpr int K0_SMEM_OFFSET      = 8192;                // 8192 B (buffer 0)
constexpr int K1_SMEM_OFFSET      = 16384;               // 8192 B (buffer 1)
// Opt-#5: scores_smem size = 64 × 65 × 4 = 16640 B (padded from 16384 B).
constexpr int SCORES_SMEM_OFFSET  = 24576;               // 16640 B ([64 rows × 65 cols] FP32)
constexpr int SCALE0_SMEM_OFFSET  = 41216;               // 256 B
constexpr int SCALE1_SMEM_OFFSET  = 41472;               // 256 B
constexpr int BT_CACHE_OFFSET     = 41728;               // 16 B
constexpr int ALLOC_SLOT_OFFSET   = 41744;               // 16 B
constexpr int MBAR_SLOT_OFFSET    = 41760;               // 16 B
constexpr int WEIGHTS_SMEM_OFFSET = 41776;               // 256 B ([H=64] FP32)
constexpr int SMEM_BYTES          = 42032;

// GT-11 8xT core-tile constants
constexpr int SLAB_BYTES = 2048;
constexpr int SBO_BYTES  = 256;
constexpr int LBO_BYTES  = 128;

// 8xT fill formula: for an FP8 byte at logical (m, k) in a [M_tile, K_tile=128] operand
// stored across 4 K-slabs, the SMEM byte offset is:
__device__ __forceinline__ int smem_8xT_offset(int m, int k) {
    return (k / 32) * SLAB_BYTES
         + (m / 8)  * SBO_BYTES
         + ((k % 32) / 16) * LBO_BYTES
         + (m % 8)  * 16
         + (k % 16);
}

// SMEM descriptor encode helper (Table 42): 14-bit field units of 16 B
__device__ __forceinline__ uint64_t desc_encode_u64(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

// Build MMA SMEM descriptor for a K-major 8xT operand at the given shared-space
// address (lower 24 bits). SBO=256, LBO=128 (GT-11). Swizzle=None (GT-10).
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr_shared) {
    uint64_t d = 0;
    d |= desc_encode_u64((uint64_t)smem_addr_shared);   // bits 0-13 matrix start
    d |= desc_encode_u64((uint64_t)LBO_BYTES) << 16;    // bits 16-29 LBO -> 8
    d |= desc_encode_u64((uint64_t)SBO_BYTES) << 32;    // bits 32-45 SBO -> 16
    d |= (uint64_t)0b001ULL << 46;                      // bits 46-48 fixed
    // bits 49+ all 0: base offset, stride mode, swizzle = NONE
    return d;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        " .reg .pred %%px;\n\t"
        " elect.sync _|%%px, %1;\n\t"
        " @%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFFu)
    );
    return pred;
}

__device__ __forceinline__ void mbarrier_init_1(uint32_t mbar_smem) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mbar_smem, uint32_t phase) {
    uint32_t ticks = 0x989680u;
    asm volatile(
        "{\n\t"
        " .reg .pred P1;\n\t"
        "LAB_WAIT_%=:\n\t"
        " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        " @P1 bra DONE_%=;\n\t"
        " bra LAB_WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar_smem), "r"(phase), "r"(ticks)
    );
}

__device__ __forceinline__ void tcgen05_alloc_64(uint32_t alloc_slot_smem) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"(alloc_slot_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_64(uint32_t taddr) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;"
                 :: "r"(taddr) : "memory");
}

// GT-18: commit form for CUDA 12.8 on sm_100a
__device__ __forceinline__ void tcgen05_commit_mbar(uint32_t mbar_smem) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Opt-#7: cp.async helpers. 16-byte async copy GMEM -> SMEM, no register round trip.
__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr_shared,
                                              const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr_shared), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// FP8 MMA. A and B are both K-major 8xT in SMEM, transpose_A=transpose_B=0 (GT-15).
// enable_input_d = 0 -> fresh accumulator (D = A*B); 1 -> accumulate (D = A*B + D).
__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, int enable_input_d)
{
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(idesc), "r"(enable_input_d)
    );
}

// tcgen05.ld.32x32b.x32 — loads 32 cols of 32 TMEM lanes into 32 FP32 regs per thread.
// Per warp thread L: TMEM lane (warp_id*32 + L), 32 consecutive columns from taddr.
__device__ __forceinline__ void tcgen05_ld_32x32b_x32(
    uint32_t taddr,
    uint32_t (&r)[32])
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),  "=r"(r[1]),  "=r"(r[2]),  "=r"(r[3]),
          "=r"(r[4]),  "=r"(r[5]),  "=r"(r[6]),  "=r"(r[7]),
          "=r"(r[8]),  "=r"(r[9]),  "=r"(r[10]), "=r"(r[11]),
          "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
          "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
          "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
          "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
    );
}

// =====================================================================
//  Fused FP8 MMA + H-reduction kernel (Tier 3 + Option C)
//  Grid:  (ceil(max_num_pages / PAGES_PER_CTA), B)
//  Block: 128 threads (4 warps)
//  SMEM:  ~33 KiB
//  Output: final_scores[B, N_max] (H-reduction done inline; no scores_all intermediate)
// =====================================================================
__global__ __launch_bounds__(128, 2)
void fp8_mma_final_scores_kernel(
    const uint8_t* __restrict__ q_fp8,          // [B, 64, 128] FP8 (as raw bytes)
    const uint8_t* __restrict__ k_cache,        // paged FP8 cache (uint8 view)
    const float*   __restrict__ weights,        // [B, 64] FP32
    const int32_t* __restrict__ block_table,    // [B, max_num_pages]
    const int32_t* __restrict__ seq_lens,       // [B]
    float*         __restrict__ final_scores,   // [B, N_max] FP32
    int max_num_pages,
    int N_max)
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    uint8_t* Q_smem          = smem_raw + Q_SMEM_OFFSET;
    uint8_t* K_smem_buf[2]   = { smem_raw + K0_SMEM_OFFSET,
                                  smem_raw + K1_SMEM_OFFSET };
    float*   scores_smem     = reinterpret_cast<float*>(smem_raw + SCORES_SMEM_OFFSET);
    float*   scale_smem_buf[2] = {
        reinterpret_cast<float*>(smem_raw + SCALE0_SMEM_OFFSET),
        reinterpret_cast<float*>(smem_raw + SCALE1_SMEM_OFFSET) };
    int32_t* bt_cache_smem   = reinterpret_cast<int32_t*>(smem_raw + BT_CACHE_OFFSET);
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + ALLOC_SLOT_OFFSET);
    float*   weights_smem    = reinterpret_cast<float*>(smem_raw + WEIGHTS_SMEM_OFFSET);
    // mbar_slot at MBAR_SLOT_OFFSET (accessed by shared-space addr, not via C++ ptr)

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int b         = blockIdx.y;
    const int cta_page_start = blockIdx.x * PAGES_PER_CTA;

    // ---- P0: kernel-scope setup (thread-private regs) ----
    const int seq_len_b = seq_lens[b];

    // Shared-space (24-bit) addresses for descriptors / asm
    const uint32_t q_shared       = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_shared_buf[2] = {
        __cvta_generic_to_shared(K_smem_buf[0]),
        __cvta_generic_to_shared(K_smem_buf[1]) };
    const uint32_t alloc_slot_s   = __cvta_generic_to_shared(alloc_slot_ptr);
    const uint32_t mbar_s         = __cvta_generic_to_shared(smem_raw + MBAR_SLOT_OFFSET);

    // ---- P1: TMEM alloc (warp 1) ----
    if (warp_id == 1) {
        tcgen05_alloc_64(alloc_slot_s);
    }
    __syncthreads();
    const uint32_t tmem_col = alloc_slot_ptr[0];   // column base returned by alloc

    // ---- P2: block_table cache (warp 0, thread 0) ----
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int p = 0; p < PAGES_PER_CTA; ++p) {
            const int pg_id = cta_page_start + p;
            bt_cache_smem[p] = (pg_id < max_num_pages)
                ? block_table[b * max_num_pages + pg_id] : 0;
        }
    }

    // ---- P3: mbarrier init (warp 0, thread 0) ----
    if (warp_id == 0 && lane_id == 0) {
        mbarrier_init_1(mbar_s);
    }
    // ---- P3b: weights cooperative load (parallel with mbar init) — Option C ----
    if (tid < H) {
        weights_smem[tid] = weights[b * H + tid];
    }
    uint32_t phase = 0;
    __syncthreads();

    // ---- P4: Q cooperative load -> Q_smem (8xT) via cp.async (Opt-#3) ----
    // 128 threads * 4 uint4 = 512 uint4 = 8192 B. Issued async; waited below
    // alongside page 0's K cp.async so both overlap with TMEM-alloc + mbar setup.
    {
        const uint8_t* q_base = q_fp8 + size_t(b) * 64 * INDEX_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u        = tid * 4 + i;
            int m        = u >> 3;
            int k_uint4  = u & 7;
            int k_base   = k_uint4 << 4;
            int off = smem_8xT_offset(m, k_base);
            cp_async_16B(q_shared + off,
                         q_base + m * INDEX_HEAD_DIM + k_base);
        }
    }
    cp_async_commit();

    // ---- Page loop (Opt-2 double-buffer + Opt-#7 cp.async K loads) ----
    // Per-token zero-fill removed: MMA output is column-independent (output[M,col]
    // depends only on K input col, not other cols), so garbage K for tokens past
    // seq_len_b doesn't contaminate valid cols' results. The final_scores mask
    // `(n_global >= seq_len_b) ? -INFINITY : acc` handles those cols.
    const int num_active = min(PAGES_PER_CTA,
                               max(0, max_num_pages - cta_page_start));

    if (num_active > 0) {
        // ---- Prologue: async-load page 0 K + sync-load scale into buffer 0 ----
        {
            const int pidx0 = 0;
            const int32_t phys0 = bt_cache_smem[pidx0];
            const size_t  pg_base0 = size_t(phys0) * PAGE_BYTES;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int u        = tid * 4 + i;
                int tok      = u >> 3;
                int k_uint4  = u & 7;
                int k_base   = k_uint4 << 4;
                int off      = smem_8xT_offset(tok, k_base);
                cp_async_16B(k_shared_buf[0] + off,
                             k_cache + pg_base0 + tok * INDEX_HEAD_DIM + k_base);
            }
            cp_async_commit();
            if (tid < PAGE_SIZE) {
                scale_smem_buf[0][tid] = *reinterpret_cast<const float*>(
                    k_cache + pg_base0 + PAGE_DATA_BYTES + tid * 4);
            }
        }
        cp_async_wait_all();
        __syncthreads();

        // ---- Pipelined main loop ----
        for (int pidx = 0; pidx < num_active; ++pidx) {
            const int curr_buf   = pidx & 1;
            const int next_buf   = curr_buf ^ 1;
            const int next_pidx  = pidx + 1;
            const bool has_next  = (next_pidx < num_active);
            const int page_id    = cta_page_start + pidx;

            // ---- Issue MMA (warp 0 elected) on curr_buf ----
            if (warp_id == 0 && elect_one_sync()) {
                const uint32_t tmem_addr = tmem_col;
                #pragma unroll 1
                for (int s = 0; s < NUM_SLABS; ++s) {
                    const uint32_t q_slab_addr = q_shared + s * SLAB_BYTES;
                    const uint32_t k_slab_addr = k_shared_buf[curr_buf] + s * SLAB_BYTES;
                    const uint64_t a_desc = make_smem_desc(q_slab_addr);
                    const uint64_t b_desc = make_smem_desc(k_slab_addr);
                    const int enable_d = (s != 0) ? 1 : 0;
                    tcgen05_mma_f8f6f4(tmem_addr, a_desc, b_desc, IDESC_VAL, enable_d);
                }
                tcgen05_commit_mbar(mbar_s);
            }

            // ---- Overlap: cp.async next page's K + sync-load scale into next_buf ----
            if (has_next) {
                const int32_t phys_n  = bt_cache_smem[next_pidx];
                const size_t  pg_base_n = size_t(phys_n) * PAGE_BYTES;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int u        = tid * 4 + i;
                    int tok      = u >> 3;
                    int k_uint4  = u & 7;
                    int k_base   = k_uint4 << 4;
                    int off      = smem_8xT_offset(tok, k_base);
                    cp_async_16B(k_shared_buf[next_buf] + off,
                                 k_cache + pg_base_n + tok * INDEX_HEAD_DIM + k_base);
                }
                cp_async_commit();
                if (tid < PAGE_SIZE) {
                    scale_smem_buf[next_buf][tid] = *reinterpret_cast<const float*>(
                        k_cache + pg_base_n + PAGE_DATA_BYTES + tid * 4);
                }
            }

            // ---- Wait for curr MMA ----
            mbarrier_wait_phase(mbar_s, phase);
            phase ^= 1;
            tcgen05_fence_after_sync();

            // ---- tcgen05.ld (all 32 lanes per warp — GT-17) ----
            const uint32_t taddr_lo = (uint32_t(warp_id) * 32u) << 16 | tmem_col;
            const uint32_t taddr_hi = (uint32_t(warp_id) * 32u) << 16 | (tmem_col + 32u);
            uint32_t rlo[32], rhi[32];
            tcgen05_ld_32x32b_x32(taddr_lo, rlo);
            tcgen05_ld_32x32b_x32(taddr_hi, rhi);
            tcgen05_wait_ld();

            // ---- scale apply + masked write to scores_smem (curr_buf scale) ----
            // Opt-#5: row stride = SCORES_STRIDE = 65 (breaks 16-way bank conflict).
            const float* Sc = scale_smem_buf[curr_buf];
            if (lane_id < 16) {
                const int row = warp_id * 16 + lane_id;
                float* row_out = scores_smem + row * SCORES_STRIDE;
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    float v = __uint_as_float(rlo[c]);
                    row_out[c] = v * Sc[c];
                }
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    float v = __uint_as_float(rhi[c]);
                    row_out[c + 32] = v * Sc[c + 32];
                }
            }
            // Barrier: scores_smem writes visible + K_smem_buf[next_buf] loads committed
            __syncthreads();

            // ---- SMEM-transpose H-reduction + GMEM write final_scores ----
            // Opt-NEW: 8-way partial-sum ILP.
            if (tid < PAGE_SIZE) {
                const int col      = tid;
                const int n_global = page_id * PAGE_SIZE + col;
                float acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
                #pragma unroll
                for (int h = 0; h < H; h += 8) {
                    float s0 = scores_smem[(h  ) * SCORES_STRIDE + col];
                    float s1 = scores_smem[(h+1) * SCORES_STRIDE + col];
                    float s2 = scores_smem[(h+2) * SCORES_STRIDE + col];
                    float s3 = scores_smem[(h+3) * SCORES_STRIDE + col];
                    float s4 = scores_smem[(h+4) * SCORES_STRIDE + col];
                    float s5 = scores_smem[(h+5) * SCORES_STRIDE + col];
                    float s6 = scores_smem[(h+6) * SCORES_STRIDE + col];
                    float s7 = scores_smem[(h+7) * SCORES_STRIDE + col];
                    acc0 = fmaf(fmaxf(s0, 0.0f), weights_smem[h  ], acc0);
                    acc1 = fmaf(fmaxf(s1, 0.0f), weights_smem[h+1], acc1);
                    acc2 = fmaf(fmaxf(s2, 0.0f), weights_smem[h+2], acc2);
                    acc3 = fmaf(fmaxf(s3, 0.0f), weights_smem[h+3], acc3);
                    acc4 = fmaf(fmaxf(s4, 0.0f), weights_smem[h+4], acc4);
                    acc5 = fmaf(fmaxf(s5, 0.0f), weights_smem[h+5], acc5);
                    acc6 = fmaf(fmaxf(s6, 0.0f), weights_smem[h+6], acc6);
                    acc7 = fmaf(fmaxf(s7, 0.0f), weights_smem[h+7], acc7);
                }
                float acc = ((acc0+acc1) + (acc2+acc3)) + ((acc4+acc5) + (acc6+acc7));
                final_scores[b * N_max + n_global] =
                    (n_global >= seq_len_b) ? -INFINITY : acc;
            }
            // Opt-J: removed the end-of-iter __syncthreads. mbarrier_wait at start
            // of next iter naturally gates warps that didn't do H-reduction
            // (warp 0 elected thread must issue MMA+commit before other warps pass
            // mbarrier_wait, and warp 0's elected thread is itself gated on H-reduction
            // completion of the current iter). The cp_async_wait_all still needed
            // for K_smem[next_buf] visibility to MMA descriptor.
            if (has_next) {
                cp_async_wait_all();
            }
        }
    }

    // ---- TMEM dealloc (single exit path) ----
    if (warp_id == 1) {
        tcgen05_dealloc_64(tmem_col);
    }
}

// =====================================================================
//  Tier 2(c): fused block radix-sort top-K + index translation.
//  Opt-#3: two variants — large (512×16=8192 cap) for N_max > 2048,
//  small (256×8=2048 cap) for N_max ≤ 2048. Host dispatches per workload.
// =====================================================================
constexpr int TOPK_BLOCK_THREADS    = 512;
constexpr int TOPK_ITEMS_PER_THREAD = 16;
constexpr int TOPK_CAPACITY         = TOPK_BLOCK_THREADS * TOPK_ITEMS_PER_THREAD;

constexpr int TOPK_SMALL_THREADS    = 256;
constexpr int TOPK_SMALL_ITEMS      = 8;
constexpr int TOPK_SMALL_CAPACITY   = TOPK_SMALL_THREADS * TOPK_SMALL_ITEMS;  // 2048

// Opt-D: tiny variant for N_max ≤ 512 (max_num_pages ≤ 8).
constexpr int TOPK_TINY_THREADS     = 128;
constexpr int TOPK_TINY_ITEMS       = 4;
constexpr int TOPK_TINY_CAPACITY    = TOPK_TINY_THREADS * TOPK_TINY_ITEMS;    // 512

__global__ __launch_bounds__(TOPK_BLOCK_THREADS, 1)
void block_topk_translate_kernel(
    const float*   __restrict__ final_scores,    // [B, N_max]
    const int32_t* __restrict__ block_table,     // [B, max_num_pages]
    const int32_t* __restrict__ seq_lens,        // [B]
    int32_t*       __restrict__ topk_indices,    // [B, TOPK] — pre-filled with -1
    int N_max,
    int max_num_pages)
{
    using BlockLoadKeys = cub::BlockLoad<float, TOPK_BLOCK_THREADS,
                                         TOPK_ITEMS_PER_THREAD,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockRadixSortT = cub::BlockRadixSort<float, TOPK_BLOCK_THREADS,
                                                TOPK_ITEMS_PER_THREAD, int32_t>;

    __shared__ union {
        typename BlockLoadKeys::TempStorage load_keys;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    const int b = blockIdx.x;
    const int sl = seq_lens[b];
    const int actual_topk = (sl < TOPK) ? sl : TOPK;

    float keys[TOPK_ITEMS_PER_THREAD];
    BlockLoadKeys(temp_storage.load_keys).Load(
        final_scores + size_t(b) * N_max,
        keys,
        N_max,
        -INFINITY);
    __syncthreads();

    int32_t vals[TOPK_ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < TOPK_ITEMS_PER_THREAD; ++i) {
        int pos = threadIdx.x * TOPK_ITEMS_PER_THREAD + i;
        vals[i] = (pos < N_max) ? pos : -1;
    }

    BlockRadixSortT(temp_storage.sort).SortDescending(keys, vals);

    // Opt-#4: fuse fill_(-1) — write every rank < TOPK, choosing phys-translated
    // index for ranks < actual_topk and -1 sentinel for padded ranks in between.
    #pragma unroll
    for (int i = 0; i < TOPK_ITEMS_PER_THREAD; ++i) {
        int rank = threadIdx.x * TOPK_ITEMS_PER_THREAD + i;
        if (rank < TOPK) {
            int32_t out_val;
            if (rank < actual_topk) {
                int32_t lid = vals[i];
                int32_t phys = block_table[b * max_num_pages + (lid >> 6)];
                out_val = phys * PAGE_SIZE + (lid & 63);
            } else {
                out_val = -1;
            }
            topk_indices[b * TOPK + rank] = out_val;
        }
    }
}

// Opt-#3 small variant: 256 threads × 8 items = 2048 capacity for N_max ≤ 2048.
__global__ __launch_bounds__(TOPK_SMALL_THREADS, 2)
void block_topk_translate_small_kernel(
    const float*   __restrict__ final_scores,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    int32_t*       __restrict__ topk_indices,
    int N_max,
    int max_num_pages)
{
    using BlockLoadKeys = cub::BlockLoad<float, TOPK_SMALL_THREADS,
                                         TOPK_SMALL_ITEMS,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockRadixSortT = cub::BlockRadixSort<float, TOPK_SMALL_THREADS,
                                                TOPK_SMALL_ITEMS, int32_t>;

    __shared__ union {
        typename BlockLoadKeys::TempStorage load_keys;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    const int b = blockIdx.x;
    const int sl = seq_lens[b];
    const int actual_topk = (sl < TOPK) ? sl : TOPK;

    float keys[TOPK_SMALL_ITEMS];
    BlockLoadKeys(temp_storage.load_keys).Load(
        final_scores + size_t(b) * N_max,
        keys,
        N_max,
        -INFINITY);
    __syncthreads();

    int32_t vals[TOPK_SMALL_ITEMS];
    #pragma unroll
    for (int i = 0; i < TOPK_SMALL_ITEMS; ++i) {
        int pos = threadIdx.x * TOPK_SMALL_ITEMS + i;
        vals[i] = (pos < N_max) ? pos : -1;
    }

    BlockRadixSortT(temp_storage.sort).SortDescending(keys, vals);

    // Opt-#4: fuse fill_(-1).
    #pragma unroll
    for (int i = 0; i < TOPK_SMALL_ITEMS; ++i) {
        int rank = threadIdx.x * TOPK_SMALL_ITEMS + i;
        if (rank < TOPK) {
            int32_t out_val;
            if (rank < actual_topk) {
                int32_t lid = vals[i];
                int32_t phys = block_table[b * max_num_pages + (lid >> 6)];
                out_val = phys * PAGE_SIZE + (lid & 63);
            } else {
                out_val = -1;
            }
            topk_indices[b * TOPK + rank] = out_val;
        }
    }
}

// Opt-D tiny variant: 128 threads × 4 items = 512 capacity for N_max ≤ 512.
__global__ __launch_bounds__(TOPK_TINY_THREADS, 4)
void block_topk_translate_tiny_kernel(
    const float*   __restrict__ final_scores,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    int32_t*       __restrict__ topk_indices,
    int N_max,
    int max_num_pages)
{
    using BlockLoadKeys = cub::BlockLoad<float, TOPK_TINY_THREADS,
                                         TOPK_TINY_ITEMS,
                                         cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockRadixSortT = cub::BlockRadixSort<float, TOPK_TINY_THREADS,
                                                TOPK_TINY_ITEMS, int32_t>;

    __shared__ union {
        typename BlockLoadKeys::TempStorage load_keys;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    const int b = blockIdx.x;
    const int sl = seq_lens[b];
    const int actual_topk = (sl < TOPK) ? sl : TOPK;

    float keys[TOPK_TINY_ITEMS];
    BlockLoadKeys(temp_storage.load_keys).Load(
        final_scores + size_t(b) * N_max,
        keys,
        N_max,
        -INFINITY);
    __syncthreads();

    int32_t vals[TOPK_TINY_ITEMS];
    #pragma unroll
    for (int i = 0; i < TOPK_TINY_ITEMS; ++i) {
        int pos = threadIdx.x * TOPK_TINY_ITEMS + i;
        vals[i] = (pos < N_max) ? pos : -1;
    }

    BlockRadixSortT(temp_storage.sort).SortDescending(keys, vals);

    // Pass 1: ranks 0..511 from sort.
    #pragma unroll
    for (int i = 0; i < TOPK_TINY_ITEMS; ++i) {
        int rank = threadIdx.x * TOPK_TINY_ITEMS + i;
        int32_t out_val;
        if (rank < actual_topk) {
            int32_t lid = vals[i];
            int32_t phys = block_table[b * max_num_pages + (lid >> 6)];
            out_val = phys * PAGE_SIZE + (lid & 63);
        } else {
            out_val = -1;
        }
        topk_indices[b * TOPK + rank] = out_val;
    }
    // Pass 2: ranks 512..TOPK-1 = -1 sentinel (actual_topk < 512 here).
    for (int rank = TOPK_TINY_CAPACITY + threadIdx.x; rank < TOPK;
         rank += TOPK_TINY_THREADS) {
        topk_indices[b * TOPK + rank] = -1;
    }
}

// =====================================================================
//  Host entry point (unchanged signature; config.toml entry_point unchanged)
// =====================================================================
void launch_topk_c(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    const at::cuda::CUDAGuard device_guard(q_index_fp8.device());
    const int B             = static_cast<int>(q_index_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int N_max         = max_num_pages * PAGE_SIZE;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // View Q as raw bytes (FP8 e4m3 interpreted as uint8 for MMA consumption).
    auto q_u8      = q_index_fp8.view(torch::kUInt8);
    auto k_cache_u8 = k_index_cache_fp8.view(torch::kUInt8);

    const uint8_t* q_ptr       = q_u8.data_ptr<uint8_t>();
    const uint8_t* k_cache_ptr = k_cache_u8.data_ptr<uint8_t>();
    const float*   w_ptr       = weights.data_ptr<float>();
    const int32_t* bt_ptr      = block_table.data_ptr<int32_t>();
    const int32_t* sl_ptr      = seq_lens.data_ptr<int32_t>();

    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q_index_fp8.device());

    // Phases 1+2+3 (fused via Option C): MMA + scale + relu + weights + sum-over-heads.
    auto final_scores = torch::empty({B, N_max}, opts_f32);

    const int grid_x = (max_num_pages + PAGES_PER_CTA - 1) / PAGES_PER_CTA;
    dim3 grid(grid_x, B);
    fp8_mma_final_scores_kernel<<<grid, 128, SMEM_BYTES, stream>>>(
        q_ptr, k_cache_ptr, w_ptr, bt_ptr, sl_ptr,
        final_scores.data_ptr<float>(),
        max_num_pages, N_max);
    AT_CUDA_CHECK(cudaGetLastError());

    // Phase 4+5 (Tier 2(c) + Opt-#3 + Opt-#4): adaptive top-K + fused fill.
    TORCH_CHECK(N_max <= TOPK_CAPACITY,
                "N_max exceeds block_topk_translate_kernel capacity (8192)");
    if (N_max <= TOPK_TINY_CAPACITY) {
        block_topk_translate_tiny_kernel<<<B, TOPK_TINY_THREADS, 0, stream>>>(
            final_scores.data_ptr<float>(),
            bt_ptr, sl_ptr,
            topk_indices.data_ptr<int32_t>(),
            N_max, max_num_pages);
    } else if (N_max <= TOPK_SMALL_CAPACITY) {
        block_topk_translate_small_kernel<<<B, TOPK_SMALL_THREADS, 0, stream>>>(
            final_scores.data_ptr<float>(),
            bt_ptr, sl_ptr,
            topk_indices.data_ptr<int32_t>(),
            N_max, max_num_pages);
    } else {
        block_topk_translate_kernel<<<B, TOPK_BLOCK_THREADS, 0, stream>>>(
            final_scores.data_ptr<float>(),
            bt_ptr, sl_ptr,
            topk_indices.data_ptr<int32_t>(),
            N_max, max_num_pages);
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_topk_c", &launch_topk_c, "DSA TopK FP8 indexer (Tier 3 + Option C + Tier 2(c))");
}
