// clustered_kernel.cu — sm_100a cluster + DSM learning exercise.
//
// Based on kernel_current_baseline.cu (0.038 ms / 128/128 PASS).
// Goal: fuse scoring + top-K into a single kernel using a 2-CTA cluster and
// DSM for cluster-level top-K merge. Eliminates both final_scores[B, N_max]
// GMEM round trip AND the separate top-K kernel launch.
//
// Design:
//   Grid:           (B * CLUSTER_SIZE, 1, 1)  — each cluster handles one batch
//   Cluster:        __cluster_dims__(2, 1, 1) — 2 paired CTAs
//   Per CTA:        128 threads, handles max_num_pages/2 pages (with tail-handling)
//   Per-CTA scores: accumulated in SMEM run_scores_smem[CAP_PER_CTA] where
//                   CAP_PER_CTA = ceil(max_num_pages/2) * PAGE_SIZE (up to ~2880)
//   Sync:           cluster.sync() (barrier.cluster.sync.aligned PTX)
//   DSM:            CTA 0 uses mapa.shared::cluster to read CTA 1's run_scores_smem
//   Top-K:          CTA 0 runs CUB BlockRadixSort on combined 5760 scores,
//                   produces top-K, translates via block_table, writes topk_indices
//
// Deferred (v2): DSM-based Q sharing (CTA 0 loads Q, CTA 1 reads via DSM).
//                Unclear whether tcgen05.mma SMEM descriptors accept DSM
//                shared-cluster addresses. Each CTA loads its own Q for MVP.
//
// Known experimental areas:
//   - cluster.sync semantics on sm_100a with concurrent tcgen05 lifecycle
//   - mapa.shared::cluster addressing for DSM reads into CUB-style BlockLoad
//   - cluster launch via __cluster_dims__ attribute vs cudaLaunchKernelEx

#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>

namespace cg = cooperative_groups;

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int PAGE_DATA_BYTES = 8192;
constexpr int PAGE_BYTES      = 8448;
constexpr int TOPK            = 2048;
constexpr int H               = 64;

// Cluster config
constexpr int CLUSTER_SIZE    = 2;
constexpr int MAX_PAGES_PER_CTA_CLUSTER = 48;   // ceil(96/2) covers max_num_pages up to 96
constexpr int MAX_SCORES_PER_CTA = MAX_PAGES_PER_CTA_CLUSTER * PAGE_SIZE;  // 3072 scores

constexpr int MMA_M           = 64;
constexpr int MMA_N           = 64;
constexpr int MMA_K           = 32;
constexpr int NUM_SLABS       = INDEX_HEAD_DIM / MMA_K;  // 4
constexpr int SCORES_STRIDE   = 65;
constexpr uint32_t IDESC_VAL  = (1u << 4) | (8u << 17) | (4u << 24);

// CUB top-K config for combined 5760-score cluster merge.
// 128 threads × 48 items = 6144 cap (covers max 2 × 2880 = 5760).
// Register pressure: 48 × 4B keys + 48 × 4B vals = 384 B/thread = ~96 regs.
// Only CTA 0 runs this; with launch_bounds(128, 1) the compiler should fit.
constexpr int MERGE_THREADS   = 128;
constexpr int MERGE_ITEMS     = 48;
constexpr int MERGE_CAPACITY  = MERGE_THREADS * MERGE_ITEMS;  // 6144

// SMEM layout (all 16-B aligned)
constexpr int C_Q_OFFSET          = 0;        // 8192 B
constexpr int C_K0_OFFSET         = 8192;     // 8192 B
constexpr int C_K1_OFFSET         = 16384;    // 8192 B
constexpr int C_SCORES_OFFSET     = 24576;    // 16640 B ([64 × 65] FP32)
constexpr int C_SCALE0_OFFSET     = 41216;    // 256 B
constexpr int C_SCALE1_OFFSET     = 41472;    // 256 B
// Opt-#2 cache larger block_table since cluster spans all pages of one batch.
constexpr int C_BT_CACHE_OFFSET   = 41728;    // 128 × 4 = 512 B
constexpr int C_ALLOC_OFFSET      = 42240;    // 16 B
constexpr int C_MBAR_OFFSET       = 42256;    // 16 B
constexpr int C_WEIGHTS_OFFSET    = 42272;    // 256 B
// Running local scores buffer — visible to the OTHER cluster CTA via DSM.
// 3072 × 4 B = 12288 B.
constexpr int C_RUNSCORES_OFFSET  = 42528;    // 12288 B
constexpr int C_SMEM_BYTES        = 54816;    // ~53.5 KB per CTA

constexpr int SLAB_BYTES = 2048;
constexpr int SBO_BYTES  = 256;
constexpr int LBO_BYTES  = 128;

__device__ __forceinline__ int smem_8xT_offset(int m, int k) {
    return (k / 32) * SLAB_BYTES
         + (m / 8)  * SBO_BYTES
         + ((k % 32) / 16) * LBO_BYTES
         + (m % 8)  * 16
         + (k % 16);
}

__device__ __forceinline__ uint64_t desc_encode_u64(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr_shared) {
    uint64_t d = 0;
    d |= desc_encode_u64((uint64_t)smem_addr_shared);
    d |= desc_encode_u64((uint64_t)LBO_BYTES) << 16;
    d |= desc_encode_u64((uint64_t)SBO_BYTES) << 32;
    d |= (uint64_t)0b001ULL << 46;
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

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(
    uint32_t taddr, uint32_t (&r)[32])
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

// DSM: translate a local SMEM address to one that targets another cluster CTA's SMEM.
// `mapa.shared::cluster` converts (local smem addr, target block rank) → cluster-scoped addr.
__device__ __forceinline__ uint32_t mapa_shared_cluster(uint32_t local_smem_addr,
                                                         int target_block_rank) {
    uint32_t dsm_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(dsm_addr)
                 : "r"(local_smem_addr), "r"(target_block_rank));
    return dsm_addr;
}

// Read FP32 from cluster-shared SMEM (DSM). Address is the u32 cluster-shared
// form returned by mapa.shared::cluster. Cannot be accessed via normal pointer
// dereference — must use ld.shared::cluster explicitly.
__device__ __forceinline__ float ld_dsm_f32(uint32_t cluster_smem_addr) {
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n"
                 : "=f"(v)
                 : "r"(cluster_smem_addr));
    return v;
}

// =====================================================================
// Clustered kernel: 2 CTAs per batch, DSM-mediated top-K merge.
// Grid: (B * CLUSTER_SIZE, 1, 1)   — cluster_id = blockIdx.x / CLUSTER_SIZE = batch
// =====================================================================
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__global__ __launch_bounds__(128, 1)
void clustered_fp8_mma_topk_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    int32_t*       __restrict__ topk_indices,   // [B, TOPK] written directly
    int max_num_pages)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int block_rank = cluster.block_rank();   // 0 or 1

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* Q_smem          = smem_raw + C_Q_OFFSET;
    uint8_t* K_smem_buf[2]   = { smem_raw + C_K0_OFFSET, smem_raw + C_K1_OFFSET };
    float*   scores_smem     = reinterpret_cast<float*>(smem_raw + C_SCORES_OFFSET);
    float*   scale_smem_buf[2] = {
        reinterpret_cast<float*>(smem_raw + C_SCALE0_OFFSET),
        reinterpret_cast<float*>(smem_raw + C_SCALE1_OFFSET) };
    int32_t* bt_cache_smem   = reinterpret_cast<int32_t*>(smem_raw + C_BT_CACHE_OFFSET);
    uint32_t* alloc_slot_ptr = reinterpret_cast<uint32_t*>(smem_raw + C_ALLOC_OFFSET);
    float*   weights_smem    = reinterpret_cast<float*>(smem_raw + C_WEIGHTS_OFFSET);
    float*   run_scores_smem = reinterpret_cast<float*>(smem_raw + C_RUNSCORES_OFFSET);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int b       = blockIdx.x / CLUSTER_SIZE;
    const int seq_len_b = seq_lens[b];

    // Page range for this CTA within the cluster.
    const int half = (max_num_pages + 1) / 2;
    const int page_start = block_rank * half;
    const int page_end   = min(page_start + half, max_num_pages);
    const int my_pages   = max(0, page_end - page_start);

    const uint32_t q_shared       = __cvta_generic_to_shared(Q_smem);
    const uint32_t k_shared_buf[2] = {
        __cvta_generic_to_shared(K_smem_buf[0]),
        __cvta_generic_to_shared(K_smem_buf[1]) };
    const uint32_t alloc_slot_s   = __cvta_generic_to_shared(alloc_slot_ptr);
    const uint32_t mbar_s         = __cvta_generic_to_shared(smem_raw + C_MBAR_OFFSET);
    const uint32_t run_scores_s   = __cvta_generic_to_shared(run_scores_smem);

    // ---- TMEM alloc (warp 1) ----
    if (warp_id == 1) {
        tcgen05_alloc_64(alloc_slot_s);
    }
    __syncthreads();
    const uint32_t tmem_col = alloc_slot_ptr[0];

    // ---- block_table cache: cooperative load up to MAX of 128 pages ----
    if (warp_id == 0 && lane_id == 0) {
        mbarrier_init_1(mbar_s);
    }
    for (int p = tid; p < max_num_pages; p += 128) {
        bt_cache_smem[p] = block_table[b * max_num_pages + p];
    }
    if (tid < H) {
        weights_smem[tid] = weights[b * H + tid];
    }
    uint32_t phase = 0;
    __syncthreads();

    // ---- Q load via cp.async (each CTA loads its own Q for MVP) ----
    {
        const uint8_t* q_base = q_fp8 + size_t(b) * 64 * INDEX_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int u        = tid * 4 + i;
            int m        = u >> 3;
            int k_uint4  = u & 7;
            int k_base   = k_uint4 << 4;
            int off      = smem_8xT_offset(m, k_base);
            cp_async_16B(q_shared + off,
                         q_base + m * INDEX_HEAD_DIM + k_base);
        }
    }
    cp_async_commit();

    // ---- Page loop: only my_pages iterations on this CTA's range ----
    if (my_pages > 0) {
        // Prologue: load page_start into K_smem_buf[0]
        {
            const int32_t phys0 = bt_cache_smem[page_start];
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

        for (int pidx = 0; pidx < my_pages; ++pidx) {
            const int curr_buf = pidx & 1;
            const int next_buf = curr_buf ^ 1;
            const int next_pidx = pidx + 1;
            const bool has_next = (next_pidx < my_pages);
            const int page_id = page_start + pidx;

            // Issue MMA on curr_buf
            if (warp_id == 0 && elect_one_sync()) {
                const uint32_t tmem_addr = tmem_col;
                #pragma unroll 1
                for (int s = 0; s < NUM_SLABS; ++s) {
                    const uint32_t q_slab = q_shared + s * SLAB_BYTES;
                    const uint32_t k_slab = k_shared_buf[curr_buf] + s * SLAB_BYTES;
                    const uint64_t a_desc = make_smem_desc(q_slab);
                    const uint64_t b_desc = make_smem_desc(k_slab);
                    const int enable_d = (s != 0) ? 1 : 0;
                    tcgen05_mma_f8f6f4(tmem_addr, a_desc, b_desc, IDESC_VAL, enable_d);
                }
                tcgen05_commit_mbar(mbar_s);
            }

            // Overlap: cp.async next K + scale
            if (has_next) {
                const int page_id_n = page_start + next_pidx;
                const int32_t phys_n  = bt_cache_smem[page_id_n];
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

            mbarrier_wait_phase(mbar_s, phase);
            phase ^= 1;
            tcgen05_fence_after_sync();

            const uint32_t taddr_lo = (uint32_t(warp_id) * 32u) << 16 | tmem_col;
            const uint32_t taddr_hi = (uint32_t(warp_id) * 32u) << 16 | (tmem_col + 32u);
            uint32_t rlo[32], rhi[32];
            tcgen05_ld_32x32b_x32(taddr_lo, rlo);
            tcgen05_ld_32x32b_x32(taddr_hi, rhi);
            tcgen05_wait_ld();

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
            __syncthreads();

            // H-reduction: write to LOCAL run_scores_smem (cluster-visible via DSM).
            if (tid < PAGE_SIZE) {
                const int col = tid;
                const int n_global = page_id * PAGE_SIZE + col;
                // Local slot: pidx * PAGE_SIZE + col — per-CTA buffer.
                const int local_slot = pidx * PAGE_SIZE + col;
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
                run_scores_smem[local_slot] = (n_global >= seq_len_b) ? -INFINITY : acc;
            }

            if (has_next) {
                cp_async_wait_all();
            }
        }
    }

    // ---- TMEM dealloc before cluster sync ----
    if (warp_id == 1) {
        tcgen05_dealloc_64(tmem_col);
    }
    __syncthreads();

    // ---- Cluster sync: ensure both CTAs finished writing run_scores_smem ----
    cluster.sync();

    // ---- Only block_rank 0 does the merge + top-K + translate + output ----
    if (block_rank == 0) {
        // Compute the total number of scores across cluster:
        // my_pages (this CTA) + peer_pages (other CTA) = max_num_pages total.
        const int peer_pages = max(0, max_num_pages - half);  // pages on CTA 1
        const int total_scores = (my_pages + peer_pages) * PAGE_SIZE;  // ≤ max_num_pages*64 = N_max

        // DSM base address of CTA 1's run_scores_smem.
        const uint32_t peer_run_scores_base = mapa_shared_cluster(run_scores_s, 1);

        using BlockRadixSortT = cub::BlockRadixSort<float, MERGE_THREADS,
                                                    MERGE_ITEMS, int32_t>;
        __shared__ typename BlockRadixSortT::TempStorage sort_temp;

        // Gather keys from two sources:
        //   pos in [0, my_count)            → local run_scores_smem[pos]
        //   pos in [my_count, my+peer)      → peer run_scores_smem[pos - my_count] via DSM
        //   pos beyond                      → -INFINITY padding
        float keys[MERGE_ITEMS];
        int32_t vals[MERGE_ITEMS];
        const int my_count   = my_pages * PAGE_SIZE;
        const int peer_count = peer_pages * PAGE_SIZE;

        #pragma unroll
        for (int i = 0; i < MERGE_ITEMS; ++i) {
            int pos = tid * MERGE_ITEMS + i;
            float key;
            int32_t val;
            if (pos < my_count) {
                key = run_scores_smem[pos];
                val = page_start * PAGE_SIZE + pos;
            } else if (pos < my_count + peer_count) {
                int peer_pos = pos - my_count;
                // DSM load: address = peer_base + peer_pos * sizeof(float)
                uint32_t peer_addr = peer_run_scores_base + uint32_t(peer_pos) * 4u;
                key = ld_dsm_f32(peer_addr);
                val = half * PAGE_SIZE + peer_pos;
            } else {
                key = -INFINITY;
                val = -1;
            }
            keys[i] = key;
            vals[i] = val;
        }

        BlockRadixSortT(sort_temp).SortDescending(keys, vals);

        const int actual_topk = (seq_len_b < TOPK) ? seq_len_b : TOPK;

        // Each thread writes MERGE_ITEMS ranks. Cover all TOPK=2048 ranks with 128 threads × 16 items.
        // But MERGE_ITEMS=48, so we only cover 6144 ranks. Ranks > TOPK ignored.
        #pragma unroll
        for (int i = 0; i < MERGE_ITEMS; ++i) {
            int rank = tid * MERGE_ITEMS + i;
            if (rank < TOPK) {
                int32_t out_val;
                if (rank < actual_topk) {
                    int32_t lid = vals[i];
                    if (lid >= 0) {
                        int32_t phys = bt_cache_smem[lid >> 6];
                        out_val = phys * PAGE_SIZE + (lid & 63);
                    } else {
                        out_val = -1;
                    }
                } else {
                    out_val = -1;
                }
                topk_indices[b * TOPK + rank] = out_val;
            }
        }
    }

    // Final cluster sync: CTA 1 must not exit before CTA 0 finishes DSM reads of
    // its run_scores_smem. Without this, CTA 1's SMEM could be deallocated mid-read.
    cluster.sync();
}

// =====================================================================
// Host entry point.
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
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(max_num_pages <= 128, "max_num_pages exceeds bt_cache capacity (128)");

    auto q_u8       = q_index_fp8.view(torch::kUInt8);
    auto k_cache_u8 = k_index_cache_fp8.view(torch::kUInt8);

    const uint8_t* q_ptr       = q_u8.data_ptr<uint8_t>();
    const uint8_t* k_cache_ptr = k_cache_u8.data_ptr<uint8_t>();
    const float*   w_ptr       = weights.data_ptr<float>();
    const int32_t* bt_ptr      = block_table.data_ptr<int32_t>();
    const int32_t* sl_ptr      = seq_lens.data_ptr<int32_t>();

    // Opt in to dynamic SMEM above 48 KB.
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaFuncSetAttribute(
            (const void*)&clustered_fp8_mma_topk_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            C_SMEM_BYTES);
        smem_attr_set = true;
    }

    // Grid: B clusters × CLUSTER_SIZE CTAs each.
    dim3 grid(B * CLUSTER_SIZE, 1, 1);
    clustered_fp8_mma_topk_kernel<<<grid, 128, C_SMEM_BYTES, stream>>>(
        q_ptr, k_cache_ptr, w_ptr, bt_ptr, sl_ptr,
        topk_indices.data_ptr<int32_t>(),
        max_num_pages);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_topk_c", &launch_topk_c,
          "DSA TopK FP8 indexer (clustered fused scoring+topk MVP)");
}
