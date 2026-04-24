// ==========================================================================
// sbo_lbo_sweep_hostloop.cu
//
// Host-side sweep: launches a SEPARATE kernel per (sbo_enc, lbo_enc)
// combination. Each launch gets a fresh CTA, fresh TMEM allocation,
// and a fresh mbarrier — eliminating the cross-iteration state
// contamination that caused the in-kernel sweep to report uniform
// error 33.2562 across all combinations.
//
// Uses the canary's proven pipeline:
//   fill SMEM → build desc → MMA → commit → mbar wait → fence → ld
//
// Input pattern: A[m][k] = 1.0 (E4M3), B[n][k] = (n+1) (E4M3)
// Expected output: D[m][n] = K * (n+1) = 32 * (n+1)
//   → row0 = {32, 64, 96, 128, 160, 192, 224, 256}
//
// COMPILE:
//   nvcc -O2 -arch=sm_100a -std=c++17 -o sweep_host sbo_lbo_sweep_hostloop.cu
//
// RUN:
//   ./sweep_host
//
// OUTPUT: JSON summary to stdout + per-combo diagnostics to stderr
// ==========================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

// ── Tile geometry ───────────────────────────────────────────────────────────
constexpr int M = 64;
constexpr int N = 64;
constexpr int K = 32;
constexpr int T = 16;              // fp8 elements per 128-bit K-tile
constexpr int SMEM_TILE = M * K;   // 2048 bytes per operand
constexpr uint32_t TMEM_NCOLS = 64;

// E4M3 encodings
constexpr uint8_t E4M3_ONE = 0x38;  // 1.0

// E4M3 encoding for small integers (1..8)
// E4M3: sign(1) exp(4) man(3), bias=7
// val = 2^(exp-7) * (1 + man/8)
//   1.0 = 0 0111 000 = 0x38
//   2.0 = 0 1000 000 = 0x40
//   3.0 = 0 1000 100 = 0x44
//   4.0 = 0 1001 000 = 0x48
//   5.0 = 0 1001 010 = 0x4A
//   6.0 = 0 1001 100 = 0x4C
//   7.0 = 0 1001 110 = 0x4E
//   8.0 = 0 1010 000 = 0x50
__constant__ uint8_t E4M3_TABLE[9] = {
    0x00,  // 0 (unused)
    0x38,  // 1.0
    0x40,  // 2.0
    0x44,  // 3.0
    0x48,  // 4.0
    0x4A,  // 5.0
    0x4C,  // 6.0
    0x4E,  // 7.0
    0x50,  // 8.0
};

// ── Result struct (copied back to host per launch) ──────────────────────────
struct SweepResult {
    float row0[64];    // first row of TMEM output
    int   non_zero;    // count of non-zero elements in row0
};

// ── SMEM layout ─────────────────────────────────────────────────────────────
//   [0      .. 2047]  smem_a
//   [2048   .. 4095]  smem_b
//   [4096   .. 4099]  tmem_addr slot (4 B)
//   [4104   .. 4111]  mbarrier (8 B, 8-byte aligned)
//   [4112   .. 4367]  result staging (256 B = 64 floats)
constexpr int SMEM_TOTAL = 2048 + 2048 + 4 + 4 + 8 + 256;

// ── PTX helpers (same as canary) ────────────────────────────────────────────

__device__ __forceinline__
void tcgen05_alloc(uint32_t smem_dst, uint32_t ncols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_dst), "r"(ncols));
}

__device__ __forceinline__
void tcgen05_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(ncols));
}

__device__ __forceinline__
void tcgen05_mma(uint32_t taddr, uint64_t a_desc, uint64_t b_desc,
                 uint32_t idesc, bool accum) {
    uint32_t p = accum ? 1u : 0u;
    asm volatile(
        "{ .reg .pred pd;\n\t"
        "  setp.ne.u32 pd, %4, 0;\n\t"
        "  tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, pd;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p)
        : "memory");
}

__device__ __forceinline__
void tcgen05_commit(uint32_t mbar_smem_addr, uint32_t taddr, uint32_t tx_bytes) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::complete_tx::bytes [%0], %1, %2;"
        :: "r"(mbar_smem_addr), "r"(taddr), "r"(tx_bytes)
        : "memory");
}

__device__ __forceinline__
void mbar_init(uint32_t mbar_smem_addr, uint32_t expected_arrivals) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_smem_addr), "r"(expected_arrivals));
}

__device__ __forceinline__
void mbar_wait(uint32_t mbar_smem_addr, uint32_t parity) {
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  LAB_WAIT:\n\t"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "  @!p bra LAB_WAIT;\n\t"
        "}"
        :: "r"(mbar_smem_addr), "r"(parity)
        : "memory");
}

__device__ __forceinline__ void fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_ld_x8(float out[8], uint32_t addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]),"=f"(out[1]),"=f"(out[2]),"=f"(out[3]),
          "=f"(out[4]),"=f"(out[5]),"=f"(out[6]),"=f"(out[7])
        : "r"(addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ bool elect_one() {
    uint32_t r;
    asm volatile(
        "{ .reg .pred p;\n\t"
        "  elect.sync _|p, 0xFFFFFFFF;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}" : "=r"(r));
    return r != 0u;
}

// ── Descriptor builder ──────────────────────────────────────────────────────
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr,
                   uint32_t sbo_enc,
                   uint32_t lbo_enc,
                   uint32_t swizzle) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);
    d |= (uint64_t)(lbo_enc     & 0x3FFFu) << 16;
    d |= (uint64_t)(sbo_enc     & 0x3FFFu) << 32;
    d |= (uint64_t)(1u)                    << 46;     // strided mode
    d |= (uint64_t)((addr >> 7) & 0x7u)    << 49;
    d |= (uint64_t)(swizzle     & 0x7u)    << 61;
    return d;
}

// ── SMEM fill: 8×T core-tile layout ────────────────────────────────────────
//
// A: all 1.0  →  every byte = E4M3_ONE
// B: B[n][k] = (n/8 + 1), i.e. each group of 8 rows has the same value.
//    This gives a distinctive output pattern: D[m][n] = K * (n/8 + 1)
//    which for M_GRPS=8 groups means row0 = {32,32,...,64,64,...,...,256,...}
//
// SIMPLIFIED: since all A values are 1.0, we just fill every byte of
// smem_a with E4M3_ONE. For B we use the core-tile layout with the
// parameterised sbo_bytes / lbo_bytes strides.

__device__ void fill_A(uint8_t* smem_a, uint32_t sbo_bytes, uint32_t lbo_bytes) {
    int tid = threadIdx.x;
    // A = all 1.0 in 8×T layout. Since every byte is the same value,
    // the layout doesn't actually matter — just fill everything.
    for (int i = tid; i < SMEM_TILE; i += blockDim.x)
        smem_a[i] = E4M3_ONE;
}

__device__ void fill_B(uint8_t* smem_b, uint32_t sbo_bytes, uint32_t lbo_bytes) {
    int tid = threadIdx.x;
    // Zero first
    for (int i = tid; i < SMEM_TILE; i += blockDim.x)
        smem_b[i] = 0;
    __syncthreads();

    // B[n][k] in 8×T core-tile layout
    // Value = (n+1) clamped to [1..8] for E4M3 representability
    // Actually: group = n/8, value = group+1
    for (int i = tid; i < N * K; i += blockDim.x) {
        int n = i / K;
        int k = i % K;

        int grp    = n / 8;
        int in_grp = n % 8;
        int k_tile = k / T;
        int k_in_t = k % T;

        uint32_t off = grp * sbo_bytes
                     + k_tile * lbo_bytes
                     + in_grp * T
                     + k_in_t;

        // Value for this row: grp+1 (1..8)
        uint8_t val = E4M3_TABLE[grp + 1];

        if (off < SMEM_TILE)
            smem_b[off] = val;
    }
}

// ── Kernel: one MMA, writes result to global memory ─────────────────────────
__global__ void sweep_kernel(uint32_t sbo_enc, uint32_t lbo_enc,
                             uint32_t sbo_bytes, uint32_t lbo_bytes,
                             SweepResult* d_result) {
    extern __shared__ uint8_t smem[];
    uint8_t*  smem_a    = smem;
    uint8_t*  smem_b    = smem + 2048;
    uint32_t* tmem_smem = (uint32_t*)(smem + 4096);
    uint32_t* mbar_smem = (uint32_t*)(smem + 4104);
    float*    result_staging = (float*)(smem + 4112);

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    uint32_t tmem_smem_addr = (uint32_t)__cvta_generic_to_shared(tmem_smem);
    uint32_t mbar_smem_addr = (uint32_t)__cvta_generic_to_shared(mbar_smem);

    // ── Allocate TMEM ──
    if (warp_id == 1)
        tcgen05_alloc(tmem_smem_addr, TMEM_NCOLS);
    __syncthreads();
    uint32_t taddr = *tmem_smem;

    // ── Fill SMEM ──
    fill_A(smem_a, sbo_bytes, lbo_bytes);
    __syncthreads();
    fill_B(smem_b, sbo_bytes, lbo_bytes);
    __syncthreads();

    // ── Build descriptors ──
    uint64_t a_desc = make_desc(smem_a, sbo_enc, lbo_enc, /*swizzle=*/0);
    uint64_t b_desc = make_desc(smem_b, sbo_enc, lbo_enc, /*swizzle=*/0);

    // ── Init mbarrier ──
    if (tid == 0)
        mbar_init(mbar_smem_addr, 1u);
    __syncthreads();

    // ── MMA → commit ──
    if (warp_id == 0 && elect_one()) {
        tcgen05_mma(taddr, a_desc, b_desc, 0x04100010u, /*accum=*/false);
        tcgen05_commit(mbar_smem_addr, taddr, M * N * 4);
    }

    // ── Wait → fence ──
    mbar_wait(mbar_smem_addr, 0u);
    fence_after();

    // ── Read TMEM row 0 (warp 0, lane 0) into shared staging ──
    // Zero staging first
    for (int i = tid; i < 64; i += blockDim.x)
        result_staging[i] = -999.0f;
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        uint32_t row_off = 0u << 16;  // row 0
        for (int g = 0; g < 8; g++) {
            float regs[8];
            tcgen05_ld_x8(regs, taddr + row_off + (uint32_t)(g * 8));
            for (int x = 0; x < 8; x++)
                result_staging[g * 8 + x] = regs[x];
        }
    }
    tcgen05_wait_ld();
    __syncthreads();

    // ── Thread 0: copy to global result ──
    if (tid == 0) {
        int nz = 0;
        for (int i = 0; i < 64; i++) {
            d_result->row0[i] = result_staging[i];
            if (result_staging[i] != 0.0f && result_staging[i] != -999.0f)
                nz++;
        }
        d_result->non_zero = nz;
    }
    __syncthreads();

    // ── Deallocate TMEM ──
    if (warp_id == 1)
        tcgen05_dealloc(taddr, TMEM_NCOLS);
}

// ── Expected output: D[0][n] = K * (n/8 + 1) ───────────────────────────────
// n=0..7   → 32*1 = 32
// n=8..15  → 32*2 = 64
// n=16..23 → 32*3 = 96
// n=24..31 → 32*4 = 128
// n=32..39 → 32*5 = 160
// n=40..47 → 32*6 = 192
// n=48..55 → 32*7 = 224
// n=56..63 → 32*8 = 256
float expected_row0(int n) {
    return (float)(K * (n / 8 + 1));
}

// ── Host ────────────────────────────────────────────────────────────────────
int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // Allocate device result buffer
    SweepResult* d_result;
    cudaMalloc(&d_result, sizeof(SweepResult));

    SweepResult h_result;

    cudaFuncSetAttribute(sweep_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    // ── Sweep ranges ────────────────────────────────────────────────────
    // SBO_enc = sbo_bytes / 16.  For 8×T layout: sbo_bytes = M_GRP_STRIDE
    // LBO_enc = lbo_bytes / 16.  For 8×T layout: lbo_bytes = K_TILE_STRIDE
    //
    // Known-correct: sbo_enc=16 (256B), lbo_enc=8 (128B)
    // Sweep a range around it to confirm uniqueness.

    uint32_t sbo_encs[] = {4, 8, 12, 16, 20, 24, 32};
    uint32_t lbo_encs[] = {1, 2, 4, 8, 12, 16};
    int n_sbo = sizeof(sbo_encs) / sizeof(sbo_encs[0]);
    int n_lbo = sizeof(lbo_encs) / sizeof(lbo_encs[0]);

    float best_error = 1e30f;
    uint32_t best_sbo = 0, best_lbo = 0;
    int n_pass = 0;

    fprintf(stderr, "\nSweeping %d SBO × %d LBO = %d combinations...\n\n",
            n_sbo, n_lbo, n_sbo * n_lbo);

    for (int si = 0; si < n_sbo; si++) {
        for (int li = 0; li < n_lbo; li++) {
            uint32_t sbo_enc = sbo_encs[si];
            uint32_t lbo_enc = lbo_encs[li];
            uint32_t sbo_bytes = sbo_enc * 16;
            uint32_t lbo_bytes = lbo_enc * 16;

            // Skip if layout would overflow SMEM_TILE
            // Largest offset: (M/8 - 1)*sbo_bytes + (K/T - 1)*lbo_bytes + 8*T
            uint32_t max_off = 7 * sbo_bytes + 1 * lbo_bytes + 8 * T;
            if (max_off > SMEM_TILE) {
                fprintf(stderr, "  sbo=%3u lbo=%3u  SKIP (overflow: max_off=%u > %d)\n",
                        sbo_enc, lbo_enc, max_off, SMEM_TILE);
                continue;
            }

            // Launch one kernel
            sweep_kernel<<<1, 128, SMEM_TOTAL>>>(
                sbo_enc, lbo_enc, sbo_bytes, lbo_bytes, d_result);

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "  sbo=%3u lbo=%3u  CUDA ERROR: %s\n",
                        sbo_enc, lbo_enc, cudaGetErrorString(err));
                continue;
            }

            cudaMemcpy(&h_result, d_result, sizeof(SweepResult),
                       cudaMemcpyDeviceToHost);

            // Compute L2 error against expected
            double sum_sq = 0.0;
            for (int n = 0; n < 64; n++) {
                double diff = (double)h_result.row0[n] - (double)expected_row0(n);
                sum_sq += diff * diff;
            }
            float l2 = (float)sqrt(sum_sq);

            bool pass = (l2 < 1e-3f);
            if (pass) n_pass++;

            if (l2 < best_error) {
                best_error = l2;
                best_sbo = sbo_enc;
                best_lbo = lbo_enc;
            }

            fprintf(stderr, "  sbo=%3u lbo=%3u  err=%.4f  nz=%d  %s",
                    sbo_enc, lbo_enc, l2, h_result.non_zero,
                    pass ? "PASS" : "");

            // Print first 8 values for quick visual check
            fprintf(stderr, "  row0[0:7]={");
            for (int i = 0; i < 8; i++)
                fprintf(stderr, "%.0f%s", h_result.row0[i], i < 7 ? "," : "");
            fprintf(stderr, "}\n");
        }
    }

    fprintf(stderr, "\n── Summary ──\n");
    fprintf(stderr, "  Combinations tested: %d\n", n_sbo * n_lbo);
    fprintf(stderr, "  Passed (err < 1e-3): %d\n", n_pass);
    fprintf(stderr, "  Best: sbo=%u lbo=%u err=%.6f\n", best_sbo, best_lbo, best_error);

    // ── JSON output to stdout (for run_diagnostic.py parsing) ───────────
    printf("{\n");
    printf("  \"best_error\": %.6f,\n", best_error);
    printf("  \"best_sbo_enc\": %u,\n", best_sbo);
    printf("  \"best_lbo_enc\": %u,\n", best_lbo);
    printf("  \"n_pass\": %d,\n", n_pass);
    printf("  \"n_tested\": %d,\n", n_sbo * n_lbo);
    printf("  \"best_row0_first8\": [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           h_result.row0[0], h_result.row0[1], h_result.row0[2], h_result.row0[3],
           h_result.row0[4], h_result.row0[5], h_result.row0[6], h_result.row0[7]);
    printf("}\n");

    cudaFree(d_result);
    return (best_error < 1e-3f) ? 0 : 1;
}
