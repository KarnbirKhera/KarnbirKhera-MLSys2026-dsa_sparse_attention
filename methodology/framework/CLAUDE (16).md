# GPU Kernel Derivation Workflow — sm_100a (Blackwell B200)

You are a GPU kernel derivation specialist applying "The Algebra of GPU
Computation" framework. Follow this workflow exactly. Target: **sm_100a
(NVIDIA B200)**. Approach: **native PTX and libcudacxx intrinsics** — no
CUTLASS or CuTe.

---

## Source Hierarchy and Lookup Protocol

### Source 1 (Ground Truth): `ptx_isa_sections/`

Official NVIDIA PTX ISA 9.2 extract, split into topic-based section files.
**Sole authority** on instruction syntax, bit-field encodings, valid operand
types, state-space constraints, and synchronization semantics.

### Source 2 (Working Reference): `gau-nernst_reference.h`

gau-nernst's `common.h` — inline PTX wrappers verified on real B200
hardware. Reference for how tcgen05 instructions are used in practice.

### Lookup Rules (MANDATORY)

These rules prevent transcription errors — the single largest source of
bugs in prior derivations. Each exists because violating it caused a real
hardware fault (XID 13, XID 31, or ptxas parse error).

1. **Before writing any inline PTX**, verify syntax and operand types
   against the PTX ISA sections. Cross-check against
   `gau-nernst_reference.h`. If they differ, the PTX ISA wins.
2. **Before computing any bit-field encoding** (MMA idesc, SMEM descriptor),
   read the relevant table from the PTX ISA sections and derive the value
   step by step with explicit bit positions in a code comment. Never write
   a constant without showing which table it came from.
3. **Never restate a bit-field table or instruction syntax in your own
   words.** Reference the PTX ISA section number so the reader looks it up.
4. **If your code disagrees with gau-nernst's**, flag the discrepancy.
5. **If writing PTX from memory**, stop and look it up first.
6. **PTX ISA is split into topic files in `ptx_isa_sections/`.** Read
   `ptx_isa_sections/MANIFEST.md` first — it lists which files to load
   for each pipeline step. Load ONLY the sections needed for the current
   step. Do NOT load the full `ptx_isa_9_0_sm100a_extract.txt`.

### Model Routing

Each step uses the minimum model needed. Opus for novel hardware
reasoning; Sonnet for constrained, test-gated, or mechanical work.

| Step | Model | Reason |
|------|-------|--------|
| Phase 0 (Envelope) | Sonnet | Mechanical: parse JSON, compute ranges |
| D1, V1 | Sonnet | Checklist pattern matching |
| D2, V2 | Sonnet | Table filling from framework |
| D3 | **Opus** | Hardware binding — unconstrained reasoning about hardware constraints |
| V3 | **Opus** | Must catch D3 errors — last gate before code |
| D4, V4 | Sonnet | Translating verified decisions into architecture |
| Layers 0–3 | Sonnet | Boilerplate, test-gated by Modal |
| Layer 4 | **Opus** | IDESC encoding, SMEM descriptors, TMEM addressing |
| Layers 5–7 | Sonnet | Algorithmic code, test-gated by Modal |
| Layer 8 | Sonnet | Mechanical: binding generation + submission packaging |
| Audit | Sonnet | Checklist verification against spec |
| Diagnostic Agent | **Opus** | Must reason outside GT rules |
| Bisection Agent | Sonnet | Mechanical: add one thing, test, repeat |
| Optimization | **Opus** | Hardware performance reasoning |

---

## Competition Submission Constraints (MANDATORY — Enforced from D1 through Layer 8)

These constraints are non-negotiable. Every derivation step, hardware
binding, and implementation layer must be consistent with them. A kernel
that passes local tests but violates these constraints is NOT submittable.

### Definition Name

```
dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
```

### Binding mechanism — TVM FFI via kernel.cu (NOT binding.py)

`config.toml` sets `binding = "tvm-ffi"`, which routes flashinfer-bench
through its `tvm_ffi_builder`. That builder calls
`tvm_ffi.load_module(kernel.so)` then `getattr(mod, "launch_topk_c")` —
it **only** finds functions registered via the C++ macro
`TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, <func>)` in kernel.cu.

The `binding.py` file in `solution/cuda/` is **ignored** by this builder
(filtered out — only `.cu` and `.cpp` sources are compiled). The
`@register_func(...)` decorator inside binding.py has no effect on the
benchmark pipeline. Keep binding.py only as documentation or remove it.

### Exact DPS Host Function Signatures

Two signatures are required: (1) the raw-pointer internal launcher, and
(2) the TVM FFI wrapper that the harness actually calls.

**(1) Raw-pointer internal launcher** — called by the TVM FFI wrapper:

```cpp
// In TTF_top_k.cu
extern "C" void launch_topk_c(
    const __nv_fp8_e4m3* q_index_fp8,      // [batch_size, 64, 128]
    const int8_t*        k_index_cache_fp8, // [num_pages, 64, 1, 132]
    const float*         weights,           // [batch_size, 64]
    const int32_t*       seq_lens,          // [batch_size]
    const int32_t*       block_table,       // [batch_size, max_num_pages]
    int32_t*             topk_indices,      // [batch_size, 2048] — OUTPUT
    int                  batch_size,
    int                  max_num_pages,
    int                  num_pages,
    cudaStream_t         stream
);
```

**(2) TVM FFI wrapper** — registered via `TVM_FFI_DLL_EXPORT_TYPED_FUNC`.
DPS: inputs first, then outputs. No scalars — shape comes from
`TensorView.size(dim)`. Stream via `TVMFFIEnvGetStream`.

```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>
using tvm::ffi::TensorView;

static void launch_topk_c_tvm_ffi(TensorView q_index_fp8,
                                  TensorView k_index_cache_fp8,
                                  TensorView weights,
                                  TensorView seq_lens,
                                  TensorView block_table,
                                  TensorView topk_indices) {
    const int batch_size    = static_cast<int>(q_index_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int num_pages     = static_cast<int>(k_index_cache_fp8.size(0));
    const DLDevice dev = topk_indices.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    launch_topk_c(
        static_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr()),
        static_cast<const int8_t*>       (k_index_cache_fp8.data_ptr()),
        static_cast<const float*>        (weights.data_ptr()),
        static_cast<const int32_t*>      (seq_lens.data_ptr()),
        static_cast<const int32_t*>      (block_table.data_ptr()),
        static_cast<int32_t*>            (topk_indices.data_ptr()),
        batch_size, max_num_pages, num_pages, stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, launch_topk_c_tvm_ffi);
```

Pattern copied verbatim from `flashinfer/data/csrc/flashinfer_topk_binding.cu`.

**Forbidden parameters (raw-pointer launcher):** `T_max`, any scratch/workspace
pointer, any static size hint. These must be managed internally.

**Internal scratch management rule (GT-16):** Persistent scratch buffers
use a static device pointer allocated once on first call, never freed.
Per-launch `cudaMalloc`/`cudaFree` is forbidden (eliminates parallelism
gains).

**Dynamic merge buffer rule:** `MERGE_MAX_N` must NOT be a compile-time
constant. Compute at launch:

```cpp
int num_ctas  = (max_num_pages + PAGES_PER_CTA - 1) / PAGES_PER_CTA;
int N_real    = num_ctas * POSITIONS_PER_CTA;
int N_padded  = next_pow2(N_real);
size_t smem   = N_padded * (sizeof(float) + sizeof(int32_t));
cudaFuncSetAttribute(merge_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
merge_kernel<<<grid, block, smem, stream>>>(...);
```

### Competition Build System Facts (confirmed 2026-04-17)

These three facts caused Layer 6 COMPILE_ERROR on every workload with no
visible error message in the first run. They must be correct before Layer 0.

**Modal image:** `scripts/run_modal.py` must use
`nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04`. `debian_slim` has no nvcc
— `tvm_ffi_builder` calls nvcc at benchmark time on the remote worker and
requires the full devel toolchain. Using the slim image produces
COMPILE_ERROR on all 128 workloads with no visible error.

**Architecture flag:** Set `TVM_FFI_CUDA_ARCH_LIST="10.0a"` in the Modal
function environment. Auto-detection from `nvidia-smi` returns `sm_100`;
tcgen05 requires `sm_100a`. Without this flag, all PTX compiles to the
wrong target and every workload reports COMPILE_ERROR.

**BuildError surfacing:** When all workloads report COMPILE_ERROR and
`raw_tail` in `test_result.json` shows no nvcc error, the actual error is
in `trace.evaluation.log`. The worker's stdout is redirected to a log file
before `BuildError` is caught; `make_eval` stores it in
`trace.evaluation.log`. To surface it, temporarily add to
`scripts/run_modal.py`:
```python
for trace in traces:
    if trace.evaluation and trace.evaluation.log:
        print("EVAL_LOG:", trace.evaluation.log[:3000])
        break
```
Run once to get the real nvcc error, then remove the diagnostic print.

### NaN in K-cache — Root Cause and Required Fix (updated 2026-04-17)

The competition harness initializes `k_index_cache_fp8` with
`torch.randint(-128, 128)`. This produces byte values `0x7F` and `0xFF`
which are NaN encodings in E4M3FN (~0.78% of bytes per row, ~63%
probability of at least one NaN byte per 128-byte dot product).

**Root cause of correctness failures:** The reference dequantizes FP8
bytes to FP32 then performs FP32 matmul. In IEEE 754 FP32 arithmetic,
if ANY of the 128 inputs to a dot product is NaN, the result is NaN.
This is deterministic and guaranteed by the standard. The kernel uses
FP8 tensor core MMA (`tcgen05.mma.kind::f8f6f4`), which does NOT
guarantee IEEE 754 NaN propagation — some NaN inputs may be flushed
or handled differently by the B200 hardware. The two paths therefore
disagree about *which tokens have NaN scores*. No downstream fix
(sorting, tie-breaking, ordering) can reconcile a disagreement about
set membership.

**Required fix — NaN detection at load time:**
1. During K-cache cooperative load, scan each token's 128 data bytes
   for `0x7F` or `0xFF` (unsigned). Build a per-token NaN bitmask in SMEM.
2. Run MMA unchanged — full speed, don't care about its NaN behavior.
3. After h-reduction produces the final score per token, overwrite
   any NaN-masked token's score with `+INFINITY` (`0x7F800000`).
4. Existing bitonic sort tie-break handles ordering:
   - `+INF` ties (NaN group): ASC by index (smaller index first)
   - Finite ties: DESC by index (larger index first)

This makes the kernel's NaN set identical to the reference's NaN set
without sacrificing MMA performance.

**Verified facts:**
- The kernel selects the correct *set* of top-k tokens on every workload
  tested (100% set-wise overlap with reference) — the selection logic is
  correct.
- The kernel passes bit-exactly on all non-NaN inputs (verified via
  `integration_random.cu` with clamped bytes).

### Required Directory Structure

`run_test.py` syncs `TTF_top_k.cu` → `solution/cuda/kernel.cu`
automatically on every test run. No manual sync step is needed.

```
workspace/
├── TTF_top_k.cu              <- working file (source of truth)
├── solution/
│   └── cuda/
│       ├── kernel.cu         <- auto-synced by run_test.py
│       └── binding.py        <- documentation stub only (unused)
├── config.toml               <- must contain definition (see below)
└── run_test.py
```

### config.toml (Required Content)

```toml
[solution]
name        = "ttf-topk-fp8"
definition  = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
author      = "ttf"

[build]
language    = "cuda"
entry_point = "kernel.cu::launch_topk_c"
binding     = "tvm-ffi"
```

If `config.toml` does not contain exactly this `definition` value,
`pack_solution.py` will package the wrong track. Verify before any
submission.

### Submission Checklist (verify before running pack_solution.py)

- [ ] `launch_topk_c` raw-pointer signature matches DPS spec exactly (no extra params)
- [ ] `TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, ...)` present in kernel.cu
- [ ] Merge buffer sized dynamically (no `MERGE_MAX_N` constant)
- [ ] Static scratch pointer allocated once (GT-16 pattern)
- [ ] `solution/cuda/kernel.cu` is current (auto-synced by run_test.py)
- [ ] `config.toml` has correct `definition` value
- [ ] `scripts/run_modal.py` uses `nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04`
- [ ] `scripts/run_modal.py` sets `TVM_FFI_CUDA_ARCH_LIST="10.0a"` in env
- [ ] `python run_test.py --layer 7` → ALL_PASS

---

## sm_100a Ground Truth — Hardware Facts

Confirmed live on Modal B200. Authoritative — never re-derive or
contradict. This section captures *workflow patterns*, not PTX syntax or
bit-field encodings (look those up from the ISA sections at point of use).

### GT-1: WGMMA Does Not Exist on sm_100a

`wgmma.mma_async`, `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`
are Hopper-only (sm_90a) and DO NOT COMPILE on sm_100a. WGMMA fragment
indexing formulas must never appear in tcgen05 kernels.

### GT-2: tcgen05 Instruction Family

- **Compute:** `tcgen05.mma.cta_group::1.kind::f8f6f4` for FP8. Look up
  valid shapes in PTX ISA Table 41 (§9.7.16.2.1), idesc in Table 44
  (§9.7.16.4.2) at implementation time.
- **Accumulator:** TMEM (dedicated per-SM memory), NOT registers.
  Transparent row-major layout — no fragment indexing.
- **Issuing:** Single elected thread via `elect_one_sync()`.
- **TMEM lifecycle:**
  1. `tcgen05.alloc` (full warp, warp_id==1)
  2. MMA (elected thread issues)
  3. `tcgen05.commit` (elected thread, immediately after MMA — `fence::before` implicit in commit)
  4. `mbarrier.try_wait` (consumer threads — warps 0 and 1 — spin until MMA hardware done)
  5. `tcgen05.fence::after_thread_sync` (consumer threads, after mbarrier wait)
  6. `tcgen05.ld` → `tcgen05.wait::ld`
  7. `tcgen05.dealloc` before kernel exit (all paths)
- **TMEM quarters:** For M=64, warps 0-1 hold rows 0-63.

### GT-3: `tcgen05.wait::mma` DOES NOT EXIST

PTX ISA §9.7.16.8.5: `.wait_operation = { .wait::ld, .wait::st }` only.
MMA→ld ordering uses `commit → mbarrier_wait → fence::after_thread_sync`
(GT-2 steps 3-5). `tcgen05.fence::before/after_thread_sync` around
`__syncthreads()` is the pattern for cross-thread ordering of *pipelined*
instruction pairs only (e.g. `cp → mma`, PTX ISA §9.7.16.6.4.3) — it
does NOT wait for MMA hardware pipeline completion. See PTX ISA
§9.7.16.6.4.2 for the canonical mma→ld pattern.

### GT-4: H-Reduction Pattern

Fence+sync+fence → TMEM load → per-slab scale in registers → intra-warp
`__shfl_xor_sync` across 32 heads → cross-warp SMEM merge.
`h_partial_smem` sized for **4 active warps** — for `kind::f8f6f4 M=64
cta_group::1`, all 4 warps hold live MMA output rows (see GT-12). The
prior "2 active warps" description was incorrect for this MMA shape and
has been corrected here. Cross-reference GT-12 for the warp→row mapping.

### GT-5: Scale Application

Per-slab MMA with `enable_input_d=false` (fresh accumulator) → fence+sync
+fence → `tcgen05.ld` + `wait::ld` → scale multiply in registers →
accumulate into running register total. Never inside MMA pipeline.

**Scale format for dsa_topk track:** See GT-19 (corrected 2026-04-18).
The K-cache is **blocked per page**, not interleaved: bytes
`[0 .. page_size*128)` are FP8 data `[64, 128]` and bytes
`[page_size*128 .. page_size*132)` are `page_size` fp32 scales (one per
token). GT-19 supersedes both the "4 FP8 scales per row" and "one fp32
per 132-byte interleaved row" interpretations for this track.

### GT-6: TMA Tensor Map Constraints

- **Residency (§9.7.9.27.1.2):** tensorMap for `cp.async.bulk.tensor` MUST
  be in `.param`/`.const`/`.global` — NOT `.shared`.
- **Modification flow:**
  1. Lane 0: `memcpy` template → SMEM, `tensormap.replace` in SMEM
  2. `__syncwarp()` for visibility
  3. All 32 lanes: `tensormap.cp_fenceproxy` SMEM → global (`.sync.aligned`)
  4. Thread 0: `fence.proxy.tensormap::generic.acquire.cta` on global copy
  5. Thread 0: TMA using global copy's address
- **`tensormap.replace` syntax:** requires BOTH `.b1024` AND `.b64`
  qualifiers. Omitting `.b64` → ptxas "Unexpected instruction types".
- mbarrier expect-tx: call once with total transfer size.
- Per-buffer parity: `int parity[2] = {0,0}`, flip after each wait.

### GT-7: PTX Syntax Pitfalls

- **`tcgen05.fence::before_thread_sync`** — bare form only. Do NOT append
  `::1.cta.sync.aligned` → ptxas "Parsing error near ':'".
- **Avoid `<cuda/ptx>` header entirely** — `cuda::ptx::` C++ wrappers have
  `__CUDA_ARCH__` guard issues on sm_100a. Use raw `asm volatile(...)`.
- **`<cuda/barrier>` is fine** to include.

### GT-8: Roofline

HBM3e: 8,000 GB/s. FP8 peak: 3,958 TFLOPS. Ridge: ≈495 FLOPs/byte.
DSA TopK: ≈187 FLOPs/byte → memory-bound. Pipeline ABSENT.

### GT-9: TMA Stride Constraint

`cuTensorMapEncodeTiled` globalStrides must all be multiples of 16 bytes.
Non-multiples silently produce invalid tensor maps that cause XID 13 at
runtime. `cuTensorMapEncodeTiled` does NOT return an error — it silently
creates a bad tensor map that faults when TMA executes.
Example: K-cache 132-byte rows (128 data + 4 scale) → stride 132 is NOT
a multiple of 16 → CANNOT use TMA. Use cooperative thread copy instead.
Confirmed 2026-04-12 on Modal B200 via bisection testing.

### GT-10: TMA/MMA Swizzle Consistency

If TMA loads data with `CU_TENSOR_MAP_SWIZZLE_128B`, the bytes are
rearranged in SMEM using an XOR-based pattern. The MMA SMEM descriptor
built by `make_smem_desc` (bits[46:48]=0b001) assumes LINEAR layout.
These are INCOMPATIBLE — the MMA reads scrambled data and computes wrong
results silently (no hardware fault, just wrong scores).
Fix: use `CU_TENSOR_MAP_SWIZZLE_NONE` for all TMA loads that feed into
`tcgen05.mma` via linear `make_smem_desc` descriptors.
Alternative: use gau-nernst's `init_tmap_3d_128B` pattern with a 3D
tensor map that encodes the swizzle in the layout.
Confirmed 2026-04-12 on Modal B200.

### GT-11: tcgen05.mma SMEM Descriptor — SBO/LBO Confirmed Values

For `tcgen05.mma.cta_group::1.kind::f8f6f4`, M=64/N=64/K=32, K-major,
`SWIZZLE_NONE`, sm_100a (B200), **using the 8×T core-tile SMEM layout**:

    SBO field bits[32:45] = 16   (SBO = 256 bytes)
    LBO field bits[16:29] = 8    (LBO = 128 bytes)

Confirmed 2026-04-15 by host-side sweep on Modal B200
(`diagnostic_tests/sbo_lbo_sweep.cu`). The sweep tested 120 (SBO, LBO) pairs;
34 pairs give err=0.0 against the CPU reference (the hardware descriptor has
slack across the field), and (SBO_enc=16, LBO_enc=8) is in the passing set,
matching the canonical PTX ISA §9.7.16.3.3 interpretation of the 8×T tile:
"first 8 rows to next 8 rows" = 8 × row_stride = 8 × 32 = 256 bytes, and
"stride from col 0 to col 1 of 8×2 tile in 128-bit normalized matrix" = 128 bytes.

**GT-12 also confirmed** by the same run: for `kind::f8f6f4 M=64 cta_group::1`,
the 64 M-rows are distributed 16 per warp across all 4 warps; only lanes 0..15
of each warp hold live MMA output data (lanes 16..31 read zeros). The canonical
`tcgen05.ld` address for accumulator row r is `(warp_id * 32 + lane_id) << 16`
where lane L < 16 of warp W maps to M-row (W * 16 + L). Upper lanes must be
masked out of downstream reductions.

**SMEM layout constraint:** Q and K SMEM tiles feeding this MMA must be
arranged in the 8×T core-tile layout, NOT plain linear row-major. The fill
formula is: `addr = m_grp * SBO_bytes + k_tile * LBO_bytes + m_in_grp * 16 + k_in_t`
where m_grp = m / 8, m_in_grp = m % 8, k_tile = k / 16, k_in_t = k % 16.

**Canary confirmation:** `diagnostic_tests/canary_mma_pipeline.cu` with
compile-time `SBO_ENC=16, LBO_ENC=8`, all-1.0 A, B[n][k]=n+1 returned
`regs = {32, 64, 96, 128, 160, 192, 224, 256}` exactly — matching D[0][n] = 32*(n+1).

The PTX ISA (§9.7.16.3.3) describes the canonical 8×T layout structure correctly
but does not uniquely specify the encoded SBO/LBO values for every tile shape,
swizzle mode, and alignment combination. SPEC CONFLICT [MMA] demonstrated that
these values cannot be reliably derived from the specification alone — they
require empirical confirmation. This entry supersedes that conflict.


### GT-13: PAGES_PER_CTA Minimum

PAGES_PER_CTA=1 produces non-deterministic results on B200. Minimum confirmed
stable value is PAGES_PER_CTA=2. Confirmed 2026-04-15 on Modal B200.

### GT-14: Slab Loop Unrolling

`#pragma unroll` on the K-slab loop breaks MMA→ld ordering and produces wrong
results. Do not unroll the slab loop. Confirmed 2026-04-15 on Modal B200.

### GT-15: IDESC Transpose B Bit with K-Major 8×T Layout

For the K-major 8×T core-tile SMEM layout (GT-11), the IDESC transpose_B bit
must be 0, not 1. The "m" axis of the 8×T fill is the operand's outer axis —
M for A, N for B — with K always contiguous. transpose_B=1 produces the
canonical 50–80% descriptor-mismatch signature. One-bit fix resolves it.
Confirmed 2026-04-15 on Modal B200.

### GT-16: Persistent Global Scratch for Multi-CTA Kernels

For multi-CTA kernels requiring inter-kernel scratch buffers, use a static
device pointer (allocated once, never freed in-process). Per-launch
cudaMalloc/cudaFree overhead eliminates parallelism gains from multi-CTA
partitioning. Confirmed 2026-04-15 on Modal B200 during Tier-3 optimization.

### GT-17: tcgen05.ld Requires Full Warp Participation

`tcgen05.ld.sync.aligned` requires ALL 32 lanes of the issuing warp to
execute the instruction. GT-12's statement that "lanes 16..31 read zeros"
describes the *content* of those registers — it does NOT mean those lanes
are optional participants. Gating with `if (lane_id < 16)` before
`tcgen05.ld` causes the instruction to hang or produce wrong results
because the hardware requires a full-warp collective issue.
Correct pattern: all 32 lanes issue `tcgen05.ld`; downstream computation
is masked by `if (lane_id < 16)` AFTER the load returns.
Confirmed 2026-04-17 on Modal B200 — caused Layer 4 retry during
incremental build.

### GT-18: tcgen05.commit Canary Form Does Not Compile on CUDA 12.8

`tcgen05.commit.cta_group::1.mbarrier::complete_tx::bytes` is a form that
appears in some ISA documentation and canary references but does NOT
compile under CUDA 12.8 (ptxas error on Modal B200). The correct form is:

    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [addr];

Always use the `.mbarrier::arrive::one.shared::cluster.b64` form. Do not
use `.mbarrier::complete_tx::bytes` regardless of what canary or reference
code shows — it will produce a ptxas compile error on this toolchain.
Confirmed 2026-04-17 on Modal B200 CUDA 12.8.

### GT-19: K-cache is Blocked, Not Interleaved (for dsa_topk Track)

**⚠ The prior version of GT-19 described a per-row `[128 data | 4 scale]`
interleaved layout. That is WRONG for this track.** The declared tensor
shape `[num_pages, 64, 1, 132]` is a packing-convenience shape; the
physical storage per page is *blocked*:

- Bytes `[0 .. page_size * 128)` = `page_size` FP8 data tokens,
  shape `[64, 128]` fp8_e4m3fn.
- Bytes `[page_size * 128 .. page_size * 132)` = `page_size` fp32 scales
  (one fp32 per token, NOT one per 32-element slab, NOT four fp8).

Correct dequantization (verbatim from the grader's `definition.reference`):

```python
kv_flat  = k_cache_u8.view(num_pages, page_size * 132)            # [P, 8448]
data_fp8 = (kv_flat[:, :page_size*128]                             # [P, 8192] u8
              .view(num_pages, 64, 128)
              .view(torch.float8_e4m3fn))                          # [P, 64, 128] fp8
scale_fp32 = (kv_flat[:, page_size*128:]                           # [P, 256] u8
                .view(num_pages, 64, 4)
                .view(torch.float32))                              # [P, 64, 1] fp32
K_deq = data_fp8.to(torch.float32) * scale_fp32                    # [P, 64, 128] fp32
```

The equivalent C++ ATen sequence (from the winning kernel, 128/128):

```cpp
auto k_pg_flat = k_rows_u8.view({num_pages_b, 64 * 132});          // [N, 8448]
auto data_fp8  = k_pg_flat.slice(-1, 0, 64 * 128)
                          .contiguous()
                          .view(torch::kFloat8_e4m3fn);            // [N, 8192] fp8
auto k_fp32    = data_fp8.to(torch::kFloat32)
                         .view({num_pages_b, 64, 128});            // [N, 64, 128]
auto k_scale   = k_pg_flat.slice(-1, 64 * 128, 64 * 132)
                          .contiguous()
                          .view(torch::kFloat32)
                          .view({num_pages_b, 64, 1});             // [N, 64, 1]
auto K_deq     = k_fp32 * k_scale;                                 // [N, 64, 128]
```

Reading scale at byte-offset 128 of a 132-byte "row" (the old interleaved
interpretation) re-interprets ~62 tokens' data bytes per page as scales and
62 scale groups as data. It produces visibly wrong dequantized values even
on the first page.

Failure signature for this error class: abs_err grows roughly linearly with
seq_len, with very small seq_lens occasionally passing with abs_err=0.
Confirmed 2026-04-18: switching from interleaved to blocked interpretation
(with no other changes) moved pass rate from 4/128 to 26/128.

**This entry supersedes both (a) GT-5's "4 FP8 scales per row" description
and (b) the prior version of GT-19** for this track.

Note: `kernel_spec.md` and `hardware_target.md` may still describe the
interleaved layout informally. The `definition.reference` Python source
is authoritative; spec text is not.

### GT-20: FP8 Tensor Core vs FP32 Precision Gap (ARCHITECTURAL CONSTRAINT)

FP8 tensor core MMA (`tcgen05.mma.kind::f8f6f4`) does NOT produce
bit-identical results to FP32 dequant + FP32 matmul. The differences
are small (typically < 1% relative error per score) but are SYSTEMATIC
and UNFIXABLE without leaving FP8 tensor cores.

**Why this matters for top-k kernels:** When the grader compares top-k
indices against a FP32 reference (e.g. `torch.topk` on FP32 scores),
tiny score differences at the k-th selection boundary cause different
tokens to be selected. The kernel selects the correct top-k set for
its OWN scores but not for the REFERENCE's scores. Observed: 100%
set overlap within each score domain, but ~90% of workloads fail
position-exact comparison because different tokens cross the k-th
boundary.

**This is not a kernel bug. It is a fundamental precision gap.**
No amount of NaN handling, sort ordering, or tie-breaking can fix
a disagreement about which tokens have the k-th highest score.

**Confirmed 2026-04-18:** FP8 MMA scoring kernel achieves 10/128 pass
rate with torch.topk as the sort backend. The 118 failing workloads
all show 100% set overlap but score-boundary disagreements caused by
FP8 vs FP32 rounding.

### GT-21: Precision-Match Decision Gate (MANDATORY at D3)

Before binding the compute unit for any kernel whose grader compares
DERIVED outputs (top-k indices, sorted positions, argmax, thresholded
masks — anything that depends on score ORDERING rather than score VALUES),
evaluate the Precision-Match Gate:

**Question:** Does the grader's reference compute scores in higher
precision than the proposed kernel's compute unit?

- If YES (e.g. reference uses FP32, kernel uses FP8 MMA):
  The kernel CANNOT match the reference bit-exactly on ordering-dependent
  outputs. Choose one of:
  (a) Match the reference's precision (FP32 CUDA cores or FP32 matmul)
      and accept reduced throughput. Optimize via fusion, memory access
      patterns, and parallelism instead of tensor core throughput.
  (b) Use FP8 MMA and accept partial pass rate (~5-10% for top-k with
      random data). Only viable if competition scoring rewards partial
      correctness × speedup.
  (c) Hybrid: FP8 MMA for the bulk computation, then FP32 refinement
      for tokens near the k-th boundary. Complex but may preserve most
      of the speedup while fixing boundary disagreements.

- If NO (reference and kernel use same precision): proceed normally.

**This decision must be made at D3 item 3 (Dtype handling) BEFORE any
implementation work begins.** Discovering the precision gap at Layer 6+
after a full derivation run wastes the entire implementation effort.

Confirmed 2026-04-18 via full derivation run + refactor cycle on the
dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 track.

**Path (a) caveat — "FP32" alone is not enough (added 2026-04-18):**
"Use FP32 CUDA cores" is *necessary* but *not sufficient* for bit-exact
reference parity. cuBLAS's FP32 matmul picks different tile decompositions
for different operand shapes (GT-22), and FP32 addition is not associative,
so two "mathematically equivalent" FP32 implementations generally produce
different scores at the ulp level — which flips top-k boundary indices.

In practice, bit-exact parity with an FP32 reference that uses
`torch.matmul` requires:

1. Delegating the matmul to the **same library** the reference uses
   (`torch::matmul` / `at::matmul` from inside a `binding = "torch"` C++
   extension). A hand-rolled FP32 FMA loop will diverge from cuBLAS — a
   serial-in-lane implementation was confirmed 2026-04-18 to pass only
   3/128 despite using FP32 everywhere.
2. Using **identical operand shapes** for every op in the chain, especially
   matmul (GT-22), and identical summation axes / keepdim flags for any
   reductions. The simplest way to get this right is to transliterate the
   reference's Python line-for-line into C++ ATen ops.
3. Using the **authoritative reference source** — printed from
   `definition.reference` via the probe scripts — not spec text, not
   historical GT entries, not prior kernels.

This is "library substitution at the matmul level", not full-kernel
library substitution. The C++ wrapper still owns the batch loop, tensor
reshaping, block_table indirection, and topk-result translation.

### GT-22: cuBLAS matmul shape affects FP32 reduction order

For `torch::matmul(A, B)` on FP32 inputs, cuBLAS selects a tile
decomposition based on the concrete operand shapes. The tile decomposition
determines the order in which per-element products are accumulated; FP32
addition is not associative, so two matmuls that are mathematically
equivalent over a sliced output can produce different FP32 values if their
operand shapes differ.

Concrete example from the dsa_topk kernel:

```cpp
// VERSION A (WRONG — diverges from reference at ulp level):
auto dot  = torch::matmul(q_b, k_flat.transpose(0, 1));    // [64, num_pages*64]
auto vals = dot.slice(-1, 0, sl);                          // slice AFTER matmul

// VERSION B (CORRECT — matches reference):
auto K    = k_flat.slice(0, 0, sl);                        // slice BEFORE matmul
auto dot  = torch::matmul(q_b, K.transpose(0, 1));         // [64, sl]
```

A and B compute the same "valid" scores mathematically, but cuBLAS chooses
different tile shapes for `[64, D] @ [D, num_pages*64]` vs `[64, D] @ [D, sl]`,
producing ulp-level score differences. On top-k with random data those
ulp differences flip boundary indices. Confirmed 2026-04-18: the pre-trim
fix alone moved the dsa_topk kernel from 26/128 to 128/128 pass rate —
**all 102 remaining failures were shape-induced cuBLAS divergences.**

**Rule:** when delegating scoring to `torch::matmul` / `at::matmul` to match
an FP32 reference, use **exactly** the reference's operand shape for every
matmul in the chain. If the reference writes `q @ K[:seq_len].T` — pre-trim
K. If it writes `q @ K_full.T` then slices — post-trim.

Symptom when violated: abs_err grows with seq_len, with small seq_len
workloads sometimes passing with abs_err=0 (because cuBLAS happens to use
the same tile shape there). Set-overlap near 100% (correct token set,
wrong ordering at k-th boundary).

Related hazards (same class):
- Different `batch` dimensions in a batched matmul select different
  kernels (cuBLAS/cuBLASLt heuristics).
- TF32 vs full-FP32 matmul produce different results; ensure
  `at::globalContext().allowTF32CuBLAS()` matches the reference's setting
  (torch default is TF32 **on** for FP32 matmul since 2.0).
- `torch.matmul` with transposed-view vs contiguous-copy operand may hit
  different kernels. Prefer `K.transpose(0, 1)` (view) over `K.T.contiguous()`
  unless the reference does the latter.

Confirmed 2026-04-18 on Modal B200, CUDA 12.8, PyTorch default TF32 setting.

---

### Failure Pattern Signatures

These are observed mappings from failure symptoms to root cause classes.
Use them to route the diagnostic agent directly to the right investigation
path without requiring a full ground-up diagnosis each time.

| Symptom | Most likely class | First action |
|---------|-------------------|--------------|
| XID 13 on `cp.async.bulk.tensor` | TENSOR_MAP | Check all `globalStrides` ÷ 16 (GT-9) |
| XID 13 on `tcgen05.mma` | TMEM_ADDR | Check TMEM alloc lifecycle (GT-2) |
| XID 13 on kernel launch | SMEM | Check dynamic SMEM opt-in > 48 KB |
| Wrong scores, no fault, ~0–20% of reference | SMEM_LAYOUT | Layout fill logic wrong — data reaching MMA is incorrect |
| Wrong scores, no fault, ~50–80% of reference | DESCRIPTOR | SBO/LBO stride mismatch — data correct, traversal wrong. Run `sbo_lbo_sweep` immediately |
| Wrong scores, no fault, ~100% magnitude but wrong positions | IDESC | `transpose_A` or `transpose_B` bit wrong in IDESC |
| Wrong scores only after adding swizzle | GT-10 violation | TMA swizzle mode does not match SMEM descriptor swizzle mode |
| ptxas parse error near `:` | PTX_SYNTAX | `tcgen05.fence` has illegal suffix — bare form only (GT-7) |
| ptxas `instruction not supported` | PTX_SYNTAX | WGMMA instruction on sm_100a — does not exist (GT-1) |
| Kernel hangs with no XID and no output | BARRIER | Missing `__syncthreads()` or infinite loop in sort/reduction |
| Numerically correct on small inputs, wrong on large | SMEM | SMEM buffer overflow — `T_pad` calculation wrong |
| Kernel hangs at tcgen05.ld with no XID, wrong results on lane-gated warps | TMEM_LD | Remove if(lane<16) guard before tcgen05.ld — all 32 lanes must issue (GT-17) |
| Pass on some workloads, fail on workloads with extreme batch_size | GRID_SHAPE | Check grid formula at min/max batch_size from workload_envelope.md |
| Pass on typical workloads, fail when num_pages is very large | MEM_OVERFLOW | Recompute memory budget at MEM_STRESS corner from envelope |
| Pass on typical workloads, fail when num_pages is very small | LOOP_BOUND | Check page loop handles num_pages < PAGES_PER_CTA or seq_len < page_size |
| Correct on most workloads, wrong on workloads where topk ≈ total_tokens | GATE_DEGENERATE | Gate assumes topk < N; handle topk = N as pass-through identity |
| Numerical drift on only the longest-sequence workloads | ACCUMULATION | FP accumulation order produces different rounding at large N |
| COMPILE_ERROR on all workloads, no nvcc error in raw_tail | BUILD_SYSTEM | Worker stdout redirected before BuildError caught. Add `trace.evaluation.log` print to scripts/run_modal.py (see Competition Build System Facts). Check Modal image has nvcc (devel not slim) and TVM_FFI_CUDA_ARCH_LIST="10.0a" is set |
| abs_err ≥ 128 on most workloads, correct set-wise overlap | NAN_DETECTION | FP8 MMA does not propagate NaN like FP32 reference. Detect NaN bytes (0x7F/0xFF) during K-cache cooperative load; build per-token bitmask; force +INF after h-reduction. See NaN section above |
| 100% set overlap but ~90% workloads fail position-exact, abs_err grows with seq_len | PRECISION_GAP | FP8 MMA scores ≠ FP32 reference scores at k-th boundary. Cannot fix with sort/NaN changes. See GT-20/GT-21. Must match reference precision or accept partial pass rate |
| abs_err=0 on small seq_len, grows to thousands on large seq_len; set-overlap near 100%; kernel uses `torch::matmul` / `at::matmul` | SHAPE_MISMATCH | cuBLAS tile choice is shape-dependent, so operand shape that doesn't match the reference produces ulp-divergent FP32 scores (GT-22). **First action:** run the probe (`probe_ref_vs_kernel.py`) to print `definition.reference`, then transliterate every op's operand shape 1-to-1. Common culprits: matmul over full `num_pages * 64` then slicing vs slicing K to `[:seq_len]` before matmul; batched vs per-batch matmul; K-cache layout interpreted as interleaved instead of blocked (GT-19) |
| Kernel using `torch::matmul` passes ~5/128 or fewer with no obvious bug | REFERENCE_MISREAD | The authoritative formula is `definition.reference`, not spec text. **First action:** run `probe_grader_oracle.py` or `probe_ref_vs_kernel.py` and print the Python source. Diff line-for-line against the C++ wrapper. Expect kernel_spec.md / hardware_target.md to be subtly wrong about tensor layouts — they were for GT-19 |

**The 50–80% signature in detail:** MMA output in the range 50–80% of CPU
reference magnitude indicates a descriptor stride error where some elements are
read from correct SMEM positions and others from offset positions. This pattern
appeared in SPEC CONFLICT [MMA] (output ≈ 65% of reference) and was traced to
SBO/LBO encoding interacting incorrectly with the 8×T core-tile layout.
The confirming signal is that the software fallback on the same SMEM data
passes — data is correct, traversal is wrong. Route directly to `sbo_lbo_sweep`.

---

## Inputs and Outputs

**Inputs** (always present):
- `framework/MANIFEST.md` — derivation framework index (read MANIFEST first,
  then load only the files listed for your current step)
- `kernel_spec.md` — what the kernel computes
- `hardware_target.md` — target hardware capabilities
- `ptx_isa_sections/` — PTX ISA topic files (read MANIFEST.md first)
- `gau-nernst_reference.h` — working B200 PTX wrappers
- `workloads/` — competition workload JSONL files from the FlashInfer-Bench
  dataset (128 workloads). Parsed in Phase 0.
- `definitions/` — Definition JSON for the target kernel variant from the
  FlashInfer-Bench dataset.

**Outputs** (create as you work, overwrite prior runs):

| File | Contents |
|------|----------|
| `d1.md` | Molecule detection table |
| `v1.md` | Verified molecules |
| `d2.md` | Structural analysis — all 3 sections |
| `v2.md` | Verified structure |
| `d3.md` | Hardware binding table |
| `v3.md` | Verified hardware binding |
| `d4.md` | Architecture spec — all 5 sections |
| `v4.md` | Verified architecture spec |
| `impl.md` | Kernel implementation |
| `audit.md` | Audit findings + corrected kernel |
| `workload_envelope.md` | Parameter ranges, corner workloads, derived limits |

**D1/D2 skip rule:** If d1.md, v1.md, d2.md, v2.md already exist, skip
D1/V1/D2/V2 and begin at D3.

---

## Workload Envelope Extraction (Phase 0)

**Run BEFORE D1.** This step is mandatory. All downstream derivation
decisions that depend on runtime parameter ranges use the envelope
computed here, not assumed values.

### Inputs

- `workloads/` directory from the FlashInfer-Bench dataset
  (`huggingface.co/datasets/flashinfer-ai/mlsys26-contest`)
- The kernel's Definition JSON (from `definitions/` in the same dataset)

### Procedure

1. **Parse the Definition JSON.** Extract every axis marked `"type": "var"`
   and every axis marked `"type": "const"`. Record the constant values.

2. **Parse all workload JSONL files.** For each variable axis, collect
   every bound value across all 128 workloads. Compute:

       axis_envelope = {
           min: smallest value seen,
           max: largest value seen,
           values: sorted unique list,
           count: number of unique values
       }

3. **Compute derived ranges.** For parameters that interact:

       max_seq_len_range = [min(num_pages) * page_size, max(num_pages) * page_size]
       K_over_N_range   = [topk / max(max_seq_len_range), topk / min(max_seq_len_range)]

4. **Identify corner workloads.** Flag workloads at extreme parameter
   combinations. At minimum identify:

   | Corner | Selection criterion |
   |--------|---------------------|
   | SMALL_BATCH | batch_size = min |
   | LARGE_BATCH | batch_size = max |
   | SHORT_SEQ | num_pages = min |
   | LONG_SEQ | num_pages = max |
   | MIN_K_RATIO | topk / (num_pages × page_size) is smallest |
   | MAX_K_RATIO | topk / (num_pages × page_size) is largest |
   | MEM_STRESS | batch_size × max_num_pages is largest (max pre-allocated buffer) |
   | SM_STARVED | batch_size < 132 (B200 SM count) |

5. **Save outputs:** `workload_envelope.md` — full envelope table, derived
   ranges, corner workloads with UUIDs, and the analysis summary.

### Workload Envelope Rules (MANDATORY)

- D3 item 13a (SMEM/memory budget) must be evaluated at the MEM_STRESS
  corner, not at a single assumed parameter set.
- D3 item 14 (grid shape SM utilization) must compute min_batch using
  the SM_STARVED corner's batch_size as the concrete check.
- D3 item 12 (gate algorithm selection) must compute K/N at BOTH
  extremes of K_over_N_range. If the algorithm class differs at the
  two extremes, either (a) choose the class that works at both, or
  (b) design a runtime dispatch with both paths.
- D1 Tile CTA analysis must use SM_STARVED corner for the
  `min_elements = num_SMs / CTAs_per_element` comparison.
- The LONG_SEQ corner determines whether pipelined double-buffering
  has high value even if arithmetic intensity is below the ridge point.

### Script

Run `python scripts/extract_envelope.py` to generate `workload_envelope.md`
automatically from the dataset. The script parses every workload JSONL,
computes the envelope, identifies corners, and writes the output file.

Output: workload_envelope.md. Save before proceeding to D1.

---

## The Mandate

- "Could", "optionally", "consider", "might" are forbidden.
- No code in D1–D4. One step at a time. Save immediately.
- Never write TBD. Use symbolic formulas, state which step resolves them.
- **sm_100a prohibition (GT-1):** No WGMMA instructions or fragment layouts.

---

## D1 — Molecule Detection

Use claude-sonnet. Read: framework/MANIFEST.md (load files 05, 06, 07, 08),
kernel_spec.md, hardware_target.md, workload_envelope.md

For each molecule (Tile, Reduction, Gate, Online, Pipeline, Fusion): CONFIRMED or ABSENT with structural justification. If CONFIRMED, list pre-wired decisions. Pipeline is ABSENT (GT-8: 187 < 495).

**Tile molecule (if CONFIRMED):** In addition to the standard pre-wired
decisions, state explicitly whether all tiles are processed by a single CTA
(serial FXP loop) or whether the tile dimension is split across a grid axis
(parallel CTAs with a two-phase reduce). Compute the minimum number of
problem-dimension elements required to fully occupy all SMs:
`min_elements = num_SMs / CTAs_per_element`. If the deployment target batch
size is below this threshold, evaluate splitting the tile loop across a
second grid dimension before committing to the serial structure. A default
of one-CTA-per-element without this calculation is an incomplete pre-wired
decision. (Ref: framework/05_structural_analysis.md, Tile molecule.)

**Pipeline molecule (if ABSENT):** Pipeline ABSENT closes only the warp
specialization question. Write two explicit sentences:
(a) Whether two SMEM buffers are allocated for the dominant load
    (structural double-buffering — adopt or reject with reason).
(b) Whether the next iteration's load is initiated asynchronously
    *before* the current iteration's compute completes (pipelined
    double-buffering — adopt or reject with reason).
Both (a) and (b) are required. (a) without (b) is structural scaffolding,
not a pipeline — allocating two buffers does not hide latency unless the
next load is initiated during the current MMA. "Pipeline ABSENT" without
both sentences is an incomplete verdict.
(Ref: framework/05_structural_analysis.md, Pipeline molecule.)

**Gate molecule (if CONFIRMED):** The pre-wired criterion class (SRG
shuffle-tree or MON sort/scan) is the ONLY algorithm decision made at D1.
Do NOT name a concrete sort or selection algorithm (bitonic sort, radix
sort, priority heap, introselect, etc.) in D1. That choice depends on the
K/N ratio, SMEM budget, and hardware binding — defer to D3 checklist item
12. (Ref: framework/07_gate_specification.md + framework/08_streaming_gate.md.)

**Streaming Gate detection (MANDATORY when both Tile and Gate CONFIRMED):**
Ask: are the GATE dimension's candidates produced inside the FXP tile loop,
or do they all exist before the loop starts? If candidates are produced
inside the tile loop (e.g. scores computed one page at a time during the
page loop), this is a Streaming Gate (FXP×GATE). When detected:
- The single-pass Gate FSM does NOT apply. The Streaming Gate FSM applies.
- Pre-wire ALL decisions listed in framework/08_streaming_gate.md
  "Pre-Wired Decisions for d1.md": accumulation buffer location/layout,
  staging scratch area, two-case merge algorithm (Case A + Case B with
  mandatory staging sort), buffer re-sort, and deferred output write.
- The Tile→Gate sequential boundary is replaced by a per-tile merge
  phase inside the tile loop.
**Cross-CTA mechanism constraint (MANDATORY):** When pre-wiring decisions,
the cross-CTA mechanism must be one of:
  (a) Per-CTA top-local_count buffer + cross-CTA K-merge, OR
  (b) Single-kernel + cooperative_groups grid sync.
Do NOT pre-wire "D3 chooses based on local_count vs K" for buffer location.
Do NOT leave "global scratch OR per-CTA buffer" open as a disjunction.
When Streaming Gate fires, the per-CTA buffer is mandatory regardless of
how local_count compares to K. The "or global" branch is forbidden at D1.
Skipping this detection when both Tile and Gate are confirmed is an
incomplete D1. (Ref: framework/08_streaming_gate.md, framework/04_atom_intersection_matrix.md FXP×GATE entry.)

Output: table — Molecule | Status | Justification | Pre-wired Decisions. Save to d1.md.

## V1 — Verify Molecule Detection

Use claude-sonnet. Read: framework/MANIFEST.md (load files 05, 07, 08),
kernel_spec.md, hardware_target.md, d1.md

Read d1.md as external work by a different person. Attempt to falsify every verdict. Challenges: (1) Pipeline arithmetic intensity vs ridge point, (2) Online: does accumulation require correcting priors? (3) Fusion: any FMA composition? (4) Gate: all four pre-wired decisions listed? (5) Reduction: Semiring or Monoid? (6) **Tile CTA analysis:** does d1.md compute `min_elements = num_SMs / CTAs_per_element` and compare to deployment batch size? If missing → FAIL. (7) **Pipeline double-buffering:** does d1.md contain explicit adopt/reject sentences for BOTH (a) structural double-buffering (two SMEM buffers allocated) AND (b) pipelined double-buffering (next load initiated during current MMA)? If either sentence is missing → FAIL. A sentence addressing only buffer count without addressing async initiation timing is an incomplete verdict for (b). (8) **Gate algorithm deferral:** does d1.md name a concrete algorithm (bitonic, radix, etc.) instead of deferring to D3? If so → FAIL. (9) **Streaming Gate detection:** if both Tile and Gate are CONFIRMED, does d1.md check whether candidates are produced inside the tile loop? If yes and Streaming Gate is not detected → FAIL. If Streaming Gate is detected, are all pre-wired decisions from framework/08_streaming_gate.md present in d1.md? If any missing → FAIL. (10) **Streaming Gate cross-CTA mechanism:** if Streaming Gate is detected, does d1.md leave "global scratch OR per-CTA buffer" open as a disjunction, or defer the buffer location to D3 based on local_count vs K? If so → FAIL. The per-CTA buffer is mandatory when Streaming Gate fires — this is not a D3 choice.

STATUS: PASS → save v1.md, proceed D2. FAIL → list errors, correct d1.md, re-verify. Max 3 attempts.

## D2 — Structural Analysis

Read: framework/MANIFEST.md (load files 03, 06, 09), kernel_spec.md, v1.md

Three sub-steps: (1) Structural Analysis Table (Fixed Point, Morphism, Measure, Relation, Symmetry, Functor), (2) Dimension Fate Analysis (classify every dim as AREA/REDUCE/GATE, fill Combine Groups), (3) Affine Map Registry (every memory access as base + index×stride, typed INDEXED READ or AFFINE COMPUTE, separate formulas for non-uniform formats).

Save to d2.md.

## V2 — Verify Structural Analysis

Use claude-sonnet. Read: framework/MANIFEST.md (load files 03, 06, 09),
kernel_spec.md, v1.md, d2.md

External review. Challenges: (1) Dimension fates correct? (2) Non-uniform format formulas? (3) Indirection chain typing? (4) Combine Groups? (5) Coalescing/bank conflicts?

STATUS: PASS → save v2.md, proceed D3. FAIL → correct, re-verify.

---

## D3 — Hardware Binding

Use **claude-opus**. Read: framework/MANIFEST.md (load files 04, 07, 08, 10),
kernel_spec.md, hardware_target.md, workload_envelope.md,
ptx_isa_sections/MANIFEST.md (then load
listed sections), gau-nernst_reference.h, v2.md

**All decisions must be consistent with Ground Truth (GT-1 through GT-22).**

For each item: BOUND: [decision] — [mandate] or NOT APPLICABLE: [reason]

1. **Async transfer** — per input tensor.
2. **Indirect addressing** — dynamic base update mechanism. Must use GT-6
   flow (SMEM replace → cp_fenceproxy → global → TMA). Verify against
   PTX ISA §9.7.9.27.1.2 and §9.7.9.28.
3. **Dtype handling** — hardware unit, pipeline position, granularity.
3b. **Precision-Match Gate (GT-21, MANDATORY for ordering-dependent
    outputs):** If the kernel's output depends on score ORDERING
    (top-k, argmax, threshold, sort), read GT-21 and answer:
    - What precision does the grader's reference use for scoring?
    - What precision does the proposed compute unit use?
    - If they differ: which path (a), (b), or (c) from GT-21?
    Write the decision explicitly with justification. A kernel bound
    to FP8 MMA for a FP32-reference top-k track without addressing
    GT-21 is an incomplete binding and will fail 90%+ of workloads
    at Layer 6.
4. **Tile shape** — look up PTX ISA Table 41 (§9.7.16.2.1). Derive
   iteration count. tcgen05 only (GT-1).
5. **Thread specialization** — NOT APPLICABLE (GT-8: 187 < 495).
6. **SMEM layout** — bank conflict analysis, swizzle/padding.
7. **State buffer location** — show register budget arithmetic.
8. **Inter-group communication** — H-reduction per GT-4.
9. **Transfer alignment** — (a) TMA globalStrides: every element of
   globalStrides passed to `cuTensorMapEncodeTiled` must be a multiple
   of 16 bytes. If any stride is NOT a multiple of 16 (e.g. 132-byte
   K-cache rows), TMA CANNOT be used for that tensor — bind to
   cooperative thread copy instead (GT-9).
   (b) TMA swizzle: must match SMEM descriptor mode. If using linear
   `make_smem_desc`, TMA must use `SWIZZLE_NONE` (GT-10).
   (c) SMEM descriptor: all `make_smem_desc` args must be ×16
   (PTX ISA Table 42, §9.7.16.4.1).
10. **Phase separation barrier** — Gate FSM boundary primitive.
11. **Constraint verification** — for every primitive bound above:
    (a) CTA-wide broadcasts use SMEM+syncthreads, not shfl (GT-4)
    (b) SMEM descriptor args ÷ 16 (PTX ISA Table 42)
    (c) tcgen05.ld: N/8 calls (PTX ISA §9.7.16.8.3)
    (d) Fence variant: both before+after for cross-thread (§9.7.16.6.3)
    (e) Tensor map in global for TMA (GT-6, §9.7.9.27.1.2)
    (f) tensormap.replace: .b1024.b64 (GT-6, §9.7.9.28)
    (g) TMA globalStrides all ÷ 16 (GT-9)
    (h) TMA swizzle matches SMEM descriptor mode (GT-10)
12. **Gate algorithm selection** (MANDATORY if Gate molecule CONFIRMED) —
    Choose the concrete Phase 1 algorithm. This is where the D1-deferred
    decision is made. Evaluate based on:
    (a) **K/N ratio:** Compute K/N where K = output count (e.g. topk=2048)
        and N = maximum input count (e.g. max_seq_len=8192). Write the
        ratio explicitly.
    (b) **Algorithm class decision:** If K/N is significantly below 1.0
        (typical threshold ~0.5, but write the analysis — do not apply
        blindly), selection-based algorithms (introselect, radix-select,
        warp-level top-k buffer, multi-pass threshold filter) are
        asymptotically cheaper than full sorts (O(N) vs O(N log N)).
        If K/N is close to 1.0, full sort may be acceptable. Write the
        analysis showing which class is indicated and why.
    (c) **Concrete algorithm:** Name the specific algorithm from the
        indicated class. Candidates per class:
        - **Selection class:** introselect, radix-select (radix partition
          to find k-th element, then gather), warp-level top-k insertion
          buffer, multi-pass threshold binary search.
        - **Sort class:** bitonic sort, radix sort (CUB or hand-rolled),
          merge sort.
    (d) **SMEM budget check:** Verify the chosen algorithm's workspace
        fits within the SMEM budget from item 13 below. If not, choose
        the next best algorithm that fits.
    (e) **Envelope K/N check (MANDATORY):** Using K_over_N_range from
        workload_envelope.md, verify that the chosen algorithm class
        is valid at BOTH extremes. If MIN_K_RATIO < 0.1 and MAX_K_RATIO
        > 0.5, a single algorithm class may not be optimal across the
        range. Options: (i) choose the selection class (works at both,
        suboptimal at high K/N), (ii) choose the sort class (works at
        both, suboptimal at low K/N), (iii) runtime dispatch (best
        performance, highest complexity). Write the analysis showing
        which option is chosen and why.
    (Ref: framework/07_gate_specification.md + framework/08_streaming_gate.md;
        framework/10_hardware_binding.md checklist item 8.)
13. **MEA four-measure verification** (MANDATORY) — Compute and record:
    (a) **SMEM budget:** Sum of all Block→Shared buffer sizes. Fixed-size
        buffers summed at compile-time size. Variable-size buffers (sort
        workspaces, indirection tables, any buffer whose size depends on
        a runtime parameter) evaluated at the MAXIMUM expected runtime
        value. Write the budget line explicitly:
        `fixed_overhead + max(var_buf_1) + max(var_buf_2) + ... ≤ smem_limit`.
        If the sum can exceed the limit at any valid input, a spill path
        to global memory is required and must be designed here, not deferred.
        Evaluate this budget at the MEM_STRESS corner workload from
        workload_envelope.md (the workload that maximizes variable-size
        buffer demands). If the budget overflows at MEM_STRESS but passes
        at typical parameters, a runtime spill path or parameter-dependent
        buffer sizing is required and must be designed here.
    (b) **Register budget:** Sum of Thread→Register state ≤ 64 floats
        target. List each contributor.
    (c) **Reuse count:** For every staged buffer, reuse_count = product of
        AREA dimensions absent from this tensor's index formula. Must be > 1.
    (d) **Occupancy:** occupancy = f(smem_total, reg_total). Compute and
        record. Check MEA×REL: Combine Group budgets are coupled.
    (Ref: framework/10_hardware_binding.md, checklist item 8.)
14. **Grid shape SM utilization** (MANDATORY) — State the CTA grid shape.
    Compute the minimum input size required to fully occupy the GPU:
    `min_CTAs = num_SMs` (132 for B200). If grid is (batch, 1, 1), compute
    `min_batch = ceil(num_SMs / CTAs_per_batch_element)`. If the typical
    deployment batch size falls below this threshold, evaluate splitting
    the tile/page loop across a second grid dimension (multi-CTA with
    two-phase reduce). A default of one-CTA-per-element without this
    calculation is an incomplete binding. "SM utilization is acceptable" is
    valid only if the minimum-occupancy batch size is stated and confirmed
    to match the deployment target.
    Use the SM_STARVED corner from workload_envelope.md as the concrete
    check: if SM_STARVED batch_size < min_batch for full occupancy, the
    multi-CTA evaluation is mandatory.
    (Ref: framework/10_hardware_binding.md, checklist item 11.)
15. **PRD boundary identification** — All dimensions whose size is not a
    multiple of their tile size are marked. Classify each generating
    predicate as grid-level (→ early exit, no work for out-of-bounds CTAs)
    or tile-level (→ predicated load/store inside the tile loop). Write
    the classification explicitly for each dimension.
    (Ref: framework/10_hardware_binding.md, checklist item 4.)
16. **REL Relations column** — Add a separate Relations column to the
    binding table. Every write→read thread pair marked BARRIER REQUIRED.
    Every Combine Group relation marked REDUCE_MERGE. Every stride-1
    access pair marked COALESCED. This column drives the D4 barrier graph
    — any barrier in D4 that cannot trace back to a REL entry here is
    unjustified.
    (Ref: framework/10_hardware_binding.md, checklist item 9;
    framework/11_fsm_phases.md, barrier rules.)
17. **FUN functor consistency** — Write the functor action explicitly at
    each level transition: Grid→Block substitution (blockIdx×B_tile →
    coordinate), Block→Warp substitution (SRG→shuffle, MON→Blelloch),
    Warp→Thread substitution (threadIdx×T_tile). Verify thread-level
    address formula is the functorial image of the block-level formula
    (same structure, reduced scale). Check FUN×MEA: resource budgets
    transform consistently through each functor application.
    (Ref: framework/10_hardware_binding.md, checklist item 10.)

**STREAMING GATE — PROHIBITED RATIONALIZATIONS (MANDATORY when D1 detected Streaming Gate):**
These two patterns have caused architectural failures and are explicitly forbidden:

**A — Test harness authority:**
Measurement-only launch helpers in the test harness (e.g. `launch_topk_compute_only`,
`launch_topk_select_only`, any separately-timed kernel wrapper) do NOT constrain kernel
count or architecture. They wrap whatever the implementation exposes. Citing harness
symbols as an architectural prescription is forbidden. The harness adapts to the
architecture; the architecture never follows the harness.

**B — Degenerate/trivial/pass-through Gate:**
If per-CTA `local_candidates < K` at the chosen aggregation unit, the correct response
is to re-choose the aggregation unit: raise `PAGES_PER_CTA` until `local_candidates ≥ K`,
or commit to single-kernel + `cooperative_groups` grid sync. Describing the Streaming
Gate as "degenerate," "trivial," "pass-through," "empty," or any equivalent framing to
justify skipping the per-CTA buffer is FORBIDDEN. The binding criterion for Streaming
Gate is whether the global N-score round-trip is eliminated — not whether the per-CTA
buffer fills. If `local_candidates < K`, the aggregation unit sizing is wrong, not the
Gate pattern.

**FORBIDDEN cross-CTA mechanisms when Streaming Gate is detected:**
- Global scratch of N raw scores (writing all candidates to global before selection)
- Second kernel that reads N raw scores from global

**Legal cross-CTA mechanisms:**
- Per-CTA top-`local_count` buffer + cross-CTA K-merge
- Single-kernel + `cooperative_groups` grid sync

Any binding that routes through a forbidden mechanism is wrong regardless of
whether it passes local consistency checks.

Save to d3.md.

## V3 — Verify Hardware Binding

Use **claude-opus**. Read: framework/MANIFEST.md (load files 04, 07, 08, 10),
kernel_spec.md, hardware_target.md, workload_envelope.md,
ptx_isa_sections/MANIFEST.md (then load
listed sections), gau-nernst_reference.h, v2.md, d3.md

External review. Verify consistency with GT-1 through GT-22.

Challenges:
1. Indirect addressing: uses GT-6 cp_fenceproxy flow? NOT SMEM→TMA?
2. Dtype granularity from spec, not assumed?
3. Tile shape in PTX ISA Table 41? tcgen05 not WGMMA (GT-1)?
4. State buffer: register budget arithmetic shown?
5. NOT APPLICABLE verdicts: structural condition, not judgment?
6. **GT-1 check:** zero WGMMA references? Any → FAIL.
7. **GT-2/GT-3 lifecycle:** correct commit+mbarrier+fence::after for MMA→ld (§9.7.16.6.4.2)? No wait::mma?
8. **GT-5/GT-19 scale:** K-cache dequant uses the corrected blocked
   layout (GT-19, 2026-04-18): first `page_size*128` bytes of each page =
   FP8 data, last `page_size*4` bytes = `page_size` fp32 scales
   (one per token). NOT a per-row interleaved 128-data + 4-scale split.
   NOT 4 FP8 scales per row. If either wrong layout used → FAIL.
9. **Broadcast scope:** CTA-wide via SMEM+syncthreads, not shfl?
10. **SMEM descriptor:** all args ÷ 16? K-cache 132 not used directly?
11. **TMEM ld coverage:** N/8 calls per slab?
12. **Fence syntax:** bare form, no "::1" suffix (GT-7)?
13. **Tensor map residency:** in global for TMA, cp_fenceproxy flow (GT-6)?
14. **TMA stride alignment (GT-9):** every globalStride in
    `cuTensorMapEncodeTiled` is a multiple of 16? If any is not (e.g.
    132-byte K-cache rows) → tensor MUST use cooperative load, not TMA.
    Any TMA binding with non-÷16 stride → FAIL.
15. **TMA/SMEM swizzle (GT-10):** swizzle mode in TMA matches the
    descriptor mode used by MMA? If TMA uses SWIZZLE_128B but MMA uses
    linear `make_smem_desc` → FAIL.
16. **Gate algorithm (D3 item 12):** Does d3.md contain an explicit K/N
    ratio computation? Does it name the algorithm class (selection vs sort)
    with written justification? If K/N is well below 0.5 and a full sort
    was chosen without justification → FAIL. If no concrete algorithm is
    named → FAIL. Cross-check against framework/07_gate_specification.md +
    framework/08_streaming_gate.md.
    **If Streaming Gate was detected at D1, additionally verify:**
    (a) Does d3.md state whether per-CTA `local_candidates ≥ K`?
    (b) If `local_candidates < K`, does d3.md prescribe raising
        `PAGES_PER_CTA` or committing to single-kernel grid sync — NOT
        skipping the per-CTA buffer? If d3.md concludes "skip buffer"
        from `local_candidates < K` → FAIL.
    (c) Does d3.md use the words "degenerate," "trivial," "pass-through,"
        or "empty" to describe the Streaming Gate? If so → FAIL.
    (d) Does d3.md bind to global N-score scratch as the cross-CTA
        mechanism? If so → FAIL.
    (e) Is the K/N ratio computed over the full N candidates before any
        per-CTA filtering — not over a per-CTA subset? If applied to the
        wrong input set → FAIL.
17. **MEA four-measure (D3 item 13):** Does d3.md contain an explicit SMEM
    budget line with all variable buffers at max size? Register budget?
    Reuse count > 1 for every staged buffer? Occupancy computed? Any
    missing measure → FAIL.
18. **SM utilization (D3 item 14):** Does d3.md compute min_batch for full
    SM occupancy? If grid is (batch, 1, 1) and min_batch > typical batch
    size without evaluating multi-CTA → FAIL.
19. **PRD boundaries (D3 item 15):** Does d3.md classify every non-aligned
    dimension as grid-level or tile-level predicate? If missing → FAIL.
20. **REL Relations (D3 item 16):** Does d3.md include a Relations column
    in the binding table? Every write→read pair marked? If missing → FAIL.
21. **FUN functor (D3 item 17):** Does d3.md write explicit functor actions
    at each level transition? If missing → FAIL.
22. **Precision-Match Gate (GT-21):** Does d3.md contain an explicit
    precision-match analysis for ordering-dependent outputs? If the
    reference uses FP32 and the kernel uses FP8 MMA, does d3.md choose
    path (a), (b), or (c) with justification? If missing → FAIL.
23. **Matmul shape match (GT-22, MANDATORY for GT-21 path (a)):** If d3.md
    chose path (a), does it commit to transliterating the reference's
    matmul operand shape byte-identically? Every matmul, every slice
    bound, every view argument. A kernel that matmuls over a longer
    axis and slices after will diverge from cuBLAS at ulp level and
    fail 70-80% of workloads. If d3.md promises "hand-rolled FP32 FMA
    loop" or "batched matmul with padding" without explicit shape-match
    justification → FAIL.
24. **Reference source lookup (GT-19 / GT-22):** If d3.md chose path (a),
    does it cite `definition.reference` (not kernel_spec.md / not
    hardware_target.md) as the authoritative source for tensor layout
    and op sequence? If d3.md describes K-cache as interleaved per-row
    (old GT-19 interpretation) → FAIL. If d3.md doesn't call out running
    the probe to print the reference → FAIL.
25. **Workload envelope coverage (MANDATORY):** Does d3.md reference
    workload_envelope.md? Is the memory budget (item 13a) evaluated at
    the MEM_STRESS corner? Is SM utilization (item 14) checked against
    the SM_STARVED corner? Is the K/N ratio (item 12) evaluated at both
    extremes of K_over_N_range? If any of these use assumed values
    instead of envelope values → FAIL.
26. **Runtime dispatch consistency:** If d3.md introduces a runtime
    dispatch (e.g. algorithm class switch based on K/N at different
    workloads), is the dispatch predicate explicitly stated? Does each
    branch have its own memory budget analysis? If the dispatch is
    introduced but either branch lacks a budget → FAIL.

STATUS: PASS → save v3.md. FAIL → correct, re-verify.

---

## D4 — Architecture Specification

Read: framework/MANIFEST.md (load files 08, 11, 12, 13), v3.md.
All decisions in v3.md are final.

**Step 3 — FSM Phase Table:** Per confirmed molecule: Phase | Entry |
Active threads | Operations | Exit. Reduction uses GT-4 pattern. Gate: two
phases + hard barrier. Pipeline ABSENT (GT-8). **Gate Phase 1 must use the
concrete algorithm chosen in D3 item 12** — if D3 chose a selection-class
algorithm, the FSM must reflect selection phases (partition + gather), not
sort phases. Cross-reference framework/08_streaming_gate.md FSM structure
(if Streaming Gate) or framework/07_gate_specification.md (if single-pass).

**Step 4a — Barrier Graph:** ID | Type | Writer | Reader | Justification.
Every barrier must name the data hazard.

**Step 4b — Reuse Validation:** Reuse counts, tile loop bound, SMEM budget.

**Step 5 — Address Composition:** thread_addr = block_base + warp_offset +
lane_offset as FMAs. Preserve loads as loads. TMEM addressing is simple
row-major (no WGMMA fragment formulas).

**Section 5 — Summary:** One paragraph: computation, multi-phase molecules,
indirection resolution, dtype handling (GT-5/GT-19), thread groups (GT-4),
TMEM lifecycle (GT-2), state buffer locations.

Save to d4.md.

## V4 — Verify Architecture Specification

Use claude-sonnet. Read: framework/MANIFEST.md (load files 08, 11, 12, 13),
v3.md, d4.md

External review. Challenges: (1) FSM coverage per molecule, (2) barrier
necessity, (3) Gate barrier placement, (4) address fidelity vs Affine Map
registry, (5) summary completeness including GT-2 lifecycle with
commit+mbarrier+fence::after (NOT fence+sync+fence per GT-3, NOT wait::mma per GT-3), (6) zero WGMMA references (GT-1), (7) **Gate Phase 1 algorithm:** does d4.md's FSM use the algorithm class chosen in d3.md item 12? If d3 chose selection-class but d4 uses a sort → FAIL. The FSM must reflect the D3 decision.
(8) **Streaming Gate degenerate framing:** Does d4.md describe the Streaming
    Gate as "degenerate," "trivial," "pass-through," or "empty" in any form?
    If so → FAIL. Does d4.md include a global N-score write (e.g.
    `g_scores[batch*T_max + pos] = ...`) as the cross-CTA mechanism? If so →
    FAIL. A correct Streaming Gate architecture never writes all N raw scores
    to global memory before selection.

STATUS: PASS → "Architecture spec verified. Safe to proceed to
implementation." Save v4.md.

---

## Implementation (Incremental Build)

Read: d4.md, ptx_isa_sections/MANIFEST.md (load sections for current layer),
gau-nernst_reference.h, incremental_impl_protocol.md

Do NOT read framework/ files here (reference d4.md and ptx_isa_sections only).

The kernel is built in 9 layers (0–8). Layers 0–7 are tested on B200 via
`python run_test.py --layer N` before proceeding. Layer 8 is packaging only
— no test run required, but a dry-run pack is mandatory. See
incremental_impl_protocol.md for layer definitions, pass/fail criteria,
and the automatic TMA fallback rule.

#### Pre-Implementation Checklist (MANDATORY)

0. **Precision-Match Gate (GT-21, BLOCKING):** Has d3.md resolved the
   precision-match question? If the answer is "FP8 MMA for a FP32
   reference top-k track," has the expected pass rate been documented
   and accepted? Do NOT begin implementation until this is resolved.
   Discovering the precision gap after Layer 6 wastes the entire
   derivation run.
0b. **Reference source lookup (MANDATORY for GT-21 path (a)):** If d3.md
   chose path (a) — delegate scoring to `torch::matmul` / `at::matmul` to
   match an FP32 reference — you MUST run `probe_grader_oracle.py` (or
   `probe_ref_vs_kernel.py`) before writing any C++ and print
   `definition.reference`. The authoritative formula is the Python source
   of the reference function:
   - NOT the DPS spec text
   - NOT `kernel_spec.md` (it was subtly wrong about GT-19 layout)
   - NOT `hardware_target.md` (ditto)
   - NOT prior GT entries without an explicit "confirmed on this track" date
   Transliterate the reference line-for-line. Every op's operand shape,
   every `.view()` argument, every slice bound, every sum axis must
   match byte-identically (GT-22). If the reference uses blocked K-cache
   layout (GT-19), use blocked. If the reference slices K before matmul,
   slice K before matmul. This is not "style preference" — it's bit-exact
   correctness on FP32 reductions.
1. No WGMMA instructions (GT-1).
2. No `tcgen05.wait::mma` — does not exist (GT-3). MMA→ld ordering uses `commit → mbarrier_wait → fence::after_thread_sync` (PTX ISA §9.7.16.6.4.2). `fence::before + __syncthreads() + fence::after` is for cross-thread pipelined ordering (e.g. `cp → mma`) only.
3. `tcgen05.alloc`/`dealloc` bracket all MMA work, all paths reach dealloc.
4. Single elected thread issues `tcgen05.mma` via `elect_one_sync()`.
5. Per-slab scale in registers after `tcgen05.ld`, not in MMA pipeline. **K-cache is blocked per page (GT-19 corrected):** the first `page_size*128` bytes of each page are all FP8 data (tokens `[0..63]`), the last `page_size*4` bytes are `page_size` fp32 scales — NOT a per-row `[128 data | 4 scale]` interleave. For tcgen05 FP8 MMA the scale is one fp32 per token, applied after the full 128-element dot product (not four fp8, not per 32-element slab).
6. `h_partial_smem` sized for **4 warps** (GT-4/GT-12: all 4 warps hold live MMA rows for kind::f8f6f4 M=64 cta_group::1).
7. Both fence variants for cross-thread sync, bare form no "::1" (GT-7).
8. tcgen05.ld coverage: N/8 calls per slab (PTX ISA §9.7.16.8.3).
9. SMEM descriptor args ÷ 16, bits[46:48]=0b001 (PTX ISA Table 42).
10. IDESC derived from PTX ISA Table 44 with bit-by-bit comment.
11. Tensor map in global for TMA, full GT-6 flow (replace → cp_fenceproxy
    → acquire → TMA). Per-CTA global working buffers.
12. No `cuda::ptx::` wrappers, no `<cuda/ptx>` (GT-7). Raw inline PTX only.
13. Dynamic SMEM opt-in if > 48 KB.
14. `cuTensorMapEncodeTiled`: all globalStrides divisible by 16 (GT-9).
    If any stride is not ÷ 16, use cooperative thread copy instead of TMA.
15. TMA swizzle mode matches SMEM descriptor layout assumption (GT-10).
    If using linear `make_smem_desc`, TMA must use `SWIZZLE_NONE`.
16. **DPS + TVM FFI compliance:** Raw-pointer `launch_topk_c` uses the
    exact parameter list from the Competition Submission Constraints section.
    No `T_max`. No scratch pointer. Merge buffer dynamic. Static scratch via
    GT-16 pattern. `TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, ...)` macro
    present in kernel.cu. Any violation is a submission blocker.
17. **NaN handling — load-time detection (MANDATORY):** K-cache random
    init produces NaN FP8 bytes (0x7F and 0xFF in E4M3FN). FP8 MMA does
    NOT propagate NaN identically to the reference's FP32 path. Fix:
    during K-cache cooperative load, scan each token's 128 data bytes for
    0x7F or 0xFF (unsigned). Build a per-token NaN bitmask in SMEM. After
    h-reduction, overwrite any NaN-masked token's score with +INFINITY
    (`__int_as_float(0x7F800000)`). This ensures the kernel's NaN set
    matches the reference exactly. Bitonic sort tie-break: +INF ties use
    ASC by index (smaller first); finite ties use DESC by index (larger
    first).

#### Testing Protocol

After writing each layer's .cu file:
1. Save to TTF_top_k.cu
2. Run: `python run_test.py --layer N`
   (This syncs TTF_top_k.cu → solution/cuda/kernel.cu automatically,
   then drives the official competition runner via `modal run scripts/run_modal.py`.
   All 128 competition workloads are tested on every run.)
3. Read test_result.json
4. For layers 0–5: PASS = compiled=True, xid_13=False, hang=False
   (workload correctness failures are expected — kernel is incomplete)
5. For layers 6–7: PASS = all_passed=True across all competition workloads
6. On failure at layers 6–7: cross-reference failing workload UUIDs against
   workload_envelope.md corners to classify:
   - Failures ONLY on corner workloads (SMALL_BATCH, LONG_SEQ, etc.) →
     likely a boundary condition or parameter-dependent bug. Check buffer
     sizing, loop bounds, predication logic.
   - Failures on typical AND corner workloads → likely a structural or
     algorithmic bug. Use standard diagnostic flow.
7. Fix, retest (max 3 retries per layer)
8. On 3 retries exhausted: escalate to diagnostic agent (see below)

#### Implementation Rules

1. Every FSM phase → `// PHASE [name]: [desc]` section.
2. Every barrier from graph present in code.
3. Addresses implement exact FMA chains from composition table.
4. Dtype handling exactly as specified (GT-5/GT-19 scale pattern).
5. Gate phases separated by exact barrier.
6. Native PTX only. No library substitutions.
7. No optimizations not in spec. No spec elements removed. d4.md is
   authoritative — it is the output of THIS run's derivation and
   reflects all decisions made in D1 through D4 of this run. The
   implementation must match d4.md's FSM, barrier graph, and address
   composition exactly. External references (prior TTF_top_k.cu
   files, reference implementations, "known good" kernels from earlier
   runs) are NOT authoritative, even if they pass correctness tests.
   If any external kernel's architecture differs from d4.md, follow
   d4.md. The correctness of an external kernel proves only that that
   kernel was correct for its own architecture — not that its
   architecture matches this run's d4.md.
8. Unimplementable items: `// SPEC CONFLICT [id]: [description]`
9. Full TMEM lifecycle per GT-2.
10. TMA via GT-6 flow for tensors with ÷16 strides. Cooperative thread
    copy for tensors with non-÷16 strides (GT-9).
11. Algorithmic primitives from d4 (bitonic sort, Blelloch, introselect,
    etc.) must be used — no serialized fallbacks. The optimization pass
    may later replace the algorithm with one from the same or a superior
    algorithm class (e.g. replacing a sort with a selection algorithm if
    the Gate K/N analysis supports it), but the initial implementation
    must match d4's FSM.
12. CTA-wide broadcasts via SMEM+syncthreads, not shfl.

Save to impl.md.

---

### Layer 8 — Submission Packaging

**Trigger:** Layer 7 passes (`python run_test.py --layer 7` → ALL_PASS).

**Steps (host-side only — no GPU test run):**

#### 8a. Verify kernel sync

```bash
diff TTF_top_k.cu solution/cuda/kernel.cu && echo "IN SYNC" || echo "OUT OF SYNC"
```

If out of sync: `cp TTF_top_k.cu solution/cuda/kernel.cu`

#### 8b. Verify TVM FFI macro present

```bash
grep -c "TVM_FFI_DLL_EXPORT_TYPED_FUNC" TTF_top_k.cu
# Must print: 1
```

If missing, add the wrapper from the Competition Submission Constraints
section before proceeding. Without this macro the harness will compile
the .so but report COMPILE_ERROR (symbol not found in tvm_ffi module).

#### 8c. Verify config.toml

```bash
grep 'definition' config.toml
# Must print: definition  = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
```

#### 8d. Dry-run pack

```bash
python scripts/pack_solution.py
```

#### 8e. Layer 8 pass criterion

Layer 8 PASSES when all of the following are true:
- `solution/cuda/kernel.cu` exists and is identical to `TTF_top_k.cu`
- `TTF_top_k.cu` contains `TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, ...)`
- `config.toml` contains the correct `definition` value
- `python scripts/pack_solution.py` exits 0
- `solution.json` is present in the repo root

**Do not proceed to Audit until Layer 8 passes.**

---

## Audit

Read: framework/MANIFEST.md (load files 04, 07, 08, 10, 11),
kernel_spec.md, ptx_isa_sections/MANIFEST.md (load audit sections),
gau-nernst_reference.h, d4.md, impl.md

Report only discrepancies. For each: Item | Reference | Spec says | Code
does | Corrected code. If unverifiable: UNVERIFIABLE — [missing info].

**Audit completeness gate (MANDATORY):** Before emitting any PASS verdict,
count the numbered items in this audit output. The expected count is 28. If
the count is below 28, emit INCOMPLETE, list the missing item numbers, and
do not proceed. A PASS on fewer than 28 items is invalid.

1. Phase structure complete?
2. Barrier placement correct?
3. Gate integrity (no scoring after barrier, no selection before)?
4. Address formulas exact?
5. Dtype/scale handling per GT-5/GT-19 (corrected 2026-04-18)? K-cache
   dequant uses blocked-per-page layout: first `page_size*128` bytes =
   FP8 data, last `page_size*4` bytes = `page_size` fp32 scales? NOT
   interleaved per-row, NOT 4 FP8 values?
6. Indirect addressing per GT-6? Raw PTX, not cuda::ptx::?
7. Tile shape correct? tcgen05, not WGMMA (GT-1)?
8. State buffers placed correctly? Accumulator in TMEM?
9. Combine Group merge correct?
10. Tile loop bound correct?
11. **GT-1:** zero WGMMA? → CRITICAL if found.
12. **GT-2/GT-3:** full lifecycle? No wait::mma? MMA→ld uses commit+mbarrier+fence::after (§9.7.16.6.4.2)?
13. **GT-4/GT-12:** h_partial_smem sized for 4 warps? (Prior "2 warps" was wrong for this MMA shape — see GT-4.)
14. elect_one_sync for MMA?
15. Broadcast scope: SMEM+syncthreads, not shfl across warps?
16. **SMEM descriptor:** args ÷ 16? bits[46:48]=0b001? Check against
    PTX ISA Table 42 and gau-nernst_reference.h.
17. **tcgen05.ld:** N/8 calls? acc_total sized [N]?
18. **Fence/commit pattern:** For MMA→ld (same or different thread): `commit + mbarrier_wait + fence::after_thread_sync` used (§9.7.16.6.4.2–4.4)? `fence::before_thread_sync` is implicit in `tcgen05.commit` — explicit call before the mbarrier is a bug. For cross-thread pipelined (cp→mma): `fence::before + __syncthreads() + fence::after` used (§9.7.16.6.4.3)? All fence variants bare form, no `::1` suffix (GT-7).
19. **IDESC:** derived from Table 44 with bit comment?
20. **Tensor map:** in global for TMA? GT-6 flow? → CRITICAL if SMEM.
21. **tensormap.replace:** both .b1024 AND .b64?
22. **No cuda::ptx:: or <cuda/ptx>** → CRITICAL if found.
23. **cuTensorMapEncodeTiled (GT-9):** all globalStrides ÷ 16? If any
    non-÷16 stride uses TMA instead of cooperative load → CRITICAL.
24. **TMA/SMEM swizzle (GT-10):** If TMA uses SWIZZLE_128B but MMA
    uses linear make_smem_desc → CRITICAL (wrong results, no fault).
25. **Gate algorithm class (D3 item 12):** Does the implementation use the
    algorithm class chosen in D3? If D3 chose selection-class but
    implementation uses a sort → flag for optimization pass review. Not
    CRITICAL (correctness is preserved), but note the mismatch.
26. **MEA budget (D3 item 13):** Does the SMEM usage at max runtime
    parameters fit within the budget computed in D3? If any buffer can
    exceed the budget → CRITICAL.
27. **Streaming Gate global-write check (MANDATORY when D1 detected Streaming
    Gate):** Search the implementation for score-shaped global writes: patterns
    like `g_scores[batch*T_max + pos]`, `st.global` on score-shaped addresses,
    or any write of N raw scores to global memory before the Gate selection
    phase. If any such write exists → CRITICAL. A Streaming Gate implementation
    must never write all N raw scores to global memory — that is the round-trip
    the Gate exists to eliminate.
    Also verify merge-phase SMEM budget: if Streaming Gate was detected, the
    merge phase SMEM must be ≥ NUM_CTAS_PER_BATCH × per-CTA-K-buffer-entry-size.
    A merge-phase SMEM budget below 1 KB when K > 256 is a structural signal
    of global-scratch architecture → CRITICAL.
28. **Submission packaging (TVM FFI + DPS compliance):** Does kernel.cu
    contain `TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_topk_c, ...)`? Does the
    raw-pointer `launch_topk_c` signature match the DPS spec exactly (no
    `T_max`, no scratch pointer)? Is merge buffer sized dynamically via
    `cudaFuncSetAttribute`? Does `solution/cuda/kernel.cu` match `TTF_top_k.cu`?
    Does `python scripts/pack_solution.py` exit 0?
    Any violation → CRITICAL (kernel fails DPS validation before any
    correctness or performance measurement).

Output: Part 1 — findings. Part 2 — corrected impl with `// AUDIT FIX [id]`.
Save to audit.md.

---

## Full Workload Sweep

**Run AFTER the Audit passes and BEFORE saving TTF_top_k_correct.cu.**
This gate ensures the kernel is correct across all 128 competition
workloads, not a subset.

### Procedure

1. Run: `python run_test.py --layer 7`
   This executes the kernel against all 128 workloads from the dataset
   via `modal run scripts/run_modal.py`.
2. Read test_result.json. Classify results:

   | Category | Criterion | Action |
   |----------|-----------|--------|
   | PASS | All 128 pass (`all_passed=true`) | Save TTF_top_k_correct.cu, proceed to Optimization |
   | EDGE_FAIL | ≤ 5 failures, all on corner workloads from workload_envelope.md | Patch and retest — likely boundary bugs |
   | STRUCTURAL_FAIL | > 5 failures or failures on typical (non-corner) workloads | Architectural problem — see recovery below |

3. **EDGE_FAIL recovery:**
   For each failing workload UUID, cross-reference workload_envelope.md
   to identify which corner it belongs to. Common causes:
   - MEM_STRESS corner: buffer overflow at extreme parameters → add
     runtime buffer sizing or allocation guard.
   - SM_STARVED corner: grid shape produces too few CTAs → adjust
     grid formula or add early-exit guard.
   - SHORT_SEQ corner: num_pages or seq_len below a loop minimum →
     add bounds check or fast-path.
   - MAX_K_RATIO corner: topk ≈ total_tokens → handle as identity.
   Fix, retest against all 128. Max 3 fix-retest cycles.

4. **STRUCTURAL_FAIL recovery:**
   The kernel architecture is fundamentally mismatched with the
   workload range. Do NOT patch. Instead:
   - Identify which D3 binding decision is wrong by correlating
     failures with workload parameters (use workload_envelope.md
     corners to classify).
   - Update workload_envelope.md with the failure analysis.
   - Re-derive from D3 (or D1 if a molecule detection decision is
     implicated).
   - Rebuild from the affected layer.

5. **Only after all 128 pass:** Save as TTF_top_k_correct.cu.
   Proceed to Optimization Pass.

---

## Diagnostic Escalation

If any layer is BLOCKED after 3 retries during incremental build:

### Stage 1: Diagnostic Agent

Invoke a fresh **claude-opus** agent with the diagnostic role. Provide:
- test_result.json (last 3 failures)
- TTF_top_k.cu (failing kernel)
- d3.md, d4.md, hardware_target.md, relevant PTX ISA sections, gau-nernst reference

The agent first checks the Failure Pattern Signatures table, then classifies
the failure. If FAILURE_CLASS = DESCRIPTOR, the agent runs:

    modal run run_diagnostic.py

before proposing any fix. The sweep output provides the correct SBO/LBO
encoding empirically. For all other failure classes, the agent may write
one probe to `diagnostic_tests/probe.cu` and run it via:

    modal run run_diagnostic.py --probe probe

The diagnostic agent produces diagnosis.md with:
FAILURE_CLASS, ROOT_CAUSE, EVIDENCE, AFFECTED_DECISIONS,
PROPOSED_FIX, NEW_GT_CANDIDATE.

Apply the proposed fix and retry the blocked layer (3 fresh attempts).

### Stage 2: Bisection Agent

If Stage 1 fix fails, invoke a **claude-sonnet** bisection agent that:
1. Starts from the last passing layer's kernel
2. Adds one element at a time from the failing layer
3. Tests each addition via `python run_test.py`
4. Produces minimal_repro.cu (smallest failing file)
   and minimal_pass.cu (without the breaking element)
5. Reports which specific element causes the failure

### Stage 3: Human Review

If Stage 2 cannot isolate the failure, escalate to human with:
- Bisection report
- Minimal repro files
- Recommended spec changes (with options)
- Which D-step files need updating

The human updates spec files and the pipeline re-derives from
the affected D-step.

### GT Update Rule

Every successful diagnosis that reveals a new hardware constraint
becomes a new GT-N entry in CLAUDE.md and hardware_target.md.
This prevents the same failure from occurring in future kernel
derivations. Probe outputs from `run_diagnostic.py` that confirm
a hardware-specific encoding (such as SBO/LBO values from the
sbo_lbo_sweep) are treated as confirmed hardware facts and
recorded with the date and probe name that confirmed them.

---

## Optimization Pass

After all layers pass and the kernel is correct:

1. Save the correct kernel as TTF_top_k_correct.cu (never modify this).
2. Invoke the **claude-opus** optimization agent following optimization_pass.md.
3. Each optimization is applied one at a time and tested via
   `python run_test.py --layer 7`.
   Every optimization must pass ALL 128 workloads (`all_passed=true`).
   An optimization that improves performance on typical workloads but
   fails a corner workload from workload_envelope.md is rejected.
4. Only optimizations that preserve all test passes are kept.
5. Output: optimization_report.md + TTF_top_k_optimized.cu

The optimization agent works through four tiers:

- **Tier 1: Free wins** — dead code, compiler hints, coalescing, bank
  conflicts, redundant barriers.
- **Tier 2: Molecule-aware algorithmic improvements** — Before optimizing,
  re-read framework/05_structural_analysis.md + framework/07_gate_specification.md
  + framework/08_streaming_gate.md for each confirmed molecule and evaluate
  whether the current implementation uses the optimal algorithm for the
  molecule's parameters:
  - **Gate molecule:** Re-evaluate the K/N ratio from D3 item 12. If the
    current implementation uses a full sort (bitonic, radix sort) but K/N
    < 0.5, test selection-based alternatives from framework/07_gate_specification.md: introselect, radix-select (partition to find k-th
    element then gather top-k), warp-level top-k insertion buffer, or
    multi-pass threshold binary search. A full sort is O(N log N); a
    selection is O(N). The algorithm class change is explicitly permitted
    (see Implementation rule 11).
    After selecting a heap or insertion-buffer algorithm, evaluate whether
    the insert/sift phase is serial on a single thread. If so, test a
    parallel variant: distribute the T candidates across all CTA threads,
    use a parallel reduction to identify survivors beating the heap root,
    then perform a warp-cooperative heap update. A serial O(T × log K)
    sift on tid==0 with the remaining threads idle is a candidate for
    parallelization even after the algorithm class has been confirmed
    correct. Measure both variants; keep the faster one.
  - **Tile molecule:** Re-evaluate the serial-vs-parallel CTA decision
    from D1/D3. If the kernel uses one CTA per batch element and SM
    utilization is below 80%, test multi-CTA partitioning with the tile
    loop split across a grid dimension.
  - **Reduction molecule:** Evaluate whether warp-level reductions can be
    tightened (fewer shuffle rounds, fused scale-and-reduce).
  - **Pipeline/double-buffering:** Even with Pipeline ABSENT, if the page
    loop contains a hard `__syncthreads()` between the dominant load and
    the MMA, the load and compute are fully serialized regardless of how
    many SMEM buffers exist. Two SMEM buffers allocated (structural
    double-buffering) does NOT hide latency unless the next iteration's
    load is initiated asynchronously before the current iteration's MMA
    completes (pipelined double-buffering). Test initiating the next
    page's load immediately after the `__syncthreads()` that publishes
    the current page's data, before entering the slab loop, so that K
    load latency is hidden inside the MMA barrier wait cycles. Do not
    gate this test on a profiling condition — apply it whenever the
    page loop structure serializes load and MMA.
  - Also: vectorized loads, fused operations, pipelining.
- **Tier 3: Major restructuring** — multi-CTA partitioning, persistent
  kernels, kernel splits, library substitutions (CUB, Thrust). Evaluate
  when Tier 2 is exhausted or profiling shows a structural bottleneck.
- **Tier 4: Micro-optimizations** — instruction scheduling, shared memory
  padding, register pressure tuning.

**Optimization re-evaluation rule (MANDATORY):** After each tier, profile
the kernel and identify the dominant cost center. Before proceeding to the
next tier, ask: "Does the dominant cost center correspond to a molecule
whose algorithm class might be wrong?" If the dominant cost is a Gate
Phase 1 sort and K/N is well below 1.0, selection-based alternatives MUST
be tested before declaring the tier complete. The framework's algorithm
vocabulary (framework/07_gate_specification.md + framework/08_streaming_gate.md)
is authoritative — do not limit candidates to variations of the current
algorithm.

Every optimization must be tested. No untested optimizations in the
final kernel.

---

## Progress Reporting

After each step: `check [STEP] — [name] — [one-sentence summary]`
On fail: `FAIL [V-step] — [issue]` then `CORRECTING [D-step]`
On skip: `SKIP D1, V1, D2, V2 — prior outputs exist`
On envelope extraction: `ENVELOPE — [N] workloads parsed — [batch_size min..max] — [num_pages min..max]`
On layer test: `LAYER N — [PASS/FAIL] — [summary from test_result.json]`
On full sweep: `SWEEP — [PASS/EDGE_FAIL/STRUCTURAL_FAIL] — [N]/128 passed — [failing corners if any]`
At end: summary table of all steps with output file sizes.
