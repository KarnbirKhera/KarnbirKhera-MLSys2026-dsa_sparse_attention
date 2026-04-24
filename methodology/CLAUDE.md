# DSA Sparse Attention BF16 Kernel — Generation and Optimization Workflow
# Target: NVIDIA B200 (sm_100a) via Modal
#
# This CLAUDE.md is seeded from the completed top-k indexer project
# (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`). Only kernel-agnostic
# infrastructure, methodology, and hardware-level GTs were carried over.
# Kernel-specific baselines, optimization outcomes, and roofline conclusions
# were intentionally left out — they belong to the top-k kernel and would
# poison reasoning about this one.

---

## Section 1 — Kernel Identity

**Definition name:** `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

**What it computes:** DeepSeek-V3.2-style MLA sparse attention. For each
query token `t`, select up to `TOPK` KV-cache rows via `sparse_indices[t]`
(with `-1` sentinels for invalid slots), then compute attention as
`softmax((q_nope @ Kc.T + q_pe @ Kp.T) * sm_scale) @ Kc`. Emit the
attention-weighted output AND the 2-base log-sum-exp of the scaled logits.
See the `reference` field in the definition JSON for the authoritative
Python implementation.

**Fixed constants — do not re-derive these:**

| Constant | Value | Meaning |
|---|---|---|
| `NUM_QO_HEADS` | 16 | Query heads after TP-8 split (128 / 8) |
| `HEAD_DIM_CKV` | 512 | Compressed KV head dimension |
| `HEAD_DIM_KPE` | 64 | Key positional-encoding dimension |
| `PAGE_SIZE` | 64 | Tokens per KV cache page |
| `TOPK` | 2048 | Max selected KV entries per query token |

**KV cache memory layout — plain strided bfloat16, no dequant:**
- `ckv_cache`: `[num_pages, 64, 512]` bf16 → each page is `64 × 512 × 2 = 65536` bytes.
- `kpe_cache`: `[num_pages, 64, 64]`  bf16 → each page is `64 × 64 × 2 = 8192` bytes.

Both caches are 16-byte aligned on every dimension, so TMA is usable on both
(unlike the top-k project's FP8 cache which had a 132-byte stride — see
discussion in [IMPORTED] GTs that reference that constraint; it does not
apply here). There is no interleaved scale field, no uint8 reinterpretation,
no per-token dequant. Decode the sparse index as `page_idx = idx / PAGE_SIZE`,
`offset = idx % PAGE_SIZE`, then `global_row = page_idx * PAGE_SIZE + offset`.
Flattening `reshape(-1, head_dim_*)` gives exactly that row ordering.

**Variable axes (differ per workload):** `num_tokens` (acts as batch-dim;
decode or prefill), `num_pages`.

**Inputs** (all bfloat16 unless noted): `q_nope[num_tokens, 16, 512]`,
`q_pe[num_tokens, 16, 64]`, `ckv_cache[num_pages, 64, 512]`,
`kpe_cache[num_pages, 64, 64]`, `sparse_indices[num_tokens, 2048]` int32,
`sm_scale` fp32 scalar (0-d tensor).

**Outputs (DPS — harness passes in pre-allocated tensors):**
- `output[num_tokens, 16, 512]` bf16 — attention-weighted value per head.
- `lse[num_tokens, 16]` fp32 — 2-base log-sum-exp of the scaled logits.

**Sentinel rules:** when the valid-index set for token `t` is empty
(all `-1`), `output[t]` is zero and `lse[t]` is `-inf`. The reference
relies on this; the kernel must reproduce it.

---

## Section 2 — GT Update Rule ★ MOST IMPORTANT RULE IN THIS FILE ★

Every time a test run confirms a new hardware constraint, immediately write a
new GT-N entry at the bottom of Section 7 (always-active) or Section 11
(Tier 3 only), depending on scope. Include the date and the probe or test
that confirmed it.

**This rule is non-negotiable and applies after every diagnostic resolution.**
It is what prevents the same failure from occurring twice. A diagnosis that
does not produce a GT entry is incomplete.

Format:
```
### GT-N: [short title]
[What was confirmed, what the correct behavior is, what the wrong behavior looks like.]
Confirmed YYYY-MM-DD on Modal B200 via [probe name or test that confirmed it].
```

---

## Section 2b — Imported GT Re-Confirmation Rule

GTs in this file prefixed **[IMPORTED]** were confirmed on the top-k indexer
kernel (`dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`). They are included
because they are either (a) pure hardware facts that cannot vary by kernel,
or (b) infrastructure constraints that apply to anything built on this
starter kit / Modal path.

**Rule:** before relying on an [IMPORTED] GT to make an optimization
decision for this kernel, re-confirm it on a sparse-attention-shaped probe
(or, for infrastructure GTs, on the first successful compile+run). If
re-confirmation succeeds, append a short "Re-confirmed YYYY-MM-DD on sparse
attention via [probe]" line to the GT. If it fails, strike the GT and write
a new one that explains why this kernel's regime is different.

The top-k project already has one instance of this pattern: GT-8's roofline
conclusion ("pipelining is ABSENT") was overturned by GT-35 once FP8 MMA
reduced HBM traffic. Assume every roofline- or regime-dependent GT can
similarly flip between kernels.

---

## Section 3 — Project Structure

```
~/CUDAExperiments/sparse_attention/
├── CLAUDE.md                                      <- this file
├── fi-bench-env/                                  <- Python venv (activate before any command)
├── mlsys26-contest/                               <- dataset
│   ├── definitions/<op_family>/                   <- single-nested (e.g. dsa_paged/)
│   │   └── <definition_name>.json                 <- Python reference + spec
│   ├── workloads/<op_family>/
│   │   └── <definition_name>.jsonl                <- workloads
│   └── blob/workloads/<op_family>/
│       └── <definition_name>/                     <- safetensors
├── flashinfer-bench-starter-kit/
│   ├── config.toml                                <- definition + entry_point (keep in sync)
│   ├── solution/cuda/
│   │   └── kernel.cu                              <- THE file the agent edits
│   └── scripts/
│       ├── run_modal.py                           <- test command
│       └── pack_solution.py                       <- packs kernel.cu into solution JSON
├── checkpoints/
│   └── kernel_naive.cu                            <- LOCKED after Phase 1 passes 23/23. NEVER modified.
├── framework/                                     <- derivation framework (Phase 2 Tier 2+)
├── ptx_isa_sections/                              <- PTX ISA reference (Phase 2 Tier 3 only)
└── gau-nernst_reference.h                         <- B200 PTX wrappers (Phase 2 Tier 3 only)
```

**Definition JSON path — write it literally, do not construct from op_type:**
`mlsys26-contest/definitions/<op_family>/<definition_name>.json`
(For this kernel: `mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json`.)

**Python reference ground truth:** The `reference` field inside the
definition JSON contains the canonical `run()` function. This is the
authoritative specification for what the kernel must compute. On any
correctness failure, diff the kernel output against this reference on the
smallest failing workload.

---

## Section 4 — Environment

Before running any Python command or test:

```bash
source ~/CUDAExperiments/sparse_attention/fi-bench-env/bin/activate
```

If this is not active, all `modal` and `flashinfer_bench` commands will fail
silently or use the wrong packages. Verify with `which python` — it must show
the venv path.

**Modal workspace:** `khera-karnbir`
**Modal volume:** `flashinfer-trace` — contains the dataset at `/data/mlsys26-contest`
**FIB_DATASET_PATH** (for local reference only): `~/CUDAExperiments/sparse_attention/mlsys26-contest`

All benchmarking runs on Modal B200. Never attempt to run benchmarks locally —
the local GPU is an RTX 4060 (sm_89) and will produce RUNTIME_ERROR on sm_100a
targeted binaries.

---

## Section 5 — The Two Phases

```
Phase 1: Write naive kernel -> pass 23/23 -> lock checkpoint -> record baseline
Phase 2: Per-optimization cycle (tiered) -> D1-V4 scoped to delta -> implement -> test -> keep/revert
```

Phase 1 is a correctness-first naive kernel. No optimization logic. Once it
passes 23/23, copy it to `checkpoints/kernel_naive.cu` and never modify that
file again — it is the revert target for every failed Phase 2 attempt.

Phase 2 uses the derivation framework — scoped per optimization and tiered by
complexity. D1-V4 fires once per optimization change, not once at the start of
the entire phase.

---

## Section 6 — Phase 1: Naive Kernel

Goal: pass 23/23 on Modal B200 with a straightforward CUDA implementation
that mirrors the Python reference. Do not optimize. Do not use tcgen05, TMA,
clusters, or any sm_100a-specific instructions yet.

**Phase 1 audit checklist (must all pass before locking checkpoint):**

1. KV cache read as bf16 (no uint8 reinterpretation — that was an FP8-project pattern).
2. Sparse-index decode: `page_idx = idx / PAGE_SIZE`, `offset = idx % PAGE_SIZE`,
   `global_row = page_idx * PAGE_SIZE + offset` — matches `reshape(-1, head_dim)`.
3. `-1` sentinel entries in `sparse_indices` are skipped (not treated as row 0).
4. Empty valid-index token: `output[t]=0`, `lse[t]=-inf` (reference parity).
5. `sm_scale` extracted as fp32 scalar once, not re-read per token.
6. `lse` written in 2-base (divide the natural `logsumexp` by `ln(2)`).
7. Output shape and dtype match the definition JSON exactly (bf16 output, fp32 lse).
8. Small-workload correctness confirmed (num_tokens=1, num_pages=1).
9. 23/23 PASS on Modal B200.

**Once Phase 1 passes, record a GT-BASELINE entry in Section 7 with:**
- mean / p50 / p95 / min / max latency in ms
- speedup range vs Python reference
- Modal app id of the confirming run

Every Phase 2 optimization is measured against this baseline. Any
optimization that does not improve mean latency AND passes 23/23 is not
worth keeping.

**Revert command (restore Phase 1 state at any time):**
```bash
cp ~/CUDAExperiments/sparse_attention/checkpoints/kernel_naive.cu \
   ~/CUDAExperiments/sparse_attention/flashinfer-bench-starter-kit/solution/cuda/kernel.cu
```

---

## Section 7 — Always-Active GT Entries

These apply in both Phase 1 and Phase 2. Read these before any implementation
work. Entries tagged **[IMPORTED]** were confirmed on the top-k indexer
project; see Section 2b for the re-confirmation rule.

### [IMPORTED] GT-13: PAGES_PER_CTA Minimum = 2

PAGES_PER_CTA=1 produces non-deterministic results on B200. The minimum
confirmed stable value is 2. Any kernel restructuring that introduces a
PAGES_PER_CTA parameter must set its minimum to 2. Confirmed 2026-04-15 on
Modal B200 (top-k project).

### [IMPORTED] GT-19: Torch Binding Requires nvidia/cuda devel Base Image on Modal

The flashinfer-bench TorchBuilder uses `torch.utils.cpp_extension.load()` to
JIT-compile CUDA kernels. This requires `nvcc`, CUDA headers (including
`cuda_fp8.h`), and build tools to be present in the Modal container. The
default `debian_slim` base image does NOT include these. The correct Modal
image uses `nvidia/cuda:12.8.1-devel-ubuntu22.04` as the base. Using any
non-devel CUDA image produces COMPILE_ERROR on all workloads with no error
message visible in the local terminal.

Additionally, `flashinfer-bench` must be installed from git source (not pip)
to match the evaluation environment:

```python
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("build-essential", "ninja-build", "git")
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /flashinfer-bench "
        "&& cd /flashinfer-bench && pip install -v -e ."
    )
    .pip_install("torch", "triton", "numpy", "ninja")
)
```

Confirmed 2026-04-19 on Modal B200 (top-k project) — resolves all
COMPILE_ERROR failures.

### [IMPORTED] GT-20: pack_solution.py Does Not Read binding Field Without Patch

The starter kit's `pack_solution.py` does not pass the `binding` field from
`config.toml` to `BuildSpec`. Without patching, `binding` defaults to None
(tvm-ffi), causing COMPILE_ERROR when the kernel uses torch binding. The fix —
add two lines to `pack_solution.py` in the `pack_solution()` function:

```python
binding = build_config.get("binding", None)
spec = BuildSpec(
    language=language,
    target_hardware=["NVIDIA_B200"],
    entry_point=entry_point,
    destination_passing_style=dps,
    **({"binding": binding} if binding else {}),
)
```

Confirmed 2026-04-19 (top-k project) — without this patch, binding=torch
kernels fail with COMPILE_ERROR despite correct kernel code.

### [IMPORTED] GT-27: TORCH_CUDA_ARCH_LIST=10.0a Required for tcgen05

Any kernel emitting `tcgen05.*` PTX (alloc/dealloc/mma/commit/fence/ld/st/wait)
MUST be compiled targeting sm_100a, not generic sm_100.
`torch.utils.cpp_extension.load()` selects gencode via the
`TORCH_CUDA_ARCH_LIST` environment variable; unset, it auto-detects from the
live GPU and on B200 defaults to `sm_100` — which ptxas rejects for every
single tcgen05 instruction. Fix: set `TORCH_CUDA_ARCH_LIST=10.0a` in the Modal
image via `.env({"TORCH_CUDA_ARCH_LIST": "10.0a"})`, and defensively re-set
it at function entry (`os.environ.setdefault(...)`). Produces
`-gencode=arch=compute_100a,code=sm_100a` which ptxas accepts. Confirmed
2026-04-19 on Modal B200 (top-k project). Applies to all future Tier 3
kernels; no separate per-optimization retry needed — treat as always-on
infrastructure.

### GT-BASELINE: Naive Kernel Performance Baseline

Locked 2026-04-19. `checkpoints/kernel_naive.cu` passes 23/23 on B200.
Mean: 1.586 ms | p50: 1.715 ms | p95: 2.003 ms | min: 0.970 ms | max: 2.042 ms
Speedup over Python reference: 1.00x–1.06x (the naive kernel is a torch-op
wrapper that mirrors the reference exactly, so abs_err and rel_err are both
0 on every workload — any regression past this point is algorithmic, not
numerical).
Modal app id: ap-HbAW1rnuFlVMMOzssPeu5S
Every optimization in Phase 2 is measured against this baseline.

### GT-28: Scalar Inputs Arrive as Python Floats, Not 0-D Tensors

Inputs declared with `shape: null` in the definition JSON (e.g. `sm_scale`)
are passed through the flashinfer-bench torch binding as raw Python scalars,
NOT as 0-d `torch.Tensor` objects, because
`flashinfer_bench/bench/utils.py` (build_inputs) does
`out.append(workload.inputs[name].value)` for any input whose workload-side
type is `"scalar"`.

**Correct C++ signature:** declare the parameter as `double` (preferred; matches
pybind11's standard Python-float conversion). `float` also works but risks
double->float truncation on the pybind side.

**Wrong behavior:** declaring the parameter as `torch::Tensor` causes every
workload to fail with `RUNTIME_ERROR` because pybind11 has no float→Tensor
conversion and raises `TypeError` on the first call. The traceback is printed
to the Modal container's stderr; no local COMPILE_ERROR because compilation
succeeds.

Add a matching row to Section 8 if the "RUNTIME_ERROR on all workloads, no
OOM, no XID" symptom recurs: check every definition input with `shape: null`
and ensure the C++ param is a plain numeric type.

Confirmed 2026-04-19 on Modal B200 via the Phase 1 naive kernel run: the
initial signature took `torch::Tensor sm_scale` and failed 0/23; swapping to
`double sm_scale` (and `static_cast<float>` at use site) passed 23/23 with
zero other changes.

### GT-37: Persistent device buffers via raw cudaMalloc beat per-call torch::empty/zeros for sub-100µs kernels

When a kernel's wall time falls below ~100 µs (well-tuned attention or short softmax kernels), per-call `torch::empty` overhead (~1-2 µs each) and `torch::zeros` overhead (~1-2 µs for the cudaMemsetAsync on a small buffer) become a measurable fraction. For inter-kernel-persistent scratch (per CLAUDE.md GT-16), switching to a static device pointer + lazy `cudaMalloc` (grow on size change) + kernel-side reset of any zero-initialized state saves ~2-3 µs per launch.

For atomic counters specifically: the kernel that consumes the counter can reset it to 0 before exiting (single-thread store, no atomicity needed when only the last-arrival CTA reaches that code path). Subsequent launches see the zeroed value due to single-stream serialization.

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A22: replacing `torch::zeros({N}, int32)` + 3× `torch::empty(...)` with one persistent counter + one consolidated scratch buffer dropped mean 0.047 → 0.045 ms (−4.3%).

### GT-43: Whole-split empty-sentinel hoist — small tail-latency win on B200

For sparse-attention kernels where a CTA's slice of sparse_indices can be entirely -1 (e.g., tokens whose valid-index count is less than the per-CTA kv_per_split granularity), skipping the TMA+compute inner loop for that split yields a small but reproducible improvement, concentrated on large-N tail workloads.

Implementation pattern (after the SMEM-idx cooperative load is complete):
```c
__shared__ int smem_any_valid;
if (tid == 0) smem_any_valid = 0;
__syncthreads();
// during cooperative idx load, each thread that sees a non-sentinel atomicOr(&smem_any_valid, 1).
__syncthreads();

if (total_iters > 0 && smem_any_valid) {
    // ... existing TMA+compute inner loop ...
}
// Running state row_max=-inf, row_sum=0, o_acc=0 is already the correct empty default.
// Downstream scratch write path is unchanged — reduce phase short-circuits on l_global==0.
```

Cost is one extra `__syncthreads()` + one SMEM write + one SMEM read per CTA when non-empty. Win comes from skipping ~32 iters × (TMA wait + compute) on fully-empty splits (~5-10 µs per skipped split).

Confirmed 2026-04-20 on Modal B200 via Phase 2 attempt A24: mean 0.045 → 0.044 ms (-2.2%), p95 0.061 → 0.059 ms, max 0.061 → 0.059 ms. Reproducible across two consecutive runs.

### GT-38: cp.async.bulk `.L2::cache_hint = evict_first` for streaming KV gather on B200

For sparse-attention KV gather where each row is read once per kernel call and never re-used, marking the TMA reads with `createpolicy.fractional.L2::evict_first.b64` (1.0 fraction) and adding `.L2::cache_hint` to `cp.async.bulk` frees L2 capacity for non-KV reads (scratch_o in the reduce phase, output bf16 writes, etc). Modest but reliable improvement — the L2 doesn't get polluted by single-use KV lines.

PTX:
```
createpolicy.fractional.L2::evict_first.b64 cache_pol, 1.0;
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    [smem], [global], size, [mbar], cache_pol;
```

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A23: mean 0.045 → 0.044 ms (−2.2%).

### GT-36: Last-arrival atomic fused-reduce saves a kernel launch on B200

For split-K kernels with a small reduce phase (≤ N CTAs of work, where N is the outer batch dimension), fusing the reduce into the split kernel via the **last-arrival atomic pattern** saves a full kernel launch (~5 µs of host driver overhead on Modal B200). Each split CTA does `__threadfence(); int prev = atomicAdd(&counter[t], 1)` after writing its scratch; the CTA observing `prev == S - 1` is the last to arrive for token `t` and inlines the reduce work itself. No `cudaLaunchCooperativeKernel` required — regular grid launch with a pre-zeroed `int[N]` counter buffer suffices.

Cost: 1 `__threadfence()` (~10-20 cycles) + 1 `atomicAdd` (~20-30 cycles) per CTA. Win condition: `launch_overhead > (fence + atomic) × CTAs_per_SM`. For B200 with ~5 µs launch overhead, this is profitable for any N≥2 workload (the launch save is ~150x the per-CTA cost).

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A21 (large workloads −6%, small workloads regressed slightly from atomic overhead, net mean improvement ~1 µs per workload).

### GT-34: SMEM-cache paged sparse-index slice — bound by `min(kv_per_split, k_end - k_start)`

When loading a paged sparse-index slice into SMEM (e.g., `for (tid < kv_per_split) smem_idx[tid] = idx_t[k_start + tid]`), the actual slice length for the LAST split can be shorter than the per-CTA `kv_per_split` upper bound — specifically when `S` does not divide `TOPK` evenly (`kv_per_split = ceil(TOPK / S)`, last split has `TOPK - (S-1)*kv_per_split` elements). Loading by `kv_per_split` reads OOB on the next token's slice (or beyond the sparse_indices buffer for the last token entirely), corrupting smem_idx with cross-token data and causing crashes when an unlucky cross-token kv_idx feeds an OOB cp.async.bulk gather. Fix: use `min(kv_per_split, k_end - k_start)` for the load count, and sentinel-pad the rest with -1.

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A17 (first run failed RUNTIME_ERROR on N=6/7 workloads; corrected version A17b passed 23/23 with −5.4% mean).

### GT-35: Per-lane SMEM read with stride = 32 bytes triggers 8-way bank conflict on B200

For SMEM laid out row-major with bf16 elements, a per-lane read pattern where lane l reads `bytes [l*32, l*32+31]` (16 contiguous bf16 = 32 bytes per lane) produces an 8-way bank conflict: lanes 0,4,8,…,28 all hit banks 0-3 simultaneously. Per LDG.128 takes 8× the cycles vs the no-conflict ideal. Switching to an INTERLEAVED layout where lane l reads `__nv_bfloat162` at byte offset `128*j + 4*l` (8 LDG.32 per lane, j=0..7) makes each LDG.32 hit 32 distinct banks. The per-lane element count and total byte volume are identical; only the per-lane data ALLOCATION (which 16 elements lane l owns) changes, requiring matching layouts in Q load, output write, and cross-kernel scratch I/O.

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A18 (mean −9.4% on this kernel; layout matched in Q LDG, scratch_o STG, reduce-pass scratch_o LDG, and final output STG).

### GT-33: TMA bulk wins where per-thread cp.async loses (on B200)

For sparse-attention-style KV gather, replacing inline LDG.128 with `cp.async.bulk` (1D TMA, sm_90+) is a **clear net win** even when adding SMEM staging + mbarrier sync — A14 → −5.1% mean. The same kernel restructure done with `cp.async.cg` (per-thread 16-byte async copies) was a **clear net loss** — A13 → +8.5% mean. The differentiator: `cp.async.bulk` issues 1 PTX instruction per 1024-byte row (handled by the dedicated TMA unit), while `cp.async.cg` requires 64 per-thread instructions per row (all going through the LSU). Per CTA per outer iter, that's 8 TMA instructions vs 192 LDG/cp.async — 24× fewer. The TMA unit also runs **independently of the LSU**, freeing the load pipe for other work.

Practical rule: when prefetching small tiles (≤4 KB) with mbarrier sync, prefer `cp.async.bulk` over `cp.async.cg`. The SMEM-staging cost is the same; the issue cost differs by 24×.

Confirmed 2026-04-19 on Modal B200 via A13 (cp.async.cg, +8.5% mean regression) and A14 (cp.async.bulk + mbarrier, −5.1% mean win) on the same A12-based double-buffered design.

### GT-30: Sparse-attention MLA decode is fundamentally split-K

For N ≤ 8 query tokens with TOPK=2048 KV indices, a single CTA per token uses only N of 132 SMs and is wall-time bound by the per-CTA serial KV loop (latency-limited, not bandwidth- or compute-limited). FlashDecoding-style split-K (S CTAs per token, each handles `2048/S` indices, with a small reduction kernel combining the partials) is mandatory. Empirical sweet spot for this kernel: `S = min(132/N, 64)` — capping at 64 because at S=132/1=132 the per-CTA work shrinks below the launch+reduction break-even and N=1 regresses (A11 attempt). Below cap-32 the medium workloads (N=2..7) are still SM-starved (A9 → A10 went −13% mean by raising cap from 16 to 32 then −4.5% by raising to 64).

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempts A2 (split-K introduced, mean 1.417 ms → 0.170 ms = −88%), A9 (cap 16 → 32, mean −13%), A10 (cap 32 → 64, mean −4.5%), A11 (cap 64 → 132 regressed +1.6%).

### GT-31: Auto-vectorization of scratch fp32 I/O is unreliable; force int4

The CUDA compiler does not always merge a `#pragma unroll`'d 16-element fp32 store/load into 4 × `STG.128`/`LDG.128`. Forcing the cast (`reinterpret_cast<int4*>(buf)` and stride-4 access) gave a measurable mean-latency win (15% on this kernel's reduce-pass scratch read in attempt A7) when the equivalent unrolled 16 × `ST/LD.32` was the prior pattern. Apply explicitly to scratch I/O, output bf16 packed writes, and any per-lane fp32 register array touching global memory in unrolled loops.

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempts A7 (vectorize scratch_o read/write: −15.1% mean) and A8 (vectorize bf16 output write: −3.8% mean).

### GT-32: Per-warp register Q ≥ cooperative SMEM Q when each warp owns one head

When the kernel partitions one query head per warp, having each warp directly LDG its own head's Q into per-lane registers (via packed int4 + int loads) is at least as fast as the cooperative-SMEM-then-broadcast pattern, AND eliminates ~36 KB SMEM allocation, the post-Q-load `__syncthreads()`, and 16x cross-warp SMEM-read amplification. The cross-CTA L1 amplification (16 warps each LDGing the same 1 KB Q row) is paid in hot L1 cycles, not HBM BW, and is cheap.

Confirmed 2026-04-19 on Modal B200 via Phase 2 attempt A6 (mean −1.1% with split kernel SMEM dropping to 0 bytes — small win on its own, important enabler for high-occupancy variants if pursued later).

### GT-29: `lse[t] = rhs` in C++ libtorch Does Not Write to Storage

In C++ libtorch, `Tensor::operator[]` returns a view `Tensor` **by value**.
Then `returned[t] = rhs` calls `Tensor::operator=(const Tensor&)`, which
REBINDS the local handle to share storage with `rhs` — it does NOT copy
`rhs`'s data into the indexed location. The destination tensor is unchanged,
silently.

**Correct pattern for DPS output writes:** `dst[t].copy_(rhs)` or
`dst.index_put_({t}, rhs)`. The Python `dst[t] = rhs` idiom does NOT
translate directly.

**Wrong behavior:** a DPS output stays at its initialised value (e.g. `-inf`
for `lse` after `fill_(-inf)`), correctness check fails silently with large
`max_abs_error`. No runtime error, no fault — just wrong numbers.

Confirmed 2026-04-19 on Modal B200 via the Phase 1 naive kernel: the initial
`lse[t] = torch::logsumexp(...) * inv_log2;` left lse full of `-inf`. The
matching `output[t].copy_(...)` on the line above worked correctly because
`.copy_()` writes to the view. Changing to `lse[t].copy_(...)` fixed it.

### GT-44: NCU PC-sample count does not equal wall-time stall cost when the stalled instruction is parallel-overlapped across SMs

For kernels running at 1 CTA/SM with many CTAs in the grid, PC-sample warp
stall counts can be misleading as an optimization target. The sampler
assigns a "warp stalled here" sample every time its clock tick observes
*any* warp on *any* SM stalled at that PC — but the wall-time saved by
removing the instruction depends on whether the stall is on the
critical-path dependency chain of a single CTA or is parallel-overlapped
across SMs.

**Canonical failure signature:** a single instruction accumulates a large
percentage of samples for its stall category (e.g., 92% of `membar`
samples), removing it via a code change passes 23/23, yet wall-time
mean does not move.

**Correct reading:** rank candidates by the **critical-path fraction** of
the stall reason, not raw sample count. Ask: if I remove this stall on
ONE CTA, does the CTA's dependency chain shorten? If the stall is
issued in parallel with other SMs' useful work (e.g., a fence or
cross-CTA barrier that every CTA pays once), removing it frees no
critical-path cycles.

**Candidates that DO translate sample count to wall-time**:
- Per-iteration stalls in a serial inner loop (e.g., a LDG stall in
  the QK inner loop — every loop iter pays this, reducing it shortens
  the chain by that much times the iter count).
- Register-spill-induced local memory access stalls (serialize the
  single CTA's compute).
- SMEM bank conflicts on hot loads (serialize the single warp's read).

**Candidates that do NOT translate**:
- One-time per-CTA cost (fence, atomicAdd, kernel prologue/epilogue)
  that overlaps with other CTAs' concurrent execution.
- "Am I last" branch waiting on an atomic's return value (L2 RTT is
  a structural cost of last-arrival pattern; no code change in this
  single CTA shortens it).

Confirmed 2026-04-22 on Modal B200 via Phase 2 Attempt 26 (N1,
`cuda::atomic_ref::fetch_add(release)` replacement of
`__threadfence() + atomicAdd`). PC-sample attributed 24 of 26 membar
samples (92%) to the single ERRBAR instruction from `__threadfence()`.
Removing it passed 23/23 but produced 0% wall-time improvement (mean
0.044 → 0.044 ms, p95 0.060 → 0.059 ms = noise). The ERRBAR is one
instruction per CTA per launch (128 total for N=8), issued in
parallel across SMs; removing it frees no critical-path cycles.

**Confirmed a second time 2026-04-22** via Phase 2 Attempt 27 (N2,
bf16 scratch_o). PC-sample attributed ~30 of 46 `lg_throttle` samples
(~65%) to the 8× STG.E.64-per-warp scratch_o store sequence. Halving
the dtype (8 STG.E.64 → 8 STG.E.32 per warp) passed 23/23 but
produced 0% wall-time improvement (mean 0.044 → 0.044 ms) AND
degraded `abs_err` by ~4 orders of magnitude (1.4e-06 → 1.56e-02 on
the small workloads). The STG sequence is issued concurrently across
16 warps × 128 CTAs; the LSU queue depth absorbs the count, and no
single warp's exit waits on its own STGs' completion time.

**Generalization**: for the DSA sparse-attention regime (1 CTA/SM, 16
warps/CTA, 128-132 CTAs in the grid), stall samples on per-CTA
one-time-cost instructions (fences, atomicAdds, STGs issued once per
warp-iter) do NOT aggregate to wall time. The only stalls that do
translate 1:1 are those on a single CTA's critical dependency chain
— e.g., serial LDG chains in an inner loop, SMEM bank conflicts that
serialize a single warp's reads, register spills that block a single
warp's compute.

---

## Section 8 — Failure Pattern Signatures

Match symptom to class. Go directly to the first action — do not perform a
general diagnosis first.

| Symptom | Most likely class | First action |
|---|---|---|
| Wrong output on smallest workload (num_tokens=1, num_pages=1) | INDEX_DECODE or SENTINEL | Diff the kernel output against the Python `reference` on that workload. Check `-1` sentinel skip and `page_idx * PAGE_SIZE + offset` decoding. |
| Correct on small workloads (num_tokens<=2), wrong on large (num_tokens>=8) | INDEX_STRIDE or LSE_BASE | Check per-token `sparse_indices[t]` addressing and that `lse` is being divided by `ln(2)` (2-base, not natural). |
| COMPILE_ERROR on all 23 workloads, no error text in terminal | MODAL_IMAGE | Check Modal image uses nvidia/cuda devel base (GT-19). Check pack_solution.py binding patch (GT-20). |
| COMPILE_ERROR with binding=None in solution.json | PACK_SOLUTION | Apply GT-20 patch to pack_solution.py. Verify solution.json shows `binding: torch` after packing. |
| XID 13 on cp.async.bulk.tensor | TENSOR_MAP | Check all globalStrides / 16. BF16 KV caches are 16-byte aligned so TMA IS usable here (unlike the top-k FP8 project). If XID 13 still fires, see GT-6 for tensormap residency + modification flow. |
| XID 13 on tcgen05.mma | TMEM_ADDR | Check TMEM alloc lifecycle (GT-2, Tier 3 section). |
| XID 13 on kernel launch | SMEM | Check dynamic SMEM opt-in > 48KB. |
| Wrong scores, no fault, ~0-20% of reference | SMEM_LAYOUT | Layout fill logic wrong — data reaching MMA is incorrect. |
| Wrong scores, no fault, ~50-80% of reference | DESCRIPTOR | SBO/LBO stride mismatch. Run sbo_lbo_sweep immediately. |
| Wrong scores, no fault, ~100% magnitude but wrong positions | IDESC | transpose_A or transpose_B bit wrong in IDESC. |
| Wrong scores only after adding swizzle | GT-10 violation | TMA swizzle mode does not match SMEM descriptor mode (GT-10, Tier 3 section). |
| ptxas parse error near `:` | PTX_SYNTAX | tcgen05.fence has illegal suffix — bare form only (GT-7, Tier 3 section). |
| ptxas instruction not supported | PTX_SYNTAX | WGMMA on sm_100a — does not exist (GT-1, Tier 3 section). |
| ptxas "tcgen05.* not supported on .target 'sm_100'" | ARCH_TARGET | `TORCH_CUDA_ARCH_LIST` unset or set to `10.0`. Set to `10.0a` per GT-27. |
| Kernel hangs, no XID, no output | BARRIER | Missing `__syncthreads()` or infinite loop in softmax / reduction / sort. |
| Numerically correct on small inputs, wrong on large | SMEM_OVERFLOW | SMEM buffer overflow — padding calculation wrong. |
| Kernel hangs at tcgen05.ld, wrong results on lane-gated warps | TMEM_LD | Remove `if(lane<16)` guard before tcgen05.ld — all 32 lanes must issue (GT-17, Tier 3 section). |
| RUNTIME_ERROR: invalid resource handle | HARDWARE_MISMATCH | Kernel compiled for sm_100a running on non-B200 GPU. Always test on Modal B200, never local. |
| Attention output NaN / Inf | SOFTMAX_STABILITY | Check max-subtract in softmax, log-sum-exp pattern, or underflow on masked rows. |
| Attention output correct shape but magnitude ~sqrt(d) off | SCALE_FACTOR | Missing `1/sqrt(head_dim)` scale on QK before softmax. |

(Last two rows are sparse-attention-specific; remove if the kernel does not
compute softmax.)

---

## Section 9 — pack_solution.py Contract

**The agent edits exactly one file: `solution/cuda/kernel.cu`.**

`run_modal.py` calls `pack_solution.py` which reads `config.toml` to determine
the entry point function name and language, then packages `kernel.cu` into a
`Solution` object for submission to Modal.

**Rules that must hold at all times:**

1. `config.toml` field `entry_point` must exactly match the function name in
   `kernel.cu`. If this function is renamed during optimization, `config.toml`
   must be updated in the same step — not after.

2. `config.toml` field `definition` must remain `<definition_name>`. Never
   change this.

3. `pack_solution.py` must have the GT-20 binding patch applied. Without it,
   `binding` defaults to tvm-ffi and compilation fails. Verify by running
   `python scripts/pack_solution.py` and checking that `solution.json` shows
   `"binding": "torch"`.

4. If additional `.cu` or `.h` files are added during optimization, verify
   `pack_solution.py` picks them up by inspecting what it packs before
   running `run_modal.py`. A mismatch causes silent wrong-function
   compilation — the symptom looks like a correctness failure but the actual
   kernel never ran.

5. Expected `config.toml` template (update name/definition/entry_point):

```toml
[solution]
name = "<author>"
definition = "<definition_name>"
author = "<author>"

[build]
language = "cuda"
entry_point = "kernel.cu::<launch_function_name>"
binding = "torch"
destination_passing_style = true
```

---

## Section 10 — Phase 2: Optimization Loop

Phase 2 begins after Phase 1 locks `checkpoints/kernel_naive.cu`.

### Per-optimization cycle structure

```
1. Scope declaration
2. Pre-implementation analysis (tiered — see below)
3. Implement in kernel.cu
4. Update config.toml if entry_point changed
5. modal run scripts/run_modal.py
6. Read output
   -> PASS 23/23: keep change, record timing vs GT-BASELINE, apply GT Update Rule if new fact found
   -> FAIL: revert kernel.cu to checkpoint immediately, write GT entry for what broke
7. Proceed to next optimization
```

**Scope declaration (mandatory before every optimization):**
Write three sentences: (a) what this optimization changes, (b) what it
explicitly does not change, (c) which checkpoint is locked for revert.

**One optimization at a time.** Never combine two independent changes in one
test cycle. If two changes are needed together for correctness, that counts
as one optimization — document why they are coupled.

**Revert command:**
```bash
cp ~/CUDAExperiments/sparse_attention/checkpoints/kernel_naive.cu \
   ~/CUDAExperiments/sparse_attention/flashinfer-bench-starter-kit/solution/cuda/kernel.cu
```

### Tier 1 — Free Wins (Sonnet, lightweight checklist)

Applies to: coalescing improvements, vectorized loads, warp utilization, dead
code removal, redundant barriers, compiler hints, collapsing host-side batch
loops into batched launches.

No full D1-D4 required. Before implementing, answer these three questions in
writing:

- Does this change affect the memory access stride of any tensor? If yes ->
  run D3 scoped to the affected access pattern before implementing.
- Does this change affect any barrier placement? If yes -> run D3 scoped to
  the barrier before implementing.
- Does this change touch the sparse-index gather path (how `sparse_indices`
  maps to KV-cache rows)? If yes -> run D3 scoped to the gather before
  implementing.

If all three answers are no -> implement directly.

**Tier 1 candidate discovery:** after Phase 1 locks, profile the naive kernel
(or just inspect its structure) and enumerate candidates. Typical starting
points for a kernel built on this starter kit:

1. Host-side per-token loop serializing kernel launches — collapse to batched
   launches or a single fused kernel across `num_tokens`.
2. Redundant bf16→fp32 casts of the whole KV cache per call — load bf16
   directly into fp32 accumulators inside the kernel instead of materialising
   a cast copy.
3. Scratch allocations made fresh per launch — evaluate `torch::empty`
   hoisting (torch caching allocator works well; do NOT substitute raw
   cudaMalloc for torch-level scratch, the top-k project confirmed this
   regresses).
4. `masked_select` + `index_select` on every token — replace with a custom
   gather kernel that handles the `-1` sentinel inline (avoids two intermediate
   allocations per token).

Do not assume these are wins. Test each; record as GTs.

### Tier 2 — Algorithm Class (Sonnet, D3 only)

Applies to: softmax algorithm (online vs two-pass), QK computation fusion,
softmax+V fusion, per-CTA buffer architecture, attention tiling strategy.

**Regime analysis — compute this once Phase 1 is locked:**

- Arithmetic intensity on the largest workload.
- B200 ridge point at the dtype used for the dominant matmul.
- Whether the kernel is above or below ridge.
- HBM utilization at peak (loads in bytes / kernel time).

Do NOT import top-k's roofline answer (GT-8 in that project). Sparse
attention has different AI because it does TWO matmuls (QK and SV) per
selected token rather than one, plus a softmax in between. Re-derive.

**Forbidden architecture unless proven necessary:** writing the full QK^T
matrix to global memory before softmax. A fused online-softmax pattern
(FlashAttention-style: running max + running sum + running output, all in
registers/SMEM) avoids this entirely. Evaluate online vs two-pass on D3
before committing.

Run D3 scoped to the attention Gate molecule change. Make the online vs
two-pass decision explicit.

### Tier 3 — Hardware-Specific (Opus, full D1-D4)

Applies to: tcgen05 MMA for bf16 Q@K^T and attention-weighted V (`kind::f16`,
not `kind::f8f6f4` — see GT-2), TMEM accumulator lifecycle, async pipelining
between QK and SV, warp specialization (producer/consumer), TMA loads for
ckv/kpe caches (valid here because bf16 strides are 16-byte aligned).

**Before starting Tier 3:**
1. Load `ptx_isa_sections/MANIFEST.md` — identifies which ISA files to load
   for each pipeline step. Load ONLY the sections needed for the current step.
2. Load `gau-nernst_reference.h` — confirmed working B200 PTX wrappers.
   Cross-check all inline PTX against this.
3. Read Section 11 (Tier 3 GT entries) in full.
4. Run full D1->V1->D2->V2->D3->V3->D4->V4 scoped to the hardware change.

Model routing for Tier 3: D3, V3, D4, V4, Layer 4 implementation, and
Diagnostic Agent -> **Opus**. All other steps -> Sonnet.

### Tier 4 — Micro-Optimizations (Sonnet, no D1-D4)

Applies to: register pressure tuning, instruction scheduling, shared memory
padding, loop unrolling decisions, SMEM bank-conflict elimination.

Apply one micro-optimization at a time. Test after each. No derivation
pipeline needed — these are local code changes with no structural impact.

Note: the top-k project found that bank-conflict analysis on WRITES was as
important as on reads (a write pattern that looked 2-way actually was 16-way
across the 32 active lanes). Check both directions on any SMEM stride choice.

### Optimization re-evaluation rule (mandatory after each tier)

After completing a tier, profile the kernel and identify the dominant cost
center. Before proceeding to the next tier, ask: "Does the dominant cost
center correspond to a molecule whose algorithm class might be wrong?" If
yes, re-visit that tier's decisions before moving on.

---

## Section 11 — Tier 3 Only: Hardware GT Entries

**DO NOT READ THIS SECTION until Tier 3 is the active optimization target.**
These entries are irrelevant noise during Tiers 1-2. Reading them early risks
premature use of tcgen05 instructions.

Entries tagged **[IMPORTED]** were confirmed on the top-k indexer kernel.
Those that encode MMA shape parameters (M/N/K, tcgen05.ld lane distribution)
may need re-confirmation if this kernel uses a different tile shape — run the
sbo_lbo_sweep probe on the new shape before relying on them.

---

### [IMPORTED] GT-1: WGMMA Does Not Exist on sm_100a

`wgmma.mma_async`, `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`
are Hopper-only (sm_90a) and DO NOT COMPILE on sm_100a. WGMMA fragment
indexing formulas must never appear in tcgen05 kernels.

### [IMPORTED] GT-2: tcgen05 Instruction Family

- **Compute:** `tcgen05.mma.cta_group::1.kind::<K>` where `<K>` depends on
  operand dtype. For this kernel's bf16 Q/K/V, use `kind::f16` (covers
  bf16+bf16→fp32 MMA). The top-k project used `kind::f8f6f4` — do NOT reuse
  that `kind` here. Look up valid shapes in PTX ISA Table 41 (S9.7.16.2.1),
  idesc in Table 44 (S9.7.16.4.2) at implementation time.
- **Accumulator:** TMEM (dedicated per-SM memory), NOT registers. Transparent
  row-major layout — no fragment indexing.
- **Issuing:** Single elected thread via `elect_one_sync()`.
- **TMEM lifecycle:**
  1. `tcgen05.alloc` (full warp, `warp_id == 1`)
  2. MMA (elected thread issues)
  3. `tcgen05.commit` (elected thread, immediately after MMA)
  4. `mbarrier.try_wait` (consumer threads — warps 0 and 1 — spin until MMA
     hardware done)
  5. `tcgen05.fence::after_thread_sync` (consumer threads, after mbarrier wait)
  6. `tcgen05.ld` -> `tcgen05.wait::ld`
  7. `tcgen05.dealloc` before kernel exit (all paths)
- **TMEM quarters:** For M=64, warps 0-1 hold rows 0-63. For sparse attention,
  if the MMA tile is different (e.g. M=128 for 128 query heads fused), the
  quartering changes — re-derive from the ISA.

### [IMPORTED] GT-3: tcgen05.wait::mma Does Not Exist

PTX ISA S9.7.16.8.5: `.wait_operation = { .wait::ld, .wait::st }` only.
MMA->ld ordering uses `commit -> mbarrier_wait -> fence::after_thread_sync`
(GT-2 steps 3-5). `tcgen05.fence::before/after_thread_sync` around
`__syncthreads()` is for cross-thread ordering of pipelined instruction pairs
only — it does NOT wait for MMA hardware pipeline completion.

### [IMPORTED] GT-4: H-Reduction / Softmax-Reduction Pattern (re-confirm for sparse attention)

For the top-k kernel, cross-head reduction used:
`fence+sync+fence -> TMEM load -> per-slab scale in registers -> intra-warp
__shfl_xor_sync across 32 heads -> two-warp cross-SMEM merge`. For sparse
attention, the equivalent reduction is softmax row-max and row-sum. The same
primitive order holds (TMEM load -> register compute -> warp shuffle -> SMEM
merge if cross-warp), but the specific shuffle topology depends on how query
heads are tiled across warps. Re-derive from the chosen MMA shape.

### [IMPORTED] GT-5: Scale Application

Per-slab MMA with `enable_input_d=false` (fresh accumulator) -> fence+sync+
fence -> `tcgen05.ld` + `wait::ld` -> scale multiply in registers -> accumulate
into running register total. Never inside MMA pipeline. For this bf16 kernel,
the only post-load scale is `sm_scale` (the `1/sqrt(head_dim_qk + head_dim_kpe)`
attention factor) — apply it in registers post-load, not via descriptor
manipulation. No FP8 dequant scale applies here.

### [IMPORTED] GT-6: TMA Tensor Map Constraints

- **Residency (S9.7.9.27.1.2):** tensorMap for `cp.async.bulk.tensor` MUST be
  in `.param`/`.const`/`.global` — NOT `.shared`.
- **Modification flow:**
  1. Lane 0: memcpy template -> SMEM, `tensormap.replace` in SMEM
  2. `__syncwarp()` for visibility
  3. All 32 lanes: `tensormap.cp_fenceproxy` SMEM -> global (`.sync.aligned`)
  4. Thread 0: `fence.proxy.tensormap::generic.acquire.cta` on global copy
  5. Thread 0: TMA using global copy's address
- **tensormap.replace syntax:** requires BOTH `.b1024` AND `.b64` qualifiers.
- mbarrier expect-tx: call once with total transfer size.

**Applicability note for this kernel:** unlike the top-k FP8 project (whose
KV cache had a 132-byte stride that ruled out TMA), this kernel's bf16 ckv
(1024 B/row) and kpe (128 B/row) caches are 16-byte aligned on every
dimension, so TMA is a valid option for every tensor in the pipeline. Still
follow the residency + modification flow above.

### [IMPORTED] GT-7: PTX Syntax Pitfalls

- **tcgen05.fence::before_thread_sync** — bare form only. Do NOT append
  `::1.cta.sync.aligned`.
- **Avoid `<cuda/ptx>` header entirely** — `cuda::ptx::` wrappers have
  `__CUDA_ARCH__` guard issues on sm_100a. Use raw `asm volatile(...)`.
- **`<cuda/barrier>` is fine** to include.

### [IMPORTED] GT-10: TMA/MMA Swizzle Consistency

If TMA loads data with `CU_TENSOR_MAP_SWIZZLE_128B`, bytes are rearranged in
SMEM. The MMA SMEM descriptor built by `make_smem_desc`
(bits[46:48]=0b001) assumes LINEAR layout. These are INCOMPATIBLE — MMA reads
scrambled data, produces wrong results silently (no fault). Fix: use
`CU_TENSOR_MAP_SWIZZLE_NONE` for all TMA loads feeding into `tcgen05.mma` via
linear `make_smem_desc`. Confirmed 2026-04-12 on Modal B200 (top-k project).

### [IMPORTED] GT-11: tcgen05.mma SMEM Descriptor — SBO/LBO for M=64/N=64/K=32 (re-confirm for sparse attention shape)

For `tcgen05.mma.cta_group::1.kind::f8f6f4`, M=64/N=64/K=32, K-major,
SWIZZLE_NONE, sm_100a, 8xT core-tile SMEM layout:

    SBO field bits[32:45] = 16   (SBO = 256 bytes)
    LBO field bits[16:29] = 8    (LBO = 128 bytes)

Confirmed 2026-04-15 by host-side sweep on Modal B200 (top-k project:
`diagnostic_tests/sbo_lbo_sweep.cu`).

**If this kernel uses a different MMA shape (e.g. M=128 for fused
multi-query-head, or different K tile), re-run sbo_lbo_sweep on the new
shape. Do not guess the SBO/LBO values — the top-k project burned multiple
runs on incorrect SBO guesses before the sweep confirmed them.**

SMEM layout fill formula (unchanged across M/N/K, 8xT core-tile):
`addr = m_grp * SBO_bytes + k_tile * LBO_bytes + m_in_grp * 16 + k_in_t`
where `m_grp = m/8, m_in_grp = m%8, k_tile = k/16, k_in_t = k%16`.

### [IMPORTED] GT-12: tcgen05.ld Lane Distribution for M=64 (re-confirm for sparse attention shape)

For `kind::f8f6f4` M=64 cta_group::1, the 64 M-rows are distributed 16 per
warp across all 4 warps. Only lanes 0..15 of each warp hold live MMA output
data (lanes 16..31 read zeros). Canonical tcgen05.ld address for accumulator
row `r`: `(warp_id * 32 + lane_id) << 16` where lane L < 16 of warp W maps to
M-row `(W * 16 + L)`. Upper lanes must be masked AFTER the load (not before —
see GT-17).

For a different M (e.g. M=128), the lane/warp distribution changes. Re-derive
from ISA Table in S9.7.16.8 before writing ld code.

### [IMPORTED] GT-14: Slab Loop Unrolling Breaks MMA Ordering

`#pragma unroll` on the K-slab loop (the inner per-accumulator-slab MMA
dispatch) breaks MMA->ld ordering and produces wrong results. Do not unroll
the slab loop. Confirmed 2026-04-15 on Modal B200 (top-k project).

### [IMPORTED] GT-15: IDESC Transpose B Bit with K-Major 8xT Layout

For the K-major 8xT core-tile SMEM layout (GT-11), the IDESC transpose_B bit
must be 0, not 1. `transpose_B=1` produces the canonical 50-80%
descriptor-mismatch signature. Confirmed 2026-04-15 on Modal B200 (top-k
project).

Sparse attention will do BOTH QK^T and PV (or SV). QK^T needs K transposed
relative to Q's orientation; PV does not. Work out the IDESC transpose bit
for EACH matmul independently using the layout you actually chose for Q, K,
P/S, and V — do not assume the top-k answer applies to both.

### [IMPORTED] GT-16: Persistent Global Scratch for Multi-CTA Kernels

For multi-CTA kernels requiring inter-kernel scratch buffers, use a static
device pointer (allocated once, never freed in-process). Per-launch
`cudaMalloc`/`cudaFree` eliminates parallelism gains from multi-CTA
partitioning. Confirmed 2026-04-15 on Modal B200 (top-k project).

**Caveat from the top-k project:** this does NOT apply to torch-managed
per-launch scratch (`torch::empty`). Torch's caching allocator is fast; raw
`cudaMalloc` is only a win for true inter-kernel scratch where the torch
allocator's size-bucketing would churn. If the scratch is only live within a
single kernel launch, use `torch::empty`.

### [IMPORTED] GT-17: tcgen05.ld Requires Full Warp Participation

`tcgen05.ld.sync.aligned` requires ALL 32 lanes of the issuing warp to
execute the instruction. Gating with `if(lane_id < 16)` before tcgen05.ld
causes hang or wrong results. Correct pattern: all 32 lanes issue
tcgen05.ld; downstream computation is masked by `if(lane_id < 16)` AFTER the
load returns. Confirmed 2026-04-17 on Modal B200 (top-k project).

### [IMPORTED] GT-18: tcgen05.commit Correct Form for CUDA 12.8

`tcgen05.commit.cta_group::1.mbarrier::complete_tx::bytes` does NOT compile
under CUDA 12.8. Correct form:

    tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [addr];

Always use `.mbarrier::arrive::one.shared::cluster.b64`. Confirmed 2026-04-17
on Modal B200 CUDA 12.8 (top-k project).

### GT-39: tcgen05 BF16 MMA regression regime for MLA-decode-shaped kernels

`tcgen05.mma.kind::f16` is expected to **regress** wall-time on kernels whose
regime matches this signature:

1. Small effective M (number of query heads per contracted MMA, M ∈ {8, 16, 32}).
2. Split-K structure with many CTAs (~100+) each contributing a small slice
   of the K axis.
3. Arithmetic intensity well below the bf16 tensor-core ridge (`AI < 700
   FLOP/byte` on B200).
4. Memory latency already hidden by TMA + multi-buffer prefetch (2+ buffers).

Why regression is structurally expected for this signature:

- ISA Table 41: `kind::f16` cta_group::1 dense requires M ∈ {64, 128}. For
  M_real ∈ {8, 16, 32}, zero-padding to M=64 wastes 50-88% of MMA compute
  AND the TMEM accumulator. Token-packing to raise M instead destroys split-K
  SM utilization for small-batch decode workloads (GT-30 violation).
- Fixed per-MMA-group lifecycle (`tcgen05.alloc` / `commit` / `mbar_wait` /
  `ld` / `dealloc`, ~200-300 cycles per outer iter) matches or exceeds the
  scalar compute it replaces when AI << tensor-core ridge.
- The MMA path also requires reorganizing SMEM (per-lane → 8×T core-tile),
  re-running `sbo_lbo_sweep` for `kind::f16` K=16 (prior GT-11 covers
  `kind::f8f6f4` K=32 only), and re-deriving lane distribution for
  `tcgen05.ld` M=64 `kind::f16` (GT-12 covers `kind::f8f6f4` M=64 only) —
  high implementation risk for a path whose D3 predicts regression.

**Practical rule:** for sparse-attention, MLA-decode, and similarly-shaped
kernels with this signature, skip T3-5/6/7 (tcgen05 MMA) directly at D3.
The conclusion is deterministic from the ISA + roofline, not an accident
of one kernel's implementation. A successful tensor-core path would require
either (a) lifting M above 64 via algorithm restructure (e.g., batched
multi-token MMA while preserving split-K — non-trivial and may still
regress), or (b) changing the KV access pattern so AI can cross the
tensor-core ridge (e.g., cache-blocking KV, which conflicts with the
random-gather spec).

Confirmed 2026-04-19 on sparse-attention kernel at A23 baseline (0.044 ms
mean, HBM-gather-bounded) via D1-D3 derivation pass
(`d1_d4_v1_v4_t3_5_6_7.md`). Supporting evidence: prior T3 attempts A15
(TMA one-shot Q), A24 (SW-pipelined reduce), A25 (4-buffer prefetch) all
regressed because the kernel is at a latency floor where any added fixed
per-CTA overhead dominates.

**2026-04-19 empirical addendum**: a full 16-warp `kind::f16 M=64 N=16 K=16`
tcgen05 kernel was built and validated on this kernel (`checkpoints/tg_kernel.cu`,
passes 23/23). Measured mean **0.084 ms — 1.9× slower than A23**. This
confirms the regression hypothesis end-to-end. See optimization_log.md
Attempt T3-567d for full debug trajectory. The working prototype is
preserved for future reference.

### GT-40: tcgen05 `kind::f16` Layout Values Confirmed for M=64 N=16 K=16

For `tcgen05.mma.cta_group::1.kind::f16`, M=64/N=16/K=16, K-major 8×T
SMEM, no swizzle, sm_100a (B200):

- **SBO field bits[32:45] = 16** (SBO = 256 bytes, same as GT-11 for f8f6f4).
- **LBO field bits[16:29] = 8**  (LBO = 128 bytes, same as GT-11 for f8f6f4).

The 8×T tile structure is BYTE-LEVEL: 8 M-rows × 16 K-bytes per K-tile =
128 bytes per tile. This is dtype-agnostic: f8f6f4 K=32 (32 bytes per MMA
K-step) and f16 K=16 (16 bf16 = 32 bytes per MMA K-step) have IDENTICAL
byte layout. SBO/LBO values therefore carry over unchanged from GT-11.

**Multi-K-step storage** (new finding — GT-11 only confirmed single-K-step
layouts for the top-k project):

When `K_total > K_per_MMA_step` (e.g., attention with `head_dim=576` bf16
requires 36 MMA K-steps per outer iter), store consecutive K-steps
CONSECUTIVELY in SMEM:

- **Q (M=64, 8 m_grps)**: per-K-step SMEM size = 8 m_grps × 256 B = **2048 B**.
  Descriptor start-addr advance per K-step = 2048 B = encoded **128**.
  Fill formula: `addr = k_step × 2048 + (m/8) × 256 + ((kb%32)/16) × 128 + (m%8) × 16 + kb%16`
  where `kb` is byte offset in [0, K_total_bytes).

- **KV (N=16, 2 n_grps)**: per-K-step SMEM size = 2 n_grps × 256 B = **512 B**.
  Descriptor start-addr advance per K-step = 512 B = encoded **32**.
  Fill formula: `addr = k_step × 512 + (n/8) × 256 + ((kb%32)/16) × 128 + (n%8) × 16 + kb%16`.

The per-K-step byte stride equals `(M_or_N/8) × 256` where M_or_N/8 is
the number of row-groups. The DESCRIPTOR's SBO remains 256 (per-K-step
m_grp/n_grp stride) regardless — only the start-address advance differs.

### GT-41: tcgen05 `kind::f16` Lane Distribution for M=64 cta_group::1

`tcgen05.ld.32x32b.x16` after `tcgen05.mma.cta_group::1.kind::f16` M=64:
warp 0 lane L<16 holds M-row L's data (16 fp32 covers all N-cols for
N=16). Same pattern as GT-12's f8f6f4 M=64. For N=16, the 16 fp32 per
lane correspond directly to the 16 N-cols of M-row L.

**Lanes 16..31 of each warp** read zeros per the padding-to-32-lanes
convention (carry over from GT-12).

**Warps 1-3** in the 4-warp group hold M-rows 16..63 (our zero-padded M
region). Their data can be ignored downstream.

Confirmed 2026-04-19 via T3-567d passing 23/23 on Modal B200 with this
interpretation.

### GT-42: tcgen05 + TMA Layout Incompatibility Requires 3D Tensor Map

1D `cp.async.bulk` writes linear (row-major) SMEM. `tcgen05.mma`
descriptors require the 8×T core-tile layout (GT-11). These layouts are
INCOMPATIBLE — TMA writes cannot be directly fed to MMA descriptors
without reformatting.

Two options exist:
1. **Cooperative HBM→SMEM copy** (thread-level LDG.128 into the 8×T
   layout). Simpler but loses TMA's LSU-independent dispatch (GT-33).
   Used in T3-567d; contributed to its 1.9× regression vs A23.
2. **3D tensor map with swizzle** per gau-nernst `init_tmap_3d_128B`:
   shape `(K/64, M, 64)` with `CU_TENSOR_MAP_SWIZZLE_128B`. The MMA
   descriptor must then use `swizzle=2` (128B) to match (GT-10).

For sparse-attention with sparse-index-gather KV (each row fetched from
a different sparse location), option 2 is harder because standard
`cp.async.bulk.tensor` assumes contiguous source strides. The per-row
sparse indirection forces one TMA call per KV row regardless of layout.

Confirmed 2026-04-19 via T3-567d. Future MMA kernels needing TMA for the
A/B operands must use option 2; kernels with sparse gather likely need
option 1.

---

## Section 12 — Testing Protocol

**The single canonical test command:**

```bash
cd ~/CUDAExperiments/sparse_attention/flashinfer-bench-starter-kit
source ~/CUDAExperiments/sparse_attention/fi-bench-env/bin/activate
modal run scripts/run_modal.py
```

**Reading the output:**
- `PASS: all workloads passed.` -> 23/23 passed
- `FAIL: N workloads failed` -> N workloads failed, per-workload detail above it
- `PASSED: X/23` in the summary header -> X passed out of 23

**Retry limit:** 3 retries per attempt before escalating to Diagnostic Agent.

**Timing output:** The script prints mean, p50, p95, min, max latency across
all workloads. Record these after every passing optimization and compare
against GT-BASELINE.

**Local smoke test (compile check only, NOT a correctness test):**
```bash
cd ~/CUDAExperiments/sparse_attention/flashinfer-bench-starter-kit
export FIB_DATASET_PATH=~/CUDAExperiments/sparse_attention/mlsys26-contest
python scripts/run_local.py
```
Local gives RUNTIME_ERROR (RTX 4060 cannot run sm_100a code) but will give
COMPILE_ERROR if the kernel fails to compile. Use local as a fast pre-check
before spending Modal compute.

**Invariants that must always hold:**
- `checkpoints/kernel_naive.cu` is never modified
- `config.toml` definition is never changed
- `config.toml` entry_point always matches the function name in `kernel.cu`
- Tests always run on Modal B200 for final validation
- Local run used only as a compile check

**Tolerance note (imported from top-k project):** the harness rel_err
tolerance sits around 1e-2. Precision-reducing optimizations (BF16/FP16
intermediates in place of FP32) may PASS on some workloads and FAIL on
others — expect binary outcomes near that boundary. For softmax in
particular, the accumulated error from exp() and running-sum can cross the
tolerance faster than a plain matmul would. Be conservative with precision
reduction in the softmax path specifically.

---

## Section 13 — 3-Stage Diagnostic Escalation

Trigger when: an optimization is BLOCKED after 3 retries.

### Stage 1: Diagnostic Agent

Invoke a fresh **claude-opus** agent. Provide:
- Last 3 failure outputs from `modal run scripts/run_modal.py`
- Current `kernel.cu`
- The failure pattern from Section 8 that best matches the symptom
- For Tier 3 failures: `d3.md`, `d4.md`, relevant PTX ISA sections,
  `gau-nernst_reference.h`

The agent checks the Failure Pattern Signatures table first, classifies the
failure, then produces `diagnosis.md` with: FAILURE_CLASS, ROOT_CAUSE,
EVIDENCE, AFFECTED_DECISIONS, PROPOSED_FIX, NEW_GT_CANDIDATE.

If FAILURE_CLASS = DESCRIPTOR, the agent runs
`modal run run_diagnostic.py` (sbo_lbo_sweep) before proposing any fix.

Apply the proposed fix. Retry 3 times.

### Stage 2: Bisection Agent

If Stage 1 fix fails, invoke a **claude-sonnet** bisection agent that:
1. Starts from `checkpoints/kernel_naive.cu` (last known passing state)
2. Adds one element at a time from the failing kernel
3. Tests each addition via `modal run scripts/run_modal.py`
4. Produces `minimal_repro.cu` (smallest failing file) and `minimal_pass.cu`
5. Reports which specific element causes the failure

### Stage 3: Human Review

If Stage 2 cannot isolate the failure, escalate to human with:
- Bisection report
- `minimal_repro.cu` and `minimal_pass.cu`
- Recommended spec changes with options
- Which D-step files need updating

### GT Update Rule on Resolution

Every successful diagnosis that reveals a new hardware constraint becomes a
new GT entry in Section 7 (always-active) or Section 11 (Tier 3 only). Probe
outputs that confirm hardware-specific behavior are recorded with date and
probe name. This is mandatory — not optional.

---

## Section 14 — Model Routing Table

| Work | Model | Reason |
|---|---|---|
| Tier 1 optimization | Sonnet | Checklist-guided, mechanical |
| Tier 2 D3 (algorithm selection: online vs two-pass softmax, tile shape) | Sonnet | Regime analysis is mechanical once AI is computed |
| Tier 3 D3, V3 | **Opus** | Hardware binding — unconstrained reasoning |
| Tier 3 D4, V4 | **Opus** | Architecture spec — must catch D3 errors |
| Tier 3 Layer 4 (IDESC/SMEM descriptors) | **Opus** | Bit-field encoding, TMEM addressing |
| Tier 3 Layers 0-3, 5-7 | Sonnet | Boilerplate and algorithmic code |
| Tier 4 micro-optimizations | Sonnet | Local changes, no structural reasoning |
| Stage 1 Diagnostic Agent | **Opus** | Must reason outside GT rules |
| Stage 2 Bisection Agent | Sonnet | Mechanical: add one thing, test, repeat |

---

## Section 15 — The Mandate

- **"Could", "optionally", "consider", "might" are forbidden.** Every
  decision is committed. Use "BOUND:", "REQUIRED:", "CONFIRMED:".
- **No tcgen05 instructions before Tier 3 is the active optimization target.**
  Writing tcgen05 code in Tiers 1-2 is an error.
- **The checkpoint file is never modified after Phase 1 locks it.** Not for
  debugging. Not for "just trying something." Never.
- **One optimization at a time.** Never combine two independent changes in
  one test cycle. If a test fails, the cause must be unambiguous.
- **No code in D1-D4 steps.** Derivation steps produce specification
  documents. Implementation begins only after V4 passes.
- **Save intermediate work immediately.** After each D or V step, save the
  output file. After each passing test, record the timing numbers vs
  GT-BASELINE.
- **Never write PTX from memory.** Stop and look it up from the PTX ISA
  sections first.
- **Local run is a compile check only.** RUNTIME_ERROR locally is expected
  and harmless. Only Modal B200 results determine pass/fail.
- **Imported GTs are provisional until re-confirmed.** Any [IMPORTED] GT
  cited in a decision must either (a) have a "Re-confirmed … on sparse
  attention" line appended, or (b) be re-verified as part of the current
  optimization cycle. Citing an unconfirmed imported GT is the same class of
  error as writing PTX from memory.
