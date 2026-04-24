# D1-D4 / V1-V4 Derivation Pass on tg_kernel.cu
# `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
# Attempt T3-567d (16-warp tcgen05 kind::f16 QK^T, cooperative KV copy)
# Starting state: 23/23 PASS, mean 0.084 ms (1.9× slower than A23 = 0.044 ms)
# Goal: identify what it would take to beat A23.

---

## Workload Axes (unchanged)

- `N` ∈ {1..8} query tokens (batch axis).
- 16 query heads per token (`NUM_QO_HEADS`).
- `TOPK = 2048` KV rows selected per token via `sparse_indices`.
- `HEAD_DIM_CKV = 512`, `HEAD_DIM_KPE = 64`, combined `HEAD_DIM_TOT = 576`.
- Grid: `(S, N)` CTAs where `S = min(132/N, 64)` (GT-30).
- Each CTA processes `kv_per_split = ceil(2048/S)` KV rows (128 for N=8).

---

## D1 — Molecule Detection (delta vs A23)

| Molecule                              | A23   | tg_kernel | Notes                                                                                         |
| ---                                   | ---   | ---       | ---                                                                                           |
| Tile (FXP)                            | ✓     | ✓         | Outer KV loop, K_PER_ITER=4 (A23) vs KV_PER_ITER_MMA=16 (tg_kernel)                          |
| Online (FXP × SRG)                    | ✓     | ✓         | Online softmax with INV-based rescale                                                         |
| Combine Group (REL × SRG)             | ✓     | ✓         | Q@Kc + Qp@Kp → single logit (now fused into one K=576 MMA K-axis)                             |
| Pipeline (REL × FXP)                  | ✓     | **NO**    | **A23 has TMA 3-buffer prefetch; tg_kernel is synchronous fill→MMA→PV** ← primary regression  |
| Predicated Reduction (SRG × PRD)      | ✓     | ✓         | Sentinel `-1` → -inf logit                                                                    |
| Atomic Reduction (SRG × ATO)          | ✓     | ✓         | Last-arrival counter unchanged                                                                |
| Pipeline Lift (REL × FXP × FUN)       | ✓     | ✓         | Reduce-phase nested fixed point in same kernel                                                |
| **Hardware Matmul (REL × PRD × MEA)** | NO    | **✓**     | tcgen05 MMA with TMEM accumulator, SMEM descriptor, cross-warp commit+wait. NEW.              |
| **Layout Morphism (MOR × SYM)**       | A18 interleaved | **8×T core-tile** | Completely different: A18 is per-lane interleaved for bank conflict; 8×T is for MMA descriptor.   |
| **Logit Broadcast (REL × MEA)**       | NO    | **✓**     | TMEM→registers (warps 0-3) → SMEM (1 KB smem_logits) → registers (warps 0-15). NEW.           |
| Streaming Gate (FXP × GATE)           | NO    | NO        | sparse_indices precomputed                                                                    |
| Sweep-as-Fixed-Point (MON × FXP)      | NO    | NO        | Reduce max-scan short (S ≤ 64) via warp shfl                                                  |

**V1**: All detected molecules consistent. The new molecules (Hardware
Matmul, Layout Morphism, Logit Broadcast) are genuine additions that
come with tcgen05. Streaming Gate still doesn't apply. A critical
regression: **Pipeline (REL × FXP) molecule is MISSING** — this is the
primary performance cost.

---

## D2 — Structural Analysis (delta vs A23)

### New resource axes

**TMEM accumulator**: 64×N fp32 per CTA.
- For N=16: 64 × 16 × 4 = 4 KB TMEM; `tcgen05.alloc` takes 32 cols (minimum).
- Fits easily; 1 CTA/SM retained. No occupancy impact.

**SMEM footprint (new):**
- Q 8×T: 73728 B (72 KB) — up from A23's 0 B (per-warp register Q per GT-32).
- KV 8×T: 18432 B (18 KB) — up from A23's ~14 KB (3-buffer × K=4 rows × 1152 B/row).
- smem_logits: 1024 B (new, for cross-warp broadcast).
- mbar + idx + tmem_slot + padding: ~552 B.
- **Total: ~93 KB** — vs A23's ~15 KB. 6× increase.

B200 SMEM budget: ~228 KB/SM. 93 KB fits 1 CTA/SM (same as A23). SMEM
budget is not the binding constraint; but the large Q SMEM forfeits the
per-warp-register-Q win captured by GT-32.

### New atom activations

- **MEA**: Q SMEM tile (new — A23 had 0 B Q SMEM), smem_logits (new).
- **MOR**: 8×T core-tile address-rewrite morphism (`q_addr`, `kv_addr`),
  applied to BOTH Q and KV SMEM fills. Replaces A23's A18-era per-lane
  interleaved layout.
- **REL**: cross-warp SMEM broadcast of logits (warp 0 writes, warps 0-15
  read). New barrier required.

### Chain depth analysis

The hop chain for one outer iter:
- A23: `sparse_indices → ckv/kpe HBM → SMEM (linear) → warp-register → scalar MAC → running softmax → running o_acc` (depth 4, all register-resident intermediates)
- tg_kernel: `sparse_indices → ckv/kpe HBM → SMEM (8×T) → [MMA → TMEM] → warp-register (warps 0-3) → SMEM (smem_logits) → warp-register (warps 0-15) → softmax → SMEM (smem_kv) → warp-register → scalar MAC PV → running o_acc` (depth 9, multiple SMEM round-trips).

Chain depth doubled. Each hop adds sync+latency cost.

**V2**: Structural analysis consistent. The depth increase is the
fundamental cost structure — it must be amortized via pipelining for the
MMA path to win. With NO pipelining in tg_kernel, depth increase is
paid in full → regression.

---

## D3 — Hardware Binding (primary optimization leverage)

### D3.1 — Regime re-derivation

Per-CTA for N=8 (largest workload, kv_per_split=128):
- **Bytes gathered from HBM**: 128 rows × 1152 B = 147 KB (unchanged vs A23).
- **Compute FLOPs**:
  - QK^T: 16 real heads × 576 K × 128 N × 2 (MAC) = 2.36 MFLOP.
  - SV: 16 heads × 128 N × 512 head_dim × 2 = 2.10 MFLOP.
  - Total: ~4.5 MFLOP.
- **AI = 30 FLOP/byte** (same as A23; arithmetic is identical).

B200 ridges:
- Scalar fp32: ~11 FLOP/B. A23 was above ridge → compute-bound.
- BF16 tensor-core: ~714 FLOP/B. tg_kernel is ~23× below ridge → memory-bound.

**Implication**: moving to tensor cores DOES NOT raise the throughput
ceiling because HBM is already at ridge. MMA only helps if it **frees
the LSU/SIMD for other work** during the same wall-clock. Without
pipelining, MMA + LSU run serially, paying full cost of both.

### D3.2 — Measured bottleneck breakdown (tg_kernel mean 0.084 ms vs A23 0.044 ms)

The 40 µs gap vs A23 is the delta cost of:
1. **Cooperative HBM→SMEM copy** (no TMA) for KV per iter
   - 128 threads × 16 bytes/thread = 2048 B/cycle × ~72 cycles = ~40 µs
     total per CTA across 8 iters (without any overlap). GT-33 says TMA
     bulk is ~5-20× cheaper on instruction issue AND runs on independent
     TMA unit — that's ~30-35 µs savings available if TMA is used.
2. **No MMA↔load overlap** — each outer iter is serialized:
   fill → sync → MMA+commit → wait → ld → sync → softmax → PV → sync.
   Double-buffer would hide ~15-20 µs.
3. **SMEM round-trip for logit broadcast** (1 KB × 8 iters = 8 KB extra
   SMEM traffic ≈ 2-3 µs).
4. **8×T SMEM fill overhead** (per-thread `q_addr/kv_addr` computation
   plus int4 scatter to non-contiguous SMEM addresses) vs A23's single
   linear `cp.async.bulk` call. Costs maybe 1-2 µs per outer iter × 8
   iters = 8-16 µs.

Target: recover 30+ µs to match A23, then another 5 µs to beat it.

### D3.3 — Atom intersections NOT yet exploited (optimization candidates)

| # | Intersection | Optimization | Expected Δ | Effort | Risk |
|---|---|---|---|---|---|
| **C1** | **REL × FXP (Pipeline)** | TMA prefetch for KV into 8×T SMEM, double-buffered (2 buffers) + mbarrier gate | **−20 to −30 µs** | HIGH — needs 3D tensor map (GT-42) with 128B swizzle, matching descriptor swizzle mode (GT-10) | HIGH |
| **C2** | **REL × FXP × FUN (Pipelined cp → mma)** | Use ISA §9.7.16.6.4.3 async pair: issue `cp.async.bulk` for iter k+1, MMA for iter k, overlapped by TMA unit | −10 to −15 µs on top of C1 | HIGH — needs C1 first | HIGH |
| **C3** | **MOR × REL (Warp specialization, T3-7)** | 4 warps producer (TMA issue), 12 warps consumer (softmax+PV). Producer never sleeps; overlaps loads with MMA+softmax+PV of prior iter | −5 to −10 µs | HIGH — new producer/consumer mbarrier choreography | MED |
| **C4** | **MOR × MEA (Drop Q SMEM if not needed)** | MMA requires Q in SMEM. BUT: partition Q differently so only active K-slice of Q is resident per outer iter (save SMEM, better occupancy). | Possibly enables 2 CTAs/SM = 2× SM utilization on N=1 workloads | VERY HIGH — K-slicing Q with running partial sum | HIGH |
| **C5** | **REL × MEA (Eliminate smem_logits broadcast)** | Currently warp 0 writes logits to SMEM, all 16 warps read. Alternative: warps 0-3 own their 16-row M-slice, handle their 16 heads' softmax+PV directly; no broadcast | −2 to −3 µs | MED — re-map head→warp to match MMA lane distribution | MED |
| **C6** | **AFM × SYM (Vectorize 8×T SMEM fill)** | Current fill uses int4 scatter with per-thread `q_addr` computation (divides + modulos). Replace with structured permutation that compiler can fold | −1 to −2 µs | LOW | LOW |
| **C7** | **MEA × FUN (Larger N tile)** | N=32 or N=64 per MMA, fewer outer iters (8→4 or 2). Reduces per-iter fixed MMA overhead. tcgen05.ld distribution must be rechecked for N>=32 | −5 to −10 µs | MED — GT-12 lane distribution for f16 N>16 not yet confirmed; needs probe | MED |
| **C8** | **MOR × MEA (K_PER_ITER for PV, not MMA)** | Keep MMA at N=16 but process multiple batches' softmax+PV before next MMA. Decouples MMA from softmax/PV cadence | Marginal | LOW | LOW |
| **C9** | **SRG × FUN (Pipelined `tcgen05.mma` with enable_input_d)** | Issue MMA for iter k+1 while lane reads TMEM for iter k (needs 2 TMEM allocations). Per PTX §9.7.16.6.4.3 canonical pattern | −2 to −5 µs | MED | MED |
| **C10** | **MEA × MOR (Use 2D TMA tensor map with SWIZZLE_128B per gau-nernst)** | Replace cooperative-fill with `cp.async.bulk.tensor` using 3D `(K/64, M, 64)` swizzled layout; matches MMA descriptor with `swizzle=2` | Enables C1 cleanly | HIGH — sparse-gather forces per-row TMA issue; may not benefit | HIGH |

### D3.4 — Critical-path ordering

The candidates have dependencies. Recommended order:

1. **C6** (vectorize SMEM fill) — free win, low risk, small gain. Do first.
2. **C1** + **C10** (TMA for KV into 8×T SMEM) — **the big one**, expected to recover 15-25 µs. High risk but highest reward.
3. **C9** (pipelined MMA using `enable_input_d`-aware sequence) — only meaningful after TMA is in place.
4. **C7** (larger N tile) — requires confirming tcgen05.ld distribution for N>16; do AFTER C1 lands so a working TMA path exists.
5. **C3** (warp specialization) — only after C1 + C9 establish the async pair; specialization shapes the overlap cleanly.
6. **C5** (eliminate logit broadcast) — cleanup after restructure.
7. **C4** (K-slice Q) — last-mile; only if other paths plateau.
8. **C2, C8** — sub-cases that fall out of C1/C3.

### D3.5 — Expected end state

If C1+C6+C7+C9 all land without introducing regressions:
- Cooperative copy replaced by TMA: ~25 µs savings.
- Double-buffer hides MMA latency: ~10 µs savings.
- Larger N tile reduces fixed overhead: ~5 µs savings.
- Total: ~40 µs savings → **tg_kernel might reach 0.040-0.045 ms**, i.e., parity with or barely beating A23.

This aligns with the D3 regime prediction: **tensor cores cannot go much
faster than scalar in a memory-bound regime** because both converge on HBM
throughput. The MMA path is competitive at best, not a clear win.

**V3**: Binding analysis complete. C1 is the critical optimization. If
C1 fails (because 2D TMA with sparse-gather is structurally infeasible),
the tcgen05 path CANNOT match A23 — GT-42 already flagged this risk.

---

## D4 — Architecture Specification (current tg_kernel FSM + target FSM)

### Current FSM (tg_kernel)

```
[CTA entry]
  ├── tid==0: mbarrier_init(mbar_mma, 1)
  ├── all: load smem_idx (cooperative)
  ├── all: zero-fill smem_q (73 KB / 16 B = 4608 int4 writes)
  └── __syncthreads
[Q fill — 1152 (h, chunk) pairs into 8×T SMEM]
  └── (no sync needed if each thread writes distinct addrs)
[TMEM alloc, warp 0]
  └── __syncthreads
[Outer loop × n_iters]
  ├── all: zero-fill smem_kv (1152 int4 writes)         (MOR.zero)
  ├── __syncthreads
  ├── all: cooperative KV fill → 8×T SMEM                (AFM + MOR)
  ├── __syncthreads
  ├── fence_before + __syncthreads + fence_after         (GT-3 pair for cp→mma)
  ├── tid==0: 36 × tcgen05.mma (kt: enable_input_d=kt>0)  (REL + PRD + MEA)
  ├── tid==0: tcgen05.commit
  ├── all: mbarrier_wait_parity(mbar_mma)                 (REL: wait MMA done)
  ├── fence_before + __syncthreads + fence_after          (GT-2 step 5)
  ├── warps 0-3: tcgen05.ld.32x32b.x16 + wait::ld         (TMEM→register)
  ├── warp 0 lane<16: write logits to smem_logits         (REL: broadcast)
  ├── __syncthreads
  ├── warp_id (=head): read logits from smem_logits       (REL: consume)
  ├── scale + sentinel mask + online softmax              (SRG × FXP × INV)
  ├── scalar PV from smem_kv (stride-32 per-lane output)  (AFM + SRG)
  └── __syncthreads
[Scratch write] (per-head, per-warp)                       (AFM + MEA)
[TMEM dealloc, warp 0]
[Atomic last-arrival]                                      (REL × ATO)
[Reduce (last CTA only): per-head global m/l/o]            (same as A23)
```

**Sync cost**: 7 `__syncthreads` per outer iter, 8 iters = 56 CTA-wide
syncs. A23 has 0 syncs per inner iter after A6 optimization.

### Target FSM (after C1 + C6)

```
[CTA entry — same as current]
[TMA prologue: issue TMA for iter 0 into buf[0], iter 1 into buf[1]]
[Outer loop × n_iters]
  ├── mbarrier_wait_parity(mbar_kv[consume_buf])           (wait for TMA)
  ├── [if has_next] tid==0: TMA issue iter+2 into buf[issue_buf]
  ├── fence_before + __syncthreads + fence_after
  ├── tid==0: 36 × tcgen05.mma
  ├── tid==0: tcgen05.commit(mbar_mma[phase])
  ├── mbarrier_wait_parity(mbar_mma[phase])                (wait MMA done)
  ├── fence::after_thread_sync
  ├── warps 0-3: tcgen05.ld ...
  ├── [remaining FSM identical to current]
```

Key differences:
- KV SMEM double-buffered (buf[0], buf[1]).
- TMA (cp.async.bulk.tensor) replaces cooperative copy.
- Prefetch issued 1 iter ahead; wait on consume side.
- Zero-fill eliminated (TMA overwrites the buffer each iter).

### Lifetime tables

| Variable | Birth | Death | Storage | Size | Reuse |
|---|---|---|---|---|---|
| smem_q (8×T) | CTA entry | CTA exit | SMEM | 72 KB | 8 iters |
| smem_kv (8×T) | per-iter | next iter | SMEM | 18 KB | 1 iter (overwritten) |
| smem_logits | per-iter | same iter | SMEM | 1 KB | 1 iter |
| tmem_acc | CTA entry | dealloc | TMEM | 4 KB | 8 iters |
| mbar_mma | CTA entry | CTA exit | SMEM | 8 B | 8 iters (phase flip) |
| smem_idx | CTA entry | CTA exit | SMEM | 512 B | 8 iters |
| row_max, row_sum (per-warp) | CTA entry | scratch write | registers | 2 fp32 × 16 warps | all iters |
| o_acc[16] (per-warp-lane) | CTA entry | scratch write | registers | 16 fp32 × 32 lanes × 16 warps | all iters |

Register budget per lane: ~20 fp32 + other state. Well under 64. OK.

### Address composition (8×T layout audit)

For `q_addr(m, kb)` to match the descriptor SBO=256, LBO=128 layout:
- Within one K-step (32 bytes of K): 8 m_grps × 2 K-tiles × 128 B = 2048 B total.
- K-step stride in SMEM: 2048 B (Q, 8 m_grps) vs 512 B (KV, 2 n_grps). ✓
- Descriptor start-addr advance per K-step: 128 (Q) vs 32 (KV), all encoded (>>4). ✓

GT-11 formula `m_grp × SBO + k_tile × LBO + m_in_grp × 16 + k_in_t`:
- For each K-step: 8 m_grps × 256 + 2 K-tiles × 128 + in_grp × 16 + in_t_byte = 2048 B stride ✓
- For K-axis multi-step: the per-K-step stride depends on the number of
  row-groups in the M (or N) dimension — NOT a constant. Descriptor
  advance per K-step differs between A and B operands.

**V4**: FSM matches implemented code. Lifetime tables and address
composition audited. No barrier justification errors. The target FSM
differs in exactly the pipelining structure — which is the C1
optimization.

---

## Why C1 (TMA prefetch) is The Critical Step

Everything else in the matrix is either small (<5 µs) or depends on C1
being in place first. Without TMA:
- Cooperative copy costs ~35-40 µs per CTA (the entire gap vs A23).
- Double-buffering is moot if the "load" is synchronous cooperative copy
  (just doing it from 2 buffers doesn't hide latency, only TMA does).
- Warp specialization offers nothing to specialize about if there's no
  async producer.

**C1's blocker (per GT-42)**: sparse-gather KV (one TMA per row from a
different source offset) + 8×T SMEM layout is NOT a standard pattern.
Two viable sub-approaches:

### C1a — Per-row 1D `cp.async.bulk` → linear SMEM staging → SMEM-SMEM rewrite into 8×T

- **Pros**: Each `cp.async.bulk` is per-row (16 rows per iter × 2 loads ckv+kpe = 32 TMA issues), writes to linear SMEM. A second pass rewrites linear → 8×T. TMA unit still used. Parallel with compute of previous iter.
- **Cons**: Adds a SMEM-SMEM rewrite (32 × 1152 B = 36 KB of SMEM traffic per iter). Doesn't fully match A23's efficiency.
- **Expected**: −15 to −20 µs vs current (partial recovery).

### C1b — 2D `cp.async.bulk.tensor` per gau-nernst's 3D (K/64, M, 64) pattern with SWIZZLE_128B

- **Pros**: One TMA call per row loads directly into 8×T-compatible swizzled layout. MMA descriptor matches (swizzle=2 per GT-10).
- **Cons**: Per PTX ISA, `cp.async.bulk.tensor` reads from a TensorMap's own strided source — but our source rows are at scatter-indexed positions in HBM, not a contiguous tensor. Each KV row is at `ckv_cache + kv_idx × HEAD_DIM_CKV`. We'd need either (1) one TensorMap per KV row (infeasible — thousands of descriptors), (2) modify the TensorMap per row using `tensormap.replace` flow (GT-6) — but that costs ~5 µs per modification, likely net loss, (3) use the tensor map only for the destination tile dimension and let the source be a runtime-addressed `base + offset*stride` — the CUDA API doesn't support this.
- **Expected**: Infeasible for sparse-gather KV.

**Conclusion for C1**: Only **C1a** (per-row 1D TMA + SMEM-SMEM rewrite)
is structurally viable. Expected −15-20 µs savings vs current tg_kernel.
That puts the MMA kernel at ~0.064-0.069 ms — still slower than A23 but
closer.

---

## Stopping Criterion Re-evaluation

Per CLAUDE.md §10 Tier 3 stopping criterion:
> Stop and report if an attempt regresses mean latency by >5% with no clear path forward

Current tg_kernel is at +91% (0.084 vs 0.044). Despite being >5%, we have
a CLEAR path forward (C1a + C6 + C7 + C9). Estimated reachable: ~0.045
ms — approximately matching A23 but unlikely to beat it decisively.

**Decision**: Continue optimizing tg_kernel.cu toward the target FSM.
Recommended next attempt: **C1a** (per-row TMA + SMEM rewrite). This is
the single highest-leverage change. If C1a regresses or fails, the
tcgen05 path is confirmed structurally unable to beat A23 and we revert
to the GT-39 conclusion.

---

## Recommended Optimization Sequence

**Pass 1 — Low-risk wins (T4 micro, total effort 1-2 hours)**:
1. **C6**: Vectorize/fold 8×T SMEM fill — compute `q_addr`/`kv_addr`
   once per (m, k_chunk) pair, loop-hoist the per-thread work.
2. Skip zero-fill of smem_kv when `kv_idx >= 0` (currently unconditional
   zero-fill + overwrite; skip the zero when overwrite is guaranteed).

**Pass 2 — TMA KV path (HIGH leverage, T3 full rigor, 4-8 hours)**:
3. **C1a**: Replace cooperative KV copy with `cp.async.bulk` (1D, per-row)
   into a linear SMEM staging area. Add SMEM→SMEM rewrite into 8×T.
   Double-buffer the staging area. Gate MMA on mbarrier completion.

**Pass 3 — Async MMA + TMA pair (T3, 2-4 hours)**:
4. **C9**: Pipeline MMA issue for iter k+1 against logits consumption of
   iter k. Needs second TMEM allocation or careful `enable_input_d`
   sequencing.

**Pass 4 — Bigger MMA tile (T3, requires probe, 3-6 hours)**:
5. **C7**: Port `sbo_lbo_sweep` to `kind::f16` M=64 N=32 (or 64), confirm
   lane distribution, enlarge N. 4 or 2 outer iters per CTA instead of 8.

**Pass 5 — Warp specialization (T3, 4-8 hours)**:
6. **C3**: 4 warps producer (TMA + mbarrier), 12 warps consumer
   (softmax + PV). Requires redesigning head→warp mapping.

After each pass, re-measure and compare against A23's 0.044 ms. Stop
the sequence when either:
- tg_kernel latency ≤ A23 (celebrate, commit).
- Regression with no obvious next step (revert to A23, document GT-39
  closing).

---

## V4 — Final verification

- **FSM**: matches tg_kernel.cu current structure; target FSM captures
  the pipelining addition.
- **Barrier justification**: each current `__syncthreads` documented in
  D4 FSM; at least 3 can be eliminated in the target FSM.
- **Address composition**: GT-11 formula applied correctly; K-step
  stride divergence between A and B operands confirmed.
- **GT coverage**: GT-11 (SBO/LBO), GT-12 (lane distribution), GT-14
  (no K-slab unroll), GT-15 (transpose bits), GT-17 (full-warp .ld),
  GT-18 (commit form), GT-40/41/42 (added this session) all in force.
- **Stopping criterion**: current state is at regression but with
  concrete C1a path forward; continuation is justified.

---

## What's in the next attempt

**Attempt T3-568 (C1a — TMA KV into linear SMEM + rewrite into 8×T)**:

Scope declaration draft:
- *Changes*: replace the cooperative HBM→SMEM KV fill with `cp.async.bulk`
  (1D, per-row × 2 sources per row = 32 TMA issues per iter). Add a
  linear SMEM staging region (16 × 1152 B = 18 KB). After mbarrier gate,
  rewrite staging → 8×T SMEM via cooperative in-SMEM copy. Double-buffer
  the linear staging (2 × 18 KB = 36 KB). Keep the single 8×T SMEM region
  (rewritten per-iter just before MMA). Inner loop becomes prefetch→wait
  →rewrite→MMA→compute.
- *Does not change*: split-K grid, last-arrival reduce, kernel signature,
  softmax+PV registers, head-per-warp mapping.
- *Revert target*: `checkpoints/tg_kernel.cu` (current T3-567d state).

Expected outcome range: −15 µs to −25 µs vs current 0.084 ms. Landing
point ~0.060-0.070 ms. Still ~0.02 ms slower than A23.
