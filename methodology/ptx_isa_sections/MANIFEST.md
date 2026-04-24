# PTX ISA Section Files

Split from `ptx_isa_9_0_sm100a_extract.txt` by topic.
Load ONLY the sections needed for the current pipeline step.

**Full file: ~41,220 tokens. Per-section: 200–2,500 tokens each.**

## Section Index

| File | Section | ~Tokens | Used By |
|------|---------|---------|--------|
| `swizzling_modes.txt` | §5.5.7 Swizzling modes for shared memory layout | 3,023 | D3 (SMEM layout), GT-10 verification |
| `shfl_sync.txt` | §9.7.9.6 shfl.sync instruction | 1,027 | Layer 5 (H-reduction) |
| `tma_cp_async_bulk_tensor.txt` | §9.7.9.24 cp.async.bulk.tensor (TMA) | 3,105 | Layer 2 (Q TMA), Layer 3 (K TMA), D3 item |
| `mbarrier.txt` | §9.7.13.15 mbarrier instructions (init, arrive, wait) | 10,900 | Layers 1-3 (barriers), D3 item 10 |
| `tcgen05_tmem.txt` | §9.7.16.1 Tensor Memory (TMEM) addressing and allocation | 397 | Layer 1 (TMEM alloc/dealloc) |
| `tcgen05_mma_shapes.txt` | §9.7.16.2 Matrix shapes — TABLE 41 | 4,658 | D3 item 4 (tile shape) |
| `tcgen05_strides_layouts.txt` | §9.7.16.3 Strides, leading dimension, canonical layouts | 2,874 | D3 item 6 (SMEM layout), make_smem_desc |
| `tcgen05_smem_descriptor.txt` | §9.7.16.4.1 Shared memory descriptor — TABLE 42 | 816 | Layer 4 (make_smem_desc), D3 item 9, Audit |
| `tcgen05_idesc.txt` | §9.7.16.4.2 Instruction descriptor (IDESC) — TABLE 44 | 1,574 | Layer 4 (IDESC encoding), D3 item 3, Audit |
| `tcgen05_issue_granularity.txt` | §9.7.16.5 Issue granularity, CTA groups | 1,687 | Layer 4 (elect_one_sync), Audit item 14 |
| `tcgen05_sync_fences.md` | §9.7.16.6 Memory consistency model, async/sync classification, pipelined pairs, canonical patterns | 2,813 | Layer 4 (MMA→ld ordering), GT-2, GT-3, sbo_lbo_sweep fix |
| `tcgen05_alloc_dealloc.txt` | §9.7.16.7.1 tcgen05.alloc, tcgen05.dealloc | 1,360 | Layer 1 (TMEM lifecycle) |
| `tcgen05_ld_st_wait.txt` | §9.7.16.8 tcgen05.ld, tcgen05.st, tcgen05.wait | 3,725 | Layer 4 (TMEM read), GT-3, Audit item 17 |
| `tcgen05_mma.txt` | §9.7.16.10 tcgen05.mma instruction | 952 | Layer 4 (MMA), D3 item 4, Audit items 7/8 |
| `tcgen05_fence_commit.md` | §9.7.16.11–12 tcgen05.fence and tcgen05.commit instructions | 900 | sbo_lbo_sweep fix (commit+mbarrier pattern), GT-2 |
| `tensormap_replace.txt` | §9.7.9.26 tensormap.replace instruction | 2,309 | GT-6 (tensormap update flow), Audit item 21 |

---

## Loading Guide by Pipeline Step

### D3 — Hardware Binding (~4,500 tokens instead of ~50,000)
```
ptx_isa_sections/tcgen05_mma_shapes.txt        # Table 41 — tile shapes
ptx_isa_sections/tcgen05_smem_descriptor.txt    # Table 42 — SMEM descriptor encoding
ptx_isa_sections/tcgen05_idesc.txt              # Table 44 — IDESC encoding
ptx_isa_sections/tensormap_replace.txt          # GT-6 — tensormap update flow
ptx_isa_sections/swizzling_modes.txt            # GT-10 — swizzle mode selection
```

### Layer 1 — TMEM Alloc/Dealloc (~1,500 tokens)
```
ptx_isa_sections/tcgen05_alloc_dealloc.txt
ptx_isa_sections/tcgen05_tmem.txt
```

### Layer 2 — Q TMA Load (~3,000 tokens)
```
ptx_isa_sections/tma_cp_async_bulk_tensor.txt
ptx_isa_sections/mbarrier.txt
```

### Layer 4 — First MMA (~5,000 tokens)
```
ptx_isa_sections/tcgen05_smem_descriptor.txt    # make_smem_desc
ptx_isa_sections/tcgen05_idesc.txt              # IDESC encoding
ptx_isa_sections/tcgen05_ld_st_wait.txt         # tcgen05.ld
ptx_isa_sections/tcgen05_sync_fences.md         # async/sync classification + canonical patterns
ptx_isa_sections/tcgen05_mma.txt                # MMA instruction
```

### Audit (~6,000 tokens)
```
ptx_isa_sections/tcgen05_smem_descriptor.txt
ptx_isa_sections/tcgen05_idesc.txt
ptx_isa_sections/tcgen05_sync_fences.md
ptx_isa_sections/tcgen05_ld_st_wait.txt
ptx_isa_sections/tensormap_replace.txt
ptx_isa_sections/tcgen05_mma.txt
```

### sbo_lbo_sweep Fix — tcgen05.commit + mbarrier pattern (~2,500 tokens)
```
ptx_isa_sections/tcgen05_sync_fences.md         # §9.7.16.6.4.2 canonical mma→ld pattern
ptx_isa_sections/tcgen05_fence_commit.md        # §9.7.16.11-12 fence + commit syntax/examples
ptx_isa_sections/tcgen05_ld_st_wait.txt         # tcgen05.ld + tcgen05.wait::ld
ptx_isa_sections/mbarrier.txt                   # mbarrier init, arrive, try_wait
```
