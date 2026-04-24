# Framework Section Manifest

This directory contains the complete "Algebra of GPU Computation" framework
(v8), split into topic files. Each file preserves the full, unedited v8
content for its topic. No condensation, no paraphrasing.

Load ONLY the files listed for your current derivation step. Do not load
the entire framework at once — it exceeds context limits and most sections
are irrelevant to any single step.

---

## File Index

| # | File | Contents | Lines |
|---|------|----------|-------|
| 01 | `01_foundational_idea.md` | Geometric determinism, what the framework derives, five failure conditions, engineer guidance | 202 |
| 02 | `02_algebraic_properties.md` | The six algebraic properties beneath atoms (BO, ASC, IDE, COM, INV, DIS) | 352 |
| 03 | `03_eleven_atoms.md` | All eleven atom definitions (AFM, SRG, MON, PRD, ATO, FXP, MOR, MEA, REL, SYM, FUN) | 638 |
| 04 | `04_atom_intersection_matrix.md` | Every pairwise atom intersection including FXP×GATE → Streaming Gate | 597 |
| 05 | `05_structural_analysis.md` | Step 0.5 — molecule detection tables, Level 0 geometry spec, recursion morphism | 208 |
| 06 | `06_dimension_fates.md` | AREA and REDUCE (SRG/MON) dimension fate rules, atoms at Level 0 | 100 |
| 07 | `07_gate_specification.md` | GATE dimension fate, single-pass Gate specification table, Phase 1/Phase 2 structure | 112 |
| 08 | `08_streaming_gate.md` | Streaming Gate (FXP×GATE) — detection rule, two-case merge, FSM, pre-wired decisions ★ | 198 |
| 09 | `09_access_patterns.md` | Step 1a atoms, Step 1b access patterns, chain depth, per-hop fields | 239 |
| 10 | `10_hardware_binding.md` | Step 2 — binding table rules, atoms at Step 2 | 93 |
| 11 | `11_fsm_phases.md` | Step 3 — FSM phase structures, GATE/chain modifications, atoms at Step 3 | 70 |
| 12 | `12_lifetime_tables.md` | Steps 4a & 4b — lifetime tables, reuse validation, atoms at Steps 4a/4b | 51 |
| 13 | `13_address_composition.md` | Step 5 — affine map composition, FMA chains, atoms at Step 5 | 58 |
| 14 | `14_boundary_map.md` | Geometric determinism boundary, future extensions, complete kernel family map | 709 |

---

## Which Files to Load Per Derivation Step

### D1 — Molecule Detection
Load: **05, 06, 07, 08**
- 05: molecule detection tables (what to look for)
- 06: AREA/REDUCE fate rules
- 07: GATE fate + single-pass Gate spec
- 08: Streaming Gate detection rule and pre-wired decisions
  **Critical:** If both Tile and Gate are CONFIRMED, check 08's detection
  rule: "are the GATE dimension's candidates produced inside the FXP tile
  loop?" If yes → Streaming Gate applies and its pre-wired decisions must
  appear in d1.md.

### V1 — Verify Molecule Detection
Load: **05, 07, 08**
- Verify Gate detection against 07
- Verify Streaming Gate detection against 08's rule
- Check all pre-wired decisions from 08 are present if Streaming Gate applies

### D2 — Structural Analysis
Load: **03, 06, 09**
- 03: atom definitions (for structural analysis table)
- 06: dimension fate classification rules
- 09: access pattern rules, chain depth analysis

### V2 — Verify Structural Analysis
Load: **03, 06, 09**
- Same files — verify classifications against definitions

### D3 — Hardware Binding
Load: **04, 07, 08, 10**
- 04: atom intersection matrix (optimization opportunities)
- 07: Gate algorithm candidates (single-pass)
- 08: Streaming Gate merge algorithm and buffer sizing (if Streaming Gate)
- 10: Step 2 binding table rules

### V3 — Verify Hardware Binding
Load: **04, 07, 08, 10**
- Verify Gate algorithm choice against K/N ratio (07 + 08)
- Verify binding table completeness against 10
- Verify atom intersections exploited from 04

### D4 — Architecture Specification
Load: **08, 11, 12, 13**
- 08: Streaming Gate FSM structure (if applicable)
- 11: Step 3 FSM phase rules
- 12: lifetime tables
- 13: address composition rules

### V4 — Verify Architecture Specification
Load: **08, 11, 12, 13**
- Verify FSM matches Streaming Gate structure from 08 (if applicable)
- Verify barrier justification against 11
- Verify address formulas against 13

### Implementation
Load: **08, 11** (reference only)
- 08: Streaming Gate two-case merge algorithm (Case A / Case B)
- 11: FSM phase ordering

### Audit
Load: **04, 07, 08, 10, 11**
- 04: check all applicable atom intersections are exploited
- 07 + 08: Gate algorithm matches specification
- 10: binding table completeness
- 11: FSM phase correctness

### Optimization
Load: **04, 05, 07, 08**
- 04: atom intersection matrix (unexploited optimizations)
- 05: molecule re-evaluation
- 07: Gate algorithm class re-evaluation (K/N ratio)
- 08: Streaming Gate — if not already implemented, evaluate whether
  fusing Gate Phase 1 into the tile loop is possible

---

## Critical Cross-References

**FXP×GATE → Streaming Gate:** Appears in both 04 (intersection matrix
entry) and 08 (full specification). The intersection matrix entry in 04
contains the detection rule; 08 contains the complete algorithm. Both
must be consulted when Gate and Tile are co-confirmed.

**Gate Algorithm Selection:** The single-pass algorithm candidates are
in 07 (introselect, radix-select, bitonic sort, etc.). The Streaming
Gate's per-tile merge algorithm is in 08. These are different algorithms
for different situations — 07 applies when all candidates exist before
the loop; 08 applies when candidates arrive during the loop.

**Barrier Justification:** FSM barriers are defined in 11, but the
Streaming Gate adds an extra barrier (after Merge phase, before next
iteration's threshold read) that only appears in 08. If Streaming Gate
is active, 08's barrier requirements supplement 11's.
