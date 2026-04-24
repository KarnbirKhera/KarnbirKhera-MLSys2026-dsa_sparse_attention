# Atom Intersection Optimization Matrix

Atom Intersection Optimization Matrix

      "When two specific atoms co-occur at the same step and level, what named optimization does their intersection enable or require?"

    

  

  
    How to read this matrix. Each cell names the optimization that emerges when the row atom and column atom appear at the same point in the derivation. The matrix is symmetric (A×B = B×A), so only the upper triangle is populated. When you see two atom chips active at the same step annotation, find their intersection here. If the cell is populated, apply that optimization before proceeding. Cells with "—" mean the two atoms do not interact productively at any standard hierarchy level. For any cell that references the Property Decomposition Reference, that table shows the algebraic property-level building blocks that compose the optimization and what breaks when individual properties are absent — consult it to understand why the optimization works, while this matrix tells you what to do.
  

  
    
      
        
          
          AFM
          SRG
          MON
          PRD
          ATO
          FXP
          MOR
          MEA
          REL
          SYM
          FUN
        

      
      
        
        
          AFM
          —
          Address Fusion
Two composable affine maps → single FMA instruction. Eliminating the intermediate register.

          Sequential Stride
Ordered scan uses stride-1 smem access; affine map must be row-major in scan dimension.

          Boundary Masking
Predicate applied to affine coordinate; replace branch with ternary. No warp divergence.

          Atomic Coalescing
When atomic target addresses are affine in thread index, warp election reduces 32× contention.

          Incremental Address
Each tile advances address by one stride; compute += stride rather than recompute full address.

          Address Composition
Compose block + thread affine maps into one expression; enables single-instruction address compute.

          Staging Threshold
Measure of data touched by affine map = smem capacity → staging is exactly break-even. Compute precisely.

          Coalescing Detection
Relation between lane index and address stride determines transaction count. Stride = 1 → 1 transaction.

          Guaranteed Coalescing
Translational symmetry in lane dimension guarantees stride-1 access pattern → automatic coalescing.

          Register Tile Lift
Functor from block-level affine map to thread-level tile address. Substitute TM for BM, threadIdx for blockIdx.

        

        
        
          SRG
          —
          —
          Commutativity Check
Before Blelloch: verify op truly lacks commutativity. If commutative → degrade to SRG, save log₂N steps.

          Predicated Reduction
Identity element for inactive threads; shuffle tree remains valid regardless of which lanes are masked. See Property Decomposition Reference for the IDE×COM foundation.

          Global Reduce
Cross-block reduction via atomicAdd. Valid only for commutative ops. Avoids second kernel launch for small problems. See Property Decomposition Reference for the COM and COM×INV foundation.

          Online Algorithm
Semiring with merge operation → single-pass fixed point. Online softmax: carry running_max, running_sum across tiles. See Property Decomposition Reference for the four property-level building blocks (ASC×DIS, ASC×COM, IDE×COM, IDE×INV) and what breaks when each is absent.

          Domain Transform
Log-space morphism: products → sums. Prevents overflow, enables more numerically stable shuffle trees.

          Reduction Level
Measure(threads × reg_size) vs Measure(smem) determines whether reduction stays at warp or needs block-level smem.

          Combine Group
Two related SRG dims at same level → REDUCE_MERGE micro-phase. Relation discovered at Step 1a shape comparison. Note: COM eliminates intra-reduction ordering dependency. See Property Decomposition Reference for the negative relation.

          Free Parallel Tree
Commutativity IS permutation symmetry. Symmetry → hardware-optimal butterfly order. Any permutation gives same result. See Property Decomposition Reference for the ASC×COM → SYM mapping through FUN.

          Hierarchical Reduce
Functor maps warp semiring to block semiring to grid semiring. Same operation, three levels, all correct.

        

        
        
          MON
          ——
          —
          Boundary Scan
Last tile in scan dimension may be partial; predicate identity-fill (IDE) in down-sweep for out-of-range positions.

          Atomic invalid
MON lacks COM → atomicAdd gives wrong results. Cross-block monoid reduction requires a second kernel with ordered merge. See Property Decomposition Reference.

          Sweep as Fixed Point
Up-sweep is a fixed point over tree depth; down-sweep is its time-reversed counterpart. Both terminate at log₂N. See Property Decomposition Reference for why absent COM forces the two-phase structure and absent INV prevents online correction.

          Sweep Morphism
Blelloch is a morphism from sequential scan to parallel prefix. Preserves output; changes order of operations.

          Sweep Depth Bound
2×log₂N iterations; known statically. Enables full loop unrolling of Blelloch phases for small N.

          Order Relation
The ordering constraint on MON is the formal relation output[i] depends on output[i-1]. Makes this a REL atom, not just MON.

          Asymmetry Cost
Broken commutativity symmetry = O(log N) not O(1). The cost difference between SRG and MON is exactly the cost of one missing property. See Property Decomposition Reference.

          Scan Lift
Functor maps sequential scan problem to Blelloch parallel problem. Functor preserves output; maps domain from serial to parallel.

        

        
        
          PRD
          ———
          —
          —

          Predicate Hoisting
Predicates constant across all tile loop iterations precomputed once before tile loop. Only boundary tiles differ. See Property Decomposition Reference for the IDE×COM uniformity foundation.

          Guard Normalization
Morphism between predicate forms: ternary select ↔ masked load ↔ early exit. Choose form with fewest instructions for this hardware generation.

          Warp Efficiency
Measure of active lanes (predicate true) / total lanes. Below 50% → consider compacting active work into fewer blocks.

          Guard Propagation
Related phases that share a predicate condition compute it once; propagate via register. Avoids recomputing the same boundary check.

          Path Specialization
Symmetric predicate pattern (only last tile is partial) → generate fast path (no predicates) + slow path (with predicates). When IDE×COM holds, symmetry is healed and both paths are identical. See Property Decomposition Reference.

          Predicate Functor
Functor lifts thread-level predicate to block-level early exit when predicate is uniform across all threads in block.

        

        
        
          ATO
          ————
          —
          —

          Contention Morphism
Morphism from unordered global atomic to warp-elected smem atomic. Reduces contention; preserves semiring correctness.

          Contention Measure
Measure(threads competing for same address) = serialization cost. Optimize by reducing this measure via address hashing or warp election.

          Atomic Clustering
Atomics targeting related (adjacent) addresses benefit from sorted dispatch. Group related atomics to exploit hardware batching.

          Symmetric Atomics
When atomic target addresses are symmetric across lanes, elect one representative lane per warp. Reduces atomic operations by 32×.

          —

        

        
        
          FXP
          ———
          Streaming Gate ★
When a GATE dimension's candidates arrive in batches across a tile loop (FXP), the single-pass two-phase FSM is not enough. The output-write phase must move to after the tile loop. A running accumulation buffer of size K is maintained across all tile iterations, with a mandatory two-case per-tile merge. See the Streaming Gate section in the Gate Specification Table for the full algorithm. Detection: are the GATE dimension's candidates produced inside the FXP tile loop? If yes → Streaming Gate applies.

          —
          —
          Convergence Transform
Morphism to a faster-converging equivalent problem. Preconditioned iterations converge in fewer steps.

          Iteration Bound
Measure(REDUCE dim) / tile_size = static iteration count. Static bound → loop unrolling. Precompute before kernel launch.

          Pipeline Overlap
Two related fixed points can overlap when only final output is shared. Stages of the two loops interleave via double buffering.

          Sweep Sharing
Time-reversal symmetric iterations (Blelloch up/down sweep) share intermediates. Store up-sweep results; reuse in down-sweep. Halves memory traffic.

          Pipeline Lift
Functor lifts single-kernel tile-loop fixed point to multi-kernel pipeline fixed point when problem exceeds single-kernel smem capacity.

        

        
        
          MOR
          ——————
          —
          Compression Staging
Morphism that reduces data measure (e.g., FP32 → FP8) before staging in smem. More elements fit in fixed budget.

          Structural Reuse
Morphically related kernels share complete derivations. Derive once; transport through the morphism to get the second kernel's structure.

          Symmetry Exploitation
Automorphism (self-morphism = symmetry) halves work: compute one half, apply the automorphism to derive the other half.

          Derivation Transport
Functor transports optimization decisions from one kernel's derivation to a structurally equivalent kernel. No need to re-run full framework.

        

        
        
          MEA
          ———————
          —
          Joint Budget Check
Related buffers (Combine Group members) share hardware capacity. Optimizing one constrains the other. Check jointly, not independently.

          Symmetry Halving
Symmetric computation halves the measure of required work. When computation is symmetric, one half is free via the symmetry transformation.

          Budget Transformation
Functor maps smem budget constraint at block level to register budget constraint at thread level. Optimizing one level ripples through the functor.

        

        
        
          REL
          ————————
          —
          Symmetric Fusion
When the relation between two computations is symmetric, fuse them into one kernel that computes both simultaneously.

          Relation Lifting
Functor lifts thread-level relations (data dependency) to warp-level relations (barrier) to block-level relations (smem ordering). Each lift adds one synchronization cost.

        

        
        
          SYM
          —————————
          —
          Symmetry Preservation
Verify that the functor mapping one level to the next preserves the relevant symmetry. If it doesn't, the optimization based on that symmetry is unavailable at the target level.

        

        
        
          FUN
          ——————————
          —
        

      
    
  

  
    OptSig

    
      Optimization Signal Matrix — 11 Atoms × 5 Levels

      "Given atom A at hierarchy level L, what optimization should I check first, and which Intersection Matrix cell amplifies it?"

    

  

  
    For each of the eleven atoms at each hierarchy level, the primary optimization is listed along with the Atom Intersection Matrix cell that unlocks it. When two atoms are both active at a level, always check the Atom Intersection Matrix for the compound optimization — the intersection is almost always stronger than either atom alone.
  

  
    
      
        
          Atom
          ThreadRegisters
          Warp32-lane
          BlockShared Mem
          GridGlobal Mem
          PipelineAsync
        

      
      
        
          AFM
          Vectorize (float4)
→ AFM×FUN: Register Tile Lift
4-element load if stride=1 at thread level. Four FMAs replace four separate loads.

          Coalescing (stride=1)
→ AFM×SYM: Guaranteed Coalescing
Verify lane-stride=1. If true, symmetry guarantees single transaction. If not, count actual transactions.

          Bank Conflict Avoidance
→ AFM×MOR: Smem Layout Transform
XOR swizzle or +1 column padding breaks harmful modular symmetry of banking.

          Tile Base Address
→ AFM×MEA: Staging Threshold
Compute global base address. Check if tile measure equals smem capacity → staging exactly breaks even.

          Prefetch Address
→ AFM×FXP: Incremental Address
Next tile's base = current base + tile_stride. Incremental update, no full recomputation.

        

        
          SRG
          ILP / Unroll
→ SRG×SYM: Free Parallel Tree
Accumulator tiles are independent → issue multiple FMAs simultaneously. Symmetry enables any ordering.

          Tensor Cores (if MMA shape)
→ SRG×FUN: Hierarchical Reduce
If K dimension matches MMA shape (16/8/4), functor maps semiring to tensor core instruction family.

          Occupancy vs. SMEM Trade-off
→ SRG×MEA: Reduction Level
warp_partial buffer size × NUM_WARPS = smem consumed for block-level reduction. Measure both sides.

          Grid-Level Atomic (if small N)
→ SRG×ATO: Global Reduce
For small problems: skip second kernel, use atomicAdd directly. Valid only for commutative SRG.

          Overlap Compute + Load
→ SRG×FXP: Online Algorithm
Online merge operator carries state forward. Load next tile while computing on current tile.

        

        
          MON
          Sequential Fallback
→ MON×REL: Order Relation
If N=1 per thread, sequential is optimal. Order relation between i and i-1 must be maintained.

          Blelloch Warp Scan
→ MON×SYM: Asymmetry Cost
Use __shfl_up + __shfl_xor. 2×log₂N steps. Naming the broken symmetry explains the unavoidable cost.

          Blelloch Block Scan
→ MON×FXP: Sweep as Fixed Point
2×log₂N barriers. Up-sweep is a fixed point over tree depth; recognize this to enable unrolling.

          Two-Kernel Scan
Outside scope — see scope limitations. Requires multi-block coordination the framework does not derive.

          Sweep Interleave
→ MON×MOR: Sweep Morphism
The morphism from sequential to parallel scan enables interleaving stages with compute when problem structure permits.

        

        
          PRD
          Ternary Select
→ PRD×FXP: Predicate Hoisting
Is this predicate constant across all tile iterations? If yes, compute once before the loop.

          Warp Vote → Hoist
→ PRD×SYM: Path Specialization
Symmetric predicate pattern (only boundary tiles need it) → fast path for interior, slow for edge.

          Block Early Exit
→ PRD×REL: Guard Propagation
Related blocks share predicate. Compute predicate once at block launch; propagate to all related phases.

          Causal Tile Skip
→ PRD×MEA: Warp Efficiency
Skip entire KV tiles beyond causal boundary. Measure of skipped tiles × tile_cost = compute saved.

          Conditional Async Issue
→ PRD×FUN: Predicate Functor
Thread-level predicate lifts to block-level early exit. Functor: if all threads in block share predicate, hoist to block level.

        

        
          ATO
          —

          Warp Election (32× reduction)
→ ATO×SYM: Symmetric Atomics
Symmetric atomic targets across lanes → elect one representative. Reduces atomic ops by 32×.

          Smem Atomic vs Global
→ ATO×MEA: Contention Measure
Measure contention = threads competing per address. Smem atomic is faster when contention is high.

          Sort by Address
→ ATO×REL: Atomic Clustering
Sort atomic operations by target address before dispatch. Related addresses → hardware batching.

          Incompatible

        

        
          FXP
          State Variable Lifetime
→ FXP×MEA: Iteration Bound
State variables alive for entire tile loop. Bound = ceil(K/BK). Static → unroll; dynamic → keep counter in register.

          Loop Counter Uniformity
→ FXP×SYM: Pipeline Period
All 32 lanes share same tile loop counter → no divergence. Period-2 symmetry of double buffer state.

          Double Buffer (Period-2)
→ FXP×SYM: Sweep Sharing
Period-2 fixed point of buffer state. Exploit time-reversal symmetry of Blelloch to share intermediates.

          Num Tiles Precompute
→ FXP×REL: Pipeline Overlap
Iteration bound is static. Related fixed points can overlap when only final output is shared.

          Async Init+Drain Phases
→ FXP×MOR: Convergence Transform
Morphism from sync tile loop to async pipeline. Preserves correctness; adds Init and Drain phases to FSM.

        

        
          MOR
          FMA Composition
→ MOR×AFM: Address Composition
Compose block + thread affine maps. The morphism composition is the FMA instruction.

          Shuffle Pattern Selection
→ MOR×SRG: Domain Transform
The morphism from problem domain to log domain transforms products to sums. Enables numerically better shuffle tree.

          Smem Layout Transform
→ MOR×MEA: Compression Staging
Morphism reduces data size (FP32 → FP8) before staging. More elements fit in the fixed smem measure.

          Kernel Fusion
→ MOR×REL: Structural Reuse
Morphically related kernels share derivation. Transport the binding table through the morphism.

          Async Morphism
→ MOR×FXP: Convergence Transform
Morphism from sync tile loop to async pipeline. Replaces __syncthreads with cp.async.wait_group.

        

        
          MEA
          Register Budget Check
→ MEA×FUN: Budget Transformation
Register budget at thread level constrains smem budget at block level through the functor. Check both.

          Reuse Count Validation
→ MEA×SRG: Reduction Level
If reuse count > 1 at warp level, keep value in register. If > N_smem, stage in smem. Measure determines level.

          Smem Budget Check
→ MEA×REL: Joint Budget Check
Related buffers (Combine Group) share smem capacity. Sum all together; optimize jointly not independently.

          Grid Occupancy
→ MEA×SYM: Symmetry Halving
If computation has symmetry, measure of work is halved. Update occupancy calculation accordingly.

          Pipeline Depth vs Budget
→ MEA×FXP: Iteration Bound
Each buffer stage multiplies smem measure by pipeline depth. Deeper pipeline = lower occupancy. Static bound guides choice.

        

        
          REL
          Dependency Chain
→ REL×FXP: Pipeline Overlap
Intra-thread dependency chain: accumulator at step i depends on step i-1. This ordering relation makes tile loop a fixed point.

          Coalescing Verification
→ REL×AFM: Coalescing Detection
Verify stride-1 relation between lane index and address. If not stride-1, count actual DRAM transactions.

          Barrier Minimization
→ REL×SYM: Symmetric Fusion
If write→read relations between blocks are symmetric, fuse the blocks into one. One barrier serves both.

          Combine Group Merge
→ REL×SRG: Combine Group
Relation between two REDUCE dims → REDUCE_MERGE micro-phase at Step 3. Identify the combining lane (lane 0 of intersection).

          Async Dependency
→ REL×FUN: Relation Lifting
Thread dependency lifts to block barrier lifts to async wait_group through the functor. Each lift = one synchronization.

        

        
          SYM
          ILP from Tile Symmetry
→ SYM×SRG: Free Parallel Tree
Accumulator tile rows are translationally symmetric. Symmetry enables ILP: issue multiple independent FMAs.

          Lockstep from AREA Symmetry
→ SYM×AFM: Guaranteed Coalescing
AREA dimension translational symmetry guarantees all 32 lanes execute identically. No divergence analysis needed.

          Break Bank Symmetry
→ SYM×MOR: Smem Layout Transform
Detect harmful bank symmetry (stride multiple of 32 banks) → apply XOR swizzle to break it.

          Boundary Symmetry → Fast Path
→ SYM×PRD: Path Specialization
Interior tiles share translational symmetry (no predicates). Boundary tiles break it. Specialize two code paths.

          Period-2 Pipeline
→ SYM×FXP: Sweep Sharing
Double buffering exploits period-2 symmetry. Recognize this to correctly generate buf = tile & 1 idiom.

        

        
          FUN
          Thread Tile from Block Tile
→ FUN×AFM: Register Tile Lift
Apply functor: replace BM→TM, blockIdx→threadIdx. Thread tile address is the functorial image of block tile address.

          Primitive Family Selection
→ FUN×SRG: Hierarchical Reduce
Functor maps SRG → shuffle tree, MON → Blelloch. This is the framework's most important single functor application.

          Smem from Global Layout
→ FUN×MOR: Derivation Transport
Functor maps global memory layout to smem layout, composed with bank-conflict-avoidance morphism.

          Grid Shape from AREA Dims
→ FUN×MEA: Budget Transformation
Functor maps AREA dimension sizes to grid dimensions. Resource budgets transform accordingly through the functor.

          Sync → Async Morphism
→ FUN×FXP: Pipeline Lift
Functor lifts synchronous tile loop to async pipeline. New FSM phases (Init, Drain) are functorial images of existing phases.

        

      
    
  

  
    Step0.5

    
      Structural Analysis — The Six Extended Atoms

      "Before binding anything to hardware: what is the global mathematical structure of this computation — its symmetries, relations, measures, morphisms, fixed points, and functorial connections?"

    

  

  
    Step 0.5 runs once, immediately after Level 0 (Geometry Specification) and before Step 1a (Dimension Fate). It does not produce any binding decisions. It produces a Structural Analysis Table — six rows, one per extended atom — that informs every subsequent step. When a later step reaches a decision point, the relevant row of this table tells you which optimization the Atom Intersection Matrix unlocks.

    Unlike Steps 1–5, Step 0.5 asks about the computation's global structure rather than its local mechanics. It is the step where you ask not "what does this kernel do?" but "what kind of thing is this computation?" The answers determine which intersections will fire and which optimizations are available before a single tile size is chosen.

  

  
    
      
        Atom
        Question
