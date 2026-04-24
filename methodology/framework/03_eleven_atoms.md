# The Eleven Atoms

The Eleven Atoms

      "What are the named structural units — built from the algebraic properties above — that every GPU computation instantiates?"

    

  

  
    Why eleven and not five? The original five atoms describe what operations exist at each level of the hierarchy. The six extended atoms describe how those operations are structured, measured, related, and transformed. Neither set is complete without the other. A kernel designed with only the original five will be correct but leave systematic optimizations undiscovered. The extended atoms are the grammar for finding those optimizations — each one names a structure that was always there, just unnamed. The six algebraic properties in the section above sit beneath the atoms — they are the mathematical foundation from which SRG and MON are composed, and checking them individually gives more precise predictions than checking the composite atoms alone.
  

  — Original Five: Local Operation Atoms —

  
    
      AFM · Affine Map
      coord × stride + offset

      Every memory address in a parallel computation is an affine map from execution coordinates to memory coordinates. It is the simplest bridge between two linear spaces that preserves their geometry. No simpler structure can express arbitrary stride-based addressing; no more complex structure is required for all regular tensor access.

      GPU manifestation: blockIdx.x × BM + threadIdx.x. Every index formula in every kernel. The stride is the coefficient; the offset is the tile position. Appears at all five hierarchy levels simultaneously — one composition per level, all composing into the final global address.

    

    
      SRG · Semiring
      Associative + Commutative Reduction

      An accumulation where the order of combination is irrelevant to the result. Any partial results from any threads in any order combine correctly. This commutativity is a permutation symmetry — the semiring is invariant under all permutations of the elements being combined, which is exactly what makes a free parallel shuffle tree possible. Composed from: ASC (tree parallelism) + IDE (initialization and padding) + COM (any-order evaluation, enabling shuffle tree) + DIS (tiling correctness, connecting × and + into the A×B+C formula). Removing COM downgrades to MON. Removing DIS breaks tiling. Each property contributes a specific GPU capability.

      GPU manifestation: __shfl_down_sync for sum/max/min; tensor core MMA for dot products. The shuffle tree is the hardware-optimal expression of the semiring's permutation symmetry. O(1) per element cost.

    

    
      MON · Monoid
      Associative, Non-Commutative Reduction

      An accumulation where grouping is free but order is fixed. The absence of commutativity symmetry forces an ordered tree structure. This is not a performance limitation imposed by hardware — it is the mathematical consequence of the broken symmetry. The Blelloch two-phase sweep is the optimal parallel expression of a monoid operation. Composed from: ASC (tree parallelism — the Blelloch tree structure is possible) + IDE (initialization and padding — the sweep writes identity to the root between phases). The critical difference from SRG is the absence of COM: without commutativity, the shuffle tree is invalid, forcing the more expensive ordered two-phase sweep.

      GPU manifestation: Blelloch up-sweep + down-sweep via __shfl_up_sync / __shfl_xor_sync. O(log N) per element. Prefix sum, running count, ordered cumulative max.

    

    
      PRD · Predicate
      Conditional Validity Gate

      A binary condition on a memory address that determines whether a value is valid or should be replaced by the identity element of the combining operation. The predicate guards the boundary between valid and invalid territory. Its triggering iterator determines where it lives: grid-level iterator → early exit; tile-level iterator → predicated load inside the loop.

      GPU manifestation: Boundary if checks for early exit; ternary masking inside tile loops. The identity element (0 for sum, −∞ for max) is what inactive threads contribute.

    

    
      ATO · Atomic Op
      Serialized Cross-Scope Write

      A write to a shared address that guarantees mutual exclusion — only one thread modifies the location at a time. The atomic operation is the mathematical structure of any system with shared mutable state where concurrent modification is possible. It trades parallelism for correctness at a single address.

      GPU manifestation: atomicAdd, atomicMax, atomicCAS. Valid only for semiring-type operations (commutative write semantics). Appears at Block→Global and Grid→Global boundaries.

    

  

  — Extended Six: Structural Relationship Atoms —

  
    
      FXP · Fixed Point
      Iterate Until Self-Consistent

      A computation whose state is repeatedly transformed by a function until the state satisfies a consistency condition — either the function returns the same value (true fixed point) or all input tiles have been consumed (bounded fixed point). The tile loop is a bounded fixed point. The framework's validation cycle (Step 4b feeding back to Step 2) is a true fixed point. Online softmax is a fixed point with a stateful merge operator.

      GPU manifestation: Every tile loop is a bounded fixed point: apply partial semiring, advance tile index, repeat. Double buffering is a period-2 fixed point of the buffer state. The pipeline Init→Compute→Drain sequence terminates when no tiles remain. Framework self-consistency terminates when all budget checks pass without revision.

    

    
      MOR · Morphism
      Structure-Preserving Map Between Levels

      A map between two mathematical objects that preserves their structure — if A and B are related in the source, their images are related in the target. The binding table is a morphism from problem geometry to execution hierarchy. The framework runs in reverse because that morphism is an isomorphism — it has an inverse. Two kernels related by a morphism share structural decisions and can share derivations.

      GPU manifestation: The entire binding table. The reverse-engineering direction (code → geometry). Kernel fusion (composing morphisms eliminates intermediate memory). FMA instruction (composing two affine maps into one hardware instruction). The atom composition matrix entries — each cell is a morphism between levels.

    

    
      MEA · Measure
      Consistent Size Assignment

      A function that assigns a consistent size to subsets of a resource space — the size of a union of disjoint parts equals the sum of their sizes. Shared memory budget, register count, occupancy, and reuse count are all discrete measures on their respective resource spaces. The staging decision is a cost-benefit analysis where both sides are measures.

      GPU manifestation: smem budget check (sum of Block→Shared buffer sizes vs. hardware limit). Register budget (sum of Thread→Register state variable sizes). Reuse count (product of absent AREA dimension sizes — the measure of temporal reuse). Occupancy (measure of simultaneous resident blocks per SM).

    

    
      REL · Relation
      Named Connection Between Operations

      A set of pairs that specifies which operations are connected and how. The FSM is a relation between phases (which can follow which). Barriers encode a relation between threads (one writes, another reads). Combine Groups are a relation between REDUCE dimensions (their outputs feed the same value). Coalescing is a relation between thread lane indices and memory addresses (stride = 1).

      GPU manifestation: FSM phase ordering. __syncthreads() barrier (relation: writer thread → reader thread mediated by smem). Combine Group (relation between two semiring reductions). Coalescing condition (relation between threadIdx.x and address stride). Dependency edges between tiling levels in the binding table.

    

    
      SYM · Symmetry
      Invariance Under Transformation

      A transformation that leaves the relevant structure of a computation unchanged. AREA dimensions are translational symmetries of the computation — computing at position m is structurally identical to computing at position m+1. The semiring's commutativity is a permutation symmetry. Coalescing exploits translational symmetry of access patterns with respect to thread indices. Bank conflicts are harmful symmetries that cause serialization.

      GPU manifestation: AREA dimensions (translation symmetry → blocks never communicate). Coalesced access (thread-index translation symmetry → single memory transaction). Bank conflict (modular symmetry of banking → harmful serialization, broken by XOR swizzle or padding). Pipeline periodicity (period-2 symmetry of double buffer state).

    

    
      FUN · Functor
      Structure-Preserving Map Between Levels of Description

      A morphism between categories — a map that translates not just objects but the relationships between them. The framework is a functor from the category of tensor operations (geometry specs) to the category of GPU kernel implementations. The atom composition matrix is its functorial structure. Framework evolution is a sequence of such functors, each preserving correct derivations and extending to new cases.

      GPU manifestation: The framework derivation sequence itself (functor from geometry to kernel). The binding table's level mapping (block tile → thread tile address formula is derived by applying the functor to the block-level affine map). Register tiling derivation (thread-tile address is the functorial image of the block-tile address). The "Feeds into →" annotations are the functor's action on morphisms between steps.

    

  

  
    Matrix11×5

    
      Atom Composition Matrix — All 11 Atoms × 5 Hierarchy Levels

      "Which primitive does each atom instantiate at each level of execution, and what does it cost?"

    

  

  
    
      
        
          Atom
          ThreadRegisters · Scalar ALU
          Warp32-lane lockstep
          BlockShared Memory
          GridGlobal Memory
          PipelineAsync / Multi-phase
        

      
      
        
        
          AFM
          Register tile offset
rRow * TN + rCol
Thread's private tile addressed by intra-tile coordinates. Scalar multiply-add per access.

          Lane-stride address
lane_id * stride
All 32 lanes compute addresses in lockstep. Stride = 1 → coalesced; stride > 1 → check transaction count.

          Smem tile index
sRow * BN + sCol
Address into shared memory tile. Must avoid bank conflicts: sCol stride should not be a multiple of 32.

          Global tile base
blockIdx * BM * stride
Block's position in global memory. This is the coarse-level affine map; thread-level offsets are composed on top.

          Prefetch address
next_tile_base
Async pipeline precomputes the next tile's global address before the current tile finishes computing.

        

        
        
          SRG
          Scalar accumulate
acc += a * b
Thread-private running accumulator. FMA instruction. Lives in registers for the full tile loop lifetime.

          Warp shuffle tree
__shfl_down_sync
O(1) per element. Free parallel tree exploits permutation symmetry. No smem, no barrier. Optimal for sum/max/min.

          Block partial merge
warp_partial[warp_id]
If NUM_WARPS > 1: warp 0 reads all warp partials from smem and combines. One extra barrier.

          Grid atomic reduce
atomicAdd(out, val)
Cross-block reduction when no second kernel launch is feasible. Valid only for commutative operations.

          Online merge
merge(acc, new_partial)
Fixed-point tile loop: accumulate partial semiring each iteration. Online softmax is this pattern — merge operation carries running max and sum forward.

        

        
        
          MON
          Sequential scan
out[i] = f(out[i-1], in[i])
Thread-serial ordered accumulation. O(N) per thread. Only correct approach when no parallelism is available.

          Blelloch warp scan
__shfl_up_sync / __shfl_xor_sync
O(log N) warp-level prefix scan. Up-sweep builds binary tree; identity written to root; down-sweep propagates prefixes. 2×log₂N shuffle instructions.

          Blelloch block scan
smem[N-1] = identity
Two-phase sweep over smem array. 2×log₂N barriers. Size-N smem array declared in Step 2. One barrier per tree step.

          Multi-block scan
scan + fixup kernel
Requires two kernel launches: first kernel produces block-level prefixes; second adds inter-block prefix to each element. Outside framework's single-kernel scope.

          Sweep phases
up-sweep → down-sweep
The Blelloch sweep is itself a two-phase fixed point: up-sweep is a fixed point over the tree depth; down-sweep is its time-reversed counterpart.

        

        
        
          PRD
          Ternary mask
val = valid ? load : identity
Thread-private predicate. Replaces conditional branch with data-select. No warp divergence when all 32 threads share the same predicate outcome.

          Warp vote
__ballot_sync
Detect whether predicate is uniform across warp. If all active or all inactive, hoist decision out of inner loop. Eliminates per-element branch cost.

          Block early exit
if (cond) return;
Grid-level conditional leaf → entire block exits before entering tile loop. Eliminates blocks whose output tile is fully out of range.

          Causal mask
if (k_pos > q_pos)
Grid-level causal masking: skip KV tiles entirely whose positions exceed query position. Saves compute proportional to mask sparsity.

          Predicate hoisting
bool edge = block_m*BM+BM > M
FXP × PRD: predicates constant across tile loop iterations computed once before the loop. Only the last tile on each edge dimension needs predication.

        

        
        
          ATO
          Register private
No atomic needed — thread owns its registers exclusively.

          Warp elect
__ffs(__activemask())
Elect one lane to perform a single shared-memory or global write on behalf of the warp. Reduces atomic contention by 32×.

          Smem atomic
atomicAdd(&smem[idx], v)
Block-scoped atomic for histogram / scatter accumulation into shared memory. Faster than global atomic; still serializes on conflict.

          Global atomic
atomicAdd(ptr, val)
Cross-block write to global memory. Only valid for semiring operations. Contention cost = Θ(threads × conflict probability).

          Atomic pipeline
Async pipelines and atomics are architecturally incompatible — do not combine.

        

        
        
          FXP
          Register state carry
acc, running_max, running_sum
State variables that persist across tile loop iterations in registers. The fixed point carries this state through each iteration until all tiles are consumed.

          Warp-uniform iteration
for tile in range(num_tiles)
All 32 lanes advance through the tile loop in lockstep. The loop counter is warp-uniform — no divergence, no ballot needed.

          Double buffer ping-pong
buf = tile & 1
Period-2 fixed point of the buffer state. One buffer is being computed while the other is being loaded. The two-buffer cycle repeats for every tile.

          Iteration count
num_tiles = ceil(K / BK)
The bounded fixed point's termination condition: REDUCE dimension size divided by tile size. Precomputable, static, enabling loop unrolling.

          Async fixed point
cp.async + wait_group
The pipeline's Init→Tile→Drain structure is a fixed point at the pipeline level: issue prefetch, advance state, check drain condition, repeat.

        

        
        
          MOR
          FMA composition
fma(a, b, c)
Two affine maps composed into one FMA instruction: (x × a + b) feeds into (y × c + d) → single hardware instruction. Morphism composition = hardware fusion.

          Warp shuffle isomorphism
The warp shuffle tree implements the semiring's permutation symmetry. The morphism from "any order" to "hardware-optimal butterfly pattern" preserves the result.

          Smem layout transform
XOR swizzle / transpose
The morphism from row-major global layout to bank-conflict-free smem layout. Preserves data values while changing address structure.

          Kernel isomorphism
Two geometry specs related by a morphism produce kernels with identical binding table structure. Derive once, transport through the morphism for the second kernel.

          Pipeline morphism
The async pipeline is a morphism from the synchronous tile loop to an overlapped implementation. The morphism preserves correctness; the async barrier replaces the sync barrier.

        

        
        
          MEA
          Register budget
TM × TN × sizeof(float)
Measure of register file consumed by accumulator + input tiles. >64 floats → register spill risk. Directly controls warp-level occupancy.

          Reuse count (temporal)
∏(absent AREA dims)
The measure of how many times a loaded value is reused before it is no longer needed. Reuse > 1 justifies keeping the value in a faster memory level.

          Smem budget
Σ(Block→Shared sizes)
Measure of shared memory consumed by all staged buffers. Must fit within hardware limit (48–164 KB). Directly controls block-level occupancy.

          Grid occupancy
SM_count × blocks_per_SM
Measure of simultaneous resident blocks. Constrained jointly by smem budget and register budget. The occupancy-throughput trade-off is a measure optimization.

          Pipeline depth
num_buffers × buffer_size
Each pipeline stage multiplies the smem measure by the pipeline depth. Deeper pipeline = more smem consumed = fewer resident blocks = potential occupancy loss.

        

        
        
          REL
          Data dependency
Thread-private dependency: the accumulator at step i depends on the accumulator at step i-1. This intra-thread ordering relation makes the tile loop a fixed point rather than an independent parallel loop.

          Coalescing relation
addr(lane) = base + lane × stride
The relation between thread lane index and memory address. When stride = 1, all 32 lanes access a contiguous block → single transaction. When stride ≠ 1, transactions multiply.

          Barrier relation
__syncthreads()
Formal write→read relation between threads. The barrier is the implementation of this relation: it guarantees that all writes visible to one thread are visible to all threads after the barrier.

          Combine Group
Relation between two REDUCE dimensions whose outputs feed the same final value (e.g., MLA's ckv + kpe). Forces a REDUCE_MERGE micro-phase. The relation is discovered at Step 1a when comparing input and output shapes.

          Pipeline dependency
cp.async.wait_group(N)
The relation between the issue-load phase and the consume-compute phase. The async wait enforces this relation without a full __syncthreads() barrier.

        

        
        
          SYM
          ILP symmetry
The thread tile's rows and columns are related by translation symmetry — computing element (r,c) is structurally identical to computing (r+1,c). This symmetry enables unrolling: the compiler can issue multiple independent FMAs simultaneously.

          Thread translation symmetry
AREA dimensions are translational symmetries of the computation with respect to the thread index. The computation at lane i is identical in structure to the computation at lane i+1. This symmetry guarantees that all 32 lanes can execute in lockstep without divergence.

          Bank conflict symmetry (harmful)
XOR swizzle breaks it
The modular symmetry of shared memory banking: addresses that differ by a multiple of 32 banks map to the same bank. XOR swizzle or column padding breaks this harmful symmetry. Detecting the symmetry tells you exactly when padding is needed.

          Block translation symmetry
AREA dimensions are also symmetric at the block level. Block (i,j) is structurally identical to block (i+1,j). This symmetry allows the same kernel to handle any block without conditional logic (except at boundary tiles where the symmetry is broken).

          Pipeline period symmetry
Double buffering exploits period-2 temporal symmetry of the tile sequence. The computation is invariant under the transformation "swap ping and pong buffers and advance one tile." This symmetry is what makes the buf = tile & 1 expression sufficient.

        

        
        
          FUN
          Thread tile derivation
The thread's register tile address formula is the functorial image of the block tile address formula. Apply the same affine structure, one level lower: replace BM with TM, replace blockIdx with threadIdx. The functor preserves the structure and reduces the scale.

          Warp primitive selection
The functor maps the Step 1a algebraic classification (semiring / monoid) to the warp-level hardware primitive (shuffle tree / Blelloch). This is the single most important functor in the framework: it converts a mathematical property into a hardware instruction family.

          Smem layout derivation
The shared memory tile's layout is the functorial image of the global memory tile's layout, modified by the bank-conflict-avoidance morphism. The functor preserves the data content while allowing the layout to change.

          Framework derivation
The entire framework is a functor: it maps geometry specifications (objects) and tensor operations (morphisms) to kernel implementations. The "Feeds into →" annotations are the functor's action on the morphisms between steps.

          Version functor
Each framework extension is a functor from the previous state: it preserves all existing correct derivations and extends to new cases by adding new objects and morphisms to the domain.

        

      
    
  

  
    BridgeTable

    
      Property Decomposition Reference — What Builds Each Atom Intersection

      "For each optimization in the 11×11 Atom Intersection Matrix, which specific algebraic property intersections compose it, and what does each one contribute?"

    

  

  
    How this table connects the two matrices. The 6×6 Algebraic Property Intersection Matrix below shows what capabilities individual property pairs produce. The 11×11 Atom Intersection Optimization Matrix above shows what named optimization to apply when two atoms co-occur. This table is the bridge: it decomposes each atom-level optimization into its property-level building blocks. When an atom intersection says "Online Algorithm," this table tells you exactly which property intersections make it work and what happens when individual properties are absent. Read this table to understand why an optimization works. Read the 11×11 matrix to know what to do.
  

  
    
      
        
          Atom intersection
          Optimization name
          Property-level building blocks
          What breaks when a property is absent
        

      
      

        
          SRG × FXPSRG = ASC + IDE + COM + DISFXP = structural

          Online Algorithm

          
            ASC × DIS → Tiling correctness. Distributivity guarantees tile boundaries don't affect the answer. Partial tile results compose correctly through the accumulation. The Tile molecule's mathematical foundation.

            ASC × COM → One-pass shuffle per tile. Commutativity enables the shuffle tree within each tile. Single-pass reduction instead of two-phase Blelloch. Half the barriers per tile iteration.

            IDE × COM → Uniform boundary handling. Identity fills inactive lanes, commutativity makes masked reduction correct. No special-case code for the last tile within the loop body.

            IDE × INV → Correction at tile boundaries. When the running state changes (e.g., new max found), the inverse enables correcting prior accumulations: output *= exp(old_max − new_max). The Online molecule's core mechanism.

          
          
            Without INV: Tiling and per-tile shuffle reduction still work, but online correction is impossible. Must recompute from scratch when state changes.

            Without COM: Tiling works, but each tile requires two-phase Blelloch instead of shuffle. Double the barrier count per tile.

            Without DIS: Tiling produces wrong answers. Cannot split the reduction dimension into tiles at all.

          
        

        
          SRG × PRDSRG = ASC + IDE + COM + DISPRD = structural

          Predicated Reduction

          
            IDE → Safe fill value. Inactive threads contribute the identity element (0.0f for sum, −INF for max). The result is unaffected by padding.

            COM → Order-free masking. The shuffle tree produces the correct result regardless of which specific lanes are active or inactive, because all orderings are equivalent.

          
          
            Without IDE: No safe fill value for inactive lanes. Must use conditional logic per lane, causing warp divergence.

            Without COM: Masking still works but requires the ordered Blelloch sweep. Which specific lanes are inactive affects the sweep structure.

          
        

        
          SRG × ATOSRG = ASC + IDE + COM + DISATO = structural

          Global Reduce

          
            COM → Order-free atomic writes. Blocks arrive at the global accumulator in unpredictable order. Commutativity guarantees the final result is the same regardless of arrival order.

            COM × INV → Abelian group (strongest guarantee). If both hold, the accumulation is both order-free and individually correctable. The strongest possible concurrent write guarantee.

          
          
            Without COM: atomicAdd gives wrong results. Must launch a second kernel with ordered merge instead. This is the MON × ATO "Atomic Invalid" cell.

          
        

        
          SRG × SYMSRG = ASC + IDE + COM + DISSYM = structural

          Free Parallel Tree

          
            ASC × COM → Permutation symmetry, expressed as SYM through FUN. COM is an algebraic property (reordering doesn't change the answer). SYM is a structural pattern (the computation is invariant under transformation). FUN is the functor that maps one to the other. The butterfly shuffle pattern is one specific permutation among all possible tree orderings, and COM guarantees all permutations are equivalent.

          
          
            Without COM: Permutation symmetry breaks. Cost jumps from O(1) to O(log N) per element. This is the MON × SYM "Asymmetry Cost" cell — the exact price of one missing property.

          
        

        
          SRG × RELSRG = ASC + IDE + COM + DISREL = structural

          Combine Group

          
            COM → Ordering dependency eliminated (negative relation). Within a single SRG reduction, COM eliminates the sequential dependency chain output[i] depends on output[i−1] that MON's REL column tracks. No ordering constraint exists between reduction elements.

            Between two SRG reductions: Combine Group relation still applies — the inter-reduction dependency (CG1 must complete before CG2) is determined by the function between them, not by the algebraic properties of the individual reductions.

          
          
            Without COM: Sequential dependency chain exists. MON × REL "Order Relation" — output[i] depends on output[i−1]. The dependency that forces the Blelloch sweep structure.

          
        

        
          MON × FXPMON = ASC + IDEFXP = structural

          Sweep as Fixed Point

          
            ASC → Tree structure. Associativity enables tree-shaped parallelism. The Blelloch sweep is a tree.

            IDE → Clean phase transition. Identity element is written to the root between up-sweep and down-sweep. The gate value that separates the two fixed-point phases.

            Absent COM → Two-phase forced. Without commutativity, a single-pass shuffle is invalid. The ordered two-phase sweep is the mathematical consequence of the broken permutation symmetry.

            Absent INV → No online correction. If the running state changes, prior results cannot be corrected. Must recompute rather than rescale.

          
          
            If COM were added: Upgrades to SRG × FXP "Online Algorithm." Sweep collapses to one-pass shuffle + optional online correction.

          
        

        
          MON × ATOMON = ASC + IDEATO = structural

          Atomic Invalid

          
            Absent COM → Arrival order matters. atomicAdd assumes any write order is correct. For a monoid, different arrival orders give different intermediate states. Cross-block accumulation must use a second kernel launch with ordered merge, not atomic writes. This is a negative constraint — the absence of a property eliminates a hardware option.

          
          
            If COM were added: Upgrades to SRG × ATO "Global Reduce." atomicAdd becomes valid.

          
        

        
          MON × SYMMON = ASC + IDESYM = structural

          Asymmetry Cost

          
            Absent COM → Broken permutation symmetry. SRG has COM → SYM holds → O(1) per element via butterfly shuffle (any permutation valid). MON lacks COM → SYM is broken → O(log N) per element via Blelloch (only one specific ordering is valid). The cost difference is exactly the cost of one missing algebraic property, detected by the SYM atom.

          
          
            If COM were added: Upgrades to SRG × SYM "Free Parallel Tree." Cost drops from O(log N) to O(1).

          
        

        
          PRD × FXPPRD = structuralFXP = structuralproperties inherited from reduction operation

          Predicate Hoisting

          
            IDE × COM → Uniform boundary handling across iterations. Identity fill + order-free evaluation means boundary tiles execute identically to interior tiles within the loop body. Predicates that are constant across iterations are hoisted out of the loop. Only the final store needs boundary checking. Eliminates a branch in the hot loop.

          
          
            Without IDE×COM: Each tile iteration may need boundary-specific logic. The predicate cannot be hoisted. Branch within the hot loop.

          
        

        
          PRD × SYMPRD = structuralSYM = structuralproperties inherited from reduction operation

          Path Specialization

          
            Full tiles have translational symmetry — every tile is structurally identical. The boundary tile breaks this symmetry.

            IDE × COM → Symmetry healed. Identity fill + order-free evaluation makes the boundary tile execute identically to interior tiles. When this holds, no path specialization is needed — uniform execution throughout. When IDE×COM does not hold, generate fast path (full tiles, no predicates) + slow path (boundary tile, with predicates).

          
          
            Without IDE×COM: Symmetry break is permanent. Two code paths required. Runtime selects based on tile position.

          
        

      
    
  

  
    The "absent property" column is a constraint document for AI generation. Each entry describes a specific structural consequence of a missing algebraic property — a GPU capability that is lost, a hardware option that is eliminated, or a cost increase that is unavoidable. When an AI generates a kernel implementation, these entries function as negative constraints: if the combining operation lacks COM, the AI must not generate shuffle trees, must not use atomicAdd for cross-block reduction, and must generate the more expensive Blelloch sweep. Negative constraints are as important as positive ones for correct kernel generation, and they are harder to discover from examples alone because examples show what works, not what would be wrong.
  

  
    Matrix11×11
