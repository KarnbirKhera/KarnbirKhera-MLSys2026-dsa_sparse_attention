# Step 0.5 — Structural Analysis & Level 0 Geometry

What to Write Down
        Where It Fires
      

    
    
      
        FXP Fixed Point
        Is the tile loop bounded (static iteration count) or convergence-based (run until consistent)? Does the framework's validation cycle need to iterate?
        Write: static count = ceil(REDUCE_dim / tile_size). Write YES/NO for online algorithm (does the combining operation carry forward state that must be merged?). If YES, identify the merge operator and its identity element.
        Step 2 (iteration bound → loop unroll decision). Step 3 (online merge → FXP state variable in FSM). Pipeline opt (FXP×FXP → double buffer). Intersection: FXP×SRG → Online Algorithm; FXP×PRD → Predicate Hoisting; FXP×MEA → Iteration Bound; FXP×GATE → Streaming Gate (when a GATE dimension's candidates arrive in batches across tile iterations rather than all at once — see Gate Specification Table).
      

      
        MOR Morphism
        Is this kernel structurally equivalent to a kernel already derived? Are any two index formulas composable into one? Does the smem layout need a structure-preserving transform from the global layout?
        Write: name any known equivalent kernel (e.g., "this is GEMM with different tile sizes"). Write YES/NO for smem layout transform needed. Write all composable affine map pairs (they become FMA instructions in Step 5).
        Step 2 (smem layout transform → MOR×AFM → bank-conflict-free layout). Step 5 (address composition → FMA chains). Intersection: MOR×MEA → Compression Staging; MOR×AFM → Address Composition; MOR×REL → Structural Reuse.
      

      
        MEA Measure
        What are the four key measures of this computation: (1) total data size per tile, (2) expected reuse count, (3) smem budget consumed at the proposed tile sizes, (4) register budget consumed?
        Write: Measure_tile = BM × BK × sizeof(dtype) for each staged tensor. Write: Reuse_count = ∏(AREA dims absent from tensor index) for each buffer. Write: Smem_total = sum of all Block→Shared buffers. Write: Reg_total = sum of all Thread→Register state variables.
        Step 2 budget checks (MEA×REL → Joint Budget Check for Combine Groups). Step 4a reuse validation. Intersection: MEA×SRG → Reduction Level; MEA×FXP → Iteration Bound; MEA×FUN → Budget Transformation.
      

      
        REL Relation
        Are any two REDUCE dimensions related (their outputs feed the same output value)? Are any two phases related by a write→read dependency not visible as a direct smem boundary crossing? Are any memory accesses related by a constant stride difference?
        Write: list all Combine Groups (pairs of REDUCE dims whose partial results must merge before the final output). Write: all write→read thread pairs that cross a phase boundary. Write: for each pair of affine map leaves that access the same buffer — are their addresses stride-related?
        Step 1a (Combine Group identification → REDUCE_MERGE in Step 3). Step 3 FSM (relation between phases → barrier placement). Intersection: REL×SRG → Combine Group; REL×AFM → Coalescing Detection; REL×FUN → Relation Lifting.
      

      
        SYM Symmetry
        Which dimensions are AREA dimensions (translational symmetry — computing at position i is identical to computing at position i+1)? Are any memory access patterns translationally symmetric with respect to the thread lane index (→ coalescing)? Are any patterns symmetric with respect to the bank index (→ harmful bank conflict)?
        Write: all AREA dimensions (already from Level 0 comparison — each one is a translational symmetry). Write YES/NO for coalescing symmetry at the warp level for each buffer. Write YES/NO for bank symmetry (stride is multiple of 32 banks) for each smem buffer. Write the symmetry-breaking action if bank symmetry exists.
        Step 1a (AREA classification is symmetry classification). Step 1b (access pattern symmetry determines coalescing). Step 2 (bank symmetry → XOR swizzle). Intersection: SYM×AFM → Guaranteed Coalescing; SYM×SRG → Free Parallel Tree; SYM×PRD → Path Specialization.
      

      
        FUN Functor
        Can this kernel's binding table be derived from a known kernel by applying the level-lowering functor (BM → TM, blockIdx → threadIdx)? Can the warp primitive choice be read directly from the algebraic classification (SRG → shuffle tree, MON → Blelloch) without additional reasoning?
        Write: for each hierarchy level transition (Grid→Block, Block→Warp, Warp→Thread), write the functor action — what changes (index variable, tile size variable) and what stays the same (structural decisions). Write: confirm that the SRG/MON classification from Level 0 directly determines the warp primitive family — no further analysis needed.
        Step 2 (functor from block to thread tile addresses). Warp primitive selection (FUN×SRG → primitive family). Step 5 (functor produces thread-level address from block-level address by substitution). Intersection: FUN×AFM → Register Tile Lift; FUN×SRG → Hierarchical Reduce; FUN×MEA → Budget Transformation.
      

    
  

  
    GEMM example — Step 0.5 completed:
    FXP: static count = ceil(K/BK), no online merge (plain sum accumulator). MOR: structurally equivalent to tiled GEMM from prior derivation — transport. Smem layout transform needed (row-major global → column-major smem for B to avoid bank conflicts). MEA: Measure_A_tile = BM×BK×4 bytes; Reuse_A = BN (N-dimension absent from A's index); Smem_total = BM×BK×4 + BK×BN×4 bytes — check against 48 KB limit. REL: no Combine Groups; write→read relations at smem boundary (Load_A → Compute → Load_A crossing); A and B addresses stride-related (A: lane stride along K, B: lane stride along N). SYM: M and N are AREA (translational symmetry); B access at warp level has stride-1 along N → guaranteed coalescing; A access has stride=K along M → not coalesced at warp level (requires smem staging). FUN: block→thread functor: replace BM→TM, BN→TN, blockIdx.y→(tid/TN), blockIdx.x→(tid%TN); SRG → __shfl_down_sync confirmed.
  

  
    Feeds into → Step 1a (Combine Groups from REL analysis; SYM confirms AREA classification) · Step 1b (coalescing SYM confirms leaf types; MOR identifies composable address pairs) · Step 2 (MEA provides all four measures for budget checks; FUN provides functor action for tile derivation; SYM identifies smem layout transform needed) · Step 3 (REL provides write→read thread pairs for barrier placement; FXP identifies online merge state variables) · Step 4a/4b (MEA provides reuse counts before the table is computed, as a pre-check) · Step 5 (MOR provides composable map pairs; FUN provides thread-level address derivation rule)
  

  
    atoms activated at Step 0.5

    
      
        FXP

        Active — primary. This is where fixed point structure is identified. The tile loop's static iteration count is computed here. Online algorithm detection happens here. If the fixed point is convergence-based rather than bounded, the framework's scope limitation at Step 1a triggers.

      

      
        MOR

        Active — primary. Structural equivalence to known kernels checked here. Composable affine map pairs identified here. Smem layout morphism planned here. This is the step that enables derivation transport — skipping the full framework for morphically related problems.

      

      
        MEA

        Active — primary. All four key measures computed here as pre-checks: tile data size, expected reuse count, smem total, register total. This turns the Step 2 budget checks from discovery into confirmation.

      

      
        REL

        Active — primary. Combine Groups identified here (the most important output of Step 0.5 for complex kernels like MLA attention). Write→read thread relations identified here. Stride-relation between accesses identified here for coalescing analysis.

      

      
        SYM

        Active — primary. AREA dimensions are named as translational symmetries here, making the symmetry interpretation of Step 1a explicit. Coalescing symmetry and harmful bank symmetry both identified here before any tile size is chosen.

      

      
        FUN

        Active — primary. The functor action at each level transition written down here. Warp primitive family confirmed here directly from algebraic classification. This makes the functor explicit so that every subsequent step knows exactly which structural substitution to apply at each level.

      

      
        AFM SRG MON PRD ATO

        Latent at Step 0.5. The affine map parameters are born at Level 0 and bound at Step 2. The semiring/monoid classification is confirmed at Step 1a. Predicates are classified at Step 1b. Atomics appear at Step 2's Block→Global boundary.

      

    

  

  
    Level0

    
      Geometry Specification — Atom Annotations

      "What tensors in, what tensor out, what operation? What is the combining operator and does the computation have a recursive self-similar structure?"

    

  

  
    Answer three geometry questions: input tensor shapes and storage formats, output tensor shape, and the mathematical operation. The atom annotations below show which of the eleven atoms are activated by each answer. There is also an optional fourth field — the recursion morphism — required only when the computation is self-similar.

  

  
  
    Level 0 · Optional Fourth Field · Recursion Morphism — complete only when the computation is self-similar

    
      Beginner explanation — what self-similar means and why it needs a special field.
      
      Imagine a recipe that says "to make a large cake, first make two medium cakes, then combine them." The recipe for a medium cake is the same as the recipe for a large cake — just smaller. And the recipe for a small cake is the same as the medium cake — just smaller still. This is self-similarity: the problem at each level is a scaled-down copy of the problem at the level above.
      
      Recursive FFT works this way. An FFT over 1024 elements is built from two FFTs over 512 elements. A recursive merge sort of N elements sorts two halves of N/2 each. The geometry of the computation — how many elements, what strides, what tile sizes — is different at every depth level, but always related to the previous level by the same rule.
      
      The standard three geometry questions at Level 0 assume one fixed geometry for the whole computation. When the geometry changes at every depth level, you cannot answer "what are the input shapes?" with a single number. You need a fourth field: the rule that says how the geometry shrinks at each depth. That rule is the recursion morphism.
    

    
      Why this field exists
      Without the recursion morphism field, the framework produces a derivation for the first depth level and stops — it has no way to know that the sub-problems have different geometries. The recursion morphism field tells the framework to derive a family of related kernels, one per depth level, connected by the scaling rule. Each kernel in the family is derived using the standard Level 0 → Code Derivation sequence with a fresh geometry spec produced by applying the morphism rule to the previous level's geometry.
    

    
      
        
          Field
          Question to Answer
          Default
          Atoms Activated
        

      
      
        
          Self-similar?
          Does solving this problem require solving a smaller version of the same problem? If NO, skip all other rows and proceed to the standard three questions.
          NO — most kernels are not recursive. Iterative FFT and iterative merge sort are NOT self-similar — see the Boundary Map for the distinction.
          FXP — are we iterating to depth, or iterating to tile count?
        

        
          Base case geometry
          What is the geometry of the smallest subproblem that doesn't need to recurse further? Write a mini Level 0 spec for it — input shapes, output shape, combining operation. This is the geometry the innermost kernel will use.
          —
          MOR — the base case is the fixed point of the morphism (the smallest geometry the morphism maps to itself).
        

        
          Morphism rule
          How does the geometry transform from depth k to depth k+1? Write the rule explicitly: "halve the sequence length dimension," "quarter the spatial dimensions," "remove the outermost dimension." This is the single rule applied at every depth level.
          —
          MOR primary — the morphism is the fundamental object. SYM — self-similar algorithms have a scale symmetry; each depth is a scaled copy of the level above. FUN — the functor maps depth-k's binding table to depth-(k+1)'s binding table using this rule.
        

        
          Recursion depth
          How many times is the morphism applied before reaching the base case? For divide-and-conquer algorithms this is typically log₂N, where N is the problem dimension that is being halved. Write the formula, not just a number — the framework uses this to count kernel launches.
          —
          FXP — the recursion is a bounded fixed point over depth levels. MEA — total work = sum of all depth levels' work measures. For divide-and-conquer with constant work per level, this is a geometric series.
        

        
          Morphism preserves combining op?
          Does the same semiring or monoid operation apply at every depth level? For most divide-and-conquer algorithms the answer is YES — the butterfly/merge/reduction is the same at every level, just on different-sized data. If NO, the combining operation itself changes with depth and must be specified separately per level.
          YES — the vast majority of recursive GPU algorithms use the same combining operation at every depth.
          SRG or MON confirmed as invariant across depths. SYM — the combining op's invariance is the algebraic expression of the scale symmetry.
        

      
    

    
      Iterative FFT example — why it does NOT need this field: A bottom-up iterative Cooley-Tukey FFT has log₂N rounds. Every round processes all N elements. Every round uses the same butterfly structure with the same affine addressing. The geometry is identical at every round — only the stride parameter changes by a factor of 2. This stride change is captured by the standard affine map coefficients in Level 0, not by a recursion morphism. The geometry spec is fixed; the affine map's coefficient changes. This is exactly a standard bounded FXP tile loop with a stride that doubles each iteration. It fits within the existing framework without the recursion morphism field.
      
      Recursive FFT example — why it DOES need this field: A top-down recursive FFT over 1024 elements creates two subproblems over 512 elements each. Each 512-element subproblem creates two 256-element subproblems. The geometry — the input shape — halves at every depth. A single Level 0 geometry spec cannot be written for the whole computation because there is no single "input shape." The recursion morphism field captures this: base case = 2 elements, morphism rule = halve the input length, recursion depth = log₂(1024) = 10.
