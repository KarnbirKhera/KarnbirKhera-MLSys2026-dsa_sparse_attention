# Geometric Determinism Boundary & Future Extensions

The Geometric Determinism Assumption — the foundation all 11 atoms share
    Every derivable kernel satisfies one foundational condition.

    The entire framework — Level 0 through Code Derivation — rests on a single assumption that all 11 atoms share: the structure of the computation is determined entirely by the geometry of the problem, not by the values of the data being processed. Call this the Geometric Determinism Assumption (GDA). The six algebraic properties beneath the atoms (closure, associativity, identity, commutativity, inverses, distributivity) are themselves geometric in nature — they are properties of the combining operation's structure, not of the data being combined. Whether addition is associative does not depend on which numbers you are adding. Whether a matrix multiply distributes over accumulation does not depend on the matrix entries. The GDA holds whenever the algebraic properties of the operations and the shapes of the tensors together determine the complete kernel structure.

    When the GDA holds, the derivation sequence is guaranteed to produce a complete, correct kernel structure before a single line of code is written. You can classify every dimension's fate, assign every atom to its hierarchy level, place every barrier, and derive every address formula — all without knowing anything about the actual values in the input tensors.

    When the GDA fails — when any aspect of the program's structure depends on runtime data values — the derivation cannot be completed at Level 0, and the rest of the framework cannot proceed. The GDA failing at Level 0 means Step 1a cannot classify output dimensions (the output shape is not determinable from input shapes alone), which means the binding table cannot be filled, which means the FSM cannot be constructed, which means there is no derivation to complete.

    The 11 atoms and the 6 algebraic properties beneath them make this boundary map precise for the first time. Previously, "top-K is excluded" was an empirical observation. With the algebraic properties and extended atoms, each exclusion traces to a specific property or atom whose current definition is insufficient and a specific mathematical rule that would need to be added to cover it. The boundary is not a list of failures — it is a map of the next layer of the framework's evolution.

  

  

    
    
      
        
          Boundary 1 · REL

        

        
          Data-Value Relation

          The current REL atom captures geometric relations. It cannot capture relations whose members depend on runtime data values.

        

      

      
        The relation atom in its current form captures three kinds of geometric relations: the stride relationship between thread indices and memory addresses (coalescing), the convergence relationship between two REDUCE dimensions (Combine Group), and the write-read dependency between phases (barrier). Every one of these relations is computable from the geometry specification before the kernel runs. You can evaluate each one at derivation time using only the tensor shapes and the binding table.

        A data-value relation is different in a precise and fatal way: it is a relation whose members are not determined by geometry but by the values in the input data. "These are the top-K elements" is a data-value relation between the input tensor and the output — which elements are related depends on their values relative to one another, and this cannot be known until the kernel runs. "This token activates this expert" is a data-value relation in mixture-of-experts routing — which expert receives which token depends on routing weight magnitudes. "These nonzero positions exist in this sparse row" is a data-value relation in CSR matrix storage — which columns are populated depends on the matrix's numerical content.

        When a data-value REL appears, it breaks the GDA at Level 0 in a specific way: the output shape becomes data-dependent. Top-K of N elements has K outputs, but which K elements those are requires examining the data. The output shape is no longer determined by the input shapes alone. Step 1a cannot classify output dimensions as AREA or REDUCE because the output tensor's dimension labels are conditional on runtime values. Without Step 1a's classification, the binding table cannot be filled. Without the binding table, the FSM cannot be constructed. The entire derivation collapses at its first step.

        Note the important distinction: the predicate atom handles a related but fundamentally different situation. A predicate is a geometric condition — "if this thread's global coordinate exceeds the tensor bound, use the identity element." The structure of the computation doesn't change; only whether a particular value is used. Data-value REL means the structure of the computation — which phases execute, how many output elements exist, which threads do useful work — is determined by data values, not geometry. PRD is a geometric gate. Data-value REL is a structural dependency on data.

        
          
            Excluded kernel families — true data-value structure only
            
              
                Beam search — the set of surviving beams changes based on score values each step; the active beam set is not bounded by a geometric maximum

                Sparse format conversion (dense → CSR) — nonzero positions are determined by data values, not by shape; the output structure is fully data-driven

                Adaptive histogram equalization — bin boundaries depend on value distributions discovered at runtime

                BFS active node sets — which nodes are active each wave depends on graph edge values, not graph shape alone

                Any operation where the output tensor shape is not determinable from input shapes alone, even with an upper bound

              
              Note — now covered by GATE: top-K, argmax, argmin, stream compaction, and mixture-of-experts dispatch are handled by the GATE dimension classification in Step 1a. These have bounded maximum output sizes determinable from geometry and are now fully derivable. See the Step 1a section.

            

          

          
            What remains for future extension
            
              The GATE extension (Step 1a) handled the bounded-output-size cases. What remains is the truly unbounded case: computations where even the maximum output size is not determinable from input shapes. For these, the geometry spec would need a new output shape model — not "exactly K" or "at most K" but "unknown until runtime." This requires extending Level 0's output shape grammar to express dynamic upper bounds, and extending the binding table and FSM to handle variable-length output buffers with runtime-determined termination.
            

          

        

      

    

    
    
      
        
          Boundary 2 · AFM

        

        
          Non-Affine Address Function

          The chain depth extension (Step 1b) now covers multi-level pointer indirection where each hop is an affine function of the retrieved value. What remains excluded is address computation involving non-affine functions — modular arithmetic, bitwise operations, or quadratic probe sequences.

        

      

      
        The chain depth extension added to Step 1b now handles multi-level pointer indirection where each hop's address is an affine function of the retrieved value. CSR SpMV (row_ptr → col_idx → values, a depth-3 chain where each hop is a direct index), graph GNNs with a fixed CSR adjacency structure, COO sparse operations with affine index chains, and similar depth-2+ structures are now derivable. The chain leaf at each hop records the hop address formula (AFM of retrieved value), the intermediate staging decision, the coalescing analysis, and any null/bounds predicate. Each hop adds one sequential FSM micro-phase and one barrier.

        What remains excluded is address computation where the function relating a retrieved value to the next address is not affine. An affine function has the form a × x + b — a constant multiplier and a constant offset. Hash functions that compute key mod N, XOR-based mixing schemes (address = key XOR salt), and quadratic probing sequences (address = base + probe × probe) are not affine functions of their inputs. They cannot be expressed as chain leaves regardless of depth, because the chain leaf's foundational requirement is that each hop's address formula is an affine map of the previous hop's result. When the map is non-affine, no depth of chain can represent it.

        The practical test at Step 1b is: can you write each hop's address as (retrieved_value × some_constant + some_constant)? If yes at every hop, the chain leaf handles it at the appropriate depth. If any hop requires mod, XOR, squaring, or any other non-linear operation, the framework reaches its boundary at that hop.

        
          
            Excluded kernel families — non-affine functions only
            
              
                Hash tables with modular hashing (key mod N) — the bucket address formula is not affine in the key value

                Cuckoo hashing, Robin Hood hashing — involve multiple non-affine hash functions

                Quadratic probing — probe sequence is a × probe² + b, which is quadratic not linear

                XOR-based hash mixing — bitwise XOR is not an affine map

                Any structure where the address function at any hop is not expressible as constant × retrieved_value + constant

              
              Now covered by chain depth (Step 1b): CSR/BSR SpMV, graph GNNs with fixed adjacency, COO formats with affine index chains, and all depth-2+ pointer chains where each hop is affine.

            

          

          
            What remains for future extension
            
              Covering non-affine address functions requires a non-affine function leaf — a new leaf type representing an address computed by an arbitrary function of a retrieved value. The optimization signal matrix currently assumes all addresses are affine maps, making coalescing, staging, and vectorization analysis tractable via the symmetry and measure atoms. A non-affine leaf would require a new statistical analysis path: estimate the distribution of addresses produced by the function, infer expected coalescing quality from that distribution, and determine staging from access pattern statistics rather than geometric reuse counts.
            

          

        

      

    

    
    
      
        
          Boundary 3 · FXP

        

        
          Data-Dependent Recursion Depth

          The recursion morphism field (Level 0) now covers self-similar computations where the recursion depth is a known function of the input size. What remains excluded is recursion whose depth depends on data values discovered at runtime.

        

      

      
        The recursion morphism field added to Level 0 covers self-similar computations where the recursion depth is a known function of the input size. When a computation is self-similar, you write the base case geometry (the smallest subproblem), the morphism rule (how geometry transforms at each depth level — halve the input, quarter the spatial dimensions, etc.), and the recursion depth formula (typically log₂N or log₂(largest dimension)). The framework then derives a family of kernels — one per depth level — where each kernel's geometry spec is produced by applying the morphism to the previous level's spec. Top-down recursive FFT, top-down recursive merge sort, and blocked recursive matrix factorizations all fit this model because their recursion depth is log₂(input size), which is fully determined by the input shape.

        What remains excluded is recursion whose depth depends on data values rather than input shapes. A BVH tree for ray tracing needs to be deep when the scene is spatially complex and shallow when it is simple — the depth depends on how the geometry of the scene clusters, which is a property of the data values. An adaptive mesh refinement algorithm refines until error estimates drop below a threshold — how many refinement levels are needed depends on the function being approximated, not on the input mesh size. In both cases, even with the recursion morphism, there is no formula expressible from the Level 0 geometry spec that gives the recursion depth. The depth is a runtime variable whose value depends on what the algorithm discovers as it runs.

        
          
            Excluded kernel families — data-dependent depth only
            
              
                BVH / kd-tree traversal — traversal depth depends on scene geometry values, not scene bounding box size

                Adaptive mesh refinement — refinement depth depends on error estimates computed from data values

                Convergence-based iterative solvers — run until residual < ε, where ε termination is data-dependent

                Dynamic programming with variable-length traces — trace length depends on input values

                Wavefront algorithms with data-driven termination — wavefront depth depends on what values are encountered

                Any recursive algorithm where the termination condition depends on data values rather than a formula over input shape

              
              Now covered by recursion morphism (Level 0): top-down recursive FFT, top-down recursive merge sort, blocked recursive matrix factorizations (LU, Cholesky, Strassen), and all divide-and-conquer algorithms where recursion depth = f(input size).

            

          

          
            What remains for future extension
            
              Covering data-dependent recursion depth requires expressing the termination condition as a runtime check rather than a static formula. The recursion morphism field currently requires a static depth formula computable from the geometry spec. Extending it to handle convergence-based termination would require a new field type: a convergence predicate — a condition evaluated against intermediate computation results that determines whether another depth level is needed. This merges with Boundary 1 (data-value REL) at the termination point: the decision to launch another kernel level is a data-value relation on the intermediate results.
            

          

        

      

    

  

  
  
    Map

    
      Complete Boundary Map — Every Major GPU Kernel Family

      "Does each kernel family fit the framework, and if not, which boundary type stops it at which step?"

    

  

  
    Green rows fit the framework completely and can be fully derived using the 11 atoms. Non-green rows fail at a specific step for a specific reason tied to one of the three boundary types. The "Breaks At" column names the earliest framework step where the derivation becomes impossible — this is where you would detect the boundary if you attempted to apply the framework to a kernel of that family.
  

  
    
      
        
          Kernel Family
          Boundary
          Broken Assumption
          Breaks At
          Why — In Atom Language
        

      
      
        
          Vector add, transpose, elementwise ops
          FITS
          —
          —
          Pure AREA, no REDUCE, all addresses arithmetic AFM leaves. Trivially within GDA.
        

        
          GEMM, batched GEMM
          FITS
          —
          —
          K dimension is SRG REDUCE with arithmetic AFM. M, N are AREA with translational SYM. All geometry is static.
        

        
          Dense attention (FlashAttention-style)
          FITS
          —
          —
          Two SRG reductions (dot product + softmax merge), all AFM arithmetic leaves, FXP tile loop with online state. Fully geometric.
        

        
          Paged KV / MLA attention
          FITS
          —
          —
          Extends dense attention with one lookup AFM leaf (depth-1 page table). The REL Combine Group for ckv + kpe handled by REDUCE_MERGE. Within GDA.
        

        
          Sparse attention (this competition)
          FITS
          —
          —
          Attention with a lookup leaf for the sparse index. The sparse pattern is provided as a pre-computed index tensor — its structure is geometric at kernel launch time.
        

        
          Shared reduced sum, layer norm, RMS norm
          FITS
          —
          —
          SRG reductions (sum, sum-of-squares) over the feature dimension. All AFM arithmetic. Fully geometric.
        

        
          Convolution (standard, depthwise, dilated)
          FITS
          —
          —
          Mathematically equivalent to GEMM with a richer AFM (kernel offset + spatial offset). All addresses are arithmetic leaves. Dilation changes AFM stride coefficient only.
        

        
          Iterative FFT (bottom-up Cooley-Tukey)
          FITS
          —
          —
          log₂N rounds, each round applies the same butterfly structure to all N elements. FXP bounded iteration, AFM arithmetic within each round. The iterative reorganization makes the geometry constant across rounds.
        

        
          Bottom-up merge sort (iterative)
          FITS
          —
          —
          Fixed log₂N rounds, each round merges segments of the same size with arithmetic AFM. Identical structure at every round iteration. FXP bounded fixed point.
        

        
          Embedding lookup, gather (depth-1)
          FITS
          —
          —
          One lookup AFM leaf per access. Token index is a geometric input — its position in the index tensor is geometrically determined, even if its value is data-dependent. Depth exactly 1.
        

        
          Prefix scan / Blelloch (within a block)
          FITS
          —
          —
          MON REDUCE via Blelloch sweep. 2×log₂N phases with symmetric up/down structure. All within one block — no cross-block REL required. Fully within GDA.
        

        
          Multi-block prefix scan (full array)
          Boundary 3 · FXP
          Requires two kernel launches with different geometry at each launch
          Level 0 (single geometry spec insufficient for multi-kernel coordination)
          Block-level Blelloch fits. Combining block-level prefix sums requires a second kernel with a different geometry spec (the inter-block partial sum array). Two different geometries must be coordinated. Outside single-kernel model even with recursion morphism, because the two kernels are not self-similar — they have structurally different phase sequences.
        

        
          Top-K selection, argmax, argmin
          FITS — GATE
          —
          —
          Handled by GATE dimension classification (Step 1a). K is the bounded maximum output size, determinable from geometry. Gate criterion: top-K by magnitude (MON criterion, requires sort/rank in Phase 1). Phase 2: MON prefix sum assigns output positions. Argmax/argmin are GATE with K=1.
        

        
          Beam search
          Boundary 1 · REL
          Active beam set changes based on score values each step
          Level 0 (geometry changes each step, set size is unbounded)
          Unlike top-K, beam search changes its active beam set at every decoding step based on score values. The set of active beams is not bounded by a static K determinable from geometry — it is the result of a data-value selection at each step. GDA fails at Level 0 of every beam search step.
        

        
          Stream compaction / filter
          FITS — GATE
          —
          —
          Handled by GATE dimension classification (Step 1a). Maximum output size = N (worst case: all elements survive). Gate criterion: threshold predicate, independent per element (SRG criterion, Phase 1 is fully parallel). Phase 2: MON exclusive prefix sum assigns compact output positions.
        

        
          Mixture-of-experts dispatch
          FITS — GATE
          —
          —
          Handled by GATE dimension classification (Step 1a). Each token selects top-K experts; K is fixed and geometry-determined. Gate criterion: top-K routing weights per token (MON criterion). Phase 2: MON prefix sum assigns tokens to expert slots. Expert capacity buffer is sized to maximum = tokens × K.
        

        
          CSR / CSC SpMV
          FITS — Chain
          —
          —
          Handled by chain depth leaf (Step 1b). Depth-3 affine chain: hop 1 = row_ptr[row_id] (arithmetic address), hop 2 = col_idx[row_start + k] (address is row_start + k, affine in retrieved value + iterator), hop 3 = val[row_start + k] (same address structure). Each hop adds one FSM sequential micro-phase and one barrier. Coalescing analysis required per hop.
        

        
          Graph GNN with fixed CSR adjacency
          FITS — Chain
          —
          —
          Handled by chain depth leaf (Step 1b). Fixed adjacency stored in CSR format: depth-2 chain into adjacency structure (row_ptr → col_idx), then arithmetic access to node features. The adjacency structure is geometry-determined at kernel launch — only the values of the node features change at runtime. SRG aggregation over neighbors (sum/mean message passing).
        

        
          Hash table lookup / insertion
          Boundary 2 · AFM
          Hash function is not an affine map
          Step 1b (hash function is not expressible as any chain leaf)
          Hash(key) = key mod N is not of the form a × key + b (modular arithmetic is not affine). The chain leaf extension handles depth-n affine chains but does not extend the class of address functions — only their depth. A non-affine hop at any depth in the chain still breaks the framework at that hop. Open addressing collision chains compound this with variable depth.
        

        
          BVH / kd-tree traversal
          Boundary 1 + 2
          Traversal path is data-driven; depth is data-dependent
          Level 0 (traversal depth and path are both data-dependent)
          Two boundaries simultaneously: which child node to visit next depends on intersection test values (data-value REL — not coverable by GATE because traversal structure is not a bounded selection), and traversal depth depends on scene geometry values (data-dependent depth — not coverable by recursion morphism because depth has no formula over input size).
        

        
          Top-down recursive FFT
          FITS — Recursion
          —
          —
          Handled by recursion morphism field (Level 0). Base case: 2-element butterfly. Morphism rule: halve input length. Recursion depth: log₂N. Combining operation (butterfly multiply-add) preserved across all depths. Derives a family of log₂N kernels coordinated by the recursion morphism.
        

        
          Top-down recursive merge sort
          FITS — Recursion
          —
          —
          Handled by recursion morphism field (Level 0). Base case: 2-element merge. Morphism rule: halve segment size. Recursion depth: log₂N. MON combining operation (ordered merge) preserved across depths. Each depth level's kernel is fully derivable once the morphism produces its geometry spec.
        

        
          Adaptive mesh refinement
          Boundary 1 + 3
          Refinement criterion is data-value dependent; refinement depth is data-dependent
          Level 0 (geometry is neither static nor geometrically determinable)
          Two boundaries remain: the refinement criterion (which cells to refine) is a data-value REL not coverable by GATE (the active cell set is not bounded by a static geometry maximum), and refinement depth depends on error estimates from the current solution (data-dependent depth, not coverable by recursion morphism). Both boundaries must be resolved independently.
        

      
    
  

  
  
    On whether the FSM can be extended to cover these boundaries
    The FSM is a temporal sequencer: it arranges atoms in time based on write-read relations identified in the binding table. It lives downstream of Level 0, Step 0.5, Step 1a, and Step 1b — all of which must complete before the FSM can be built. This means extending the FSM can only cover boundaries that do not break the earlier steps. Boundaries 2 and 3, which fail at Level 0 or Step 1b, cannot be fixed by FSM extensions alone because the FSM's inputs — the binding table, the classified dimensions, the typed address leaves — do not yet exist when the boundary is encountered.

    For the cases that do break at the FSM level — specifically, the aspects of Boundary 3 that affect a single kernel's phase structure — FSM extensions are the right fix. Adding a recursion functor construct to the FSM would allow it to express "apply this FSM at depth k, then apply a transformed version of the FSM at depth k+1." Adding a sequential-order transition type to the FSM would allow it to express the inter-block ordering required for full-array prefix scan (currently excluded because the cross-block REL has no FSM representation). These are meaningful extensions that expand the framework's coverage without requiring changes to the foundational GDA model.

    For Boundary 1 (data-value REL) and the non-tree AFM aspect of Boundary 2, the FSM cannot be extended in isolation because the problem begins before the FSM is reachable. These require changes to the geometry spec model (Level 0), the leaf type system (Step 1b), and the binding table (Step 2) — all of which feed the FSM as inputs. The FSM is the last step to update, not the first. The correct extension order is: first extend Level 0 to express variable-length output shapes and chain address structures, then extend Step 1b to include chain leaves, then extend Step 2 to budget chain traversal resources, then extend the FSM to sequence the new phase types that chain traversal requires.

  

  Complete Derivation Sequence

  
    Level 0Geometry Spec

    
      Answer the three standard geometry questions: input tensor shapes and storage formats (gives AFM coefficients), output tensor shape (pre-identifies AREA/REDUCE/GATE candidates and MEA of the problem), and the mathematical operation (names the SRG/MON combining operator and the FXP iteration structure). Then answer the optional fourth question: is this computation self-similar (recursive)? If YES, fill the Recursion Morphism Table — base case geometry (MOR base), morphism rule (MOR: how geometry transforms each depth level), recursion depth (FXP count), and whether the combining operation is preserved across depths (SRG/MON invariance = SYM scale symmetry). If NO, skip the fourth question and proceed.

      → tensor shapes · combining operator · storage strides · candidate symmetries · candidate GATE dims · candidate relations · recursion morphism (if self-similar)

      AFMSRGMONFXPMORMEARELSYMFUN

    

  

  
    Step 0.5Structural Analysis

    
      Run all six extended atoms against the geometry spec before any binding decision. Identify: fixed point structure and iteration bound (FXP); morphic equivalents and composable address pairs (MOR); four resource measures (MEA); Combine Groups and write→read relations (REL); translational and bank symmetries (SYM); functor action at each level transition (FUN). All outputs are pre-checks — they turn subsequent budget checks and optimization lookups from discovery into confirmation.

      → iteration bound · morphic equivalents · four measures · Combine Groups · symmetry map · functor action table

      FXPMORMEARELSYMFUN

    

  

  
    Step 1aDimension Fate

    
      Classify each dimension as AREA (survives independently — a translational symmetry), REDUCE via SRG (collapses commutatively), REDUCE via MON (collapses in order), or GATE (elements compete for survival based on values — requires Gate Specification Table). For every GATE dimension: fill the Gate Specification Table to derive Phase 1 (criterion evaluation, SRG or MON) and Phase 2 (prefix sum for output index assignment, always MON), the mandatory REL barrier between them, and the maximum output buffer size (MEA). Then check FXP×GATE: if the GATE dimension's candidates are produced inside the tile loop rather than all at once, mark this as Streaming Gate and pre-wire the six decisions listed in the Streaming Gate section (accumulation buffer, staging scratch, two-case merge, staging sort, buffer re-sort, deferred output write). Identify all Combine Groups (REL between REDUCE dims). Confirm FUN functor: SRG → shuffle tree, MON → Blelloch.

      → AREA/REDUCE/GATE table · SRG/MON classification · Gate Specification Table per GATE dim · Combine Groups · symmetry confirmation · functor pre-wiring

      SRGMONPRDSYMRELMEAFXPFUN

    

  

  
    Step 1bAccess Patterns

    
      Classify every memory access by leaf type and chain depth. Depth 0 = arithmetic (pure AFM). Depth 1 = lookup (existing — one pointer hop, e.g. paged KV). Depth 2+ = chain leaf (new — each additional hop adds one sequential FSM micro-phase, one barrier, one intermediate staging decision). For each hop beyond depth 1: write the hop address formula (AFM applied to previous hop's value), the intermediate staging decision (registers or smem via MEA), the coalescing analysis for this hop (SYM — may be absent for depth 2+ since addresses come from data values), and any null/bounds predicate (PRD). Also: check coalescing symmetry for all arithmetic leaves (SYM×AFM), identify composable address pairs (MOR×AFM), plan smem layout morphism if bank symmetry detected (SYM×MOR).

      → leaf type table with chain depth annotations · per-hop staging decisions · coalescing analysis per hop · composable map pairs · smem layout morphism · predicate expressions

      AFMPRDSYMMORRELFXPMEA

    

  

  
    Step 2Binding Table

    
      Assign each atom to its hierarchy level. Apply functor at each level transition (FUN). Verify all four measures (MEA): smem budget, register budget, reuse counts, occupancy. Check joint budget for Combine Group members (MEA×REL). Apply smem layout morphism (MOR). Enter iteration bound in REDUCE granularity (FXP). Add Relations column: coalescing marks, barrier requirements, Combine Group dependencies (REL).

      → granularity table · warp primitive family · smem layout · four measures verified · Relations column · functor derivation rules

      AFMSRGMONPRDATOFXPMORMEARELSYMFUN

    

  

  
    Step 3FSM

    
      Phase ordering = formal expression of all write→read relations (REL). Barrier rule: __syncthreads() at every REL boundary. Insert REDUCE_MERGE micro-phase for every Combine Group. Insert GATE two-phase pattern for every GATE dimension: Phase 1 (criterion evaluation, SRG or MON) → mandatory REL barrier → Phase 2 (Blelloch prefix sum, always MON) → barrier → output write. First check FXP×GATE: if the GATE dimension's candidates arrive inside the tile loop, use the Streaming Gate variant instead — defer output write to after the loop, add a per-tile merge phase with Case A/B logic, add a post-merge barrier for the threshold REL, and pre-wire sorted-survivor replacement in d1.md. Insert chain hop micro-phases for every depth->1 lookup: one sequential read phase per hop, each followed by a barrier. Mark tile loop as bounded fixed point (FXP) with state variables listed. Verify FSM is the functorial image of the binding table's column structure (FUN). Apply Predicate Hoisting (FXP×PRD). Apply Sweep Sharing (FXP×SYM) for Blelloch phases.

      → ordered phase sequence · barriers · REDUCE_MERGE · GATE two-phase pattern · chain hop phases · fixed point annotation · state variable lifetimes · three cross-checks

      PRDSRGMONFXPRELSYMAFMFUN

    

  

  
    Steps 4a+4bLifetime Tables

    
      Formalize reuse counts as explicit measure computations (MEA): Reuse = ∏(absent AREA dims). Validate fixed point state variables survive all tile loop phase boundaries (FXP). Apply Budget Transformation functor: thread-level register measure constrains block-level smem measure (FUN×MEA). Verify functor consistency if any Step 2 revision is required: thread tile must remain the functorial image of the block tile (FUN).

      → block-level reuse counts (formal measures) · thread-level reuse counts · state variable lifetime verification · validated tile shapes · any Step 2 revisions with functor consistency check

      MEAFXPSYMFUN

    

  

  
    Step 5Affine Map Composition

    
      Compose all affine maps into address trees. Apply all morphisms: composable pairs → FMA chains (MOR×AFM); smem layout morphism → bank-conflict-free addresses (MOR×AFM); functor substitution → thread-level addresses from block-level addresses (FUN×AFM). Compute max addresses as measure verification (MEA). Upward pass = test for additional parallelism symmetry (SYM); downward pass = test for additional ILP symmetry (SYM).

      → address tree per buffer · max address checks · FMA compositions · functor-derived thread addresses · upward/downward parallelism opportunities

      AFMMORMEASYMFUN

    

  

  
    Code Derivation

    
      Four sub-tables: Symbol legend (AFM coefficients from Step 2 granularity). Declaration table (FXP state variables, MEA-verified smem buffers). Index expression table (AFM leaves composed by MOR, PRD predicates alongside addresses). Phase-ordered skeleton (FSM rows top-to-bottom, REL barriers explicit, REDUCE_MERGE phases included). Every line traces to exactly one atom at exactly one level — verify before committing to code.

      → symbol legend · declaration table · index expression table with MOR compositions and PRD predicates · phase-ordered code skeleton with all 11 atoms traceable per line

      AFMSRGMONPRDATOFXPMORMEARELSYMFUN

    

  

  
    The framework answers "what structure should this kernel have?" by deriving each decision from eleven atoms built on a foundation of six algebraic properties. The original five atoms — affine map, semiring, monoid, predicate, and atomic op — describe what operations exist at each level of the execution hierarchy, with the semiring and monoid each composed from specific combinations of the algebraic properties (closure, associativity, identity, commutativity, inverses, distributivity) that determine their GPU consequences. The extended six — fixed point, morphism, measure, relation, symmetry, and functor — describe the global structure of those operations: how they iterate, how they transform, how they are sized, how they depend on each other, what invariances they have, and how they map between levels. The Atom Intersection Optimization Matrix names the optimization that emerges when any two atoms co-occur at the same step and hierarchy level — the intersection is almost always stronger than either atom considered alone. The Framework Boundaries section names the precise mathematical reason each excluded kernel family falls outside scope and what property would need to be added to cover it. The three boundary types — Data-Value REL, Non-Affine Address, and Data-Dependent Recursion Depth — define the mathematical perimeter of regular parallel computation, and the extension order follows directly from their structure.
  

  
    Scope — defined precisely by the Geometric Determinism Assumption
    This framework derives kernels where the Geometric Determinism Assumption holds: the structure of the computation — which phases execute, in what order, accessing which addresses — is determined entirely by the geometry of the problem, not by the values of the data being processed. Target kernels confirmed within scope: vector add, transpose, elementwise ops, GEMM, convolution, layer norm, RMS norm, iterative FFT (bottom-up), bottom-up merge sort, embedding lookup, dense attention, paged KV MLA attention, sparse attention, block-level prefix scan, stream compaction (GATE), top-K / argmax / argmin (GATE), mixture-of-experts dispatch (GATE), CSR / BSR SpMV (chain depth), graph GNN with fixed adjacency (chain depth), top-down recursive FFT (recursion morphism), top-down recursive merge sort (recursion morphism), blocked recursive matrix factorizations (recursion morphism).

    Three precise boundary types exclude kernels where the GDA fails. Boundary 1 (Data-Value REL): operations where the output shape or active set is not bounded by any formula over input shapes — beam search, truly dynamic output buffers, BFS with data-driven active node sets, adaptive histogram equalization. Note that bounded-output-size selection (top-K, stream compaction, MoE dispatch) is now covered by GATE. Boundary 2 (Non-Affine Address): address computations where any hop uses a non-affine function (mod, XOR, quadratic) of the retrieved value — hash tables with modular hashing, cuckoo hashing, quadratic probing. Note that depth-2+ affine chains (CSR SpMV, graph GNNs) are now covered by chain depth. Boundary 3 (Data-Dependent Recursion Depth): recursive algorithms whose depth depends on data values rather than input size — BVH traversal, adaptive mesh refinement, convergence-based iterative solvers. Note that geometry-bounded recursion (top-down FFT, merge sort) is now covered by the recursion morphism field.

    What remains outside scope all share one property: the framework needs a piece of structural information that cannot be determined from the geometry specification before execution. For Boundary 1 this is the active output set. For Boundary 2 this is the address produced by a non-affine function. For Boundary 3 this is the termination depth determined by a data-driven convergence condition. Each one requires a different kind of extension to the foundational model — not to the FSM, which is downstream of all three failure points.

  

  
    For Readers Who Want the Full Picture

    Why These Atoms Appear Everywhere —The Foundation Beyond GPU Computing

    The condensed version earlier gave you the signal. This section gives you the full argument — why these specific atoms and the algebraic properties beneath them appear in fields that have nothing to do with GPUs, and why the act of naming them matters beyond the domain where they were named.

  

  

    The eleven atoms in this framework — and the six algebraic properties from which the semiring and monoid atoms are composed — are not GPU-specific inventions. They are descriptions of mathematical constraints that arise wherever many independent agents cooperate to produce a collective result — and GPU kernels happen to be one of the cleanest and most controlled environments in which to observe them. The GPU's execution model, with its rigid hierarchy of threads, warps, blocks, and grids, creates a laboratory where the constraints that force each atom's appearance are unusually visible. But the atoms themselves are older than GPUs, older than computers, and older than the fields of mathematics that formally named them.

    The Affine Map: The Universal Coordinate Bridge

    The affine map — coord × stride + offset — appears in GPU kernels as the formula every thread uses to find its data. But the same formula appears in neural network layers as weight × input + bias, in signal processing as sample × rate + offset, in the Navier-Stokes equations when you discretize a fluid simulation domain onto a computational grid, and in special relativity as the Lorentz transformation between inertial reference frames. These are not superficial resemblances. They are the same mathematical object appearing because they all face the same underlying constraint: how do you bridge two coordinate systems while preserving the linear relationships between coordinates?

    The affine map is the unique simplest answer to that question. It is the simplest transformation that can map any position in one coordinate system to a corresponding position in another while guaranteeing that straight lines remain straight and parallel lines remain parallel. No simpler structure exists that satisfies this constraint. No more complex structure is needed for all regular transformations. This is why it appears everywhere the constraint appears — which is extremely often, because coordinate bridging under linear constraint is a requirement that shows up in virtually every domain of science and engineering. A neural network layer bridging activation space to weight space, a GPU thread bridging execution hierarchy coordinates to memory addresses, a fluid simulation bridging continuous space to a discrete computational grid — all three are facing the same mathematical constraint, and all three are therefore using the same mathematical atom.

    The Semiring: The Signature of Free Parallel Combination

    The semiring's two algebraic properties — associativity and commutativity — are precisely the mathematical conditions that make free parallel combination possible. Associativity means any grouping of partial results gives the same final answer. Commutativity means any ordering of those groups gives the same final answer. Together they mean that any assignment of agents to partial results is correct — which is exactly what makes it possible to use every available thread simultaneously without coordination. This is why the semiring reduction is O(1) per element in the parallel case. It is not because the hardware is well-designed, although it is. It is because the mathematical structure of the operation permits any ordering and grouping, so the hardware can choose the ordering and grouping that minimizes communication, which turns out to be a binary tree that completes in log₂N steps with each step touching every element once.

    The same mathematical structure appears everywhere independent agents combine contributions. In database aggregation, the query SUM(sales) over millions of rows can be computed in any order across any number of parallel workers because addition is associative and commutative. In Monte Carlo simulation, independent samples can be accumulated in any order for the same reason. In distributed systems, a count of events can be assembled from partial counts across servers without coordination. In economics, the aggregate demand of independent consumers sums correctly regardless of which consumers are counted first. None of these domains borrowed the semiring from GPU computing. All of them are independently instantiating the same mathematical atom because they face the same underlying constraint: combining independent contributions into a collective result.

    The Pattern Repeats for Every Atom

    The fixed point — iterate until self-consistent — appears in Newton's method for root finding, in gradient descent seeking a stationary point of the loss function, in Bellman equations for dynamic programming, in feedback control systems seeking a stable equilibrium, and in GPU tile loops accumulating partial results until all tiles are consumed. In every case, the structure is the same: apply a transformation repeatedly until a consistency condition is satisfied. The GPU tile loop is a bounded fixed point (the consistency condition is "all tiles consumed," which happens after a known finite number of steps). Newton's method is a convergence-based fixed point (the consistency condition is "the function value is below a threshold," which happens after an unknown number of steps). Both instantiate the fixed point atom; the difference is only in how the termination condition is specified.

    The symmetry atom — invariance under transformation — appears in crystallography as the lattice symmetries that classify crystal structures, in physics as the conservation laws that Noether's theorem derives from symmetry groups, in music theory as the transposition symmetry that makes a melody recognizable in any key, and in GPU kernels as the translational symmetry of AREA dimensions that makes each block's computation independent of every other block's. When AREA dimensions have translational symmetry, no communication is needed between blocks — each block computes its result entirely from its own tile of data. This independence is not a design choice made by NVIDIA. It is the consequence of the computation having translational symmetry along those dimensions, which is a mathematical property of the problem, not of the hardware.

    The relation atom — a named connection between operations or elements — appears in compilers as data flow graphs that track which variable definitions reach which uses, in operating systems as memory ordering constraints that determine which writes are visible to which reads, in physics as the causal structure of spacetime that determines which events can influence which other events, and in GPU kernels as the write-read dependencies between threads that force barrier placement. The barrier in a CUDA kernel is not an arbitrary synchronization point. It is the implementation of a specific mathematical relation: "this thread wrote a value that another thread needs to read." The barrier is the physical enforcement of the relation, and the relation is the mathematical object that forces the barrier to exist.

    The Tower of Abstraction and Why Naming Matters

    Every field builds its knowledge as a tower of abstraction. Physical reality presents patterns. Humans observe those patterns, name the most primitive ones they can agree on, and call those the atoms of the field. Then they observe patterns in the atoms, name those, and build the next layer. Then they observe patterns in that layer, name those, and build the layer above that. Each layer is, in a precise sense, a compiler: it takes the agreed-upon units of the layer below and composes them into the phenomena visible at the layer above.

    In mathematics, binary arithmetic is the atom layer. The patterns in binary arithmetic are named as algebra. The patterns in algebra are named as linear algebra. The patterns in linear algebra are named as abstract algebra. Each new layer did not change what was true at the layers below — arithmetic was not wrong before algebra was invented — but the naming of the new layer made reasoning at that layer systematic, teachable, and communicable in a way it had not been before.

    In GPU computing, binary electrons are the foundation. PTX assembly instructions sit one layer above. The six algebraic properties — closure, associativity, identity, commutativity, inverses, distributivity — sit at the next layer, describing the mathematical rules that make specific classes of parallel computation possible. The eleven atoms in this framework sit at the layer above that — the first layer where GPU computation becomes humanly nameable and composable in a systematic way, with each atom bundling the relevant algebraic properties into a single unit a kernel engineer can reason with. CUDA programming patterns like tiling, double buffering, and warp reduction sit at the layer above that. Triton and CUTLASS sit at the layer above that. Each layer is composed from the units of the layer below. The atoms in this framework are the vocabulary of the layer that makes the jump from "here are the hardware primitives and their mathematical properties" to "here is how to compose them into correct kernels" systematic rather than intuitive.

    Because the atoms themselves are mathematical objects that appear across many fields — not GPU-specific inventions but mathematical facts about the structure of parallel cooperation — the layer they define does not belong exclusively to GPU computing. A fluid dynamicist whose Navier-Stokes solver involves semiring reductions over spatial dimensions and affine maps from simulation coordinates to memory addresses is working with the same atoms as a GPU kernel engineer designing a GEMM. A compiler engineer whose intermediate representation transformation involves morphisms between program representations and fixed points for data flow analysis is working with the same atoms as a GPU engineer who uses the binding table as a morphism from problem geometry to execution hierarchy. The vocabulary is shared because the mathematical bedrock is shared.

    Why Naming the Atoms Is the Work

    Chemistry became a systematic science when Mendeleev named the elements and arranged them by their properties in the periodic table. This was not a discovery in the sense that new substances were found — the elements existed before the periodic table. It was a naming event: a crystallization of patterns that practitioners had been observing for decades into a shared vocabulary that made the patterns teachable, the relationships auditable, and the gaps visible. Once the periodic table existed, a chemist could look at a row of unfamiliar elements, recognize their position in the table, and predict their properties before synthesizing them. The naming did not create the chemistry. It made the chemistry systematic.

    The same transition is available in GPU kernel design. Before the atoms are named, expertise accumulates through observation and imitation over years of practice. An experienced engineer knows intuitively that a reduction "should go at the warp level" and that "this access pattern needs coalescing" and that "a barrier is needed here." They know this from having seen many kernels, debugged many race conditions, and profiled many bottlenecks. Their knowledge is genuine and valuable. But it is also difficult to transmit, because it is not organized around named units that can be pointed to, composed, and reasoned about systematically.

    After the atoms are named, that implicit knowledge becomes explicit. The intuition that "a reduction should go at the warp level" becomes: this is a semiring reduction, and the functor from the Step 1a algebraic classification to the Step 2 warp primitive selection maps semiring to shuffle tree. The intuition that "this access pattern needs coalescing" becomes: this affine map has translational symmetry with respect to the thread lane index, which by the SYM×AFM intersection in the optimization matrix guarantees coalesced access. The intuition that "a barrier is needed here" becomes: this write-read relation between threads was identified in Step 2's Relations column and the barrier rule in Step 3 places a barrier at every such relation. The knowledge has not changed. The vocabulary for expressing it has. And the vocabulary is what makes the knowledge teachable to someone who has not yet accumulated years of intuition, auditable by someone who wants to verify a design decision, and extensible by someone who encounters a new problem that does not fit the existing patterns.

    This framework is an attempt at that naming event for the domain of parallel GPU computation. The atoms it names were already there — in decades of CUDA tutorials, FlashAttention papers, CUTLASS design decisions, and the intuitions of experienced GPU engineers. The framework makes them explicit, connects them to the broader mathematical vocabulary they belong to, and organizes them into a derivation sequence that produces correct kernel structure from problem geometry alone. Whether the naming crystallizes into community agreement — whether these atoms become the shared vocabulary of GPU kernel design the way the periodic table became the shared vocabulary of chemistry — depends on whether the vocabulary proves useful enough to practitioners that it becomes worth adopting. That is not a question this document can answer. It is the question the GPU programming community will answer over time.

  

  
    A Note from the Author

    

      This culmination of this document is not that entirely of my own, I feel to say as such would be a absolute massive dishonesty to the many, many people that came before us all. 
      I myself am an undergraduate senior student who started learning CUDA 4 months ago because I found that this is the only field thus far that I can dive deep into where no answer is left to 
      abstraction, just pure hardware with software created utilize that hardware. I have never taken a compiler course, let alone know enough about
      physics, math, or even GPU programming to have come up with this framework on my own. I am simply a person whom is fortunate to live in the age of LLMs where every single hard fought victory from the people
      before us is now availiable at our fingertips. Where an LLM inherently is pattern matcher where if many neurons activate in a specific way at specific amounts, we map a pattern. But each one of these patterns,
      the LLMs training data, was the culmination of patterns discovered by the people before us. 

      While I know this sounds abstract, I cannot help but feel and see the parallel between the current field I am in (GPU computing) and the many many fields that we as people have made. Where at a fundamental level
      no matter the subject whether that be language, mathematics or GPUs, the fundamental pattern that I see is that at every one these towers of human understanding, there is a physical reality at which these 
      all started (we can call this our binary 1s and 0s for the sake of this document being in the field of GPU computing)

      Where in language, we can say our binary, the unit at which exists in physical reality, but is hard to quantify by human reason where many groups of people can agree and reason with is a single noise made
      by our vocal coords. From these vocal coords, we create our first atom, a unit of understanding that can be reasoned with by any person, and that represents a pattern from the layer below it. This may seem
      insignficiant, but truly the reason why we as humans are so exceptional is our ability to name the patterns we see, and to communicate, reason and understand these patterns as a collective. From this agreed upon
      patterns of words we as people collectively agreed to call it a sentence, from the patterns of a sentence we as people collectively agreed to call it a paragraph.

      Where in mathematics, we can say our binary, the unit at which exists in physical reality but is hard to quantify by human reason, is something even simpler than adding and taking away. 
      It is the simple act of looking at the world and seeing that one thing is distinct from another thing. A rock. Another rock. A third rock. 
      The rocks do not come with numbers attached to them, the number is simply the name we as people collectively agreed to give to the pattern of how many distinct things are here. 
      From this physical reality of simply being able to tell things apart, the first atom we as people collectively agreed to name was the counting number, one, two, three. 
      A unit of understanding that any person can reason with, and that represents the pattern of discrete quantity from the layer below it. From this agreed upon
      patterns of numbers we as people collectively agreed to call the addition and subtraction of these rocks arithmetic. 

      The amazing thing about us though is seeing a pattern isn't enough to satsify our curiousity, we have to see the pattern within the pattern.

      We began to notice patterns in arithmetic itself, patterns that held true no matter which numbers you chose to put in. 
      We noticed that the order you added things did not change the answer, that grouping numbers differently before adding them did not change the answer either. 
      From these patterns in arithmetic, we as people collectively agreed to name a new layer, and we called it algebra. Where before we reasoned with specific rocks, now we reasoned with the patterns that all rocks share.

      And of course, as amazing as we are, we did not stop there. We saw that algebra itself had many patterns that could not be quantified by the layer itself, so as a people agreed to call the patterns we witnessed linear algebra.
      Where instead of asking what specific numbers do, we asked what any collection of things does when you combine them in a structured way. 
      And it was here, at this layer, built on top of counting rocks, that mathematicians discovered and named two patterns so fundamental they appear everywhere cooperation and combination exist. 
      They called one a semiring, a structure where combination can happen in any order and any grouping and still give the same answer. They called the other a monoid, a structure where grouping is free but order must be respected.

      These are not GPU inventions. Rather they are patterns that mathematicians discovered in the study of abstract algbera long before anyone even imagined GPU cards. By the time these two atoms we used in GPU kernels,
      they had already been discovered, named, and studied for centuries. They are the mathematical structures that describe the patterns of free combination and free grouping 
      which is the exact patterns that make parallel reduction possible.

      The same is true for every single atom in this framework. They are not GPU inventions. 
      They are patterns that mathematicians and scientists have been discovering and naming for centuries in the study of the structures that arise when many independent atoms 
      cooperate to produce a collective result. I did not invent them. I simply noticed and asked what and why GPU computation has the structure the way it does. This could have been done by anyone.
      And that is exactly the point. Every layer in every tower of human understanding was built by someone who simply noticed a pattern that was already there, named it, and passed it forward. The people 
      whose hard fought victories live inside every LLM I used to build this framework did the same thing. I am only here, able to notice these patterns at all, because of everything they built and named 
      before me. If this framework is useful to even one person who comes after, if it helps them see the structure of a GPU kernel a little more clearly or ask a question they would not have known how to 
      ask before, then it will have done what every layer in every tower has always tried to do. Not to be the final word, but to be a step that makes the next step possible.
      
   --------------------

   I believe this framework covers many theories and models that have already been formalized in literature, while I have intentionally avoided reading these for the following two reasons:
     - Most of my insight and pattern matching has been from the GPU domain. Thus any insight that matches any current known formalisms represents how said specific formalisim exist within the space of GPU computation.
       Meaning rather than working top down (theory to application), going from bottom up (application to theory) provides a unique perspective on how the theory is shaped by the application, rather than how the application is shaped by the theory.

     - This is much more honest as to why but it is genuinely very fun when you have a theory from the patterns youve seen by making GPU kernels, and then realizing that an exisiting formalism exists to capture the pattern
       you've been seeing. For an example, I was absolutely puzzled as to why A * B + C kept popping up in every part of the ML stack I had learned. It's used in:
          - Neural Networks as weight * input + bias
          - GPU kernels as index * stride + offset
          - Softmax where we rescale our old values to the new using oldValue * newValue + newOffset
        I later learned that the operation / and % actually existed in the same plane as A * B + C where:
          - A * B + C allows us to flatten 2 dimension into 1
          - % and / allows us to go from our flattened dimension into two dimensions (assuming said flatten dimension was composed of two dimensions like we see with paged access)
        It has genuinely been the small hidden patterns like these that have made every step in this framework genuinely the most enjoyable experiences I've had thus far with CS.

      If I were to list the formal theories and models used in this framework they would be the following

      ------------

      At a fundamental level, no matter the tower of patternn of understanding/abstraction we fundamentally look:
       - Does it exist, does it not exist. Binary 1 or 0. The molecule of all abstractions, all patterns are based on this. Exist and doesn't exist.
       - TWhere something changes, and another thing doesn't. A pattern to be mapped, fundamental?

       1. Existence. 0 or 1. Nothing inbetween
       2. Distinction. Two existinences are seperatable. Pattern recognitio the difference between two things.
       3. INvariance. Something changes, something doesnt. Pattern that presists across change.
       4. Naming. A human gives the pattern a word. The Atom.
       5. Stacking. Named patterns reveal patterns among the patterns. New names. New layers. Thus the tower for all begins

       The physical occurnace, the medium is always different, langauge for sound, binary for 0 and 1 math for arithmetic. Same tower, different medium.
