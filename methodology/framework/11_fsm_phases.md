# Step 3 — FSM Phase Structures

What changes in Step 3 for GATE dimensions and chain leaves.
    
    The FSM sequences phases based on which memory levels are crossed: Load from global → Compute in registers → Store to global, with barriers wherever a write in one phase is read in a subsequent phase. Two additional mandatory phase patterns appear automatically when Step 1a identifies a GATE dimension or Step 1b identifies a chain depth > 1.
    
    The GATE two-phase pattern: Every GATE dimension produces exactly two mandatory phases inserted into the FSM at the point where the GATE's output is produced. Phase 1 evaluates the gate criterion across all elements (parallel, using SRG or MON depending on criterion parallelizability). A mandatory barrier separates Phase 1 from Phase 2. Phase 2 runs the exclusive prefix sum (MON Blelloch sweep) over Phase 1's boolean mask to assign output positions. A second barrier separates Phase 2 from any subsequent phase that reads the output positions. You cannot reorder these phases, merge them, or remove the barriers between them — the REL between Phase 1 and Phase 2 is what the barriers are implementing.
    
    Streaming Gate variant (FXP×GATE) — check this before writing the FSM: If the GATE dimension's candidates are produced inside the tile loop rather than all at once before it, the single-pass pattern above is insufficient. The output-write phase must be deferred to after the tile loop, and a per-tile merge phase (with its own barrier) is added inside the loop. The merge has two cases depending on buffer fill state (see Gate Specification Table — Streaming Gate section). The additional barrier after the merge phase is mandatory and is not present in the single-pass pattern: it guards the threshold value that the next iteration's Phase 1 will read. Failure to detect Streaming Gate when it applies will produce a kernel that appears structurally correct but retains the first K candidates seen rather than the top K.
    
    The chain hop phase pattern: Every chain depth > 1 produces (depth - 1) additional sequential micro-phases inserted before the phase that uses the final address. Each micro-phase reads one intermediate value from memory; a barrier follows each micro-phase before the next one can begin. Chain micro-phases are purely sequential — they cannot be pipelined or overlapped with each other because each one's address is the previous one's retrieved value. They can, however, be overlapped with unrelated computation happening in other dimensions if the binding table assigns those dimensions to separate phases.
  

  
    atoms activated at Step 3

    
      
        PRD

        Primary. Barrier rule: __syncthreads() wherever one thread writes shared memory that another thread reads. This is the PRD atom at the block level: the barrier is the implementation of the write→read relation. Every barrier placement traces to a specific REL entry from Step 2's Relations column.

      

      
        FXP

        The tile loop in the FSM is the bounded fixed point. Write explicitly in the FSM: "Fixed Point: iterate num_tiles = ceil(K/BK) times. State variables carried across iterations: [list all register-held accumulators and online algorithm state]." The FSM's loop structure IS the fixed point structure — naming it makes it auditable. If FXP×PRD (Predicate Hoisting) fired at Step 0.5, verify that hoisted predicates appear before the tile loop row in the FSM, not inside it.

      

      
        REL

        The FSM's phase ordering is the formal expression of all Relations identified at Step 2. For each write→read relation: verify a barrier appears between the writer phase and the reader phase. For each Combine Group: insert a REDUCE_MERGE micro-phase between the two warp-reduce phases and before the block-merge phase. The REDUCE_MERGE micro-phase is the implementation of the Combine Group relation — it has no other justification. Check REL×SRG (Combine Group) for the exact implementation detail (which lane is the combining thread).

      

      
        SYM

        The FSM's phase sequence should reflect the computation's symmetry structure. Phases that are symmetric (identical structure, different data) can be written once with a loop rather than duplicated. The Blelloch sweep's up-sweep and down-sweep are time-reversal symmetric — recognize this in the FSM to enable the FXP×SYM (Sweep Sharing) optimization: store up-sweep intermediates for reuse in down-sweep.

      

      
        FUN

        The FSM is the functorial image of the binding table's column structure. Each column in the binding table corresponds to a set of FSM phases: Grid→Global becomes Bounds Check and Store phases; Block→Shared becomes Load and partial-result Write phases; Warp→Primitives becomes the Warp Reduce phase; Thread→Registers becomes the Compute phase. The functor maps the static structure of the binding table to the dynamic structure of the FSM. Verify: every FSM phase traces to exactly one binding table column.

      

      
        AFM SRG MON ATO MOR MEA

        AFM: addresses derived at Step 5, not Step 3. SRG/MON: warp primitive choice already made at Step 2; FSM uses that choice. ATO: appears as a specific phase row if cross-block reduction was selected at Step 2. MOR: the pipeline morphism (sync→async) is applied after Step 3 as an optional transformation. MEA: budget validation complete from Step 2; Step 3 adds no new measures.

      

    

  

  
    Steps4a+4b

    
      Lifetime Tables — Atom Annotations

      "Validate block-level staging decisions (4a) and thread-level tile shapes (4b) using formal measure computations."
