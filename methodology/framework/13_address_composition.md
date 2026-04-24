# Step 5 — Affine Map Composition

atoms activated at Step 5

    
      
        AFM

        Primary. Write every memory address as a tree of typed leaves. Compose: global_addr = grid_term + block_term + thread_term, where each term is one level's affine map contribution. Upward pass: can any REDUCE leaf move up one level (parallelization)? Downward pass: can any AREA leaf move down into the thread register tile (ILP)? These passes are asking: does the affine map have slack in its level assignment?

      

      
        MOR

        Apply all composable pairs identified at Steps 0.5 and 1b. For each pair: write the composed map as a single expression (this becomes one FMA instruction). Apply the smem layout morphism: the smem address formula is the global address formula with the morphism rule substituted for the smem dimension stride. Write both forms side by side — the global form (from storage format) and the smem form (from morphism application). Check MOR×AFM (Address Composition) in the matrix for the exact FMA form.

      

      
        FUN

        The functor derivation rule from Step 2 is applied here to derive thread-level addresses from block-level addresses. Rule: substitute TM for BM, threadIdx for blockIdx. Apply this substitution to every block-level address formula to produce the corresponding thread-level formula. This guarantees the thread addresses are consistent with the block addresses — a correctness guarantee provided by the functor, not by inspection. Verify: the functor-derived thread address has the same structure as the block address, just with different scale parameters.

      

      
        MEA

        Compute max addresses for every buffer (the upward pass reveals the maximum value of each address expression). Verify: max_address ≤ allocated buffer size. This is a measure check: the measure of the address space accessed must not exceed the measure of the allocated buffer. Write the max address computation explicitly alongside each address formula.

      

      
        SYM

        The upward pass (can a REDUCE leaf move up?) is asking: does this computation have additional parallelism symmetry that we haven't exploited? An REDUCE leaf that can move up without breaking correctness represents a translational symmetry of the partial result with respect to the level above. The downward pass (can an AREA leaf move down?) is asking: does this address have a symmetry that allows it to be replicated across the thread tile without additional overhead?

      

      
        SRG MON PRD ATO FXP REL

        SRG/MON: reduction structure fully determined at Step 3. Step 5 addresses the data access side, not the computation side. PRD: predicate expressions are written in the Index Expression Table alongside address formulas — they reference the max addresses computed here. ATO: not active at address composition stage. FXP: tile loop structure complete from Step 3; Step 5 derives the address for each tile iteration. REL: Combine Group relations fully handled by REDUCE_MERGE in Step 3; Step 5 addresses individual buffer accesses.

      

    

  

  
    WhereIt Stops

    
      Framework Boundaries — A Precise Map of What Cannot Be Derived and Why

      "Which atom's current definition is insufficient for each excluded kernel family, and exactly what mathematical property would that atom need to gain to cover it?"
