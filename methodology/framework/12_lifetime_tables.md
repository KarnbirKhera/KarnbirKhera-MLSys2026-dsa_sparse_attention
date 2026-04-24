# Steps 4a & 4b — Lifetime Tables

atoms activated at Steps 4a and 4b

    
      
        MEA

        Step 4a: for every Block→Shared buffer, compute Reuse_block = ∏(AREA dims absent from buffer's index). If Reuse_block = 1 and no coalescing justification exists → buffer should be removed from Block→Shared. If Reuse_block > hardware_latency_in_flops → buffer is well-justified. Write both the measure value and its justification. Step 4b: for every Thread→Register value, compute Reuse_thread = ∏(thread-owned AREA dims absent from value's index). If Reuse_thread > 1 but tile = 1 → increase tile. Check MEA×FXP (Iteration Bound): is the reuse count equal to the iteration bound? If yes, the buffer's entire useful lifetime is exactly one tile loop execution.

      

      
        FXP

        Step 4a validates the fixed point state variables as well as the smem buffers. For each state variable (running_max, running_sum, accumulator): does it survive across all iterations of the tile loop fixed point? Verify: the FSM shows the variable alive in every phase within the tile loop. If a state variable is not alive in some phase, it is either unnecessary (remove) or the FSM is missing a phase (add it). The fixed point's state is what the variable analysis is measuring.

      

      
        SYM

        The reuse count for AREA dimensions is a measure of translational symmetry exploitation. High reuse count = the computation has strong translational symmetry at this level = significant staging benefit. Low reuse count = weak symmetry at this level = staging not justified. Write: "the staging decision for buffer X exploits the M-dimension's translational symmetry: each loaded value is reused M_tile times, once per position along M."

      

      
        FUN

        The functor from Step 2 determines how corrections to Step 2 propagate. If Step 4b requires reducing the thread tile size, apply the inverse functor: the block tile size may need to change accordingly (since the functor from block to thread was the original derivation rule). Verify that functor consistency is maintained after any Step 2 revision: the thread-level address formula must remain the functorial image of the block-level address formula.

      

      
        AFM SRG MON PRD ATO REL MOR

        AFM: addressed in Step 5 after tile sizes are finalized. SRG/MON: reduction structure validated at Step 3, not revisited here. PRD: boundary predicates valid as long as tile sizes don't change (if they change, revisit Step 1b). ATO: not active at lifetime analysis stage. REL: Combine Group relations validated at Step 3. MOR: smem layout morphism confirmed at Step 2.

      

    

  

  
    Step5

    
      Affine Map Composition — Atom Annotations

      "Compose all affine maps explicitly. Every address is a tree of typed leaves. Apply all morphisms identified in Steps 0.5 and 1b. Verify address ranges."
