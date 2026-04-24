# Streaming Gate (FXP × GATE)
# ★ New in v8 — required when GATE is nested inside a tile loop

Streaming Gate — What Changes vs. Single-Pass Gate

      
        
          
            Property
            Single-Pass Gate
            Streaming Gate (FXP×GATE)
          

        
        
          
            When candidates arrive
            All at once, before the loop
            In batches, one per tile iteration
          

          
            Output write phase
            Inside the Gate FSM
            Deferred to after the tile loop
          

          
            Buffer state across iterations
            None — single pass
            Running best-K buffer in SMEM, maintained sorted descending by score. K-th score is the current threshold.
          

          
            Per-tile Phase 1
            Full sort / rank of all N elements
            Threshold comparison: does score > current K-th best? Produces boolean mask for this tile's batch.
          

          
            Per-tile Phase 2
            Blelloch prefix sum → final output positions
            Blelloch prefix sum → staging positions → merge survivors into running buffer (two cases; see below)
          

          
            Threshold state
            Not needed
            Maintained in SMEM. Updated after every Case B merge. Always equals buffer[K−1].score after a full sort.
          

        
      
    

    
      Per-Tile Merge — The Two Cases

      
        After the Blelloch prefix sum identifies which candidates in this tile's batch survive the threshold check, they must be merged into the running buffer. The merge has exactly two cases depending on how full the buffer currently is. Both cases must be pre-wired in d1.md — skipping Case B is the most common implementation error with Streaming Gate kernels.
      

      
        
          CASE A — Buffer has room

          
            Condition: current_count + n_survivors ≤ K
            Algorithm:
            
              Write survivors directly into buffer at positions current_count + prefix[t]

              Update buffer count

              If buffer just became exactly full (count = K): sort buffer descending by score, set threshold = buffer[K−1].score

            
            
            Why it is simple: No existing buffer entry can be displaced — there is space for everyone who survived the threshold check.
          

        

        
          CASE B — Buffer is full ★ do not skip

          
            Condition: current_count = K
            Algorithm:
            
              Write survivors into a staging area (separate SMEM scratch buffer)

              Sort staging area descending by score (this step is mandatory — see warning below)

              For i = 0 … n_survivors−1: if staged_score[i] > buffer[K−1−i].score → replace buffer[K−1−i] with the staged entry

              Re-sort buffer descending

              Update threshold = buffer[K−1].score

            
          

        

      

      
        ⚠ WHY SORTING SURVIVORS BEFORE REPLACEMENT IS MANDATORY

        
          Survivors arrive in page-position order, not score order. If you skip the staging sort and match survivor i directly against buffer position K−1−i, a low-scoring survivor at page position 0 can displace a buffer entry that a higher-scoring survivor at page position 1 should have displaced instead.
        

        
          Concrete counterexample. K=4. Buffer (sorted descending): [10, 5, 3, 1]. Threshold = 1. Two survivors arrive in page order: survivor A (score=2) at page position 0, survivor B (score=4) at page position 1. Both pass the threshold check (both > 1).
        

        
          
            ❌ Without sorting survivors first

            
              A(2) vs buffer[3]=1 → 2>1, replace
              B(4) vs buffer[2]=3 → 4>3, replace
              Buffer after: [10, 5, 4, 2]
              Entry 3 is gone. Wrong.
            

          

          
            ✓ With sorting survivors first

            
              Sort survivors: B(4), A(2)
              B(4) vs buffer[3]=1 → 4>1, replace
              A(2) vs buffer[2]=3 → 2<3, keep
              Buffer after: [10, 5, 4, 3] ← sort
              Correct top-4.
            

          

        

        
          The sort ensures the strongest survivors compete for the weakest buffer positions. Without it, the positional matching is arbitrary and can produce wrong results in any case where survivor scores are not already in descending order — which is the common case.
        

      

    

    
      Streaming Gate — Pre-Wired Decisions for d1.md

      When Streaming Gate is detected, the following decisions must be pre-wired in d1.md as part of the Gate molecule confirmation. They must not be left to the implementer.

      
        Accumulation buffer location and layout: SMEM, sized to K entries, sorted descending by score at all times. Layout must include both score (float32) and original index (int32) per entry. Threshold state variable (the K-th score) stored alongside in SMEM.

        Staging scratch area: A separate SMEM buffer sized to the maximum batch size (= tile size). Used in Case B to hold survivors before sorting. Must be allocated separately from the accumulation buffer.

        Two-case merge algorithm: Case A (buffer not full) and Case B (buffer full, with mandatory staging sort). Both cases must be explicitly specified. Describing only Case A is incomplete.

        Sort algorithm for Case B staging: Specify the sort used for the staging area (e.g., bitonic sort, insertion sort). Batch size is bounded by tile size — typically 32–128 elements — so any simple in-SMEM sort suffices. The sort must be descending by score.

        Buffer re-sort after Case B: After replacing tail entries, the full K-element buffer must be re-sorted descending. Specify the algorithm (typically bitonic sort over K elements, cooperative across all threads).

        Output write deferred to after tile loop: The output write phase writes the final buffer contents to global memory after all pages are processed, not inside the per-tile Gate FSM.

      
    

    
      Streaming Gate — FSM Structure

      The Streaming Gate replaces the single-pass Gate's output-write phase with a per-tile merge inside the tile loop, and adds a final output-write phase after the loop ends.

      
        // Before tile loop
        Initialize accumulation buffer (empty, threshold = −∞)
        // Inside tile loop (once per tile iteration)
        [existing tile phases: Load → Compute → SRG/MON Reduce]
        → scores for this tile's batch now available
        Phase: Streaming Gate Phase 1 — threshold comparison → boolean mask
            barrier [mandatory REL: mask written before prefix sum reads it]
        Phase: Streaming Gate Phase 2 — Blelloch prefix sum over boolean mask → staging positions
            barrier [mandatory REL: positions written before merge reads them]
        Phase: Streaming Gate Merge — Case A or Case B (per above)
            barrier [mandatory REL: buffer updated before next iteration reads threshold]
        // After tile loop
        Phase: Output Write — write buffer[0..K−1].index to global topk_indices
      

      
        The barrier after the Merge phase is new — it does not appear in the single-pass Gate. It is required because the threshold value written in the Merge phase is read at the top of the next iteration's Phase 1. This is a write→read relation across tile loop iterations.
