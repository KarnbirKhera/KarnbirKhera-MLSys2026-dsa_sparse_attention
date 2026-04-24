# Step 1a Atoms & Step 1b — Access Patterns

atoms activated at Step 1a

    
      
        SRG

        Primary. REDUCE dimensions with associative+commutative operations are classified SRG. The algebraic structure check is the formal test of this classification. FUN will map this to the warp shuffle tree primitive at Step 2.

      

      
        MON

        Primary. REDUCE dimensions with associative-only operations are classified MON. FUN will map this to the Blelloch sweep primitive at Step 2.

      

      
        SYM

        AREA classification IS symmetry classification. Write it both ways: "M is an AREA dimension" and "M is a translational symmetry of the computation." The symmetry framing immediately yields: no inter-block communication needed (symmetry → independence), path specialization available at boundary tiles (symmetry breaks at edges), coalescing available if M maps to thread lane index (SYM×AFM → Guaranteed Coalescing).

      

      
        REL

        Combine Groups are relations between REDUCE dimensions. After classifying all dimensions, check: do any two REDUCE dimensions' partial results converge into the same output expression? If yes, write the Combine Group: {dim_A, dim_B} → REDUCE_MERGE at Step 3. The relation between the two dimensions forces the REDUCE_MERGE micro-phase. Check intersection REL×SRG in the matrix for the exact implementation.

      

      
        FUN

        The classification at Step 1a is the first application of the framework functor. Confirm: SRG classification → functor will select shuffle tree at Step 2 (no further analysis needed). MON classification → functor will select Blelloch sweep. Write this confirmation explicitly so Step 2 warp primitive selection is a lookup, not a decision.

      

      
        AFM PRD ATO FXP MOR MEA

        AFM: latent — address leaves classified at Step 1b. PRD: latent — predicate conditions depend on dimension fate (now known) and access patterns (Step 1b). ATO: not active. FXP: iteration count already written at Step 0.5, confirmed here. MOR: composable pairs confirmed once dimension fate is known. MEA: reuse counts not yet computable (need tile sizes from Step 2).

      

    

  

  
    Step1b

    
      Access Patterns — Leaf Types and Chain Depth

      "Arithmetic, lookup, conditional, or type-conversion leaf? And if lookup — how deep is the indirection chain?"

    

  

  
    Step 1b classifies how each tensor is addressed using four leaf types — arithmetic, lookup, conditional, and type-conversion — each carrying a chain depth annotation. The depth annotation is what distinguishes a simple one-hop lookup (depth 1, as in a paged KV cache) from a multi-hop chain (depth 2+, as in CSR sparse access). Naming the depth explicitly ensures the FSM generates the correct number of sequential memory-access phases and the binding table allocates the correct number of intermediate staging buffers.

  

  
    Beginner explanation — why address depth matters.
    
    Imagine finding a book in a library. An arithmetic access is like knowing the exact shelf number — you go directly there. A depth-1 lookup is like a catalog card that says "see shelf B7" — you check the card first, then go to the shelf. A depth-2 chain is like a card that says "see card box 3," where card box 3 contains a card that says "see shelf B7" — two lookups before you reach the book. A depth-3 chain adds one more level: the first lookup tells you where to find the card box, the second lookup finds the card, the third lookup finds the book.
    
    Every additional level of indirection means one more sequential memory read — and sequential means you cannot start the next read until the previous one returns its result. On a GPU, this creates a new FSM phase per hop, because all 32 threads in a warp must complete each hop before any thread can compute the address for the next hop. Missing a hop in the FSM means threads will read stale or garbage addresses for subsequent hops.
  

  
    Why chain depth must be explicit in the framework
    A depth-1 lookup was already in the framework for paged KV attention. CSR sparse matrices require depth 2 (row pointer → column index). Some graph algorithms require depth 3. Without the depth annotation, the framework silently treats all lookups as depth 1 — generating an FSM with only one sequential-read phase when two or three are needed. The resulting kernel reads the wrong addresses on every hop beyond the first. Making depth explicit turns this silent correctness failure into an auditable design decision.
  

  
  
    
      
        
          Depth
          Leaf Name
          Address Formula
          GPU FSM Phases Added
          Key Examples
        

      
      
        
          Depth 0existing
          Arithmetic
          addr = blockIdx × tile + threadIdx × stride + offset — fully computed from thread coordinates and constants. No data reads before the final access.
          No sequential phases added. The address is computed in registers before any memory access occurs.
          GEMM tile address, attention Q/K/V load, any tensor with regular stride-based access.
        

        
          Depth 1existing
          Lookup
          page_addr = page_table[geometric_index]final_addr = page_addr + intra_page_offset — one data read before the final access.
          One sequential micro-phase added to the FSM: "Read page table entry." A barrier follows before the final access phase.
          Paged KV cache (depth-1 page table lookup), standard embedding table lookup.
        

        
          Depth 2new
          Two-hop chain
          ptr1 = table1[geometric_index]ptr2 = table2[ptr1]final_addr = ptr2 + offset — two data reads before the final access.
          Two sequential micro-phases added: "Read table1 entry" → barrier → "Read table2 entry" → barrier → final access. Each hop must complete before the next address can be computed.
          CSR row pointer → column index (ptr1 = row_ptr[row_id]; ptr2 = col_idx[ptr1]). Blocked sparse formats with two-level indexing.
        

        
          Depth 3+new
          n-hop chain
          ptr1 = table1[geometric_index]ptr2 = table2[ptr1]ptr3 = table3[ptr2]... one read per hop.
          n sequential micro-phases, one per hop, each followed by a barrier. FSM grows linearly with chain depth. Total sequential time = n × memory_latency at minimum.
          CSR SpMV with separate value array (depth 3: row_ptr → col_idx → values). Irregular graph neural networks with multi-level adjacency. Nested hash table structures.
        

      
    
  

  For every chain depth > 1, write down the following for each hop when completing Step 1b:

  
    
      
        
          Per-Hop Field
          What to Write
          Atoms Active at This Hop
          What It Drives Downstream
        

      
      
        
          Hop address formula
          Write the address used to retrieve this hop's value. For hop k, this is: "address of this read = f(result of hop k-1)." For hop 1, it is: "address of this read = f(thread coordinates)." This must be an affine function of the previous hop's result — if it is not affine, a new leaf type is needed beyond the current chain model.
          AFM — each hop's address formula is an affine map applied to the previously retrieved value. The chain is a composition of affine maps where each map's input is the previous map's output.
          Step 3 FSM: generates one sequential read micro-phase for this hop. Step 2: this hop's result must be staged somewhere while the next hop's address is computed — add a register or smem entry for the intermediate value.
        

        
          Intermediate staging decision
          Where does the result of this hop live between when it is retrieved and when the next hop uses it? Options: register (fast, one value per thread), shared memory (allows reuse across threads in the block). Write: "registers — no reuse needed" or "shared memory — reuse count = [N]."
          MEA — each staged intermediate value adds to the register or smem budget. FXP — the chain traversal is a sequential fixed point; each hop's result is the state that the next hop's phase reads from.
          Step 2 binding table: add one row to the staging section per intermediate value. Step 4a: compute reuse count for the intermediate — if reuse > 1, smem staging is justified; if reuse = 1, register staging is sufficient.
        

        
          Coalescing at this hop
          For this hop's memory access: do all 32 threads in a warp access addresses that are stride-1 in the lane dimension? For hop 1, this is the same analysis as any arithmetic leaf. For hop 2+, the addresses come from the previous hop's retrieved values — coalescing is now data-dependent and may be unpredictable unless the data has been pre-sorted by access pattern.
          SYM — coalescing requires translational symmetry in the lane dimension. For hop 2+, this symmetry may be absent (the retrieved values from hop 1 may not be stride-1 in the lane dimension). REL — the relation between this hop's addresses and the previous hop's values determines whether coalescing is achievable.
          Step 2 optimization signal matrix: if coalescing symmetry is absent at any hop, the actual DRAM transaction count per warp is proportional to the number of distinct cache lines accessed, which in the worst case equals the warp size (32 transactions per warp instead of 1).
        

        
          Null / bounds predicate
          Can the result of this hop ever be an invalid index — a null pointer, an out-of-range index, or a sentinel value? If YES, write the predicate check: "if ptr1 == null, use identity element" or "if ptr1 >= N, skip this row." This is a conditional leaf at this hop — it uses the PRD atom to guard the subsequent hop.
          PRD — the null/bounds check is a predicate applied to the retrieved value rather than to a geometric coordinate. The identity element is substituted when the predicate is false, exactly as with standard conditional leaves.
          Step 1b: marks this hop as a conditional hop in addition to a chain hop. Step 3 FSM: the predicate check occurs in the same micro-phase as the hop read and gates whether the next hop is issued.
        

      
    
  

  
    CSR SpMV example — chain depth annotation completed:
    Accessing values in a CSR sparse matrix requires three hops. Hop 1: row_start = row_ptr[row_id] — arithmetic address (row_id is from blockIdx, fully geometric), result lives in registers, coalescing holds if consecutive threads process consecutive rows (stride-1 in row_id). Hop 2: col_id = col_idx[row_start + k] — address is row_start + k where row_start came from Hop 1 and k is the intra-row iterator; result lives in registers; coalescing depends on data layout (not guaranteed). Hop 3: value = val_array[row_start + k] — same address structure as Hop 2; result enters the accumulator register. The final multiply-add uses val_array[row_start + k] × input_vector[col_id], where col_id came from Hop 2. FSM consequence: three sequential read micro-phases, two barriers, before the multiply-add compute phase.
  

  
    atoms activated at Step 1b

    
      
        AFM

        Primary. Four leaf types: Arithmetic (coord × stride + offset — pure AFM), Lookup (one-level indirection into a page table — AFM composed with a data-dependent offset), Conditional (AFM address valid only when predicate is true), Type-conversion (AFM address followed by a dtype cast). The leaf type determines which optimization columns in the Optimization Signal Matrix apply.

      

      
        PRD

        Primary. Conditional leaves trigger predicate derivation. Classify by triggering iterator: grid-level → early exit block; tile-level → predicated load inside loop. Write predicate expression and identity element for each conditional leaf.

      

      
        SYM

        For each arithmetic leaf: does the address formula have translational symmetry with respect to the thread lane index? If addr(lane) = base + lane × stride and stride = 1 → coalescing symmetry confirmed → check intersection SYM×AFM → Guaranteed Coalescing in the matrix. If stride ≠ 1 → symmetry broken → count actual DRAM transactions. If stride is a multiple of 32 in the smem formula → harmful bank symmetry → plan XOR swizzle at Step 2 (SYM×MOR intersection).

      

      
        MOR

        For each pair of arithmetic leaves: are they composable? If leaf A's output address is leaf B's input coordinate, they are composable into a single FMA. Write all such composable pairs (they will be fused at Step 5). Also: does the smem layout need to differ from the global layout for bank-conflict avoidance? If yes, the smem layout formula is a morphism of the global layout. Write the morphism rule (XOR swizzle, transpose, +1 padding) here so Step 2 can allocate the correct smem shape.

      

      
        REL

        For lookup leaves (paged access): the page table index is a relation between the logical token position and the physical page address. Write this relation explicitly: logical_pos → page_table[logical_pos / PAGE_SIZE] × PAGE_SIZE + (logical_pos % PAGE_SIZE). This relation is the formal definition of the lookup leaf, and it determines whether the access is coalesced within a page (it is) or across page boundaries (it is not).

      

      
        SRG MON ATO FXP MEA FUN

        SRG/MON: classification complete from Step 1a. ATO: not at leaf classification stage. FXP: tile loop structure already captured at Step 0.5. MEA: reuse measurement not yet possible (needs tile sizes). FUN: functor from Step 1a classification to warp primitive will fire at Step 2.

      

    

  

  
    Step2

    
      Binding Table — Atom Annotations

      "Assign each atom to its hierarchy level, verify all six measures, confirm the functor action at each level transition, check all relations between buffers."
