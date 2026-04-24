# Dimension Fate Rules — AREA and REDUCE

atoms activated at Level 0 — all 11

    
      
        AFM

        The storage format answer (row-major, column-major, paged) gives the literal stride coefficients that appear in every affine map formula downstream. Level 0 is where the affine map's parameters are named before the map is composed.

      

      
        SRG

        The operation type answer names the combining operator. "Sum over k", "softmax-weighted sum", "max" — each is a semiring multiplication + accumulation. The semiring is named here; its algebraic properties are verified at Step 1a.

      

      
        MON

        If the combining operator is ordered (prefix scan, running count), the monoid atom is named here alongside the semiring. Write the combining operator precisely enough that the commutativity question is answerable from the Level 0 text alone.

      

      
        PRD Latent

        The output shape establishes the valid index ranges that predicates will guard. Every "if i < N" downstream traces to the shape bounds written here. The predicate is latent — it exists but is not yet expressed.

      

      
        ATO Latent

        Not active at Level 0. Atomics only appear once execution units share addresses, determined at Step 2.

      

      
        FXP

        The operation type answer reveals whether the computation is a fixed number of applications (GEMM tiles) or a convergence computation (iterative solver). Write this explicitly: "bounded fixed point, count = ceil(K/BK)" or "online algorithm — carries running_max, running_sum across iterations."

      

      
        MOR

        The storage format answer describes the morphism between logical coordinate space and hardware address space. Row-major storage IS the affine morphism coord_row × stride_M + coord_col × stride_N. Write the morphism explicitly: the storage format is not bookkeeping, it is the affine map's coefficients named in domain language.

      

      
        MEA

        The input and output tensor shapes give the first measure: total data size. Write: Measure_total = ∏(all dims) × sizeof(dtype) for each tensor. This is the measure of the problem, against which all subsequent staging decisions are measured.

      

      
        REL

        The input vs. output shape comparison (which drives Step 1a) is also the first place to look for Combine Group relations. If two input dimensions are absent from the output but their partial results are combined by the same output expression, they are related. Write any Combine Groups identified here — they will be confirmed at Step 1a.

      

      
        SYM

        Every dimension that appears in both the input and output shapes is a candidate translational symmetry. Write explicitly: "M and N are translational symmetries of the computation — computing at (m,n) is structurally identical to computing at (m+1,n)." This makes the AREA classification at Step 1a a symmetry confirmation rather than a fresh analysis.

      

      
        FUN

        The Level 0 geometry spec is the domain of the framework functor. Write: "the framework functor maps this geometry spec to a kernel implementation." The three questions of Level 0 are the three components of the functor's input object: source tensors (objects), operation (morphism), target tensor (image object).

      

    

  

  
    Step1a

    
      Dimension Fate — AREA, REDUCE, or GATE

      "What happens to each dimension? Does it survive independently (AREA), collapse into one combined value (REDUCE), or selectively filter elements based on their values (GATE)?"

    

  

  
    Step 1a classifies every dimension in your computation into one of three categories: AREA, REDUCE, or GATE. AREA and REDUCE cover standard parallel operations. GATE handles the class of operations where elements compete for survival based on their values — every kernel that involves selecting, filtering, or ranking elements requires a GATE dimension.
