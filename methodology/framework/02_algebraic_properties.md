# Algebraic Properties Beneath the Atoms

BelowtheAtoms

    
      The Algebraic Properties Beneath the Atoms

      "If the atoms are the smallest units a kernel engineer can reason with, what are the atoms built from? Why do SRG and MON have the GPU consequences they do?"

    

  

  
    When counting rocks on the ground, we do not start with algebra. We start with a simpler question: can I combine two rocks into one pile? That question — two things in, one thing out — is the most primitive possible description of computation. Mathematicians call this a binary operation. A set of things, and a rule for combining any two of them. No restrictions on how the combining works. No promises about the result. Just: two in, one out.

    A binary operation alone is almost useless for parallel computing, because without any rules about how the combination works, you cannot reorder it, regroup it, or split it across threads. You would have to compute everything sequentially, left to right, exactly as written. The entire GPU would be idle except for one thread. The six algebraic properties below start with that bare binary operation (closure) and then add exactly one rule at a time to this starting point. Each rule unlocks a specific GPU capability that was impossible without it. The properties accumulate — each one builds on the ones before it — and the atoms in this framework (SRG, MON, and the others) are specific combinations of these properties, named at the level where GPU engineers can reason with them.

  

  
    Why show this layer? Two of the eleven atoms — SRG (semiring) and MON (monoid) — are composites built from the properties below. Understanding the individual properties tells you exactly which GPU capability each property enables, rather than treating SRG and MON as indivisible bundles. When you encounter a new combining operation, checking these properties one at a time tells you precisely what optimizations are available and which are not — more precisely than asking "is it SRG or MON?"
  

  
  

    
    
      
        BO
        Binary operation (closure) — two things in, one thing out
      

      The rule: for any two elements a and b in a set S, the result a · b is also in S. This is called closure — the result of combining two things is the same kind of thing as the inputs. Combining two floating-point numbers gives you a floating-point number, not a string. Combining two thread results gives you a thread result, not an error. This is the most primitive possible starting point for computation: a combining operation exists and produces valid output. No other rules are imposed — no guarantees about ordering, grouping, identity, or reversibility. Mathematicians call a set with just this property a magma.

      Every combining operation in every GPU kernel satisfies closure. If it didn't — if combining two partial results could produce something outside the valid set — the kernel would be fundamentally broken. Closure is always present. It is the ground floor on which every other property is built. The reason it appears in the matrix and in this list is that the intersections in the BO row show what each individual property contributes on its own, before combining with any other property. The BO × ASC cell shows what associativity alone gives you (tree parallelism). The BO × IDE cell shows what an identity element alone gives you (safe padding). These are the first-order contributions, and the rest of the matrix shows the second-order interactions between properties.

      
        What it unlocks: The ability to combine two partial results into one. Without this, no reduction of any kind is possible — not even sequential.
        What happens without it: No combining operation exists. Elements cannot be aggregated. There is no computation to parallelize. This is below Failure Condition 4 — it is the absence of the problem itself.
      

    

    
    
      
        ASC
        Associativity — regrouping doesn't change the answer
      

      The rule: (a · b) · c = a · (b · c). You can move the parentheses anywhere without changing the result. This single rule is what transforms a sequential chain into a parallel tree. Without associativity, you must compute element 0 combined with element 1, then that result combined with element 2, and so on — pure sequential, depth N. With associativity, you can regroup into a tree: combine pairs simultaneously, then combine the results — depth log₂N. That is an exponential speedup from one algebraic property.

      
        What it unlocks: Tree-shaped parallel reduction. The Blelloch sweep structure. Every reduction tree in every GPU kernel exists because this property holds for the combining operation.
        What happens without it: Strictly sequential computation. One thread does all the work. No parallel reduction possible. This is Failure Condition 4 in the framework.
      

    

    
    
      
        IDE
        Identity element — a "do nothing" value exists
      

      The rule: there exists a special value e such that e · a = a · e = a for any element a. Combining anything with the identity leaves it unchanged. For addition, the identity is 0. For multiplication, it is 1. For max, it is negative infinity. For min, it is positive infinity.

      This matters for GPU computing in two specific ways. First, it solves the padding problem: when a reduction tree needs a power-of-two number of elements but the input has 13 elements, you fill three slots with the identity, and the result is unchanged. Second, it solves the initialization problem: when a tile loop starts, the accumulator must be set to some value before the first tile arrives. That value is the identity element — acc = 0.0f for sum, acc = -INFINITY for max.

      
        What it unlocks: Safe accumulator initialization. Predicated reduction (inactive lanes hold identity). Boundary handling without special cases.
        What happens without it: No clean way to initialize accumulators. Boundary tiles require conditional logic. The PRD atom's masking strategy (replace invalid elements with identity) becomes impossible.
      

    

    
    
      
        COM
        Commutativity — reordering doesn't change the answer
      

      The rule: a · b = b · a. You can swap the order of any two elements without changing the result. When combined with associativity, this means any permutation of the elements produces the same answer. This is the property the framework calls permutation symmetry, and it is what separates the SRG atom from the MON atom.

      With commutativity, the warp shuffle tree (__shfl_down_sync) is valid — the hardware's butterfly reduction pattern is just one particular permutation of a binary tree, and commutativity guarantees it gives the same answer as any other permutation. Without commutativity, the shuffle tree is invalid, because reordering the elements would change the result. You must use the more expensive Blelloch sweep, which preserves order through its up-sweep and down-sweep structure.

      
        What it unlocks: Shuffle tree reduction (any butterfly pattern valid). atomicAdd for cross-block reduction (concurrent writes in any order are correct). The difference between O(log N) with one barrier (SRG) and O(log N) with 2×log₂N barriers (MON).
        What happens without it: Order must be preserved. Blelloch sweep required instead of shuffle tree. This is the SRG→MON downgrade in the framework.
      

    

    
    
      
        INV
        Inverses — every operation can be undone
      

      The rule: for every element a, there exists an element a⁻¹ such that a · a⁻¹ = e (the identity). For addition, the inverse of 5 is −5. For multiplication (excluding zero), the inverse of 3 is 1/3. For rotation, the inverse of "rotate 90° clockwise" is "rotate 90° counterclockwise."

      Inverses matter for GPU computing because they enable correction factors. Online softmax's rescaling step — multiplying by e^(old_max - new_max) — works because subtraction is the additive inverse and division is the multiplicative inverse. The old accumulation can be corrected rather than recomputed from scratch. Without inverses, the Online molecule is impossible: you cannot undo the effect of the old state when the new tile reveals new information. This is why max-reduction (which has no inverse — you cannot "un-max") can be tiled but cannot support an online correction algorithm. RoPE uses rotation inverses for relative position encoding. Normalization (layer norm, RMS norm) uses division. These operations are present throughout deep learning but were unnamed in the original framework.

      
        What it unlocks: Online algorithms (correction factors at tile boundaries). Normalization. Relative position encoding (RoPE). Residual connections (subtraction is the additive inverse).
        What happens without it: No online correction — when running state changes, prior accumulations must be recomputed from scratch rather than corrected. Tiling still works (ASC), but the Online molecule cannot be confirmed.
      

    

    
    
      
        DIS
        Distributivity — two operations connect cleanly
      

      The rule: a × (b + c) = (a × b) + (a × c). One operation (multiplication) distributes over another (addition). This property connects two binary operations into a single coherent structure — the semiring. The formula A × B + C is one step of a semiring computation: the × is the semiring's multiplication and the + is the semiring's addition. The FMA (fused multiply-add) hardware instruction executes this in a single cycle because distributivity guarantees the two operations compose correctly.

      Distributivity is what makes tiling correct. When you split the K dimension of a matrix multiply into tiles and accumulate partial sums, you are relying on distributivity: a × (b₁ + b₂ + … + bₖ) = (a × b₁) + (a × b₂) + … + (a × bₖ). Each tile computes one partial product (a × bᵢ), and the partial sums compose correctly because of distributivity. Without it, splitting a dimension into tiles and recombining partial results would give a different answer than computing the full dimension at once. The Tile molecule exists because distributivity holds.

      
        What it unlocks: Tiling correctness (split a dimension, recombine partials). The FMA instruction (two ops in one cycle). Sparsity optimization (0 × a = 0, skip computation). The Tile molecule's structural validity. The A × B + C formula.
        What happens without it: Tiling produces wrong answers. The K dimension cannot be split. Each GEMM tile is no longer a valid partial result. The Tile molecule cannot be confirmed for this operation.
      

    

  

  
  
    How the properties compose into atoms. The SRG atom is the combination of ASC + IDE + COM + DIS — associativity gives you the tree, identity gives you initialization, commutativity gives you the shuffle tree, and distributivity connects the multiplication and addition into the A × B + C formula. The MON atom is ASC + IDE without commutativity — associativity gives you the tree structure, identity gives you initialization, but the absence of commutativity forces the more expensive Blelloch ordered sweep instead of the free shuffle tree. The Online molecule additionally requires INV — without inverses, the correction factor at tile boundaries cannot be computed, and online accumulation is impossible. Each property's presence or absence has a specific, named GPU consequence. Checking them individually is more precise than checking the composite atom.
  

  
    Matrix6×6

    
      Algebraic Property Intersection Matrix

      "When two algebraic properties co-occur in a combining operation, what specific mathematical structure and GPU capability does their intersection produce?"

    

  

  
    How to read this matrix. Each cell names the mathematical structure that emerges when the row property and column property both hold for a combining operation, and the GPU capability that structure enables. The matrix is symmetric (row × column = column × row), so only the upper triangle is populated. When you encounter a new combining operation, check each property individually, then look up every pairwise intersection of the properties that hold — each intersection tells you a specific optimization or capability that is available. If a property is absent, every intersection involving that property is unavailable, and the corresponding GPU capability is lost. This matrix sits beneath the Atom Intersection Optimization Matrix: the atom-level intersections are built from these property-level intersections the same way the atoms are built from these properties.
  

  
    
      
        
          
          BO
          ASC
          IDE
          COM
          INV
          DIS
        

      
      
        
        
          BO
          Magma(closure only)
          
            Parallel tree

            Regrouping freedom enables tree-shaped computation: depth drops from N to log₂N. The single most important structural transformation in GPU computing — sequential becomes parallel.

            GPU: Reduction tree possible. Without ASC: depth N (sequential). With ASC: depth log₂N (parallel). Exponential speedup from one algebraic property.

          
          
            Safe padding

            A "do nothing" value exists. Non-power-of-2 inputs can be padded without affecting the result. Accumulators can be initialized to a known-safe starting value.

            GPU: PRD mask fills inactive lanes with identity. acc = 0.0f for sum, -INFINITY for max. No special-case logic for boundary tiles.

          
          
            Any-order eval

            Two elements can be combined in either order. But without associativity, still can't regroup into a tree — just evaluate the same sequential chain in either direction.

            GPU: Lanes can evaluate in any order, but still need sequential accumulation without ASC. Limited parallelism — order-free but not tree-free.

          
          
            Undo exists

            Every operation can be reversed. But without associativity, reversal doesn't compose cleanly across a chain of operations.

            GPU: Single-step error correction or rollback. Rare in ML kernels. Present in checkpoint-and-restore patterns.

          
          
            Two ops linked

            One operation distributes over another: a×(b+c) = a×b + a×c. Two separate operations become one coherent structure — the foundation of the A × B + C formula.

            GPU: FMA instruction — two operations fused into a single hardware cycle. The hardware expression of distributivity. Every acc += a * b is this intersection.

          
        

        
        
          ASC
          
          Semigroup(tree only)
          
            Monoid = MON atom

            Tree reduction + clean init. Accumulator starts at identity. Blelloch sweep possible: up-sweep builds tree, identity at root, down-sweep propagates prefixes.

            GPU: __shfl_up_sync + __shfl_xor_sync Blelloch. acc = 0.0f or -INFINITY. 2×log₂N barriers per sweep. Prefix sum, running count.

          
          
            Any-permutation tree = SRG atom

            Regroup freely AND reorder freely. Any permutation of the reduction tree gives the same answer. This IS the permutation symmetry that makes the shuffle tree valid.

            GPU: __shfl_down_sync shuffle tree. Any butterfly pattern valid. O(log N), no smem needed within warp, no barrier within warp. The SRG atom's core mechanism.

          
          
            Reversible tree

            Tree reduction where every node can be undone. Can recover individual inputs from any partial result in the tree.

            GPU: Reversible reduction. Cryptographic kernels, error-correcting codes. Rare in ML kernels but present in gradient checkpointing.

          
          
            Tiling correctness

            Split a dimension into tiles, compute partial results per tile, combine partials. Distributivity guarantees tile boundaries don't affect the answer. The mathematical reason the Tile molecule works.

            GPU: Tiled GEMM is correct. K-dim split into BK tiles, partial sums recombine via acc += partial. Without DIS, tiling produces wrong answers.

          
        

        
        
          IDE
          
          
          Identity(padding only)
          
            Predicated shuffle

            Inactive lanes filled with identity + any-order evaluation. Masked reduction produces correct result regardless of which lanes are active or inactive.

            GPU: The PRD × SRG intersection. Boundary tiles: inactive threads hold identity (0.0f for sum, -INF for max). Shuffle tree runs uniformly — no branch, no warp divergence.

          
          
            Full group

            Identity + inverse together: every element has a unique "undo" that returns to the identity. The structure that makes modular arithmetic, rotation, and cyclic indexing possible.

            GPU: RoPE's rotation group (rotate + inverse rotation = identity). Ring buffers and circular page indexing. Modular arithmetic in cryptographic kernels.

          
          
            Zero annihilation

            The identity of addition (0) annihilates multiplication: 0 × a = 0. Entire branches of computation can be skipped when an input is the additive identity.

            GPU: Sparsity optimization. If score = 0, skip the entire multiply-accumulate for that element. Sparse attention skips zero-score KV tiles. The mathematical basis for structured sparsity.

          
        

        
        
          COM
          
          
          
          Commutative(order-free only)
          
            Abelian group

            Full reorder + full undo. The structure of ordinary integer arithmetic under addition. Any rearrangement and any undoing gives the same result. The richest single-operation algebraic structure common in GPU computing.

            GPU: atomicAdd valid across blocks — concurrent writes in any order are correct because COM + INV together guarantee the final sum is independent of arrival order and correctable. The intersection that makes cross-block reduction work without a second kernel launch.

          
          
            Independent output lanes

            Each output position computable independently and in any order. No communication needed between output elements. The AREA dimension pattern: computing at position m is structurally identical to computing at position m+1.

            GPU: Grid-level parallelism. Each block computes its output tile independently. The SYM translational symmetry of AREA dimensions. grid.x = B, grid.y = num_tiles_M — each block is a self-contained computation.

          
        

        
        
          INV
          
          
          
          
          Invertible(undo only)
          
            Field structure

            Division distributes over addition. Full arithmetic with all four operations (+, −, ×, ÷). Exact linear solvers become possible. The richest two-operation algebraic structure.

            GPU: LU factorization, Gaussian elimination, matrix inversion, conjugate gradient. Operations that ML largely avoids (prefers approximate methods like SGD) but scientific computing and inference-time exact solvers require.

          
        

        
        
          DIS
          
          
          
          
          
          Semiring(A×B+C)
        

      
    
  

  
    How to use this matrix during kernel derivation. When you identify a combining operation at Step 1a (dimension fate classification), check which of the six algebraic properties it satisfies. The first property — BO (binary operation, closure) — is satisfied by any combining operation; it is the starting point. The remaining five are the ones you check: does the operation have associativity? An identity element? Commutativity? Inverses? Does a second operation distribute over it? Then look up every pairwise intersection of the properties that hold. Each populated cell tells you a specific GPU capability that is available for this operation. If a property is absent, every intersection involving that property is unavailable — and the corresponding GPU capability is structurally impossible, not just unoptimized. For example: if your combining operation is max (which has BO, ASC, IDE, and COM, but lacks INV and DIS), the available intersections are BO×ASC (parallel tree), BO×IDE (safe padding), BO×COM (any-order evaluation), ASC×IDE (monoid — Blelloch sweep valid), ASC×COM (any-permutation tree — shuffle tree valid), and IDE×COM (predicated shuffle — boundary masking valid). The unavailable intersections include ASC×INV (no reversible tree), IDE×INV (no group structure — no correction factor), ASC×DIS (no tiling guarantee — cannot split max across tiles and recombine unless the operation independently satisfies a weaker form of decomposability), and INV×DIS (no field structure). This tells you immediately: max-reduction supports shuffle trees and boundary masking but does not support online correction or standard tiling. The matrix makes the prediction mechanical rather than requiring algebraic expertise.
  

  
    All11
