# Foundational Idea — Geometric Determinism

The Foundational Idea

  Start Here — One Idea Drives the Entire Framework

  Every table, every step, every atom, every optimization in this framework exists to serve one foundational idea. Understanding it first makes every other piece of the document immediately clear.

  
    The Foundational Idea

    A GPU kernel can be fully derived — without writing a single line of code — whenever the structure of the computation is determined by the shape of the problem rather than the values stored in the data.

  

  Let that distinction land before reading further. Shape means the sizes and layouts of your tensors — a matrix that is 1024×512, stored row-major, being multiplied by another matrix that is 512×256. Values means what numbers are actually stored in those matrices — the weights, the activations, the attention scores. The framework works when the structure of the program — which phases execute, in what order, accessing which memory addresses — can be completely figured out from the shapes alone, before you have ever looked at a single number in the data.

  This principle has a name: Geometric Determinism. When it holds, you can sit down with a blank piece of paper, write out your tensor shapes and your combining operation, and then follow a sequence of structured questions — the steps in this framework — and arrive at a complete, correct, well-optimized kernel structure. No guessing. No trial and error. No "I'll just try this tile size and see what happens." The structure falls out of the geometry. And because the geometry belongs to the problem rather than to any particular hardware platform or programming language, the first layers of this derivation are the same whether you are targeting CUDA on an NVIDIA GPU, ROCm on AMD hardware, or XLA on a Google TPU. The hardware enters only when atoms are bound to a specific execution hierarchy — until that point, the derivation is entirely about the mathematics of the computation itself.

  
    A Concrete Example of the Difference

    Shape-determined (Geometric Determinism holds): GEMM — multiplying two matrices of known sizes. You know the output shape before you start. You know the inner dimension K will be reduced by summing products. You know every thread accesses addresses that are a linear function of its position in the grid. Nothing about the actual matrix values changes any of this. The framework can derive the complete kernel structure: grid shape, tile sizes, shared memory staging, barrier placement, warp primitive choice — all of it, from shapes alone.

    Value-determined (Geometric Determinism fails): Top-K selection — finding the K largest values in an array. You know the output will have K elements. But which K positions they come from is completely unknown until you have compared every element to every other element. The structure of the program — which elements get written to the output, at which output positions — depends on the actual values in the array. The framework cannot derive this from shapes alone, because the shape of the problem does not determine the program's behavior.

  

  What this framework can derive

  The framework covers the complete transformer inference stack and substantially more. Every kernel in this list satisfies Geometric Determinism — the structure of each computation is determined entirely by tensor shapes and combining operators, not by data values.

  
    
      Transformer Inference (complete)

      GEMM and batched GEMM · All attention variants (dense, paged KV, MLA, GQA, MQA, sparse, flash-decoding) · Layer norm, RMS norm, group norm · Rotary embeddings (RoPE) · Embedding lookup · Mixture-of-experts dispatch and routing · Elementwise activations (GELU, SiLU, ReLU, sigmoid)

    

    
      Scientific Computing

      Convolution (standard, depthwise, dilated, transposed, Winograd) · Pooling (max, average, global) · Iterative FFT (bottom-up Cooley-Tukey) · Iterative merge sort (bottom-up) · Conjugate gradient, Jacobi iteration · CSR / BSR sparse matrix-vector multiply

    

    
      Selection and Filtering

      Stream compaction / filter (threshold-based) · Top-K with bounded maximum output size · Argmax, argmin · Non-maximum suppression · Prefix scan within a block · Graph message passing with fixed adjacency structure

    

    
      Recursive Algorithms (geometry-bounded)

      Top-down recursive FFT · Top-down recursive merge sort · Blocked recursive matrix factorizations (LU, Cholesky, Strassen) — any divide-and-conquer algorithm where recursion depth is bounded by a function of the input size

    

  

  The precise boundary — five failure conditions

  The framework cannot derive a kernel when Geometric Determinism fails. After all current extensions, this happens in exactly five specific ways. Each one corresponds to a piece of structural information that the framework needs before code can be written, but that is unknowable without examining the data. Note which step each failure occurs at — this tells you exactly where in the derivation the framework hits its limit.

  

    
      Failure 1Truly UnboundedOutput StructureBreaks at Level 0

      The output tensor's shape cannot be bounded by any formula over the input shapes. The GATE extension covers selection with a known maximum (top-K where K is geometry-determined, stream compaction where maximum = N). What remains excluded is output whose maximum size is itself a runtime variable — a dynamically growing buffer, a graph traversal where the active set has no geometric upper bound, an algorithm that generates new work based on values encountered. Examples: beam search with data-driven pruning, BFS with data-driven active node sets, truly dynamic output buffers.

    

    
      Failure 2Non-Affine AddressFunction in ChainBreaks at Step 1b

      The chain depth extension (Step 1b) now covers multi-level pointer indirection where each hop is an affine function of the retrieved value — CSR SpMV, graph GNNs with fixed adjacency, depth-2+ affine chains. What remains excluded is any hop whose address function is non-affine: modular arithmetic, XOR, squaring, or any other non-linear operation on the retrieved value. Examples: hash tables using key mod N, cuckoo hashing, quadratic probing, XOR-based mixing schemes.

    

    
      Failure 3Data-DependentRecursion DepthBreaks at Level 0

      The recursion morphism field (Level 0) now covers self-similar computations where depth = f(input size) — top-down recursive FFT, recursive merge sort, blocked recursive matrix factorizations. What remains excluded is recursion whose depth depends on data values: how deep a BVH tree needs to be depends on scene clustering, how many refinement levels adaptive mesh needs depends on error estimates. Examples: BVH traversal, adaptive mesh refinement, convergence-based iterative solvers (run until residual < ε).

    

    
      Failure 4Non-AssociativeReductionBreaks at Step 1a

      The combining operation is neither a semiring (associative + commutative) nor a monoid (associative only). Without associativity, no parallel tree structure produces a correct result — the combining must be strictly sequential, which makes the parallel hardware useless for that operation. Examples: selective scan (Mamba / SSM) with data-dependent mixing matrices, LSTM recurrence with gating (sigmoid and tanh are non-linear and non-associative), any recurrence where output[i] depends non-linearly on output[i-1].

    

    
      Failure 5Dynamic WorkGenerationBreaks at Level 0

      The computation generates new work — new threads, new kernel launches, new tasks — based on results discovered during execution. The framework assumes the complete structure of the computation (grid dimensions, phase sequence, address formulas) is fixed at kernel launch time. When the structure itself changes at runtime, no Level 0 geometry spec can be written for the work that does not yet exist. Examples: CUDA dynamic parallelism, work-stealing task queues, graph algorithms that discover new active vertices during each wave.

    

  

  
    The Single Test — Apply This Before Reading Any Further

    Before using this framework for any kernel, ask one question: can I write down the complete structure of my computation — which phases execute, in what order, accessing which addresses — without looking at any data values?

    If yes, proceed. The framework will guide you from your tensor shapes to a complete, correct kernel. If no, identify which of the five failure conditions applies, because that tells you what kind of extension would be needed. The framework does not fail silently — if you try to apply it to a computation outside its boundary, the derivation will stop at a specific step and ask a question that cannot be answered from geometry alone. That stopping point is a signal, not an error. It is the framework telling you exactly where Geometric Determinism breaks down for your specific problem.

  

  For Engineers Using Triton, CUTLASS, CuTe, or PyTorch

  A Pre-Derivation Methodology — Works Alongside Your Existing Tools

  This framework operates at the step before implementation: it derives the kernel's structure from the geometry of the computation, producing a complete architectural specification — grid shape, phase sequence, barrier placement, memory staging, warp primitives. That specification is then the input to Triton, CUTLASS, CuTe, or whatever implementation tool fits your context. The two sit at adjacent levels of the same workflow and hand off cleanly.

  Triton, CUTLASS, and CuTe are genuinely excellent tools, and the depth of understanding encoded in them is remarkable. CuTe's layout algebra captures the affine map structure at the heart of GPU addressing with extraordinary precision. Triton's tile-based model reflects a deep and correct understanding of how memory hierarchies constrain kernel design. These tools represent years of hard-won knowledge about GPU hardware, and they make that knowledge accessible and reusable at scale. This framework draws on the same intellectual foundation they do.

  What this framework adds is a systematic way to arrive at the design decisions those tools then help you express. Triton and CUTLASS excel at translating a kernel design into efficient, correct code — and this framework is what you run before that translation begins, to derive the design itself from the geometry of your problem. The two operate at adjacent levels of the same workflow, handing off to each other cleanly: the framework produces the architectural specification; your tool of choice implements it. Using them together gives you both the rigorous derivation and the powerful implementation machinery.

  
    Hardware Agnostic · Language Agnostic · Company Agnostic

    The first layers of this framework — Level 0 through Step 1b — are entirely about the problem, not the hardware. They ask what the geometry of the computation is, what the algebraic structure of each reduction is, and how memory is addressed. None of those questions have answers that depend on whether you are targeting CUDA, ROCm, a Google TPU, or any other platform. The geometry of a matrix multiplication is the same regardless of which chip executes it.

    The hardware enters at Step 2, when atoms are bound to a specific execution hierarchy — and even there, the binding table simply gets different rows for different hardware. The thread/warp/block/grid column names change; the underlying dimension fate analysis and algebraic classification from the earlier steps stay identical. The examples in this document use CUDA because CUDA is the most widely used GPU programming language, but every derivation from Level 0 through Step 1b applies equally to ROCm kernels on AMD hardware, to XLA programs targeting TPUs, or to any future parallel computing platform built on the same mathematical foundations.

  

  
    
      1 · This Framework

      Derive the kernel structure from the geometry of your problem — phase sequence, barrier placement, memory staging, tile sizes, warp primitive choice. Done entirely on paper before any code is written. Output: a complete architectural specification.

    

    
      2 · Your Implementation Tool

      Implement the derived structure using whichever tool fits your platform — Triton, CUTLASS / CuTe, or raw CUDA on NVIDIA; HIP, ROCm, or composable_kernel on AMD; XLA or MLIR-based stacks on TPUs and custom accelerators. The framework's output is the architectural specification your tool implements. Nothing about your existing workflow changes.

    

    
      3 · Profile and Tune

      Profile with Nsight Compute on your target hardware. Apply hardware-specific tuning. This step is not optional and this framework does not replace it. See the note below — this is the most important thing on this page.

    

  

  
    Profiling Is Irreplaceable — This Cannot Be Stated Strongly Enough

    This framework derives correct kernel structure. It tells you where barriers go, how memory is staged, what the phase sequence is, and which warp primitives to use. What it cannot tell you — and what no pre-planning methodology can ever tell you — is how fast your kernel actually runs on your specific hardware.

    Occupancy, register spill counts, instruction throughput, vectorization width, L2 cache hit rates, and warp stall distributions are not derivable from geometry. They are measured. The framework's scope ends precisely where systematic derivation gives way to hardware-specific measurement — and that boundary is not a weakness. It is an honest acknowledgment that the hardware knows things the geometry does not.

    Use this framework to produce a structurally correct kernel with justified design decisions. Then open Nsight Compute and let the hardware tell you what to tune. The framework gives you a correct skeleton. Profiling gives you a fast one. Both are necessary. Neither is sufficient alone.

  

  An analogy: an architect's blueprints tell you what to build and why each structural decision was made. They do not tell you how the building will perform under real operating conditions — that requires measuring instruments and iterative testing on the actual structure. The blueprints make the construction process systematic and every design decision defensible. The measurement process makes the result excellent. This framework is the blueprints.

  
    What Isan Atom?

    
      What an Atom Is and Why These Eleven Are the Right Ones

      "Why call them atoms? What makes something an atom rather than just a useful pattern? And why do these eleven appear in GPU computing and not some other set?"

    

  

  
    The word atom has a precise meaning here that is worth establishing before you encounter the specific eleven. In chemistry, atoms are not the smallest things that exist — protons and electrons exist below them — but they are the smallest things that retain the properties that chemistry cares about. A hydrogen atom is the smallest unit that behaves like hydrogen. Split it further and you no longer have chemistry; you have physics. The atom is defined not by being the absolute bottom of reality but by being the smallest unit that is meaningful and nameable at a particular layer of description.

    The same logic applies here. In GPU computing, binary digits exist below the atoms in this framework. PTX assembly instructions exist below them. But neither binary nor PTX gives a programmer a unit they can hold in their mind and reason with when designing a kernel. You cannot look at a sequence of binary bits and immediately know whether they represent a semiring reduction or an affine address computation. The atoms in this framework are the layer where GPU computation first becomes humanly legible — the smallest units that a kernel engineer can recognize, name, agree on, and use as the starting point for structured reasoning.

  

  
    The defining properties of an atom — what separates an atom from just a useful pattern.
    
    Three properties together distinguish an atom from a pattern that happens to be common. First, an atom is irreducible within its layer — it cannot be expressed as a composition of other atoms at the same layer without losing the essential property that makes it useful. The affine map cannot be broken into simpler address operations while remaining a coherent unit of reasoning. The semiring cannot be replaced by combining the predicate and some other operation. Each atom is genuinely primitive at its layer; that is the criterion that limits the count to a small, stable set rather than growing indefinitely as you notice more and more useful patterns. Note that "irreducible within its layer" does not mean "has no internal structure." Two of the atoms — the semiring and the monoid — are composed from algebraic properties at the layer below (associativity, identity, commutativity, inverses, distributivity), as shown in the preceding section. The atom is the level where those properties become a single nameable unit that a kernel engineer can reason with. Below the atom, the individual properties are too fine-grained for practical kernel design. Above the atom, the molecules are too coarse to distinguish between fundamentally different hardware primitives.
    
    Second, an atom requires community agreement to become an atom. An atom is not discovered by one person working alone — it is crystallized when a community of practitioners reaches consensus that this pattern deserves a name because it keeps appearing, cannot be reduced further, and makes reasoning easier once named. The GPU programming community had been implicitly using the affine map, the semiring, and the monoid for decades before this framework named them as atoms. The patterns were always there in CUDA tutorials, in FlashAttention papers, in CUTLASS design decisions. The naming is the crystallization event — the moment a recurring pattern stops being a private intuition and becomes a shared unit of communication.
    
    Third, an atom sits at a specific layer boundary — below it is a lower layer of description that requires different vocabulary to reason about, and above it is the grammar that composes atoms into the phenomena visible at the next layer up. The atoms in this framework sit between the algebraic properties (the layer below, where individual mathematical rules like associativity and commutativity describe the raw capabilities of combining operations) and CUDA programming patterns like tiling and pipelining (the layer above, which is entirely composed from these atoms). The algebraic properties are too fine-grained for a kernel engineer to track during design — you do not want to separately verify associativity, identity, and commutativity at every step. The atoms bundle the properties into nameable units (SRG, MON) that sit at the right granularity for practical reasoning. Remove the atoms and the gap between the algebraic foundations and CUDA patterns becomes a void with no systematic bridge.
  

  
    With those three properties in mind, a deeper question arises: why do these specific eleven atoms keep appearing in GPU computing, rather than some other set? The short answer is that they are not GPU-specific inventions. They are descriptions of constraints that arise wherever many independent agents cooperate to produce a collective result — and GPU kernels are one particularly clean instance of that general structure. The affine map appears in neural network equations, fluid simulation discretizations, and signal processing for the same reason it appears in CUDA: it is the unique simplest structure that bridges two coordinate systems while preserving linear relationships, and that requirement appears across all of those domains. The semiring's shuffle tree is O(1) per element not because NVIDIA designed good hardware, but because the mathematical properties of associativity and commutativity are precisely the conditions that make free parallel combination possible — the same reason the same structure appears in database aggregation, distributed systems, and Monte Carlo simulation. This cross-domain connection is not incidental context. It is the reason the framework's atoms are durable: they are forced by the mathematics of the problem, not chosen by the engineers who designed the hardware. Before examining the eleven atoms individually, the next section looks at the algebraic properties beneath them — the six mathematical rules whose specific combinations produce the SRG and MON atoms and determine which GPU optimizations each combination enables.
