# The Third Dimension Fate — GATE & Gate Specification Table

The Third Dimension Fate — GATE
    What GATE means: A dimension is GATE when each element's fate — whether it appears in the output — depends on its value relative to other elements. You cannot know which elements survive until you have examined the data.

    The clearest way to see the difference between the three fates is through a concrete example. Suppose you have a list of 1000 test scores. An AREA operation keeps every score independently — computing score[5] does not affect score[7]. A REDUCE operation combines all 1000 scores into a single average — every score contributes to the same final value. A GATE operation keeps only the top 50 scores — but you cannot know which 50 they are without comparing all 1000 to each other first.

  

  
    Beginner analogy — the audition versus the assembly line.
    
    An AREA dimension is like an assembly line where each station works independently. Station 3 doesn't need to know what station 7 is doing. A REDUCE dimension is like a voting system where everyone contributes to one final decision. A GATE dimension is like an audition where you can't know who makes the cast until you've seen everyone perform. The GATE's defining property is that elements compete — the fate of one element depends on how it compares to all the others.
    
    This competition is precisely what makes GATE different from a predicate (PRD). A predicate checks whether one element satisfies a fixed threshold — "is this score above 70?" — and doesn't require looking at any other element. A GATE checks whether one element belongs to the best group — "is this score in the top 50?" — which does require looking at all other elements to know what "top 50" means.
  

  
    Why GATE needs to be a formal dimension fate, not just a predicate
    Without naming GATE as a distinct fate, the framework has no way to derive the two-phase structure that all GATE computations require. Phase 1 evaluates each element's criterion in parallel (this is a semiring-like operation — independent per element). Phase 2 assigns output positions to survivors using a prefix sum (this is a monoid operation — ordered, because position 5 in the output depends on how many elements before it survived). The two phases use different atoms, execute in a mandatory sequence, and produce different data types. Calling GATE a "special predicate" would obscure all of this and lead to incorrect phase structure derivation.
  

  When you identify a GATE dimension, you must complete the Gate Specification Table before proceeding. This table captures exactly what the gate does and derives the two phases it always requires.

  
  
    
      
        
          Field
          Question to Answer
          Atoms Active
          What It Drives Downstream
        

      
      
        
          Gate Criterion
          What condition must an element satisfy to survive? Write it precisely: "score > threshold," "value is in top-K by magnitude," "absolute value exceeds running maximum." The criterion determines whether Phase 1 is independent per element (→ semiring evaluation) or requires comparing elements to each other (→ requires a separate sorting or ranking step).
          PRD — the criterion is a generalized predicate evaluated per element. SRG if the criterion is a threshold comparison (fully independent per element). MON if the criterion involves ranking (elements must be compared to their neighbors in some order).
          Determines Phase 1's structure in the FSM and which warp primitive implements it.
        

        
          Maximum Output Size
          What is the largest number of elements that can survive the gate? For top-K, this is K exactly. For threshold filters, this is the full input size N (every element could theoretically survive). Write a static upper bound — this is how large the output buffer must be allocated, even if the actual survivor count at runtime is smaller.
          MEA — the maximum output size determines the output buffer measure. PRD — the buffer is allocated at maximum size; unused positions are predicated out of the final write.
          Step 2 binding table: the output buffer is sized to the maximum, not the actual runtime size. This is the key difference between a GATE output dimension and an AREA output dimension in the granularity table.
        

        
          Criterion Parallelizability
          Can each element's criterion be evaluated independently of other elements' results? For threshold filters and absolute-value comparisons, YES — each element's decision requires no knowledge of any other element's value. For rank-based criteria like top-K, NO — to know whether element i is in the top-K, you need to know the K-th largest value across all elements.
          SRG if YES — Phase 1 is a free-parallel evaluation, same as a semiring reduction. MON if NO — Phase 1 requires a sorted or ranked intermediate, which is a monoid-like ordered operation.
          Determines whether Phase 1 uses a shuffle tree (SRG, O(1) per element) or a bitonic/radix sort (MON, O(log²N) per element).
        

        
          Phase 1: Criterion Evaluation
          Derived automatically from the Criterion and Parallelizability fields. Write the specific operation: "threshold comparison producing a boolean mask," "warp-level maximum-finding reduction," "radix sort to rank elements." This is the operation that produces the boolean mask or the ranked order that Phase 2 will use to assign output positions.
          SRG + PRD + AFM — parallel criterion evaluation with affine addressing and identity substitution for non-survivors.
          Step 3 FSM: Phase 1 is the first phase in the GATE's two-phase sequence. A barrier follows Phase 1 before Phase 2 can begin — Phase 2 cannot start until all of Phase 1's boolean mask values are written and visible.
        

        
          Phase 2: Index Assignment
          Derived automatically once Phase 1 is defined. Always: an exclusive prefix sum over the boolean mask produced by Phase 1. The prefix sum converts a boolean mask like [1, 0, 1, 1, 0, 1] into output positions [0, -, 1, 2, -, 3] — each surviving element learns its position in the compacted output array. Write: "exclusive prefix sum over Phase 1 boolean mask."
          MON — prefix sum is always a monoid operation (associative, non-commutative — position 3 in the output depends on how many elements before position 3 survived, so order must be respected). AFM — the output write address for each survivor is their prefix-sum result × output element stride.
          Step 3 FSM: Phase 2 follows Phase 1's barrier. Phase 2's FSM structure is the Blelloch two-phase sweep (up-sweep + down-sweep), which is the standard MON template already in the framework.
        

        
          REL between phases
          Automatically derived: Phase 1 and Phase 2 are always related by a mandatory ordering constraint. Phase 2 cannot begin until Phase 1's boolean mask is complete and visible to all threads. Write this as a Combine Group relation: {Phase 1 mask, Phase 2 index assignment} → mandatory barrier between them in Step 3.
          REL — the most important relation in the GATE specification. The relation between Phase 1 and Phase 2 is what forces the barrier. FXP — the two-phase GATE pattern is a bounded fixed point of depth 2: always exactly two phases in this order, never more, never fewer.
          Step 3 FSM: the REL between phases adds one mandatory barrier between Phase 1 and Phase 2. This barrier is not optional and cannot be moved. Removing it produces a race condition where Phase 2 reads incomplete mask data from Phase 1.
        

      
    
  

  
    Top-K example — Gate Specification Table completed:
    Gate Criterion: "value has rank among top K by magnitude." Maximum Output Size: K (exactly K survivors, no variability). Criterion Parallelizability: NO — knowing whether element i is in the top K requires knowing the K-th largest value, which requires examining all N elements. Phase 1: radix sort or bitonic sort to rank all N elements by magnitude (MON — ordered operation). Phase 2: exclusive prefix sum over the binary "is this element in the top K?" mask — assigns contiguous output positions 0 through K-1 to the K survivors (MON). REL: mandatory barrier between Phase 1 (sort) and Phase 2 (index assignment). Note: this is why top-K requires warp min-heaps or radix select — the Phase 1 criterion is not a simple threshold comparison but a rank-based operation, which is a MON rather than a SRG.
    
    Stream compaction example — Gate Specification Table completed:
    Gate Criterion: "value satisfies threshold condition predicate(x)." Maximum Output Size: N (worst case: all elements survive). Criterion Parallelizability: YES — each element's threshold check is independent of all other elements. Phase 1: parallel predicate evaluation producing boolean mask (SRG / PRD — one independent comparison per element, no inter-element communication). Phase 2: exclusive prefix sum over boolean mask (MON — ordered). REL: barrier between Phase 1 and Phase 2.
  

  
  
  

  
    
      FXP
      × GATE → Streaming Gate
      ★ New in v8 — required when GATE is nested inside a tile loop
    

    
      The Gate Specification Table above describes the single-pass Gate: all candidates are available at once, you evaluate the criterion, run Blelloch, and write the output. That works perfectly when the candidates all exist in memory before the kernel starts — for example, finding the top-K elements in an array that is fully loaded into shared memory.
    

    
      But some kernels process candidates in batches across a tile loop. The candidates for the Gate dimension do not all exist at once — they arrive one batch at a time, page by page, tile by tile. The kernel must compare each new batch against everything it has seen so far and maintain a running best-K list across all iterations. This is a fundamentally different problem that the single-pass Gate FSM cannot handle. It needs its own named variant: the Streaming Gate.
    

    
      How to detect it: Ask one question — are the GATE dimension's candidates produced inside the FXP tile loop, or do they all exist before the loop starts? If inside, you have a Streaming Gate. The top-K sparse attention indexer is the canonical example: scores for each KV cache page are computed one page at a time inside the page loop; you cannot evaluate the top-K criterion until you have processed all pages, but you also cannot store all scores before starting.
