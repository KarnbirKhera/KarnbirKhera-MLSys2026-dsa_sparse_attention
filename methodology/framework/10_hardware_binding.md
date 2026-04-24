# Step 2 — Hardware Binding Table

atoms activated at Step 2 — all 11 active

    
      
        AFM

        Primary. Each Granularity Table cell gives one affine map coefficient: blockIdx.x × BM is the grid-level AFM term; threadIdx.x is the thread-level term. Choose tile sizes now — you are choosing the coefficients of the affine maps that will appear in Step 5. Check AFM×MEA (Staging Threshold) before finalizing tile sizes: does tile measure = smem capacity?

      

      
        SRG

        Primary. Warp→Primitives column: enter "shuffle tree / __shfl_down_sync" for SRG REDUCE dimensions. This is the FUN functor application: SRG classification → shuffle tree primitive, automatically. Check SRG×MEA (Reduction Level): if NUM_WARPS > 1, add warp_partial buffer to Block→Shared and note two extra FSM sync points.

      

      
        MON

        Primary. Warp→Primitives column: enter "Blelloch / __shfl_up_sync + __shfl_xor_sync" for MON REDUCE dimensions. Add smem[BLOCK_SIZE] to Block→Shared for Blelloch sweep. Note 2×log₂N extra barriers in FSM. MON×MEA: the smem array is the measure of the Blelloch algorithm's working memory.

      

      
        PRD

        Primary. Grid→Global column: note boundary dimensions (those whose size is not a multiple of the tile size) — these generate predicates at the block or tile level. Verify that conditional leaf types from Step 1b are correctly assigned: grid-level iterator → early exit in Grid→Global column; tile-level iterator → predicated load inside Warp or Thread phases.

      

      
        ATO

        Appears at Grid→Global boundary if cross-block reduction is required. Check ATO×MEA (Contention Measure) before choosing atomics over a second kernel launch: contention = threads × P(address conflict). High contention → second kernel; low contention → atomic. Check ATO×SYM (Symmetric Atomics) — if addresses are symmetric across warp lanes, elect one representative.

      

      
        FXP

        The REDUCE dimension's tile size determines the fixed point iteration count: num_tiles = ceil(REDUCE_dim / tile_size). Write this in the Granularity Table's REDUCE row alongside the tile size. If the fixed point is online (carries state), add the state variables to Thread→Registers now. Check FXP×MEA (Iteration Bound): is the count static? If yes, mark for loop unrolling in Code Derivation.

      

      
        MOR

        Apply the smem layout morphism identified at Step 1b: if the global layout causes bank conflicts in smem, write the morphism rule (XOR swizzle, +1 padding, transpose) in the Block→Shared column alongside the buffer dimensions. This morphism changes the smem address formula without changing the data values. Check MOR×MEA (Compression Staging): can the dtype morphism (FP32→FP8) be applied before staging to fit more data in smem?

      

      
        MEA

        Four measure checks are mandatory: (1) Smem budget: Σ(Block→Shared buffer sizes) ≤ hardware limit. (2) Register budget: Σ(Thread→Register state variable sizes) ≤ 64 floats target. (3) Reuse counts: computed for every Block→Shared buffer as ∏(absent AREA dims) — verify > 1 for all staged buffers. (4) Occupancy: given smem_total and reg_total, compute max resident blocks per SM. Check MEA×REL (Joint Budget Check) for Combine Group buffers: their budgets are coupled.

      

      
        REL

        Add a Relations column to the Binding Table alongside Grid/Block/Warp/Thread. For each pair of buffers in the Block→Shared column that are related (Combine Group members): write the REDUCE_MERGE relation explicitly. For each pair of access patterns that are coalescing-related (stride-1 at warp level): mark COALESCED. For each cross-phase write→read pair: mark BARRIER REQUIRED (feeds directly into Step 3 barrier placement). Check REL×MEA (Joint Budget Check) for Combine Group members.

      

      
        SYM

        For each AREA dimension: confirm its translational symmetry means no inter-block communication (zero synchronization cost along this dimension). For the smem buffer assigned to each REDUCE dimension: verify the access pattern breaks harmful bank symmetry (apply the MOR from Step 1b if needed). Check SYM×AFM (Guaranteed Coalescing) for each buffer at the warp level — if the lane-stride = 1, coalescing is guaranteed by symmetry, no further analysis needed.

      

      
        FUN

        The Binding Table is where the framework functor applies most explicitly. Each level transition in the table is a functor application: Grid→Block applies the block-tile functor (substitute blockIdx × B_tile for the grid-level coordinate). Block→Warp applies the warp-primitive functor (SRG → shuffle tree, MON → Blelloch). Warp→Thread applies the thread-tile functor (substitute threadIdx × T_tile for the warp-level coordinate). Write each functor action explicitly in the corresponding column. Check FUN×MEA (Budget Transformation): resource measures transform through each functor application — verify consistency at every level.

      

    

  

  
    Step3

    
      FSM — Phase Ordering, Barriers, GATE Phases, and Chain Phases

      "Sequence the phases. Place barriers. Add REDUCE_MERGE for Combine Groups. Add GATE phases for GATE dimensions. Add chain hop phases for depth->1 lookups. Verify fixed point state variables survive all boundaries."
