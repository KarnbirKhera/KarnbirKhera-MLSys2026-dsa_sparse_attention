# §9.7.16.11–12 tcgen05.fence and tcgen05.commit

*PTX ISA, Release 9.2 — pages 743–745*

---

## §9.7.16.11 TensorCore 5th Generation Specialized Synchronization Operations

### §9.7.16.11.1 tcgen05.fence

**Purpose:** Specialized fence for ordering asynchronous tcgen05 operations across threads. Does NOT wait for MMA hardware completion — use `tcgen05.commit` for that.

#### Syntax

```ptx
tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync ;
```

#### Description

`tcgen05.fence::before_thread_sync` orders all **prior** asynchronous tcgen05 operations with respect to subsequent tcgen05 and execution ordering operations.

`tcgen05.fence::after_thread_sync` orders all **subsequent** asynchronous tcgen05 operations with respect to prior tcgen05 and execution ordering operations.

The `tcgen05.fence::*` instructions compose with execution ordering instructions across a thread scope and provide ordering between tcgen05 instructions across the same scope.

`tcgen05.fence::before_thread_sync` behaves as a code motion fence — prior tcgen05 instructions cannot be hoisted across it. `tcgen05.fence::after_thread_sync` behaves as a code motion fence — subsequent tcgen05 instructions cannot be sunk across it.

#### PTX ISA Notes

Introduced in PTX ISA version 8.6.

#### Target ISA Notes

Supported on: sm_100a, sm_101a (renamed to sm_110a from PTX ISA 9.0), sm_100f or higher in the same family, sm_101f or higher in the same family (renamed to sm_110f from PTX ISA 9.0), sm_110f or higher in the same family.

#### Example

```ptx
// Producer thread: write data into TMEM via tcgen05.cp, then signal consumer
tcgen05.cp.cta_group::1.128x256b [taddr0], sdesc0;
tcgen05.fence::before_thread_sync;   // prior cp must be visible before flag write
st.relaxed.b32 [flag], 1;

// Consumer thread: wait for flag, then fence before consuming with MMA
loop:
    ld.relaxed.b32 r, [flag];
    setp.eq.u32 p, r, 1;
    @!p bra loop;
tcgen05.fence::after_thread_sync;    // ensure cp result is visible before MMA
tcgen05.mma.cta_group.kind [taddr0], adesc, bdesc, idesc, p;
```

---

## §9.7.16.12 TensorCore 5th Generation Async Synchronization Operations

### §9.7.16.12.1 tcgen05.commit

**Purpose:** Makes the mbarrier object track the completion of all prior async-tcgen05 operations initiated by the executing thread. This is the mechanism for waiting until the MMA hardware has finished writing results to TMEM.

#### Syntax

```ptx
tcgen05.commit.cta_group.completion_mechanism{.shared::cluster}{.multicast}.b64
    [mbar] {, ctaMask};

.completion_mechanism = { .mbarrier::arrive::one }
.cta_group            = { .cta_group::1, .cta_group::2 }
.multicast            = { .multicast::cluster }
```

#### Description

`tcgen05.commit` is an asynchronous instruction that makes the mbarrier object at address `mbar` track the completion of all prior asynchronous tcgen05 operations (`.mma`, `.cp`, `.shift`) initiated by the executing thread. When those operations complete, the hardware triggers the specified `.completion_mechanism` on the mbarrier.

`tcgen05.commit.cta_group::1` tracks completion of all prior `.cta_group::1` operations from the current thread. `tcgen05.commit.cta_group::2` tracks completion of all prior `.cta_group::2` operations.

All tcgen05 instructions within a kernel must use the same `.cta_group` value.

The qualifier `.mbarrier::arrive::one` causes an arrive-on operation with count=1 to be signaled on the mbarrier upon completion. The scope of the arrive-on operation is cluster scope.

The optional `.multicast::cluster` qualifier allows signaling mbarrier objects in multiple CTAs simultaneously. `ctaMask` is a 16-bit mask where each bit corresponds to the `%cluster_ctarank` of a destination CTA; the signal is multicast to the same `mbar` offset in each destination CTA's shared memory.

If no state space is specified, Generic Addressing is used. If `mbar` does not fall within `.shared::cluster` address space, behavior is undefined.

#### PTX ISA Notes

Introduced in PTX ISA version 8.6.

#### Target ISA Notes

Supported on: sm_100a, sm_101a (renamed to sm_110a from PTX ISA 9.0), sm_100f or higher in the same family, sm_101f or higher in the same family (renamed to sm_110f from PTX ISA 9.0), sm_110f or higher in the same family.

#### Examples

**Example 1 — cp → commit → mbarrier wait:**
```ptx
tcgen05.cp.cta_group::1.128x256b [taddr0], sdesc0;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbarObj1];
loop:
    mbarrier.try_wait.parity.b64 p, [mbarObj1], 0;
    @!p bra loop;
// cp is now guaranteed complete; safe to read taddr0
```

**Example 2 — mma → commit → mbarrier wait (the canonical TMEM readback pattern):**
```ptx
tcgen05.mma.cta_group::2.kind::tf32 [taddr0], adesc, bdesc, idesc, p;
tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [mbarObj2];
loop:
    mbarrier.try_wait.parity.b64 p, [mbarObj2], 0;
    @!p bra loop;
// MMA is now guaranteed complete; safe to call tcgen05.ld on taddr0
```
