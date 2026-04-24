# §9.7.16.6 Memory Consistency Model for 5th generation of TensorCore operations

*PTX ISA, Release 9.2 — pages 657–662*

Ordering of tcgen05 instructions is described in terms of two key concepts:

1. Pipelined tcgen05 instructions
2. Specialized tcgen05-specific inter-thread synchronization mechanisms.

These concepts combine to form four canonical synchronization patterns, as described further below.

---

## §9.7.16.6.1 Asynchronous Operations

The tcgen05 family of instructions are divided into 2 categories:

1. **Asynchronous instructions:** These tcgen05 operations are not inherently ordered with respect to other tcgen05 operations in the same thread (unless pipelined as mentioned below).
2. **Synchronous instructions:** These tcgen05 operations are inherently ordered with respect to other tcgen05 operations in the same order.

The Tensor Memory allocation related instructions that access shared memory maintain same-address ordering with respect to non-tcgen05 instructions.

The following table lists the category of each tcgen05 instruction:

| tcgen05.* operation          | Category                  |
|------------------------------|---------------------------|
| `.alloc`                     | Synchronous instructions  |
| `.dealloc`                   |                           |
| `.relinquish_alloc_permit`   |                           |
| `.fence::*`                  |                           |
| `.wait::*`                   |                           |
| `.commit`                    |                           |
| `.mma`                       | Asynchronous instructions |
| `.cp`                        |                           |
| `.shift`                     |                           |
| `.ld`                        |                           |
| `.st`                        |                           |

> **Key implication:** `tcgen05.mma` and `tcgen05.ld` are both asynchronous. There is no implicit ordering between them — explicit synchronization is always required when reading TMEM after an MMA. See §9.7.16.6.4.2 for the canonical fix pattern.

---

## §9.7.16.6.2 Pipelined tcgen05 Instructions

The asynchronous tcgen05 operations may execute and complete in a different order than they were issued. However, some specific pairs of asynchronous tcgen05 instructions form tcgen05 pipelines, where the two operations are guaranteed to execute in the same order as the instructions that issued them. The specific pairings are:

1. `tcgen05.mma.cta_group::N` → `tcgen05.mma.cta_group::N` (same N, accumulator, shape, and kind)
2. `tcgen05.cp.cta_group::N` → `tcgen05.mma.cta_group::N` (same N)
3. `tcgen05.shift.cta_group::N` → `tcgen05.mma.cta_group::N` (same N)
4. `tcgen05.shift.cta_group::N` → `tcgen05.cp.4x256b.cta_group::N` (same N)
5. `tcgen05.mma.cta_group::N` → `tcgen05.shift.cta_group::N` (same N)

> **Notable absence:** `tcgen05.mma → tcgen05.ld` is NOT a pipelined pair. Reading TMEM after MMA is always a non-pipelined operation and requires explicit synchronization.

### §9.7.16.6.2.1 Implicitly pipelined tcgen05 Instructions

Instructions `tcgen05.commit` and `tcgen05.wait` are implicitly pipelined with respect to previously issued `tcgen05.{mma,cp,shift}` and `tcgen05.{ld,st}` instructions respectively, from the same thread.

### §9.7.16.6.2.2 mbarrier based completion mechanism

Completion of the following instructions' asynchronous operations is observed through the mbarrier based waiting mechanism:

1. `tcgen05.mma`
2. `tcgen05.cp`
3. `tcgen05.shift`

`tcgen05.commit` is used to track the completion of the above asynchronous instructions.

The following are the implicitly pipelined pairings that use the mbarrier based completion mechanism:

- `tcgen05.mma.cta_group::N` → `tcgen05.commit.cta_group::N` (same N)
- `tcgen05.cp.cta_group::N` → `tcgen05.commit.cta_group::N` (same N)
- `tcgen05.shift.cta_group::N` → `tcgen05.commit.cta_group::N` (same N)

### §9.7.16.6.2.3 tcgen05.wait instruction based completion mechanism

Completion of the following instructions' asynchronous operations is observed through the `tcgen05.wait` based waiting mechanism:

1. `tcgen05.ld`
2. `tcgen05.st`

`tcgen05.wait::ld` and `tcgen05.wait::st` track completion of `tcgen05.ld` and `tcgen05.st` respectively.

Implicitly pipelined pairings using `tcgen05.wait`:

- `tcgen05.ld` → `tcgen05.wait::ld`
- `tcgen05.st` → `tcgen05.wait::st`

---

## §9.7.16.6.3 Specialized Inter-thread Synchronization for tcgen05 instructions

The tcgen05 instructions support a specialized inter-thread synchronization optimized for the tcgen05 family. Standard memory consistency model synchronization mechanisms also apply.

`tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` compose with execution ordering instructions (morally strong ld/st/atom instructions, mbarrier instructions, barrier instructions, etc.) to establish ordering between tcgen05 operations across threads. Asynchronous tcgen05 instructions ordered across threads also form a tcgen05 pipeline.

- An asynchronous tcgen05 operation **prior to** a `tcgen05.fence::before_thread_sync` is ordered before all subsequent tcgen05 and execution ordering operations.
- An asynchronous tcgen05 operation **subsequent to** a `tcgen05.fence::after_thread_sync` is ordered after all prior tcgen05 and execution ordering operations.

> **Important:** The fence instructions handle *cross-thread visibility of already-completed TMEM state*. They do NOT wait for the MMA hardware pipeline to finish writing. Use `tcgen05.commit` + mbarrier wait for that purpose.

---

## §9.7.16.6.4 Canonical synchronization patterns

### §9.7.16.6.4.1 Pipelined instructions, same thread

No explicit ordering mechanism is needed; the pipelined instruction pairing provides the guarantee.

```ptx
tcgen05.mma
tcgen05.mma  // same shape and accumulator — executes in program order
```

### §9.7.16.6.4.2 Non-pipelined instructions, same thread

Explicit waiting mechanisms are required to wait for completion of asynchronous tcgen05 operations.

**Example 1 (st → ld):**
```ptx
tcgen05.st
tcgen05.wait::st        // wait for st to complete before proceeding
tcgen05.ld
```

**Example 2 (mma → ld) — the canonical pattern for reading TMEM after MMA:**
```ptx
tcgen05.mma [d], ...
tcgen05.commit.mbarrier::arrive::one          // track MMA completion via mbarrier
mbarrier.try_wait.relaxed.cluster             // spin until MMA hardware is done
tcgen05.fence::after_thread_sync              // order: subsequent ld sees completed TMEM
tcgen05.ld [d], ...
```

Notes from the PTX ISA:
- `tcgen05.commit` is used to track completion of the asynchronous `tcgen05.mma`.
- `tcgen05.fence::after_thread_sync` is needed because `tcgen05.ld` is itself asynchronous.
- **No explicit `tcgen05.fence::before_thread_sync` is needed** — it is implicitly performed by `tcgen05.commit`.
- The combination of `tcgen05.mma` and `tcgen05.commit` forms a conceptual asynchronous pipeline and establishes execution ordering.

**Alternate form (mma producer signaling another thread):**
```ptx
tcgen05.mma [d], ...
tcgen05.fence::before_thread_sync
mbarrier::arrive
```

### §9.7.16.6.4.3 Pipelined instructions, different thread

No explicit waiting mechanism is needed, but proper inter-thread synchronization is required.

| Thread 0                                   | Thread 1                                             |
|--------------------------------------------|------------------------------------------------------|
| `tcgen05.cp`                               |                                                      |
| `tcgen05.fence::before_thread_sync`        |                                                      |
| `mbarrier.arrive.relaxed.cluster`          |                                                      |
|                                            | `mbarrier.try_wait.relaxed.cluster // loop till success` |
|                                            | `tcgen05.fence::after_thread_sync`                   |
|                                            | `tcgen05.mma`                                        |

### §9.7.16.6.4.4 Non-pipelined instructions, different thread

The producer thread must explicitly wait for asynchronous instruction completion before synchronizing with the consumer thread.

**Example 1 (ld in Thread 0, mma in Thread 1):**

| Thread 0                                        | Thread 1                                             |
|-------------------------------------------------|------------------------------------------------------|
| `tcgen05.ld`                                    |                                                      |
| `tcgen05.wait::ld`                              |                                                      |
| `tcgen05.fence::before_thread_sync`             |                                                      |
| `mbarrier.arrive.relaxed.cluster`               |                                                      |
|                                                 | `mbarrier.try_wait.relaxed.cluster // loop till success` |
|                                                 | `tcgen05.fence::after_thread_sync`                   |
|                                                 | `tcgen05.mma`                                        |

**Example 2 (mma in Thread 0, ld in Thread 1):**

| Thread 0                                             | Thread 1                                             |
|------------------------------------------------------|------------------------------------------------------|
| `tcgen05.mma`                                        |                                                      |
| `tcgen05.commit.mbarrier::arrive::one [mbar]`        |                                                      |
|                                                      | `mbarrier.try_wait.relaxed.cluster [mbar] // loop`   |
|                                                      | `tcgen05.fence::after_thread_sync`                   |
|                                                      | `tcgen05.ld`                                         |

**Composed synchronization (bidirectional handshake):**

| Thread 0                                             | Thread 1                                             |
|------------------------------------------------------|------------------------------------------------------|
| `tcgen05.mma`                                        |                                                      |
| `tcgen05.commit.mbarrier::arrive::one [bar1]`        |                                                      |
|                                                      | `mbarrier.try_wait.relaxed.cluster [bar1] // loop`   |
|                                                      | `...`                                                |
|                                                      | `tcgen05.fence::after_thread_sync`                   |
|                                                      | `... // MMA completion is guaranteed here`           |
|                                                      | `tcgen05.fence::before_thread_sync`                  |
|                                                      | `mbarrier.arrive.relaxed.cluster [bar2]`             |
| `mbarrier.try_wait.relaxed.cluster [bar2] // loop`   |                                                      |
| `...`                                                |                                                      |
| `tcgen05.fence::after_thread_sync`                   |                                                      |
| `tcgen05.ld`                                         |                                                      |

### §9.7.16.6.4.5 Register dependencies, same thread

For `tcgen05.ld`, an intra-thread ordering through true register dependency will be respected regardless of other synchronization. However, a register dependency does **not** imply that memory accesses will be performed in dependency order. To enforce memory ordering and avoid anti-dependency hazards around `tcgen05.ld`, `tcgen05.wait::ld` must be used.

```ptx
tcgen05.ld %r1, ...;
tcgen05.mma ..., %r1, ...;  // register dependency on %r1 is respected,
                             // but does NOT guarantee memory ordering
```
