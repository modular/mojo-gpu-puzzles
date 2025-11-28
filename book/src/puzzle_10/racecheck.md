# 🏁 Debugging Race Conditions

## Overview

Debug failing GPU programs using NVIDIA's `compute-sanitizer` to identify race conditions that cause incorrect results. You'll learn to use the `racecheck` tool to find concurrency bugs in shared memory operations.

You have a GPU kernel that should accumulate values from multiple threads using shared memory. The test fails, but the logic seems correct. Your task is to identify and fix the race condition causing the failure.

## Configuration

```mojo
comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)  # 9 threads, but only 4 are active
comptime dtype = DType.float32
```

## The failing kernel

```mojo
{{#include ../../../problems/p10/p10.mojo:shared_memory_race}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p10/p10.mojo" class="filename">View full file: problems/p10/p10.mojo</a>

## Running the code

```bash
pixi run p10 --race-condition
```

and the output will look like

```txt
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p10/p10.mojo:122:33: AssertionError: `left == right` comparison failed:
   left: 0.0
  right: 6.0
```

Let's see how `compute-sanitizer` can help us detection issues in our GPU code.

## Debugging with `compute-sanitizer`

### Step 1: Identify the race condition with `racecheck`

Use `compute-sanitizer` with the `racecheck` tool to identify race conditions:

```bash
pixi run compute-sanitizer --tool racecheck mojo problems/p10/p10.mojo --race-condition
```

the output will look like

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
========= Error: Race reported between Write access at p10_shared_memory_race_...+0x140
=========     and Read access at p10_shared_memory_race_...+0xe0 [4 hazards]
=========     and Write access at p10_shared_memory_race_...+0x140 [5 hazards]
=========
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

**Analysis**: The program has **1 race condition** with **9 individual hazards**:
- **4 read-after-write hazards** (threads reading while others write)
- **5 write-after-write hazards** (multiple threads writing simultaneously)


### Step 2: Compare with `synccheck`

Verify this is a race condition, not a synchronization issue:

```bash
pixi run compute-sanitizer --tool synccheck mojo problems/p10/p10.mojo --race-condition
```

and the output will be like

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= ERROR SUMMARY: 0 errors
```

**Key insight**: `synccheck` found **0 errors** - there are no synchronization issues like deadlocks. The problem is **race conditions**, not synchronization bugs.


## Deadlock vs Race Condition: Understanding the Difference

| Aspect | Deadlock | Race Condition |
|--------|----------|----------------|
| **Symptom** | Program hangs forever | Program produces wrong results |
| **Execution** | Never completes | Completes successfully |
| **Timing** | Deterministic hang | Non-deterministic results |
| **Root Cause** | Synchronization logic error | Unsynchronized data access |
| **Detection Tool** | `synccheck` | `racecheck` |
| **Example** | [Puzzle 09: Third case](../puzzle_09/third_case.md) barrier deadlock | Our shared memory `+=` operation |

**In our specific case:**
- **Program completes** → No deadlock (threads don't get stuck)
- **Wrong results** → Race condition (threads corrupt each other's data)
- **Tool confirms** → `synccheck` reports 0 errors, `racecheck` reports 9 hazards

**Why this distinction matters for debugging:**
- **Deadlock debugging**: Focus on barrier placement, conditional synchronization, thread coordination
- **Race condition debugging**: Focus on shared memory access patterns, atomic operations, data dependencies


## Challenge

Equiped with these tools, fix the kernel failing kernel.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### Understanding the hazard breakdown

The `shared_sum[0] += a[row, col]` operation creates hazards because it's actually **three separate memory operations**:
1. **READ** `shared_sum[0]`
2. **ADD** `a[row, col]` to the read value
3. **WRITE** the result back to `shared_sum[0]`

With 4 active threads (positions (0,0), (0,1), (1,0), (1,1)), these operations can interleave:
- **Thread timing overlap** → Multiple threads read the same initial value (0.0)
- **Lost updates** → Each thread writes back `0.0 + their_value`, overwriting others' work
- **Non-atomic operation** → The `+=` compound assignment isn't atomic in GPU shared memory

**Why we get exactly 9 hazards:**
- Each thread tries to perform read-modify-write
- 4 threads × 2-3 hazards per thread = 9 total hazards
- `compute-sanitizer` tracks every conflicting memory access pair

### Race condition debugging tips

1. **Use racecheck for data races**: Detects shared memory hazards and data corruption
2. **Use synccheck for deadlocks**: Detects synchronization bugs (barrier issues, deadlocks)
3. **Focus on shared memory access**: Look for unsynchronized `+=`, `=` operations to shared variables
4. **Identify the pattern**: Read-modify-write operations are common race condition sources
5. **Check barrier placement**: Barriers must be placed BEFORE conflicting operations, not after

**Why this distinction matters for debugging:**
- **Deadlock debugging**: Focus on barrier placement, conditional synchronization, thread coordination
- **Race condition debugging**: Focus on shared memory access patterns, atomic operations, data dependencies

**Common race condition patterns to avoid:**
- Multiple threads writing to the same shared memory location
- Unsynchronized read-modify-write operations (`+=`, `++`, etc.)
- Barriers placed after the race condition instead of before

</div>
</details>

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p10/p10.mojo:shared_memory_race_solution}}
```

<div class="solution-explanation">

### Understanding what went wrong

#### The race condition problem pattern

The original failing code had this critical line:

```mojo
shared_sum[0] += a[row, col]  # RACE CONDITION!
```

This single line creates multiple hazards among the 4 valid threads:
1. **Thread (0,0) reads** `shared_sum[0]` (value: 0.0)
2. **Thread (0,1) reads** `shared_sum[0]` (value: 0.0) ← **Read-after-write hazard!**
3. **Thread (0,0) writes** back `0.0 + 0`
4. **Thread (1,0) writes** back `0.0 + 2` ← **Write-after-write hazard!**

#### Why the test failed

- Multiple threads corrupt each other's writes during the `+=` operation
- The `+=` operation gets interrupted, causing lost updates
- Expected sum of 6.0 (0+1+2+3), but race conditions resulted in 0.0
- The `barrier()` comes too late - after the race condition already occurred

#### What are race conditions?

**Race conditions** occur when multiple threads access shared data concurrently, and the result depends on the unpredictable timing of thread execution.

**Key characteristics:**
- **Non-deterministic behavior**: Same code can produce different results on different runs
- **Timing-dependent**: Results depend on which thread "wins the race"
- **Hard to reproduce**: May only manifest under specific conditions or hardware

#### GPU-specific dangers

**Massive parallelism impact:**
- **Warp-level corruption**: Race conditions can affect entire warps (32 threads)
- **Memory coalescing issues**: Races can disrupt efficient memory access patterns
- **Kernel-wide failures**: Shared memory corruption can affect the entire GPU kernel

**Hardware variations:**
- **Different GPU architectures**: Race conditions may manifest differently across GPU models
- **Memory hierarchy**: L1 cache, L2 cache, and global memory can all exhibit different race behaviors
- **Warp scheduling**: Different thread scheduling can expose different race condition scenarios

### Strategy: Single writer pattern

The key insight is to eliminate concurrent writes to shared memory:

1. **Single writer**: Only one thread (thread at position (0,0)) does all accumulation work
2. **Local accumulation**: Thread at position (0,0) uses a local variable to avoid repeated shared memory access
3. **Single shared memory write**: One write operation eliminates write-write races
4. **Barrier synchronization**: Ensures writer completes before others read
5. **Multiple readers**: All threads safely read the final result

#### Step-by-step solution breakdown

**Step 1: Thread identification**
```mojo
if row == 0 and col == 0:
```
Use direct coordinate check to identify thread at position (0,0).

**Step 2: Single-threaded accumulation**
```mojo
if row == 0 and col == 0:
    local_sum = Scalar[dtype](0.0)
    for r in range(size):
        for c in range(size):
            local_sum += rebind[Scalar[dtype]](a[r, c])
    shared_sum[0] = local_sum  # Single write operation
```
Only thread at position (0,0) performs all accumulation work, eliminating race conditions.

**Step 3: Synchronization barrier**
```mojo
barrier()  # Ensure thread (0,0) completes before others read
```
All threads wait for thread at position (0,0) to finish accumulation.

**Step 4: Safe parallel reads**
```mojo
if row < size and col < size:
    output[row, col] = shared_sum[0]
```
All threads can safely read the result after synchronization.

### Important note on efficiency

**This solution prioritizes correctness over efficiency**. While it eliminates race conditions, using only thread at position (0,0) for accumulation is **not optimal** for GPU performance - we're essentially doing serial computation on a massively parallel device.

**Coming up in [Puzzle 11: Pooling](../../puzzle_11/puzzle_11.md)**: You'll learn efficient parallel reduction algorithms that leverage **all threads** for high-performance summation operations while maintaining race-free execution. This puzzle teaches the foundation of **correctness first** - once you understand how to avoid race conditions, Puzzle 11 will show you how to achieve both **correctness AND performance**.

### Verification

```bash
pixi run compute-sanitizer --tool racecheck mojo solutions/p10/p10.mojo --race-condition
```

**Expected output:**
```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([6.0, 6.0, 6.0, 6.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
✅ Race condition test PASSED! (racecheck will find hazards)
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

**✅ SUCCESS:** Test passes and no race conditions detected!

</div>
</details>
