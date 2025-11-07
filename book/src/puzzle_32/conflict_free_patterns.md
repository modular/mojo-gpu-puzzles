# Conflict-Free Patterns

> **Note: This section is specific to NVIDIA GPUs**
>
> Bank conflict analysis and profiling techniques covered here apply specifically to NVIDIA GPUs. The profiling commands use NSight Compute tools that are part of the NVIDIA CUDA toolkit.

## Building on your profiling skills

You've learned GPU profiling fundamentals in [Puzzle 30](../puzzle_30/puzzle_30.md) and understood resource optimization in [Puzzle 31](../puzzle_31/puzzle_31.md). Now you're ready to apply those detective skills to a new performance mystery: **shared memory bank conflicts**.

**The detective challenge:** You have two GPU kernels that perform identical mathematical operations (`(input + 10) * 2`). Both produce exactly the same results. Both use the same amount of shared memory. Both have identical occupancy. Yet one experiences systematic performance degradation due to **how** it accesses shared memory.

**Your mission:** Use the profiling methodology you've learned to uncover this hidden performance trap and understand when bank conflicts matter in real-world GPU programming.

## Overview

Shared memory bank conflicts occur when multiple threads in a warp simultaneously access different addresses within the same memory bank. This detective case explores two kernels with contrasting access patterns:

```mojo
{{#include ../../../problems/p32/p32.mojo:no_conflict_kernel}}
```

```mojo
{{#include ../../../problems/p32/p32.mojo:two_way_conflict_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p32/p32.mojo" class="filename">View full file: problems/p32/p32.mojo</a>

**The mystery:** These kernels compute identical results but have dramatically different shared memory access efficiency. Your job is to discover why using systematic profiling analysis.

## Configuration

**Requirements:**

- NVIDIA GPU with CUDA toolkit and NSight Compute from [Puzzle 30](../puzzle_30/puzzle_30.md)
- Understanding of shared memory banking concepts from the [previous section](./shared_memory_bank.md)

**Kernel specifications:**

```mojo
alias SIZE = 8 * 1024      # 8K elements - focus on shared memory patterns
alias TPB = 256            # 256 threads per block (8 warps)
alias BLOCKS_PER_GRID = (SIZE // TPB, 1)  # 32 blocks
```

**Key insight:** The problem size is deliberately smaller than previous puzzles to highlight shared memory effects rather than global memory bandwidth limitations.

## The investigation

### Step 1: Verify correctness

```bash
pixi shell -e nvidia
mojo problems/p32/p32.mojo --test
```

Both kernels should produce identical results. This confirms that bank conflicts affect **performance** but not **correctness**.

### Step 2: Benchmark performance baseline

```bash
mojo problems/p32/p32.mojo --benchmark
```

Record the execution times. You may notice similar performance due to the workload being dominated by global memory access, but bank conflicts will be revealed through profiling metrics.

### Step 3: Build for profiling

```bash
mojo build --debug-level=full problems/p32/p32.mojo -o problems/p32/p32_profiler
```

### Step 4: Profile bank conflicts

Use NSight Compute to measure shared memory bank conflicts quantitatively:

```bash
# Profile no-conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --no-conflict

```

and

```bash
# Profile two-way conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --two-way
```

**Key metrics to record:**

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` - Load conflicts
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` - Store conflicts

### Step 5: Analyze access patterns

Based on your profiling results, analyze the mathematical access patterns:

**No-conflict kernel access pattern:**

```mojo
# Thread mapping: thread_idx.x directly maps to shared memory index
shared_buf[thread_idx.x]  # Thread 0→Index 0, Thread 1→Index 1, etc.
# Bank mapping: Index % 32 = Bank ID
# Result: Thread 0→Bank 0, Thread 1→Bank 1, ..., Thread 31→Bank 31
```

**Two-way conflict kernel access pattern:**

```mojo
# Thread mapping with stride-2 modulo operation
shared_buf[(thread_idx.x * 2) % TPB]
# For threads 0-31: Index 0,2,4,6,...,62, then wraps to 64,66,...,126, then 0,2,4...
# Bank mapping examples:
# Thread 0  → Index 0   → Bank 0
# Thread 16 → Index 32  → Bank 0  (conflict!)
# Thread 1  → Index 2   → Bank 2
# Thread 17 → Index 34  → Bank 2  (conflict!)
```

## Your task: solve the bank conflict mystery

**After completing the investigation steps above, answer these analysis questions:**

### Performance analysis (Steps 1-2)

1. Do both kernels produce identical mathematical results?
2. What are the execution time differences (if any) between the kernels?
3. Why might performance be similar despite different access patterns?

### Bank conflict profiling (Step 4)

4. How many bank conflicts does the no-conflict kernel generate for loads and stores?
5. How many bank conflicts does the two-way conflict kernel generate for loads and stores?
6. What is the total conflict count difference between the kernels?

### Access pattern analysis (Step 5)

7. In the no-conflict kernel, which bank does Thread 0 access? Thread 31?
8. In the two-way conflict kernel, which threads access Bank 0? Which access Bank 2?
9. How many threads compete for the same bank in the conflict kernel?

### The bank conflict detective work

10. Why does the two-way conflict kernel show measurable conflicts while the no-conflict kernel shows zero?
11. How does the stride-2 access pattern `(thread_idx.x * 2) % TPB` create systematic conflicts?
12. Why do bank conflicts matter more in compute-intensive kernels than memory-bound kernels?

### Real-world implications

13. When would you expect bank conflicts to significantly impact application performance?
14. How can you predict bank conflict patterns before implementing shared memory algorithms?
15. What design principles help avoid bank conflicts in matrix operations and stencil computations?

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

**Bank conflict detective toolkit:**

- **NSight Compute metrics** - Quantify conflicts with precise measurements
- **Access pattern visualization** - Map thread indices to banks systematically
- **Mathematical analysis** - Use modulo arithmetic to predict conflicts
- **Workload characteristics** - Understand when conflicts matter vs when they don't

**Key investigation principles:**

- **Measure systematically:** Use profiling tools rather than guessing about conflicts
- **Visualize access patterns:** Draw thread-to-bank mappings for complex algorithms
- **Consider workload context:** Bank conflicts matter most in compute-intensive shared memory algorithms
- **Think prevention:** Design algorithms with conflict-free access patterns from the start

**Access pattern analysis approach:**

1. **Map threads to indices:** Understand the mathematical address calculation
2. **Calculate bank assignments:** Use the formula `bank_id = (address / 4) % 32`
3. **Identify conflicts:** Look for multiple threads accessing the same bank
4. **Validate with profiling:** Confirm theoretical analysis with NSight Compute measurements

**Common conflict-free patterns:**

- **Sequential access:** `shared[thread_idx.x]` - each thread different bank
- **Broadcast access:** `shared[0]` for all threads - hardware optimization
- **Power-of-2 strides:** Stride-32 often maps cleanly to banking patterns
- **Padded arrays:** Add padding to shift problematic access patterns

</div>
</details>

## Solution

<details class="solution-details">
<summary><strong>Complete Solution with Bank Conflict Analysis</strong></summary>

This bank conflict detective case demonstrates how shared memory access patterns affect GPU performance and reveals the importance of systematic profiling for optimization.

## **Investigation results from profiling**

**Step 1: Correctness Verification**
Both kernels produce identical mathematical results:

```
✅ No-conflict kernel: PASSED
✅ Two-way conflict kernel: PASSED
✅ Both kernels produce identical results
```

**Step 2: Performance Baseline**
Benchmark results show similar execution times:

```
| name             | met (ms)           | iters |
| ---------------- | ------------------ | ----- |
| no_conflict      | 2.1930616745886655 | 547   |
| two_way_conflict | 2.1978922967032966 | 546   |
```

**Key insight:** Performance is nearly identical (~2.19ms vs ~2.20ms) because this workload is **global memory bound** rather than shared memory bound. Bank conflicts become visible through profiling metrics rather than execution time.

## **Bank conflict profiling evidence**

**No-Conflict Kernel (Optimal Access Pattern):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    0
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    0
```

**Result:** Zero conflicts for both loads and stores - perfect shared memory efficiency.

**Two-Way Conflict Kernel (Problematic Access Pattern):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    256
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    256
```

**Result:** 256 conflicts each for loads and stores - clear evidence of systematic banking problems.

**Total conflict difference:** 512 conflicts (256 + 256) demonstrate measurable shared memory inefficiency.

## **Access pattern mathematical analysis**

### No-conflict kernel access pattern

**Thread-to-index mapping:**

```mojo
shared_buf[thread_idx.x]
```

**Bank assignment analysis:**

```
Thread 0  → Index 0   → Bank 0 % 32 = 0
Thread 1  → Index 1   → Bank 1 % 32 = 1
Thread 2  → Index 2   → Bank 2 % 32 = 2
...
Thread 31 → Index 31  → Bank 31 % 32 = 31
```

**Result:** Perfect bank distribution - each thread accesses a different bank within each warp, enabling parallel access.

### Two-way conflict kernel access pattern

**Thread-to-index mapping:**

```mojo
shared_buf[(thread_idx.x * 2) % TPB]  # TPB = 256
```

**Bank assignment analysis for first warp (threads 0-31):**

```
Thread 0  → Index (0*2)%256 = 0   → Bank 0
Thread 1  → Index (1*2)%256 = 2   → Bank 2
Thread 2  → Index (2*2)%256 = 4   → Bank 4
...
Thread 16 → Index (16*2)%256 = 32 → Bank 0  ← CONFLICT with Thread 0
Thread 17 → Index (17*2)%256 = 34 → Bank 2  ← CONFLICT with Thread 1
Thread 18 → Index (18*2)%256 = 36 → Bank 4  ← CONFLICT with Thread 2
...
```

**Conflict pattern:** Each bank serves exactly 2 threads, creating systematic 2-way conflicts across all 32 banks.

**Mathematical explanation:** The stride-2 pattern with modulo 256 creates a repeating access pattern where:

- Threads 0-15 access banks 0,2,4,...,30
- Threads 16-31 access the **same banks** 0,2,4,...,30
- Each bank collision requires hardware serialization

## **Why this matters: workload context analysis**

### Memory-bound vs compute-bound implications

**This workload characteristics:**

- **Global memory dominant:** Each thread performs minimal computation relative to memory transfer
- **Shared memory secondary:** Bank conflicts add overhead but don't dominate total execution time
- **Identical performance:** Global memory bandwidth saturation masks shared memory inefficiency

**When bank conflicts matter most:**

1. **Compute-intensive shared memory algorithms** - Matrix multiplication, stencil computations, FFT
2. **Tight computational loops** - Repeated shared memory access within inner loops
3. **High arithmetic intensity** - Significant computation per memory access
4. **Large shared memory working sets** - Algorithms that heavily utilize shared memory caching

### Real-world performance implications

**Applications where bank conflicts significantly impact performance:**

**Matrix Multiplication:**

```mojo
# Problematic: All threads in warp access same column
for k in range(tile_size):
    acc += a_shared[local_row, k] * b_shared[k, local_col]  # b_shared[k, 0] conflicts
```

**Stencil Computations:**

```mojo
# Problematic: Stride access in boundary handling
shared_buf[thread_idx.x * stride]  # Creates systematic conflicts
```

**Parallel Reductions:**

```mojo
# Problematic: Power-of-2 stride patterns
if thread_idx.x < stride:
    shared_buf[thread_idx.x] += shared_buf[thread_idx.x + stride]  # Conflict potential
```

## **Conflict-free design principles**

### Prevention strategies

**1. Sequential access patterns:**

```mojo
shared[thread_idx.x]  # Optimal - each thread different bank
```

**2. Broadcast optimization:**

```mojo
constant = shared[0]  # All threads read same address - hardware optimized
```

**3. Padding techniques:**

```mojo
shared = LayoutTensor[dtype, Layout.row_major(TPB + 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()  # Shift access patterns
```

**4. Access pattern analysis:**

- Calculate bank assignments before implementation
- Use modulo arithmetic: `bank_id = (address_bytes / 4) % 32`
- Visualize thread-to-bank mappings for complex algorithms

### Systematic optimization workflow

**Design Phase:**

1. **Plan access patterns** - Sketch thread-to-memory mappings
2. **Calculate bank assignments** - Use mathematical analysis
3. **Predict conflicts** - Identify problematic access patterns
4. **Design alternatives** - Consider padding, transpose, or algorithm changes

**Implementation Phase:**

1. **Profile systematically** - Use NSight Compute conflict metrics
2. **Measure impact** - Compare conflict counts across implementations
3. **Validate performance** - Ensure optimizations improve end-to-end performance
4. **Document patterns** - Record successful conflict-free algorithms for reuse

## **Key takeaways: from detective work to optimization expertise**

**The Bank Conflict Investigation revealed:**

1. **Measurement trumps intuition** - Profiling tools reveal conflicts invisible to performance timing
2. **Pattern analysis works** - Mathematical prediction accurately matched NSight Compute results
3. **Context matters** - Bank conflicts matter most in compute-intensive shared memory workloads
4. **Prevention beats fixing** - Designing conflict-free patterns easier than retrofitting optimizations

**Universal shared memory optimization principles:**

**When to worry about bank conflicts:**

- **High-computation kernels** using shared memory for data reuse
- **Iterative algorithms** with repeated shared memory access in tight loops
- **Performance-critical code** where every cycle matters
- **Memory-intensive operations** that are compute-bound rather than bandwidth-bound

**When bank conflicts are less critical:**

- **Memory-bound workloads** where global memory dominates performance
- **Simple caching scenarios** with minimal shared memory reuse
- **One-time access patterns** without repeated conflict-prone operations

**Professional development methodology:**

1. **Profile before optimizing** - Measure conflicts quantitatively with NSight Compute
2. **Understand access mathematics** - Use bank assignment formulas to predict problems
3. **Design systematically** - Consider banking in algorithm design, not as afterthought
4. **Validate optimizations** - Confirm that conflict reduction improves actual performance

This detective case demonstrates that **systematic profiling reveals optimization opportunities invisible to performance timing alone** - bank conflicts are a perfect example of where measurement-driven optimization beats guesswork.

</details>
