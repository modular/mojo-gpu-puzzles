# 📚 Understanding Shared Memory Banks

## Building on what you've learned

You've come a long way in your GPU optimization journey. In [Puzzle 8](../puzzle_08/puzzle_08.md), you discovered how shared memory provides fast, block-local storage that dramatically outperforms global memory. [Puzzle 16](../puzzle_16/puzzle_16.md) showed you how matrix multiplication kernels use shared memory to cache data tiles, reducing expensive global memory accesses.

But there's a hidden performance trap lurking in shared memory that can serialize your parallel operations: **bank conflicts**.

**The performance mystery:** You can write two kernels that access shared memory in seemingly identical ways - both use the same amount of data, both have perfect occupancy, both avoid race conditions. Yet one runs 32× slower than the other. The culprit? How threads access shared memory banks.

## What are shared memory banks?

Think of shared memory as a collection of 32 independent memory units called **banks**, each capable of serving one memory request per clock cycle. This banking system exists for a fundamental reason: **hardware parallelism**.

When a warp of 32 threads needs to access shared memory simultaneously, the GPU can serve all 32 requests in parallel, **if each thread accesses a different bank**. When multiple threads try to access the same bank, the hardware must **serialize** these accesses, turning what should be a 1-cycle operation into multiple cycles.

### Bank address mapping

Each 4-byte word in shared memory belongs to a specific bank according to this formula:

```
bank_id = (byte_address / 4) % 32
```

Here's how the first 128 bytes of shared memory map to banks:

| Address Range | Bank ID | Example `float32` Elements |
|---------------|---------|---------------------------|
| 0-3 bytes     | Bank 0  | `shared[0]` |
| 4-7 bytes     | Bank 1  | `shared[1]` |
| 8-11 bytes    | Bank 2  | `shared[2]` |
| ...           | ...     | ... |
| 124-127 bytes | Bank 31 | `shared[31]` |
| 128-131 bytes | Bank 0  | `shared[32]` |
| 132-135 bytes | Bank 1  | `shared[33]` |

**Key insight:** The banking pattern repeats every 32 elements for `float32` arrays, which perfectly matches the 32-thread warp size. This is not a coincidence - it's designed for optimal parallel access.

## Types of bank conflicts

### No conflict: the ideal case

When each thread in a warp accesses a different bank, all 32 accesses complete in 1 cycle:

```mojo
# Perfect case: each thread accesses a different bank
shared[thread_idx.x]  # Thread 0→Bank 0, Thread 1→Bank 1, ..., Thread 31→Bank 31
```

**Result:** 32 parallel accesses, 1 cycle total

### N-way bank conflicts

When N threads access different addresses in the same bank, the hardware serializes these accesses:

```mojo
# 2-way conflict: stride-2 access pattern
shared[thread_idx.x * 2]  # Thread 0,16→Bank 0; Thread 1,17→Bank 1; etc.
```

**Result:** 2 accesses per bank, 2 cycles total (50% efficiency)

```mojo
# Worst case: all threads access different addresses in Bank 0
shared[thread_idx.x * 32]  # All threads→Bank 0
```

**Result:** 32 serialized accesses, 32 cycles total (3% efficiency)

### The broadcast exception

There's one important exception to the conflict rule: **broadcast access**. When all threads read the **same address**, the hardware optimizes this into a single memory access:

```mojo
# Broadcast: all threads read the same value
constant = shared[0]  # All threads read shared[0]
```

**Result:** 1 access broadcasts to 32 threads, 1 cycle total

This optimization exists because broadcasting is a common pattern (loading constants, reduction operations), and the hardware can duplicate a single value to all threads without additional memory bandwidth.

## Why bank conflicts matter

### Performance impact

Bank conflicts directly multiply your shared memory access time:

| Conflict Type | Access Time | Efficiency | Performance Impact |
|---------------|-------------|------------|-------------------|
| No conflict | 1 cycle | 100% | Baseline |
| 2-way conflict | 2 cycles | 50% | 2× slower |
| 4-way conflict | 4 cycles | 25% | 4× slower |
| 32-way conflict | 32 cycles | 3% | **32× slower** |

### Real-world context

From [Puzzle 30](../puzzle_30/puzzle_30.md), you learned that memory access patterns can create dramatic performance differences. Bank conflicts are another example of this principle operating at the shared memory level.

Just as global memory coalescing affects DRAM bandwidth utilization, bank conflicts affect shared memory throughput. The difference is scale: global memory latency is hundreds of cycles, while shared memory conflicts add only a few cycles per access. However, in compute-intensive kernels that heavily use shared memory, these "few cycles" accumulate quickly.

### Connection to warp execution

Remember from [Puzzle 24](../puzzle_24/puzzle_24.md) that warps execute in SIMT (Single Instruction, Multiple Thread) fashion. When a warp encounters a bank conflict, **all 32 threads must wait** for the serialized memory accesses to complete. This waiting time affects the entire warp's progress, not just the conflicting threads.

This connects to the occupancy concepts from [Puzzle 31](../puzzle_31/puzzle_31.md): bank conflicts can prevent warps from hiding memory latency effectively, reducing the practical benefit of high occupancy.

## Detecting bank conflicts

### Visual pattern recognition

You can often predict bank conflicts by analyzing access patterns:

**Sequential access (no conflicts):**
```mojo
# Thread ID:  0  1  2  3  ...  31
# Address:    0  4  8 12  ... 124
# Bank:       0  1  2  3  ...  31  ✅ All different banks
```

**Stride-2 access (2-way conflicts):**
```mojo
# Thread ID:  0  1  2  3  ...  15 16 17 18 ... 31
# Address:    0  8 16 24  ... 120  4 12 20 ... 124
# Bank:       0  2  4  6  ...  30  1  3  5 ...  31
# Conflict:   Banks 0,2,4... have 2 threads each  ❌
```

**Stride-32 access (32-way conflicts):**
```mojo
# Thread ID:  0   1   2   3  ...  31
# Address:    0  128 256 384 ... 3968
# Bank:       0   0   0   0  ...   0  ❌ All threads→Bank 0
```

### Profiling with NSight Compute (`ncu`)

Building on the profiling methodology from [Puzzle 30](../puzzle_30/puzzle_30.md), you can measure bank conflicts quantitatively:

```bash
# Key metrics for shared memory bank conflicts
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st your_kernel

# Additional context metrics
ncu --metrics=smsp__sass_average_branch_targets_threads_uniform.pct your_kernel
ncu --metrics=smsp__warps_issue_stalled_membar_per_warp_active.pct your_kernel
```

The `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` metrics directly count the number of bank conflicts for load and store operations during kernel execution. Combined with the number of shared memory accesses, these give you the conflict ratio - a critical performance indicator.

## When bank conflicts matter most

### Compute-intensive kernels

Bank conflicts have the greatest impact on kernels where:
- Shared memory is accessed frequently within tight loops
- Computational work per shared memory access is minimal
- The kernel is compute-bound rather than memory-bound

**Example scenarios:**
- Matrix multiplication inner loops (like the tiled versions in [Puzzle 16](../puzzle_16/puzzle_16.md))
- Stencil computations with shared memory caching
- Parallel reduction operations

### Memory-bound vs compute-bound trade-offs

Just as [Puzzle 31](../puzzle_31/puzzle_31.md) showed that occupancy matters less for memory-bound workloads, bank conflicts matter less when your kernel is bottlenecked by global memory bandwidth or arithmetic intensity is very low.

However, many kernels that use shared memory do so precisely **because** they want to shift from memory-bound to compute-bound execution. In these cases, bank conflicts can prevent you from achieving the performance gains that motivated using shared memory in the first place.

## The path forward

Understanding shared memory banking gives you the foundation to:
1. **Predict performance** before writing code by analyzing access patterns
2. **Diagnose slowdowns** using systematic profiling approaches
3. **Design conflict-free algorithms** that maintain high shared memory throughput
4. **Make informed trade-offs** between algorithm complexity and memory efficiency

In the next section, you'll apply this knowledge through hands-on exercises that demonstrate common conflict patterns and their solutions - turning theoretical understanding into practical optimization skills.
