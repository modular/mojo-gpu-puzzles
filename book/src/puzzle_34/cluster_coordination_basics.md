# Multi-Block Coordination Basics

## Overview

Welcome to your first **cluster programming challenge**! This section introduces the fundamental building blocks of inter-block coordination using SM90+ cluster APIs.

**The Challenge**: Implement a multi-block histogram algorithm where **4 thread blocks coordinate** to process different ranges of data and store results in a shared output array.

**Key Learning**: Learn the essential cluster synchronization pattern: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) → process → [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait), extending the synchronization concepts from [barrier() in Puzzle 29](../puzzle_29/barrier.md).

## The problem: multi-block histogram binning

Traditional single-block algorithms like those in [Puzzle 27](../puzzle_27/puzzle_27.md) can only process data that fits within one block's thread capacity (e.g., 256 threads). For larger datasets exceeding [shared memory capacity from Puzzle 8](../puzzle_08/puzzle_08.md), we need **multiple blocks to cooperate**.

**Your task**: Implement a histogram where each of 4 blocks processes a different data range, scales values by its unique block rank, and coordinates with other blocks using [synchronization patterns from Puzzle 29](../puzzle_29/barrier.md) to ensure all processing completes before any block reads the final results.

### Problem specification

**Multi-Block Data Distribution:**

- **Block 0**: Processes elements 0-255, scales by 1
- **Block 1**: Processes elements 256-511, scales by 2
- **Block 2**: Processes elements 512-767, scales by 3
- **Block 3**: Processes elements 768-1023, scales by 4

**Coordination Requirements:**

1. Each block must signal completion using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)
2. All blocks must wait for others using [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
3. Final output shows each block's processed sum in a 4-element array

## Configuration

- **Problem Size**: `SIZE = 1024` elements (1D array)
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks per cluster `(4, 1)`
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(CLUSTER_SIZE)`

**Thread Block Distribution:**

- Block 0: threads 0-255 → elements 0-255
- Block 1: threads 0-255 → elements 256-511
- Block 2: threads 0-255 → elements 512-767
- Block 3: threads 0-255 → elements 768-1023

## Code to complete

```mojo
{{#include ../../../problems/p34/p34.mojo:cluster_coordination_basics}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">View full file: problems/p34/p34.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### **Block identification patterns**

- Use [`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster) to get the cluster rank (0-3)
- Use `Int(block_idx.x)` for reliable block indexing in grid launch
- Scale data processing by block position for distinct results

### **Shared memory coordination**

- Allocate shared memory using `LayoutTensor[dtype, Layout.row_major(tpb), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()` (see [shared memory basics from Puzzle 8](../puzzle_08/puzzle_08.md))
- Process input data scaled by `block_id + 1` to create distinct scaling per block
- Use bounds checking when accessing input data (pattern from [guards in Puzzle 3](../puzzle_03/puzzle_03.md))

### **Cluster synchronization pattern**

1. **Process**: Each block works on its portion of data
2. **Signal**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) announces processing completion
3. **Compute**: Block-local operations (reduction, aggregation)
4. **Wait**: [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) ensures all blocks complete before proceeding

### **Thread coordination within blocks**

- Use `barrier()` for intra-block synchronization before cluster operations (from [barrier concepts in Puzzle 29](../puzzle_29/barrier.md))
- Only thread 0 should write the final block result (single-writer pattern from [block programming](../puzzle_27/block_sum.md))
- Store results at `output[block_id]` for reliable indexing

</div>
</details>

## Running the code

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --coordination
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --coordination
```

  </div>
</div>

**Expected Output:**

```
Testing Multi-Block Coordination
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Block coordination results:
  Block 0 : 127.5
  Block 1 : 255.0
  Block 2 : 382.5
  Block 3 : 510.0
✅ Multi-block coordination tests passed!
```

**Success Criteria:**

- All 4 blocks produce **non-zero results**
- Results show **scaling pattern**: Block 1 > Block 0, Block 2 > Block 1, etc.
- No race conditions or coordination failures

## Solution

<details class="solution-details">
<summary>Click to reveal solution</summary>

```mojo
{{#include ../../../solutions/p34/p34.mojo:cluster_coordination_basics_solution}}
```

<div class="solution-explanation">

**The cluster coordination solution demonstrates the fundamental multi-block synchronization pattern using a carefully orchestrated two-phase approach:**

## **Phase 1: Independent block processing**

**Thread and block identification:**

```mojo
global_i = block_dim.x * block_idx.x + thread_idx.x  # Global thread index
local_i = thread_idx.x                               # Local thread index within block
my_block_rank = Int(block_rank_in_cluster())         # Cluster rank (0-3)
block_id = Int(block_idx.x)                          # Block index for reliable addressing
```

**Shared memory allocation and data processing:**

- Each block allocates its own shared memory workspace: `LayoutTensor[dtype, Layout.row_major(tpb), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **Scaling strategy**: `data_scale = Float32(block_id + 1)` ensures each block processes data differently
  - Block 0: multiplies by 1.0, Block 1: by 2.0, Block 2: by 3.0, Block 3: by 4.0
- **Bounds checking**: `if global_i < size:` prevents out-of-bounds memory access
- **Data processing**: `shared_data[local_i] = input[global_i] * data_scale` scales input data per block

**Intra-block synchronization:**

- `barrier()` ensures all threads within each block complete data loading before proceeding
- This prevents race conditions between data loading and subsequent cluster coordination

## **Phase 2: Cluster coordination**

**Inter-block signaling:**

- [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) signals that this block has completed its local processing phase
- This is a **non-blocking** operation that registers completion with the cluster hardware

**Local aggregation (Thread 0 only):**

```mojo
if local_i == 0:
    var block_sum: Float32 = 0.0
    for i in range(tpb):
        block_sum += shared_data[i][0]  # Sum all elements in shared memory
    output[block_id] = block_sum        # Store result at unique block position
```

- Only thread 0 performs the sum to avoid race conditions
- Results stored at `output[block_id]` ensures each block writes to unique location

**Final synchronization:**

- [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) blocks until ALL blocks in the cluster have completed their work
- This ensures deterministic completion order across the entire cluster

## **Key technical insights**

**Why use `block_id` instead of `my_block_rank`?**

- `block_idx.x` provides reliable grid-launch indexing (0, 1, 2, 3)
- [`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster) may behave differently depending on cluster configuration
- Using `block_id` guarantees each block gets unique data portions and output positions

**Memory access pattern:**

- **Global memory**: Each thread reads `input[global_i]` exactly once
- **Shared memory**: Used for intra-block communication and aggregation
- **Output memory**: Each block writes to `output[block_id]` exactly once

**Synchronization hierarchy:**

1. **`barrier()`**: Synchronizes threads within each block (intra-block)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Signals completion to other blocks (inter-block, non-blocking)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Waits for all blocks to complete (inter-block, blocking)

**Performance characteristics:**

- **Compute complexity**: O(TPB) per block for local sum, O(1) for cluster coordination
- **Memory bandwidth**: Each input element read once, minimal inter-block communication
- **Scalability**: Pattern scales to larger cluster sizes with minimal overhead

</div>
</details>

## Understanding the pattern

The essential cluster coordination pattern follows a simple but powerful structure:

1. **Phase 1**: Each block processes its assigned data portion independently
2. **Signal**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) announces completion of processing
3. **Phase 2**: Blocks can safely perform operations that depend on other blocks' results
4. **Synchronize**: [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) ensures all blocks finish before proceeding

**Next step**: Ready for more advanced coordination? Continue to **[Cluster-Wide Collective Operations](./cluster_collective_ops.md)** to learn how to extend [`block.sum()` patterns from Puzzle 27](../puzzle_27/block_sum.md) to cluster scale, building on [warp-level reductions from Puzzle 24](../puzzle_24/warp_sum.md)!
