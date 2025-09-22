# ☸️ Cluster-Wide Collective Operations

## Overview

Building on basic cluster coordination from the previous section, this challenge teaches you to implement **cluster-wide collective operations** - extending the familiar [`block.sum`](https://docs.modular.com/mojo/stdlib/gpu/block/sum) pattern from [Puzzle 27](../puzzle_27/block_sum.md) to coordinate across **multiple thread blocks**.

**The Challenge**: Implement a cluster-wide reduction that processes 1024 elements across 4 coordinated blocks, combining their individual reductions into a single global result.

**Key Learning**: Learn [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) for full cluster coordination and [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient final reductions.

## The problem: large-scale global sum

Single blocks (as learned in [Puzzle 27](../puzzle_27/puzzle_27.md)) are limited by their thread count and [shared memory capacity from Puzzle 8](../puzzle_08/puzzle_08.md). For **large datasets** requiring global statistics (mean, variance, sum) beyond [single-block reductions](../puzzle_27/block_sum.md), we need **cluster-wide collective operations**.

**Your task**: Implement a cluster-wide sum reduction where:

1. Each block performs local reduction (like [`block.sum()` from Puzzle 27](../puzzle_27/block_sum.md))
2. Blocks coordinate to combine their partial results using [synchronization from Puzzle 29](../puzzle_29/barrier.md)
3. One elected thread computes the final global sum using [warp election patterns](../puzzle_24/warp_sum.md)

### Problem specification

**Algorithmic Flow:**

**Phase 1 - Local Reduction (within each block):**
\\[R_i = \sum_{j=0}^{TPB-1} input[i \times TPB + j] \quad \text{for block } i\\]

**Phase 2 - Global Aggregation (across cluster):**
\\[\text{Global Sum} = \sum_{i=0}^{\text{CLUSTER_SIZE}-1} R_i\\]

**Coordination Requirements:**

1. **Local reduction**: Each block computes partial sum using tree reduction
2. **Cluster sync**: [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) ensures all partial results are ready
3. **Final aggregation**: One elected thread combines all partial results

## Configuration

- **Problem Size**: `SIZE = 1024` elements
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks per cluster `(4, 1)`
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(1)`
- **Temporary Storage**: `Layout.row_major(CLUSTER_SIZE)` for partial results

**Expected Result**: Sum of sequence `0, 0.01, 0.02, ..., 10.23` = **523,776**

## Code to complete

```mojo
{{#include ../../../problems/p34/p34.mojo:cluster_collective_operations}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">View full file: problems/p34/p34.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### **Local reduction pattern**

- Use [tree reduction pattern from Puzzle 27's block sum](../puzzle_27/block_sum.md)
- Start with stride = `tpb // 2` and halve each iteration (classic [reduction from Puzzle 12](../puzzle_12/puzzle_12.md))
- Only threads with `local_i < stride` participate in each step
- Use `barrier()` between reduction steps (from [barrier concepts in Puzzle 29](../puzzle_29/barrier.md))

### **Cluster coordination strategy**

- Store partial results in `temp_storage[block_id]` for reliable indexing
- Use [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) for full cluster synchronization (stronger than arrive/wait)
- Only one thread should perform the final global aggregation

### **Election pattern for efficiency**

- Use [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) within the first block (`my_block_rank == 0`) (pattern from [warp programming](../puzzle_24/warp_sum.md))
- This ensures only one thread performs the final sum to avoid redundancy
- The elected thread reads all partial results from `temp_storage` (similar to [shared memory access from Puzzle 8](../puzzle_08/puzzle_08.md))

### **Memory access patterns**

- Each thread reads `input[global_i]` with bounds checking (from [guards in Puzzle 3](../puzzle_03/puzzle_03.md))
- Store intermediate results in [shared memory for intra-block reduction](../puzzle_08/puzzle_08.md)
- Store partial results in `temp_storage[block_id]` for inter-block communication
- Final result goes to `output[0]` (single-writer pattern from [block coordination](../puzzle_27/block_sum.md))

</div>
</details>

## Cluster APIs reference

**From [`gpu.cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/) module:**

- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)**: Full cluster synchronization - stronger than arrive/wait pattern
- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Elects single thread within warp for efficient coordination
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster)**: Returns unique block identifier within cluster

## Tree reduction pattern

Recall the **tree reduction pattern** from [Puzzle 27's traditional dot product](../puzzle_27/puzzle_27.md):

```
Stride 128: [T0] += [T128], [T1] += [T129], [T2] += [T130], ...
Stride 64:  [T0] += [T64],  [T1] += [T65],  [T2] += [T66],  ...
Stride 32:  [T0] += [T32],  [T1] += [T33],  [T2] += [T34],  ...
Stride 16:  [T0] += [T16],  [T1] += [T17],  [T2] += [T18],  ...
...
Stride 1:   [T0] += [T1] → Final result at T0
```

**Now extend this pattern to cluster scale** where each block produces one partial result, then combine across blocks.

## Running the code

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --reduction
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --reduction
```

  </div>
</div>

**Expected Output:**

```
Testing Cluster-Wide Reduction
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Expected sum: 523776.0
Cluster reduction result: 523776.0
Expected: 523776.0
Error: 0.0
✅ Passed: Cluster reduction accuracy test
✅ Cluster-wide collective operations tests passed!
```

**Success Criteria:**

- **Perfect accuracy**: Result exactly matches expected sum (523,776)
- **Cluster coordination**: All 4 blocks contribute their partial sums
- **Efficient final reduction**: Single elected thread computes final result

## Solution

<details class="solution-details">
<summary>Click to reveal solution</summary>

```mojo
{{#include ../../../solutions/p34/p34.mojo:cluster_collective_operations_solution}}
```

<div class="solution-explanation">

**The cluster collective operations solution demonstrates the classic distributed computing pattern: local reduction → global coordination → final aggregation:**

## **Phase 1: Local block reduction (traditional tree reduction)**

**Data loading and initialization:**

```mojo
var my_value: Float32 = 0.0
if global_i < size:
    my_value = input[global_i][0]  # Load with bounds checking
shared_mem[local_i] = my_value     # Store in shared memory
barrier()                          # Ensure all threads complete loading
```

**Tree reduction algorithm:**

```mojo
var stride = tpb // 2  # Start with half the threads (128)
while stride > 0:
    if local_i < stride and local_i + stride < tpb:
        shared_mem[local_i] += shared_mem[local_i + stride]
    barrier()          # Synchronize after each reduction step
    stride = stride // 2
```

**Tree reduction visualization (TPB=256):**

```
Step 1: stride=128  [T0]+=T128, [T1]+=T129, ..., [T127]+=T255
Step 2: stride=64   [T0]+=T64,  [T1]+=T65,  ..., [T63]+=T127
Step 3: stride=32   [T0]+=T32,  [T1]+=T33,  ..., [T31]+=T63
Step 4: stride=16   [T0]+=T16,  [T1]+=T17,  ..., [T15]+=T31
Step 5: stride=8    [T0]+=T8,   [T1]+=T9,   ..., [T7]+=T15
Step 6: stride=4    [T0]+=T4,   [T1]+=T5,   [T2]+=T6,  [T3]+=T7
Step 7: stride=2    [T0]+=T2,   [T1]+=T3
Step 8: stride=1    [T0]+=T1    → Final result at shared_mem[0]
```

**Partial result storage:**

- Only thread 0 writes: `temp_storage[block_id] = shared_mem[0]`
- Each block stores its sum at `temp_storage[0]`, `temp_storage[1]`, `temp_storage[2]`, `temp_storage[3]`

## **Phase 2: Cluster synchronization**

**Full cluster barrier:**

- [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) provides **stronger guarantees** than [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)/[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
- Ensures **all blocks complete their local reductions** before any block proceeds
- Hardware-accelerated synchronization across all blocks in the cluster

## **Phase 3: Final global aggregation**

**Thread election for efficiency:**

```mojo
if elect_one_sync() and my_block_rank == 0:
    var total: Float32 = 0.0
    for i in range(CLUSTER_SIZE):
        total += temp_storage[i][0]  # Sum: temp[0] + temp[1] + temp[2] + temp[3]
    output[0] = total
```

**Why this election strategy?**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Hardware primitive that selects exactly one thread per warp
- **`my_block_rank == 0`**: Only elect from the first block to ensure single writer
- **Result**: Only ONE thread across the entire cluster performs the final summation
- **Efficiency**: Avoids redundant computation across all 1024 threads

## **Key technical insights**

**Three-level reduction hierarchy:**

1. **Thread → Warp**: Individual threads contribute to warp-level partial sums
2. **Warp → Block**: Tree reduction combines warps into single block result (256 → 1)
3. **Block → Cluster**: Simple loop combines block results into final sum (4 → 1)

**Memory access patterns:**

- **Input**: Each element read exactly once (`input[global_i]`)
- **Shared memory**: High-speed workspace for intra-block tree reduction
- **Temp storage**: Low-overhead inter-block communication (only 4 values)
- **Output**: Single global result written once

**Synchronization guarantees:**

- **`barrier()`**: Ensures all threads in block complete each tree reduction step
- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)**: **Global barrier** - all blocks reach same execution point
- **Single writer**: Election prevents race conditions on final output

**Algorithm complexity analysis:**

- **Tree reduction**: O(log₂ TPB) = O(log₂ 256) = 8 steps per block
- **Cluster coordination**: O(1) synchronization overhead
- **Final aggregation**: O(CLUSTER_SIZE) = O(4) simple additions
- **Total**: Logarithmic within blocks, linear across blocks

**Scalability characteristics:**

- **Block level**: Scales to thousands of threads with logarithmic complexity
- **Cluster level**: Scales to dozens of blocks with linear complexity
- **Memory**: Temp storage requirements scale linearly with cluster size
- **Communication**: Minimal inter-block data movement (one value per block)

</div>
</details>

## Understanding the collective pattern

This puzzle demonstrates the classic **two-phase reduction pattern** used in distributed computing:

1. **Local aggregation**: Each processing unit (block) reduces its data portion
2. **Global coordination**: Processing units synchronize and exchange results
3. **Final reduction**: One elected unit combines all partial results

**Comparison to single-block approaches:**

- **Traditional `block.sum()`**: Works within 256 threads maximum
- **Cluster collective**: Scales to 1000+ threads across multiple blocks
- **Same accuracy**: Both produce identical mathematical results
- **Different scale**: Cluster approach handles larger datasets

**Performance benefits**:

- **Larger datasets**: Process arrays that exceed single-block capacity
- **Better utilization**: Use more GPU compute units simultaneously
- **Scalable patterns**: Foundation for complex multi-stage algorithms

**Next step**: Ready for the ultimate challenge? Continue to **[Advanced Cluster Algorithms](./advanced_cluster_patterns.md)** to learn hierarchical [warp programming](../puzzle_24/warp_sum.md)+[block coordination](../puzzle_27/block_sum.md)+cluster synchronization, building on [performance optimization techniques](../puzzle_30/profile_kernels.md)!
