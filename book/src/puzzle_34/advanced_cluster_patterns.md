# 🧠 Advanced Cluster Algorithms

## Overview

This final challenge combines **all levels of GPU programming hierarchy** from [warp-level (Puzzles 24-26)](../puzzle_24/puzzle_24.md), [block-level (Puzzle 27)](../puzzle_27/puzzle_27.md), and cluster coordination - to implement a sophisticated multi-level algorithm that maximizes GPU utilization.

**The Challenge**: Implement a hierarchical cluster algorithm using **warp-level optimization** (`elect_one_sync()`), **block-level aggregation**, and **cluster-level coordination** in a single unified pattern.

**Key Learning**: Learn the complete GPU programming stack with production-ready coordination patterns used in advanced computational workloads.

## The problem: multi-level data processing pipeline

Real-world GPU algorithms often require **hierarchical coordination** where different levels of the GPU hierarchy ([warps from Puzzle 24](../puzzle_24/warp_simt.md), [blocks from Puzzle 27](../puzzle_27/block_sum.md), clusters) perform specialized roles in a coordinated computation pipeline, extending [multi-stage processing from Puzzle 29](../puzzle_29/barrier.md).

**Your task**: Implement a multi-stage algorithm where:

1. **[Warp-level](../puzzle_24/warp_sum.md)**: Use [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient intra-warp coordination (from [SIMT execution](../puzzle_24/warp_simt.md))
2. **[Block-level](../puzzle_27/block_sum.md)**: Aggregate warp results using [shared memory coordination](../puzzle_08/puzzle_08.md)
3. **Cluster-level**: Coordinate between blocks using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) [staged synchronization from Puzzle 29](../puzzle_29/barrier.md)

### Algorithm specification

**Multi-Stage Processing Pipeline:**

1. **Stage 1 ([Warp-level](../puzzle_24/puzzle_24.md))**: Each warp elects one thread to sum 32 consecutive elements
2. **Stage 2 ([Block-level](../puzzle_27/puzzle_27.md))**: Aggregate all warp sums within each block
3. **Stage 3 (Cluster-level)**: Coordinate between blocks with [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)

**Input**: 1024 float values with pattern `(i % 50) * 0.02` for testing
**Output**: 4 block results showing hierarchical processing effects

## Configuration

- **Problem Size**: `SIZE = 1024` elements
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks `(4, 1)`
- **Warp Size**: `WARP_SIZE = 32` threads per warp (NVIDIA standard)
- **Warps per Block**: `TPB / WARP_SIZE = 8` warps
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(CLUSTER_SIZE)`

**Processing Distribution:**

- **Block 0**: 256 threads → 8 warps → elements 0-255
- **Block 1**: 256 threads → 8 warps → elements 256-511
- **Block 2**: 256 threads → 8 warps → elements 512-767
- **Block 3**: 256 threads → 8 warps → elements 768-1023

## Code to complete

```mojo
{{#include ../../../problems/p34/p34.mojo:advanced_cluster_patterns}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">View full file: problems/p34/p34.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### **Warp-level optimization patterns**

- Use [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) to select one thread per warp for computation (from [warp programming basics](../puzzle_24/warp_sum.md))
- The elected thread should process 32 consecutive elements (leveraging [SIMT execution](../puzzle_24/warp_simt.md))
- Compute warp start with `(local_i // 32) * 32` to find warp boundaries (lane indexing from [warp concepts](../puzzle_24/puzzle_24.md))
- Store warp results back in [shared memory at elected thread's position](../puzzle_08/puzzle_08.md)

### **Block-level aggregation strategy**

- After warp processing, aggregate across all warp results (extending [block coordination from Puzzle 27](../puzzle_27/block_sum.md))
- Read from elected positions: indices 0, 32, 64, 96, 128, 160, 192, 224
- Use loop `for i in range(0, tpb, 32)` to iterate through warp leaders (pattern from [reduction algorithms](../puzzle_12/puzzle_12.md))
- Only thread 0 should compute the final block total (single-writer pattern from [barrier coordination](../puzzle_29/barrier.md))

### **Cluster coordination flow**

1. **Process**: Each block processes its data with hierarchical warp optimization
2. **Signal**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) indicates completion of local processing
3. **Store**: Thread 0 writes the block result to output
4. **Wait**: [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) ensures all blocks complete before termination

### **Data scaling and bounds checking**

- Scale input by `Float32(block_id + 1)` to create distinct block patterns
- Always check `global_i < size` before reading input (from [guards in Puzzle 3](../puzzle_03/puzzle_03.md))
- Use `barrier()` between processing phases within blocks (from [synchronization patterns](../puzzle_29/barrier.md))
- Handle warp boundary conditions carefully in loops (considerations from [warp programming](../puzzle_24/warp_simt.md))

</div>
</details>

## Advanced cluster APIs

**From [`gpu.cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/) module:**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Warp-level thread election for efficient computation
- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Signal completion for staged cluster coordination
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Wait for all blocks to reach synchronization point
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster)**: Get unique block identifier within cluster

## Hierarchical coordination pattern

This puzzle demonstrates **three-level coordination hierarchy**:

### **Level 1: Warp Coordination** ([Puzzle 24](../puzzle_24/puzzle_24.md))

```
Warp (32 threads) → elect_one_sync() → 1 elected thread → processes 32 elements
```

### **Level 2: Block Coordination** ([Puzzle 27](../puzzle_27/puzzle_27.md))

```
Block (8 warps) → aggregate warp results → 1 block total
```

### **Level 3: Cluster Coordination** (This puzzle)

```
Cluster (4 blocks) → cluster_arrive/wait → synchronized completion
```

**Combined Effect:** 1024 threads → 32 warp leaders → 4 block results → coordinated cluster completion

## Running the code

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --advanced
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --advanced
```

  </div>
</div>

**Expected Output:**

```
Testing Advanced Cluster Algorithms
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Advanced cluster algorithm results:
  Block 0 : 122.799995
  Block 1 : 247.04001
  Block 2 : 372.72
  Block 3 : 499.83997
✅ Advanced cluster patterns tests passed!
```

**Success Criteria:**

- **Hierarchical scaling**: Results show multi-level coordination effects
- **Warp optimization**: `elect_one_sync()` reduces redundant computation
- **Cluster coordination**: All blocks complete processing successfully
- **Performance pattern**: Higher block IDs produce proportionally larger results

## Solution

<details class="solution-details">
<summary>Click to reveal solution</summary>

```mojo
{{#include ../../../solutions/p34/p34.mojo:advanced_cluster_patterns_solution}}
```

<div class="solution-explanation">

**The advanced cluster patterns solution demonstrates a sophisticated three-level hierarchical optimization that combines warp, block, and cluster coordination for maximum GPU utilization:**

## **Level 1: Warp-Level Optimization (Thread Election)**

**Data preparation and scaling:**

```mojo
var data_scale = Float32(block_id + 1)  # Block-specific scaling factor
if global_i < size:
    shared_data[local_i] = input[global_i] * data_scale
else:
    shared_data[local_i] = 0.0  # Zero-pad for out-of-bounds
barrier()  # Ensure all threads complete data loading
```

**Warp-level thread election:**

```mojo
if elect_one_sync():  # Hardware elects exactly 1 thread per warp
    var warp_sum: Float32 = 0.0
    var warp_start = (local_i // 32) * 32  # Calculate warp boundary
    for i in range(32):  # Process entire warp's data
        if warp_start + i < tpb:
            warp_sum += shared_data[warp_start + i][0]
    shared_data[local_i] = warp_sum  # Store result at elected thread's position
```

**Warp boundary calculation explained:**

- **Thread 37** (in warp 1): `warp_start = (37 // 32) * 32 = 1 * 32 = 32`
- **Thread 67** (in warp 2): `warp_start = (67 // 32) * 32 = 2 * 32 = 64`
- **Thread 199** (in warp 6): `warp_start = (199 // 32) * 32 = 6 * 32 = 192`

**Election pattern visualization (TPB=256, 8 warps):**

```
Warp 0 (threads 0-31):   elect_one_sync() → Thread 0   processes elements 0-31
Warp 1 (threads 32-63):  elect_one_sync() → Thread 32  processes elements 32-63
Warp 2 (threads 64-95):  elect_one_sync() → Thread 64  processes elements 64-95
Warp 3 (threads 96-127): elect_one_sync() → Thread 96  processes elements 96-127
Warp 4 (threads 128-159):elect_one_sync() → Thread 128 processes elements 128-159
Warp 5 (threads 160-191):elect_one_sync() → Thread 160 processes elements 160-191
Warp 6 (threads 192-223):elect_one_sync() → Thread 192 processes elements 192-223
Warp 7 (threads 224-255):elect_one_sync() → Thread 224 processes elements 224-255
```

## **Level 2: Block-level aggregation (Warp Leader Coordination)**

**Inter-warp synchronization:**

```mojo
barrier()  # Ensure all warps complete their elected computations
```

**Warp leader aggregation (Thread 0 only):**

```mojo
if local_i == 0:
    var block_total: Float32 = 0.0
    for i in range(0, tpb, 32):  # Iterate through warp leader positions
        if i < tpb:
            block_total += shared_data[i][0]  # Sum warp results
    output[block_id] = block_total
```

**Memory access pattern:**

- Thread 0 reads from: `shared_data[0]`, `shared_data[32]`, `shared_data[64]`, `shared_data[96]`, `shared_data[128]`, `shared_data[160]`, `shared_data[192]`, `shared_data[224]`
- These positions contain the warp sums computed by elected threads
- Result: 8 warp sums → 1 block total

## **Level 3: Cluster-level staged synchronization**

**Staged synchronization approach:**

```mojo
cluster_arrive()  # Non-blocking: signal this block's completion
# ... Thread 0 computes and stores block result ...
cluster_wait()    # Blocking: wait for all blocks to complete
```

**Why staged synchronization?**

- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)** called **before** final computation allows overlapping work
- Block can compute its result while other blocks are still processing
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)** ensures deterministic completion order
- More efficient than [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) for independent block computations

## **Advanced pattern characteristics**

**Hierarchical computation reduction:**

1. **256 threads** → **8 elected threads** (32x reduction per block)
2. **8 warp sums** → **1 block total** (8x reduction per block)
3. **4 blocks** → **staged completion** (synchronized termination)
4. **Total efficiency**: 256x reduction in redundant computation per block

**Memory access optimization:**

- **Level 1**: Coalesced reads from `input[global_i]`, scaled writes to shared memory
- **Level 2**: Elected threads perform warp-level aggregation (8 computations vs 256)
- **Level 3**: Thread 0 performs block-level aggregation (1 computation vs 8)
- **Result**: Minimized memory bandwidth usage through hierarchical reduction

**Synchronization hierarchy:**

1. **`barrier()`**: Intra-block thread synchronization (after data loading and warp processing)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Inter-block signaling (non-blocking, enables work overlap)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Inter-block synchronization (blocking, ensures completion order)

**Why this is "advanced":**

- **Multi-level optimization**: Combines warp, block, and cluster programming techniques
- **Hardware efficiency**: Leverages [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for optimal warp utilization
- **Staged coordination**: Uses advanced cluster APIs for flexible synchronization
- **Production-ready**: Demonstrates patterns used in real-world GPU libraries

**Real-world performance benefits:**

- **Reduced memory pressure**: Fewer threads accessing shared memory simultaneously
- **Better warp utilization**: Elected threads perform focused computation
- **Scalable coordination**: Staged synchronization handles larger cluster sizes
- **Algorithm flexibility**: Foundation for complex multi-stage processing pipelines

**Complexity analysis:**

- **Warp level**: O(32) operations per elected thread = O(256) total per block
- **Block level**: O(8) aggregation operations per block
- **Cluster level**: O(1) synchronization overhead per block
- **Total**: Linear complexity with massive parallelization benefits

</div>
</details>

## The complete GPU hierarchy

Congratulations! By completing this puzzle, you've learned **the complete GPU programming stack**:

✅ **Thread-level programming**: Individual execution units
✅ **[Warp-level programming](../puzzle_24/puzzle_24.md)**: 32-thread SIMT coordination
✅ **[Block-level programming](../puzzle_27/puzzle_27.md)**: Multi-warp coordination and shared memory
✅ **🆕 Cluster-level programming**: Multi-block coordination with SM90+ APIs
✅ **Coordinate multiple thread blocks** with cluster synchronization primitives
✅ **Scale algorithms beyond single-block limitations** using cluster APIs
✅ **Implement hierarchical algorithms** combining warp + block + cluster coordination
✅ **Utilize next-generation GPU hardware** with SM90+ cluster programming

## Real-world applications

The hierarchical coordination patterns from this puzzle are fundamental to:

**High-Performance Computing:**

- **Multi-grid solvers**: Different levels handle different resolution grids
- **Domain decomposition**: Hierarchical coordination across problem subdomains
- **Parallel iterative methods**: Warp-level local operations, cluster-level global communication

**Deep Learning:**

- **Model parallelism**: Different blocks process different model components
- **Pipeline parallelism**: Staged processing across multiple transformer layers
- **Gradient aggregation**: Hierarchical reduction across distributed training nodes

**Graphics and Visualization:**

- **Multi-pass rendering**: Staged processing for complex visual effects
- **Hierarchical culling**: Different levels cull at different granularities
- **Parallel geometry processing**: Coordinated transformation pipelines

## Next steps

You've now learned the **cutting-edge GPU programming techniques** available on modern hardware!

**Ready for more challenges?** Explore other advanced GPU programming topics, revisit [performance optimization techniques from Puzzles 30-32](../puzzle_30/puzzle_30.md), apply [profiling methodologies from NVIDIA tools](../puzzle_30/nvidia_profiling_basics.md), or build upon these cluster programming patterns for your own computational workloads!
