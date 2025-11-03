# Tile - Memory-Efficient Tiled Processing

## Overview

Building on the **elementwise** pattern, this puzzle introduces **tiled processing** - a fundamental technique for optimizing memory access patterns and cache utilization on GPUs. Instead of each thread processing individual SIMD vectors across the entire array, tiling organizes data into smaller, manageable chunks that fit better in cache memory.

You've already seen tiling in action with **[Puzzle 16's tiled matrix multiplication](../puzzle_16/tiled.md)**, where we used tiles to process large matrices efficiently. Here, we apply the same tiling principles to vector operations, demonstrating how this technique scales from 2D matrices to 1D arrays.

Implement the same vector addition operation using Mojo's tiled approach. Each GPU thread will process an entire tile of data sequentially, demonstrating how memory locality can improve performance for certain workloads.

**Key insight:** _Tiling trades parallel breadth for memory locality - fewer threads each doing more work with better cache utilization._

## Key concepts

In this puzzle, you'll learn:

- **Tile-based memory organization** for cache optimization
- **Sequential SIMD processing** within tiles
- **Memory locality principles** and cache-friendly access patterns
- **Thread-to-tile mapping** vs thread-to-element mapping
- **Performance trade-offs** between parallelism and memory efficiency

The same mathematical operation as elementwise:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

But with a completely different execution strategy optimized for memory hierarchy.

## Configuration

- Vector size: `SIZE = 1024`
- Tile size: `TILE_SIZE = 32`
- Data type: `DType.float32`
- SIMD width: GPU-dependent (for operations within tiles)
- Layout: `Layout.row_major(SIZE)` (1D row-major)

## Code to complete

```mojo
{{#include ../../../problems/p23/p23.mojo:tiled_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">View full file: problems/p23/p23.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding tile organization**

The tiled approach divides your data into fixed-size chunks:

```mojo
num_tiles = (size + tile_size - 1) // tile_size  # Ceiling division
```

For a 1024-element vector with `TILE_SIZE=32`: `1024 ÷ 32 = 32` tiles exactly.

### 2. **Tile extraction pattern**

Check out the [LayoutTensor `.tile` documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile).

```mojo
tile_id = indices[0]  # Each thread gets one tile to process
out_tile = output.tile[tile_size](tile_id)
a_tile = a.tile[tile_size](tile_id)
b_tile = b.tile[tile_size](tile_id)
```

The `tile[size](id)` method creates a view of `size` consecutive elements starting at `id × size`.

### 3. **Sequential processing within tiles**

Unlike elementwise, you process the tile sequentially:

```mojo
@parameter
for i in range(tile_size):
    # Process element i within the current tile
```

This `@parameter` loop unrolls at compile-time for optimal performance.

### 4. **SIMD operations within tile elements**

```mojo
a_vec = a_tile.load[simd_width](i, 0)  # Load from position i in tile
b_vec = b_tile.load[simd_width](i, 0)  # Load from position i in tile
result = a_vec + b_vec                 # SIMD addition (GPU-dependent width)
out_tile.store[simd_width](i, 0, result)  # Store to position i in tile
```

### 5. **Thread configuration difference**

```mojo
elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)
```

Note the `1` instead of `SIMD_WIDTH` - each thread processes one entire tile sequentially.

### 6. **Memory access pattern insight**

Each thread accesses a contiguous block of memory (the tile), then moves to the next tile. This creates excellent **spatial locality** within each thread's execution.

### 7. **Key debugging insight**

With tiling, you'll see fewer thread launches but each does more work:

- Elementwise: ~256 threads (for SIMD_WIDTH=4), each processing 4 elements
- Tiled: ~32 threads, each processing 32 elements sequentially

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --tiled
```

  </div>
</div>

Your output will look like this when not yet solved:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0
tile_id: 1
tile_id: 2
tile_id: 3
...
tile_id: 29
tile_id: 30
tile_id: 31
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p23/p23.mojo:tiled_elementwise_add_solution}}
```

<div class="solution-explanation">

The tiled processing pattern demonstrates advanced memory optimization techniques for GPU programming:

### 1. **Tiling philosophy and memory hierarchy**

Tiling represents a fundamental shift in how we think about parallel processing:

**Elementwise approach:**

- **Wide parallelism**: Many threads, each doing minimal work
- **Global memory pressure**: Threads scattered across entire array
- **Cache misses**: Poor spatial locality across thread boundaries

**Tiled approach:**

- **Deep parallelism**: Fewer threads, each doing substantial work
- **Localized memory access**: Each thread works on contiguous data
- **Cache optimization**: Excellent spatial and temporal locality

### 2. **Tile organization and indexing**

```mojo
tile_id = indices[0]
out_tile = output.tile[tile_size](tile_id)
a_tile = a.tile[tile_size](tile_id)
b_tile = b.tile[tile_size](tile_id)
```

**Tile mapping visualization (TILE_SIZE=32):**

```
Original array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ..., 1023]

Tile 0 (thread 0): [0, 1, 2, ..., 31]      ← Elements 0-31
Tile 1 (thread 1): [32, 33, 34, ..., 63]   ← Elements 32-63
Tile 2 (thread 2): [64, 65, 66, ..., 95]   ← Elements 64-95
...
Tile 31 (thread 31): [992, 993, ..., 1023] ← Elements 992-1023
```

**Key insights:**

- Each `tile[size](id)` creates a **view** into the original tensor
- Views are zero-copy - no data movement, just pointer arithmetic
- Tile boundaries are always aligned to `tile_size` boundaries

### 3. **Sequential processing deep dive**

```mojo
@parameter
for i in range(tile_size):
    a_vec = a_tile.load[simd_width](i, 0)
    b_vec = b_tile.load[simd_width](i, 0)
    ret = a_vec + b_vec
    out_tile.store[simd_width](i, 0, ret)
```

**Why sequential processing?**

- **Cache optimization**: Consecutive memory accesses maximize cache hit rates
- **Compiler optimization**: `@parameter` loops unroll completely at compile-time
- **Memory bandwidth**: Sequential access aligns with memory controller design
- **Reduced coordination**: No need to synchronize between SIMD groups

**Execution pattern within one tile (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Thread processes tile sequentially:
Step 0: Process elements [0:4] with SIMD
Step 1: Process elements [4:8] with SIMD
Step 2: Process elements [8:12] with SIMD
...
Step 7: Process elements [28:32] with SIMD
Total: 8 SIMD operations per thread (32 ÷ 4 = 8)
```

### 4. **Memory access pattern analysis**

**Cache behavior comparison:**

**Elementwise pattern:**

```
Thread 0: accesses global positions [0, 4, 8, 12, ...]    ← Stride = SIMD_WIDTH
Thread 1: accesses global positions [4, 8, 12, 16, ...]   ← Stride = SIMD_WIDTH
...
Result: Memory accesses spread across entire array
```

**Tiled pattern:**

```
Thread 0: accesses positions [0:32] sequentially         ← Contiguous 32-element block
Thread 1: accesses positions [32:64] sequentially       ← Next contiguous 32-element block
...
Result: Perfect spatial locality within each thread
```

**Cache efficiency implications:**

- **L1 cache**: Small tiles often fit better in L1 cache, reducing cache misses
- **Memory bandwidth**: Sequential access maximizes effective bandwidth
- **TLB efficiency**: Fewer translation lookbook buffer misses
- **Prefetching**: Hardware prefetchers work optimally with sequential patterns

### 5. **Thread configuration strategy**

```mojo
elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)
```

**Why `1` instead of `SIMD_WIDTH`?**

- **Thread count**: Launch exactly `num_tiles` threads, not `num_tiles × SIMD_WIDTH`
- **Work distribution**: Each thread handles one complete tile
- **Load balancing**: More work per thread, fewer threads total
- **Memory locality**: Each thread's work is spatially localized

**Performance trade-offs:**

- **Fewer logical threads**: May not fully utilize all GPU cores at low occupancy
- **More work per thread**: Better cache utilization and reduced coordination overhead
- **Sequential access**: Optimal memory bandwidth utilization within each thread
- **Reduced overhead**: Less thread launch and coordination overhead

**Important note**: "Fewer threads" refers to the logical programming model. The GPU scheduler can still achieve high hardware utilization by running multiple warps and efficiently switching between them during memory stalls.

### 6. **Performance characteristics**

**When tiling helps:**

- **Memory-bound operations**: When memory bandwidth is the bottleneck
- **Cache-sensitive workloads**: Operations that benefit from data reuse
- **Complex operations**: When compute per element is higher
- **Limited parallelism**: When you have fewer threads than GPU cores

**When tiling hurts:**

- **Highly parallel workloads**: When you need maximum thread utilization
- **Simple operations**: When memory access dominates over computation
- **Irregular access patterns**: When tiling doesn't improve locality

**For our simple addition example (TILE_SIZE=32):**

- **Thread count**: 32 threads instead of 256 (8× fewer)
- **Work per thread**: 32 elements instead of 4 (8× more)
- **Memory pattern**: Sequential vs strided access
- **Cache utilization**: Much better spatial locality

### 7. **Advanced tiling considerations**

**Tile size selection:**

- **Too small**: Poor cache utilization, more overhead
- **Too large**: May not fit in cache, reduced parallelism
- **Sweet spot**: Usually 16-64 elements for L1 cache optimization
- **Our choice**: 32 elements balances cache usage with parallelism

**Hardware considerations:**

- **Cache size**: Tiles should fit in L1 cache when possible
- **Memory bandwidth**: Consider memory controller width
- **Core count**: Ensure enough tiles to utilize all cores
- **SIMD width**: Tile size should be multiple of SIMD width

**Comparison summary:**

```
Elementwise: High parallelism, scattered memory access
Tiled:       Moderate parallelism, localized memory access
```

The choice between elementwise and tiled patterns depends on your specific workload characteristics, data access patterns, and target hardware capabilities.

</div>
</details>

## Next steps

Now that you understand both elementwise and tiled patterns:

- **[Vectorization](./vectorize.md)**: Fine-grained control over SIMD operations
- **[🧠 GPU Threading vs SIMD](./gpu-thread-vs-simd.md)**: Understanding the execution hierarchy
- **[📊 Benchmarking](./benchmarking.md)**: Performance analysis and optimization

💡 **Key takeaway**: Tiling demonstrates how memory access patterns often matter more than raw computational throughput. The best GPU code balances parallelism with memory hierarchy optimization.
