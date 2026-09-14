# Tile - Memory-Efficient Tiled Processing

## Overview

Building on the **elementwise** pattern, this puzzle introduces
**tiled processing** - a fundamental technique for optimizing memory access
patterns and cache utilization on GPUs. Instead of each thread processing
individual SIMD vectors across the entire array, tiling organizes data into
smaller, manageable chunks that fit better in cache memory.

You've already seen tiling in action with
**[Puzzle 16's tiled matrix multiplication](../puzzle_16/tiled.md)**, where we
used tiles to process large matrices efficiently. Here, we apply the same tiling
principles to vector operations, demonstrating how this technique scales from 2D
matrices to 1D arrays.

Implement the same vector addition operation using Mojo's tiled approach. Each
GPU thread will process an entire tile of data sequentially, demonstrating how
memory locality can improve performance for certain workloads.

**Key insight:** _Tiling trades parallel breadth for memory locality - fewer
threads each doing more work with better cache utilization._

## Key concepts

In this puzzle, you'll learn:

- **Tile-based memory organization** for cache optimization
- **Sequential processing** within tiles
- **Memory locality principles** and cache-friendly access patterns
- **Thread-to-tile mapping** vs thread-to-element mapping
- **Performance trade-offs** between parallelism and memory efficiency

The same mathematical operation as elementwise:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

But with a completely different execution strategy optimized for memory
hierarchy.

## Configuration

- Vector size: `SIZE = 1024`
- Tile size: `TILE_SIZE = 32`
- Data type: `DType.float32`
- SIMD width: GPU-dependent
- Layout: `row_major[SIZE]()` (1D row-major)

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
var num_tiles = (size + tile_size - 1) // tile_size  # Ceiling division
```

For a 1024-element vector with `TILE_SIZE=32`: `1024 ÷ 32 = 32` tiles exactly.

### 2. **Tile extraction pattern**

Check out the
[TileTensor `.tile` documentation](https://max.modular.com/api/mojo/layout/tile_tensor/TileTensor/#tile).

```mojo
var tile_id = Int(indices[0].value())  # Each thread gets one tile to process
var output_tile = output.tile[tile_size](tile_id).to_layout_tensor()
var a_tile = a.tile[tile_size](tile_id).to_layout_tensor()
var b_tile = b.tile[tile_size](tile_id).to_layout_tensor()
```

The `tile[size](id)` method creates a view of `size` consecutive elements
starting at `id × size`.

### 3. **Sequential processing within tiles**

Unlike elementwise, you process the tile sequentially:

```mojo
comptime for i in range(tile_size):
    # Process element i within the current tile
```

This `comptime for` loop unrolls at compile-time for optimal performance.

### 4. **Load and store within tile elements**

```mojo
var a_vec = a_tile.aligned_load[width=simd_width](Index(i))  # Load from position i in tile
var b_vec = b_tile.aligned_load[width=simd_width](Index(i))  # Load from position i in tile
var result = a_vec + b_vec                       # Addition at width simd_width
output_tile.store[simd_width](Index(i), result)  # Store to position i in tile
```

Here `simd_width` is the inner function's own parameter, which the launch below
binds to 1—see the solution for why.

### 5. **Thread configuration difference**

```mojo
elementwise[simd_width=1, target="gpu"](process_tiles, Coord(num_tiles), ctx)
```

Note the `simd_width=1` instead of `SIMD_WIDTH` - each thread processes one
entire tile sequentially.

### 6. **Memory access pattern insight**

Each thread accesses a contiguous block of memory (the tile), then moves to the
next tile. This creates excellent **spatial locality** within each thread's
execution.

### 7. **Key debugging insight**

With tiling, you'll see fewer thread launches but each does more work:

- Elementwise: 256 invocations (for SIMD_WIDTH=4), each covering 4 elements
- Tiled: 32 invocations, each walking 32 elements one at a time

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

The tiled processing pattern demonstrates advanced memory optimization
techniques for GPU programming:

### 1. **Tiling philosophy and memory hierarchy**

Tiling represents a fundamental shift in how we think about parallel processing:

**Elementwise approach:**

- **Wide parallelism**: Many threads, each doing minimal work
- **Small per-thread footprint**: Each thread touches only `SIMD_WIDTH` elements
- **Coalesced access**: Consecutive threads cover one contiguous run of memory

**Tiled approach:**

- **Deep parallelism**: Fewer threads, each doing substantial work
- **Localized memory access**: Each thread works on contiguous data
- **Per-thread locality**: Spatial locality within a thread, traded against
  warp-level coalescing (see below)

### 2. **Tile organization and indexing**

```mojo
var tile_id = Int(indices[0].value())
var output_tile = output.tile[tile_size](tile_id).to_layout_tensor()
var a_tile = a.tile[tile_size](tile_id).to_layout_tensor()
var b_tile = b.tile[tile_size](tile_id).to_layout_tensor()
```

**Tile mapping visualization (TILE_SIZE=32):**

```text
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
comptime for i in range(tile_size):
    var a_vec = a_tile.aligned_load[width=simd_width](Index(i))
    var b_vec = b_tile.aligned_load[width=simd_width](Index(i))
    var ret = a_vec + b_vec
    output_tile.store[simd_width](Index(i), ret)
```

**Why sequential processing?**

- **Cache optimization**: Consecutive memory accesses maximize cache hit rates
- **Compiler optimization**: `comptime for` loops unroll completely at
  compile-time
- **Predictable addresses**: Each thread's addresses advance by one element per
  step, which the compiler folds into the unrolled loop
- **Reduced coordination**: One thread owns a whole tile, so there is nothing to
  synchronize

**Execution pattern within one tile (TILE_SIZE=32):**

```text
Thread processes tile sequentially, one element per iteration:
Step 0:  Load/store element [0]
Step 1:  Load/store element [1]
Step 2:  Load/store element [2]
...
Step 31: Load/store element [31]
Total: 32 scalar operations per thread (comptime for i in range(tile_size))
```

The width here is 1, not `SIMD_WIDTH`. The inner `process_tiles` declares its
own `simd_width` parameter, which shadows the enclosing function's, and
`elementwise[simd_width=1, target="gpu"](process_tiles, Coord(num_tiles), ctx)`
instantiates it with 1. So `aligned_load[width=simd_width]` loads a single element and the loop
walks the tile one element at a time. The two vectorized kernels later in this
puzzle avoid the shadowing by naming their inner parameter
`num_threads_per_tile`, which is why they really do load `SIMD_WIDTH` elements
at a time.

### 4. **Memory access pattern analysis**

**Cache behavior comparison:**

**Elementwise pattern:**

```text
Thread 0: accesses positions [0:4]                        ← One SIMD chunk
Thread 1: accesses positions [4:8]                        ← Next SIMD chunk
...
Result: Each thread touches one small chunk, and the chunks spread across
        the entire array
```

**Tiled pattern:**

```text
Thread 0: accesses positions [0:32] sequentially         ← Contiguous 32-element block
Thread 1: accesses positions [32:64] sequentially       ← Next contiguous 32-element block
...
Result: Perfect spatial locality within each thread
```

**Cache efficiency implications:**

- **L1 cache**: Small tiles often fit better in L1 cache, reducing cache misses
- **TLB efficiency**: Fewer translation lookaside buffer misses
- **Coalescing caveat**: GPUs have no hardware prefetcher, and coalescing is a
  per-warp property, not a per-thread one. Because each thread walks its own
  tile, on any single iteration the lanes of a warp sit `tile_size` elements
  apart - this pattern buys per-thread sequentiality at the cost of warp-level
  coalescing

### 5. **Thread configuration strategy**

```mojo
elementwise[simd_width=1, target="gpu"](process_tiles, Coord(num_tiles), ctx)
```

**Why `simd_width=1` instead of `SIMD_WIDTH`?**

- **Thread count**: Launch exactly `num_tiles` threads, not
  `num_tiles × SIMD_WIDTH`
- **Work distribution**: Each thread handles one complete tile
- **Load balancing**: More work per thread, fewer threads total
- **Memory locality**: Each thread's work is spatially localized

**Performance trade-offs:**

- **Fewer logical threads**: May not fully utilize all GPU cores at low
  occupancy
- **More work per thread**: Better cache utilization and reduced coordination
  overhead
- **Sequential access**: Each thread's own stream of addresses is contiguous,
  though the warp's are not
- **Reduced overhead**: Less thread launch and coordination overhead

**Important note**: "Fewer threads" refers to the logical programming model. The
GPU scheduler can still achieve high hardware utilization by running multiple
warps and efficiently switching between them during memory stalls.

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

```text
Elementwise: High parallelism, scattered memory access
Tiled:       Moderate parallelism, localized memory access
```

The choice between elementwise and tiled patterns depends on your specific
workload characteristics, data access patterns, and target hardware
capabilities.

</div>
</details>

## Next steps

Now that you understand both elementwise and tiled patterns:

- **[Vectorization](./vectorize.md)**: Fine-grained control over SIMD operations
- **[🧠 GPU Threading vs SIMD](./gpu-thread-vs-simd.md)**: Understanding the
  execution hierarchy
- **[📊 Benchmarking](./benchmarking.md)**: Performance analysis and
  optimization

💡 **Key takeaway**: Tiling demonstrates how memory access patterns often matter
more than raw computational throughput. The best GPU code balances parallelism
with memory hierarchy optimization.
