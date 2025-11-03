# Vectorization - Fine-Grained SIMD Control

## Overview

This puzzle explores **advanced vectorization techniques** using manual vectorization and [vectorize](https://docs.modular.com/mojo/stdlib/algorithm/functional/vectorize/) that give you precise control over SIMD operations within GPU kernels. You'll implement two different approaches to vectorized computation:

1. **Manual vectorization**: Direct SIMD control with explicit index calculations
2. **Mojo's vectorize function**: High-level vectorization with automatic bounds checking

Both approaches build on tiling concepts but with different trade-offs between control, safety, and performance optimization.

**Key insight:** _Different vectorization strategies suit different performance requirements and complexity levels._

## Key concepts

In this puzzle, you'll learn:

- **Manual SIMD operations** with explicit index management
- **Mojo's vectorize function** for safe, automatic vectorization
- **Chunk-based memory organization** for optimal SIMD alignment
- **Bounds checking strategies** for edge cases
- **Performance trade-offs** between manual control and safety

The same mathematical operation as before:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

But with sophisticated vectorization strategies for maximum performance.

## Configuration

- Vector size: `SIZE = 1024`
- Tile size: `TILE_SIZE = 32`
- Data type: `DType.float32`
- SIMD width: GPU-dependent
- Layout: `Layout.row_major(SIZE)` (1D row-major)

## 1. Manual vectorization approach

### Code to complete

```mojo
{{#include ../../../problems/p23/p23.mojo:manual_vectorized_tiled_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">View full file: problems/p23/p23.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding chunk organization**

```mojo
alias chunk_size = tile_size * simd_width  # 32 * 4 = 128 elements per chunk
```

Each tile now contains multiple SIMD groups, not just sequential elements.

### 2. **Global index calculation**

```mojo
global_start = tile_id * chunk_size + i * simd_width
```

This calculates the exact global position for each SIMD vector within the chunk.

### 3. **Direct tensor access**

```mojo
a_vec = a.load[simd_width](global_start, 0)     # Load from global tensor
output.store[simd_width](global_start, 0, ret)  # Store to global tensor
```

Note: Access the original tensors, not the tile views.

### 4. **Key characteristics**

- More control, more complexity, global tensor access
- Perfect SIMD alignment with hardware
- Manual bounds checking required

</div>
</details>

### Running manual vectorization

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --manual-vectorized
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
tile_id: 4
tile_id: 5
tile_id: 6
tile_id: 7
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### Manual vectorization solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p23/p23.mojo:manual_vectorized_tiled_elementwise_add_solution}}
```

<div class="solution-explanation">

### Manual vectorization deep dive

**Manual vectorization** gives you direct control over SIMD operations with explicit index calculations:

- **Chunk-based organization**: `chunk_size = tile_size * simd_width`
- **Global indexing**: Direct calculation of memory positions
- **Manual bounds management**: You handle edge cases explicitly

**Architecture and memory layout:**

```mojo
alias chunk_size = tile_size * simd_width  # 32 * 4 = 128
```

**Chunk organization visualization (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Original array: [0, 1, 2, 3, ..., 1023]

Chunk 0 (thread 0): [0:128]    ← 128 elements = 32 SIMD groups of 4
Chunk 1 (thread 1): [128:256]  ← Next 128 elements
Chunk 2 (thread 2): [256:384]  ← Next 128 elements
...
Chunk 7 (thread 7): [896:1024] ← Final 128 elements
```

**Processing within one chunk:**

```mojo
@parameter
for i in range(tile_size):  # i = 0, 1, 2, ..., 31
    global_start = tile_id * chunk_size + i * simd_width
    # For tile_id=0: global_start = 0, 4, 8, 12, ..., 124
    # For tile_id=1: global_start = 128, 132, 136, 140, ..., 252
```

**Performance characteristics:**

- **Thread count**: 8 threads (1024 ÷ 128 = 8)
- **Work per thread**: 128 elements (32 SIMD operations of 4 elements each)
- **Memory pattern**: Large chunks with perfect SIMD alignment
- **Overhead**: Minimal - direct hardware mapping
- **Safety**: Manual bounds checking required

**Key advantages:**

- **Predictable indexing**: Exact control over memory access patterns
- **Optimal alignment**: SIMD operations perfectly aligned to hardware
- **Maximum throughput**: No overhead from safety checks
- **Hardware optimization**: Direct mapping to GPU SIMD units

**Key challenges:**

- **Index complexity**: Manual calculation of global positions
- **Bounds responsibility**: Must handle edge cases explicitly
- **Debugging difficulty**: More complex to verify correctness

</div>
</details>

## 2. Mojo vectorize approach

### Code to complete

```mojo
{{#include ../../../problems/p23/p23.mojo:vectorize_within_tiles_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">View full file: problems/p23/p23.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Tile boundary calculation**

```mojo
tile_start = tile_id * tile_size
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

Handle cases where the last tile might be smaller than `tile_size`.

### 2. **Vectorized function pattern**

```mojo
@parameter
fn vectorized_add[width: Int](i: Int):
    global_idx = tile_start + i
    if global_idx + width <= size:  # Bounds checking
        # SIMD operations here
```

The `width` parameter is automatically determined by the vectorize function.

### 3. **Calling vectorize**

```mojo
vectorize[vectorized_add, simd_width](actual_tile_size)
```

This automatically handles the vectorization loop with the provided SIMD width.

### 4. **Key characteristics**

- Automatic remainder handling, built-in safety, tile-based access
- Takes explicit SIMD width parameter
- Built-in bounds checking and automatic remainder element processing

</div>
</details>

### Running Mojo vectorize

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p23 --vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run p23 --vectorized
```

  </div>
</div>

Your output will look like this when not yet solved:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0 tile_start: 0 tile_end: 32 actual_tile_size: 32
tile_id: 1 tile_start: 32 tile_end: 64 actual_tile_size: 32
tile_id: 2 tile_start: 64 tile_end: 96 actual_tile_size: 32
tile_id: 3 tile_start: 96 tile_end: 128 actual_tile_size: 32
...
tile_id: 29 tile_start: 928 tile_end: 960 actual_tile_size: 32
tile_id: 30 tile_start: 960 tile_end: 992 actual_tile_size: 32
tile_id: 31 tile_start: 992 tile_end: 1024 actual_tile_size: 32
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### Mojo vectorize solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p23/p23.mojo:vectorize_within_tiles_elementwise_add_solution}}
```

<div class="solution-explanation">

### Mojo vectorize deep dive

**Mojo's vectorize function** provides automatic vectorization with built-in safety:

- **Explicit SIMD width parameter**: You provide the simd_width to use
- **Built-in bounds checking**: Prevents buffer overruns automatically
- **Automatic remainder handling**: Processes leftover elements automatically
- **Nested function pattern**: Clean separation of vectorization logic

**Tile-based organization:**

```mojo
tile_start = tile_id * tile_size    # 0, 32, 64, 96, ...
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

**Automatic vectorization mechanism:**

```mojo
@parameter
fn vectorized_add[width: Int](i: Int):
    global_idx = tile_start + i
    if global_idx + width <= size:
        # Automatic SIMD optimization
```

**How vectorize works:**

- **Automatic chunking**: Divides `actual_tile_size` into chunks of your provided `simd_width`
- **Remainder handling**: Automatically processes leftover elements with smaller widths
- **Bounds safety**: Automatically prevents buffer overruns
- **Loop management**: Handles the vectorization loop automatically

**Execution visualization (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Tile 0 processing:
  vectorize call 0: processes elements [0:4]   with SIMD_WIDTH=4
  vectorize call 1: processes elements [4:8]   with SIMD_WIDTH=4
  ...
  vectorize call 7: processes elements [28:32] with SIMD_WIDTH=4
  Total: 8 automatic SIMD operations
```

**Performance characteristics:**

- **Thread count**: 32 threads (1024 ÷ 32 = 32)
- **Work per thread**: 32 elements (automatic SIMD chunking)
- **Memory pattern**: Smaller tiles with automatic vectorization
- **Overhead**: Slight - automatic optimization and bounds checking
- **Safety**: Built-in bounds checking and edge case handling

</div>
</details>

## Performance comparison and best practices

### When to use each approach

**Choose manual vectorization when:**

- **Maximum performance** is critical
- You have **predictable, aligned data** patterns
- **Expert-level control** over memory access is needed
- You can **guarantee bounds safety** manually
- **Hardware-specific optimization** is required

**Choose Mojo vectorize when:**

- **Development speed** and safety are priorities
- Working with **irregular or dynamic data sizes**
- You want **automatic remainder handling** instead of manual edge case management
- **Bounds checking** complexity would be error-prone
- You prefer **cleaner vectorization patterns** over manual loop management

### Advanced optimization insights

**Memory bandwidth utilization:**

```
Manual:    8 threads × 32 SIMD ops = 256 total SIMD operations
Vectorize: 32 threads × 8 SIMD ops = 256 total SIMD operations
```

Both achieve similar total throughput but with different parallelism strategies.

**Cache behavior:**

- **Manual**: Large chunks may exceed L1 cache, but perfect sequential access
- **Vectorize**: Smaller tiles fit better in cache, with automatic remainder handling

**Hardware mapping:**

- **Manual**: Direct control over warp utilization and SIMD unit mapping
- **Vectorize**: Simplified vectorization with automatic loop and remainder management

### Best practices summary

**Manual vectorization best practices:**

- Always validate index calculations carefully
- Use compile-time constants for `chunk_size` when possible
- Profile memory access patterns for cache optimization
- Consider alignment requirements for optimal SIMD performance

**Mojo vectorize best practices:**

- Choose appropriate SIMD width for your data and hardware
- Focus on algorithm clarity over micro-optimizations
- Use nested parameter functions for clean vectorization logic
- Trust automatic bounds checking and remainder handling for edge cases

Both approaches represent valid strategies in the GPU performance optimization toolkit, with manual vectorization offering maximum control and Mojo's vectorize providing safety and automatic remainder handling.

## Next steps

Now that you understand all three fundamental patterns:

- **[🧠 GPU Threading vs SIMD](./gpu-thread-vs-simd.md)**: Understanding the execution hierarchy
- **[📊 Benchmarking](./benchmarking.md)**: Performance analysis and optimization

💡 **Key takeaway**: Different vectorization strategies suit different performance requirements. Manual vectorization gives maximum control, while Mojo's vectorize function provides safety and automatic remainder handling. Choose based on your specific performance needs and development constraints.
