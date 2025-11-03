# Elementwise - Basic GPU Functional Operations

This puzzle implements vector addition using Mojo's functional `elementwise` pattern. Each thread automatically processes multiple SIMD elements, showing how modern GPU programming abstracts low-level details while preserving high performance.

**Key insight:** _The [elementwise](https://docs.modular.com/mojo/stdlib/algorithm/functional/elementwise/) function automatically handles thread management, SIMD vectorization, and memory coalescing for you._

## Key concepts

This puzzle covers:

- **Functional GPU programming** with `elementwise`
- **Automatic SIMD vectorization** within GPU threads
- **LayoutTensor operations** for safe memory access
- **GPU thread hierarchy** vs SIMD operations
- **Capturing semantics** in nested functions

The mathematical operation is simple element-wise addition:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

The implementation covers fundamental patterns applicable to all GPU functional programming in Mojo.

## Configuration

- Vector size: `SIZE = 1024`
- Data type: `DType.float32`
- SIMD width: Target-dependent (determined by GPU architecture and data type)
- Layout: `Layout.row_major(SIZE)` (1D row-major)

## Code to complete

```mojo
{{#include ../../../problems/p23/p23.mojo:elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">View full file: problems/p23/p23.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding the function structure**

The `elementwise` function expects a nested function with this exact signature:

```mojo
@parameter
@always_inline
fn your_function[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
    # Your implementation here
```

**Why each part matters:**

- `@parameter`: Enables compile-time specialization for optimal GPU code generation
- `@always_inline`: Forces inlining to eliminate function call overhead in GPU kernels
- `capturing`: Allows access to variables from the outer scope (the input/output tensors)
- `IndexList[rank]`: Provides multi-dimensional indexing (rank=1 for vectors, rank=2 for matrices)

### 2. **Index extraction and SIMD processing**

```mojo
idx = indices[0]  # Extract linear index for 1D operations
```

This `idx` represents the **starting position** for a SIMD vector, not a single element. If `SIMD_WIDTH=4` (GPU-dependent), then:

- Thread 0 processes elements `[0, 1, 2, 3]` starting at `idx=0`
- Thread 1 processes elements `[4, 5, 6, 7]` starting at `idx=4`
- Thread 2 processes elements `[8, 9, 10, 11]` starting at `idx=8`
- And so on...

### 3. **SIMD loading pattern**

```mojo
a_simd = a.aligned_load[simd_width](idx, 0)  # Load 4 consecutive floats (GPU-dependent)
b_simd = b.aligned_load[simd_width](idx, 0)  # Load 4 consecutive floats (GPU-dependent)
```

The second parameter `0` is the dimension offset (always 0 for 1D vectors). This loads a **vectorized chunk** of data in a single operation. The exact number of elements loaded depends on your GPU's SIMD capabilities.

### 4. **Vector arithmetic**

```mojo
result = a_simd + b_simd  # SIMD addition of 4 elements simultaneously (GPU-dependent)
```

This performs element-wise addition across the entire SIMD vector (if supported) in parallel - much faster than 4 separate scalar additions.

### 5. **SIMD storing**

```mojo
output.store[simd_width](idx, 0, result)  # Store 4 results at once (GPU-dependent)
```

Writes the entire SIMD vector back to memory in one operation.

### 6. **Calling the elementwise function**

```mojo
elementwise[your_function, SIMD_WIDTH, target="gpu"](total_size, ctx)
```

- `total_size` should be `a.size()` to process all elements
- The GPU automatically determines how many threads to launch: `total_size // SIMD_WIDTH`

### 7. **Key debugging insight**

Notice the `print("idx:", idx)` in the template. When you run it, you'll see:

```
idx: 0, idx: 4, idx: 8, idx: 12, ...
```

This shows that each thread handles a different SIMD chunk, automatically spaced by `SIMD_WIDTH` (which is GPU-dependent).

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
pixi run p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --elementwise
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
SIZE: 1024
simd_width: 4
...
idx: 404
idx: 408
idx: 412
idx: 416
...

out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p23/p23.mojo:elementwise_add_solution}}
```

<div class="solution-explanation">

The elementwise functional pattern in Mojo introduces several fundamental concepts for modern GPU programming:

### 1. **Functional abstraction philosophy**

The `elementwise` function represents a paradigm shift from traditional GPU programming:

**Traditional CUDA/HIP approach:**

```mojo
# Manual thread management
idx = thread_idx.x + block_idx.x * block_dim.x
if idx < size:
    output[idx] = a[idx] + b[idx];  // Scalar operation
```

**Mojo functional approach:**

```mojo
# Automatic management + SIMD vectorization
elementwise[add_function, simd_width, target="gpu"](size, ctx)
```

**What `elementwise` abstracts away:**

- **Thread grid configuration**: No need to calculate block/grid dimensions
- **Bounds checking**: Automatic handling of array boundaries
- **Memory coalescing**: Optimal memory access patterns built-in
- **SIMD orchestration**: Vectorization handled transparently
- **GPU target selection**: Works across different GPU architectures

### 2. **Deep dive: nested function architecture**

```mojo
@parameter
@always_inline
fn add[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
```

**Parameter Analysis:**

- **`@parameter`**: This decorator provides **compile-time specialization**. The function is generated separately for each unique `simd_width` and `rank`, allowing aggressive optimization.
- **`@always_inline`**: Critical for GPU performance - eliminates function call overhead by embedding the code directly into the kernel.
- **`capturing`**: Enables **lexical scoping** - the inner function can access variables from the outer scope without explicit parameter passing.
- **`IndexList[rank]`**: Provides **dimension-agnostic indexing** - the same pattern works for 1D vectors, 2D matrices, 3D tensors, etc.

### 3. **SIMD execution model deep dive**

```mojo
idx = indices[0]                                  # Linear index: 0, 4, 8, 12... (GPU-dependent spacing)
a_simd = a.aligned_load[simd_width](idx, 0)       # Load: [a[0:4], a[4:8], a[8:12]...] (4 elements per load)
b_simd = b.aligned_load[simd_width](idx, 0)       # Load: [b[0:4], b[4:8], b[8:12]...] (4 elements per load)
ret = a_simd + b_simd                             # SIMD: 4 additions in parallel (GPU-dependent)
output.aligned_store[simd_width](idx, 0, ret)     # Store: 4 results simultaneously (GPU-dependent)
```

**Execution Hierarchy Visualization:**

```
GPU Architecture:
├── Grid (entire problem)
│   ├── Block 1 (multiple warps)
│   │   ├── Warp 1 (32 threads) --> We'll learn about Warp in the next Part VI
│   │   │   ├── Thread 1 → SIMD[4 elements]  ← Our focus (GPU-dependent width)
│   │   │   ├── Thread 2 → SIMD[4 elements]
│   │   │   └── ...
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (multiple warps)
```

**For a 1024-element vector with SIMD_WIDTH=4 (example GPU):**

- **Total SIMD operations needed**: 1024 ÷ 4 = 256
- **GPU launches**: 256 threads (1024 ÷ 4)
- **Each thread processes**: Exactly 4 consecutive elements
- **Memory bandwidth**: SIMD_WIDTH× improvement over scalar operations

**Note**: SIMD width varies by GPU architecture (e.g., 4 for some GPUs, 8 for RTX 4090, 16 for A100).

### 4. **Memory access pattern analysis**

```mojo
a.aligned_load[simd_width](idx, 0)  // Coalesced memory access
```

**Memory Coalescing Benefits:**

- **Sequential access**: Threads access consecutive memory locations
- **Cache optimization**: Maximizes L1/L2 cache hit rates
- **Bandwidth utilization**: Achieves near-theoretical memory bandwidth
- **Hardware efficiency**: GPU memory controllers optimized for this pattern

**Example for SIMD_WIDTH=4 (GPU-dependent):**

```
Thread 0: loads a[0:4]   → Memory bank 0-3
Thread 1: loads a[4:8]   → Memory bank 4-7
Thread 2: loads a[8:12]  → Memory bank 8-11
...
Result: Optimal memory controller utilization
```

### 5. **Performance characteristics & optimization**

**Computational Intensity Analysis (for SIMD_WIDTH=4):**

- **Arithmetic operations**: 1 SIMD addition per 4 elements
- **Memory operations**: 2 SIMD loads + 1 SIMD store per 4 elements
- **Arithmetic intensity**: 1 add ÷ 3 memory ops = 0.33 (memory-bound)

**Why This Is Memory-Bound:**

```
Memory bandwidth >>> Compute capability for simple operations
```

**Optimization Implications:**

- Focus on memory access patterns rather than arithmetic optimization
- SIMD vectorization provides the primary performance benefit
- Memory coalescing is critical for performance
- Cache locality matters more than computational complexity

### 6. **Scaling and adaptability**

**Automatic Hardware Adaptation:**

```mojo
alias SIMD_WIDTH = simd_width_of[dtype, target = _get_gpu_target()]()
```

- **GPU-specific optimization**: SIMD width adapts to hardware (e.g., 4 for some cards, 8 for RTX 4090, 16 for A100)
- **Data type awareness**: Different SIMD widths for float32 vs float16
- **Compile-time optimization**: Zero runtime overhead for hardware detection

**Scalability Properties:**

- **Thread count**: Automatically scales with problem size
- **Memory usage**: Linear scaling with input size
- **Performance**: Near-linear speedup until memory bandwidth saturation

### 7. **Advanced insights: why this pattern matters**

**Foundation for Complex Operations:**
This elementwise pattern is the building block for:

- **Reduction operations**: Sum, max, min across large arrays
- **Broadcast operations**: Scalar-to-vector operations
- **Complex transformations**: Activation functions, normalization
- **Multi-dimensional operations**: Matrix operations, convolutions

**Compared to Traditional Approaches:**

```mojo
// Traditional: Error-prone, verbose, hardware-specific
__global__ void add_kernel(float* output, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];  // No vectorization
    }
}

// Mojo: Safe, concise, automatically vectorized
elementwise[add, SIMD_WIDTH, target="gpu"](size, ctx)
```

**Benefits of Functional Approach:**

- **Safety**: Automatic bounds checking prevents buffer overruns
- **Portability**: Same code works across GPU vendors/generations
- **Performance**: Compiler optimizations often exceed hand-tuned code
- **Maintainability**: Clean abstractions reduce debugging complexity
- **Composability**: Easy to combine with other functional operations

This pattern represents the future of GPU programming - high-level abstractions that don't sacrifice performance, making GPU computing accessible while maintaining optimal efficiency.

</div>
</details>

## Next steps

Once you've learned elementwise operations, you're ready for:

- **[Tile Operations](./tile.md)**: Memory-efficient tiled processing patterns
- **[Vectorization](./vectorize.md)**: Fine-grained SIMD control
- **[🧠 GPU Threading vs SIMD](./gpu-thread-vs-simd.md)**: Understanding the execution hierarchy
- **[📊 Benchmarking](./benchmarking.md)**: Performance analysis and optimization

💡 **Key Takeaway**: The `elementwise` pattern shows how Mojo combines functional programming elegance with GPU performance, automatically handling vectorization and thread management while maintaining full control over the computation.
