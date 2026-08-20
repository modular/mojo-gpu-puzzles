# Elementwise - Basic GPU Functional Operations

This puzzle implements vector addition using Mojo's functional `elementwise`
pattern. Each thread automatically processes multiple SIMD elements, showing how
modern GPU programming abstracts low-level details while preserving high
performance.

**Key insight:** _The
[elementwise](https://max.modular.com/api/mojo/max/algorithm/functional/elementwise/)
function automatically handles thread management, SIMD vectorization, and memory
coalescing for you._

## Key concepts

This puzzle covers:

- **Functional GPU programming** with `elementwise`
- **Automatic SIMD vectorization** within GPU threads
- **TileTensor operations** for safe memory access
- **GPU thread hierarchy** vs SIMD operations
- **Capture semantics** in nested closures

The mathematical operation is simple element-wise addition:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

The implementation covers fundamental patterns applicable to all GPU functional
programming in Mojo.

**Where to start:** You begin from the `elementwise` template in the problem file
— there is no manual shared memory or thread-index math here. The key shift from
earlier puzzles is that each invocation of your nested function processes a whole
SIMD vector, not a single element. That's why you load and store with
`aligned_load[simd_width]` / `store[simd_width]` (vectorized) instead of indexing
one scalar at a time.

## Configuration

- Vector size: `SIZE = 1024`
- Data type: `DType.float32`
- SIMD width: Target-dependent (determined by GPU architecture and data type)
- Layout: `row_major[SIZE]()` (1D row-major)

> **Scope:** This is a single-kernel, per-element operation. The `elementwise`
> abstraction handles thread, block, and grid configuration for you — there is no
> cross-thread or cross-block communication to reason about here.

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
@always_inline
def your_function[
    simd_width: Int, alignment: Int = 1
](indices: Coord) {var} -> None:
    # Your implementation here
```

**Why each part matters:**

- `@always_inline`: Forces inlining to eliminate function call overhead in GPU
  kernels
- `{var}`: The capture list - allows access to variables from the outer scope
  (the input/output tensors)
- `Coord`: Carries the per-dimension indices for the current SIMD chunk; use
  `indices[0]` for 1D operations

### 2. **Index extraction and SIMD processing**

```mojo
var idx = Int(indices[0].value())  # Extract linear index for 1D operations
```

This `idx` represents the **starting position** for a SIMD vector, not a single
element. If `SIMD_WIDTH=4` (GPU-dependent), then:

- Thread 0 processes elements `[0, 1, 2, 3]` starting at `idx=0`
- Thread 1 processes elements `[4, 5, 6, 7]` starting at `idx=4`
- Thread 2 processes elements `[8, 9, 10, 11]` starting at `idx=8`
- And so on...

### 3. **SIMD loading pattern**

```mojo
var a_simd = a_lt.aligned_load[width=simd_width](Index(idx))  # Load 4 consecutive floats (GPU-dependent)
var b_simd = b_lt.aligned_load[width=simd_width](Index(idx))  # Load 4 consecutive floats (GPU-dependent)
```

`aligned_load` is a `LayoutTensor` method, so the receivers are the `a_lt` /
`b_lt` handles produced by `to_layout_tensor()` inside the kernel—not the
`TileTensor` parameters. This loads a **vectorized chunk** of data in a single
operation. The exact number of elements loaded depends on your GPU's SIMD
capabilities.

### 4. **Vector arithmetic**

```mojo
var result = a_simd + b_simd  # SIMD addition of 4 elements simultaneously (GPU-dependent)
```

This performs element-wise addition across the entire SIMD vector (if supported)
in parallel - much faster than 4 separate scalar additions.

### 5. **SIMD storing**

```mojo
out_lt.store[simd_width](Index(idx), result)  # Store 4 results at once (GPU-dependent)
```

Writes the entire SIMD vector back to memory in one operation.

### 6. **Calling the elementwise function**

```mojo
elementwise[simd_width=SIMD_WIDTH, target="gpu"](your_function, Coord(total_size), ctx)
```

- Your nested function is passed as a runtime argument, and the problem shape
  is wrapped in a `Coord`
- `total_size` is the `size` compile-time parameter of the enclosing
  function—the template already passes it
- `elementwise` sizes the grid itself and drives a grid-stride loop, so your
  function is invoked once per `SIMD_WIDTH`-wide chunk:
  `total_size // SIMD_WIDTH` invocations, spread over however many threads the
  GPU can keep busy

### 7. **Key debugging insight**

Add `print("idx:", idx)` to the nested function and re-run. You'll see values
like:

```text
idx: 0, idx: 4, idx: 8, idx: 12, ...
```

The values are multiples of `SIMD_WIDTH` (which is GPU-dependent), showing that
each invocation handles a different SIMD chunk. The order they print in is
arbitrary, since the threads run concurrently.

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

The elementwise functional pattern in Mojo introduces several fundamental
concepts for modern GPU programming:

### 1. **Functional abstraction philosophy**

The `elementwise` function represents a paradigm shift from traditional GPU
programming:

**Traditional CUDA/HIP approach:**

```mojo
# Manual thread management
var idx = thread_idx.x + block_idx.x * block_dim.x
if idx < size:
    output[idx] = a[idx] + b[idx]  # Scalar operation
```

**Mojo functional approach:**

```mojo
# Automatic management + SIMD vectorization
elementwise[simd_width=simd_width, target="gpu"](add_function, Coord(size), ctx)
```

**What `elementwise` abstracts away:**

- **Thread grid configuration**: No need to calculate block/grid dimensions
- **Tail handling**: Elements left over when `SIMD_WIDTH` doesn't divide the
  shape are re-invoked one at a time, with `width=1`
- **Memory coalescing**: Threads walk the array with a grid stride, so
  neighboring threads touch neighboring chunks
- **SIMD orchestration**: Vectorization handled transparently
- **GPU target selection**: Works across different GPU architectures

### 2. **Deep dive: nested function architecture**

```mojo
@always_inline
def add[
    simd_width: Int, alignment: Int = 1
](indices: Coord) {var} -> None:
```

**Parameter Analysis:**

- **`simd_width: Int`**: A compile-time parameter, so the function is
  instantiated separately for each unique `simd_width`, allowing aggressive
  optimization.
- **`@always_inline`**: Critical for GPU performance - eliminates function call
  overhead by embedding the code directly into the kernel.
- **`{var}`**: The capture list enables **lexical scoping** - the inner function
  can access variables from the outer scope without explicit parameter passing.
- **`Coord`**: Carries the per-dimension indices for the SIMD chunk being
  processed; `indices[0]` is the linear start position for 1D operations.

### 3. **SIMD execution model deep dive**

```mojo
var idx = Int(indices[0].value())                            # Linear index: 0, 4, 8, 12... (GPU-dependent spacing)
var a_lt = a.to_layout_tensor()                              # LayoutTensor views for vectorized access
var b_lt = b.to_layout_tensor()
var out_lt = output.to_layout_tensor()
var a_simd = a_lt.aligned_load[width=simd_width](Index(idx))  # Load: [a[0:4], a[4:8], a[8:12]...] (4 elements per load)
var b_simd = b_lt.aligned_load[width=simd_width](Index(idx))  # Load: [b[0:4], b[4:8], b[8:12]...] (4 elements per load)
var ret = a_simd + b_simd                                    # SIMD: 4 additions in parallel (GPU-dependent)
out_lt.store[simd_width](Index(idx), ret)                    # Store: 4 results simultaneously (GPU-dependent)
```

**Execution Hierarchy Visualization:**

```text
GPU Architecture:
├── Grid (entire problem)
│   ├── Block 1 (multiple warps)
│   │   ├── Warp 1 (32 threads) --> We'll learn about Warp in Part VII
│   │   │   ├── Thread 1 → SIMD[4 elements]  ← Our focus (GPU-dependent width)
│   │   │   ├── Thread 2 → SIMD[4 elements]
│   │   │   └── ...
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (multiple warps)
```

**For a 1024-element vector with SIMD_WIDTH=4 (example GPU):**

- **Total SIMD operations needed**: 1024 ÷ 4 = 256
- **Body invocations**: 256, one per SIMD chunk
- **Each invocation processes**: Exactly 4 consecutive elements
- **Threads**: However many `elementwise` decides it needs to saturate the
  GPU - the 256 chunks are handed out across that grid, so a thread may run the
  body once or several times

**Note**: SIMD width varies with the target's vector register width and the data
type. Current NVIDIA and AMD GPU targets report a 128-bit vector width, which
gives `SIMD_WIDTH = 4` for `float32`.

### 4. **Memory access pattern analysis**

```mojo
a_lt.aligned_load[width=simd_width](Index(idx))  # Coalesced memory access
```

**Memory Coalescing Benefits:**

- **Sequential access**: Threads access consecutive memory locations
- **Cache optimization**: Maximizes L1/L2 cache hit rates
- **Bandwidth utilization**: Achieves near-theoretical memory bandwidth
- **Hardware efficiency**: GPU memory controllers optimized for this pattern

**Example for SIMD_WIDTH=4 (GPU-dependent):**

```text
Thread 0: loads a[0:4]   → bytes 0-15
Thread 1: loads a[4:8]   → bytes 16-31
Thread 2: loads a[8:12]  → bytes 32-47
...
Result: A 32-lane warp covers one contiguous 512-byte span, so the memory
        controller fetches whole cache lines instead of scattered words
```

### 5. **Performance characteristics & optimization**

**Computational Intensity Analysis (for SIMD_WIDTH=4):**

- **Arithmetic operations**: 1 SIMD addition per 4 elements
- **Memory operations**: 2 SIMD loads + 1 SIMD store per 4 elements
- **Arithmetic intensity**: 1 add ÷ 3 memory ops = 0.33 (memory-bound)

**Why This Is Memory-Bound:**

```text
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
comptime SIMD_WIDTH = simd_width_of[dtype, target=get_gpu_target()]()
```

- **GPU-specific optimization**: SIMD width adapts to hardware - it is the
  target's SIMD register width divided by the size of `dtype`, so a 128-bit
  vector register gives 4 for `float32` and 8 for `float16`
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

```cpp
// Traditional: error-prone, verbose, hardware-specific
__global__ void add_kernel(float* output, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];  // No vectorization
    }
}
```

```mojo
# Mojo: safe, concise, automatically vectorized
elementwise[simd_width=SIMD_WIDTH, target="gpu"](add, Coord(size), ctx)
```

**Benefits of Functional Approach:**

- **Safety**: The tail is handled for you, so a shape that isn't a multiple of
  `SIMD_WIDTH` doesn't need a hand-written remainder branch
- **Portability**: Same code works across GPU vendors/generations
- **Performance**: Grid sizing and the SIMD width come from the target, so the
  same source adapts instead of being retuned by hand
- **Maintainability**: Clean abstractions reduce debugging complexity
- **Composability**: Easy to combine with other functional operations

This pattern represents the future of GPU programming - high-level abstractions
that don't sacrifice performance, making GPU computing accessible while
maintaining optimal efficiency.

</div>
</details>

## Next steps

Once you've learned elementwise operations, you're ready for:

- **[Tile Operations](./tile.md)**: Memory-efficient tiled processing patterns
- **[Vectorization](./vectorize.md)**: Fine-grained SIMD control
- **[🧠 GPU Threading vs SIMD](./gpu-thread-vs-simd.md)**: Understanding the
  execution hierarchy
- **[📊 Benchmarking](./benchmarking.md)**: Performance analysis and
  optimization

💡 **Key Takeaway**: The `elementwise` pattern shows how Mojo combines
functional programming elegance with GPU performance, automatically handling
vectorization and thread management while maintaining full control over the
computation.
