# warp.sum() Essentials - Warp-Level Dot Product

Implement the dot product we saw in [puzzle 12](../puzzle_12/puzzle_12.md) using Mojo's warp operations to replace complex shared memory patterns with simple function calls. Each warp lane will process one element and use `warp.sum()` to combine results automatically, demonstrating how warp programming transforms GPU synchronization.

**Key insight:** _The [warp.sum()](https://docs.modular.com/mojo/stdlib/gpu/warp/sum) operation leverages SIMT execution to replace shared memory + barriers + tree reduction with a single hardware-accelerated instruction._

## Key concepts

In this puzzle, you'll master:
- **Warp-level reductions** with `warp.sum()`
- **SIMT execution model** and lane synchronization
- **Cross-architecture compatibility** with `WARP_SIZE`
- **Performance transformation** from complex to simple patterns
- **Lane ID management** and conditional writes

The mathematical operation is a dot product (inner product):
\\[\Large \text{output}[0] = \sum_{i=0}^{N-1} a[i] \times b[i]\\]

But the implementation teaches fundamental patterns for all warp-level GPU programming in Mojo.

## Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU architecture)
- Data type: `DType.float32`
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Grid configuration: `(1, 1)` blocks per grid
- Layout: `Layout.row_major(SIZE)` (1D row-major)

## The traditional complexity (from Puzzle 12)

Recall the complex approach from [solutions/p12/p12.mojo](../../../solutions/p12/p12.mojo) that required shared memory, barriers, and tree reduction:

```mojo
{{#include ../../../problems/p24/p24.mojo:traditional_approach_from_p12}}
```

**What makes this complex:**
- **Shared memory allocation**: Manual memory management within blocks
- **Explicit barriers**: `barrier()` calls to synchronize threads
- **Tree reduction**: Complex loop with stride-based indexing
- **Conditional writes**: Only thread 0 writes the final result

This works, but it's verbose, error-prone, and requires deep understanding of GPU synchronization.

**Test the traditional approach:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --traditional
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --traditional
```

  </div>
</div>

## Code to complete

### 1. Simple warp kernel approach

Transform the complex traditional approach into a simple warp kernel using `warp_sum()`:

```mojo
{{#include ../../../problems/p24/p24.mojo:simple_warp_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p24/p24.mojo" class="filename">View full file: problems/p24/p24.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding the simple warp kernel structure**

You need to complete the `simple_warp_dot_product` function with **6 lines or fewer**:

```mojo
fn simple_warp_dot_product[...](output, a, b):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL IN (6 lines at most)
```

**Pattern to follow:**
1. Compute partial product for this thread's element
2. Use `warp_sum()` to combine across all warp lanes
3. Lane 0 writes the final result

### 2. **Computing partial products**
```mojo
var partial_product: Scalar[dtype] = 0
if global_i < size:
    partial_product = (a[global_i] * b[global_i]).reduce_add()
```

**Why `.reduce_add()`?** Values in Mojo are SIMD-based, so `a[global_i] * b[global_i]` returns a SIMD vector. Use `.reduce_add()` to sum the vector into a scalar.

**Bounds checking:** Essential because not all threads may have valid data to process.

### 3. **Warp reduction magic**
```mojo
total = warp_sum(partial_product)
```

**What `warp_sum()` does:**
- Takes each lane's `partial_product` value
- Sums them across all lanes in the warp (hardware-accelerated)
- Returns the same total to **all lanes** (not just lane 0)
- Requires **zero explicit synchronization** (SIMT handles it)

### 4. **Writing the result**
```mojo
if lane_id() == 0:
    output[0] = total
```

**Why only lane 0?** All lanes have the same `total` value after `warp_sum()`, but we only want to write once to avoid race conditions.

**`lane_id()`:** Returns 0-31 (NVIDIA) or 0-63 (AMD) - identifies which lane within the warp.

</div>
</details>

**Test the simple warp kernel:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --kernel
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --kernel
```

  </div>
</div>

Expected output when solved:
```txt
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
=== RESULT ===
out: 10416.0
expected: 10416.0
🚀 Notice how simple the warp version is compared to p12.mojo!
   Same kernel structure, but warp_sum() replaces all the complexity!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p24/p24.mojo:simple_warp_kernel_solution}}
```

<div class="solution-explanation">

The simple warp kernel demonstrates the fundamental transformation from complex synchronization to hardware-accelerated primitives:

**What disappeared from the traditional approach:**
- **15+ lines → 6 lines**: Dramatic code reduction
- **Shared memory allocation**: Zero memory management required
- **3+ barrier() calls**: Zero explicit synchronization
- **Complex tree reduction**: Single function call
- **Stride-based indexing**: Eliminated entirely

**SIMT execution model:**
```
Warp lanes (SIMT execution):
Lane 0: partial_product = a[0] * b[0]    = 0.0
Lane 1: partial_product = a[1] * b[1]    = 4.0
Lane 2: partial_product = a[2] * b[2]    = 16.0
...
Lane 31: partial_product = a[31] * b[31] = 3844.0

warp_sum() hardware operation:
All lanes → 0.0 + 4.0 + 16.0 + ... + 3844.0 = 10416.0
All lanes receive → total = 10416.0 (broadcast result)
```

**Why this works without barriers:**
1. **SIMT execution**: All lanes execute each instruction simultaneously
2. **Hardware synchronization**: When `warp_sum()` begins, all lanes have computed their `partial_product`
3. **Built-in communication**: GPU hardware handles the reduction operation
4. **Broadcast result**: All lanes receive the same `total` value

</div>
</details>

### 2. Functional approach

Now implement the same warp dot product using Mojo's functional programming patterns:

```mojo
{{#include ../../../problems/p24/p24.mojo:functional_warp_approach}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding the functional approach structure**

You need to complete the `compute_dot_product` function with **10 lines or fewer**:

```mojo
@parameter
@always_inline
fn compute_dot_product[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
    idx = indices[0]
    # FILL IN (10 lines at most)
```

**Functional pattern differences:**
- Uses `elementwise` to launch exactly `WARP_SIZE` threads
- Each thread processes one element based on `idx`
- Same warp operations, different launch mechanism

### 2. **Computing partial products**
```mojo
var partial_product: Scalar[dtype] = 0.0
if idx < size:
    a_val = a.load[1](idx, 0)
    b_val = b.load[1](idx, 0)
    partial_product = (a_val * b_val).reduce_add()
else:
    partial_product = 0.0
```

**Loading pattern:** `a.load[1](idx, 0)` loads exactly 1 element at position `idx` (not SIMD vectorized).

**Bounds handling:** Set `partial_product = 0.0` for out-of-bounds threads so they don't contribute to the sum.

### 3. **Warp operations and storing**
```mojo
total = warp_sum(partial_product)

if lane_id() == 0:
    output.store[1](0, 0, total)
```

**Storage pattern:** `output.store[1](0, 0, total)` stores 1 element at position (0, 0) in the output tensor.

**Same warp logic:** `warp_sum()` and lane 0 writing work identically in functional approach.

### 4. **Available functions from imports**
```mojo
from gpu import lane_id
from gpu.warp import sum as warp_sum, WARP_SIZE

# Inside your function:
my_lane = lane_id()           # 0 to WARP_SIZE-1
total = warp_sum(my_value)    # Hardware-accelerated reduction
warp_size = WARP_SIZE         # 32 (NVIDIA) or 64 (AMD)
```

</div>
</details>

**Test the functional approach:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --functional
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --functional
```

  </div>
</div>

Expected output when solved:
```txt
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
=== RESULT ===
out: 10416.0
expected: 10416.0
🔧 Functional approach shows modern Mojo style with warp operations!
   Clean, composable, and still leverages warp hardware primitives!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p24/p24.mojo:functional_warp_approach_solution}}
```

<div class="solution-explanation">

The functional warp approach showcases modern Mojo programming patterns with warp operations:

**Functional approach characteristics:**
```mojo
elementwise[compute_dot_product, 1, target="gpu"](WARP_SIZE, ctx)
```

**Benefits:**
- **Type safety**: Compile-time tensor layout checking
- **Composability**: Easy integration with other functional operations
- **Modern patterns**: Leverages Mojo's functional programming features
- **Automatic optimization**: Compiler can apply high-level optimizations

**Key differences from kernel approach:**
- **Launch mechanism**: Uses `elementwise` instead of `enqueue_function`
- **Memory access**: Uses `.load[1]()` and `.store[1]()` patterns
- **Integration**: Seamlessly works with other functional operations

**Same warp benefits:**
- **Zero synchronization**: `warp_sum()` works identically
- **Hardware acceleration**: Same performance as kernel approach
- **Cross-architecture**: `WARP_SIZE` adapts automatically

</div>
</details>

## Performance comparison with benchmarks

Run comprehensive benchmarks to see how warp operations scale:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --benchmark
```

  </div>
</div>

Here's example output from a complete benchmark run:
```
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
--------------------------------------------------------------------------------
Testing SIZE=1 x WARP_SIZE, BLOCKS=1
Running traditional_1x
Running simple_warp_1x
Running functional_warp_1x
--------------------------------------------------------------------------------
Testing SIZE=4 x WARP_SIZE, BLOCKS=4
Running traditional_4x
Running simple_warp_4x
Running functional_warp_4x
--------------------------------------------------------------------------------
Testing SIZE=32 x WARP_SIZE, BLOCKS=32
Running traditional_32x
Running simple_warp_32x
Running functional_warp_32x
--------------------------------------------------------------------------------
Testing SIZE=256 x WARP_SIZE, BLOCKS=256
Running traditional_256x
Running simple_warp_256x
Running functional_warp_256x
--------------------------------------------------------------------------------
Testing SIZE=2048 x WARP_SIZE, BLOCKS=2048
Running traditional_2048x
Running simple_warp_2048x
Running functional_warp_2048x
--------------------------------------------------------------------------------
Testing SIZE=16384 x WARP_SIZE, BLOCKS=16384 (Large Scale)
Running traditional_16384x
Running simple_warp_16384x
Running functional_warp_16384x
--------------------------------------------------------------------------------
Testing SIZE=65536 x WARP_SIZE, BLOCKS=65536 (Massive Scale)
Running traditional_65536x
Running simple_warp_65536x
Running functional_warp_65536x
| name                   | met (ms)           | iters |
| ---------------------- | ------------------ | ----- |
| traditional_1x         | 1.0263419180000002 | 1000  |
| simple_warp_1x         | 1.025756103        | 1000  |
| functional_warp_1x     | 1.027618774        | 1000  |
| traditional_4x         | 1.026372558        | 1000  |
| simple_warp_4x         | 1.0274108880000001 | 1000  |
| functional_warp_4x     | 1.0272440180000002 | 1000  |
| traditional_32x        | 1.029869628        | 1000  |
| simple_warp_32x        | 1.029203002        | 1000  |
| functional_warp_32x    | 1.0293903800000002 | 1000  |
| traditional_256x       | 1.055470581        | 1000  |
| simple_warp_256x       | 1.0549002680000001 | 1000  |
| functional_warp_256x   | 1.054106567        | 1000  |
| traditional_2048x      | 1.170297851        | 1000  |
| simple_warp_2048x      | 1.1691909169999999 | 1000  |
| functional_warp_2048x  | 1.166839843        | 1000  |
| traditional_16384x     | 6.470711037837837  | 185   |
| simple_warp_16384x     | 6.482257572972973  | 185   |
| functional_warp_16384x | 6.414636946524065  | 187   |
| traditional_65536x     | 22.48350437735849  | 53    |
| simple_warp_65536x     | 22.561115754716983 | 53    |
| functional_warp_65536x | 22.399149188679246 | 53    |

Benchmarks completed!

WARP OPERATIONS PERFORMANCE ANALYSIS:
   GPU Architecture: NVIDIA (WARP_SIZE=32) vs AMD (WARP_SIZE=64)
   - 1 x WARP_SIZE: Single warp baseline
   - 4 x WARP_SIZE: Few warps, warp overhead visible
   - 32 x WARP_SIZE: Medium scale, warp benefits emerge
   - 256 x WARP_SIZE: Large scale, dramatic warp advantages
   - 2048 x WARP_SIZE: Massive scale, warp operations dominate
   - 16384 x WARP_SIZE: Large scale (512K-1M elements)
   - 65536 x WARP_SIZE: Massive scale (2M-4M elements)
   - Note: AMD GPUs process 2 x elements per warp vs NVIDIA!

   Expected Results at Large Scales:
   • Traditional: Slower due to more barrier overhead
   • Warp operations: Faster, scale better with problem size
   • Memory bandwidth becomes the limiting factor
```

**Performance insights from this example:**
- **Small scales (1x-4x)**: Warp operations show modest improvements (~10-15% faster)
- **Medium scale (32x-256x)**: Functional approach often performs best
- **Large scales (16K-65K)**: All approaches converge as memory bandwidth dominates
- **Variability**: Performance depends heavily on specific GPU architecture and memory subsystem

**Note:** Your results will vary significantly depending on your hardware (GPU model, memory bandwidth, `WARP_SIZE`). The key insight is observing the relative performance trends rather than absolute timings.



## Next Steps

Once you've mastered warp sum operations, you're ready for:

- **[When to Use Warp Programming](./warp_extra.md)**: Strategic decision framework for warp vs traditional approaches
- **Advanced warp operations**: `shuffle_idx()`, `shuffle_down()`, `prefix_sum()` for complex communication patterns
- **Multi-warp algorithms**: Combining warp operations with block-level synchronization
- **Part VII: Memory Coalescing**: Optimizing memory access patterns for maximum bandwidth

💡 **Key Takeaway**: Warp operations transform GPU programming by replacing complex synchronization patterns with hardware-accelerated primitives, demonstrating how understanding the execution model enables dramatic simplification without sacrificing performance.
