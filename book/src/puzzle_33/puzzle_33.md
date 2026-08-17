# Puzzle 33: Tensor Core Operations

## Introduction

Welcome to the final frontier of GPU matrix multiplication optimization! In this
puzzle, we'll explore **Tensor Cores** - specialized hardware units designed to
accelerate mixed-precision matrix operations at unprecedented speeds.

Building on everything we've learned so far, especially from
[Puzzle 16's idiomatic tiled matrix multiplication](../puzzle_16/puzzle_16.md),
we'll see how modern GPUs provide dedicated silicon to make matrix operations
blazingly fast.

## What are tensor cores?

Tensor Cores (also known as Matrix Cores on AMD hardware) are specialized
processing units that can perform mixed-precision matrix-matrix operations in a
single instruction. These units are available on modern GPU architectures:

- **NVIDIA**: Tensor Cores (Volta, Turing, Ampere, Ada, Hopper, Blackwell)
- **AMD**: Matrix Cores (CDNA architectures, plus WMMA on RDNA 3 and RDNA 4)

Think of them as hardware-accelerated GEMM (General Matrix Multiply) engines
built directly into the GPU.

### Key characteristics

- **Warp-level operations**: Each instruction operates on data from an entire
  warp (32 threads on NVIDIA, 32 or 64 on AMD)
- **Fixed tile sizes**: Operations work on specific matrix fragment sizes (e.g.,
  16×8×8 for FP32)
- **Mixed precision**: Can mix input and output precisions for optimal
  performance
- **Massive throughput**: Peak matrix throughput is roughly an order of
  magnitude above the general-purpose FP32 pipeline—though, as the profiling
  section below shows, peak throughput is not the same as achieved performance

## From tiled to tensor cores

Let's trace our journey from basic matrix multiplication to Tensor Cores:

1. **Puzzle 16**: We learned idiomatic tiled matrix multiplication using shared
   memory
2. **Shared memory optimization**: We used `copy_dram_to_sram_async` for
   efficient memory transfers
3. **Thread cooperation**: We coordinated warps using barriers and async
   operations
4. **Now**: We'll use specialized hardware (Tensor Cores) to accelerate the core
   computation

## The tensor core programming model

Tensor Cores expose a different programming paradigm:

### Traditional compute core approach

```mojo
# Each thread computes one element
acc += a_shared[local_row, k] * b_shared[k, local_col]
```

### Tensor core approach

```mojo
# Entire warp cooperates on matrix fragments
var a_reg = mma_op.load_a(A_mma_tile)           # Load 16×8 fragment
var b_reg = mma_op.load_b(B_mma_tile)           # Load 8×8 fragment
var c_reg = mma_op.load_c(C_mma_tile)           # Load 16×8 accumulator
var d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)  # D = A×B + C
mma_op.store_d(C_mma_tile, d_reg)           # Store result
```

## Tensor core API in Mojo

Mojo provides a clean interface to Tensor Cores through the
[`TensorCore`](https://docs.modular.com/api/mojo/layout/tensor_core/TensorCore/)
type:

```mojo
from layout.tensor_core import TensorCore

# Create a Tensor Core operator for specific tile sizes
var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

# Core operations:
# - load_a(): Load matrix A fragment from shared memory
# - load_b(): Load matrix B fragment from shared memory
# - load_c(): Load matrix C fragment (accumulator)
# - mma_op(): Perform D = A×B + C operation
# - store_d(): Store result fragment to memory
```

**Advanced features:** The TensorCore API also supports quantized operations,
different swizzle patterns for memory access optimization, and mixed-precision
arithmetic. For complete documentation of all supported shapes, data types, and
methods, see the
[official TensorCore API reference](https://docs.modular.com/api/mojo/layout/tensor_core/TensorCore/).

### Matrix fragment sizes

The TensorCore API supports different shapes and data types depending on the GPU
hardware:

**NVIDIA GPUs:**

- **float32**: 16×8×8 or 16×8×4
- **half-precision**: 16×8×16
- **float8**: 16×8×32

**AMD GPUs:**

- **float32**: 16×16×4
- **half-precision**: 16×16×16 or 32×32×8

**This puzzle uses FP32 with 16×8×8 fragments:**

- **MMA_M = 16**: Matrix A height (and output height)
- **MMA_N = 8**: Matrix B width (and output width)
- **MMA_K = 8**: Inner dimension (A width = B height)

**What is MMA?** MMA stands for "Mixed-precision Matrix-Multiply-Accumulate" -
the fundamental operation that Tensor Cores perform. Each MMA instruction
computes: `D = A × B + C` where A, B, C, and D are matrix fragments.

**Fragment visualization:**

```txt
A fragment (16×8)  ×  B fragment (8×8)  +  C fragment (16×8)  =  D fragment (16×8)

    16 rows             8 rows               16 rows              16 rows
    8 cols              8 cols               8 cols               8 cols
      |                   |                    |                    |
   [A data]         ×   [B data]         +   [C data]         =  [D result]
```

This means each Tensor Core instruction computes a 16×8 output tile by
multiplying a 16×8 tile from A with an 8×8 tile from B, then adding it to the
existing 16×8 accumulator.

## Warp organization for tensor cores

**What is a warp?** A warp is a group of threads (32 on NVIDIA, 32 or 64 on AMD)
that execute instructions together in lockstep. Tensor Cores require all threads
in a warp to cooperate on a single matrix operation.

**Why warp-level?** Unlike regular operations where each thread works
independently, Tensor Cores need the entire warp to collectively load matrix
fragments, perform the MMA operation, and store results.

Since Tensor Cores operate at warp-level, we need to organize our threads
differently:

```mojo
# Calculate warp coordinates within the block
var warp_id = thread_idx.x // WARP_SIZE
var warps_in_n = BN // WN  # Number of warps along N dimension
var warps_in_m = BM // WM  # Number of warps along M dimension
var warp_y = warp_id // warps_in_n  # Warp's row
var warp_x = warp_id % warps_in_n   # Warp's column

# Each warp handles a WM×WN tile of the output
var C_warp_tile = C_block_tile.tile[WM, WN](warp_y, warp_x)
```

**Warp organization example** (with BM=128, BN=64, WM=32, WN=32):

```txt
Block (128×64) contains 8 warps arranged as:

    32 cols    32 cols
     |          |
[  Warp 0  ][  Warp 1  ]  ← 32 rows each
[  Warp 2  ][  Warp 3  ]  ← 32 rows each
[  Warp 4  ][  Warp 5  ]  ← 32 rows each
[  Warp 6  ][  Warp 7  ]  ← 32 rows each

Total: 4×2 = 8 warps, each handling 32×32 output region
```

## Memory hierarchy with tensor cores

Tensor Cores add another layer to our memory optimization:

1. **Global Memory** → **Shared Memory**: Use `copy_dram_to_sram_async` (from
   Puzzle 16)
2. **Shared Memory** → **Register Fragments**: Use `mma_op.load_a/load_b`
3. **Computation**: Use `mma_op.mma_op` on register fragments
4. **Register Fragments** → **Global Memory**: Use `mma_op.store_d`

## The challenge

Your task is to complete the `tensor_core_matrix_multiplication` function. The
skeleton builds on the tiled approach but uses actual Tensor Core hardware
operations.

### Key requirements

1. **Use the actual Tensor Core API**: Don't simulate - use real
   `mma_op.load_a()`, `mma_op.mma_op()`, etc.
2. **Maintain correctness**: Your result must match the CPU reference
   implementation
3. **Proper warp coordination**: Handle multiple warps per block correctly
4. **Memory efficiency**: Use the same async copy patterns from Puzzle 16
5. **Warp-width independence**: Ensure tiling parameters are multiples of
   `WARP_SIZE`

## Configuration

- Matrix size: \\(\\text{SIZE} = 1024\\)
- Block tiling: \\(\\text{BM} = 4 \\times \\text{WARP\_SIZE}\\),
  \\(\\text{BN} = 2 \\times \\text{WARP\_SIZE}\\),
  \\(\\text{BK} = \\text{WARP\_SIZE}\\)
- Warp tiling: \\(\\text{WM} = \\text{WN} = \\text{WARP\_SIZE}\\)
- MMA fragments: \\(16 \\times 8 \\times 8\\) for FP32
- Threads per block: \\(8 \\times \\text{WARP\_SIZE}\\) (8 warps per block)
- Grid dimensions: Covers full matrix with block tiles

Deriving every tile size from `WARP_SIZE` keeps the block and warp
decomposition portable across warp widths. The concrete numbers used throughout
this page assume \\(\\text{WARP\_SIZE} = 32\\), which gives
\\(\\text{BM} = 128\\), \\(\\text{BN} = 64\\), \\(\\text{BK} = 32\\), and
\\(\\text{WM} = \\text{WN} = 32\\). On a 64-wide AMD wavefront they all double,
and the fragment counts derived from them change accordingly.

The MMA shape does not follow that rule. `16×8×8` is an NVIDIA shape; AMD's
matrix cores take `16×16×4` for FP32, so this puzzle is NVIDIA-only—see the
[support matrix](../howto.md).

Layout configuration:

- Input A: `row_major[SIZE, SIZE]()`
- Input B: `row_major[SIZE, SIZE]()`
- Output C: `row_major[SIZE, SIZE]()`
- Shared Memory: Block-sized tiles with async copy operations

## The challenge

In this puzzle, you'll transform the idiomatic tiled matrix multiplication from
Puzzle 16 into a Tensor Core implementation. Let's break this down step by step:

### Step 1: Understanding your tiled baseline

The puzzle provides a complete idiomatic tiled implementation as your reference:

```mojo
{{#include ../../../problems/p33/p33.mojo:matmul_idiomatic_tiled_solution}}
```

**What this baseline does:**

- **Correctness**: This implementation works perfectly and passes all tests
- **Thread cooperation**: Uses `copy_dram_to_sram_async` for efficient memory
  transfers
- **Shared memory**: Coordinates threads with barriers and async operations
- **Tiled computation**: Each thread computes one output element using shared
  memory tiles

### Step 2: Your tensor core mission

Transform the above approach using specialized hardware acceleration:

- **From:** Thread-level computation → **To:** Warp-level matrix fragments
- **From:** Standard FP32 arithmetic → **To:** Hardware-accelerated GEMM
  operations
- **From:** Individual element results → **To:** 16×8 matrix fragment results

### Step 3: Configuration understanding

The tensor core version uses different tiling parameters optimized for hardware:

- **Block tiling**: `BM=128, BN=64, BK=32` (larger blocks for more reuse per
  shared memory tile)
- **Warp tiling**: `WM=32, WN=32` (each warp handles a 32×32 output region)
- **MMA fragments**: `16×8×8` (hardware-defined matrix fragment sizes)
- **Warps per block**: 8 warps (organized as 4×2 in the BM×BN block)

**Why these specific sizes?**

- **BM=128, BN=64**: Larger than tiled version (32×32) to better utilize Tensor
  Cores
- **WM=WN=32**: Multiple of WARP_SIZE and contains 2×4=8 MMA fragments (32÷16=2,
  32÷8=4)
- **MMA 16×8×8**: Fixed by hardware - this is what the Tensor Cores physically
  compute
- **8 warps**: BM÷WM × BN÷WN = 128÷32 × 64÷32 = 4×2 = 8 warps per block

**How warps map to MMA fragments:**

```txt
Each 32×32 warp tile contains multiple 16×8 MMA fragments
(MMA_M = 16 rows, MMA_N = 8 cols):

  8 cols    8 cols    8 cols    8 cols
    |         |         |         |
[ MMA 0,0 ][ MMA 0,1 ][ MMA 0,2 ][ MMA 0,3 ]  ← 16 rows each
[ MMA 1,0 ][ MMA 1,1 ][ MMA 1,2 ][ MMA 1,3 ]  ← 16 rows each

4 fragments across (32÷8=4) × 2 fragments down (32÷16=2) = 8 MMA operations per warp per K-slice
```

### Step 4: Code to complete

```mojo
{{#include ../../../problems/p33/p33.mojo:tensor_core_matrix_multiplication}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p33/p33.mojo" class="filename">View full file: problems/p33/p33.mojo</a>

**Your task**: Complete the missing section (marked with
`# FILL IN (roughly 8 lines)`) inside the triple nested loops.

**What you need to understand:**

- The skeleton handles all memory management, warp organization, and
  synchronization
- You only need to implement the core Tensor Core computation
- The loops iterate over MMA fragments: `mma_k`, `mma_m`, `mma_n`
- Each iteration processes one 16×8×8 matrix fragment

**Understanding the triple nested loops:**

```mojo
comptime for mma_k in range(BK // MMA_K):     # 32÷8 = 4 iterations (K dimension)
    comptime for mma_m in range(WM // MMA_M): # 32÷16 = 2 iterations (M dimension)
        comptime for mma_n in range(WN // MMA_N): # 32÷8 = 4 iterations (N dimension)
            # YOUR CODE HERE: Process one 16×8×8 MMA fragment
```

**What each loop does:**

- `mma_k`: Iterates through K-slices of the current K-tile (4 slices of 8
  elements each)
- `mma_m`: Iterates through M-slices of the warp's output (2 slices of 16 rows
  each)
- `mma_n`: Iterates through N-slices of the warp's output (4 slices of 8 columns
  each)
- **Total**: 4×2×4 = 32 MMA operations per warp per K-tile

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

Think about the Tensor Core workflow - you need to:

1. **Get the right matrix fragments**:
   - From the warp tiles (`A_warp_tile`, `B_warp_tile`, `C_warp_accum`), extract
     the specific MMA-sized fragments
   - Use the loop indices (`mma_m`, `mma_k`, `mma_n`) to get the correct tile
     coordinates
   - Remember: A needs [MMA_M, MMA_K], B needs [MMA_K, MMA_N], C needs
     [MMA_M, MMA_N]

2. **Load fragments into Tensor Core registers**:
   - The `mma_op` object has methods to load each matrix type
   - Each load method takes a tile and returns register fragments
   - Think: `load_a()`, `load_b()`, `load_c()` - what do they each take?

3. **Perform the hardware operation and store**:
   - Use the MMA operation to compute the result
   - Store the result back to the accumulator tile
   - The operation follows the pattern: result = A × B + C

**Key insight**: You're replacing 1024 individual multiply-add operations (a
16×8 output tile × 8 K-steps) with a single hardware instruction!

**Debugging tip**: If you get dimension errors, double-check your tile
indexing—the order of `mma_m`, `mma_k`, `mma_n` matters for getting the right
fragments.

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p33 --test
```

  </div>
  <div class="tab-content">

```bash
uv run poe p33 --test
```

  </div>
</div>

Your output will show accuracy test results once completed:

```txt
=== Running All Accuracy Tests ===
--- Test 1: Tensor Core vs CPU Reference ---
Tensor core test: passed
--- Test 2: Idiomatic Tiled vs CPU Reference ---
Idiomatic tiled test: passed

=== ACCURACY TEST SUMMARY ===
ALL TESTS PASSED!
Puzzle 33 complete ✅
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p33/p33.mojo:tensor_core_matrix_multiplication_solution}}
```

<div class="solution-explanation">

This solution demonstrates the Tensor Core programming model:

1. **Warp organization**
   - Calculates warp coordinates within the block using
     `warp_id = thread_idx.x // WARP_SIZE`
   - Maps warps to output tiles: each warp handles a `WM×WN` region
   - Uses a `warp_is_active` guard to skip any warp whose row lands outside the
     block's `BM // WM` warp rows

2. **Memory hierarchy optimization**
   - **Global → Shared**: Uses `copy_dram_to_sram_async` for efficient
     block-level transfers
   - **Shared → Registers**: Uses `mma_op.load_a/load_b` for warp-level fragment
     loading
   - **Register computation**: Uses `mma_op.mma_op` for hardware-accelerated
     matrix operations
   - **Registers → Global**: Uses `mma_op.store_d` for efficient result storage

3. **Tensor Core operations**
   - `load_a(A_mma_tile)`: Loads 16×8 matrix A fragment into registers
   - `load_b(B_mma_tile)`: Loads 8×8 matrix B fragment into registers
   - `load_c(C_mma_tile)`: Loads 16×8 accumulator fragment
   - `mma_op(a_reg, b_reg, c_reg)`: Computes D = A×B + C using specialized
     hardware
   - `store_d(C_mma_tile, d_reg)`: Stores 16×8 result fragment

4. **Warp-width independence**
   - All tiling parameters are multiples of `WARP_SIZE` (32 on NVIDIA, 32 on
     AMD RDNA, 64 on AMD CDNA)
   - Mojo abstracts hardware differences through the `TensorCore` interface,
     which exposes Tensor Cores and Matrix Cores through one API
   - The `16×8×8` fragment shape hardcoded here is NVIDIA-only, so porting to
     AMD Matrix Cores means picking an AMD shape such as `16×16×4`

The key insight is that Tensor Cores operate on entire matrix fragments at the
warp level, rather than individual elements at the thread level. This enables
massive parallelism and specialized hardware acceleration.

</div>
</details>

## Performance analysis: Are we done?

Now let's see if Tensor Cores deliver their promised performance advantage over
the idiomatic tiled approach.

### Building for profiling

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run mojo build problems/p33/p33.mojo -o problems/p33/p33_profiler
```

  </div>
  <div class="tab-content">

```bash
pixi run mojo build problems/p33/p33.mojo -o problems/p33/p33_profiler
```

  </div>
</div>

### Profiling with NVIDIA Nsight Compute (NVIDIA only)

First, enter the CUDA environment for `ncu` access:

```bash
# Enter CUDA environment
pixi shell -e nvidia

# Profile tensor core version
ncu --set full --metrics sm__cycles_elapsed.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor_op_hmma.sum ./problems/p33/p33_profiler --tensor-core

# Profile tiled version for comparison
ncu --set full --metrics sm__cycles_elapsed.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./problems/p33/p33_profiler --tiled
```

### Key metrics to compare

**Performance metrics:**

- **Duration**: Total kernel execution time (lower is better)
- **SM Active %**: Streaming multiprocessor utilization (higher is better)
- **DRAM Throughput**: Memory bandwidth utilization (shows if memory-bound)
- **Tensor Op Instructions**: Number of actual tensor core operations (tensor
  core only)

**What the results typically show:**

**Tensor Core version (slower):**

- **Duration**: ~13.9 ms (much slower!)
- **SM Active**: 83.7% (good utilization)
- **DRAM Throughput**: 72.5% (memory-bound!)
- **Occupancy**: 26.3% (poor - limited by registers)
- **Tensor Op Instructions**: 1,048,576 (confirms tensor cores are working)

**Tiled version (faster):**

- **Duration**: ~1.62 ms (8.6× faster!)
- **SM Active**: 98.0% (excellent utilization)
- **DRAM Throughput**: 1.7% (compute-bound, as expected)
- **Occupancy**: 66.7% (much better)
- **L2 Hit Rate**: 96.9% vs 29.7% (much better cache locality)

**Why is Tensor Core slower?**

- **Memory bottleneck**: 72% DRAM usage shows it's memory-bound, not
  compute-bound
- **Poor occupancy**: 26% vs 67% - high register usage (68 vs 38 per thread)
  limits concurrent warps
- **Cache misses**: 29% L2 hit rate vs 97% shows poor memory locality
- **Shared memory conflicts**: a plausible contributor, but none of the metrics
  above measure it—add the bank-conflict counters from
  [Puzzle 32](../puzzle_32/conflict_free_patterns.md) if you want to confirm it
- **Launch configuration**: Suboptimal block/warp organization for this problem
  size

## The performance reality

As you can see from the profiling results, the "specialized hardware" isn't
automatically faster! The Tensor Core version is significantly slower (~8.6×)
than the simple tiled approach. This is a common reality in GPU optimization -
raw hardware capability doesn't guarantee better performance.

**Key insights:**

- **Memory bottleneck**: 72% DRAM usage shows tensor cores are memory-bound, not
  compute-bound
- **Poor occupancy**: 26% vs 67% due to high register usage limits concurrent
  warps
- **Cache misses**: 29% vs 97% L2 hit rate shows poor memory locality
- **Resource waste**: suboptimal launch configuration, and possibly shared
  memory bank conflicts—which the profile above does not measure

**The lesson**: Understanding performance bottlenecks and systematic
optimization matter more than using the "latest and greatest" APIs. Hardware
features are tools that require careful tuning, not magic bullets.

## Next step

Ready for a rewarding GPU optimization challenge? Head to the
[🎯 Performance Bonus Challenge](../bonuses/part5.md) to learn how to transform
your memory-bound Tensor Core implementation into something that actually beats
the simple tiled version!
