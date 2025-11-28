# Puzzle 28: Async Memory Operations & Copy Overlap

**The GPU Memory Bottleneck:** Most real-world GPU algorithms hit a frustrating wall - they're not limited by compute power, but by **memory bandwidth**. Your expensive GPU cores sit idle, waiting for data to arrive from slow DRAM.

Consider this common scenario in GPU programming:

```mojo
# The performance killer - sequential memory operations
load_input_tile()     # ← 500 cycles waiting for DRAM
load_kernel_data()    # ← Another 100 cycles waiting
barrier()             # ← All threads wait idle
compute()             # ← Finally, 50 cycles of actual work
# Total: 650 cycles, only 7.7% compute utilization!
```

**What if you could do this instead?**

```mojo
# The performance win - overlapped operations
launch_async_load()   # ← Start 500-cycle transfer in background
load_small_data()     # ← 100 cycles of useful work while waiting
wait_and_compute()    # ← Only wait for remaining ~400 cycles, then compute
# Total: ~550 cycles, 45% better utilization!
```

**This is the power of async memory operations** - the difference between a sluggish algorithm and one that maximizes your GPU's potential.

## Why this matters

In this puzzle, you'll transform a memory-bound 1D convolution from [Puzzle 13](../puzzle_13/puzzle_13.md) into a high-performance implementation that **hides memory latency behind computation**. This isn't just an academic exercise - these patterns are fundamental to:

- **Deep learning**: Efficiently loading weights and activations
- **Scientific computing**: Overlapping data transfers in stencil operations
- **Image processing**: Streaming large datasets through memory hierarchies
- **Any memory-bound algorithm**: Converting waiting time into productive work

## Prerequisites

Before diving in, ensure you have solid foundation in:

**Essential GPU programming concepts:**

- **Shared memory programming** ([Puzzle 8](../puzzle_08/puzzle_08.md), [Puzzle 16](../puzzle_16/puzzle_16.md)) - You'll extend matmul patterns
- **Memory coalescing** ([Puzzle 21](../puzzle_21/puzzle_21.md)) - Critical for optimal async transfers
- **Tiled processing** ([Puzzle 23](../puzzle_23/puzzle_23.md)) - The foundation for this optimization

**Hardware understanding:**

- GPU memory hierarchy (DRAM → Shared Memory → Registers)
- Thread block organization and synchronization
- Basic understanding of memory latency vs. bandwidth

**API familiarity:** [Mojo GPU Memory Operations](https://docs.modular.com/mojo/stdlib/gpu/memory/)

> **⚠️ Hardware compatibility note:** This puzzle uses async copy operations (`copy_dram_to_sram_async`, `async_copy_wait_all`) that may require modern GPU architectures. If you encounter compilation errors related to `.async` modifiers or unsupported operations, your GPU may not support these features. The concepts remain valuable for understanding memory optimization patterns.
>
> **Check your GPU compute capability:**
>
> ```bash
> nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
> ```
>
> - **SM_70 and above** (e.g., V100, T4, A10G, RTX 20+ series): Basic async copy supported
> - **SM_80 and above** (e.g., A100, RTX 30+ series): Full async copy features
> - **SM_90 and above** (e.g., H100, RTX 40+ series): Advanced TMA operations supported

## What you'll focus

By the end of this puzzle, you'll have hands-on experience with:

### **Core techniques**

- **Async copy primitives**: Launch background DRAM→SRAM transfers
- **Latency hiding**: Overlap expensive memory operations with useful computation
- **Thread layout optimization**: Match memory access patterns to hardware
- **Pipeline programming**: Structure algorithms for maximum memory utilization

### **Key APIs you'll focus**

Building on the async copy operations introduced in [Puzzle 16's idiomatic matmul](../puzzle_16/tiled.md#solution-idiomatic-layouttensor-tiling), you'll now focus specifically on their memory optimization potential:

- **[`copy_dram_to_sram_async()`](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/)**: Launch background DRAM→SRAM transfers using dedicated copy engines
- **[`async_copy_wait_all()`](https://docs.modular.com/mojo/stdlib/gpu/memory/async_copy_wait_all)**: Synchronize transfer completion before accessing shared memory

**What's different from Puzzle 16?** While Puzzle 16 used async copy for clean tile loading in matmul, this puzzle focuses specifically on **latency hiding** - structuring algorithms to overlap expensive memory operations with useful computation work.

### **Performance impact**

These techniques can provide **significant speedups** for memory-bound algorithms by:

- **Hiding DRAM latency**: Convert idle waiting into productive computation time
- **Maximizing bandwidth**: Optimal memory access patterns prevent cache misses
- **Pipeline efficiency**: Keep compute units busy while memory transfers happen in parallel

> **What are async copy operations?** [Asynchronous copy operations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) allow GPU blocks to initiate memory transfers that execute in the background while the block continues with other work. This enables overlapping computation with memory movement, a fundamental optimization technique for memory-bound algorithms.

💡 **Success tip**: Think of this as **pipeline programming for GPU memory** - overlap stages, hide latencies, and maximize throughput. The goal is to keep your expensive compute units busy while data moves in the background.

## Understanding halo regions

Before diving into async copy operations, it's essential to understand **halo regions** (also called ghost cells or guard cells), which are fundamental to tile-based processing with stencil operations like convolution.

### What is a halo region?

A **halo region** consists of **extra elements** that extend beyond the boundaries of a processing tile to provide necessary neighboring data for stencil computations. When processing elements near tile edges, the stencil operation requires access to data from adjacent tiles.

### Why halo regions are necessary

Consider a 1D convolution with a 5-point kernel on a tile:

```
Original data:   [... | a b c d e f g h i j k l m n o | ...]
Processing tile:       [c d e f g h i j k l m n o]
                            ^                 ^
                     Need neighbors    Need neighbors
                     from left tile    from right tile

With halo:       [a b | c d e f g h i j k l m n o | p q]
                  ^^^                               ^^^
                  Left halo                      Right halo
```

**Key characteristics:**

- **Halo size**: Typically `KERNEL_SIZE // 2` elements on each side
- **Purpose**: Enable correct stencil computation at tile boundaries
- **Content**: Copies of data from neighboring tiles or boundary conditions
- **Memory overhead**: Small additional storage for significant computational benefit

### Halo region in convolution

For a 5-point convolution kernel \\([k_0, k_1, k_2, k_3, k_4]\\):

- **Center element**: \\(k_2\\) aligns with the current processing element
- **Left neighbors**: \\(k_0, k_1\\) require 2 elements to the left
- **Right neighbors**: \\(k_3, k_4\\) require 2 elements to the right
- **Halo size**: `HALO_SIZE = 5 // 2 = 2` elements on each side

**Without halo regions:**

- Tile boundary elements cannot perform full convolution
- Results in incorrect output or complex boundary handling logic
- Performance suffers from scattered memory access patterns

**With halo regions:**

- All tile elements can perform full convolution using local data
- Simplified, efficient computation with predictable memory access
- Better cache utilization and memory coalescing

This concept becomes particularly important when implementing async copy operations, as halo regions must be properly loaded and synchronized to ensure correct parallel computation across multiple tiles.

## Async copy overlap with 1D convolution

**Building on [Puzzle 13](../puzzle_13/puzzle_13.md):** This puzzle revisits the 1D convolution from Puzzle 13, but now optimizes it using async copy operations to hide memory latency behind computation. Instead of simple synchronous memory access, we'll use hardware acceleration to overlap expensive DRAM transfers with useful work.

### Configuration

- Vector size: `VECTOR_SIZE = 16384` (16K elements across multiple blocks)
- Tile size: `CONV_TILE_SIZE = 256` (processing tile size)
- Block configuration: `(256, 1)` threads per block
- Grid configuration: `(VECTOR_SIZE // CONV_TILE_SIZE, 1)` blocks per grid (64 blocks)
- Kernel size: `KERNEL_SIZE = 5` (simple 1D convolution, same as Puzzle 13)
- Data type: `DType.float32`
- Layout: `Layout.row_major(VECTOR_SIZE)` (1D row-major)

### The async copy opportunity

**Building on Puzzle 16:** You've already seen `copy_dram_to_sram_async` used for clean tile loading in matmul. Now we'll focus on its **latency hiding capabilities** - the key to high-performance memory-bound algorithms.

Traditional synchronous memory loading forces compute units to wait idle during transfers. Async copy operations enable overlapping transfers with useful work:

```mojo
# Synchronous approach - INEFFICIENT:
for i in range(CONV_TILE_SIZE):
    input_shared[i] = input[base_idx + i]  # Each load waits for DRAM
for i in range(KERNEL_SIZE):
    kernel_shared[i] = kernel[i]           # More waiting for DRAM
barrier()  # All threads wait before computation begins
# ↑ Total time = input_transfer_time + kernel_transfer_time

# Async copy approach - EFFICIENT:
copy_dram_to_sram_async[thread_layout](input_shared, input_tile)  # Launch background transfer
# While input transfers in background, load kernel synchronously
for i in range(KERNEL_SIZE):
    kernel_shared[i] = kernel[i]  # Overlaps with async input transfer
async_copy_wait_all()  # Wait only when both operations complete
# ↑ Total time = MAX(input_transfer_time, kernel_transfer_time)
```

**Why async copy works so well:**

- **Dedicated copy engines**: Modern GPUs have specialized hardware that bypasses registers and enables true compute-memory overlap (as explained in [Puzzle 16](../puzzle_16/tiled.md#solution-idiomatic-layouttensor-tiling))
- **Latency hiding**: Memory transfers happen while GPU threads execute other operations
- **Optimal coalescing**: Thread layouts ensure efficient DRAM access patterns
- **Resource utilization**: Compute units stay busy instead of waiting idle

### Code to complete

Implement 1D convolution that uses async copy operations to overlap memory transfers with computation, following patterns from Puzzle 16's matmul implementation.

**Mathematical operation:** Compute 1D convolution across large vector using async copy for efficiency:
\\[\\text{output}[i] = \\sum_{k=0}^{\\text{KERNEL_SIZE}-1} \\text{input}[i+k-\\text{HALO_SIZE}] \\times \\text{kernel}[k]\\]

**Async copy algorithm:**

1. **Async tile loading:** Launch background DRAM→SRAM transfer for input data
2. **Overlapped operations:** Load small kernel data while input transfers
3. **Synchronization:** Wait for transfers, then compute using shared memory

```mojo
{{#include ../../../problems/p28/p28.mojo:async_copy_overlap_convolution}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p28/p28.mojo" class="filename">View full file: problems/p28/p28.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding async copy mechanics**

Async copy operations initiate background transfers while your block continues executing other code.

**Key questions to explore:**

- What data needs to be transferred from DRAM to shared memory?
- Which operations can execute while the transfer happens in the background?
- How does the hardware coordinate multiple concurrent operations?

**Thread layout considerations:**

- Your block has `THREADS_PER_BLOCK_ASYNC = 256` threads
- The tile has `CONV_TILE_SIZE = 256` elements
- What layout pattern ensures optimal memory coalescing?

### 2. **Identifying overlap opportunities**

The goal is to hide memory latency behind useful computation.

**Analysis approach:**

- What operations must happen sequentially vs. in parallel?
- Which data transfers are large (expensive) vs. small (cheap)?
- How can you structure the algorithm to maximize parallel execution?

**Memory hierarchy considerations:**

- Large input tile: 256 elements × 4 bytes = 1KB transfer
- Small kernel: 5 elements × 4 bytes = 20 bytes
- Which transfer benefits most from async optimization?

### 3. **Synchronization strategy**

Proper synchronization ensures correctness without sacrificing performance.

**Timing analysis:**

- When does each operation actually need its data to be ready?
- What's the minimum synchronization required for correctness?
- How do you avoid unnecessary stalls while maintaining data dependencies?

**Race condition prevention:**

- What happens if computation starts before transfers complete?
- How do memory fences and barriers coordinate different memory operations?

</div>
</details>

**Test the async copy overlap:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p28
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p28
```

  </div>
  <div class="tab-content">

```bash
uv run poe p28
```

  </div>
</div>

### Solution

<details class="solution-details">
<summary><strong>Complete Solution with Detailed Explanation</strong></summary>

The async copy overlap solution demonstrates how to hide memory latency by overlapping expensive DRAM transfers with useful computation:

```mojo
{{#include ../../../solutions/p28/p28.mojo:async_copy_overlap_convolution_solution}}
```

#### **Phase-by-phase breakdown**

**Phase 1: Async Copy Launch**

```mojo
# Phase 1: Launch async copy for input tile
input_tile = input.tile[CONV_TILE_SIZE](block_idx.x)
comptime load_layout = Layout.row_major(THREADS_PER_BLOCK_ASYNC)
copy_dram_to_sram_async[thread_layout=load_layout](input_shared, input_tile)
```

- **Tile Creation**: `input.tile[CONV_TILE_SIZE](block_idx.x)` creates a 256-element view of the input array starting at `block_idx.x * 256`. The Mojo [`tile` method](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile) does **NOT** perform bounds checking or zero-padding. Accessing out-of-bounds indices results in undefined behavior. The implementation must ensure the tile size and offset remain within valid array bounds.

- **Thread Layout**: `Layout.row_major(THREADS_PER_BLOCK_ASYNC, 1)` creates a `256 x 1` layout that matches our block organization. This is **critical** - the layout must match the physical thread arrangement for optimal coalesced memory access. When layouts mismatch, threads may access non-contiguous memory addresses, breaking coalescing and severely degrading performance.

- **Async Copy Launch**: `copy_dram_to_sram_async` initiates a background transfer from DRAM to shared memory. The hardware copies 256 floats (1KB) while the block continues executing.

**Phase 2: Overlapped Operation**

```mojo
# Phase 2: Load kernel synchronously (small data)
if local_i < KERNEL_SIZE:
    kernel_shared[local_i] = kernel[local_i]
```

- **Simultaneous Execution**: While the 1KB input tile transfers in the background, threads load the small 20-byte kernel synchronously. This overlap is the key optimization.

- **Size-Based Strategy**: Large transfers (input tile) use async copy; small transfers (kernel) use synchronous loading. This balances complexity with performance benefit.

**Phase 3: Synchronization**

```mojo
# Phase 3: Wait for async copy to complete
async_copy_wait_all()  # Always wait since we always do async copy
barrier()  # Sync all threads
```

- **Transfer Completion**: `async_copy_wait_all()` blocks until all async transfers complete. This is essential before accessing `input_shared`.

- **Thread Synchronization**: `barrier()` ensures all threads see the completed transfer before proceeding to computation.

**Phase 4: Computation**

```mojo
# Phase 4: Compute convolution
global_i = block_idx.x * CONV_TILE_SIZE + local_i
if local_i < CONV_TILE_SIZE and global_i < output.shape[0]():
    var result: output.element_type = 0

    if local_i >= HALO_SIZE and local_i < CONV_TILE_SIZE - HALO_SIZE:
        # Full convolution for center elements
        for k in range(KERNEL_SIZE):
            input_idx = local_i + k - HALO_SIZE
            if input_idx >= 0 and input_idx < CONV_TILE_SIZE:
                result += input_shared[input_idx] * kernel_shared[k]
    else:
        # For boundary elements, just copy input (no convolution)
        result = input_shared[local_i]

    output[global_i] = result
```

- **Fast Shared Memory Access**: All computation uses pre-loaded shared memory data, avoiding slow DRAM access during the compute-intensive convolution loop.

- **Simplified Boundary Handling**: The implementation uses a pragmatic approach to handle elements near tile boundaries:
  - **Center elements** (`local_i >= HALO_SIZE` and `local_i < CONV_TILE_SIZE - HALO_SIZE`): Apply full 5-point convolution using shared memory data
  - **Boundary elements** (first 2 and last 2 elements in each tile): Copy input directly without convolution to avoid complex boundary logic

  **Educational rationale**: This approach prioritizes demonstrating async copy patterns over complex boundary handling. For a 256-element tile with `HALO_SIZE = 2`, elements 0-1 and 254-255 use input copying, while elements 2-253 use full convolution. This keeps the focus on memory optimization while providing a working implementation.

#### **Performance analysis**

**Without Async Copy (Synchronous):**

```
Total Time = Input_Transfer_Time + Kernel_Transfer_Time + Compute_Time
           = Large_DRAM_transfer + Small_DRAM_transfer + convolution
           = Major_latency + Minor_latency + computation_work
```

**With Async Copy (Overlapped):**

```
Total Time = MAX(Input_Transfer_Time, Kernel_Transfer_Time) + Compute_Time
           = MAX(Major_latency, Minor_latency) + computation_work
           = Major_latency + computation_work
```

**Speedup**: Performance improvement from hiding the smaller kernel transfer latency behind the larger input transfer. The actual speedup depends on the relative sizes of transfers and available memory bandwidth. In memory-bound scenarios with larger overlaps, speedups can be much more significant.

#### **Key technical insights**

1. **Thread Layout Matching**: The `Layout.row_major(256, 1)` layout precisely matches the block's `(256, 1)` thread organization, enabling optimal memory coalescing.

2. **Race Condition Avoidance**: Proper sequencing (async copy → kernel load → wait → barrier → compute) eliminates all race conditions that could corrupt shared memory.

3. **Hardware Optimization**: Modern GPUs have dedicated hardware for async copy operations, allowing true parallelism between memory and compute units.

4. **Memory Hierarchy Exploitation**: The pattern moves data through the hierarchy efficiently: DRAM → Shared Memory → Registers → Computation.

5. **Test-Implementation Consistency**: The test verification logic matches the boundary handling strategy by checking `local_i_in_tile = i % CONV_TILE_SIZE` to determine whether each element should expect convolution results (center elements) or input copying (boundary elements). This ensures accurate validation of the simplified boundary approach.

This solution transforms a naive memory-bound convolution into an optimized implementation that hides memory latency behind useful work, demonstrating fundamental principles of high-performance GPU programming.

</details>
