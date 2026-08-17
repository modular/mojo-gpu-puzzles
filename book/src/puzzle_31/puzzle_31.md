# Puzzle 31: GPU Occupancy Optimization

## Why this puzzle matters

**Building on Puzzle 30:** You've just learned GPU profiling tools and
discovered how memory access patterns can create dramatic performance
differences. Now you're ready for the next level: **resource optimization**.

**The Learning Journey:**

- **Puzzle 30** taught you to **diagnose** performance problems using Nsight
  profiling (`nsys` and `ncu`)
- **Puzzle 31** teaches you to **predict and control** performance through
  resource management
- **Together**, they give you the complete toolkit for GPU optimization

**What You'll Discover:** GPU performance isn't just about algorithmic
efficiency - it's about **how your code uses limited hardware resources**. Every
GPU has finite registers, shared memory, and execution units. Understanding
**occupancy** - _the ratio of active warps to maximum possible warps per SM_ -
is crucial for:

- **Latency hiding**: Keeping the GPU busy while waiting for memory
- **Resource allocation**: Balancing registers, shared memory, and thread blocks
- **Performance prediction**: Understanding bottlenecks before they happen
- **Optimization strategy**: Knowing when to focus on occupancy vs other factors

**Why This Matters Beyond GPUs:** The principles you learn here apply to any
parallel computing system where resources are shared among many execution
units—from CPUs with hyperthreading to distributed computing clusters.

## Overview

**GPU Occupancy** is the ratio of active warps to the maximum possible warps per
SM. It determines how well your GPU can hide memory latency through warp
switching.

**SAXPY** is a mnemonic for Single-precision Alpha times X plus Y. This puzzle
explores three SAXPY kernels (`y[i] = alpha * x[i] + y[i]`) with nearly
equivalent math but very different resource usage:

```mojo
{{#include ../../../problems/p31/p31.mojo:minimal_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p31/p31.mojo" class="filename">View full file: problems/p31/p31.mojo</a>

```mojo
{{#include ../../../problems/p31/p31.mojo:sophisticated_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p31/p31.mojo" class="filename">View full file: problems/p31/p31.mojo</a>

```mojo
{{#include ../../../problems/p31/p31.mojo:balanced_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p31/p31.mojo" class="filename">View full file: problems/p31/p31.mojo</a>

## Your task

Use profiling tools to investigate three kernels and answer analysis questions
about occupancy optimization. The kernels compute equivalent results (to within
test tolerance) but use resources very differently - your job is to discover
why performance and occupancy behave counterintuitively!

> The worked occupancy calculation in this puzzle is based on
> **NVIDIA A10G (Ampere 8.6)** hardware; the measured resource and timing
> figures come from a **B200 (Blackwell 10.0)**. Your results will vary
> depending on your GPU vendor and architecture (NVIDIA:
> Pascal/Turing/Ampere/Ada/Hopper/Blackwell, AMD: RDNA/GCN, Apple:
> M1/M2/M3/M4/M5), but the **fundamental concepts, methodology, and insights
> remain universally applicable** across modern GPUs. Use `pixi run gpu-specs`
> to get your specific hardware values.
>
> **Hardware is not the only variable — the build is too.** Register counts and
> timings differ between a `--debug-level=full` build and a release one, and
> occupancy follows the registers. Shared memory per block does not move. Any
> number below that you intend to compare against your own must say which build
> it came from; see
> [the debug-build trade-off](../puzzle_30/nvidia_profiling_basics.md#step-1-prepare-your-code-for-profiling).

## Configuration

**Requirements:**

- NVIDIA GPU with CUDA toolkit
- Nsight Compute from [Puzzle 30](../puzzle_30/puzzle_30.md)

> **⚠️ GPU compatibility note:** The default configuration uses aggressive
> settings that may fail on older or lower-capability GPUs:
>
> ```mojo
> comptime SIZE = 32 * 1024 * 1024  # 32M elements (~128MB per array)
> comptime THREADS_PER_BLOCK = (1024, 1)  # 1024 threads per block
> comptime BLOCKS_PER_GRID = (SIZE // 1024, 1)  # 32768 blocks
> ```
>
> **If you encounter launch failures, reduce these values in
> `problems/p31/p31.mojo`:**
>
> - **For older GPUs:** Use `THREADS_PER_BLOCK = (512, 1)` and `SIZE = 16 * 1024 * 1024`
> - **For limited memory GPUs (< 2GB):** Use `SIZE = 8 * 1024 * 1024` or `SIZE = 4 * 1024 * 1024`
> - **For grid dimension limits:** The `BLOCKS_PER_GRID` will automatically adjust with `SIZE`

**Occupancy Formula:**

```text
Theoretical Occupancy = min(
    Registers Per SM / (Registers Per Thread × Threads Per Block),
    Shared Memory Per SM / Shared Memory Per Block,
    Max Blocks Per SM
) × Threads Per Block / Max Threads Per SM
```

## The investigation

### Step 1: Test the kernels

```bash
pixi shell -e nvidia
mojo problems/p31/p31.mojo --all
```

All three compute the same SAXPY result to within test tolerance—the
sophisticated and balanced kernels add small correction terms, so the tests
compare with `rtol=1e-3` and `rtol=1e-4` rather than exactly. The mystery: why
do they have different performance?

### Step 2: Benchmark performance

```bash
mojo problems/p31/p31.mojo --benchmark
```

Record the reported time for each kernel. Step 4 is where you connect those
numbers to the resources each kernel consumes.

### Step 3: Build for profiling

```bash
mojo build problems/p31/p31.mojo -o problems/p31/p31_profiler
```

**No `--debug-level=full` here**, unlike the profiling walkthrough in
[Puzzle 30](../puzzle_30/nvidia_profiling_basics.md). This puzzle's subject
*is* resource usage, and debug metadata inflates register counts — enough, on a
B200, to drag the sophisticated kernel's measured occupancy from 82% down to
50% and invert the comparison you are about to make. Profile a release build
whenever the numbers you want are the resource numbers.

### Step 4: Profile resource usage

```bash
# Profile each kernel's resource usage
ncu --set basic --section=LaunchStats problems/p31/p31_profiler --minimal
ncu --set basic --section=LaunchStats problems/p31/p31_profiler --sophisticated
ncu --set basic --section=LaunchStats problems/p31/p31_profiler --balanced
```

Record the resource usage for occupancy analysis.

### Step 5: Calculate theoretical occupancy

First, identify your GPU architecture and detailed specs:

```bash
pixi run gpu-specs
```

**Note**: `gpu-specs` automatically detects your GPU vendor (NVIDIA/AMD/Apple)
and shows **all architectural details** derived from your hardware - no lookup
tables needed!

**Common Architecture Specs (Reference):**

| Architecture                      | Compute Cap | Registers/SM | Shared Mem/SM | Max Threads/SM | Max Blocks/SM |
|-----------------------------------|-------------|--------------|---------------|----------------|---------------|
| **Hopper (H100)**                 | 9.0         | 65,536       | 228KB         | 2,048          | 32            |
| **Ada (RTX 40xx)**                | 8.9         | 65,536       | 100KB         | 1,536          | 24            |
| **Ampere (RTX 30xx, A10G)**       | 8.6         | 65,536       | 100KB         | 1,536          | 16            |
| **Ampere (A100)**                 | 8.0         | 65,536       | 164KB         | 2,048          | 32            |
| **Turing (RTX 20xx)**             | 7.5         | 65,536       | 64KB          | 1,024          | 16            |
| **Pascal (GTX 10xx)**             | 6.1         | 65,536       | 96KB          | 2,048          | 32            |

Note that the two Ampere compute capabilities differ: the data-center 8.0 part
(A100) has 164KB of shared memory and 2,048 threads per SM, while the 8.6 parts
(RTX 30xx, A10G) have 100KB and 1,536. That distinction drives the worked
solution below.

**📚 Official Documentation:**

- [NVIDIA CUDA Compute Capability Table](https://developer.nvidia.com/cuda-gpus)
- [CUDA Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Ampere Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)

**⚠️ Note:** These are theoretical maximums. Actual occupancy may be lower due
to hardware scheduling constraints, driver overhead, and other factors.

Using your GPU specs and the occupancy formula:

- **Threads Per Block:** 1024 (from our kernel)

Use the occupancy formula and your hardware specifications to predict each
kernel's theoretical occupancy.

### Step 6: Measure actual occupancy

```bash
# Measure actual occupancy for each kernel
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --minimal
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --sophisticated
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --balanced
```

Compare the actual measured occupancy with your theoretical calculations - this
is where the mystery reveals itself!

## Key insights

💡 **Occupancy Threshold:** Once you have sufficient occupancy for latency
hiding (~25-50%), additional occupancy provides diminishing returns.

💡 **Memory Bound vs Compute Bound:** SAXPY is memory-bound. Memory bandwidth
often matters more than occupancy for memory-bound kernels.

💡 **Resource Efficiency:** Modern GPUs can handle moderate register pressure
(20-40 registers/thread) without dramatic occupancy loss.

## Your task: Answer the following questions

**After completing the investigation steps above, answer these analysis
questions to solve the occupancy mystery:**

**Performance Analysis (Step 2):**

1. Which kernel is fastest? Which is slowest? Record the timing differences.

**Resource Profiling (Step 4):**

2. Record for each kernel: Registers Per Thread, Shared Memory Per Block, Warps
   Per SM

**Theoretical Calculations (Step 5):**

3. Calculate theoretical occupancy for each kernel using your GPU specs and the
   occupancy formula. Which should be highest/lowest?

**Measured Occupancy (Step 6):**

4. How do the measured occupancy values compare to your calculations?

**The Occupancy Mystery:**

5. Why do all three kernels land in a similar occupancy band (the exact figures
   vary by GPU architecture) despite dramatically different resource usage?
6. Why does performance vary so little — about 14% between fastest and slowest
   on a B200 — when shared memory usage varies from 0KB to 49KB?
7. What does this reveal about the relationship between theoretical occupancy
   calculations and real-world GPU behavior?
8. For this SAXPY workload, what is the actual performance bottleneck if it's
   not occupancy?

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

**Your detective toolkit:**

- **Nsight Compute (`ncu`)** - Measure occupancy and resource usage
- **GPU architecture specs** - Calculate theoretical limits using
  `pixi run gpu-specs`
- **Occupancy formula** - Predict resource bottlenecks
- **Performance benchmarks** - Validate theoretical analysis

**Key optimization principles:**

- **Calculate before optimizing:** Use the occupancy formula to predict resource
  limits before writing code
- **Measure to validate:** Theoretical calculations don't account for compiler
  optimizations and hardware details
- **Consider workload characteristics:** Memory-bound workloads need less
  occupancy than compute-bound operations
- **Don't optimize for maximum occupancy:** Optimize for sufficient occupancy +
  other performance factors
- **Think in terms of thresholds:** 25-50% occupancy is often sufficient for
  latency hiding
- **Profile resource usage:** Use Nsight Compute to understand actual register
  and shared memory consumption

**Investigation approach:**

1. **Start with benchmarking** - See the performance differences first
2. **Profile with Nsight Compute** - Get actual resource usage and occupancy
   data
3. **Calculate theoretical occupancy** - Use your GPU specs and the occupancy
   formula
4. **Compare theory vs reality** - This is where the mystery reveals itself!
5. **Think about workload characteristics** - Why might theory not match
   practice?

</div>
</details>

## Solution

<details class="solution-details">
<summary><strong>Complete Solution with Enhanced Explanation</strong></summary>

This occupancy detective case demonstrates how resource usage affects GPU
performance and reveals the complex relationship between theoretical occupancy
and actual performance.

> The worked calculations below are for **NVIDIA A10G (Ampere 8.6)**. The
> measurements are from a **B200 (Blackwell 10.0)**, driver 595.71.05, MAX
> 26.5.0 / Mojo 1.0.0, release build. Your results will vary based on your GPU
> architecture, but the methodology and insights apply universally. Use
> `pixi run gpu-specs` to get your specific hardware values.

## **Profiling evidence from resource analysis**

**Nsight Compute Resource Analysis:**

**Actual Profiling Results (NVIDIA B200, release build - your results will vary
by GPU):**

- **Minimal:** 16 registers, 0KB shared → **74.66%** occupancy, **0.1399ms**
- **Balanced:** 16 registers, 16.38KB shared → **83.51%** occupancy,
  **0.1497ms**
- **Sophisticated:** 16 registers, 49.15KB shared → **81.95%** occupancy,
  **0.1589ms**

**Performance Evidence from Benchmarking:**

- **All kernels perform within about 14% of each other** despite one using no
  shared memory and another using 49KB
- **All land in a similar occupancy band** (75-84%) despite those resource
  differences
- **Memory bandwidth becomes the limiting factor** for all kernels

> **Registers are not the interesting variable here, and the build is why.**
> Profile the same three kernels with `--debug-level=full` and the register
> counts spread out — 16 minimal, 24 balanced, 40 sophisticated on this B200 —
> which drags sophisticated's occupancy down to 49.70%, below both others, and
> breaks the very pattern this puzzle teaches. The
> release build shows the compiler assigning all three the same 16 registers.
> Shared memory per block is identical in both builds, which is what makes it
> the honest resource contrast.

## **Occupancy calculations revealed**

**Theoretical Occupancy Analysis (NVIDIA A10G, Ampere 8.6):**

This is a worked example of the formula on one card, so you can see each limit
computed before running the arithmetic for your own. The per-kernel register
counts it feeds in (19 / 25 / 40) are from a debug build, which is why they
differ from the 16 / 16 / 16 measured above — watch what that changes in the
result, which is nothing: the register limit never becomes the binding one.

**GPU Specifications (from `pixi run gpu-specs`):**

- **Registers Per SM:** 65,536
- **Shared Memory Per SM:** 100KB (compute capability 8.6)
- **Max Threads Per SM:** 1,536 (hardware limit on A10G)
- **Threads Per Block:** 1,024 (our configuration)
- **Max Blocks Per SM:** 16

**Minimal Kernel Calculation:**

```text
Register Limit = 65,536 / (19 × 1,024) = 3.36 blocks per SM
Shared Memory Limit = 100KB / 0KB = ∞ blocks per SM
Hardware Block Limit = 16 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(3, ∞, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Balanced Kernel Calculation:**

```text
Register Limit = 65,536 / (25 × 1,024) = 2.56 blocks per SM
Shared Memory Limit = 100KB / 16.4KB = 6.10 blocks per SM
Hardware Block Limit = 16 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(2, 6, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Sophisticated Kernel Calculation:**

```text
Register Limit = 65,536 / (40 × 1,024) = 1.6 blocks per SM
Shared Memory Limit = 100KB / 49.2KB = 2.03 blocks per SM
Hardware Block Limit = 16 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(1, 2, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Key Discovery: One Limit Binds, and It Isn't the Interesting One**

- **Theoretical**: all three kernels 66.7% on the A10G, limited by its thread
  capacity — not by registers and not by shared memory
- **The 0KB and the 49KB kernel land on the same answer**, which is the whole
  point: a resource only matters when it is the one that runs out first

You can only fit 1 block of 1,024 threads per SM when the maximum is 1,536, so
the thread limit decides the outcome and the dramatic resource differences never
get to. Run the same arithmetic with your own card's specs before assuming a
resource is your problem.

## **Why measured occupancy falls short of theoretical**

Measured occupancy always lands somewhat under the theoretical figure:

1. **Hardware Scheduling Overhead**: Real warp schedulers have practical
   limitations beyond theoretical calculations
2. **CUDA Runtime Reservations**: Driver and runtime overhead reduce available
   SM resources slightly
3. **Memory Controller Pressure**: The memory subsystem creates slight
   scheduling constraints
4. **Power and Thermal Management**: Dynamic frequency scaling affects peak
   performance
5. **Instruction Cache Effects**: Real kernels have instruction fetch overhead
   not captured in occupancy calculations

**Key Insight**: the gap is small enough that the calculation is worth doing.
Identifying *which* limit binds is what the formula buys you — and here it is
thread capacity, for every kernel, regardless of their register and shared
memory differences.

## **The occupancy mystery explained**

**The Real Mystery Revealed:**

- **All kernels land in the same occupancy band** despite dramatic resource
  differences
- **Performance varies by about 14%** across all three, where shared memory
  usage varies by a factor of infinity (0KB) to 49KB
- **Theory correctly identifies the binding limit** - thread capacity, not
  registers or shared memory
- **The mystery isn't occupancy mismatch** - it's why similar occupancy and
  near-identical performance despite huge resource differences!

**Why Near-Identical Performance Despite Different Resource Usage:**

**SAXPY Workload Characteristics:**

- **Memory-bound operation:** Each thread does minimal computation
  (`y[i] = alpha * x[i] + y[i]`)
- **High memory traffic:** Reading 2 values, writing 1 value per thread
- **Low arithmetic intensity:** Only 2 FLOPS per 12 bytes of memory traffic

**Memory Bandwidth Analysis:**

```text
Single Kernel Pass Analysis:
- Input arrays: 32M × 4 bytes × 2 arrays = 256MB read
- Output array: 32M × 4 bytes × 1 array = 128MB write
- Total per kernel: 384MB memory traffic

Measured (B200, minimal kernel): 0.1399ms per iteration
Implied bandwidth: 384MB / 0.1399ms ≈ 2.7 TB/s
```

Divide your own card's peak bandwidth (`pixi run gpu-specs`) into that 384MB to
get the floor this kernel could ever reach, then compare it with what you
measured. All three kernels move the same 384MB, which is why none of them can
get far from the others.

**The Real Performance Factors:**

1. **Memory Bandwidth Utilization**: All kernels saturate available memory
   bandwidth
2. **Computational Overhead**: the sophisticated kernel does extra work — more
   instructions and more shared memory traffic, not more registers
3. **Shared Memory Without Reuse**: the balanced kernel writes each value into
   shared memory and reads it straight back, so the allocation buys no data
   reuse—only extra instructions
4. **Compiler Optimizations**: Modern compilers minimize register usage when
   possible

## **Understanding the occupancy threshold concept**

**Critical Insight: Occupancy is About "Sufficient" Not "Maximum"**

**Latency Hiding Requirements:**

- **Memory latency:** ~500-800 cycles on modern GPUs
- **Warp scheduling:** GPU needs enough warps to hide this latency
- **Sufficient threshold:** Usually 25-50% occupancy provides effective latency
  hiding

**Why Higher Occupancy Doesn't Always Help:**

**Resource Competition:**

- More active threads compete for same memory bandwidth
- Cache pressure increases with more concurrent accesses
- Register/shared memory pressure can hurt individual thread performance

**Workload-Specific Optimization:**

- **Compute-bound:** Higher occupancy helps hide ALU pipeline latency
- **Memory-bound:** Memory bandwidth limits performance regardless of occupancy
- **Mixed workloads:** Balance occupancy with other optimization factors

## **Real-world occupancy optimization principles**

**Systematic Occupancy Analysis Approach:**

**Phase 1: Calculate Theoretical Limits**

```bash
# Find your GPU specs
pixi run gpu-specs
```

**Phase 2: Profile Actual Usage**

```bash
# Measure resource consumption
ncu --set basic --section=LaunchStats your_kernel

# Measure achieved occupancy
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active your_kernel
```

**Phase 3: Performance Validation**

```bash
# Always validate with actual performance measurements
ncu --set roofline --section=MemoryWorkloadAnalysis your_kernel
```

**Evidence-to-Decision Framework:**

```text
OCCUPANCY ANALYSIS → OPTIMIZATION STRATEGY:

High occupancy (>70%) + Good performance:
→ Occupancy is sufficient, focus on other bottlenecks

Low occupancy (<30%) + Poor performance:
→ Increase occupancy through resource optimization

Good occupancy (50-70%) + Poor performance:
→ Look for memory bandwidth, cache, or computational bottlenecks

Low occupancy (<30%) + Good performance:
→ Workload doesn't need high occupancy (memory-bound)
```

## **Practical occupancy optimization techniques**

**Register Optimization:**

- **Use appropriate data types**: `float32` vs `float64`, `int32` vs `int64`
- **Minimize intermediate variables**: Let compiler optimize temporary storage
- **Loop unrolling consideration**: Balance occupancy vs instruction-level
  parallelism

**Shared Memory Optimization:**

- **Calculate required sizes**: Avoid over-allocation
- **Consider tiling strategies**: Balance occupancy vs data reuse
- **Bank conflict avoidance**: Design access patterns for conflict-free access

**Block Size Tuning:**

- **Test multiple configurations**: 256, 512, 1024 threads per block
- **Consider warp utilization**: Avoid partial warps when possible
- **Balance occupancy vs resource usage**: Larger blocks may hit resource limits

## **Key takeaways: From one occupancy mystery to universal principles**

This occupancy investigation reveals a clear progression of insights that apply
to all GPU optimization:

**The Discovery Chain:**

1. **Thread limits dominated everything** - Despite 0KB vs 49KB shared memory
   differences, all kernels hit the same 1-block-per-SM limit on the A10G, set
   by its 1,536-thread capacity
2. **Theory identified the binding limit** - the calculation is worth doing not
   because it predicts occupancy to the decimal, but because it tells you which
   resource is actually deciding
3. **Memory bandwidth ruled performance** - all three kernels move the same
   384MB, and SAXPY's memory-bound nature explains why performance stayed
   within about 14% despite the resource differences

**Universal GPU Optimization Principles:**

**Identify the Real Bottleneck:**

- Calculate occupancy limits from **all resources**: registers, shared memory,
  AND thread capacity
- The most restrictive limit wins - don't assume it's always registers or shared
  memory
- Memory-bound workloads (like SAXPY) are limited by bandwidth, not occupancy,
  once you have sufficient threads for latency hiding

**When Occupancy Matters vs When It Doesn't:**

- **High occupancy critical**: Compute-intensive kernels (GEMM, scientific
  simulations) that need latency hiding for ALU pipeline stalls
- **Occupancy less critical**: Memory-bound operations (BLAS Level 1, memory
  copies) where bandwidth saturation occurs before occupancy becomes limiting
- **Sweet spot**: 25-50% occupancy is often enough for latency hiding - beyond
  that, focus on the real bottleneck

**Practical Optimization Workflow:**

1. **Profile first** (`ncu --set basic`) - measure actual resource usage
   and occupancy
2. **Calculate theoretical limits** using your GPU's specs
   (`pixi run gpu-specs`)
3. **Identify the dominant constraint** - registers, shared memory, thread
   capacity, or memory bandwidth
4. **Optimize the bottleneck** - don't waste time on non-limiting resources
5. **Validate with end-to-end performance** - occupancy is a means to
   performance, not the goal

This case demonstrates why
**systematic bottleneck analysis beats intuition** - the sophisticated kernel's
49KB of shared memory was irrelevant because thread capacity dominated, and
similar occupancy plus memory bandwidth saturation explained the performance
mystery completely.

</details>
