# Puzzle 31: GPU Occupancy Optimization

## Why this puzzle matters

**Building on Puzzle 30:** You've just learned GPU profiling tools and discovered how memory access patterns can create dramatic performance differences. Now you're ready for the next level: **resource optimization**.

**The Learning Journey:**

- **Puzzle 30** taught you to **diagnose** performance problems using NSight profiling (`nsys` and `ncu`)
- **Puzzle 31** teaches you to **predict and control** performance through resource management
- **Together**, they give you the complete toolkit for GPU optimization

**What You'll Discover:**
GPU performance isn't just about algorithmic efficiency - it's about **how your code uses limited hardware resources**. Every GPU has finite registers, shared memory, and execution units. Understanding **occupancy** - _the ratio of active warps to maximum possible warps per SM_ - is crucial for:

- **Latency hiding**: Keeping the GPU busy while waiting for memory
- **Resource allocation**: Balancing registers, shared memory, and thread blocks
- **Performance prediction**: Understanding bottlenecks before they happen
- **Optimization strategy**: Knowing when to focus on occupancy vs other factors

**Why This Matters Beyond GPUs:**
The principles you learn here apply to any parallel computing system where resources are shared among many execution units - from CPUs with hyperthreading to distributed computing clusters.

## Overview

**GPU Occupancy** is the ratio of active warps to the maximum possible warps per SM. It determines how well your GPU can hide memory latency through warp switching.

**SAXPY** is a mnemonic for Single-precision Alpha times X plus Y. This puzzle explores three SAXPY kernels (`y[i] = alpha * x[i] + y[i]`) with identical math but different resource usage:

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

Use profiling tools to investigate three kernels and answer analysis questions about occupancy optimization. The kernels compute identical results but use resources very differently - your job is to discover why performance and occupancy behave counterintuitively!

> The specific numerical results shown in this puzzle are based on **NVIDIA A10G (Ampere 8.6)** hardware. Your results will vary depending on your GPU vendor and architecture (NVIDIA: Pascal/Turing/Ampere/Ada/Hopper, AMD: RDNA/GCN, Apple: M1/M2/M3/M4), but the **fundamental concepts, methodology, and insights remain universally applicable** across modern GPUs. Use `pixi run gpu-specs` to get your specific hardware values.

## Configuration

**Requirements:**

- NVIDIA GPU with CUDA toolkit
- NSight Compute from [Puzzle 30](../puzzle_30/puzzle_30.md)

> **⚠️ GPU compatibility note:**
> The default configuration uses aggressive settings that may fail on older or lower-capability GPUs:
>
> ```mojo
> comptime SIZE = 32 * 1024 * 1024  # 32M elements (~256MB memory per array)
> comptime THREADS_PER_BLOCK = (1024, 1)  # 1024 threads per block
> comptime BLOCKS_PER_GRID = (SIZE // 1024, 1)  # 32768 blocks
> ```
>
> **If you encounter launch failures, reduce these values in `problems/p31/p31.mojo`:**
>
> - **For older GPUs (Compute Capability < 3.0):** Use `THREADS_PER_BLOCK = (512, 1)` and `SIZE = 16 * 1024 * 1024`
> - **For limited memory GPUs (< 2GB):** Use `SIZE = 8 * 1024 * 1024` or `SIZE = 4 * 1024 * 1024`
> - **For grid dimension limits:** The `BLOCKS_PER_GRID` will automatically adjust with `SIZE`

**Occupancy Formula:**

```
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

All three should produce identical results. The mystery: why do they have different performance?

### Step 2: Benchmark performance

```bash
mojo problems/p31/p31.mojo --benchmark
```

All three should produce identical results. The mystery: why do they have different performance?

### Step 3: Build for profiling

```bash
mojo build --debug-level=full problems/p31/p31.mojo -o problems/p31/p31_profiler
```

### Step 4: Profile resource usage

```bash
# Profile each kernel's resource usage
ncu --set=@occupancy --section=LaunchStats problems/p31/p31_profiler --minimal
ncu --set=@occupancy --section=LaunchStats problems/p31/p31_profiler --sophisticated
ncu --set=@occupancy --section=LaunchStats problems/p31/p31_profiler --balanced
```

Record the resource usage for occupancy analysis.

### Step 5: Calculate theoretical occupancy

First, identify your GPU architecture and detailed specs:

```bash
pixi run gpu-specs
```

**Note**: `gpu-specs` automatically detects your GPU vendor (NVIDIA/AMD/Apple) and shows **all architectural details** derived from your hardware - no lookup tables needed!

**Common Architecture Specs (Reference):**

| Architecture | Compute Cap | Registers/SM | Shared Mem/SM | Max Threads/SM | Max Blocks/SM |
|--------------|-------------|--------------|---------------|----------------|---------------|
| **Hopper (H100)** | 9.0 | 65,536 | 228KB | 2,048 | 32 |
| **Ada (RTX 40xx)** | 8.9 | 65,536 | 128KB | 2,048 | 32 |
| **Ampere (RTX 30xx, A100, A10G)** | 8.0, 8.6 | 65,536 | 164KB | 2,048 | 32 |
| **Turing (RTX 20xx)** | 7.5 | 65,536 | 96KB | 1,024 | 16 |
| **Pascal (GTX 10xx)** | 6.1 | 65,536 | 96KB | 2,048 | 32 |

**📚 Official Documentation:**

- [NVIDIA CUDA Compute Capability Table](https://developer.nvidia.com/cuda-gpus)
- [CUDA Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Ampere Architecture Whitepaper](https://developer.nvidia.com/ampere-architecture)

**⚠️ Note:** These are theoretical maximums. Actual occupancy may be lower due to hardware scheduling constraints, driver overhead, and other factors.

Using your GPU specs and the occupancy formula:

- **Threads Per Block:** 1024 (from our kernel)

Use the occupancy formula and your hardware specifications to predict each kernel's theoretical occupancy.

### Step 6: Measure actual occupancy

```bash
# Measure actual occupancy for each kernel
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --minimal
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --sophisticated
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active problems/p31/p31_profiler --balanced
```

Compare the actual measured occupancy with your theoretical calculations - this is where the mystery reveals itself!

## Key insights

💡 **Occupancy Threshold:** Once you have sufficient occupancy for latency hiding (~25-50%), additional occupancy provides diminishing returns.

💡 **Memory Bound vs Compute Bound:** SAXPY is memory-bound. Memory bandwidth often matters more than occupancy for memory-bound kernels.

💡 **Resource Efficiency:** Modern GPUs can handle moderate register pressure (20-40 registers/thread) without dramatic occupancy loss.

## Your task: Answer the following questions

**After completing the investigation steps above, answer these analysis questions to solve the occupancy mystery:**

**Performance Analysis (Step 2):**

1. Which kernel is fastest? Which is slowest? Record the timing differences.

**Resource Profiling (Step 4):**

2. Record for each kernel: Registers Per Thread, Shared Memory Per Block, Warps Per SM

**Theoretical Calculations (Step 5):**

3. Calculate theoretical occupancy for each kernel using your GPU specs and the occupancy formula. Which should be highest/lowest?

**Measured Occupancy (Step 6):**

4. How do the measured occupancy values compare to your calculations?

**The Occupancy Mystery:**

5. Why do all three kernels achieve similar occupancy (~64-66% results may vary depending on gpu architecture) despite dramatically different resource usage?
6. Why is performance nearly identical (<2% difference) when resource usage varies so dramatically (19 vs 40 registers, 0KB vs 49KB shared memory)?
7. What does this reveal about the relationship between theoretical occupancy calculations and real-world GPU behavior?
8. For this SAXPY workload, what is the actual performance bottleneck if it's not occupancy?

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

**Your detective toolkit:**

- **NSight Compute (`ncu`)** - Measure occupancy and resource usage
- **GPU architecture specs** - Calculate theoretical limits using `pixi run gpu-specs`
- **Occupancy formula** - Predict resource bottlenecks
- **Performance benchmarks** - Validate theoretical analysis

**Key optimization principles:**

- **Calculate before optimizing:** Use the occupancy formula to predict resource limits before writing code
- **Measure to validate:** Theoretical calculations don't account for compiler optimizations and hardware details
- **Consider workload characteristics:** Memory-bound workloads need less occupancy than compute-bound operations
- **Don't optimize for maximum occupancy:** Optimize for sufficient occupancy + other performance factors
- **Think in terms of thresholds:** 25-50% occupancy is often sufficient for latency hiding
- **Profile resource usage:** Use NSight Compute to understand actual register and shared memory consumption

**Investigation approach:**

1. **Start with benchmarking** - See the performance differences first
2. **Profile with NSight Compute** - Get actual resource usage and occupancy data
3. **Calculate theoretical occupancy** - Use your GPU specs and the occupancy formula
4. **Compare theory vs reality** - This is where the mystery reveals itself!
5. **Think about workload characteristics** - Why might theory not match practice?

</div>
</details>

## Solution

<details class="solution-details">
<summary><strong>Complete Solution with Enhanced Explanation</strong></summary>

This occupancy detective case demonstrates how resource usage affects GPU performance and reveals the complex relationship between theoretical occupancy and actual performance.

> The specific calculations below are for **NVIDIA A10G (Ampere 8.6)** - the GPU used for testing. Your results will vary based on your GPU architecture, but the methodology and insights apply universally. Use `pixi run gpu-specs` to get your specific hardware values.

## **Profiling evidence from resource analysis**

**NSight Compute Resource Analysis:**

**Actual Profiling Results (NVIDIA A10G - your results will vary by GPU):**

- **Minimal:** 19 registers, ~0KB shared → **63.87%** occupancy, **327.7ms**
- **Balanced:** 25 registers, 16.4KB shared → **65.44%** occupancy, **329.4ms**
- **Sophisticated:** 40 registers, 49.2KB shared → **65.61%** occupancy, **330.9ms**

**Performance Evidence from Benchmarking:**

- **All kernels perform nearly identically** (~327-331ms, <2% difference)
- **All achieve similar occupancy** (~64-66%) despite huge resource differences
- **Memory bandwidth becomes the limiting factor** for all kernels

## **Occupancy calculations revealed**

**Theoretical Occupancy Analysis (NVIDIA A10G, Ampere 8.6):**

**GPU Specifications (from `pixi run gpu-specs`):**

- **Registers Per SM:** 65,536
- **Shared Memory Per SM:** 164KB (architectural maximum)
- **Max Threads Per SM:** 1,536 (hardware limit on A10G)
- **Threads Per Block:** 1,024 (our configuration)
- **Max Blocks Per SM:** 32

**Minimal Kernel Calculation:**

```
Register Limit = 65,536 / (19 × 1,024) = 3.36 blocks per SM
Shared Memory Limit = 164KB / 0KB = ∞ blocks per SM
Hardware Block Limit = 32 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(3, ∞, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Balanced Kernel Calculation:**

```
Register Limit = 65,536 / (25 × 1,024) = 2.56 blocks per SM
Shared Memory Limit = 164KB / 16.4KB = 10 blocks per SM
Hardware Block Limit = 32 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(2, 10, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Sophisticated Kernel Calculation:**

```
Register Limit = 65,536 / (40 × 1,024) = 1.64 blocks per SM
Shared Memory Limit = 164KB / 49.2KB = 3.33 blocks per SM
Hardware Block Limit = 32 blocks per SM

Thread Limit = 1,536 / 1,024 = 1 block per SM (floor)
Actual Blocks = min(1, 3, 1) = 1 block per SM
Theoretical Occupancy = (1 × 1,024) / 1,536 = 66.7%
```

**Key Discovery: Theory Matches Reality!**

- **Theoretical**: All kernels ~66.7% (limited by A10G's thread capacity)
- **Actual Measured**: All ~64-66% (very close match!)

This reveals that **A10G's thread limit dominates** - you can only fit 1 block of 1,024 threads per SM when the maximum is 1,536 threads. The small difference (66.7% theoretical vs ~65% actual) comes from hardware scheduling overhead and driver limitations.

## **Why theory closely matches reality**

**Why the small gap between theoretical (66.7%) and actual (~65%) occupancy:**

1. **Hardware Scheduling Overhead**: Real warp schedulers have practical limitations beyond theoretical calculations
2. **CUDA Runtime Reservations**: Driver and runtime overhead reduce available SM resources slightly
3. **Memory Controller Pressure**: A10G's memory subsystem creates slight scheduling constraints
4. **Power and Thermal Management**: Dynamic frequency scaling affects peak performance
5. **Instruction Cache Effects**: Real kernels have instruction fetch overhead not captured in occupancy calculations

**Key Insight**: The close match (66.7% theoretical vs ~65% actual) shows that **A10G's thread limit truly dominates** all three kernels, regardless of their register and shared memory differences. This is an excellent example of identifying the real bottleneck!

## **The occupancy mystery explained**

**The Real Mystery Revealed:**

- **All kernels achieve nearly identical occupancy** (~64-66%) despite dramatic resource differences
- **Performance is essentially identical** (<2% variation) across all kernels
- **Theory correctly predicts occupancy** (66.7% theoretical ≈ 65% actual)
- **The mystery isn't occupancy mismatch** - it's why identical occupancy and performance despite huge resource differences!

**Why Identical Performance Despite Different Resource Usage:**

**SAXPY Workload Characteristics:**

- **Memory-bound operation:** Each thread does minimal computation (`y[i] = alpha * x[i] + y[i]`)
- **High memory traffic:** Reading 2 values, writing 1 value per thread
- **Low arithmetic intensity:** Only 2 FLOPS per 12 bytes of memory traffic

**Memory Bandwidth Analysis (A10G):**

```
Single Kernel Pass Analysis:
- Input arrays: 32M × 4 bytes × 2 arrays = 256MB read
- Output array: 32M × 4 bytes × 1 array = 128MB write
- Total per kernel: 384MB memory traffic

Peak Bandwidth (A10G): 600 GB/s
Single-pass time: 384MB / 600 GB/s ≈ 0.64ms theoretical minimum
Benchmark time: ~328ms (includes multiple iterations + overhead)
```

**The Real Performance Factors:**

1. **Memory Bandwidth Utilization**: All kernels saturate available memory bandwidth
2. **Computational Overhead**: Sophisticated kernel does extra work (register pressure effects)
3. **Shared Memory Benefits**: Balanced kernel gets some caching advantages
4. **Compiler Optimizations**: Modern compilers minimize register usage when possible

## **Understanding the occupancy threshold concept**

**Critical Insight: Occupancy is About "Sufficient" Not "Maximum"**

**Latency Hiding Requirements:**

- **Memory latency:** ~500-800 cycles on modern GPUs
- **Warp scheduling:** GPU needs enough warps to hide this latency
- **Sufficient threshold:** Usually 25-50% occupancy provides effective latency hiding

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
ncu --set=@occupancy --section=LaunchStats your_kernel

# Measure achieved occupancy
ncu --metrics=smsp__warps_active.avg.pct_of_peak_sustained_active your_kernel
```

**Phase 3: Performance Validation**

```bash
# Always validate with actual performance measurements
ncu --set=@roofline --section=MemoryWorkloadAnalysis your_kernel
```

**Evidence-to-Decision Framework:**

```
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
- **Loop unrolling consideration**: Balance occupancy vs instruction-level parallelism

**Shared Memory Optimization:**

- **Calculate required sizes**: Avoid over-allocation
- **Consider tiling strategies**: Balance occupancy vs data reuse
- **Bank conflict avoidance**: Design access patterns for conflict-free access

**Block Size Tuning:**

- **Test multiple configurations**: 256, 512, 1024 threads per block
- **Consider warp utilization**: Avoid partial warps when possible
- **Balance occupancy vs resource usage**: Larger blocks may hit resource limits

## **Key takeaways: From A10G mystery to universal principles**

This A10G occupancy investigation reveals a clear progression of insights that apply to all GPU optimization:

**The A10G Discovery Chain:**

1. **Thread limits dominated everything** - Despite 19 vs 40 registers and 0KB vs 49KB shared memory differences, all kernels hit the same 1-block-per-SM limit due to A10G's 1,536-thread capacity
2. **Theory matched reality closely** - 66.7% theoretical vs ~65% measured occupancy shows our calculations work when we identify the right bottleneck
3. **Memory bandwidth ruled performance** - With identical 66.7% occupancy, SAXPY's memory-bound nature (600 GB/s saturated) explained identical performance despite resource differences

**Universal GPU Optimization Principles:**

**Identify the Real Bottleneck:**

- Calculate occupancy limits from **all resources**: registers, shared memory, AND thread capacity
- The most restrictive limit wins - don't assume it's always registers or shared memory
- Memory-bound workloads (like SAXPY) are limited by bandwidth, not occupancy, once you have sufficient threads for latency hiding

**When Occupancy Matters vs When It Doesn't:**

- **High occupancy critical**: Compute-intensive kernels (GEMM, scientific simulations) that need latency hiding for ALU pipeline stalls
- **Occupancy less critical**: Memory-bound operations (BLAS Level 1, memory copies) where bandwidth saturation occurs before occupancy becomes limiting
- **Sweet spot**: 60-70% occupancy often sufficient for latency hiding - beyond that, focus on the real bottleneck

**Practical Optimization Workflow:**

1. **Profile first** (`ncu --set=@occupancy`) - measure actual resource usage and occupancy
2. **Calculate theoretical limits** using your GPU's specs (`pixi run gpu-specs`)
3. **Identify the dominant constraint** - registers, shared memory, thread capacity, or memory bandwidth
4. **Optimize the bottleneck** - don't waste time on non-limiting resources
5. **Validate with end-to-end performance** - occupancy is a means to performance, not the goal

The A10G case perfectly demonstrates why **systematic bottleneck analysis beats intuition** - the sophisticated kernel's high register pressure was irrelevant because thread capacity dominated, and identical occupancy plus memory bandwidth saturation explained the performance mystery completely.

</details>
