# 🕵 The Cache Hit Paradox

## Overview

Welcome to your first **profiling detective case**! You have three GPU kernels
that all compute the same simple vector addition: `output[i] = a[i] + b[i]`.
They should all perform identically, right?

**Wrong!** These kernels have dramatically different performance - one is
**orders of magnitude slower** than the others. Your mission: use the
[profiling tools](./nvidia_profiling_basics.md) you just learned to discover
**why**.

## The challenge

Welcome to a **performance mystery** that will challenge everything you think
you know about GPU optimization! You're confronted with three seemingly
identical vector addition kernels that compute the exact same mathematical
operation:

```text
output[i] = a[i] + b[i]  // Simple arithmetic - what could go wrong?
```

**The shocking reality:**

- **All three kernels produce identical, correct results**
- **One kernel runs thousands of times slower than the others**
- **The slowest kernel has the highest cache hit rates** (counterintuitive!)
- **Standard performance intuition completely fails**

**Your detective mission:**

1. **Identify the performance culprit** - Which kernel is catastrophically slow?
2. **Uncover the cache paradox** - Why do high cache hits indicate poor
   performance?
3. **Account for the work** - What makes identical operations behave so
   differently?
4. **Learn profiling methodology** - Use Nsight tools to gather evidence, not
   guesses

**Why this matters:** This puzzle reveals a fundamental GPU performance
principle that challenges CPU-based intuition. The skills you develop here apply
to real-world GPU optimization, where a metric that looks healthy can be hiding
the actual cost.

**The twist:** We approach this **without looking at the source code first** -
using only profiling tools as your guide, just like debugging production
performance issues. After we obtained the profiling results, we look at the code
for further analysis.

## Your detective toolkit

From the profiling tutorial, you have:

- **Nsight Systems (`nsys`)** - Find which kernels are slow
- **Nsight Compute (`ncu`)** - Analyze why kernels are slow
- **Memory efficiency metrics** - Detect poor access patterns

## Getting started

### Step 1: Run the benchmark

```bash
pixi shell -e nvidia
mojo problems/p30/p30.mojo --benchmark
```

You'll see dramatic timing differences between kernels! One kernel is
**much slower** than the others. Your job is to figure out why using profiling
tools **without** looking at the code.

**Example output** (B200, MAX 26.5.0 / Mojo 1.0.0 — your absolute times will
differ, the ratio between them is the point):

```text
| name    | met (ms)  | iters |
| ------- | --------- | ----- |
| kernel1 | 0.0735    | 100   |
| kernel2 | 371.16    | 100   |  <- This one is much slower!
| kernel3 | 0.0809    | 100   |
```

### Step 2: Prepare your code for profiling

**Critical**: For source-level attribution, build with full debug information:

```bash
mojo build --debug-level=full problems/p30/p30.mojo -o problems/p30/p30_profiler
```

**Why this matters**:

- **Full debug info**: Provides complete symbol tables, variable names, and
  source line mapping for profilers
- **Comprehensive analysis**: Enables Nsight tools to correlate performance data
  with specific code locations

The debug metadata costs real time — see
[the debug-build trade-off](./nvidia_profiling_basics.md#step-1-prepare-your-code-for-profiling).
It does not distort this investigation, because the effect you are hunting is a
several-thousand-fold gap between kernels built the same way.

### Step 3: System-wide investigation (Nsight Systems)

Profile each kernel to see the big picture:

```bash
# Profile each kernel individually using the optimized build (with warmup to avoid cold start effects)
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel1_profile ./problems/p30/p30_profiler --kernel1
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel2_profile ./problems/p30/p30_profiler --kernel2
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel3_profile ./problems/p30/p30_profiler --kernel3

# Analyze the results
nsys stats --force-export=true ./problems/p30/kernel1_profile.nsys-rep > ./problems/p30/kernel1_profile.txt
nsys stats --force-export=true ./problems/p30/kernel2_profile.nsys-rep > ./problems/p30/kernel2_profile.txt
nsys stats --force-export=true ./problems/p30/kernel3_profile.nsys-rep > ./problems/p30/kernel3_profile.txt
```

**Look for:**

- **GPU Kernel Summary** - Which kernels take longest?
- **Kernel execution times** - How much do they vary?
- **Memory transfer patterns** - Are they similar across implementations?

### Step 4: Kernel deep-dive (Nsight Compute)

Once you identify the slow kernel, analyze it with Nsight Compute:

```bash
# Deep-dive into memory patterns for each kernel using the optimized build
ncu --set roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel1_analysis ./problems/p30/p30_profiler --kernel1
ncu --set roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel2_analysis ./problems/p30/p30_profiler --kernel2
ncu --set roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel3_analysis ./problems/p30/p30_profiler --kernel3

# View the results
ncu --import ./problems/p30/kernel1_analysis.ncu-rep --page details
ncu --import ./problems/p30/kernel2_analysis.ncu-rep --page details
ncu --import ./problems/p30/kernel3_analysis.ncu-rep --page details
```

**When you run these commands, you'll see output like this** (B200; the
absolute figures depend on your GPU, the gap between them does not):

```text
Kernel1: Memory Throughput: ~2.06 Tbyte/s, Max Bandwidth: ~31%
Kernel2: Memory Throughput: ~8.7 Gbyte/s,  Max Bandwidth: ~27%
Kernel3: Memory Throughput: ~1.78 Tbyte/s, Max Bandwidth: ~27%
```

**Key metrics to investigate:**

- **Memory Throughput (Gbyte/s)** - Actual memory bandwidth achieved
- **Max Bandwidth (%)** - Percentage of theoretical peak bandwidth utilized
- **L1/TEX Hit Rate (%)** - L1 cache efficiency
- **L2 Hit Rate (%)** - L2 cache efficiency

**🤔 The Counterintuitive Result**: You'll notice Kernel2 has the **highest**
cache hit rates but the **lowest** performance! This is the key mystery to
solve.

### Step 5: Detective questions

Use your profiling evidence to answer these questions by looking at the kernel
code
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p30/p30.mojo" class="filename">problems/p30/p30.mojo</a>:

#### Performance analysis

1. **Which kernel achieves the highest Memory Throughput?** (Look at the
   Gbyte/s values)
2. **Which kernel achieves the lowest?** And how far apart are the three Max
   Bandwidth percentages by comparison — does that metric separate them at all?
3. **What's the performance gap in memory throughput?** (Factor difference
   between fastest and slowest)

#### The cache paradox

4. **Which kernel has the highest L1/TEX Hit Rate?**
5. **Which kernel has the highest L2 Hit Rate?**
6. **🤯 Why does the kernel with the BEST cache hit rates perform the WORST?**

#### Redundant work detective work

7. **Can high cache hit rates actually indicate a performance problem?**
8. **Count the element operations each kernel performs. How does kernel2's
   loop bound compare to the number of threads in the grid?**
9. **If every kernel's warps are equally coalesced, what is left to explain the
   throughput gap?**

#### The "Aha!" Moment

10. **Based on the profiling evidence, what fundamental GPU memory principle
    does this demonstrate?**

**Key insight to discover**: Sometimes **high cache hit rates are a red flag**,
not a performance victory!

## Solution

The mystery reveals a fundamental GPU performance principle:
**for memory-bound operations, the total volume of work dominates performance**,
even when every kernel computes the same result from the same coalesced
accesses.

**The profiling evidence reveals:**

1. **Performance hierarchy**: Kernel1 and Kernel3 are fast, Kernel2 is
   catastrophically slow (orders of magnitude difference)
2. **Memory throughput tells the story**: Fast kernels achieve high bandwidth
   utilization, slow kernel achieves minimal utilization
3. **The cache paradox**: The slowest kernel has the **highest** cache hit
   rates—revealing that high cache hits can indicate **wasted** work
4. **How much work a kernel does matters more than how well it caches** for
   memory-bound GPU workloads

<details class="solution-details">
<summary><strong>Complete Solution with Enhanced Explanation</strong></summary>

This profiling detective case demonstrates how the amount of work a kernel
issues creates orders-of-magnitude performance differences, even when kernels
perform identical mathematical operations.

## **Performance evidence from profiling**

**Nsight Systems Timeline Analysis:**

- **Kernel 1**: Short execution time - **EFFICIENT**
- **Kernel 3**: Similar to Kernel 1 - **EFFICIENT**
- **Kernel 2**: Dramatically longer execution time - **INEFFICIENT**

**Nsight Compute Memory Analysis (Hardware-Agnostic Patterns):**

- **Efficient kernels (1 & 3)**: High memory throughput, good bandwidth
  utilization, moderate cache hit rates
- **Inefficient kernel (2)**: Very low memory throughput, poor bandwidth
  utilization, **extremely high cache hit rates**

## **The cache paradox revealed**

**🤯 The Counterintuitive Discovery:**

- **Kernel2 has the HIGHEST cache hit rates** but **WORST performance**
- **This challenges conventional wisdom**: "High cache hits = good performance"
- **The truth**: High cache hit rates can be a
  **symptom of re-reading data the grid has already fetched**

**Why the Cache Paradox Occurs:**

**Traditional CPU intuition (INCORRECT for GPUs):**

- Higher cache hit rates always mean better performance
- Cache hits reduce memory traffic, improving efficiency

**GPU memory reality (CORRECT understanding):**

- **Delivered bandwidth matters more than hit rate** for memory-bound workloads
- **Redundant work** can inflate cache hit rates while starving throughput
- **Memory bandwidth utilization** is the real performance indicator

## **Root cause analysis - redundant work**

**Actual Kernel Implementations from p30.mojo:**

**Kernel 1 - Efficient Coalesced Access:**

```mojo
{{#include ../../../problems/p30/p30.mojo:kernel1}}
```

*Standard thread indexing - adjacent threads access adjacent memory*

**Kernel 2 - Redundant Grid-Stride Loop:**

```mojo
{{#include ../../../problems/p30/p30.mojo:kernel2}}
```

*Grid-stride loop with `stride = 512`. The stride separates successive
iterations of a single thread, not the threads within a warp—so each warp
access is still contiguous. The cost is redundant work: the grid is already
`SIZE` threads wide, one per element, so having every thread additionally walk
the whole buffer performs 16,384x the element operations kernel1 does. The last
element is written by 32,768 separate threads.*

**Kernel 3 - Efficient Reverse Access:**

```mojo
{{#include ../../../problems/p30/p30.mojo:kernel3}}
```

*Reverse indexing but still predictable - adjacent threads access adjacent
addresses (just backwards)*

**Pattern Analysis:**

- **Kernel 1**: Classic coalesced access - adjacent threads access adjacent
  memory
- **Kernel 2**: Coalesced within each warp access, but the grid-stride loop
  makes the whole grid re-traverse the buffer thousands of times
- **Kernel 3**: Reverse but still coalesced within warps - predictable pattern

## **Understanding the memory system**

**GPU Memory Architecture Fundamentals:**

- **Warp execution**: 32 threads execute together
- **Cache line size**: 128 bytes (32 float32 values)
- **Coalescing requirement**: Adjacent threads should access adjacent memory

**p30.mojo Configuration Details:**

```mojo
comptime SIZE = 16 * 1024 * 1024          # 16M elements (64MB of float32 data)
comptime THREADS_PER_BLOCK = (1024, 1)    # 1024 threads per block
comptime BLOCKS_PER_GRID = (SIZE // 1024, 1)  # 16,384 blocks total
comptime dtype = DType.float32             # 4 bytes per element
```

**Why these settings matter:**

- **Large dataset (16M)**: Makes memory access patterns clearly visible
- **1024 threads/block**: Maximum CUDA threads per block
- **32 warps/block**: Each block contains 32 warps of 32 threads each

**Memory Access Efficiency Visualization:**

```text
KERNEL 1 (Coalesced):           KERNEL 2 (Grid-stride loop, stride 512):
Warp threads 0-31:             Warp threads 0-31, iteration 0:
  Thread 0: Memory[0]            Thread 0: Memory[0]
  Thread 1: Memory[1]            Thread 1: Memory[1]
  Thread 2: Memory[2]            Thread 2: Memory[2]
  ...                           ...
  Thread 31: Memory[31]          Thread 31: Memory[31]

                               Thread 0 across iterations:
                                 iter 0: Memory[0]
                                 iter 1: Memory[512]
                                 iter 2: Memory[1024]

Result: 1 cache line fetch       Result: 1 cache line fetch, ×16,384 passes
Status: ~2.06 TB/s throughput    Status: ~8.7 GB/s throughput
Cache: Efficient utilization     Cache: Same lines hit repeatedly!
```

**KERNEL 3 (Reverse but Coalesced):**

```text
Warp threads 0-31 (first iteration):
  Thread 0: Memory[SIZE-1]     (reverse_i = SIZE-1-0)
  Thread 1: Memory[SIZE-2]     (reverse_i = SIZE-1-1)
  Thread 2: Memory[SIZE-3]     (reverse_i = SIZE-1-2)
  ...
  Thread 31: Memory[SIZE-32]   (reverse_i = SIZE-1-31)

Result: Adjacent addresses (just backwards)
Status: ~1.78 TB/s throughput (same order as Kernel 1)
Cache: Efficient utilization despite reverse order
```

## **The cache paradox explained**

**Why Kernel2 (stride=512) has high cache hit rates but poor performance:**

**The stride=512 disaster explained:**

```mojo
# Each thread processes multiple elements with huge gaps:
Thread 0: elements [0, 512, 1024, 1536, 2048, ...]
Thread 1: elements [1, 513, 1025, 1537, 2049, ...]
Thread 2: elements [2, 514, 1026, 1538, 2050, ...]
...
```

**Why this creates the cache paradox:**

1. **Cache line reuse across the grid**: a 512-element jump on `float32` is
   2048 bytes—16 cache lines further on—but every one of those lines has
   already been pulled in by another thread, since the grid spans the whole
   buffer
2. **False efficiency illusion**: the same cache lines are accessed over and
   over = artificially high "hit rates"
3. **Redundant work**: the grid already has one thread per element, so the loop
   makes it re-traverse the buffer 16,384 times over—this is the dominant
   cost, and it is what the repeated cache hits are hits *on*
4. **Long residency**: the loop keeps every block resident far longer than the
   one-element-per-thread kernels, so the grid drains much more slowly

**Concrete example with float32 (4 bytes each):**

- **Cache line**: 128 bytes = 32 float32 values
- **Stride 512**: a single thread jumps 512×4 = 2048 bytes = 16 cache lines
  between its own iterations
- **Warp impact**: a warp still reads 32 consecutive floats per iteration, so
  each access is one cache line—the waste is in how many times the grid
  revisits those lines, not in how many lines a warp touches

**The key insight**: High cache hits in Kernel2 come from
**re-reading data the grid has already fetched**, not from smart caching!

## **Profiling methodology insights**

**Systematic Detective Approach:**

**Phase 1: Nsight Systems (Big Picture)**

- Identify which kernels are slow
- Rule out obvious bottlenecks (memory transfers, API overhead)
- Focus on kernel execution time differences

**Phase 2: Nsight Compute (Deep Analysis)**

- Analyze memory throughput metrics
- Compare bandwidth utilization percentages
- Investigate cache hit rates and patterns

**Phase 3: Connect Evidence to Theory**

```text
PROFILING EVIDENCE → CODE ANALYSIS:

Nsight Compute Results:           Actual Code Pattern:
- Kernel1: ~2.06 TB/s            → i = block_idx*block_dim + thread_idx (coalesced)
- Kernel2: ~8.7 GB/s, 99% L2 hits → i += 512 (redundant re-traversal)
- Kernel3: ~1.78 TB/s            → reverse_i = size-1-forward_i (reverse coalesced)

The profiler data directly reveals how much work each kernel issues!
```

**Evidence-to-Code Connection:**

- **High throughput + normal cache rates** = Coalesced access (Kernels 1 & 3)
- **Low throughput + high cache rates** = Redundant re-traversal of data already
  fetched (Kernel 2)
- **Memory bandwidth utilization** reveals true efficiency regardless of cache
  statistics

## **Real-world performance implications**

**This pattern affects many GPU applications:**

**Scientific Computing:**

- **Stencil computations**: Neighbor access patterns in grid simulations
- **Linear algebra**: Matrix traversal order (row-major vs column-major)
- **PDE solvers**: Grid point access patterns in finite difference methods

**Graphics and Image Processing:**

- **Texture filtering**: Sample access patterns in shaders
- **Image convolution**: Filter kernel memory access
- **Color space conversion**: Channel interleaving strategies

**Machine Learning:**

- **Matrix operations**: Memory layout optimization in GEMM
- **Tensor contractions**: Multi-dimensional array access patterns
- **Data loading**: Batch processing and preprocessing pipelines

## **Fundamental GPU optimization principles**

**Memory-First Optimization Strategy:**

1. **Count the work first**: the number of element operations a grid issues
   often dwarfs every other effect, as kernel2 shows
2. **Coalescing is critical**: Design for adjacent threads accessing adjacent
   memory
3. **Measure bandwidth utilization**: Focus on actual throughput, not just cache
   statistics
4. **Profile systematically**: Use Nsight tools to identify real bottlenecks

**Key Technical Insights:**

- **Memory-bound workloads**: Bandwidth utilization determines performance
- **Cache metrics can mislead**: High hit rates don't always indicate efficiency
- **Warp-level thinking**: Design access patterns for 32-thread execution groups
- **Hardware-aware programming**: Understanding GPU memory hierarchy is
  essential

## **Key takeaways**

This detective case reveals that
**GPU performance optimization requires abandoning CPU intuition** for
**memory-centric thinking**:

**Critical insights:**

- High cache hit rates can indicate wasted work (not good performance)
- Memory bandwidth utilization matters more than cache statistics
- Doing the work once beats re-reading data the grid has already fetched
- Profiling tools reveal counterintuitive performance truths

**Practical methodology:**

- Profile systematically with Nsight Systems and Nsight Compute
- Design for adjacent threads accessing adjacent memory (coalescing)
- Let profiler evidence guide optimization decisions, not intuition

The cache paradox demonstrates that
**high-level metrics can mislead without architectural understanding** -
applicable far beyond GPU programming.

</details>
