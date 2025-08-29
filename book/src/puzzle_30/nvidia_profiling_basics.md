# 📚 NVIDIA Profiling Basics

## Overview

You've learned GPU programming fundamentals and advanced patterns. Part II taught you debugging techniques for **correctness** using `compute-sanitizer` and `cuda-gdb`, while other parts covered different GPU features like warp programming, memory systems, and block-level operations. Your kernels work correctly - but are they **fast**?

> This tutorial follows NVIDIA's recommended profiling methodology from the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling).

**Key Insight**: A correct kernel can still be orders of magnitude slower than optimal. Profiling bridges the gap between working code and high-performance code.

## The profiling toolkit

Since you have `cuda-toolkit` via pixi, you have access to NVIDIA's professional profiling suite:

### NSight Systems (`nsys`) - the "big picture" tool

**Purpose**: System-wide performance analysis ([NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/))
- Timeline view of CPU-GPU interaction
- Memory transfer bottlenecks
- Kernel launch overhead
- Multi-GPU coordination
- API call tracing

**Available interfaces**: Command-line (`nsys`) and GUI (`nsys-ui`)

**Use when**:
- Understanding overall application flow
- Identifying CPU-GPU synchronization issues
- Analyzing memory transfer patterns
- Finding kernel launch bottlenecks

```bash
# See the help
pixi run nsys --help

# Basic system-wide profiling
pixi run nsys profile --trace=cuda,nvtx --output=timeline mojo your_program.mojo

# Interactive analysis
pixi run nsys stats --force-export=true timeline.nsys-rep
```

### NSight Compute (`ncu`) - the "kernel deep-dive" tool

**Purpose**: Detailed single-kernel performance analysis ([NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/))
- Roofline model analysis
- Memory hierarchy utilization
- Warp execution efficiency
- Register/shared memory usage
- Compute unit utilization

**Available interfaces**: Command-line (`ncu`) and GUI (`ncu-ui`)

**Use when**:
- Optimizing specific kernel performance
- Understanding memory access patterns
- Analyzing compute vs memory bound kernels
- Identifying warp divergence issues

```bash
# See the help
pixi run ncu --help

# Detailed kernel profiling
pixi run ncu --set full --output kernel_profile mojo your_program.mojo

# Focus on specific kernels
pixi run ncu --kernel-name regex:your_kernel_name mojo your_program.mojo
```

## Tool selection decision tree

```
Performance Problem
        |
        v
Know which kernel?
    |           |
   No          Yes
    |           |
    v           v
NSight    Kernel-specific issue?
Systems       |           |
    |        No          Yes
    v         |           |
Timeline      |           v
Analysis <----+     NSight Compute
                          |
                          v
                   Kernel Deep-Dive
```

**Quick Decision Guide**:
- **Start with NSight Systems (`nsys`)** if you're unsure where the bottleneck is
- **Use NSight Compute (`ncu`)** when you know exactly which kernel to optimize
- **Use both** for comprehensive analysis (common workflow)

## Hands-on: system-wide profiling with NSight Systems

Let's profile the Matrix Multiplication implementations from [Puzzle 16](../puzzle_16/puzzle_16.md) to understand performance differences.

> **GUI Note**: The NSight Systems and Compute GUIs (`nsys-ui`, `ncu-ui`) require a display and OpenGL support. On headless servers or remote systems without X11 forwarding, use the command-line versions (`nsys`, `ncu`) with text-based analysis via `nsys stats` and `ncu --import --page details`. You can also transfer `.nsys-rep` and `.ncu-rep` files to local machines for GUI analysis.

### Step 1: Prepare your code for profiling

**Critical**: For accurate profiling, build with full debug information while keeping optimizations enabled:

```bash
pixi shell -e cuda
# Build with full debug info (for comprehensive source mapping) with optimizations enabled
mojo build --debug-level=full solutions/p16/p16.mojo -o solutions/p16/p16_optimized

# Test the optimized build
./solutions/p16/p16_optimized --naive
```

**Why this matters**:
- **Full debug info**: Provides complete symbol tables, variable names, and source line mapping for profilers
- **Comprehensive analysis**: Enables NSight tools to correlate performance data with specific code locations
- **Optimizations enabled**: Ensures realistic performance measurements that match production builds

### Step 2: Capture system-wide profile

```bash
# Profile the optimized build with comprehensive tracing
nsys profile \
  --trace=cuda,nvtx \
  --output=matmul_naive \
  --force-overwrite=true \
  ./solutions/p16/p16_optimized --naive
```

**Command breakdown**:
- `--trace=cuda,nvtx`: Capture CUDA API calls and custom annotations
- `--output=matmul_naive`: Save profile as `matmul_naive.nsys-rep`
- `--force-overwrite=true`: Replace existing profiles
- Final argument: Your Mojo program

### Step 3: Analyze the timeline

```bash
# Generate text-based statistics
nsys stats --force-export=true matmul_naive.nsys-rep

# Key metrics to look for:
# - GPU utilization percentage
# - Memory transfer times
# - Kernel execution times
# - CPU-GPU synchronization gaps
```

**What you'll see** (actual output from a 2×2 matrix multiplication):

```txt
** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)  Min (ns)  Max (ns)  StdDev (ns)          Name
 --------  ---------------  ---------  ---------  --------  --------  --------  -----------  --------------------
     81.9          8617962          3  2872654.0    2460.0      1040   8614462    4972551.6  cuMemAllocAsync
     15.1          1587808          4   396952.0    5965.5      3810   1572067     783412.3  cuMemAllocHost_v2
      0.6            67152          1    67152.0   67152.0     67152     67152          0.0  cuModuleLoadDataEx
      0.4            44961          1    44961.0   44961.0     44961     44961          0.0  cuLaunchKernelEx

** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------
    100.0             1920          1    1920.0    1920.0      1920      1920          0.0  p16_naive_matmul_Layout_Int6A6AcB6A6AsA6A6A

** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     49.4             4224      3    1408.0    1440.0      1312      1472         84.7  [CUDA memcpy Device-to-Host]
     36.0             3072      4     768.0     528.0       416      1600        561.0  [CUDA memset]
     14.6             1248      3     416.0     416.0       416       416          0.0  [CUDA memcpy Host-to-Device]
```

**Key Performance Insights**:
- **Memory allocation dominates**: 81.9% of total time spent on `cuMemAllocAsync`
- **Kernel is lightning fast**: Only 1,920 ns (0.000001920 seconds) execution time
- **Memory transfer breakdown**: 49.4% Device→Host, 36.0% memset, 14.6% Host→Device
- **Tiny data sizes**: All memory operations are < 0.001 MB (4 float32 values = 16 bytes)

### Step 4: Compare implementations

Profile different versions and compare:

```bash
# Make sure you've in pixi shell still `pixi run -e cuda`

# Profile shared memory version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_shared ./solutions/p16/p16_optimized --single-block

# Profile tiled version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_tiled ./solutions/p16/p16_optimized --tiled

# Profile idiomatic tiled version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_idiomatic_tiled ./solutions/p16/p16_optimized --idiomatic-tiled

# Analyze each implementation separately (nsys stats processes one file at a time)
nsys stats --force-export=true matmul_shared.nsys-rep
nsys stats --force-export=true matmul_tiled.nsys-rep
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep
```

**How to compare the results**:
1. **Look at GPU Kernel Summary** - Compare execution times between implementations
2. **Check Memory Operations** - See if shared memory reduces global memory traffic
3. **Compare API overhead** - All should have similar memory allocation patterns

**Manual comparison workflow**:
```bash
# Run each analysis and save output for comparison
nsys stats --force-export=true matmul_naive.nsys-rep > naive_stats.txt
nsys stats --force-export=true matmul_shared.nsys-rep > shared_stats.txt
nsys stats --force-export=true matmul_tiled.nsys-rep > tiled_stats.txt
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep > idiomatic_tiled_stats.txt
```

**Fair Comparison Results** (actual output from profiling):

### Comparison 1: 2 x 2 matrices
| Implementation | Memory Allocation | Kernel Execution | Performance |
|----------------|------------------|------------------|-------------|
| **Naive**      | 81.9% cuMemAllocAsync | ✅ 1,920 ns | Baseline |
| **Shared** (`--single-block`) | 81.8% cuMemAllocAsync | ✅ 1,984 ns | **+3.3% slower** |

### Comparison 2: 9 x 9 matrices
| Implementation | Memory Allocation | Kernel Execution | Performance |
|----------------|------------------|------------------|-------------|
| **Tiled** (manual) | 81.1% cuMemAllocAsync | ✅ 2,048 ns | Baseline |
| **Idiomatic Tiled** | 81.6% cuMemAllocAsync | ✅ 2,368 ns | **+15.6% slower** |

**Key Insights from Fair Comparisons**:

**Both Matrix Sizes Are Tiny for GPU Work!**:
- **2×2 matrices**: 4 elements - completely overhead-dominated
- **9×9 matrices**: 81 elements - still completely overhead-dominated
- **Real GPU workloads**: Thousands to millions of elements per dimension

**What These Results Actually Show**:
- **All variants dominated by memory allocation** (>81% of time)
- **Kernel execution is irrelevant** compared to setup costs
- **"Optimizations" can hurt**: Shared memory adds 3.3% overhead, async_copy adds 15.6%
- **The real lesson**: For tiny workloads, algorithm choice doesn't matter - overhead dominates everything

**Why This Happens**:
- GPU setup cost (memory allocation, kernel launch) is fixed regardless of problem size
- For tiny problems, this fixed cost dwarfs computation time
- Optimizations designed for large problems become overhead for small ones

**Real-World Profiling Lessons**:
- **Problem size context matters**: Both 2×2 and 9×9 are tiny for GPUs
- **Fixed costs dominate small problems**: Memory allocation, kernel launch overhead
- **"Optimizations" can hurt tiny workloads**: Shared memory, async operations add overhead
- **Don't optimize tiny problems**: Focus on algorithms that scale to real workloads
- **Always benchmark**: Assumptions about "better" code are often wrong

**Understanding Small Kernel Profiling**:
This 2×2 matrix example demonstrates a **classic small-kernel pattern**:
- The actual computation (matrix multiply) is extremely fast (1,920 ns)
- Memory setup overhead dominates the total time (97%+ of execution)
- This is why **real-world GPU optimization** focuses on:
  - **Batching operations** to amortize setup costs
  - **Memory reuse** to reduce allocation overhead
  - **Larger problem sizes** where compute becomes the bottleneck

## Hands-on: kernel deep-dive with NSight Compute

Now let's dive deep into a specific kernel's performance characteristics.

### Step 1: Profile a specific kernel

```bash
# Make sure you're in an active shell
pixi shell -e cuda

# Profile the naive MatMul kernel in detail (using our optimized build)
ncu \
  --set full \
  -o kernel_analysis \
  --force-overwrite \
  ./solutions/p16/p16_optimized --naive
```

> **Common Issue: Permission Error**
>
> If you get `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`, try these > solutions:
>
> ```bash
> # Add NVIDIA driver option (safer than rmmod)
> echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
>
> # Set kernel parameter
> sudo sysctl -w kernel.perf_event_paranoid=0
>
> # Make permanent
> echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf
>
> # Reboot required for driver changes to take effect
> sudo reboot
>
> # Then run the ncu command again
> ncu \
>   --set full \
>   -o kernel_analysis \
>   --force-overwrite \
>   ./solutions/p16/p16_optimized --naive
> ```

### Step 2: Analyze key metrics

```bash
# Generate detailed report (correct syntax)
ncu --import kernel_analysis.ncu-rep --page details
```

**Real NSight Compute Output** (from your 2×2 naive MatMul):

```txt
GPU Speed Of Light Throughput
----------------------- ----------- ------------
DRAM Frequency              Ghz         6.10
SM Frequency                Ghz         1.30
Elapsed Cycles            cycle         3733
Memory Throughput             %         1.02
DRAM Throughput               %         0.19
Duration                     us         2.88
Compute (SM) Throughput       %         0.00
----------------------- ----------- ------------

Launch Statistics
-------------------------------- --------------- ---------------
Block Size                                                     9
Grid Size                                                      1
Threads                           thread               9
Waves Per SM                                                0.00
-------------------------------- --------------- ---------------

Occupancy
------------------------------- ----------- ------------
Theoretical Occupancy                 %        33.33
Achieved Occupancy                    %         2.09
------------------------------- ----------- ------------
```

**Critical Insights from Real Data**:

#### Performance analysis - the brutal truth!
- **Compute Throughput: 0.00%** - GPU is completely idle computationally
- **Memory Throughput: 1.02%** - Barely touching memory bandwidth
- **Achieved Occupancy: 2.09%** - Using only 2% of GPU capability
- **Grid Size: 1 block** - Completely underutilizing 80 multiprocessors!

#### Why performance is so poor
- **Tiny problem size**: 2×2 matrix = 4 elements total
- **Poor launch configuration**: 9 threads in 1 block (should be multiples of 32)
- **Massive underutilization**: 0.00 waves per SM (need thousands for efficiency)

#### Key optimization recommendations from NSight Compute
- **"Est. Speedup: 98.75%"** - Increase grid size to use all 80 SMs
- **"Est. Speedup: 71.88%"** - Use thread blocks as multiples of 32
- **"Kernel grid is too small"** - Need much larger problems for GPU efficiency

### Step 3: The reality check

**What This Profiling Data Teaches Us**:

1. **Tiny problems are GPU poison**: 2×2 matrices completely waste GPU resources
2. **Launch configuration matters**: Wrong thread/block sizes kill performance
3. **Scale matters more than algorithm**: No optimization can fix a fundamentally tiny problem
4. **NSight Compute is honest**: It tells us when our kernel performance is poor

**The Real Lesson**:
- **Don't optimize toy problems** - they're not representative of real GPU workloads
- **Focus on realistic workloads** - 1000×1000+ matrices where optimizations actually matter
- **Use profiling to guide optimization** - but only on problems worth optimizing

**For our tiny 2×2 example**: All the sophisticated algorithms (shared memory, tiling) just add overhead to an already overhead-dominated workload.

## Reading profiler output like a performance detective

### Common performance patterns

#### Pattern 1: Memory-bound kernel
**NSight Systems shows**: Long memory transfer times
**NSight Compute shows**: High memory throughput, low compute utilization
**Solution**: Optimize memory access patterns, use shared memory

#### Pattern 2: Low occupancy
**NSight Systems shows**: Short kernel execution with gaps
**NSight Compute shows**: Low achieved occupancy
**Solution**: Reduce register usage, optimize block size

#### Pattern 3: Warp divergence
**NSight Systems shows**: Irregular kernel execution patterns
**NSight Compute shows**: Low warp execution efficiency
**Solution**: Minimize conditional branches, restructure algorithms

### Profiling detective workflow

```
Performance Issue
        |
        v
NSight Systems: Big Picture
        |
        v
GPU Well Utilized?
    |           |
   No          Yes
    |           |
    v           v
Fix CPU-GPU    NSight Compute: Kernel Detail
Pipeline            |
                    v
            Memory or Compute Bound?
                |       |       |
             Memory  Compute  Neither
                |       |       |
                v       v       v
           Optimize  Optimize  Check
           Memory    Arithmetic Occupancy
           Access
```

## Profiling best practices

For comprehensive profiling guidelines, refer to the [Best Practices Guide - Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics).

### Do's

1. **Profile representative workloads**: Use realistic data sizes and patterns
2. **Build with full debug info**: Use `--debug-level=full` for comprehensive profiling data and source mapping with optimizations
3. **Warm up the GPU**: Run kernels multiple times, profile later iterations
4. **Compare alternatives**: Always profile multiple implementations
5. **Focus on hotspots**: Optimize the kernels that take the most time

### Don'ts

1. **Don't profile without debug info**: You won't be able to map performance back to source code (`mojo build --help`)
2. **Don't profile single runs**: GPU performance can vary between runs
3. **Don't ignore memory transfers**: CPU-GPU transfers often dominate
4. **Don't optimize prematurely**: Profile first, then optimize

### Common pitfalls and solutions

#### Pitfall 1: Cold start effects

```bash
# Wrong: Profile first run
nsys profile mojo your_program.mojo

# Right: Warm up, then profile
nsys profile --delay=5 mojo your_program.mojo  # Let GPU warm up
```

#### Pitfall 2: Wrong build configuration

```bash
# Wrong: Full debug build (disables optimizations) i.e. `--no-optimization`
mojo build -O0 your_program.mojo -o your_program

# Wrong: No debug info (can't map to source)
mojo build your_program.mojo -o your_program

# Right: Optimized build with full debug info for profiling
mojo build --debug-level=full your_program.mojo -o optimized_program
nsys profile ./optimized_program
```

#### Pitfall 3: Ignoring memory transfers
```txt
# Look for this pattern in NSight Systems:
CPU -> GPU transfer: 50ms
Kernel execution: 2ms
GPU -> CPU transfer: 48ms
# Total: 100ms (kernel is only 2%!)
```
**Solution**: Overlap transfers with compute, reduce transfer frequency (covered in Part IX)

#### Pitfall 4: Single kernel focus
```bash
# Wrong: Only profile the "slow" kernel
ncu --kernel-name regex:slow_kernel program

# Right: Profile the whole application first
nsys profile mojo program.mojo  # Find real bottlenecks
```

## Best practices and advanced options

### Advanced NSight Systems profiling

For comprehensive system-wide analysis, use these advanced `nsys` flags:

```bash
# Production-grade profiling command
nsys profile \
  --gpu-metrics-devices=all \
  --trace=cuda,osrt,nvtx \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --opengl-gpu-workload=false \
  --delay=2 \
  --duration=30 \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output=comprehensive_profile \
  --force-overwrite=true \
  ./your_program
```

**Flag explanations**:
- `--gpu-metrics-devices=all`: Collect GPU metrics from all devices
- `--trace=cuda,osrt,nvtx`: Comprehensive API tracing
- `--cuda-memory-usage=true`: Track memory allocation/deallocation
- `--cuda-um-cpu/gpu-page-faults=true`: Monitor Unified Memory page faults
- `--delay=2`: Wait 2 seconds before profiling (avoid cold start)
- `--duration=30`: Profile for 30 seconds max
- `--sample=cpu`: Include CPU sampling for hotspot analysis
- `--cpuctxsw=process-tree`: Track CPU context switches

### Advanced NSight Compute profiling

For detailed kernel analysis with comprehensive metrics:

```bash
# Full kernel analysis with all metric sets
ncu \
  --set full \
  --import-source=on \
  --kernel-id=:::1 \
  --launch-skip=0 \
  --launch-count=1 \
  --target-processes=all \
  --replay-mode=kernel \
  --cache-control=all \
  --clock-control=base \
  --apply-rules=yes \
  --check-exit-code=yes \
  --export=detailed_analysis \
  --force-overwrite \
  ./your_program

# Focus on specific performance aspects
ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:your_kernel_.* \
  --export=targeted_analysis \
  ./your_program
```

**Key NSight Compute flags**:
- `--set full`: Collect all available metrics (comprehensive but slow)
- `--set @roofline`: Optimized set for roofline analysis
- `--import-source=on`: Map results back to source code
- `--replay-mode=kernel`: Replay kernels for accurate measurements
- `--cache-control=all`: Control GPU caches for consistent results
- `--clock-control=base`: Lock clocks to base frequencies
- `--section=SpeedOfLight`: Include Speed of Light analysis
- `--metrics=...`: Collect specific metrics only
- `--kernel-name regex:pattern`: Target kernels using regex patterns (not `--kernel-regex`)

### Profiling workflow best practices

#### 1. Progressive profiling strategy
```bash
# Step 1: Quick overview (fast)
nsys profile --trace=cuda --duration=10 --output=quick_look ./program

# Step 2: Detailed system analysis (medium)
nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true --output=detailed ./program

# Step 3: Kernel deep-dive (slow but comprehensive)
ncu --set=@roofline --kernel-name regex:hotspot_kernel ./program
```

#### 2. Multi-run analysis for reliability
```bash
# Profile multiple runs and compare
for i in {1..5}; do
  nsys profile --output=run_${i} ./program
  nsys stats run_${i}.nsys-rep > stats_${i}.txt
done

# Compare results
diff stats_1.txt stats_2.txt
```

#### 3. Targeted kernel profiling
```bash
# First, identify hotspot kernels
nsys profile --trace=cuda,nvtx --output=overview ./program
nsys stats overview.nsys-rep | grep -A 10 "GPU Kernel Summary"

# Then profile specific kernels
ncu --kernel-name="identified_hotspot_kernel" --set full ./program
```

### Environment and build best practices

#### Optimal build configuration
```bash
# For profiling: optimized with full debug info
mojo build --debug-level=full --optimization-level=3 program.mojo -o program_profile

# Verify build settings
mojo build --help | grep -E "(debug|optimization)"
```

#### Profiling environment setup

```bash
# Disable GPU boost for consistent results
sudo nvidia-smi -ac 1215,1410  # Lock memory and GPU clocks

# Set deterministic behavior
export CUDA_LAUNCH_BLOCKING=1  # Synchronous launches for accurate timing

# Increase driver limits for profiling
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
```

#### Memory and performance isolation
```bash
# Clear GPU memory before profiling
nvidia-smi --gpu-reset

# Disable other GPU processes
sudo fuser -v /dev/nvidia*  # Check what's using GPU
sudo pkill -f cuda  # Kill CUDA processes if needed

# Run with high priority
sudo nice -n -20 nsys profile ./program
```

### Analysis and reporting best practices

#### Comprehensive report generation
```bash
# Generate multiple report formats
nsys stats --report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format=csv --output=. profile.nsys-rep

# Export for external analysis
nsys export --type=sqlite profile.nsys-rep
nsys export --type=json profile.nsys-rep

# Generate comparison reports
nsys stats --report=cuda_gpu_kern_sum baseline.nsys-rep > baseline_kernels.txt
nsys stats --report=cuda_gpu_kern_sum optimized.nsys-rep > optimized_kernels.txt
diff -u baseline_kernels.txt optimized_kernels.txt
```

#### Performance regression testing
```bash
#!/bin/bash
# Automated profiling script for CI/CD
BASELINE_TIME=$(nsys stats baseline.nsys-rep | grep "Total Time" | awk '{print $3}')
CURRENT_TIME=$(nsys stats current.nsys-rep | grep "Total Time" | awk '{print $3}')

REGRESSION_THRESHOLD=1.10  # 10% slowdown threshold
if (( $(echo "$CURRENT_TIME > $BASELINE_TIME * $REGRESSION_THRESHOLD" | bc -l) )); then
    echo "Performance regression detected: ${CURRENT_TIME}ns vs ${BASELINE_TIME}ns"
    exit 1
fi
```

## Next steps

Now that you understand profiling fundamentals:

1. **Practice with your existing kernels**: Profile puzzles you've already solved
2. **Prepare for optimization**: Puzzle 31 will use these insights for occupancy optimization
3. **Understand the tools**: Experiment with different NSight Systems and NSight Compute options

**Remember**: Profiling is not just about finding slow code - it's about understanding your program's behavior and making informed optimization decisions.

For additional profiling resources, see:
- [NVIDIA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NSight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
