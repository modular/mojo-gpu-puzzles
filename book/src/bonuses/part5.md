# 🎯 Performance Bonus Challenge

## The discovery

You've just completed [Puzzle 33](../puzzle_33/puzzle_33.md) and implemented
actual Tensor Core matrix multiplication using Mojo's `TensorCore` API. The
implementation works correctly, passes all accuracy tests, and uses real
hardware-accelerated matrix operations. But when you profile it against the
simple idiomatic tiled version from [Puzzle 16](../puzzle_16/tiled.md)...

**The "specialized hardware" is orders of magnitude slower!**

### What went wrong?

Your profiling with (NVIDIA only) `ncu` revealed the brutal truth (if you need a
refresher on profiling techniques, see
[Puzzle 30's GPU profiling](../puzzle_30/puzzle_30.md) and its
[NVIDIA profiling basics](../puzzle_30/nvidia_profiling_basics.md)):

**Tensor Core version (the disappointment):**

- **Duration**: ~13.9 ms
- **Memory bound**: 72.5% DRAM throughput (should be compute-bound!)
- **Poor occupancy**: 26.3% (wasted hardware)
- **Cache disaster**: 29.7% L2 hit rate
- **Register pressure**: 68 registers per thread
- **Shared memory conflicts**: Bank conflicts destroying performance

**Tiled version (the winner):**

- **Duration**: ~1.62 ms (8.6x faster!)
- **Compute bound**: 1.7% DRAM throughput (as expected)
- **Excellent occupancy**: 66.7%
- **Cache friendly**: 96.9% L2 hit rate
- **Efficient**: 38 registers per thread
- **Clean memory**: No significant bank conflicts

### The harsh reality

This is a common story in GPU optimization:
**raw hardware capability ≠ actual performance**. Tensor Cores are incredibly
powerful, but they're also incredibly demanding:

- **Memory wall**: They're so fast they expose every memory bottleneck
- **Resource hungry**: High register usage kills occupancy
- **Access sensitive**: Poor memory patterns destroy cache behavior
- **Configuration critical**: Launch parameters must be perfectly tuned

### Your mission: Fix the tensor core performance

**The challenge:** Transform your memory-bound, low-occupancy Tensor Core
implementation into something that actually beats the simple tiled version.

**What you need to beat:**

- **Target duration**: < 1.62 ms
- **Occupancy**: > 26.3% baseline
- **DRAM pressure**: < 72.5% baseline
- **Cache performance**: > 29.7% L2 hit rate baseline

**Optimization strategies to explore:**

1. **Register pressure reduction**
   - Use smaller accumulator tiles
   - Minimize intermediate storage
   - Consider mixed-precision to reduce register footprint
   - Review [Puzzle 16's tiled approach](../puzzle_16/tiled.md) for efficient
     accumulation patterns

2. **Memory pattern optimization**
   - Add shared memory padding to eliminate bank conflicts (see
     [shared memory concepts](../puzzle_16/shared_memory.md))
   - Optimize `copy_dram_to_sram_async` layouts
   - Improve coalescing patterns (memory access fundamentals from
     [early puzzles](../puzzle_01/puzzle_01.md))

3. **Occupancy improvements**
   - Tune block sizes for better warp utilization
   - Balance shared memory vs register usage
   - Optimize warp-to-SM mapping
   - Apply the occupancy lessons from
     [Puzzle 31](../puzzle_31/puzzle_31.md)

4. **Cache optimization**
   - Improve data reuse patterns
   - Optimize tile sizes for cache hierarchy
   - Consider data layout transformations
   - Build on the cache behavior analyzed in
     [Puzzle 30's cache hit paradox](../puzzle_30/profile_kernels.md)

5. **Advanced techniques**
   - Implement double buffering to overlap memory and compute
   - Use software pipelining
   - Explore async execution patterns
   - Apply the overlap and double-buffering patterns from
     [Puzzle 28](../puzzle_28/puzzle_28.md) and
     [Puzzle 29's double-buffered stencil](../puzzle_29/memory_barrier.md)

### Success criteria

- **Correctness**: All accuracy tests still pass
- **Performance**: Tensor Core duration < 1.62 ms
- **Efficiency**: Higher occupancy (>26.3%)
- **Memory**: Lower DRAM pressure (<72.5%)
- **Cache**: Better hit rates (>29.7% L2)

### The deeper lesson

This bonus challenge teaches the most important lesson in GPU optimization:
**understanding bottlenecks matters more than using the latest APIs**.

The goal isn't just to make Tensor Cores faster - it's to understand why they
can be slower, how to systematically diagnose performance problems, and how to
apply principled optimization techniques.

Complete this challenge, and you'll have the skills to optimize any GPU
workload, regardless of the hardware features available.
