# 📊 Benchmarking - Performance Analysis and Optimization

## Overview

After learning **elementwise**, **tiled**, **manual vectorization**, and **Mojo vectorize** patterns, it's time to measure their actual performance. Here's how to use the built-in benchmarking system in `p21.mojo` to scientifically compare these approaches and understand their performance characteristics.

> **Key insight:** _Theoretical analysis is valuable, but empirical benchmarking reveals the true performance story on your specific hardware._

## Running benchmarks

To execute the comprehensive benchmark suite:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p23 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --benchmark
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --benchmark
```

  </div>
</div>

Your output will show performance measurements for each pattern:

```txt
SIZE: 1024
simd_width: 4
Running P21 GPU Benchmarks...
SIMD width: 4
--------------------------------------------------------------------------------
Testing SIZE=16, TILE=4
Running elementwise_16_4
Running tiled_16_4
Running manual_vectorized_16_4
Running vectorized_16_4
--------------------------------------------------------------------------------
Testing SIZE=128, TILE=16
Running elementwise_128_16
Running tiled_128_16
Running manual_vectorized_128_16
--------------------------------------------------------------------------------
Testing SIZE=128, TILE=16, Vectorize within tiles
Running vectorized_128_16
--------------------------------------------------------------------------------
Testing SIZE=1048576 (1M), TILE=1024
Running elementwise_1M_1024
Running tiled_1M_1024
Running manual_vectorized_1M_1024
Running vectorized_1M_1024
| name                      | met (ms)              | iters |
| ------------------------- | --------------------- | ----- |
| elementwise_16_4          | 0.0033248             | 100   |
| tiled_16_4                | 0.00327392            | 100   |
| manual_vectorized_16_4    | 0.0036169600000000002 | 100   |
| vectorized_16_4           | 0.0037209599999999997 | 100   |
| elementwise_128_16        | 0.00351999            | 100   |
| tiled_128_16              | 0.00370431            | 100   |
| manual_vectorized_128_16  | 0.0043696             | 100   |
| vectorized_128_16         | 0.00378048            | 100   |
| elementwise_1M_1024       | 0.03130143            | 100   |
| tiled_1M_1024             | 0.6892189000000001    | 100   |
| manual_vectorized_1M_1024 | 0.5923888             | 100   |
| vectorized_1M_1024        | 0.1876688             | 100   |

Benchmarks completed!
```

## Benchmark configuration

The benchmarking system uses Mojo's built-in `benchmark` module:

```mojo
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
```

- **`max_iters=10`**: Up to 10 iterations for statistical reliability
- **`num_warmup_iters=1`**: GPU warmup before measurement
- Check out the [benchmark documentation](https://docs.modular.com/mojo/stdlib/benchmark/)

## Benchmarking implementation essentials

### Core workflow pattern

Each benchmark follows a streamlined pattern:

```mojo
@parameter
fn benchmark_pattern_parameterized[test_size: Int, tile_size: Int](mut b: Bencher) raises:
    bench_ctx = DeviceContext()
    # Setup: Create buffers and initialize data
    @parameter
    fn pattern_workflow(ctx: DeviceContext) raises:
      # Compute: Execute the algorithm being measured

    b.iter_custom[pattern_workflow](bench_ctx)
    # Prevent optimization: keep(out.unsafe_ptr())
    # Synchronize: ctx.synchronize()
```

**Key phases:**

1. **Setup**: Buffer allocation and data initialization
2. **Computation**: The actual algorithm being benchmarked
3. **Prevent optimization**: Critical for accurate measurement
4. **Synchronization**: Ensure GPU work completes

> **Critical: The `keep()` function**
> `keep(out.unsafe_ptr())` prevents the compiler from optimizing away your computation as "unused code." Without this, you might measure nothing instead of your algorithm! This is essential for accurate GPU benchmarking because kernels are launched asynchronously.

### Why custom iteration works for GPU

Standard benchmarking assumes CPU-style synchronous execution. GPU kernels launch asynchronously, so we need:

- **GPU context management**: Proper DeviceContext lifecycle
- **Memory management**: Buffer cleanup between iterations
- **Synchronization handling**: Accurate timing of async operations
- **Overhead isolation**: Separate setup cost from computation cost

## Test scenarios and thread analysis

The benchmark suite tests three scenarios to reveal performance characteristics:

### Thread utilization summary

| Problem Size | Pattern | Threads | SIMD ops/thread | Total SIMD ops |
|-------------|---------|---------|-----------------|----------------|
| **SIZE=16** | Elementwise | 4 | 1 | 4 |
|             | Tiled | 4 | 1 | 4 |
|             | Manual | 1 | 4 | 4 |
|             | Vectorize | 4 | 1 | 4 |
| **SIZE=128** | Elementwise | 32 | 1 | 32 |
|              | Tiled | 8 | 4 | 32 |
|              | Manual | 2 | 16 | 32 |
|              | Vectorize | 8 | 4 | 32 |
| **SIZE=1M** | Elementwise | 262,144 | 1 | 262,144 |
|             | Tiled | 1,024 | 256 | 262,144 |
|             | Manual | 256 | 1,024 | 262,144 |
|             | Vectorize | 1,024 | 256 | 262,144 |

### Performance characteristics by problem size

**Small problems (SIZE=16):**

- Launch overhead dominates (~0.003ms baseline)
- Thread count differences don't matter
- Tiled/vectorize show slightly lower overhead

**Medium problems (SIZE=128):**

- Still overhead-dominated (~0.003ms for all)
- Performance differences nearly disappear
- Transitional behaviour between overhead and computation

**Large problems (SIZE=1M):**

- Real algorithmic differences emerge
- Impact of uncoalesced loads becomes apparent
- Clear performance ranking appears

## What the data shows

Based on empirical benchmark results across different hardware:

### Performance rankings (large problems)

| Rank | Pattern | Typical time | Key insight |
|------|---------|-------------|-------------|
| 🥇 | **Elementwise** | ~0.03ms | Coalesced memory access wins for memory-bound ops |
| 🥈 | **Mojo vectorize** | ~0.19ms | Uncoalesced memory access hurts performance |
| 🥉 | **Manual vectorized** | ~0.59ms | Uncoalesced memory access and manual optimization reduces performance |
| 4th | **Tiled** | ~0.69ms | Uncoalesced memory access, manual optimization without SIMD loads reduces performance further |

### Key performance insights

> **For simple memory-bound operations:** Maximum parallelism (elementwise) outperforms complex memory optimizations at scale.

**Why elementwise wins:**

- **262,144 threads** provide excellent latency hiding
- **Simple memory patterns** achieve good coalescing
- **Minimal overhead** per thread
- **Scales naturally** with GPU core count

**Why tiled and vectorize are competitive:**

- **Balanced approach** between parallelism and memory locality
- **Automatic optimization** (vectorize) performs nearly as well as manual tiling
- **Good thread utilization** without excessive complexity

**Why manual vectorization struggles:**

- **Only 256 threads** limit parallelism
- **Complex indexing** adds computational overhead
- **Cache pressure** from large chunks per thread
- **Diminishing returns** for simple arithmetic

**Framework intelligence:**

- Automatic iteration count adjustment (91-100 iterations)
- Statistical reliability across different execution times
- Handles thermal throttling and system variation

## Interpreting your results

### Reading the output table

```txt
| name                     | met (ms)           | iters |
| elementwise_1M_1024      | 0.03130143         | 100   |
```

- **`met (ms)`**: Execution time for a single iteration
- **`iters`**: Number of iterations performed
- **Compare within problem size**: Same-size comparisons are most meaningful

### Making optimization decisions

**Choose patterns based on empirical evidence:**

**For production workloads:**

- **Large datasets (>100K elements)**: Elementwise typically optimal
- **Small/startup datasets (<1K elements)**: Tiled or vectorize for lower overhead
- **Development speed priority**: Mojo vectorize for automatic optimization
- **Avoid manual vectorization**: Complexity rarely pays off for simple operations

**Performance optimization workflow:**

1. **Profile first**: Measure before optimizing
2. **Test at scale**: Small problems mislead about real performance
3. **Consider total cost**: Include development and maintenance effort
4. **Validate improvements**: Confirm with benchmarks on target hardware

## Advanced benchmarking techniques

### Custom test scenarios

Modify parameters to test different conditions:

```mojo
# Different problem sizes
benchmark_elementwise_parameterized[1024, 32]  # Large problem
benchmark_elementwise_parameterized[64, 8]     # Small problem

# Different tile sizes
benchmark_tiled_parameterized[256, 8]   # Small tiles
benchmark_tiled_parameterized[256, 64]  # Large tiles
```

### Hardware considerations

Your results will vary based on:

- **GPU architecture**: SIMD width, core count, memory bandwidth
- **System configuration**: PCIe bandwidth, CPU performance
- **Thermal state**: GPU boost clocks vs sustained performance
- **Concurrent workloads**: Other processes affecting GPU utilization

## Best practices summary

**Benchmarking workflow:**

1. **Warm up GPU** before critical measurements
2. **Run multiple iterations** for statistical significance
3. **Test multiple problem sizes** to understand scaling
4. **Use `keep()` consistently** to prevent optimization artifacts
5. **Compare like with like** (same problem size, same hardware)

**Performance decision framework:**

- **Start simple**: Begin with elementwise for memory-bound operations
- **Measure don't guess**: Theoretical analysis guides, empirical data decides
- **Scale matters**: Small problem performance doesn't predict large problem behaviour
- **Total cost optimization**: Balance development time vs runtime performance

## Next steps

With benchmarking skills:

- **Profile real applications**: Apply these patterns to actual workloads
- **Advanced GPU patterns**: Explore reductions, convolutions, and matrix operations
- **Multi-GPU scaling**: Understand distributed GPU computing patterns
- **Memory optimization**: Dive deeper into shared memory and advanced caching

💡 **Key takeaway**: Benchmarking transforms theoretical understanding into practical performance optimization. Use empirical data to make informed decisions about which patterns work best for your specific hardware and workload characteristics.

## Looking ahead: when you need more control

The functional patterns in Part V provide excellent performance for most workloads, but some algorithms require **direct thread communication**:

### **Algorithms that benefit from warp programming:**

- **Reductions**: Sum, max, min operations across thread groups
- **Prefix operations**: Cumulative sums, running maximums
- **Data shuffling**: Reorganizing data between threads
- **Cooperative algorithms**: Where threads must coordinate closely

### **Performance preview:**

In Part VI, we'll revisit several algorithms from Part II and show how warp operations can:

- **Simplify code**: Replace complex shared memory patterns with single function calls
- **Improve performance**: Eliminate barriers and reduce memory traffic
- **Enable new algorithms**: Unlock patterns impossible with pure functional approaches

**Coming up next**: [Part VII: Warp-Level Programming](../puzzle_24/puzzle_24.md) - starting with a dramatic reimplementation of Puzzle 14's prefix sum.
