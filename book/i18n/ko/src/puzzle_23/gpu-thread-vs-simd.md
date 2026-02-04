# 🧠 GPU Threading vs SIMD - Understanding the Execution Hierarchy

## Overview

After exploring **elementwise**, **tiled**, and **vectorization** patterns, you've seen different ways to organize GPU computation. This section clarifies the fundamental relationship between **GPU threads** and **SIMD operations** - two distinct but complementary levels of parallelism that work together for optimal performance.

> **Key insight:** _GPU threads provide the parallelism structure, while SIMD operations provide the vectorization within each thread._

## Core concepts

### GPU threading hierarchy

GPU execution follows a well-defined hierarchy that abstracts hardware complexity:

```
GPU Device
├── Grid (your entire problem)
│   ├── Block 1 (group of threads, shared memory)
│   │   ├── Warp 1 (32 threads, lockstep execution)
│   │   │   ├── Thread 1 → SIMD operations
│   │   │   ├── Thread 2 → SIMD operations
│   │   │   └── ... (32 threads total)
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (independent group)
```

💡 **Note**: While this Part focuses on functional patterns, **warp-level programming** and advanced GPU memory management will be covered in detail in **[Part VII](../puzzle_24/puzzle_24.md)**.

**What Mojo abstracts for you:**
- **Grid/Block configuration**: Automatically calculated based on problem size
- **Warp management**: Hardware handles 32-thread groups transparently
- **Thread scheduling**: GPU scheduler manages execution automatically
- **Memory hierarchy**: Optimal access patterns built into functional operations

### SIMD within GPU threads

Each GPU thread can process multiple data elements simultaneously using **SIMD (Single Instruction, Multiple Data)** operations:

```mojo
# Within one GPU thread:
a_simd = a.load[simd_width](idx, 0)      # Load 4 floats simultaneously
b_simd = b.load[simd_width](idx, 0)      # Load 4 floats simultaneously
result = a_simd + b_simd                 # Add 4 pairs simultaneously
output.store[simd_width](idx, 0, result) # Store 4 results simultaneously
```

## Pattern comparison and thread-to-work mapping

> **Critical insight:** All patterns perform the **same total work** - 256 SIMD operations for 1024 elements with SIMD_WIDTH=4. The difference is in how this work is distributed across GPU threads.

### Thread organization comparison (`SIZE=1024`, `SIMD_WIDTH=4`)

| Pattern | Threads | SIMD ops/thread | Memory pattern | Trade-off |
|---------|---------|-----------------|----------------|-----------|
| **Elementwise** | 256 | 1 | Distributed access | Max parallelism, poor locality |
| **Tiled** | 32 | 8 | Small blocks | Balanced parallelism + locality |
| **Manual vectorized** | 8 | 32 | Large chunks | High bandwidth, fewer threads |
| **Mojo vectorize** | 32 | 8 | Smart blocks | Automatic optimization |

### Detailed execution patterns

**Elementwise pattern:**
```
Thread 0: [0,1,2,3] → Thread 1: [4,5,6,7] → ... → Thread 255: [1020,1021,1022,1023]
256 threads × 1 SIMD op = 256 total SIMD operations
```

**Tiled pattern:**
```
Thread 0: [0:32] (8 SIMD) → Thread 1: [32:64] (8 SIMD) → ... → Thread 31: [992:1024] (8 SIMD)
32 threads × 8 SIMD ops = 256 total SIMD operations
```

**Manual vectorized pattern:**
```
Thread 0: [0:128] (32 SIMD) → Thread 1: [128:256] (32 SIMD) → ... → Thread 7: [896:1024] (32 SIMD)
8 threads × 32 SIMD ops = 256 total SIMD operations
```

**Mojo vectorize pattern:**
```
Thread 0: [0:32] auto-vectorized → Thread 1: [32:64] auto-vectorized → ... → Thread 31: [992:1024] auto-vectorized
32 threads × 8 SIMD ops = 256 total SIMD operations
```

## Performance characteristics and trade-offs

### Core trade-offs summary

| Aspect | High thread count (Elementwise) | Moderate threads (Tiled/Vectorize) | Low threads (Manual) |
|--------|--------------------------------|-----------------------------------|----------------------|
| **Parallelism** | Maximum latency hiding | Balanced approach | Minimal parallelism |
| **Cache locality** | Poor between threads | Good within tiles | Excellent sequential |
| **Memory bandwidth** | Good coalescing | Good + cache reuse | Maximum theoretical |
| **Complexity** | Simplest | Moderate | Most complex |

### When to choose each pattern

**Use elementwise when:**
- Simple operations with minimal arithmetic per element
- Maximum parallelism needed for latency hiding
- Scalability across different problem sizes is important

**Use tiled/vectorize when:**
- Cache-sensitive operations that benefit from data reuse
- Balanced performance and maintainability desired
- Automatic optimization (vectorize) is preferred

**Use manual vectorization when:**
- Expert-level control over memory patterns is needed
- Maximum memory bandwidth utilization is critical
- Development complexity is acceptable

## Hardware considerations

Modern GPU architectures include several levels that Mojo abstracts:

**Hardware reality:**
- **Warps**: 32 threads execute in lockstep
- **Streaming Multiprocessors (SMs)**: Multiple warps execute concurrently
- **SIMD units**: Vector processing units within each SM
- **Memory hierarchy**: L1/L2 caches, shared memory, global memory

**Mojo's abstraction benefits:**
- Automatically handles warp alignment and scheduling
- Optimizes memory access patterns transparently
- Manages resource allocation across SMs
- Provides portable performance across GPU vendors

## Performance mental model

Think of GPU programming as managing two complementary types of parallelism:

**Thread-level parallelism:**
- Provides the parallel structure (how many execution units)
- Enables latency hiding through concurrent execution
- Managed by GPU scheduler automatically

**SIMD-level parallelism:**
- Provides vectorization within each thread
- Maximizes arithmetic throughput per thread
- Utilizes vector processing units efficiently

**Optimal performance formula:**
```
Performance = (Sufficient threads for latency hiding) ×
              (Efficient SIMD utilization) ×
              (Optimal memory access patterns)
```

## Scaling considerations

| Problem size | Optimal pattern | Reasoning |
|-------------|----------------|-----------|
| Small (< 1K) | Tiled/Vectorize | Lower launch overhead |
| Medium (1K-1M) | Any pattern | Similar performance |
| Large (> 1M) | Usually Elementwise | Parallelism dominates |

The optimal choice depends on your specific hardware, workload complexity, and development constraints.

## Next steps

With a solid understanding of GPU threading vs SIMD concepts:

- **[📊 Benchmarking](./benchmarking.md)**: Measure and compare actual performance

💡 **Key takeaway**: GPU threads and SIMD operations work together as complementary levels of parallelism. Understanding their relationship allows you to choose the right pattern for your specific performance requirements and constraints.
