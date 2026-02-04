# Puzzle 34: GPU Cluster Programming (SM90+)

## Introduction

> **Hardware requirement: ⚠️ NVIDIA SM90+ Only**
>
> This puzzle requires **NVIDIA Hopper architecture** (H100, H200) or newer GPUs with SM90+ compute capability. The cluster programming APIs are hardware-accelerated and will raise errors on unsupported hardware. If you're unsure about the underlying architecture, run `pixi run gpu-specs` and must have at least `Compute Cap: 9.0` (see [GPU profiling basics](../puzzle_30/nvidia_profiling_basics.md) for hardware identification)

Building on your journey from **[warp-level programming (Puzzles 24-26)](../puzzle_24/puzzle_24.md)** through **[block-level programming (Puzzle 27)](../puzzle_27/puzzle_27.md)**, you'll now learn **cluster-level programming** - coordinating multiple thread blocks to solve problems that exceed single-block capabilities.

## What are thread block clusters?

Thread Block Clusters are a revolutionary SM90+ feature that enable **multiple thread blocks to cooperate** on a single computational task with hardware-accelerated synchronization and communication primitives.

**Key capabilities:**
- **Inter-block synchronization**: Coordinate multiple blocks with [`cluster_sync`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync), [`cluster_arrive`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), [`cluster_wait`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
- **Block identification**: Use [`block_rank_in_cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster) for unique block coordination
- **Efficient coordination**: [`elect_one_sync`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for optimized warp-level cooperation
- **Advanced patterns**: [`cluster_mask_base`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_mask_base) for selective block coordination


## The cluster programming model

### Traditional GPU programming hierarchy:

```
Grid (Multiple Blocks)
├── Block (Multiple Warps) - barrier() synchronization
    ├── Warp (32 Threads) - SIMT lockstep execution
    │   ├── Lane 0  ─┐
    │   ├── Lane 1   │ All execute same instruction
    │   ├── Lane 2   │ at same time (SIMT)
    │   │   ...      │ warp.sum(), warp.broadcast()
    │   └── Lane 31 ─┘
        └── Thread (SIMD operations within each thread)
```

### **New: Cluster programming hierarchy:**
```
Grid (Multiple Clusters)
├── 🆕 Cluster (Multiple Blocks) - cluster_sync(), cluster_arrive()
    ├── Block (Multiple Warps) - barrier() synchronization
        ├── Warp (32 Threads) - SIMT lockstep execution
        │   ├── Lane 0  ─┐
        │   ├── Lane 1   │ All execute same instruction
        │   ├── Lane 2   │ at same time (SIMT)
        │   │   ...      │ warp.sum(), warp.broadcast()
        │   └── Lane 31 ─┘
            └── Thread (SIMD operations within each thread)
```

**Execution Model Details:**
- **Thread Level**: [SIMD operations](../puzzle_23/gpu-thread-vs-simd.md) within individual threads
- **Warp Level**: [SIMT execution](../puzzle_24/warp_simt.md) - 32 threads in lockstep coordination
- **Block Level**: [Multi-warp coordination](../puzzle_27/puzzle_27.md) with shared memory and barriers
- **🆕 Cluster Level**: Multi-block coordination with SM90+ cluster APIs

## Learning progression

This puzzle follows a carefully designed **3-part progression** that builds your cluster programming expertise:

### **[🔰 Multi-Block Coordination Basics](./cluster_coordination_basics.md)**

**Focus**: Understanding fundamental cluster synchronization patterns

Learn how multiple thread blocks coordinate their execution using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) and [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) for basic inter-block communication and data distribution.

**Key APIs**: [`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster), [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)

---

### **[📊 Cluster-Wide Collective Operations](./cluster_collective_ops.md)**

**Focus**: Extending block-level patterns to cluster scale

Learn cluster-wide reductions and collective operations that extend familiar `block.sum()` concepts to coordinate across multiple thread blocks for large-scale computations.

**Key APIs**: [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync), [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient cluster coordination

---

### **[🚀 Advanced Cluster Algorithms](./advanced_cluster_patterns.md)**

**Focus**: Production-ready multi-level coordination patterns

Implement sophisticated algorithms combining warp-level, block-level, and cluster-level coordination for maximum GPU utilization and complex computational workflows.

**Key APIs**: [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync), [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), advanced coordination patterns

## Why cluster programming matters

**Problem Scale**: Modern AI and scientific workloads often require computations that exceed single thread block capabilities:
- **Large matrix operations** requiring inter-block coordination (like [matrix multiplication from Puzzle 16](../puzzle_16/puzzle_16.md))
- **Multi-stage algorithms** with [producer-consumer dependencies from Puzzle 29](../puzzle_29/barrier.md)
- **Global statistics** across datasets larger than [shared memory from Puzzle 8](../puzzle_08/puzzle_08.md)
- **Advanced stencil computations** requiring neighbor block communication

**Hardware Evolution**: As GPUs gain more compute units (see [GPU architecture profiling in Puzzle 30](../puzzle_30/nvidia_profiling_basics.md)), **cluster programming becomes essential** for utilizing next-generation hardware efficiently.

## Educational value

By completing this puzzle, you'll have learned the complete **GPU programming hierarchy**:

- **Thread-level**: [Individual computation units with SIMD operations](../puzzle_23/gpu-thread-vs-simd.md)
- **[Warp-level](../puzzle_24/puzzle_24.md)**: [32-thread SIMT coordination](../puzzle_24/warp_simt.md) (Puzzles 24-26)
- **[Block-level](../puzzle_27/puzzle_27.md)**: [Multi-warp coordination with shared memory](../puzzle_27/block_sum.md) (Puzzle 27)
- **🆕 Cluster-level**: Multi-block coordination (Puzzle 34)
- **Grid-level**: Independent block execution across [multiple streaming multiprocessors](../puzzle_30/profile_kernels.md)

This progression prepares you for **next-generation GPU programming** and **large-scale parallel computing** challenges, building on the [performance optimization techniques from Puzzles 30-32](../puzzle_30/puzzle_30.md).

## Getting started

**Prerequisites**:
- Complete understanding of [block-level programming (Puzzle 27)](../puzzle_27/puzzle_27.md)
- Experience with [warp-level programming (Puzzles 24-26)](../puzzle_24/puzzle_24.md)
- Familiarity with GPU memory hierarchy from [shared memory concepts (Puzzle 8)](../puzzle_08/puzzle_08.md)
- Understanding of [GPU synchronization from barriers (Puzzle 29)](../puzzle_29/puzzle_29.md)
- Access to NVIDIA SM90+ hardware or compatible environment

**Recommended approach**: Follow the 3-part progression sequentially, as each part builds essential concepts for the next level of complexity.

**Hardware note**: If running on non-SM90+ hardware, the puzzles serve as **educational examples** of cluster programming concepts and API usage patterns.

Ready to learn the future of GPU programming? Start with **[Multi-Block Coordination Basics](./cluster_coordination_basics.md)** to learn fundamental cluster synchronization patterns!
