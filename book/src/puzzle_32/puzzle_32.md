# Puzzle 32: Bank Conflicts

## Why this puzzle matters

**Completing the performance trilogy:** You've learned GPU profiling tools in [Puzzle 30](../puzzle_30/puzzle_30.md) and understood occupancy optimization in [Puzzle 31](../puzzle_31/puzzle_31.md). Now you're ready for the final piece of the performance optimization puzzle: **shared memory efficiency**.

**The hidden performance trap:** You can write GPU kernels with perfect occupancy, optimal global memory coalescing, and identical mathematical operations - yet still experience dramatic performance differences due to **how threads access shared memory**. Bank conflicts represent one of the most subtle but impactful performance pitfalls in GPU programming.

**The learning journey:**

- **Puzzle 30** taught you to **measure and diagnose** performance with NSight profiling
- **Puzzle 31** taught you to **predict and control** resource usage through occupancy analysis
- **Puzzle 32** teaches you to **optimize shared memory access patterns** for maximum efficiency

**Why this matters beyond GPU programming:** The principles of memory banking, conflict detection, and systematic access pattern optimization apply across many parallel computing systems - from CPU cache hierarchies to distributed memory architectures.

> **Note: This puzzle is specific to NVIDIA GPUs**
>
> Bank conflict analysis uses NVIDIA's 32-bank shared memory architecture and NSight Compute profiling tools. While the optimization principles apply broadly, the specific techniques and measurements are NVIDIA CUDA-focused.

## Overview

**Shared memory bank conflicts** occur when multiple threads in a warp simultaneously access different addresses within the same memory bank, forcing the hardware to serialize these accesses. This can transform what should be a single-cycle memory operation into multiple cycles of serialized access.

**What you'll discover:**

- How GPU shared memory banking works at the hardware level
- Why identical kernels can have vastly different shared memory efficiency
- How to predict and measure bank conflicts before they impact performance
- Professional optimization strategies for designing conflict-free algorithms

**The detective methodology:** This puzzle follows the same evidence-based approach as previous performance puzzles - you'll use profiling tools to uncover hidden inefficiencies, then apply systematic optimization principles to eliminate them.

## Key concepts

**Shared memory architecture fundamentals:**

- **32-bank design**: NVIDIA GPUs organize shared memory into 32 independent banks
- **Conflict types**: No conflict (optimal), N-way conflicts (serialized), broadcast (optimized)
- **Access pattern mathematics**: Bank assignment formulas and conflict prediction
- **Performance impact**: From optimal 1-cycle access to worst-case 32-cycle serialization

**Professional optimization skills:**

- **Pattern analysis**: Mathematical prediction of banking behavior
- **Profiling methodology**: NSight Compute metrics for conflict measurement
- **Design principles**: Conflict-free algorithm patterns and prevention strategies
- **Performance validation**: Evidence-based optimization using systematic measurement

## Puzzle structure

This puzzle contains two complementary sections that build your expertise progressively:

### **[📚 Understanding Shared Memory Banks](./shared_memory_bank.md)**

Learn the theoretical foundations of GPU shared memory banking through clear explanations and practical examples.

**You'll learn:**

- How NVIDIA's 32-bank architecture enables parallel access
- The mathematics of bank assignment and conflict prediction
- Types of conflicts and their performance implications
- Connection to previous concepts (warp execution, occupancy, profiling)

**Key insight:** Understanding the hardware enables you to predict performance before writing code.

### **[Conflict-Free Patterns](./conflict_free_patterns.md)**

Apply your banking knowledge to solve a performance mystery using professional profiling techniques.

**The detective challenge:** Two kernels compute identical results but have dramatically different shared memory access efficiency. Use NSight Compute to uncover why one kernel experiences systematic bank conflicts while the other achieves optimal performance.

**Skills developed:** Pattern analysis, conflict measurement, systematic optimization, and evidence-based performance improvement.

## Getting started

**Learning path:**

1. **[Understanding Shared Memory Banks](./shared_memory_bank.md)** - Build theoretical foundation
2. **[Conflict-Free Patterns](./conflict_free_patterns.md)** - Apply detective skills to real optimization

**Prerequisites:**

- GPU profiling experience from [Puzzle 30](../puzzle_30/puzzle_30.md)
- Resource optimization understanding from [Puzzle 31](../puzzle_31/puzzle_31.md)
- Shared memory programming experience from [Puzzle 8](../puzzle_08/puzzle_08.md) and [Puzzle 16](../puzzle_16/puzzle_16.md)

**Hardware requirements:**

- NVIDIA GPU with CUDA toolkit
- NSight Compute profiling tools
- The dependencies such as profiling are managed by `pixi`
- [Compatible GPU architecture](https://docs.modular.com/max/packages/#gpu-compatibility)

## The optimization impact

**When bank conflicts matter most:**

- **Matrix multiplication** with shared memory tiling
- **Stencil computations** using shared memory caching
- **Parallel reductions** with stride-based memory patterns

**Professional development value:**

- **Systematic optimization**: Evidence-based performance improvement methodology
- **Hardware awareness**: Understanding how software maps to hardware constraints
- **Pattern recognition**: Identifying problematic access patterns in algorithm design

**Learning outcome:** Complete your GPU performance optimization toolkit with the ability to design, measure, and optimize shared memory access patterns - the final piece for professional-level GPU programming expertise.

This puzzle demonstrates that **optimal GPU performance requires understanding hardware at multiple levels** - from global memory coalescing through occupancy management to shared memory banking efficiency.
