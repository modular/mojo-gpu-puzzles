# Puzzle 30: GPU Performance Profiling

> **Beyond Correct Code**
>
> Note: **This part is specific to compatible NVIDIA GPUs**
>
> This chapter introduces **systematic performance analysis** that transforms working GPU code into high-performance code. Unlike previous puzzles that focused on correctness and GPU features, these challenges explore **profiling methodologies** used in production GPU software development.
>
>
> **What you'll learn:**
> - **Professional profiling tools**: NSight Systems and NSight Compute for comprehensive performance analysis
> - **Performance detective work**: Using profiler data to identify bottlenecks and optimization opportunities
> - **Memory system insights**: Understanding how memory access patterns dramatically impact performance
> - **Counter-intuitive discoveries**: Learning when "good" metrics actually indicate performance problems
> - **Evidence-based optimization**: Making optimization decisions based on profiler data, not assumptions
>
> **Why this matters:** Most GPU tutorials teach basic performance concepts, but real-world GPU development requires **systematic profiling methodologies** to identify actual bottlenecks, understand memory system behavior, and make informed optimization decisions. These skills bridge the gap between academic examples and production GPU computing.

## Overview

GPU performance profiling transforms correct code into high-performance code through systematic analysis. This chapter explores professional profiling tools and detective methodologies used in production GPU development.

**Core learning objectives:**
- **Master profiling tool selection** and understand when to use NSight Systems vs NSight Compute
- **Develop performance detective skills** using real profiler output to identify bottlenecks
- **Discover counter-intuitive insights** about GPU memory systems and caching behavior
- **Learn evidence-based optimization** based on profiler data rather than assumptions

## Key concepts

**Professional profiling tools:**
- **[NSight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`)**: System-wide timeline analysis for CPU-GPU coordination and memory transfers
- **[NSight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`)**: Detailed kernel analysis for memory efficiency and compute utilization
- **Systematic methodology**: Evidence-based bottleneck identification and optimization validation

**Key insights you'll discover:**
- **Counter-intuitive behavior**: When high cache hit rates actually indicate poor performance
- **Memory access patterns**: How coalescing dramatically impacts bandwidth utilization
- **Tool-guided optimization**: Using profiler data to make decisions rather than performance assumptions

## Configuration

**Requirements:**
- **NVIDIA GPU**: CUDA-compatible hardware with profiling enabled
- **CUDA Toolkit**: NSight Systems and NSight Compute tools
- **Build setup**: Optimized code with debug info (`--debug-level=full`)

**Methodology:**
1. **System-wide analysis** with NSight Systems to identify major bottlenecks
2. **Kernel deep-dives** with NSight Compute for memory system analysis
3. **Evidence-based conclusions** using profiler data to guide optimization

## Puzzle structure

This chapter contains two interconnected components that build upon each other:

### **[NVIDIA Profiling Basics Tutorial](nvidia_profiling_basics.md)**

Master the essential NVIDIA profiling ecosystem through hands-on examples with actual profiler output.

**You'll learn:**
- NSight Systems for system-wide timeline analysis and bottleneck identification
- NSight Compute for detailed kernel analysis and memory system insights
- Professional profiling workflows and best practices from production GPU development

### **[The Cache Hit Paradox Detective Case](profile_kernels.md)**

Apply profiling skills to solve a performance mystery where three identical vector addition kernels have dramatically different performance.

**The challenge:** Discover why the kernel with the **highest cache hit rates** has the **worst performance** - a counter-intuitive insight that challenges traditional CPU-based performance thinking.

**Detective skills:** Use real NSight Systems and NSight Compute data to understand memory coalescing effects and evidence-based optimization.

## Getting started

**Learning path:**
1. **[Profiling Basics Tutorial](nvidia_profiling_basics.md)** - Master NSight Systems and NSight Compute
2. **[Cache Hit Paradox Detective Case](profile_kernels.md)** - Apply skills to solve performance mysteries

**Prerequisites:**
- GPU memory hierarchies and access patterns
- GPU programming fundamentals (threads, blocks, warps, shared memory)
- Command-line profiling tools experience

**Learning outcome:** Professional-level profiling skills for systematic bottleneck identification and evidence-based optimization used in production GPU development.

This chapter teaches that **systematic profiling reveals truths that intuition misses** - GPU performance optimization requires tool-guided discovery rather than assumptions.

**Additional resources:**
- [NVIDIA CUDA Best Practices Guide - Profiling](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NSight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
