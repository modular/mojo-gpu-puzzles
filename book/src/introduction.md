<p align="center">
  <img src="images/puzzle-mark.svg" alt="Mojo GPU Puzzles Logo" width="150" class="puzzle-image">
</p>

<p align="center">
  <h1 align="center">Mojo🔥 GPU Puzzles</h1>
</p>

<p align="center" class="social-buttons" style="display: flex; justify-content: center; gap: 8px;">
  <a href="https://github.com/modular/mojo-gpu-puzzles">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub Repository">
  </a>
  <a href="https://docs.modular.com/mojo">
    <img src="https://img.shields.io/badge/Powered%20by-Mojo-FF5F1F" alt="Powered by Mojo">
  </a>
  <a href="https://docs.modular.com/max/get-started/#stay-in-touch">
    <img src="https://img.shields.io/badge/Subscribe-Updates-00B5AD?logo=mail.ru" alt="Subscribe for Updates">
  </a>
  <a href="https://forum.modular.com/c/">
    <img src="https://img.shields.io/badge/Modular-Forum-9B59B6?logo=discourse" alt="Modular Forum">
  </a>
  <a href="https://discord.com/channels/1087530497313357884/1098713601386233997">
    <img src="https://img.shields.io/badge/Discord-Join_Chat-5865F2?logo=discord" alt="Discord">
  </a>
</p>

> 🚧 This book is a work in progress! Some sections may be incomplete or subject to change. 🚧

> _"For the things we have to learn before we can do them, we learn by doing them."_
> Aristotle, (Nicomachean Ethics)

Welcome to **Mojo 🔥 GPU Puzzles**, a hands-on guide to mastering GPU programming using [Mojo](https://docs.modular.com/mojo/manual/) 🔥 — the innovative programming language that combines Pythonic syntax with systems-level performance. GPU programming remains one of the most powerful skills in modern computing, driving advances in artificial intelligence, scientific simulation, and high-performance computing.

This book takes a unique approach to teaching GPU programming: learning by solving increasingly challenging puzzles. Rather than traditional textbook learning, you'll immediately start writing real GPU code and seeing the results.

The early chapters of this book are heavily inspired by [GPU Puzzles](https://github.com/srush/GPU-Puzzles), an interactive CUDA learning project by Sasha Rush. This adaptation reimplements these concepts using Mojo's powerful abstractions and performance capabilities, while expanding on advanced topics with Mojo-specific optimizations.

## Why Mojo 🔥 for GPU Programming?

The computing industry has reached a critical inflection point. We can no longer rely on new CPU generations to automatically increase application performance through higher clock speeds. As power and heat constraints have plateaued CPU speeds, hardware manufacturers have shifted toward increasing the number of physical cores. This multi-core revolution has reached its zenith in modern GPUs, which contain thousands of cores operating in parallel. The NVIDIA H100, for example, can run an astonishing 16,896 threads simultaneously in a single clock cycle, with over 270,000 threads queued and ready for execution.

Mojo represents a fresh approach to GPU programming, making this massive parallelism more accessible and productive:

- **Python-like Syntax** with systems programming capabilities that feels familiar to the largest programming community
- **Zero-cost Abstractions** that compile to efficient machine code without sacrificing performance
- **Strong Type System** that catches errors at compile time while maintaining expressiveness
- **Built-in Tensor Support** with hardware-aware optimizations specifically designed for GPU computation
- **Direct Access** to low-level CPU and GPU intrinsics for systems-level programming
- **Cross-Hardware Portability** allowing you to write code that can run on both CPUs and GPUs
- **Ergonomic and Safety Improvements** over traditional C/C++ GPU programming
- **Lower Barrier to Entry** enabling more programmers to harness GPU power effectively

>**Mojo 🔥 aims to fuel innovation by democratizing GPU programming.**
>**By expanding on Python's familiar syntax while adding direct GPU access, Mojo empowers programmers with minimal specialized knowledge to build high-performance, heterogeneous (CPU, GPU-enabled) applications.**

## The GPU Programming Mindset

Effective GPU programming requires a fundamental shift in how we think about computation. Here are some key mental models that will guide your journey:

### From Sequential to Parallel: Eliminating Loops with Threads

In traditional CPU programming, we process data sequentially through loops:

```python
# CPU approach
for i in range(data_size):
    result[i] = process(data[i])
```

With GPUs, we flip this model entirely. Instead of moving sequentially through data, we map thousands of parallel threads directly onto the data:

```mojo
# GPU approach (conceptual)
thread_id = get_global_id()
if thread_id < data_size:
    result[thread_id] = process(data[thread_id])
```

Each thread becomes responsible for computing a single element, eliminating the need for explicit loops. This mental shift—from "stepping through data" to "blanketing data with compute"—is central to GPU programming.

### Fitting a Mesh of Compute Over Data

Imagine your data as a grid, and GPU threads as another grid that overlays it. Your task is to design this "compute mesh" to efficiently cover your data:

- **Threads**: Individual compute units that process single data elements
- **Blocks**: Organized groups of threads that share fast memory
- **Grid**: The entire collection of blocks that covers your dataset

The art of GPU programming lies in crafting this mesh to maximize parallelism while respecting memory and synchronization constraints.

### Data Movement vs. Computation

In GPU programming, data movement is often more expensive than computation:
- Moving data between CPU and GPU is slow
- Moving data between global and shared memory is faster
- Operating on data already in registers or shared memory is extremely fast

This inverts another common assumption in programming: computation is no longer the bottleneck—data movement is.

Through the puzzles in this book, you'll develop an intuitive understanding of these principles, transforming how you approach computational problems.

## What You Will Learn

This book takes you on a journey from first principles to advanced GPU programming techniques. Rather than treating the GPU as a mysterious black box, we'll build your understanding layer by layer—starting with how individual threads operate and culminating in sophisticated parallel algorithms. By mastering both low-level memory management and high-level tensor abstractions, you'll gain the versatility to tackle any GPU programming challenge.

Your learning path includes:

- **GPU Programming Fundamentals**: Thread organization, memory hierarchies, and kernel execution models
- **Dual Implementation Paths**: Beginning with raw memory approaches using pointers, then transitioning to LayoutTensor abstractions
- **Memory Management**: Working with global, shared, and thread-local memory for optimal performance
- **Low-level to High-level Progression**: Understanding the foundation with UnsafePointer before leveraging LayoutTensor's elegant abstractions
- **Layout Tensors**: Mastering Mojo's powerful tensor abstractions for simplified, efficient GPU computation
- **Parallel Algorithms**: Implementing and optimizing parallel reductions, convolutions, matrix operations, and more
- **Performance Optimization**: Advanced techniques for memory coalescing, tiling, bank conflict avoidance, and minimizing thread divergence
- **Real-world Applications**: Applying these concepts to machine learning, signal processing, and computational tasks

The book uniquely challenges the status quo approach by first building understanding with low-level memory manipulation, then gradually transitioning to Mojo's powerful LayoutTensor abstractions. This gives you both deep understanding of GPU memory patterns and practical knowledge of modern tensor-based approaches.

## 🏆 Prizes and Rewards 🎉

Have you completed the available puzzles? We're giving away free sticker packs to celebrate your achievement!

To claim your free stickers:

1. Fork the GitHub repository [https://github.com/modular/mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles)
2. Add your solutions to the available puzzles
3. Submit your solutions through [this form](https://forms.gle/bchQpB3GanHMNY3x9) and we'll send you exclusive Modular stickers!

_Note: More puzzles are being added regularly - complete the currently available ones to claim your reward!_
