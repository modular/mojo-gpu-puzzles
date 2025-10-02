<p align="center">
  <img src="puzzles_images/puzzle-mark.svg" alt="Mojo GPU Puzzles Logo" width="150" class="puzzle-image">
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

> _"For the things we have to learn before we can do them, we learn by doing them."_
> Aristotle, (Nicomachean Ethics)

Welcome to **Mojo 🔥 GPU Puzzles, Edition 1**, a hands-on guide to GPU programming using [Mojo](https://docs.modular.com/mojo/manual/) 🔥, the programming language that combines Python syntax with systems-level performance.

## Why GPU programming?

GPU programming has evolved from a specialized skill into fundamental infrastructure for modern computing. From large language models processing billions of parameters to computer vision systems analyzing real-time video streams, GPU acceleration drives the computational breakthroughs we see today. Scientific advances in climate modeling, drug discovery, and quantum simulation depend on the massive parallel processing capabilities that GPUs uniquely provide. Financial institutions rely on GPU computing for real-time risk analysis and algorithmic trading, while autonomous vehicles process sensor data through GPU-accelerated neural networks for critical decision-making.

The economic implications are substantial. Organizations that effectively leverage GPU computing achieve significant competitive advantages: accelerated development cycles, reduced computational costs, and the capacity to address previously intractable computational challenges. In an era where computational capability directly correlates with business value, GPU programming skills represent a strategic differentiator for engineers, researchers, and organizations.

## Why Mojo🔥 for GPU programming?

The computing industry has reached a critical point. CPU performance no longer increases through higher clock speeds due to power and heat constraints. Hardware manufacturers have shifted toward increasing physical cores. This multi-core approach reaches its peak in modern GPUs, which contain thousands of cores operating in parallel. The NVIDIA H100, for example, can run 16,896 threads simultaneously in a single clock cycle, with over 270,000 threads queued for execution.

Mojo provides a practical approach to GPU programming, making this parallelism more accessible:

- **Python-like Syntax** with systems programming capabilities
- **Zero-cost Abstractions** that compile to efficient machine code
- **Strong Type System** that catches errors at compile time
- **Built-in Tensor Support** with hardware-aware optimizations for GPU computation
- **Direct Access** to low-level CPU and GPU intrinsics
- **Cross-Hardware Portability** for code that runs on both CPUs and GPUs
- **Improved Safety** over traditional C/C++ GPU programming
- **Lower Barrier to Entry** for more programmers to access GPU power

> **Mojo🔥 aims to fuel innovation by democratizing GPU programming.**
>**By expanding on Python's familiar syntax while adding direct GPU access, Mojo allows programmers with minimal specialized knowledge to build high-performance, heterogeneous (CPU, GPU-enabled) applications.**

## Why learn through puzzles?

Most GPU programming resources start with extensive theory before practical implementation. This can overwhelm newcomers with abstract concepts that only become clear through direct application.

This book uses a different approach: immediate engagement with practical problems that progressively introduce concepts through guided discovery.

**Advantages of puzzle-based learning:**

- **Direct experience**: Immediate execution on GPU hardware provides concrete feedback
- **Incremental complexity**: Each challenge builds on previously established concepts
- **Applied focus**: Problems mirror real-world computational scenarios
- **Diagnostic skills**: Systematic debugging practice develops troubleshooting capabilities
- **Knowledge retention**: Active problem-solving reinforces understanding more effectively than passive consumption

The methodology emphasizes discovery over memorization. Concepts emerge naturally through experimentation, creating deeper understanding and practical competency.

> **Acknowledgement**: The Part I and III of this book are heavily inspired by [GPU Puzzles](https://github.com/srush/GPU-Puzzles), an interactive
NVIDIA GPU learning project. This adaptation reimplements these concepts using Mojo's abstractions and performance capabilities, while
expanding on advanced topics with Mojo-specific optimizations.

## The GPU programming mindset

Effective GPU programming requires a fundamental shift in how we think about computation. Here are some key mental models that will guide your journey:

### From sequential to parallel: Eliminating loops with threads

In traditional CPU programming, we process data sequentially through loops:

```python
# CPU approach
for i in range(data_size):
    result[i] = process(data[i])
```

GPU programming inverts this paradigm completely. Rather than iterating sequentially through data, we assign thousands of parallel threads to process data elements simultaneously:

```mojo
# GPU approach (conceptual)
thread_id = get_global_id()
if thread_id < data_size:
    result[thread_id] = process(data[thread_id])
```

Each thread handles a single data element, replacing explicit iteration with massive parallelism. This fundamental reframing—from sequential processing to concurrent execution across all data elements—represents the core conceptual shift in GPU programming.

### Fitting a mesh of compute over data

Consider your data as a structured grid, with GPU threads forming a corresponding computational grid that maps onto it. Effective GPU programming involves designing this thread organization to optimally cover your data space:

- **Threads**: Individual processing units, each responsible for specific data elements
- **Blocks**: Coordinated thread groups with shared memory access and synchronization capabilities
- **Grid**: The complete thread hierarchy spanning the entire computational problem

Successful GPU programming requires balancing this thread organization to maximize parallel efficiency while managing memory access patterns and synchronization requirements.

### Data movement vs. computation

In GPU programming, data movement is often more expensive than computation:

- Moving data between CPU and GPU is slow
- Moving data between global and shared memory is faster
- Operating on data already in registers or shared memory is extremely fast

This inverts another common assumption in programming: computation is no longer the bottleneck—data movement is.

Through the puzzles in this book, you'll develop an intuitive understanding of these principles, transforming how you approach computational problems.

## What you will learn

This book takes you on a journey from first principles to advanced GPU programming techniques. Rather than treating the GPU as a mysterious black box, the content builds understanding layer by layer—starting with how individual threads operate and culminating in sophisticated parallel algorithms. Learning both low-level memory management and high-level tensor abstractions provides the versatility to tackle any GPU programming challenge.

### Your current learning path

| Essential Skill | Status | Puzzles |
|-----------------|--------|---------|
| Thread/Block basics | ✅ **Available** | Part I (1-8) |
| Debugging GPU Programs | ✅ **Available** | Part II (9-10) |
| Core algorithms | ✅ **Available** | Part III (11-16) |
| MAX Graph integration | ✅ **Available** | Part IV (17-19) |
| PyTorch integration | ✅ **Available** | Part V (20-22) |
| Functional patterns & benchmarking | ✅ **Available** | Part VI (23) |
| Warp programming | ✅ **Available** | Part VII (24-26) |
| Block-level programming | ✅ **Available** | Part VIII (27) |
| Advanced memory operations | ✅ **Available** | Part IX (28-29) |
| Performance analysis | ✅ **Available** | Part X (30-32) |
| Modern GPU features | ✅ **Available** | Part XI (33-34) |

### Detailed learning objectives

**Part I: GPU fundamentals (Puzzles 1-8) ✅**

- Learn thread indexing and block organization
- Understand memory access patterns and guards
- Work with both raw pointers and LayoutTensor abstractions
- Learn shared memory basics for inter-thread communication

**Part II: Debugging GPU programs (Puzzles 9-10) ✅**

- Learn GPU debugger and debugging techniques
- Learn to use sanitizers for catching memory errors and race conditions
- Develop systematic approaches to identifying and fixing GPU bugs
- Build confidence for tackling complex GPU programming challenges

> **Note**: Debugging puzzles require `pixi` for access to NVIDIA's GPU debugging tools. These puzzles work exclusively on NVIDIA GPUs with CUDA support.

**Part III: GPU algorithms (Puzzles 11-16) ✅**

- Implement parallel reductions and pooling operations
- Build efficient convolution kernels
- Learn prefix sum (scan) algorithms
- Optimize matrix multiplication with tiling strategies

**Part IV: MAX Graph integration (Puzzles 17-19) ✅**

- Create custom MAX Graph operations
- Interface GPU kernels with Python code
- Build production-ready operations like softmax and attention

**Part V: PyTorch integration (Puzzles 20-22) ✅**

- Bridge Mojo GPU kernels with PyTorch tensors
- Use CustomOpLibrary for seamless tensor marshalling
- Integrate with torch.compile for optimized execution
- Learn kernel fusion and custom backward passes

**Part VI: Mojo functional patterns & benchmarking (Puzzle 23) ✅**

- Learn functional patterns: elementwise, tiled processing, vectorization
- Learn systematic performance optimization and trade-offs
- Develop quantitative benchmarking skills for performance analysis
- Understand GPU threading vs SIMD execution hierarchies

**Part VII: Warp-level programming (Puzzles 24-26) ✅**

- Learn warp fundamentals and SIMT execution models
- Learn essential warp operations: sum, shuffle_down, broadcast
- Implement advanced patterns with shuffle_xor and prefix_sum
- Combine warp programming with functional patterns effectively

**Part VIII: Block-level programming (Puzzle 27) ✅**

- Learn block-wide reductions with `block.sum()` and `block.max()`
- Learn block-level prefix sum patterns and communication
- Implement efficient block.broadcast() for intra-block coordination

**Part IX: Advanced memory systems (Puzzles 28-29) ✅**

- Achieve optimal memory coalescing patterns
- Use async memory operations for overlapping compute with latency hiding
- Learn memory fences and synchronization primitives
- Learn prefetching and cache optimization strategies

**Part X: Performance analysis & optimization (Puzzles 30-32) ✅**

- Profile GPU kernels and identify bottlenecks
- Optimize occupancy and resource utilization
- Eliminate shared memory bank conflicts

**Part XI: Advanced GPU features (Puzzles 33-34) ✅**

- Program tensor cores for AI workloads
- Learn cluster programming in modern GPUs

The book uniquely challenges the status quo approach by first building understanding with low-level memory manipulation, then gradually transitioning to Mojo's LayoutTensor abstractions. This provides both deep understanding of GPU memory patterns and practical knowledge of modern tensor-based approaches.

## Ready to get started?

You now understand why GPU programming matters, why Mojo is suitable for this work, and how puzzle-based learning functions. You're prepared to begin.

**Next step**: Head to [How to Use This Book](howto.md) for setup instructions, system requirements, and guidance on running your first puzzle.
