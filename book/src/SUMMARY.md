# Summary

# Getting Started
- [🔥 Introduction](./introduction.md)
- [🧭 Puzzles Usage Guide](./howto.md)

# Part I: GPU Fundamentals
- [Puzzle 1: Map](./puzzle_01/puzzle_01.md)
  - [🔰 Raw Memory Approach](./puzzle_01/raw.md)
  - [💡 Preview: Modern Approach with LayoutTensor](./puzzle_01/layout_tensor_preview.md)
- [Puzzle 2: Zip](./puzzle_02/puzzle_02.md)
- [Puzzle 3: Guards](./puzzle_03/puzzle_03.md)
- [Puzzle 4: 2D Map](./puzzle_04/puzzle_04.md)
  - [🔰 Raw Memory Approach](./puzzle_04/raw.md)
  - [📚 Learn about LayoutTensor](./puzzle_04/introduction_layout_tensor.md)
  - [🚀 Modern 2D Operations](./puzzle_04/layout_tensor.md)
- [Puzzle 5: Broadcast](./puzzle_05/puzzle_05.md)
  - [🔰 Raw Memory Approach](./puzzle_05/raw.md)
  - [📐 LayoutTensor Version](./puzzle_05/layout_tensor.md)
- [Puzzle 6: Blocks](./puzzle_06/puzzle_06.md)
- [Puzzle 7: 2D Blocks](./puzzle_07/puzzle_07.md)
  - [🔰 Raw Memory Approach](./puzzle_07/raw.md)
  - [📐 LayoutTensor Version](./puzzle_07/layout_tensor.md)
- [Puzzle 8: Shared Memory](./puzzle_08/puzzle_08.md)
  - [🔰 Raw Memory Approach](./puzzle_08/raw.md)
  - [📐 LayoutTensor Version](./puzzle_08/layout_tensor.md)

# Part II: 🐞 Debugging GPU Programs
- [Puzzle 9: GPU Debugging Workflow](./puzzle_09/puzzle_09.md)
  - [📚 Mojo GPU Debugging Essentials](./puzzle_09/essentials.md)
  - [🧐 Detective Work: First Case](./puzzle_09/first_case.md)
  - [🔍 Detective Work: Second Case](./puzzle_09/second_case.md)
  - [🕵 Detective Work: Third Case](./puzzle_09/third_case.md)
- [Puzzle 10: Memory Error Detection & Race Conditions with Sanitizers](./puzzle_10/puzzle_10.md)
  - [👮🏼‍♂️ Detect Memory Violation](./puzzle_10/memcheck.md)
  - [🏁 Debug Race Condition](./puzzle_10/racecheck.md)

# Part III: 🧮 GPU Algorithms
- [Puzzle 11: Pooling](./puzzle_11/puzzle_11.md)
  - [🔰 Raw Memory Approach](./puzzle_11/raw.md)
  - [📐 LayoutTensor Version](./puzzle_11/layout_tensor.md)
- [Puzzle 12: Dot Product](./puzzle_12/puzzle_12.md)
  - [🔰 Raw Memory Approach](./puzzle_12/raw.md)
  - [📐 LayoutTensor Version](./puzzle_12/layout_tensor.md)
- [Puzzle 13: 1D Convolution](./puzzle_13/puzzle_13.md)
  - [🔰 Simple Version](./puzzle_13/simple.md)
  - [⭐ Block Boundary Version](./puzzle_13/block_boundary.md)
- [Puzzle 14: Prefix Sum](./puzzle_14/puzzle_14.md)
  - [🔰 Simple Version](./puzzle_14/simple.md)
  - [⭐ Complete Version](./puzzle_14/complete.md)
- [Puzzle 15: Axis Sum](./puzzle_15/puzzle_15.md)
- [Puzzle 16: Matrix Multiplication (MatMul)](./puzzle_16/puzzle_16.md)
    - [🔰 Naïve Version with Global Memory](./puzzle_16/naïve.md)
    - [📚 Learn about Roofline Model](./puzzle_16/roofline.md)
    - [🤝 Shared Memory Version](./puzzle_16/shared_memory.md)
    - [📐 Tiled Version](./puzzle_16/tiled.md)

# Part IV: 🐍 Interfacing with Python via MAX Graph Custom Ops
- [Puzzle 17: 1D Convolution Op](./puzzle_17/puzzle_17.md)
- [Puzzle 18: Softmax Op](./puzzle_18/puzzle_18.md)
- [Puzzle 19: Attention Op](./puzzle_19/puzzle_19.md)
- [🎯 Bonus Challenges](./bonuses/part4.md)

# Part V: 🔥 PyTorch Custom Ops Integration
- [Puzzle 20: 1D Convolution Op](./puzzle_20/puzzle_20.md)
- [Puzzle 21: Embedding Op](./puzzle_21/puzzle_21.md)
  - [🔰 Coaleasced vs non-Coaleasced Kernel](./puzzle_21/simple_embedding_kernel.md)
  - [📊 Performance Comparison](./puzzle_21/performance.md)
- [Puzzle 22: Kernel Fusion and Custom Backward Pass](./puzzle_22/puzzle_22.md)
  - [⚛️ Fused vs Unfused Kernels](./puzzle_22/forward_pass.md)
  - [⛓️ Autograd Integration & Backward Pass](./puzzle_22/backward_pass.md)

# Part VI: 🌊 Mojo Functional Patterns and Benchmarking
- [Puzzle 23: GPU Functional Programming Patterns](./puzzle_23/puzzle_23.md)
  - [elementwise - Basic GPU Functional Operations](./puzzle_23/elementwise.md)
  - [tile - Memory-Efficient Tiled Processing](./puzzle_23/tile.md)
  - [vectorize - SIMD Control](./puzzle_23/vectorize.md)
  - [🧠 GPU Threading vs SIMD Concepts](./puzzle_23/gpu-thread-vs-simd.md)
  - [📊 Benchmarking in Mojo](./puzzle_23/benchmarking.md)

# Part VII: ⚡ Warp-Level Programming
- [Puzzle 24: Warp Fundamentals](./puzzle_24/puzzle_24.md)
  - [🧠 Warp lanes & SIMT execution](./puzzle_24/warp_simt.md)
  - [🔰 warp.sum() Essentials](./puzzle_24/warp_sum.md)
  - [🤔 When to Use Warp Programming](./puzzle_24/warp_extra.md)
- [Puzzle 25: Warp Communication](./puzzle_25/puzzle_25.md)
  - [⬇️ warp.shuffle_down()](./puzzle_25/warp_shuffle_down.md)
  - [📢 warp.broadcast()](./puzzle_25/warp_broadcast.md)
- [Puzzle 26: Advanced Warp Patterns](./puzzle_26/puzzle_26.md)
  - [🦋 warp.shuffle_xor() Butterfly Networks](./puzzle_26/warp_shuffle_xor.md)
  - [🔢 warp.prefix_sum() Scan Operations](./puzzle_26/warp_prefix_sum.md)

# Part VIII: 🧱 Block-Level Programming
- [Puzzle 27: Block-Wide Patterns](./puzzle_27/puzzle_27.md)
  - [🔰 block.sum() Essentials](./puzzle_27/block_sum.md)
  - [📈 block.prefix_sum() Parallel Histogram Binning](./puzzle_27/block_prefix_sum.md)
  - [📡 block.broadcast() Vector Normalization](./puzzle_27/block_broadcast.md)

# Part IX: 🧠 Advanced Memory Systems
- [Puzzle 28: Async Memory Operations & Copy Overlap](./puzzle_28/puzzle_28.md)
- [Puzzle 29: GPU Synchronization Primitives](./puzzle_29/puzzle_29.md)
  - [📶 Multi-Stage Pipeline Coordination](./puzzle_29/barrier.md)
  - [Double-Buffered Stencil Computation](./puzzle_29/memory_barrier.md)

# Part X: 📊 Performance Analysis & Optimization
- [Puzzle 30: GPU Profiling](./puzzle_30/puzzle_30.md)
  - [📚 NVIDIA Profiling Basics](./puzzle_30/nvidia_profiling_basics.md)
  - [🕵 The Cache Hit Paradox](./puzzle_30/profile_kernels.md)
- [Puzzle 31: Occupancy Optimization]()
- [Puzzle 32: Bank Conflicts]()
  - [📚 Understanding Shared Memory Banks]()
  - [Conflict-Free Patterns]()

# Part XI: 🚀 Advanced GPU Features
- [Puzzle 33: Tensor Core Operations]()
- [Puzzle 34: Essential TMA Operations (H100+)]()
- [Puzzle 35: GPU Cluster Programming (SM90+)]()
  - [Thread Block Clusters]()
  - [cluster_sync() Coordination]()
  - [Advanced Cluster Patterns]()
