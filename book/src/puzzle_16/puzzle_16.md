# Puzzle 16: Matrix Multiplication (MatMul)

## Overview

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. Given two matrices \\(A\\) and \\(B\\), we want to compute their product \\(C = A \\times B.\\)

For matrices \\(A_{m\\times k}\\) and \\(B_{k\\times n}\\), each element of the result \\(C_{m\\times n}\\) is computed as:

\\[\Large C_{ij} = \sum_{l=0}^{k-1} A_{il} \\cdot B_{lj} \\]

<img src="./media/16-w.png" alt="Matmul visualization" class="light-mode-img">
<img src="./media/16-b.png" alt="Matmul visualization" class="dark-mode-img">

This puzzle explores different approaches to implementing matrix multiplication on GPUs, each with its own performance characteristics:

- [Naive Version](./naïve.md)
  The straightforward implementation where each thread computes one element of the output matrix. While simple to understand, this approach makes many redundant memory accesses.

- [Shared Memory Version](./shared_memory.md)
  Improves performance by loading blocks of input matrices into fast shared memory, reducing global memory accesses. Each thread still computes one output element but reads from shared memory.

- [Tiled Version](./tiled.md)
  Further optimizes by dividing the computation into tiles, allowing threads to cooperate on loading and computing blocks of the output matrix. This approach better utilizes memory hierarchy and thread cooperation.

Each version builds upon the previous one, introducing new optimization techniques common in GPU programming. You'll learn how different memory access patterns and thread cooperation strategies affect performance.

The progression illustrates a common pattern in GPU optimization:
1. Start with a correct but naive implementation
2. Reduce global memory access with shared memory
3. Improve data locality and thread cooperation with tiling
4. Use high-level abstractions while maintaining performance

Choose a version to begin your matrix multiplication journey!
