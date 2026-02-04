# Puzzle 21: Embedding Op

> ## Memory access patterns and performance
>
> We're continuing Part IV with a focus on **memory-bound operations** and **GPU memory access optimization**.
>
> Building on [Puzzle 20](../puzzle_20/puzzle_20.md), you'll now explore how different kernel implementations of the same operation can have dramatically different performance characteristics. You'll learn:
> - How GPU memory coalescing affects performance
> - Why grid configuration matters for memory-bound operations
> - How to design kernels with optimal memory access patterns
> - The performance implications of different threading strategies
>
> This puzzle demonstrates that **how you access memory** can be more important than **what computation you perform**.

## Overview

In this puzzle, you'll implement two different GPU kernels for embedding operations - a fundamental component in neural networks. While both kernels produce identical results, they use different memory access patterns that lead to significant performance differences.

You'll compare:
- **1D coalesced kernel**: Optimized for memory bandwidth with consecutive memory accesses
- **2D non-coalesced kernel**: Suboptimal memory access pattern for comparison

This comparison teaches the critical importance of memory coalescing in GPU kernel performance.

## Background: Embedding operations

An embedding operation converts discrete token indices into dense vector representations:

```python
# Input: token indices
indices = [[1, 5, 2], [7, 1, 9]]           # Shape: [batch_size, seq_len]

# Embedding table (learned parameters)
embedding_table = [                        # Shape: [vocab_size, embed_dim]
    [0.1, 0.2, 0.3, 0.4],  # Token 0
    [0.5, 0.6, 0.7, 0.8],  # Token 1
    [0.9, 1.0, 1.1, 1.2],  # Token 2
    # ... more tokens
]

# Output: embedded vectors
output[0,0] = embedding_table[1]  # [0.5, 0.6, 0.7, 0.8]
output[0,1] = embedding_table[5]  # lookup token 5's embedding
output[0,2] = embedding_table[2]  # [0.9, 1.0, 1.1, 1.2]
# ... and so on
```

This operation is **memory-bound** - performance depends on how efficiently you can read from the embedding table and write to the output tensor.

## Learning path

This puzzle is structured in two parts to build your understanding systematically:

### **[Simple embedding kernel](./simple_embedding_kernel.md)**

Start here to implement the actual puzzle code and understand the kernel implementations.

**What you'll do:**
- Complete two different GPU embedding kernels (1D coalesced vs 2D non-coalesced)
- Learn fundamental memory access patterns for GPU programming
- See the same algorithm implemented with different threading strategies
- Understand custom operation registration in Mojo

### **[Performance comparison](./performance.md)**

Deep dive into why the kernels perform differently and the theory behind memory coalescing.

**What you'll learn:**
- Why memory coalescing matters for GPU performance
- How thread organization affects memory bandwidth utilization
- Real-world implications for neural network optimization
- Optimization strategies for memory-bound operations

## Getting started

Ready to explore GPU memory optimization? Start with the **[Simple embedding kernel](./simple_embedding_kernel.md)** to implement the code, then move to **[Performance comparison](./performance.md)** to understand the performance implications.

💡 **Success tip:** Pay attention to how the different grid configurations (1D vs 2D) affect memory access patterns - this insight applies to many GPU programming scenarios beyond embeddings.
