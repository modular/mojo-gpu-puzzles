# Puzzle 22: Kernel Fusion and Custom Backward Pass

> ## Kernel fusion and autograd integration
>
> We're continuing Part IV with a focus on **kernel fusion** and **autograd integration**.
>
> Building on [Puzzle 21](../puzzle_21/puzzle_21.md), you'll now explore how to combine multiple operations into a single efficient kernel and integrate it with PyTorch's autograd system. You'll learn:
>
> - How kernel fusion improves performance in both forward and backward passes
> - Why custom backward passes are crucial for fused operations
> - How to design fused kernels with proper gradient flow
> - The performance implications of different fusion strategies
>
> This puzzle demonstrates that **how you combine operations** can be as important as **how you implement them**.

## Overview

In this puzzle, you'll implement fused LayerNorm + Linear operations with both forward and backward passes. While both fused and unfused implementations produce identical results, they use different strategies that lead to significant performance differences.

You'll compare:

- **Unfused approach**: Separate kernels for LayerNorm and Linear
- **Fused kernel**: Combined operation in a single kernel
- **Custom backward pass**: Gradient computation for fused operations

This comparison teaches the critical importance of kernel fusion and proper gradient computation in deep learning operations.

## Background: LayerNorm + Linear operations

LayerNorm and Linear are fundamental operations in transformer architectures, particularly in attention mechanisms and feed-forward networks. Here's how they're typically used:

```python
import torch
import torch.nn.functional as F

# Input: hidden states
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm parameters
ln_weight = torch.ones(hidden_dim)  # scale parameter (γ)
ln_bias = torch.zeros(hidden_dim)   # shift parameter (β)

# Linear layer parameters
linear_weight = torch.randn(output_dim, hidden_dim)
linear_bias = torch.zeros(output_dim)

# Unfused operations (with autograd)
ln_output = F.layer_norm(x, [hidden_dim], weight=ln_weight, bias=ln_bias)
output = F.linear(ln_output, linear_weight, linear_bias)

# Fused operation (custom implementation)
# This is what you'll implement in this puzzle
output_fused = fused_layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias)
```

When fused, these operations are combined into a single efficient kernel that:

- Reduces memory bandwidth usage
- Minimizes kernel launch overhead
- Improves cache utilization
- Eliminates intermediate allocations

In practice, this fusion can provide up to 1.5-2x speedup in both forward and backward passes, which is crucial for transformer training efficiency.

### Why custom backward passes matter

PyTorch's autograd system automatically computes gradients for individual operations, but fused operations require custom backward passes to:

- Maintain numerical stability
- Ensure proper gradient flow
- Optimize memory access patterns
- Handle atomic operations for gradient accumulation

## Learning path

This puzzle is structured in two parts to build your understanding systematically:

### **[Forward pass implementation](./forward_pass.md)**

Start here to implement the fused forward kernel and understand kernel fusion benefits.

**What you'll do:**

- Implement both unfused and fused forward kernels
- Learn fundamental kernel fusion techniques
- See the same operations implemented with different strategies
- Understand performance implications of fusion
- Learn memory access patterns for optimal performance

### **[Backward pass implementation](./backward_pass.md)**

Deep dive into autograd integration and gradient computation.

**What you'll learn:**

- How to implement custom backward passes
- Why proper gradient flow is crucial
- Real-world implications for training efficiency
- Optimization strategies for backward operations
- Mathematical foundations of gradient computation
- Atomic operations for gradient accumulation
- Numerical stability in backward passes

## Getting started

Ready to explore kernel fusion and autograd integration? Start with the **[Forward pass implementation](./forward_pass.md)** to implement the fused kernel, then move to **[Backward pass implementation](./backward_pass.md)** to understand gradient computation.

The puzzle includes a comprehensive testing framework that verifies:

- Numerical correctness against PyTorch's implementation for both forward and backward passes
- Performance comparison between our CPU and GPU implementations
- Gradient computation accuracy for all parameters (input, LayerNorm weights/bias, Linear weights/bias)
- Memory usage optimization through kernel fusion

💡 **Success tip:** Pay attention to how the different implementations (fused vs unfused) affect both forward and backward pass performance - this insight applies to many deep learning operations beyond LayerNorm + Linear. The backward pass implementation is particularly important as it directly impacts training efficiency and numerical stability.
