# ⛓️ Autograd Integration & Backward Pass

## Overview

In this puzzle, we explore the backward pass implementation of the fused LayerNorm + Linear operation. The backward pass computes gradients with respect to:

- Input tensor
- LayerNorm scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters
- Linear layer weight matrix and bias

The mathematical operations we're implementing are:

1. LayerNorm backward (details of derivation in [Detailed derivation of LayerNorm backward pass](#detailed-derivation-of-layernorm-backward-pass)):
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]

2. Linear backward:
\\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
\\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
\\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]

3. Chain Rule for Fused Operation:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_{linear}} \frac{\partial y_{linear}}{\partial y_{norm}} \frac{\partial y_{norm}}{\partial x} \\]
where:

- \\(y_{norm}\\) is the LayerNorm output
- \\(y_{linear}\\) is the Linear layer output
- The chain rule ensures proper gradient flow through both operations

## Key concepts

- **Thread organization**:
  - One thread block per sequence position (grid: `[batch_size, seq_len]`)
  - Single thread per sequence position to avoid redundancy
  - Compute all gradients for each sequence position in one thread
  - Ensure proper thread synchronization for atomic operations

- **Memory access**:
  - Access input tensor with `[batch_idx, seq_idx, h]`
  - Access output tensor with `[batch_idx, seq_idx, out_idx]`
  - Access weights with `[out_idx, h]` for linear layer
  - Ensure memory alignment for atomic operations
  - Use shared memory for frequently accessed data

- **Computation flow**:
  - Compute LayerNorm statistics in same order as forward pass
  - Reuse normalized values for all output dimensions
  - Combine normalization and linear transformation
  - Maintain numerical stability throughout
  - Handle edge cases properly

- **Performance**:
  - Avoid redundant computation of statistics
  - Minimize memory traffic by fusing operations
  - Use proper type casting with `rebind[Scalar[dtype]]`
  - Ensure proper memory alignment
  - Optimize for autograd integration

## Configuration

- Batch size: `BATCH_SIZE = 4`
- Sequence length: `SEQ_LEN = 4`
- Hidden dimension: `HIDDEN_DIM = 8`
- Output dimension: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- Data type: `DType.float32`

## Implementation (challenging)

The fused backward kernel combines LayerNorm and Linear backward operations into a single GPU kernel. This is a challenging implementation that requires careful handling of:

- [Atomic operations](https://docs.modular.com/mojo/stdlib/os/atomic/Atomic/) for gradient accumulation
- Numerical stability in gradient computations
- Memory access patterns for efficient GPU utilization
- Proper synchronization between operations

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:minimal_fused_backward_kernel}}
```

**Key optimizations:**

- Single kernel launch for all gradient computations
- Atomic operations for safe gradient accumulation
- Coalesced memory access patterns
- Reduced memory bandwidth usage
- No intermediate tensor allocations

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Thread organization**:
   - One thread block per sequence position
   - Single thread per sequence position
   - Compute all gradients in one thread

2. **Memory access**:
   - Coalesced access for input/output tensors
   - Strided access for weight matrix
   - Proper alignment for atomic operations

3. **Computation flow**:
   - Compute statistics in same order as forward pass
   - Reuse normalized values
   - Maintain numerical stability

4. **Performance**:
   - Minimize memory traffic
   - Use proper type casting
   - Ensure proper alignment

</div>
</details>

### Running the code

To test your fused backward implementation, run:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --backward
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --backward
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --backward
```

  </div>
</div>

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
           Comprehensive Backward Pass Test
           Testing Custom LayerNorm + Linear Gradients
============================================================
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]

Testing CPU Backward Pass:

Testing CPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (CPU)...
✅ CPU Backward Implementation backward completed
   Forward max difference: 1.49e-08
   grad_input: 2.98e-08 ✅
   grad_ln_weight: 5.96e-08 ✅
   grad_ln_bias: 2.38e-07 ✅
   grad_linear_weight: 9.54e-07 ✅
   grad_linear_bias: 0.00e+00 ✅

   Forward pass: ✅ CORRECT
   Gradients:    ✅ CORRECT
   Overall:      ✅ CORRECT

Testing GPU Backward Pass:

Testing GPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (GPU)...

✅ GPU Backward Implementation backward completed
   Forward max difference: 1.86e-08
   grad_input: 4.47e-08 ✅
   grad_ln_weight: 5.96e-08 ✅
   grad_ln_bias: 3.58e-07 ✅
   grad_linear_weight: 9.54e-07 ✅
   grad_linear_bias: 0.00e+00 ✅

   Forward pass: ✅ CORRECT
   Gradients:    ✅ CORRECT
   Overall:      ✅ CORRECT

Backward Pass Test Summary:
   - CPU Backward:  ✅ CORRECT
   - GPU Backward:  ✅ CORRECT

   Overall Result: ✅ ALL CORRECT

BACKWARD PASS Test Completed!
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p22/op/layernorm_linear.mojo:minimal_fused_backward_kernel_solution}}
```

<div class="solution-explanation">

The fused backward implementation combines operations efficiently:

1. **Thread organization and memory layout**:
   - Grid dimensions: `[batch_size, seq_len]` for one thread block per sequence position
   - Thread indices: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`
   - Memory layout:
     - Input tensor: `[batch_size, seq_len, hidden_dim]`
     - Output tensor: `[batch_size, seq_len, output_dim]`
     - Weight matrix: `[output_dim, hidden_dim]`
     - Gradients: `[batch_size, seq_len, hidden_dim]` for input gradients
     - Parameter gradients: `[hidden_dim]` for LayerNorm, `[output_dim, hidden_dim]` for Linear

2. **LayerNorm backward phase**:
   - Recompute forward pass statistics in same order as forward pass:
     - Mean: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
     - Variance: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
     - Inverse standard deviation: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]
   - Compute normalized values: \\[\Large \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \\]
   - Calculate gradients:
     - Input gradient: \\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]
     - Scale gradient: \\[\Large \frac{\partial L}{\partial \gamma} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \odot \hat{x}_i \\]
     - Shift gradient: \\[\Large \frac{\partial L}{\partial \beta} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \\]

3. **Linear backward phase**:
   - For each output dimension:
     - Bias gradient: \\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
     - Weight gradient: \\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
     - Input gradient: \\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]
   - Use atomic operations for gradient accumulation:
     - `atomic_add` for bias gradients with proper alignment
     - `atomic_add` for weight gradients with proper alignment
     - `atomic_add` for LayerNorm parameter gradients with proper alignment

4. **Memory access patterns**:
   - Coalesced access for input/output tensors
   - Strided access for weight matrix
   - Atomic operations for gradient accumulation
   - Shared memory for intermediate results
   - Register usage for frequently accessed values
   - Proper memory alignment for all operations

5. **Numerical stability**:
   - Careful handling of epsilon in denominator
   - Proper scaling of gradients
   - Stable computation of statistics
   - Type casting with `rebind[Scalar[dtype]]`
   - Proper handling of edge cases
   - Maintain same computation order as forward pass

6. **Performance optimizations**:
   - Single kernel launch for all operations
   - Reuse of computed statistics
   - Minimized memory traffic
   - No intermediate tensor allocations
   - Efficient thread utilization
   - Reduced synchronization points
   - Optimized memory access patterns
   - Proper memory alignment

7. **Implementation details**:
   - Use of `@parameter` for compile-time constants
   - Proper handling of tensor dimensions
   - Efficient type casting and conversions
   - Careful management of shared memory
   - Proper synchronization between operations
   - Error handling and boundary checks
   - Integration with PyTorch's autograd system

This implementation achieves better performance than the unfused version by:

- Reducing memory bandwidth usage through kernel fusion
- Minimizing kernel launch overhead
- Optimizing memory access patterns
- Efficient use of GPU resources
- Maintaining numerical stability
- Proper handling of gradient accumulation
- Ensuring proper memory alignment
- Efficient autograd integration

The fused backward pass is particularly important in transformer architectures where LayerNorm + Linear operations are frequently used together, making the performance benefits significant for real-world applications.
</div>
</details>

## Performance considerations

The backward pass implementation uses `torch.compile` with optimizations to minimize overhead:

```python
# Compilation configuration
torch._dynamo.config.cache_size_limit = 64  # Increase cache
torch._dynamo.config.suppress_errors = True  # Handle errors gracefully
torch._dynamo.config.automatic_dynamic_shapes = True  # Dynamic shapes
```

These optimizations are particularly important for the backward pass because:

- Small tensor operations benefit from compilation caching
- Dynamic shapes are common in backward passes
- Error handling needs to be robust for gradient computation
- Cache size helps with repeated backward operations
- Proper error handling is crucial for gradient computation
- Compilation overhead can significantly impact training time

The backward pass is compiled with `reduce-overhead` mode to minimize the compilation overhead while maintaining correctness. This is especially important because:

- Backward passes are called frequently during training
- Gradient computation needs to be numerically stable
- Memory access patterns need to be optimized
- Atomic operations require proper synchronization
- Autograd integration needs to be efficient

## Detailed derivation of LayerNorm backward pass

The backward pass gradient for LayerNorm is derived through careful application of the chain rule. Here's the step-by-step derivation:

### Forward pass operations

- Mean: \\(\mu = \frac{1}{H} \sum_{i=1}^{H} x_i\\)
- Variance: \\(\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2\\)
- Normalized value: \\(\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\\)
- Final output: \\(y = \gamma \odot \hat{x} + \beta\\)

### Chain rule application

To compute \\(\frac{\partial L}{\partial x}\\), we apply the chain rule:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial x}\\]

### Gradient components

#### Output to normalized value

- \\(\frac{\partial y}{\partial \hat{x}} = \gamma\\) (element-wise multiplication)

#### Normalized value to input

The gradient \\(\frac{\partial \hat{x}}{\partial x}\\) has three components:

- Direct effect through numerator: \\(\frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- Indirect effect through mean: \\(-\frac{1}{H} \frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- Indirect effect through variance: \\(-\frac{(x - \mu)}{H(\sigma^2 + \epsilon)^{3/2}} (x - \mu)\\)

### Combining terms

The gradient through the normalization term can be simplified to:
\\[\Large \frac{\partial \hat{x}}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

### Final gradient expression

Combining all terms:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

### Key insights

- The chain rule accounts for all paths through which x affects the output
- The normalization term \\(\sqrt{\sigma^2 + \epsilon}\\) appears in both numerator and denominator
- The mean and variance terms create additional paths for gradient flow
- The final expression combines all effects into a single efficient computation

### Implementation considerations

- The gradient properly accounts for the scaling effect of \\(\gamma\\)
- The normalization effect of mean and variance is preserved
- The numerical stability term \\(\epsilon\\) is maintained
- Gradients are properly scaled across the hidden dimension H
- The computation order matches the forward pass for numerical stability

This derivation ensures that the backward pass maintains the same numerical properties as the forward pass while efficiently computing all necessary gradients.
