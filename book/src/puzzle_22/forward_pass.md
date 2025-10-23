# ⚛️ Fused vs Unfused Kernels

## Overview

In this puzzle, we explore the performance benefits of kernel fusion by implementing and comparing two approaches to the [LayerNorm](https://arxiv.org/abs/1607.06450) and Linear operation:

1. **Unfused approach**: Executes LayerNorm and Linear as separate operations
2. **Fused kernel**: Combines LayerNorm and Linear operations into a single GPU kernel

This comparison demonstrates how kernel fusion can significantly improve performance by:

- Reducing memory bandwidth usage
- Minimizing kernel launch overhead
- Improving cache utilization
- Eliminating intermediate memory allocations

## Key concepts

In this puzzle, you'll learn:

- **Kernel fusion techniques** for combining multiple operations
- **Memory bandwidth optimization** through fused operations
- **Performance benchmarking** of different kernel implementations
- **Numerical stability** in fused operations
- **PyTorch custom operation integration**

The mathematical operations we're fusing are:

1. LayerNorm:
\\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]

2. Linear:
\\[\Large \text{Linear}(x) = Wx + b \\]

When fused, we compute:
\\[\Large \text{Fused}(x) = W(\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta) + b \\]

## Understanding LayerNorm

LayerNorm is a normalization technique that helps stabilize and accelerate the training of deep neural networks. Let's break down its components and parameters:

### What LayerNorm does

1. **Normalization**: LayerNorm normalizes the activations across the features (hidden dimensions) for each sample independently. This means:
   - For each sequence position, it computes statistics across the hidden dimension
   - Each sample in the batch is normalized independently
   - This is different from [BatchNorm](https://arxiv.org/abs/1502.03167), which normalizes across the batch dimension

2. **Parameters**:
   - \\(\gamma\\) (scale): A learnable parameter vector that allows the network to learn the optimal scale for each feature
   - \\(\beta\\) (shift): A learnable parameter vector that allows the network to learn the optimal shift for each feature
   - \\(\epsilon\\): A small constant (1e-5) added to the variance to prevent division by zero

### What LayerNorm does in practice

LayerNorm performs several crucial functions in deep neural networks:

1. **Feature standardization**:
   - Transforms each feature to have zero mean and unit variance
   - Makes the network's learning process more stable
   - Helps prevent the "internal covariate shift" problem where the distribution of layer inputs changes during training

2. **Gradient flow**:
   - Improves gradient flow through the network
   - Prevents vanishing/exploding gradients
   - Makes training more efficient by allowing higher learning rates

3. **Regularization effect**:
   - Acts as a form of implicit regularization
   - Helps prevent overfitting by normalizing the feature distributions
   - Makes the network more robust to input variations

4. **Sequence modeling**:
   - Particularly effective in transformer architectures
   - Helps maintain consistent signal magnitude across different sequence lengths
   - Enables better handling of variable-length sequences

5. **Training dynamics**:
   - Accelerates training convergence
   - Reduces the need for careful learning rate tuning
   - Makes the network less sensitive to weight initialization

### Mathematical components

1. **Mean Calculation** (\\(\mu\\)):
   \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - Computes the mean across the hidden dimension (H)
   - Each sequence position has its own mean

2. **Variance Calculation** (\\(\sigma^2\\)):
   \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - Computes the variance across the hidden dimension
   - Used to scale the normalized values

3. **Normalization and Scaling**:
   \\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - First normalizes the input to have zero mean and unit variance
   - Then applies learnable scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters
   - The \\(\odot\\) symbol represents elementwise multiplication (Hadamard product)
   - For example, if \\(\gamma = [1.2, 0.8, 1.5]\\)  and normalized input is \\([0.5, -0.3, 0.7]\\), then \\(\gamma \odot x = [0.6, -0.24, 1.05]\\)

### Why LayerNorm is important

1. **Training Stability**:
   - Prevents activations from growing too large or small
   - Helps maintain consistent signal magnitude throughout the network

2. **Feature Learning**:
   - The scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters allow the network to learn which features are important
   - Can effectively learn to ignore or emphasize certain features

3. **Independence**:
   - Unlike BatchNorm, LayerNorm's statistics are computed independently for each sample
   - Makes it more suitable for variable-length sequences and small batch sizes

## Configuration

- Batch size: `BATCH_SIZE = 4`
- Sequence length: `SEQ_LEN = 4`
- Hidden dimension: `HIDDEN_DIM = 8`
- Output dimension: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- Data type: `DType.float32`

## Implementation approaches

### 1. Unfused implementation

The unfused approach executes operations separately using multiple kernels. Here are some of the kernels we wrote in the previous chapters:

#### Matrix multiplication kernel

From [Puzzle 16](../puzzle_16/puzzle_16.md), we reuse the tiled matrix multiplication kernel for the linear transformation. This kernel includes bounds checking to handle variable matrix dimensions safely:

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:matmul_idiomatic_tiled}}
```

#### Transpose kernel

For efficient memory access patterns, we use a transpose kernel with shared memory tiling:

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:transpose_kernel}}
```

#### Bias addition kernel

A simple elementwise addition kernel for adding the bias term:

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:add_bias_kernel}}
```

#### LayerNorm kernel

Now complete this kernel to implement the LayerNorm operation. You'll need to:

1. Compute mean \\(\mu\\) and variance \\(\sigma^2\\) for each sequence position
2. Normalize the input using these statistics
3. Apply the scale \\(\gamma\\) and shift \\(\beta\\) parameters

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:layernorm_kernel}}
```

**Implementation steps:**

1. First, compute mean and variance using parallel reduction
2. Then normalize the input using these statistics
3. Finally, apply the scale and shift parameters

**Characteristics of unfused approach:**

- Multiple kernel launches (LayerNorm → MatMul → Bias)
- Intermediate tensor allocations between operations
- More memory bandwidth usage due to separate passes
- Simpler implementation with clear separation of concerns
- Easier to debug as each operation is isolated

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Thread organization**:
   - Use one thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Each thread handles one hidden dimension element
   - Avoid redundant computation by computing statistics once per sequence

2. **Memory access**:
   - Access input tensor with `[batch_idx, seq_idx, hidden_idx]`
   - Access output tensor with `[batch_idx, seq_idx, hidden_idx]`
   - Access LayerNorm parameters with `[hidden_idx]`

3. **Numerical stability**:
   - Add epsilon (1e-5) before taking square root
   - Use `rebind[Scalar[dtype]]` for proper type casting
   - Compute variance as (sq_sum / hidden_dim) - (mean * mean)

4. **Performance**:
   - Compute mean and variance in a single pass
   - Reuse computed statistics for all elements in sequence
   - Avoid unnecessary memory barriers

</div>
</details>

### Running the code

To test your unfused implementation, run:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --unfused
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --unfused
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --unfused
```

  </div>
</div>

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
   Puzzle 22: UNFUSED Algorithm Test & Benchmark
============================================================

🧪 Correctness Testing for UNFUSED Algorithm
====================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
✅ Reference PyTorch
   Max difference: 0.00e+00
   Result: ✅ CORRECT

Testing CPU Implementation
---------------------------------
✅ Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Testing GPU Unfused Implementation
-----------------------------------------
✅ Using Mojo unfused kernel (GPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Correctness Summary:
   - Reference:   ✅ CORRECT
   - CPU:         ✅ CORRECT
   - GPU unfused: ✅ CORRECT

   Overall Correctness: ✅ ALL CORRECT

Benchmarking CPU vs GPU UNFUSED
------------------------------------------
   Testing CPU performance...
   CPU: 3173.70ms (50 iterations)
   Testing GPU unfused performance...
   GPU unfused: 3183.57ms (50 iterations)

   GPU unfused vs CPU: 1.00x slower
   CPU wins (GPU overhead > computation benefit)

UNFUSED Algorithm Test Completed!
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p22/op/layernorm_linear.mojo:layernorm_kernel_solution}}
```

<div class="solution-explanation">

The unfused implementation follows a straightforward approach where each thread handles one element of the output tensor. Let's break down the key components:

1. **Thread and Block Organization**:

   ```mojo
   batch_idx = block_idx.x
   seq_idx = block_idx.y
   hidden_idx = thread_idx.x
   ```

   - Each thread block handles one sequence position in the batch
   - Grid dimensions: `[batch_size, seq_len]`
   - Each thread processes one element in the hidden dimension
   - Early return if indices are out of bounds:

     ```mojo
     if (batch_idx >= batch_size or seq_idx >= seq_len or hidden_idx >= hidden_dim):
         return
     ```

2. **Statistics Computation**:

   ```mojo
   var sum_val: Scalar[dtype] = 0
   var sq_sum: Scalar[dtype] = 0

   @parameter
   for h in range(hidden_dim):
       val = input[batch_idx, seq_idx, h]
       sum_val += rebind[Scalar[dtype]](val)
       sq_sum += rebind[Scalar[dtype]](val * val)
   ```

   - Compute sum and squared sum in a single pass
   - Use `@parameter` for compile-time loop unrolling
   - Proper type casting with `rebind[Scalar[dtype]]`
   - Calculate mean and variance:

     ```mojo
     mean_val = sum_val / hidden_dim
     var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
     inv_std = 1.0 / sqrt(var_val + 1e-5)
     ```

3. **Normalization and Scaling**:

   ```mojo
   input_val = input[batch_idx, seq_idx, hidden_idx]
   normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
       ln_weight[hidden_idx]
   ) + rebind[Scalar[dtype]](ln_bias[hidden_idx])
   output[batch_idx, seq_idx, hidden_idx] = normalized
   ```

   - Apply normalization: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - Scale with learnable parameter `γ` (ln_weight)
   - Add learnable bias `β` (ln_bias)
   - Store result in output tensor

4. **Performance Characteristics**:
   - Each thread computes statistics independently
   - No shared memory usage (simple but less efficient)
   - Memory access pattern:
     - Input: `[batch_idx, seq_idx, h]`
     - Output: `[batch_idx, seq_idx, hidden_idx]`
     - Parameters: `[hidden_idx]`
   - Numerical stability ensured by:
     - Adding epsilon (1e-5) before square root
     - Using proper type casting
     - Computing variance in a numerically stable way

5. **Implementation Details**:
   - **Type Safety**:
     - Use `Scalar[dtype]` for intermediate calculations
     - `rebind[Scalar[dtype]]` for proper type casting
     - Ensures consistent floating-point precision

   - **Memory Access**:
     - Coalesced reads from input tensor
     - Coalesced writes to output tensor
     - Sequential access to LayerNorm parameters

   - **Computation Flow**:
     - Statistics computation: \\[\Large O(H) \text{ operations per thread} \\]
     - Normalization: \\[\Large O(1) \text{ operations per thread} \\]
     - Total complexity: \\[\Large O(H) \text{ per output element} \\]

   - **Limitations**:
     - Redundant computation of statistics
     - No shared memory for intermediate results
     - High memory bandwidth usage
     - Multiple kernel launches required

This implementation is correct but not optimal for performance, as shown in the benchmark results where it's slightly slower than the CPU version. The fused implementation will address these performance limitations by:

- Computing statistics once per sequence
- Reusing normalized values
- Reducing memory traffic
- Eliminating intermediate tensor allocations

</div>
</details>

### 2. Fused kernel implementation

The fused kernel combines LayerNorm and Linear operations into a single GPU kernel:

```mojo
{{#include ../../../problems/p22/op/layernorm_linear.mojo:minimal_fused_forward_kernel}}
```

**Key optimizations:**

- Single kernel launch instead of two
- Shared memory for intermediate results
- Coalesced memory access patterns
- Reduced memory bandwidth usage
- No intermediate tensor allocations

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Thread organization**:
   - One thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Single thread per sequence position to avoid redundancy
   - Compute all outputs for each sequence position in one thread

2. **Memory access**:
   - Access input tensor with `[batch_idx, seq_idx, h]`
   - Access output tensor with `[batch_idx, seq_idx, out_idx]`
   - Access weights with `[out_idx, h]` for linear layer

3. **Computation flow**:
   - Compute LayerNorm statistics once per sequence
   - Reuse normalized values for all output dimensions
   - Combine normalization and linear transformation

4. **Performance**:
   - Avoid redundant computation of statistics
   - Minimize memory traffic by fusing operations
   - Use proper type casting with `rebind[Scalar[dtype]]`

</div>
</details>

### Running the code

To test your fused implementation, run:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --fused
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --fused
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --fused
```

  </div>
</div>

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
   Puzzle 22: FUSED Algorithm Test & Benchmark
============================================================

🧪 Correctness Testing for FUSED Algorithm
==================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
✅ Reference PyTorch
   Max difference: 0.00e+00
   Result: ✅ CORRECT

Testing CPU Implementation
---------------------------------
✅ Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Testing GPU Fused Implementation
---------------------------------------
✅ Using Mojo fused kernel (GPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Correctness Summary:
   - Reference:   ✅ CORRECT
   - CPU:         ✅ CORRECT
   - GPU fused: ✅ CORRECT

   Overall Correctness: ✅ ALL CORRECT

⚡ Benchmarking CPU vs GPU FUSED
----------------------------------------
   Testing CPU performance...
   CPU: 3144.75ms (50 iterations)
   Testing GPU fused performance...
   GPU fused: 3116.11ms (50 iterations)

   GPU fused vs CPU: 1.01x faster
   GPU fused wins!

FUSED Algorithm Test Completed!
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p22/op/layernorm_linear.mojo:minimal_fused_forward_kernel_solution}}
```

<div class="solution-explanation">

The fused implementation combines operations efficiently:

1. **Thread organization**:
   - One thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Single thread per sequence position
   - Thread indices: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`

2. **LayerNorm phase**:
   - Compute sum and squared sum for the sequence position
   - Calculate mean: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - Calculate variance: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - Compute inverse standard deviation: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]

3. **Linear phase**:
   - For each output dimension:
     - Compute normalized value: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
     - Multiply with linear weight and accumulate: \\[\Large \text{acc} = \sum_{h=1}^{H} \text{normalized}_h \cdot W_{out,h} \\]
     - Add linear bias: \\[\Large \text{output} = \text{acc} + b_{out} \\]
   - Store result in `output[batch_idx, seq_idx, out_idx]`

4. **Performance optimizations**:
   - Single kernel launch for both operations
   - Reuse computed statistics
   - Minimize memory traffic
   - No intermediate tensor allocations
   - Efficient memory access patterns

This implementation achieves better performance than the unfused version by reducing memory bandwidth usage and kernel launch overhead.
</div>
</details>

## Advantages of kernel fusion

In this puzzle, we've explored two approaches to implementing LayerNorm + Linear operations:

1. **Unfused implementation**:
   - Separate kernels for LayerNorm and Linear
   - Simpler implementation but less efficient
   - Higher memory bandwidth usage
   - Multiple kernel launches
   - Benchmark results: 3183.57ms (GPU)

2. **Fused implementation**:
   - Single kernel combining both operations
   - More complex but significantly more efficient
   - Reduced memory bandwidth usage
   - Single kernel launch
   - Benchmark results: 3116.11ms (GPU)

### Memory bandwidth optimization

1. **Eliminated memory traffic**:
   - No intermediate tensor allocations between operations
   - Reduced global memory reads/writes
   - Reuse of normalized values for linear transformation
   - Memory bandwidth reduction: \\[\Large \text{reduction} = \frac{\text{unfused\_bandwidth} - \text{fused\_bandwidth}}{\text{unfused\_bandwidth}}\\]

2. **Cache efficiency**:
   - Better L1/L2 cache utilization
   - Reduced cache misses
   - Improved memory access patterns
   - Higher arithmetic intensity

### Reduced overhead

1. **Kernel launch optimization**:
   - Single kernel launch instead of multiple
   - Lower driver overhead
   - Reduced synchronization points
   - Fewer memory allocations

2. **Resource management**:
   - Shared memory reuse between operations
   - Better register utilization
   - Improved thread occupancy
   - Higher GPU utilization

### Performance characteristics

1. **Scalability**:
   - Better performance scaling with input size
   - Reduced memory bandwidth bottleneck
   - More efficient use of GPU resources
   - Improved throughput for large models

2. **Numerical efficiency**:
   - Maintained numerical stability
   - Reduced rounding errors
   - Better precision in intermediate results
   - Optimized computation order

💡 **Key insight**: Kernel fusion is particularly beneficial for operations that are frequently used together in neural networks, like LayerNorm + Linear in transformer architectures. The performance benefits become more significant with larger input sizes and more complex models.
