# Puzzle 18: Softmax Op

## Overview

In this puzzle, we'll implement the softmax function as a custom MAX Graph operation. Softmax takes a vector of real numbers and normalizes it into a probability distribution.

The softmax function works by performing two main steps:

1. Exponentiation: It applies the exponential function to each element of the input vector. This ensures all values are positive and amplifies the differences between them. Larger input values result in significantly larger exponential outputs, while smaller or negative values result in outputs closer to zero.

2. Normalization: It then divides each exponentiated value by the sum of all the exponentiated values. This normalization step ensures that the resulting values are a valid probability distribution, meaning they are all between 0 and 1 and their sum is exactly 1.

Mathematically, the softmax function is defined as:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Where:

- \\(x_i\\) is the \\(i\\)-th element of the input vector
- \\(n\\) is the length of the input vector

However, this direct implementation can lead to numerical overflow issues when values are large. To address this, we use a more numerically stable version:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

Our GPU implementation uses parallel reduction for both finding the maximum value and computing the sum of exponentials, making it highly efficient for large vectors.

## Key concepts

- Parallel reduction for efficient maximum and sum calculations
- Numerical stability through max-subtraction technique
- Shared memory usage for thread communication
- Custom MAX Graph operation integration with Python
- Thread synchronization with barriers

## Configuration

- Vector size: `SIZE = 128`
- Threads per block: `BLOCK_DIM_X = 1 << log2_ceil(SIZE)`. Tree-based reduction requires `BLOCK_DIM_X` to be the next power of two `>= SIZE` for correctness.
- Grid dimensions: \\(1 \times 1\\) block
- Shared memory: Two shared variables for max and sum

Layout configuration:

- Input tensor: `Layout.row_major(SIZE)`
- Output tensor: `Layout.row_major(SIZE)`
- Custom op parameters: `{"input_size": input_tensor.shape[0]}`

Key aspects of this puzzle include:

1. **Numerical stability**: Understanding how to handle potential numerical issues
2. **Parallel reductions**: Using shared memory for efficient max and sum calculations
3. **Custom op integration**: Completing the Python interface for our Mojo GPU kernel
4. **Testing and verification**: Ensuring our implementation matches the expected results

Our softmax custom operation will:

- Accept NumPy arrays from Python
- Process them efficiently on the GPU
- Return normalized probability distributions
- Match the results of SciPy's softmax implementation

## Code to complete

To complete this puzzle, you need to implement both the GPU and CPU kernels in the Mojo file and complete the graph definition in the Python code.

### 1. Implement the GPU kernel in `softmax.mojo`

```mojo
{{#include ../../../problems/p18/op/softmax.mojo:softmax_gpu_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/op/softmax.mojo" class="filename">View full file: problems/p18/op/softmax.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use shared memory for both the maximum value and sum to ensure all threads can access these values
2. Remember to call `barrier()` at appropriate points to synchronize threads
3. Implement parallel reduction by having each thread process a portion of the input array
4. Use a tree-based reduction pattern to minimize thread divergence
5. Handle out-of-bounds access carefully, especially for large inputs
6. For numerical stability, calculate \\(e^{x_i - max}\\) instead of \\(e^{x_i}\\)

</div>
</details>

### 2. Implement the CPU kernel in `softmax.mojo`

```mojo
{{#include ../../../problems/p18/op/softmax.mojo:softmax_cpu_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/op/softmax.mojo" class="filename">View full file: problems/p18/op/softmax.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Create a sequential implementation that follows the same mathematical steps as the GPU version
2. First find the maximum value across all inputs
3. Then compute \\(e^{x_i - max}\\) for each element and accumulate the sum
4. Finally, normalize by dividing each element by the sum
5. Use scalar operations since we don't have parallel threads in the CPU implementation

</div>
</details>

### Test the CPU and GPU kernels

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p18-test-kernels
```

  </div>
  <div class="tab-content">

```bash
pixi run p18-test-kernels
```

  </div>
</div>

when done correctly you'll see

```txt
Total Discovered Tests: 1

Passed : 1 (100.00%)
Failed : 0 (0.00%)
Skipped: 0 (0.00%)
```

### 3. Complete the graph definition in `p18.py`

```python
{{#include ../../../problems/p18/p18.py:softmax_custom_op_graph}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/p18.py" class="filename">View full file: problems/p18/p18.py</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use `graph.inputs[0]` to access the input tensor passed to the graph
2. Call `ops.custom()` with the name matching your registered custom op ("softmax")
3. Pass the input tensor as a value to the custom operation
4. Specify the output type to match the input shape
5. Include the "input_size" parameter which is required by the kernel
6. Set `graph.outputs` to a list containing your operation's output tensor

</div>
</details>

You can run the puzzle with:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p18
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p18
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p18
```

  </div>
  <div class="tab-content">

```bash
uv run poe p18
```

  </div>
</div>

When successful, you should see output similar to on CPU and GPU:

```
Input shape: (128,)
First few random input values: [ 1.1810775   0.60472375  0.5718309   0.6644599  -0.08899796]
Compiling softmax graph on Device(type=cpu,id=0)
Executing softmax on Device(type=cpu,id=0)
====================================================================================================
Compiling softmax graph on Device(type=gpu,id=0)
Executing softmax on Device(type=gpu,id=0)
====================================================================================================
First few softmax results on CPU (custom Mojo kernel): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
First few softmax results on GPU (custom Mojo kernel): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
First few expected results (SciPy calculation): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
Verification passed: Custom kernel results match SciPy calculation
Sum of all probabilities on CPU: 1.0
Sum of all probabilities on GPU: 1.0
```

This indicates that your custom MAX Graph operation correctly implements the softmax algorithm and produces a valid probability distribution.

## Solution

<details class="solution-details">
<summary></summary>

To solve this puzzle, we need to implement both the Mojo kernels (GPU and CPU) and the Python graph definition for our softmax custom operation. Similar to what we did in [Puzzle 17](../puzzle_17/puzzle_17.md), we're creating a bridge between Python's ecosystem and Mojo's GPU-accelerated computing capabilities.

The softmax operation we're implementing is mathematically defined as:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

However, to prevent numerical overflow, we use the more stable form:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

### GPU kernel implementation

```mojo
{{#include ../../../solutions/p18/op/softmax.mojo:softmax_gpu_kernel_solution}}
```

<div class="solution-explanation">
Our GPU kernel implements the numerically stable softmax algorithm with highly optimized parallel reduction techniques. Let's dissect the kernel in detail:

#### Kernel signature and memory management

```mojo
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
)
```

The kernel is parameterized with:

- Common layout parameter for both input and output tensors
- Vector size as an Integer parameter
- Configurable data type with float32 as default
- Mutable output tensor for in-place computation
- Non-mutable input tensor (mut=False)

#### Shared memory allocation

```mojo
shared_max = LayoutTensor[dtype, Layout.row_major(BLOCK_DIM_X), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
shared_sum = LayoutTensor[dtype, Layout.row_major(BLOCK_DIM_X), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
```

The kernel allocates two shared memory buffers:

- `shared_max`: For parallel maximum finding reduction
- `shared_sum`: For parallel sum computation
- Both use `BLOCK_DIM_X = 128` as their size
- Shared memory provides fast access for all threads within a block

#### Thread indexing

```mojo
global_i = thread_idx.x
```

This implementation of softmax operates on a single 1d thread block. i.e. The global and local index are the same.

#### Maximum-finding phase

```mojo
var val: Scalar[dtype] = min_finite[dtype]()
if global_i < input_size:
    val = rebind[Scalar[dtype]](input[global_i])

shared_max[local_i] = val
barrier()
```

This initializes each thread with:

- The minimum finite value for elements outside the valid range
- The actual input value for threads that map to valid elements
- Storage in shared memory for the reduction process
- A barrier synchronization to ensure all threads complete memory writes

#### Parallel max reduction

```mojo
stride = BLOCK_DIM_X // 2
while stride > 0:
    if local_i < stride:
        shared_max[local_i] = max(shared_max[local_i], shared_max[local_i + stride])
    barrier()
    stride = stride // 2
```

This implements a parallel tree-reduction pattern:

1. Start with `stride = 64` (half of `BLOCK_DIM_X`)
2. Each active thread compares two values separated by the stride
3. Store the maximum in the lower index
4. Synchronize all threads with a barrier
5. Halve the stride and repeat
6. After \\(\log_2(BLOCK\\_DIM\\_X)~\\) steps, `shared_max[0]` contains the global maximum

This logarithmic reduction is significantly faster than a linear scan on large inputs.

#### Exponentiation with numerical stability

```mojo
block_max = shared_max[0]

var exp_val: Scalar[dtype] = 0.0
if global_i < input_size:
    exp_val = rebind[Scalar[dtype]](exp(val - block_max))
```

Each thread:

1. Reads the global maximum from shared memory
2. Subtracts it from its input value before taking the exponential
3. This subtraction is crucial for numerical stability - it prevents overflow
4. The largest exponent becomes \\(e^0 = 1\\), and all others are \\(e^{negative} < 1\\)

#### Parallel sum reduction

```mojo
shared_sum[local_i] = exp_val
barrier()

stride = BLOCK_DIM_X // 2
while stride > 0:
    if local_i < stride:
        shared_sum[local_i] += shared_sum[local_i + stride]
    barrier()
    stride = stride // 2
```

The second reduction phase:

1. Stores all exponential values in shared memory
2. Uses the same tree-based reduction pattern as for max
3. But performs addition instead of maximum comparison
4. After \\(\log_2(BLOCK\\_DIM\\_X)~\\) steps, `shared_sum[0]` contains the total sum of all exponentials

#### Final normalization

```mojo
block_sum = shared_sum[0]

if global_i < input_size:
    output[global_i] = exp_val / block_sum
```

Each thread:

1. Reads the total sum from shared memory
2. Divides its exponential value by this sum
3. Writes the normalized probability to the output buffer
4. This produces a valid probability distribution that sums to 1

#### Performance characteristics

The implementation has excellent performance characteristics:

- **Complexity**: \\(O(\log n)\\) for both max and sum calculations vs \\(O(n)\\) in a sequential approach
- **Memory efficiency**: Uses only \\(2 \times BLOCK\\_DIM\\_X~\\) elements of shared memory
- **Work efficiency**: Each thread performs approximately \\(2 \times \log_2(BLOCK\\_DIM\\_X)~\\) operations
- **Load balancing**: Each thread handles the same amount of work
- **Synchronization**: Uses minimal barriers, only where necessary
- **Memory access**: Coalesced global memory access pattern for optimal bandwidth

The algorithm is also numerically robust, handling potential overflow/underflow cases by applying the max-subtraction technique that maintains precision across the wide range of values common in neural network activations.
</div>

### CPU fallback implementation

```mojo
{{#include ../../../solutions/p18/op/softmax.mojo:softmax_cpu_kernel_solution}}
```

<div class="solution-explanation">
Our CPU implementation provides a sequential fallback that follows the same mathematical approach but is optimized for single-threaded execution. Let's analyze each phase:

1. **Maximum Finding**:

   ```mojo
   var max_val: Scalar[dtype] = min_finite[dtype]()
   for i in range(input_size):
       max_val = max(max_val, rebind[Scalar[dtype]](input[i]))
   ```

   We initialize with the minimum finite value and perform a linear scan through the array, keeping track of the maximum value encountered. This has \\(O(n)\\) complexity but works efficiently on CPU where we don't have many cores to parallelize across.

2. **Exponential Computation and Summation**:

   ```mojo
   var sum_exp: Scalar[dtype] = 0.0
   for i in range(input_size):
       var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
       output[i] = exp_val
       sum_exp += exp_val
   ```

   We compute \\(e^{x_i - max}\\) for each element, store the result in the output buffer, and accumulate the sum \\(\sum_{j=1}^{n} e^{x_j - max}\\) in a single pass. This approach minimizes memory operations compared to using separate loops.

3. **Normalization**:

   ```mojo
   for i in range(input_size):
       output[i] = output[i] / sum_exp
   ```

   Finally, we normalize each element by dividing by the sum, producing a proper probability distribution according to the softmax formula:

   $$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

The CPU implementation uses the same numerical stability technique (subtracting the maximum) but with sequential operations rather than parallel ones. It's simpler than the GPU version since it doesn't need to handle shared memory or thread synchronization, but it's also less efficient for large inputs.

Both implementations are registered with MAX Graph's custom operation system through the `@compiler.register("softmax")` decorator, allowing seamless execution on either device type based on availability.
</div>

### Python integration

```python
{{#include ../../../solutions/p18/p18.py:softmax_custom_op_graph_solution}}
```

<div class="solution-explanation">
The Python integration creates a seamless bridge between NumPy arrays and our optimized Mojo GPU kernel. The implementation consists of several key components:

1. **Graph Setup and Configuration**:

   ```python
   with Graph(
       "softmax_graph",
       input_types=[
           TensorType(
               dtype,
               shape=input_tensor.shape,
               device=DeviceRef.from_device(device),
           ),
       ],
       custom_extensions=[mojo_kernels],
   ) as graph:
   ```

   This creates a computation graph named "softmax_graph" that:
   - Defines the input tensor type with proper dtype and shape
   - Maps the tensor to the target device (CPU or GPU)
   - Loads our custom Mojo operations from the specified directory
   - The `custom_extensions` parameter is crucial for linking to our Mojo implementation

2. **Custom Operation Configuration**:

   ```python
   output = ops.custom(
       name="softmax",
       values=[input_value],
       out_types=[
           TensorType(
               dtype=input_value.tensor.dtype,
               shape=input_value.tensor.shape,
               device=DeviceRef.from_device(device),
           )
       ],
       parameters={
           "target": "gpu" if device == Accelerator() else "cpu",
           "input_size": input_tensor.shape[0],
           "dtype": dtype,
       },
   )[0].tensor
   ```

   This sets up our custom operation with:
   - Name matching the `@compiler.register("softmax")` in our Mojo code
   - Input values passed as a list
   - Output type definition matching the input shape and type
   - Parameters required by our kernel, including the target device, vector size and data type
   - We extract the tensor from the first returned element with `[0].tensor`

3. **Graph Output Definition**:

   ```python
   graph.output(output)
   ```

   This registers our operation's result as the graph's output.

The main script includes comprehensive testing that:

- Generates random input data: `np.random.randn(INPUT_SIZE).astype(np.float32)`
- Calculates expected results with SciPy: `scipy_softmax(input_array)`
- Verifies numerical accuracy: `np.testing.assert_allclose(..., rtol=1e-5)`
- Confirms the output is a valid probability distribution: `np.sum(result.to_numpy())`

This implementation showcases the power of MAX Graph for integrating high-performance Mojo kernels with Python's scientific computing ecosystem, providing both efficiency and ease of use.
</div>

</details>
