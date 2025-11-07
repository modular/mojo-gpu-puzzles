# Puzzle 17: 1D Convolution Op

> ## Bridging to Python with MAX Graph
>
> We're now entering Part IV of our GPU puzzle journey: **Interfacing with Python via MAX Graph Custom Ops**.
>
> In previous puzzles, we've learned how to write efficient GPU kernels in Mojo. Now we'll explore how to:
>
> - Package these kernels as custom operations that can be called from Python
> - Integrate with the MAX Graph system for accelerated machine learning
> - Bridge the gap between high-level Python APIs and low-level GPU code
>
> This integration allows us to leverage the performance of Mojo GPU kernels while working in familiar Python environments.

## Overview

In [Puzzle 13](../puzzle_13/puzzle_13.md), we implemented a 1D convolution kernel that runs efficiently on the GPU. Now we'll take this kernel and transform it into a custom operation that can be called directly from Python using [MAX Graph](https://docs.modular.com/max/api/python/graph/).

The 1D convolution kernel we'll be working with is already implemented:

```mojo
{{#include ../../../problems/p17/op/conv1d.mojo:conv1d_kernel}}
```

The key aspects of this puzzle include:

1. **Custom op registration**: Understanding how to expose Mojo functions to Python via the `@compiler.register` decorator
2. **Packaging custom ops**: Learning how to package Mojo code for use with MAX Graph
3. **Python integration**: Calling custom operations from Python through MAX Graph
4. **Cross-language data flow**: Managing data types and memory between Python and GPU

This custom operation will:

- Accept [NumPy](https://numpy.org/doc/stable/) arrays as input from Python
- Transfer this data to the GPU
- Execute our optimized convolution kernel
- Return the results back to Python

When you complete this puzzle, you'll have created a seamless bridge between Python's rich ecosystem and Mojo's powerful GPU performance.

## Code to complete

To complete this puzzle, you only need to fill one line in `conv1d.mojo` to call the `conv1d_kernel`:

```mojo
{{#include ../../../problems/p17/op/conv1d.mojo:conv1d_custom_op}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p17/op/conv1d.mojo" class="filename">View full file: problems/p17/op/conv1d.mojo</a>

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
pixi run p17
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p17
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p17
```

  </div>
  <div class="tab-content">

```bash
uv run poe p17
```

  </div>
</div>

When successful, you should see output similar to:

```
Input array: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
Convolution kernel: [0. 1. 2. 3.]
Expected result (NumPy calculation): [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
Compiling 1D convolution graph...
Executing 1D convolution...
1D Convolution result (custom Mojo kernel): [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
Verification passed: Custom kernel results match NumPy calculation
```

This indicates that your custom MAX Graph operation correctly implements the 1D convolution algorithm.

## Solution

<details class="solution-details">
<summary></summary>

To solve this puzzle, we need to integrate our 1D convolution kernel with the MAX Graph system. The key is to properly call our kernel from the `execute` method in the `Conv1DCustomOp` struct.

The solution is:

```mojo
{{#include ../../../solutions/p17/op/conv1d.mojo:conv1d_custom_op_solution}}
```

<div class="solution-explanation">
This single line does several important things:

1. Calls [enqueue_function](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/#enqueue_function) on the GPU context (`gpu_ctx` is of type [DeviceContext](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/)) to schedule our kernel execution
2. Passes the necessary layout and size information as **compile-time** parameters
3. Provides the output, input, and kernel tensors as runtime arguments
4. Configures the execution grid with the appropriate dimensions

Let's break down how this works in the larger context:

### Python-Mojo integration flow

1. **Python side (<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p17/p17.py" class="filename">problems/p17/p17.py</a>)**:
   - Creates NumPy arrays for input and kernel
   - Calls `conv_1d()` function which wraps our operation in MAX Graph
   - Converts NumPy arrays to [MAX driver](https://docs.modular.com/max/api/python/driver) Tensors with `Tensor.from_numpy(input).to(device)`
   - Loads the custom operation package with `custom_extensions=[mojo_kernels]`

2. **Graph building**:
   - Defines input and output tensor types with [TensorType](https://docs.modular.com/max/api/python/graph/type/#max.graph.type.TensorType)
   - Specifies parameters for our operation via `parameters={...}`
   - Creates a computation graph with [`Graph("conv_1d_graph", ...)`](https://docs.modular.com/max/api/python/graph/Graph)
   - Calls our operation using [`ops.custom(name="conv1d", ...)`](https://docs.modular.com/max/api/python/graph/ops#custom)

3. **Custom op registration**:
   - The `@compiler.register("conv1d")` decorator exposes our operation to MAX Graph. See [@compiler.register](https://docs.modular.com/mojo/manual/decorators/compiler-register/)
   - The `execute` method parameters define the interface (inputs, outputs, context)
   - Input/output tensors are converted to LayoutTensors for use in our kernel
   - Device context manages GPU memory allocation and kernel execution

4. **Kernel execution**:
   - When [model.execute(...)]() is called, our `conv1d_kernel` receives the data
   - GPU thread configuration is set with `grid_dim` and `block_dim`
   - Results are transferred back to CPU with `result.to(CPU())`
   - NumPy verification compares our results with the expected output

### Key components in detail

1. **Custom Op Structure**:

   ```mojo
   @compiler.register("conv1d")
   struct Conv1DCustomOp:
       @staticmethod
       fn execute[target: StaticString, input_size: Int, conv_size: Int, dtype: DType = DType.float32](
           output: OutputTensor[rank=1],
           input: InputTensor[dtype = output.dtype, rank = output.rank],
           kernel: InputTensor[dtype = output.dtype, rank = output.rank],
           ctx: DeviceContextPtr,
       ) raises:
           # Implementation
   ```

   - `target` indicates the device type ("gpu" or "cpu")
   - `input_size` and `conv_size` are parameters passed from Python
   - Tensor types ensure correct shape and type checking
   - Return type is `raises` for proper error handling

2. **Tensor Conversion**:

   ```mojo
   output_tensor = output.to_layout_tensor()
   input_tensor = input.to_layout_tensor()
   kernel_tensor = kernel.to_layout_tensor()
   ```

   - MAX Graph tensors are converted to Mojo LayoutTensors
   - This allows our kernel to work with them directly
   - The layouts are extracted for compile-time optimization

3. **Device Context Usage**:

   ```mojo
   gpu_ctx = ctx.get_device_context()
   gpu_ctx.enqueue_memset(...)  # Zero output buffer
   gpu_ctx.enqueue_function_checked[..., ...](...) # Schedule kernel
   ```

   - Device context manages GPU resources
   - Memory operations ensure correct buffer state
   - Function enqueueing schedules our kernel for execution

This solution demonstrates the complete flow from Python data through MAX Graph to GPU execution and back, leveraging Mojo's powerful type system and parametric functions to create efficient, type-safe, accelerated operations.

</details>

## Understanding MAX Graph custom ops

> Check out the follow tutorials for more details:
>
> - [Get started with MAX Graph in Python](https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python/)
> - [MAX Graph custom op for GPUs](https://docs.modular.com/max/tutorials/build-custom-ops/)

### Custom op registration

The core of creating a custom operation is the `@compiler.register` decorator and the associated structure:

```mojo
@compiler.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    fn execute[...](
        output: OutputTensor[rank=1],
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        kernel: InputTensor[type = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Implementation here
```

Key components of the registration:

- The **name** passed to the decorator (`"conv1d"`) is what Python code will use to call this operation
- The **struct** must have an `execute` method with the correct signature
- **OutputTensor** and **InputTensor** types define the interface for Python data
- **DeviceContextPtr** provides access to the execution environment

### Packaging custom ops

Before the custom operation can be used from Python, it needs to be packaged:

```bash
mojo package op -o op.mojopkg
```

This command:

1. Compiles the Mojo code into a deployable package
2. Creates the necessary metadata for MAX Graph to understand the operation
3. Produces a binary artifact (`op.mojopkg`) that can be loaded by Python

The package must be placed in a location where MAX Graph can find it, typically in a directory accessible to the Python code.

### Python integration

On the Python side, here's how the custom operation is used:

```python
# Path to the directory containing our Mojo operations
mojo_kernels = Path(__file__).parent / "op"

# Configure our graph with the custom conv1d operation
with Graph(
    "conv_1d_graph",
    input_types=[...],
    custom_extensions=[mojo_kernels],  # Load our custom op package
) as graph:
    # Define inputs to the graph
    input_value, kernel_value = graph.inputs

    # Use our custom operation by name
    output = ops.custom(
        name="conv1d",  # Must match the name in @compiler.register
        values=[input_value, kernel_value],
        out_types=[...],
        parameters={
            "input_size": input_tensor.shape[0],
            "conv_size": kernel_tensor.shape[0],
            "dtype": dtype,
        },
    )[0].tensor
```

The key elements are:

1. Specifying the path to our custom operations with `custom_extensions`
2. Calling `ops.custom` with the registered operation name
3. Passing input values and parameters that match our operation's signature
