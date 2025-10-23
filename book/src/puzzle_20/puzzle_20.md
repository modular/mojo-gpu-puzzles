# Puzzle 20: 1D Convolution Op

> ## From MAX Graph to PyTorch custom ops
>
> We're now entering Part V of our GPU puzzle journey: **PyTorch Custom Operations**.
>
> In [Puzzle 17](../puzzle_17/puzzle_17.md), we learned how to integrate Mojo GPU kernels with Python using MAX Graph. Now we'll explore how to:
>
> - Use the same Mojo kernel with PyTorch's CustomOpLibrary
> - Integrate with PyTorch's tensor system and autograd
> - Compare MAX Graph vs PyTorch approaches for custom operations
> - Understand the critical pattern of explicit output tensor allocation
>
> This transition shows how the same optimized GPU kernel can work with different Python integration approaches.

## Overview

In this puzzle, we'll take the exact same 1D convolution kernel from [Puzzle 17](../puzzle_17/puzzle_17.md) and integrate it with PyTorch using the [CustomOpLibrary](https://docs.modular.com/max/api/python/torch/CustomOpLibrary/) instead of MAX Graph.

The key learning here is that **the same Mojo kernel works unchanged** - only the Python integration layer differs between MAX Graph and PyTorch approaches.

## Code to complete

To complete this puzzle, you need to fill in one line to call the custom operation:

```python
{{#include ../../../problems/p20/p20.py:conv1d_pytorch}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p20/p20.py" class="filename">View full file: problems/p20/p20.py</a>

You can run the puzzle with:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p20
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p20
```

  </div>
  <div class="tab-content">

```bash
uv run poe p20
```

  </div>
</div>

When successful, you should see output similar to:

```
Puzzle 20: From MAX Graph to PyTorch Custom Ops
============================================================
Input array: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
Convolution kernel: [0. 1. 2. 3.]

NumPy reference result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]

Testing PyTorch Custom Op (device: cuda)
----------------------------------------
PyTorch custom op result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
✅ PyTorch custom op verification PASSED

Comparing with MAX Graph approach (like p15)
--------------------------------------------
MAX Graph result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
✅ MAX Graph verification PASSED
✅ PyTorch and MAX Graph results MATCH
```

## Solution

<details class="solution-details">
<summary></summary>

The solution requires calling the compiled custom operation with the proper arguments:

```python
{{#include ../../../solutions/p20/p20.py:conv1d_pytorch_call}}
```

<div class="solution-explanation">

This solution demonstrates several critical concepts:

### 1. **torch.compile() integration**

The solution shows `torch.compile` integration

```python
torch.compile(conv1d)(output_tensor, input_tensor, kernel_tensor)
```

### 2. **Explicit Output Tensor Allocation**

```python
output_tensor = torch.empty_like(input_tensor)
```

- Unlike MAX Graph which handles output allocation automatically
- PyTorch CustomOpLibrary requires **pre-allocated output tensors**
- The Mojo operation signature expects `(out, input, kernel)` order

### 3. **Parameter Dictionary**

```python
ops.conv1d[{"input_size": input_tensor.shape[0], "conv_size": kernel_tensor.shape[0]}]
```

- Parameters are passed as a dictionary to the operation
- These become compile-time parameters in the Mojo kernel
- Must match the parameter names in the Mojo `@staticmethod fn execute` signature

### 4. **Same Kernel, Different Integration**

The underlying Mojo kernel (`conv1d_kernel`) is identical to Puzzle 17:

- Same GPU kernel code
- Same memory access patterns
- Same computational logic
- Only the Python wrapper layer changes

</div>

</details>

## Key concepts

This puzzle illustrates several important patterns for PyTorch custom operations:

| Concept | MAX Graph (p15) | PyTorch CustomOpLibrary (p18) |
|---------|-----------------|-------------------------------|
| **Output Allocation** | Automatic | Manual (`torch.empty_like()`) |
| **Operation Call** | `ops.custom(...)` | `torch.compile(op)(...)` |
| **Parameter Passing** | `parameters={...}` | `op[{...}]` |
| **Device Management** | Explicit device context | PyTorch tensor device |
| **Memory Management** | MAX Graph tensors | PyTorch tensors |

### Critical pattern: Explicit output tensor allocation

The most important difference is that PyTorch CustomOpLibrary requires **explicit output tensor allocation**:

```python
# ❌ This won't work - no output tensor
result = torch.compile(conv1d)(input_tensor, kernel_tensor)

# ✅ This works - pre-allocated output tensor
output_tensor = torch.empty_like(input_tensor)
torch.compile(conv1d)(output_tensor, input_tensor, kernel_tensor)
```

This pattern ensures:

- Memory is allocated on the correct device
- Output tensor has the right shape and dtype
- The Mojo kernel can write directly to the output buffer

### torch.compile() integration

`torch.compile()` is essential because it:

- Handles memory layout conversion between PyTorch and Mojo
- Manages device synchronization (CPU ↔ GPU)
- Optimizes tensor format conversion
- Provides proper error handling for memory operations

_Note: Without `torch.compile()`, you might encounter `std::bad_alloc` errors because the raw operation can't handle PyTorch's tensor memory management._

## Debugging custom operations

Common issues and solutions:

1. **Memory Allocation Errors**: Always use `torch.compile()`
2. **Wrong Output Shape**: Ensure output tensor matches expected dimensions
3. **Device Mismatch**: All tensors must be on the same device
4. **Parameter Errors**: Verify parameter names match Mojo operation signature

The debug approach: Compare your PyTorch results with the MAX Graph reference implementation that runs the same kernel.
