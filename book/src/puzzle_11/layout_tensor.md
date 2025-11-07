## Overview

Implement a kernel that compute the running sum of the last 3 positions of 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:** _You have 1 thread per position. You only need 1 global read and 1 global write per thread._

## Key concepts

In this puzzle, you'll learn about:

- Using LayoutTensor for sliding window operations
- Managing shared memory with LayoutTensor address_space that we saw in [puzzle_08](../puzzle_08/layout_tensor.md)
- Efficient neighbor access patterns
- Boundary condition handling

The key insight is how LayoutTensor simplifies shared memory management while maintaining efficient window-based operations.

## Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Window size: 3 elements
- Shared memory: `TPB` elements

Notes:

- **LayoutTensor allocation**: Use `LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **Window access**: Natural indexing for 3-element windows
- **Edge handling**: Special cases for first two positions
- **Memory pattern**: One shared memory load per thread

## Code to complete

```mojo
{{#include ../../../problems/p11/p11_layout_tensor.mojo:pooling_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p11/p11_layout_tensor.mojo" class="filename">View full file: problems/p11/p11_layout_tensor.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Create shared memory with LayoutTensor using address_space
2. Load data with natural indexing: `shared[local_i] = a[global_i]`
3. Handle special cases for first two elements
4. Use shared memory for window operations
5. Guard against out-of-bounds access

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p11_layout_tensor
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p11/p11_layout_tensor.mojo:pooling_layout_tensor_solution}}
```

<div class="solution-explanation">

The solution implements a sliding window sum using LayoutTensor with these key steps:

1. **Shared memory setup**
   - LayoutTensor creates block-local storage with address_space:

     ```txt
     shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
     ```

   - Each thread loads one element:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - `barrier()` ensures all data is loaded

2. **Boundary cases**
   - Position 0: Single element

     ```txt
     output[0] = shared[0] = 0.0
     ```

   - Position 1: Sum of first two elements

     ```txt
     output[1] = shared[0] + shared[1] = 0.0 + 1.0 = 1.0
     ```

3. **Main window operation**
   - For positions 2 and beyond:

     ```txt
     Position 2: shared[0] + shared[1] + shared[2] = 0.0 + 1.0 + 2.0 = 3.0
     Position 3: shared[1] + shared[2] + shared[3] = 1.0 + 2.0 + 3.0 = 6.0
     Position 4: shared[2] + shared[3] + shared[4] = 2.0 + 3.0 + 4.0 = 9.0
     ...
     ```

   - Natural indexing with LayoutTensor:

     ```txt
     # Sliding window of 3 elements
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **Memory access pattern**
   - One global read per thread into shared tensor
   - Efficient neighbor access through shared memory
   - LayoutTensor benefits:
     - Automatic bounds checking
     - Natural window indexing
     - Layout-aware memory access
     - Type safety throughout

This approach combines the performance of shared memory with LayoutTensor's safety and ergonomics:

- Minimizes global memory access
- Simplifies window operations
- Handles boundaries cleanly
- Maintains coalesced access patterns

The final output shows the cumulative window sums:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

</div>
</details>
