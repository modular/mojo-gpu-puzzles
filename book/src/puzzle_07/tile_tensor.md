# LayoutTensor Version

## Overview

Implement a kernel that adds 10 to each position of 2D LayoutTensor `a` and stores it in 2D LayoutTensor `output`.

**Note:** _You have fewer threads per block than the size of `a` in both directions._

## Key concepts

In this puzzle, you'll learn about:

- Using `LayoutTensor` with multiple blocks
- Handling large matrices with 2D block organization
- Combining block indexing with `LayoutTensor` access

The key insight is that `LayoutTensor` simplifies 2D indexing while still requiring proper block coordination for large matrices.

## Configuration

- **Matrix size**: \\(5 \times 5\\) elements
- **Layout handling**: `LayoutTensor` manages row-major organization
- **Block coordination**: Multiple blocks cover the full matrix
- **2D indexing**: Natural \\((i,j)\\) access with bounds checking
- **Total threads**: \\(36\\) for \\(25\\) elements
- **Thread mapping**: Each thread processes one matrix element

## Code to complete

```mojo
{{#include ../../../problems/p07/p07_layout_tensor.mojo:add_10_blocks_2d_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p07/p07_layout_tensor.mojo" class="filename">View full file: problems/p07/p07_layout_tensor.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate global indices: `row = block_dim.y * block_idx.y + thread_idx.y`, `col = block_dim.x * block_idx.x + thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard: think about how to add 10 to 2D LayoutTensor

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
pixi run p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p07_layout_tensor
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, ... , 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, ... , 34.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p07/p07_layout_tensor.mojo:add_10_blocks_2d_layout_tensor_solution}}
```

<div class="solution-explanation">

This solution demonstrates how LayoutTensor simplifies 2D block-based processing:

1. **2D thread indexing**
   - Global row: `block_dim.y * block_idx.y + thread_idx.y`
   - Global col: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps thread grid to tensor elements:

     ```txt
     5×5 tensor with 3×3 blocks:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = thread exists but outside tensor bounds)

2. **LayoutTensor benefits**
   - Natural 2D indexing: `tensor[row, col]` instead of manual offset calculation
   - Automatic memory layout optimization
   - Example access pattern:

     ```txt
     Raw memory:         LayoutTensor:
     row * size + col    tensor[row, col]
     (2,1) -> 11        (2,1) -> same element
     ```

3. **Bounds checking**
   - Guard `row < size and col < size` handles:
     - Excess threads in partial blocks
     - Edge cases at tensor boundaries
     - Automatic memory layout handling by LayoutTensor
     - 36 threads (2×2 blocks of 3×3) for 25 elements

4. **Block coordination**
   - Each 3×3 block processes part of 5×5 tensor
   - LayoutTensor handles:
     - Memory layout optimization
     - Efficient access patterns
     - Block boundary coordination
     - Cache-friendly data access

This pattern shows how LayoutTensor simplifies 2D block processing while maintaining optimal memory access patterns and thread coordination.
</div>
</details>
