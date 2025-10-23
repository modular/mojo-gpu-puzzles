## Overview

Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`.

**Note:** _You have fewer threads per block than the size of `a` in both directions._

## Key concepts

In this puzzle, you'll learn about:

- Working with 2D block and thread arrangements
- Handling matrix data larger than block size
- Converting between 2D and linear memory access

The key insight is understanding how to coordinate multiple blocks of threads to process a 2D matrix that's larger than a single block's dimensions.

## Configuration

- **Matrix size**: \\(5 \times 5\\) elements
- **2D blocks**: Each block processes a \\(3 \times 3\\) region
- **Grid layout**: Blocks arranged in \\(2 \times 2\\) grid
- **Total threads**: \\(36\\) for \\(25\\) elements
- **Memory pattern**: Row-major storage for 2D data
- **Coverage**: Ensuring all matrix elements are processed

## Code to complete

```mojo
{{#include ../../../problems/p07/p07.mojo:add_10_blocks_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p07/p07.mojo" class="filename">View full file: problems/p07/p07.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate global indices: `row = block_dim.y * block_idx.y + thread_idx.y`, `col = block_dim.x * block_idx.x + thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard: think about how to add 10 in row-major way!

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
pixi run p07
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p07
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p07
```

  </div>
  <div class="tab-content">

```bash
uv run poe p07
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
{{#include ../../../solutions/p07/p07.mojo:add_10_blocks_2d_solution}}
```

<div class="solution-explanation">

This solution demonstrates key concepts of 2D block-based processing with raw memory:

1. **2D thread indexing**
   - Global row: `block_dim.y * block_idx.y + thread_idx.y`
   - Global col: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps thread grid to matrix elements:

     ```txt
     5×5 matrix with 3×3 blocks:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = thread exists but outside matrix bounds)

2. **Memory layout**
   - Row-major linear memory: `index = row * size + col`
   - Example for 5×5 matrix:

     ```txt
     2D indices:    Linear memory:
     (2,1) -> 11   [00 01 02 03 04]
                   [05 06 07 08 09]
                   [10 11 12 13 14]
                   [15 16 17 18 19]
                   [20 21 22 23 24]
     ```

3. **Bounds checking**
   - Guard `row < size and col < size` handles:
     - Excess threads in partial blocks
     - Edge cases at matrix boundaries
     - 2×2 block grid with 3×3 threads each = 36 threads for 25 elements

4. **Block coordination**
   - Each 3×3 block processes part of 5×5 matrix
   - 2×2 grid of blocks ensures full coverage
   - Overlapping threads handled by bounds check
   - Efficient parallel processing across blocks

This pattern shows how to handle 2D data larger than block size while maintaining efficient memory access and thread coordination.
</div>
</details>
