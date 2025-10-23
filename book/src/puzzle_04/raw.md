## Overview

Implement a kernel that adds 10 to each position of 2D square matrix `a` and stores it in 2D square matrix `output`.

**Note:** _You have more threads than positions_.

## Key concepts

In this puzzle, you'll learn about:

- Working with 2D thread indices (`thread_idx.x`, `thread_idx.y`)
- Converting 2D coordinates to 1D memory indices
- Handling boundary checks in two dimensions

The key insight is understanding how to map from 2D thread coordinates \\((i,j)\\) to elements in a row-major matrix of size \\(n \times n\\), while ensuring thread indices are within bounds.

- **2D indexing**: Each thread has a unique \\((i,j)\\) position
- **Memory layout**: Row-major ordering maps 2D to 1D memory
- **Guard condition**: Need bounds checking in both dimensions
- **Thread bounds**: More threads \\((3 \times 3)\\) than matrix elements \\((2 \times 2)\\)

## Code to complete

```mojo
{{#include ../../../problems/p04/p04.mojo:add_10_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04.mojo" class="filename">View full file: problems/p04/p04.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Get 2D indices: `row = thread_idx.y`, `col = thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard add 10 in row-major way!

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
pixi run p04
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p04
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p04
```

  </div>
  <div class="tab-content">

```bash
uv run poe p04
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p04/p04.mojo:add_10_2d_solution}}
```

<div class="solution-explanation">

This solution:

1. Get 2D indices:  `row = thread_idx.y`, `col = thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard: `output[row * size + col] = a[row * size + col] + 10.0`

</div>
</details>
