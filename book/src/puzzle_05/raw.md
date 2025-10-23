## Overview

Implement a kernel that broadcast adds vector `a` and vector `b` and stores it in 2D matrix `output`.

**Note:** _You have more threads than positions._

## Key concepts

In this puzzle, you'll learn about:

- Broadcasting 1D vectors across different dimensions
- Using 2D thread indices for broadcast operations
- Handling boundary conditions in broadcast patterns

The key insight is understanding how to map elements from two 1D vectors to create a 2D output matrix through broadcasting, while handling thread bounds correctly.

- **Broadcasting**: Each element of `a` combines with each element of `b`
- **Thread mapping**: 2D thread grid \\((3 \times 3)\\) for \\(2 \times 2\\) output
- **Vector access**: Different access patterns for `a` and `b`
- **Bounds checking**: Guard against threads outside matrix dimensions

## Code to complete

```mojo
{{#include ../../../problems/p05/p05.mojo:broadcast_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p05/p05.mojo" class="filename">View full file: problems/p05/p05.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Get 2D indices: `row = thread_idx.y`, `col = thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard: think about how to broadcast values of `a` and `b`

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
pixi run p05
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p05
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p05
```

  </div>
  <div class="tab-content">

```bash
uv run poe p05
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 2.0, 11.0, 12.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p05/p05.mojo:broadcast_add_solution}}
```

<div class="solution-explanation">

This solution demonstrates fundamental GPU broadcasting concepts without LayoutTensor abstraction:

1. **Thread to matrix mapping**
   - Uses `thread_idx.y` for row access and `thread_idx.x` for column access
   - Direct mapping from 2D thread grid to output matrix elements
   - Handles excess threads (3×3 grid) for 2×2 output matrix

2. **Broadcasting mechanics**
   - Vector `a` broadcasts horizontally: same `a[col]` used across each row
   - Vector `b` broadcasts vertically: same `b[row]` used across each column
   - Output combines both vectors through addition

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **Bounds checking**
   - Single guard condition `row < size and col < size` handles both dimensions
   - Prevents out-of-bounds access for both input vectors and output matrix
   - Required due to 3×3 thread grid being larger than 2×2 data

Compare this with the LayoutTensor version to see how the abstraction simplifies broadcasting operations while maintaining the same underlying concepts.
</div>
</details>
