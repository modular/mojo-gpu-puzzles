# Puzzle 2: Zip

## Overview

Implement a kernel that adds together each position of vector `a` and vector `b` and stores it in `output`.

**Note:** _You have 1 thread per position._

<img src="./media/02.png" alt="Zip" class="light-mode-img">
<img src="./media/02d.png" alt="Zip" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- Processing multiple input arrays in parallel
- Element-wise operations with multiple inputs
- Thread-to-data mapping across arrays
- Memory access patterns with multiple arrays

For each thread \\(i\\): \\[\Large output[i] = a[i] + b[i]\\]

### Memory access pattern

```txt
Thread 0:  a[0] + b[0] → output[0]
Thread 1:  a[1] + b[1] → output[1]
Thread 2:  a[2] + b[2] → output[2]
...
```

💡 **Note**: Notice how we're now managing three arrays (`a`, `b`, `output`) in our kernel. As we progress to more complex operations, managing multiple array accesses will become increasingly challenging.

## Code to complete

```mojo
{{#include ../../../problems/p02/p02.mojo:add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p02/p02.mojo" class="filename">View full file: problems/p02/p02.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `thread_idx.x` in `i`
2. Add `a[i]` and `b[i]`
3. Store result in `output[i]`

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
pixi run p02
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p02
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p02
```

  </div>
  <div class="tab-content">

```bash
uv run poe p02
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 2.0, 4.0, 6.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p02/p02.mojo:add_solution}}
```

<div class="solution-explanation">

This solution:

- Gets thread index with `i = thread_idx.x`
- Adds values from both arrays: `output[i] = a[i] + b[i]`

</div>
</details>

### Looking ahead

While this direct indexing works for simple element-wise operations, consider:

- What if arrays have different layouts?
- What if we need to broadcast one array to another?
- How to ensure coalesced access across multiple arrays?

These questions will be addressed when we [introduce LayoutTensor in Puzzle 4](../puzzle_04/).
