# Puzzle 6: Blocks

## Overview

Implement a kernel that adds 10 to each position of vector `a` and stores it in `output`.

**Note:** _You have fewer threads per block than the size of a._

<img src="./media/06.png" alt="Blocks visualization" class="light-mode-img">
<img src="./media/06d.png" alt="Blocks visualization" class="dark-mode-img">

## Key concepts

This puzzle covers:

- Processing data larger than thread block size
- Coordinating multiple blocks of threads
- Computing global thread positions

The key insight is understanding how blocks of threads work together to process data that's larger than a single block's capacity, while maintaining correct element-to-thread mapping.

## Code to complete

```mojo
{{#include ../../../problems/p06/p06.mojo:add_10_blocks}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p06/p06.mojo" class="filename">View full file: problems/p06/p06.mojo</a>

> Note: The `LayoutTensor` variant of this puzzle is very similar so we leave it to the reader.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate global index: `i = block_dim.x * block_idx.x + thread_idx.x`
2. Add guard: `if i < size`
3. Inside guard: `output[i] = a[i] + 10.0`

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
pixi run p06
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p06
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p06
```

  </div>
  <div class="tab-content">

```bash
uv run poe p06
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p06/p06.mojo:add_10_blocks_solution}}
```

<div class="solution-explanation">

This solution covers key concepts of block-based GPU processing:

1. **Global thread indexing**
   - Combines block and thread indices: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps each thread to a unique global position
   - Example for 3 threads per block:

     ```txt
     Block 0: [0 1 2]
     Block 1: [3 4 5]
     Block 2: [6 7 8]
     ```

2. **Block coordination**
   - Each block processes a contiguous chunk of data
   - Block size (3) < Data size (9) requires multiple blocks
   - Automatic work distribution across blocks:

     ```txt
     Data:    [0 1 2 3 4 5 6 7 8]
     Block 0: [0 1 2]
     Block 1:       [3 4 5]
     Block 2:             [6 7 8]
     ```

3. **Bounds checking**
   - Guard condition `i < size` handles edge cases
   - Prevents out-of-bounds access when size isn't perfectly divisible by block size
   - Essential for handling partial blocks at the end of data

4. **Memory access pattern**
   - Coalesced memory access: threads in a block access contiguous memory
   - Each thread processes one element: `output[i] = a[i] + 10.0`
   - Block-level parallelism provides efficient memory bandwidth utilization

This pattern forms the foundation for processing large datasets that exceed the size of a single thread block.
</div>
</details>
