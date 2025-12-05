# Puzzle 15: Axis Sum

## Overview

Implement a kernel that computes a sum over each row of 2D matrix `a` and stores it in `output` using LayoutTensor.

<img src="./media/15-w.png" alt="Axis sum visualization" class="light-mode-img">
<img src="./media/15-b.png" alt="Axis sum visualization" class="dark-mode-img">

## Key concepts

This puzzle covers:

- Parallel reduction along matrix dimensions using LayoutTensor
- Using block coordinates for data partitioning
- Efficient shared memory reduction patterns
- Working with multi-dimensional tensor layouts

The key insight is understanding how to map thread blocks to matrix rows and perform efficient parallel reduction within each block while leveraging LayoutTensor's dimensional indexing.

## Configuration

- Matrix dimensions: \\(\\text{BATCH} \\times \\text{SIZE} = 4 \\times 6\\)
- Threads per block: \\(\\text{TPB} = 8\\)
- Grid dimensions: \\(1 \\times \\text{BATCH}\\)
- Shared memory: \\(\\text{TPB}\\) elements per block
- Input layout: `Layout.row_major(BATCH, SIZE)`
- Output layout: `Layout.row_major(BATCH, 1)`

Matrix visualization:

```txt
Row 0: [0, 1, 2, 3, 4, 5]       → Block(0,0)
Row 1: [6, 7, 8, 9, 10, 11]     → Block(0,1)
Row 2: [12, 13, 14, 15, 16, 17] → Block(0,2)
Row 3: [18, 19, 20, 21, 22, 23] → Block(0,3)
```

## Code to Complete

```mojo
{{#include ../../../problems/p15/p15.mojo:axis_sum}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p15/p15.mojo" class="filename">View full file: problems/p15/p15.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use `batch = block_idx.y` to select row
2. Load elements: `cache[local_i] = a[batch, local_i]`
3. Perform parallel reduction with halving stride
4. Thread 0 writes final sum to `output[batch]`

</div>
</details>

## Running the Code

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
pixi run p15
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p15
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p15
```

  </div>
  <div class="tab-content">

```bash
uv run poe p15
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: DeviceBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([15.0, 51.0, 87.0, 123.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p15/p15.mojo:axis_sum_solution}}
```

<div class="solution-explanation">

The solution implements a parallel row-wise sum reduction for a 2D matrix using LayoutTensor. Here's a comprehensive breakdown:

### Matrix layout and block mapping

```txt
Input Matrix (4×6) with LayoutTensor:                Block Assignment:
[[ a[0,0]  a[0,1]  a[0,2]  a[0,3]  a[0,4]  a[0,5] ] → Block(0,0)
 [ a[1,0]  a[1,1]  a[1,2]  a[1,3]  a[1,4]  a[1,5] ] → Block(0,1)
 [ a[2,0]  a[2,1]  a[2,2]  a[2,3]  a[2,4]  a[2,5] ] → Block(0,2)
 [ a[3,0]  a[3,1]  a[3,2]  a[3,3]  a[3,4]  a[3,5] ] → Block(0,3)
```

### Parallel reduction process

1. **Initial Data Loading**:

   ```txt
   Block(0,0): cache = [a[0,0] a[0,1] a[0,2] a[0,3] a[0,4] a[0,5] * *]  // * = padding
   Block(0,1): cache = [a[1,0] a[1,1] a[1,2] a[1,3] a[1,4] a[1,5] * *]
   Block(0,2): cache = [a[2,0] a[2,1] a[2,2] a[2,3] a[2,4] a[2,5] * *]
   Block(0,3): cache = [a[3,0] a[3,1] a[3,2] a[3,3] a[3,4] a[3,5] * *]
   ```

2. **Reduction Steps** (for Block 0,0):

   ```txt
   Initial:  [0  1  2  3  4  5  *  *]
   Stride 4: [4  5  6  7  4  5  *  *]
   Stride 2: [10 12 6  7  4  5  *  *]
   Stride 1: [15 12 6  7  4  5  *  *]
   ```

### Key implementation features

1. **Layout Configuration**:
   - Input: row-major layout (BATCH × SIZE)
   - Output: row-major layout (BATCH × 1)
   - Each block processes one complete row

2. **Memory Access Pattern**:
   - LayoutTensor 2D indexing for input: `a[batch, local_i]`
   - Shared memory for efficient reduction
   - LayoutTensor 2D indexing for output: `output[batch, 0]`

3. **Parallel Reduction Logic**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           cache[local_i] += cache[local_i + stride]
       barrier()
       stride //= 2
   ```

   **Note**: This implementation has a potential race condition where threads simultaneously read from and write to shared memory during the same iteration. A safer approach would separate the read and write phases:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       var temp_val: output.element_type = 0
       if local_i < stride:
           temp_val = cache[local_i + stride]  # Read phase
       barrier()
       if local_i < stride:
           cache[local_i] += temp_val  # Write phase
       barrier()
       stride //= 2
   ```

4. **Output Writing**:

   ```mojo
   if local_i == 0:
       output[batch, 0] = cache[0]  --> One result per batch
   ```

### Performance optimizations

1. **Memory Efficiency**:
   - Coalesced memory access through LayoutTensor
   - Shared memory for fast reduction
   - Single write per row result

2. **Thread Utilization**:
   - Perfect load balancing across rows
   - No thread divergence in main computation
   - Efficient parallel reduction pattern

3. **Synchronization**:
   - Minimal barriers (only during reduction)
   - Independent processing between rows
   - No inter-block communication needed
   - **Race condition consideration**: The current implementation may have read-write hazards during parallel reduction that could be resolved with explicit read-write phase separation

### Complexity analysis

- Time: \\(O(\log n)\\) per row, where n is row length
- Space: \\(O(TPB)\\) shared memory per block
- Total parallel time: \\(O(\log n)\\) with sufficient threads

</div>
</details>
