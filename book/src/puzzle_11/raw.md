## Overview

Implement a kernel that compute the running sum of the last 3 positions of vector `a` and stores it in vector `output`.

**Note:** _You have 1 thread per position. You only need 1 global read and 1 global write per thread._

## Key concepts

In this puzzle, you'll learn about:

- Using shared memory for sliding window operations
- Handling boundary conditions in pooling
- Coordinating thread access to neighboring elements

The key insight is understanding how to efficiently access a window of elements using shared memory, with special handling for the first elements in the sequence.

## Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Window size: 3 elements
- Shared memory: `TPB` elements

Notes:

- **Window access**: Each output depends on up to 3 previous elements
- **Edge handling**: First two positions need special treatment
- **Memory pattern**: One shared memory load per thread
- **Thread sync**: Coordination before window operations

## Code to complete

```mojo
{{#include ../../../problems/p11/p11.mojo:pooling}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p11/p11.mojo" class="filename">View full file: problems/p11/p11.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Load data and call `barrier()`
2. Special cases: `output[0] = shared[0]`, `output[1] = shared[0] + shared[1]`
3. General case: `if 1 < global_i < size`
4. Sum three elements: `shared[local_i - 2] + shared[local_i - 1] + shared[local_i]`

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
pixi run p11
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p11
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p11
```

  </div>
  <div class="tab-content">

```bash
uv run poe p11
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
{{#include ../../../solutions/p11/p11.mojo:pooling_solution}}
```

<div class="solution-explanation">

The solution implements a sliding window sum using shared memory with these key steps:

1. **Shared memory setup**
   - Allocates `TPB` elements in shared memory:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - Each thread loads one element from global memory
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

   - Window calculation using local indices:

     ```txt
     # Sliding window of 3 elements
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **Memory access pattern**
   - One global read per thread into shared memory
   - One global write per thread from shared memory
   - Uses shared memory for efficient neighbor access
   - Maintains coalesced memory access pattern

This approach optimizes performance through:

- Minimal global memory access
- Fast shared memory neighbor lookups
- Clean boundary handling
- Efficient memory coalescing

The final output shows the cumulative window sums:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

</div>
</details>
