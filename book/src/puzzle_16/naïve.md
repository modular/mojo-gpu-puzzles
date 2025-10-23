# Naïve Matrix Multiplication

## Overview

Implement a kernel that multiplies square matrices \\(A\\) and \\(B\\) and stores the result in \\(\text{output}\\).
This is the most straightforward implementation where each thread computes one element of the output matrix.

## Key concepts

This puzzle covers:

- 2D thread organization for matrix operations
- Global memory access patterns
- Matrix indexing in row-major layout
- Thread-to-output element mapping

The key insight is understanding how to map 2D thread indices to matrix elements and compute dot products in parallel.

## Configuration

- Matrix size: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- Threads per block: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- Grid dimensions: \\(1 \\times 1\\)

Layout configuration:

- Input A: `Layout.row_major(SIZE, SIZE)`
- Input B: `Layout.row_major(SIZE, SIZE)`
- Output: `Layout.row_major(SIZE, SIZE)`

## Code to complete

```mojo
{{#include ../../../problems/p16/p16.mojo:naive_matmul}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">View full file: problems/p16/p16.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate `row` and `col` from thread indices
2. Check if indices are within `size`
3. Accumulate products in a local variable
4. Write final sum to correct output position

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p16 --naive
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p16 --naive
```

  </div>
  <div class="tab-content">

```bash
uv run poe p16 --naive
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([4.0, 6.0, 12.0, 22.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p16/p16.mojo:naive_matmul_solution}}
```

<div class="solution-explanation">

The naive matrix multiplication using LayoutTensor follows this basic approach:

### Matrix layout (2×2 example)

```txt
Matrix A:          Matrix B:                   Output C:
[a[0,0] a[0,1]]    [b[0,0] b[0,1]]             [c[0,0] c[0,1]]
[a[1,0] a[1,1]]    [b[1,0] b[1,1]]             [c[1,0] c[1,1]]
```

### Implementation details

1. **Thread mapping**:

   ```mojo
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x
   ```

2. **Memory access pattern**:
   - Direct 2D indexing: `a[row, k]`
   - Transposed access: `b[k, col]`
   - Output writing: `output[row, col]`

3. **Computation flow**:

   ```mojo
   # Use var for mutable accumulator with tensor's element type
   var acc: output.element_type = 0

   # @parameter for compile-time loop unrolling
   @parameter
   for k in range(size):
       acc += a[row, k] * b[k, col]
   ```

### Key language features

1. **Variable declaration**:
   - The use of `var` in `var acc: output.element_type = 0` allows for type inference with `output.element_type` ensures type compatibility with the output tensor
   - Initialized to zero before accumulation

2. **Loop pptimization**:
   - [`@parameter`](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement) decorator unrolls the loop at compile time
   - Improves performance for small, known matrix sizes
   - Enables better instruction scheduling

### Performance characteristics

1. **Memory access**:
   - Each thread makes `2 x SIZE` global memory reads
   - One global memory write per thread
   - No data reuse between threads

2. **Computational efficiency**:
   - Simple implementation but suboptimal performance
   - Many redundant global memory accesses
   - No use of fast shared memory

3. **Limitations**:
   - High global memory bandwidth usage
   - Poor data locality
   - Limited scalability for large matrices

This naive implementation serves as a baseline for understanding matrix multiplication on GPUs, highlighting the need for optimization in memory access patterns.
</div>
</details>
