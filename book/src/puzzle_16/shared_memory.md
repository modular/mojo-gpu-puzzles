# Shared Memory Matrix Multiplication

## Overview

This puzzle implements matrix multiplication for square matrices \\(A\\) and \\(B\\), storing results in \\(\text{output}\\) while leveraging shared memory to optimize memory access patterns. The implementation preloads matrix blocks into shared memory before performing computations.

## Key concepts

This puzzle covers:

- Block-local memory management with LayoutTensor
- Thread synchronization patterns
- Memory access optimization using shared memory
- Collaborative data loading with 2D indexing
- Efficient use of LayoutTensor for matrix operations

The central concept involves utilizing fast shared memory through LayoutTensor to minimize costly global memory accesses.

## Configuration

- Matrix size: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- Threads per block: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- Grid dimensions: \\(1 \\times 1\\)

Layout configuration:

- Input A: `Layout.row_major(SIZE, SIZE)`
- Input B: `Layout.row_major(SIZE, SIZE)`
- Output: `Layout.row_major(SIZE, SIZE)`
- Shared Memory: Two `TPB × TPB` LayoutTensors

Memory organization:

```txt
Global Memory (LayoutTensor):          Shared Memory (LayoutTensor):
A[i,j]: Direct access                  a_shared[local_row, local_col]
B[i,j]: Direct access                  b_shared[local_row, local_col]
```

## Code to complete

```mojo
{{#include ../../../problems/p16/p16.mojo:single_block_matmul}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">View full file: problems/p16/p16.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Load matrices to shared memory using global and local indices
2. Call `barrier()` after loading
3. Compute dot product using shared memory indices
4. Check array bounds for all operations

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
pixi run p16 --single-block
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p16 --single-block
```

  </div>
  <div class="tab-content">

```bash
uv run poe p16 --single-block
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
{{#include ../../../solutions/p16/p16.mojo:single_block_matmul_solution}}
```

<div class="solution-explanation">

The shared memory implementation with LayoutTensor improves performance through efficient memory access patterns:

### Memory organization

```txt
Input Tensors (2×2):                Shared Memory (3×3):
Matrix A:                           a_shared:
 [a[0,0] a[0,1]]                     [s[0,0] s[0,1] s[0,2]]
 [a[1,0] a[1,1]]                     [s[1,0] s[1,1] s[1,2]]
                                     [s[2,0] s[2,1] s[2,2]]
Matrix B:                           b_shared: (similar layout)
 [b[0,0] b[0,1]]                     [t[0,0] t[0,1] t[0,2]]
 [b[1,0] b[1,1]]                     [t[1,0] t[1,1] t[1,2]]
                                     [t[2,0] t[2,1] t[2,2]]
```

### Implementation phases

1. **Shared Memory Setup**:

   ```mojo
   # Create 2D shared memory tensors using LayoutTensor with address_space
   a_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   b_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

2. **Thread Indexing**:

   ```mojo
   # Global indices for matrix access
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x

   # Local indices for shared memory
   local_row = thread_idx.y
   local_col = thread_idx.x
   ```

3. **Data Loading**:

   ```mojo
   # Load data into shared memory using LayoutTensor indexing
   if row < size and col < size:
       a_shared[local_row, local_col] = a[row, col]
       b_shared[local_row, local_col] = b[row, col]
   ```

4. **Computation with Shared Memory**:

   ```mojo
   # Guard ensures we only compute for valid matrix elements
   if row < size and col < size:
       # Initialize accumulator with output tensor's type
       var acc: output.element_type = 0

       # Compile-time unrolled loop for matrix multiplication
       @parameter
       for k in range(size):
           acc += a_shared[local_row, k] * b_shared[k, local_col]

       # Write result only for threads within matrix bounds
       output[row, col] = acc
   ```

   Key aspects:
   - **Boundary check**: `if row < size and col < size`
     - Prevents out-of-bounds computation
     - Only valid threads perform work
     - Essential because TPB (3×3) > SIZE (2×2)

   - **Accumulator Type**: `var acc: output.element_type`
     - Uses output tensor's element type for type safety
     - Ensures consistent numeric precision
     - Initialized to zero before accumulation

   - **Loop Optimization**: `@parameter for k in range(size)`
     - Unrolls the loop at compile time
     - Enables better instruction scheduling
     - Efficient for small, known matrix sizes

   - **Result Writing**: `output[row, col] = acc`
     - Protected by the same guard condition
     - Only valid threads write results
     - Maintains matrix bounds safety

### Thread safety and synchronization

1. **Guard conditions**:
   - Input Loading: `if row < size and col < size`
   - Computation: Same guard ensures thread safety
   - Output Writing: Protected by the same condition
   - Prevents invalid memory access and race conditions

2. **Memory access safety**:
   - Shared memory: Accessed only within TPB bounds
   - Global memory: Protected by size checks
   - Output: Guarded writes prevent corruption

### Key language features

1. **LayoutTensor benefits**:
   - Direct 2D indexing simplifies code
   - Type safety through `element_type`
   - Efficient memory layout handling

2. **Shared memory allocation**:
   - LayoutTensor with address_space for structured allocation
   - Row-major layout matching input tensors
   - Proper alignment for efficient access

3. **Synchronization**:
   - `barrier()` ensures shared memory consistency
   - Proper synchronization between load and compute
   - Thread cooperation within block

### Performance optimizations

1. **Memory Access Efficiency**:
   - Single global memory load per element
   - Multiple reuse through shared memory
   - Coalesced memory access patterns

2. **Thread cooperation**:
   - Collaborative data loading
   - Shared data reuse
   - Efficient thread synchronization

3. **Computational benefits**:
   - Reduced global memory traffic
   - Better cache utilization
   - Improved instruction throughput

This implementation significantly improves performance over the naive version by:

- Reducing global memory accesses
- Enabling data reuse through shared memory
- Using efficient 2D indexing with LayoutTensor
- Maintaining proper thread synchronization

</div>
</details>
