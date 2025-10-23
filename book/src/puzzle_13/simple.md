# Simple Case with Single Block

Implement a kernel that computes a 1D convolution between 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 1D LayoutTensor `output`.

**Note:** _You need to handle the general case. You only need 2 global reads and 1 global write per thread._

## Key concepts

This puzzle covers:

- Implementing sliding window operations on GPUs
- Managing data dependencies across threads
- Using shared memory for overlapping regions

The key insight is understanding how to efficiently access overlapping elements while maintaining correct boundary conditions.

## Configuration

- Input array size: `SIZE = 6` elements
- Kernel size: `CONV = 3` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Shared memory: Two arrays of size `SIZE` and `CONV`

Notes:

- **Data loading**: Each thread loads one element from input and kernel
- **Memory pattern**: Shared arrays for input and convolution kernel
- **Thread sync**: Coordination before computation

## Code to complete

```mojo
{{#include ../../../problems/p13/p13.mojo:conv_1d_simple}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p13/p13.mojo" class="filename">View full file: problems/p13/p13.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use `tb[dtype]().row_major[SIZE]().shared().alloc()` for shared memory allocation
2. Load input to `shared_a[local_i]` and kernel to `shared_b[local_i]`
3. Call `barrier()` after loading
4. Sum products within bounds: `if local_i + j < SIZE`
5. Write result if `global_i < SIZE`

</div>
</details>

### Running the code

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
pixi run p13 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p13 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p13 --simple
```

  </div>
  <div class="tab-content">

```bash
uv run poe p13 --simple
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([5.0, 8.0, 11.0, 14.0, 5.0, 0.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p13/p13.mojo:conv_1d_simple_solution}}
```

<div class="solution-explanation">

The solution implements a 1D convolution using shared memory for efficient access to overlapping elements. Here's a detailed breakdown:

### Memory layout

```txt
Input array a:   [0  1  2  3  4  5]
Kernel b:        [0  1  2]
```

### Computation steps

1. **Data Loading**:

   ```txt
   shared_a: [0  1  2  3  4  5]  // Input array
   shared_b: [0  1  2]           // Convolution kernel
   ```

2. **Convolution Process** for each position i:

   ```txt
   output[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] = 0*0 + 1*1 + 2*2 = 5
   output[1] = a[1]*b[0] + a[2]*b[1] + a[3]*b[2] = 1*0 + 2*1 + 3*2 = 8
   output[2] = a[2]*b[0] + a[3]*b[1] + a[4]*b[2] = 2*0 + 3*1 + 4*2 = 11
   output[3] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2] = 3*0 + 4*1 + 5*2 = 14
   output[4] = a[4]*b[0] + a[5]*b[1] + 0*b[2]    = 4*0 + 5*1 + 0*2 = 5
   output[5] = a[5]*b[0] + 0*b[1]   + 0*b[2]     = 5*0 + 0*1 + 0*2 = 0
   ```

### Implementation details

1. **Thread Participation and Efficiency Considerations**:
   - The inefficient approach without proper thread guard:

     ```mojo
     # Inefficient version - all threads compute even when results won't be used
     local_sum = Scalar[dtype](0)
     for j in range(CONV):
         if local_i + j < SIZE:
             local_sum += shared_a[local_i + j] * shared_b[j]
     # Only guard the final write
     if global_i < SIZE:
         output[global_i] = local_sum
     ```

   - The efficient and correct implementation:

     ```mojo
     if global_i < SIZE:
         var local_sum: output.element_type = 0  # Using var allows type inference
         @parameter  # Unrolls loop at compile time since CONV is constant
         for j in range(CONV):
             if local_i + j < SIZE:
                 local_sum += shared_a[local_i + j] * shared_b[j]
         output[global_i] = local_sum
     ```

   The key difference is that the inefficient version has **all threads perform the convolution computation** (including those where `global_i >= SIZE`), and only the final write is guarded. This leads to:
   - **Wasteful computation**: Threads beyond the valid range still perform unnecessary work
   - **Reduced efficiency**: Extra computations that won't be used
   - **Poor resource utilization**: GPU cores working on meaningless calculations

   The efficient version ensures that only threads with valid `global_i` values perform any computation, making better use of GPU resources.

2. **Key Implementation Features**:
   - Uses `var` for proper type inference with `output.element_type`
   - Employs `@parameter` decorator to unroll the convolution loop at compile time
   - Maintains strict bounds checking for memory safety
   - Leverages LayoutTensor's type system for better code safety

3. **Memory Management**:
   - Uses shared memory for both input array and kernel
   - Single load per thread from global memory
   - Efficient reuse of loaded data

4. **Thread Coordination**:
   - `barrier()` ensures all data is loaded before computation
   - Each thread computes one output element
   - Maintains coalesced memory access pattern

5. **Performance Optimizations**:
   - Minimizes global memory access
   - Uses shared memory for fast data access
   - Avoids thread divergence in main computation loop
   - Loop unrolling through `@parameter` decorator

</div>
</details>
