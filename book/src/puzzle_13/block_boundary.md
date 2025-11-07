# Block Boundary Version

Implement a kernel that computes a 1D convolution between 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 1D LayoutTensor `output`.

**Note:** _You need to handle the general case. You only need 2 global reads and 1 global write per thread._

## Configuration

- Input array size: `SIZE_2 = 15` elements
- Kernel size: `CONV_2 = 4` elements
- Threads per block: `TPB = 8`
- Number of blocks: 2
- Shared memory: `TPB + CONV_2 - 1` elements for input

Notes:

- **Extended loading**: Account for boundary overlap
- **Block edges**: Handle data across block boundaries
- **Memory layout**: Efficient shared memory usage
- **Synchronization**: Proper thread coordination

## Code to complete

```mojo
{{#include ../../../problems/p13/p13.mojo:conv_1d_block_boundary}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p13/p13.mojo" class="filename">View full file: problems/p13/p13.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use `LayoutTensor[dtype, Layout.row_major(TPB + CONV_2 - 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()` for shared memory
2. Load main data: `shared_a[local_i] = a[global_i]`
3. Load boundary: `if local_i < CONV_2 - 1` handle next block data
4. Load kernel: `shared_b[local_i] = b[local_i]`
5. Sum within input bounds: `if global_i + j < SIZE_2`

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
pixi run p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
uv run poe p13 --block-boundary
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0, 80.0, 41.0, 14.0, 0.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p13/p13.mojo:conv_1d_block_boundary_solution}}
```

<div class="solution-explanation">

The solution handles block boundary cases in 1D convolution using extended shared memory. Here's a detailed analysis:

### Memory layout and sizing

```txt
Test Configuration:
- Full array size: SIZE_2 = 15 elements
- Grid: 2 blocks × 8 threads
- Convolution kernel: CONV_2 = 4 elements

Block 0 shared memory:  [0 1 2 3 4 5 6 7|8 9 10]  // TPB(8) + (CONV_2-1)(3) padding
Block 1 shared memory:  [8 9 10 11 12 13 14|0 0]  // Second block with padding

Size calculation:
- Main data: TPB elements (8)
- Overlap: CONV_2 - 1 elements (4 - 1 = 3)
- Total: TPB + CONV_2 - 1 = 8 + 4 - 1 = 11 elements
```

### Implementation details

1. **Shared Memory Allocation**:

   ```mojo
   # First: account for padding needed for convolution window
   shared_a = LayoutTensor[dtype, Layout.row_major(TPB + CONV_2 - 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   shared_b = LayoutTensor[dtype, Layout.row_major(CONV_2), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

   This allocation pattern ensures we have enough space for both the block's data and the overlap region.

2. **Data Loading Strategy**:

   ```mojo
   # Main block data
   if global_i < SIZE_2:
       shared_a[local_i] = a[global_i]

   # Boundary data from next block
   if local_i < CONV_2 - 1:
       next_idx = global_i + TPB
       if next_idx < SIZE_2:
           shared_a[TPB + local_i] = a[next_idx]
       else:
           # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
           # which is an undefined behavior
           shared_a[TPB + local_i] = 0
   ```

   - Only threads with `local_i < CONV_2 - 1` load boundary data
   - Prevents unnecessary thread divergence
   - Maintains memory coalescing for main data load
   - Explicitly zeroes out-of-bounds elements to avoid undefined behavior

3. **Kernel Loading**:

   ```mojo
   if local_i < b_size:
       shared_b[local_i] = b[local_i]
   ```

   - Single load per thread
   - Bounded by kernel size

4. **Convolution Computation**:

   ```mojo
   if global_i < SIZE_2:
       var local_sum: output.element_type = 0
       @parameter
       for j in range(CONV_2):
           if global_i + j < SIZE_2:
               local_sum += shared_a[local_i + j] * shared_b[j]
   ```

   - Uses `@parameter` for compile-time loop unrolling
   - Proper type inference with `output.element_type`
   - Semantically correct bounds check: only compute convolution for valid input positions

### Memory access pattern analysis

1. **Block 0 Access Pattern**:

   ```txt
   Thread 0: [0 1 2 3] × [0 1 2 3]
   Thread 1: [1 2 3 4] × [0 1 2 3]
   Thread 2: [2 3 4 5] × [0 1 2 3]
   ...
   Thread 7: [7 8 9 10] × [0 1 2 3]  // Uses overlap data
   ```

2. **Block 1 Access Pattern**:

   ```txt
   Thread 0: [8 9 10 11] × [0 1 2 3]
   Thread 1: [9 10 11 12] × [0 1 2 3]
   ...
   Thread 7: [14 0 0 0] × [0 1 2 3]  // Zero padding at end
   ```

### Performance optimizations

1. **Memory Coalescing**:
   - Main data load: Consecutive threads access consecutive memory
   - Boundary data: Only necessary threads participate
   - Single barrier synchronization point

2. **Thread Divergence Minimization**:
   - Clean separation of main and boundary loading
   - Uniform computation pattern within warps
   - Efficient bounds checking

3. **Shared Memory Usage**:
   - Optimal sizing to handle block boundaries
   - No bank conflicts in access pattern
   - Efficient reuse of loaded data

4. **Boundary Handling**:
   - Explicit zero initialization for out-of-bounds elements which prevents reading from uninitialized shared memory
   - Semantically correct boundary checking using `global_i + j < SIZE_2` instead of shared memory bounds
   - Proper handling of edge cases without over-computation

### Boundary condition improvement

The solution uses `if global_i + j < SIZE_2:` rather than checking shared memory bounds. This pattern is:

- **Mathematically correct**: Only computes convolution where input data actually exists
- **More efficient**: Avoids unnecessary computations for positions beyond the input array
- **Safer**: Prevents reliance on zero-padding behavior in shared memory

This implementation achieves efficient cross-block convolution while maintaining:

- Memory safety through proper bounds checking
- High performance through optimized memory access
- Clean code structure using LayoutTensor abstractions
- Minimal synchronization overhead
- Mathematically sound boundary handling

</div>
</details>
