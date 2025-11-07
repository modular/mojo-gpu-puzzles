## Overview

Implement a kernel that adds 10 to each position of a 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:** _You have fewer threads per block than the size of `a`._

## Key concepts

In this puzzle, you'll learn about:

- Using LayoutTensor's shared memory features with address_space
- Thread synchronization with shared memory
- Block-local data management with LayoutTensor

The key insight is how LayoutTensor simplifies shared memory management while maintaining the performance benefits of block-local storage.

## Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 4`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

## Key differences from raw approach

1. **Memory allocation**: We will use [LayoutTensor](https://docs.modular.com/mojo/stdlib/layout/layout_tensor/LayoutTensor) with address_space instead of [stack_allocation](https://docs.modular.com/mojo/stdlib/memory/memory/stack_allocation/)

   ```mojo
   # Raw approach
   shared = stack_allocation[TPB, Scalar[dtype]]()

   # LayoutTensor approach
   shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

2. **Memory access**: Same syntax

   ```mojo
   # Raw approach
   shared[local_i] = a[global_i]

   # LayoutTensor approach
   shared[local_i] = a[global_i]
   ```

3. **Safety features**:

   - Type safety
   - Layout management
   - Memory alignment handling

> **Note**: LayoutTensor handles memory layout, but you still need to manage thread synchronization with `barrier()` when using shared memory.

**Educational Note**: In this specific puzzle, the `barrier()` isn't strictly necessary since each thread only accesses its own shared memory location. However, it's included to teach proper shared memory synchronization patterns for more complex scenarios where threads need to coordinate access to shared data.

## Code to complete

```mojo
{{#include ../../../problems/p08/p08_layout_tensor.mojo:add_10_shared_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p08/p08_layout_tensor.mojo" class="filename">View full file: problems/p08/p08_layout_tensor.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Create shared memory with LayoutTensor using address_space parameter
2. Load data with natural indexing: `shared[local_i] = a[global_i]`
3. Synchronize with `barrier()` (educational - not strictly needed here)
4. Process data using shared memory indices
5. Guard against out-of-bounds access

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
pixi run p08_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p08_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p08_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p08_layout_tensor
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p08/p08_layout_tensor.mojo:add_10_shared_layout_tensor_solution}}
```

<div class="solution-explanation">

This solution demonstrates how LayoutTensor simplifies shared memory usage while maintaining performance:

1. **Memory hierarchy with LayoutTensor**
   - Global tensors: `a` and `output` (slow, visible to all blocks)
   - Shared tensor: `shared` (fast, thread-block local)
   - Example for 8 elements with 4 threads per block:

     ```txt
     Global tensor a: [1 1 1 1 | 1 1 1 1]  # Input: all ones

     Block (0):         Block (1):
     shared[0..3]       shared[0..3]
     [1 1 1 1]          [1 1 1 1]
     ```

2. **Thread coordination**
   - Load phase with natural indexing:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()    ↓         ↓        ↓         ↓   # Wait for all loads
     ```

   - Process phase: Each thread adds 10 to its shared tensor value
   - Result: `output[global_i] = shared[local_i] + 10 = 11`

   **Note**: In this specific case, the `barrier()` isn't strictly necessary since each thread only writes to and reads from its own shared memory location (`shared[local_i]`). However, it's included for educational purposes to demonstrate proper shared memory synchronization patterns that are essential when threads need to access each other's data.

3. **LayoutTensor benefits**
   - Shared memory allocation:

     ```txt
     # Clean LayoutTensor API with address_space
     shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
     ```

   - Natural indexing for both global and shared:

     ```txt
     Block 0 output: [11 11 11 11]
     Block 1 output: [11 11 11 11]
     ```

   - Built-in layout management and type safety

4. **Memory access pattern**
   - Load: Global tensor → Shared tensor (optimized)
   - Sync: Same `barrier()` requirement as raw version
   - Process: Add 10 to shared values
   - Store: Write 11s back to global tensor

This pattern shows how LayoutTensor maintains the performance benefits of shared memory while providing a more ergonomic API and built-in features.
</div>
</details>
