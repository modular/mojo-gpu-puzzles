## Overview

Implement a kernel that adds 10 to each position of a vector `a` and stores it in `output`.

**Note:** _You have fewer threads per block than the size of `a`._

## Key concepts

In this puzzle, you'll learn about:

- Using shared memory within thread blocks
- Synchronizing threads with barriers
- Managing block-local data storage

The key insight is understanding how shared memory provides fast, block-local storage that all threads in a block can access, requiring careful coordination between threads.

## Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 4`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

Notes:

- **Shared memory**: Fast storage shared by threads in a block
- **Thread sync**: Coordination using `barrier()`
- **Memory scope**: Shared memory only visible within block
- **Access pattern**: Local vs global indexing

> **Warning**: Each block can only have a _constant_ amount of shared memory that threads in that block can read and write to. This needs to be a literal python constant, not a variable. After writing to shared memory you need to call [barrier](https://docs.modular.com/mojo/stdlib/gpu/sync/barrier/) to ensure that threads do not cross.

**Educational Note**: In this specific puzzle, the `barrier()` isn't strictly necessary since each thread only accesses its own shared memory location. However, it's included to teach proper shared memory synchronization patterns for more complex scenarios where threads need to coordinate access to shared data.

## Code to complete

```mojo
{{#include ../../../problems/p08/p08.mojo:add_10_shared}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p08/p08.mojo" class="filename">View full file: problems/p08/p08.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Wait for shared memory load with `barrier()` (educational - not strictly needed here)
2. Use `local_i` to access shared memory: `shared[local_i]`
3. Use `global_i` for output: `output[global_i]`
4. Add guard: `if global_i < size`

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
pixi run p08
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p08
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p08
```

  </div>
  <div class="tab-content">

```bash
uv run poe p08
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
{{#include ../../../solutions/p08/p08.mojo:add_10_shared_solution}}
```

<div class="solution-explanation">

This solution demonstrates key concepts of shared memory usage in GPU programming:

1. **Memory hierarchy**
   - Global memory: `a` and `output` arrays (slow, visible to all blocks)
   - Shared memory: `shared` array (fast, thread-block local)
   - Example for 8 elements with 4 threads per block:

     ```txt
     Global array a: [1 1 1 1 | 1 1 1 1]  # Input: all ones

     Block (0):      Block (1):
     shared[0..3]    shared[0..3]
     [1 1 1 1]       [1 1 1 1]
     ```

2. **Thread coordination**
   - Load phase:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()    ↓         ↓        ↓         ↓   # Wait for all loads
     ```

   - Process phase: Each thread adds 10 to its shared memory value
   - Result: `output[i] = shared[local_i] + 10 = 11`

   **Note**: In this specific case, the `barrier()` isn't strictly necessary since each thread only writes to and reads from its own shared memory location (`shared[local_i]`). However, it's included for educational purposes to demonstrate proper shared memory synchronization patterns that are essential when threads need to access each other's data.

3. **Index mapping**
   - Global index: `block_dim.x * block_idx.x + thread_idx.x`

     ```txt
     Block 0 output: [11 11 11 11]
     Block 1 output: [11 11 11 11]
     ```

   - Local index: `thread_idx.x` for shared memory access

     ```txt
     Both blocks process: 1 + 10 = 11
     ```

4. **Memory access pattern**
   - Load: Global → Shared (coalesced reads of 1s)
   - Sync: `barrier()` ensures all loads complete
   - Process: Add 10 to shared values
   - Store: Write 11s back to global memory

This pattern shows how to use shared memory to optimize data access while maintaining thread coordination within blocks.
</div>
</details>
