## Overview

Implement a kernel that computes the dot product of vector `a` and vector `b` and stores it in `output` (single number).

**Note:** _You have 1 thread per position. You only need 2 global reads per thread and 1 global write per thread block._

## Key concepts

This puzzle covers:

- Implementing parallel reduction operations
- Using shared memory for intermediate results
- Coordinating threads for collective operations

The key insight is understanding how to efficiently combine multiple values into a single result using parallel computation and shared memory.

## Configuration

- Vector size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Output size: 1 element
- Shared memory: `TPB` elements

Notes:

- **Element access**: Each thread reads corresponding elements from `a` and `b`
- **Partial results**: Computing and storing intermediate values
- **Thread coordination**: Synchronizing before combining results
- **Final reduction**: Converting partial results to scalar output

_Note: For this problem, you don't need to worry about number of shared reads. We will
handle that challenge later._

## Code to complete

```mojo
{{#include ../../../problems/p12/p12.mojo:dot_product}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p12/p12.mojo" class="filename">View full file: problems/p12/p12.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `a[global_i] * b[global_i]` in `shared[local_i]`
2. Call `barrier()` to synchronize
3. Use thread 0 to sum all products in shared memory
4. Write final sum to `output[0]`

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
pixi run p12
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p12
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p12
```

  </div>
  <div class="tab-content">

```bash
uv run poe p12
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0])
expected: HostBuffer([140.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p12/p12.mojo:dot_product_solution}}
```

<div class="solution-explanation">

The solution implements a parallel reduction algorithm for dot product computation using shared memory. Here's a detailed breakdown:

### Phase 1: Element-wise Multiplication

Each thread performs one multiplication:

```txt
Thread i: shared[i] = a[i] * b[i]
```

### Phase 2: Parallel Reduction

The reduction uses a tree-based approach that halves active threads in each step:

```txt
Initial:  [0*0  1*1  2*2  3*3  4*4  5*5  6*6  7*7]
        = [0    1    4    9    16   25   36   49]

Step 1:   [0+16 1+25 4+36 9+49  16   25   36   49]
        = [16   26   40   58   16   25   36   49]

Step 2:   [16+40 26+58 40   58   16   25   36   49]
        = [56   84   40   58   16   25   36   49]

Step 3:   [56+84  84   40   58   16   25   36   49]
        = [140   84   40   58   16   25   36   49]
```

### Key implementation features

1. **Memory Access Pattern**:
   - Each thread loads exactly two values from global memory (`a[i]`, `b[i]`)
   - Uses shared memory for intermediate results
   - Final result written once to global memory

2. **Thread Synchronization**:
   - `barrier()` after initial multiplication
   - `barrier()` after each reduction step
   - Prevents race conditions between reduction steps

3. **Reduction Logic**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           shared[local_i] += shared[local_i + stride]
       barrier()
       stride //= 2
   ```

   - Halves stride in each step
   - Only active threads perform additions
   - Maintains work efficiency

4. **Performance Considerations**:
   - \\(\log_2(n)\\) steps for \\(n\\) elements
   - Coalesced memory access pattern
   - Minimal thread divergence
   - Efficient use of shared memory

This implementation achieves \\(O(\log n)\\) time complexity compared to \\(O(n)\\) in sequential execution, demonstrating the power of parallel reduction algorithms.

### Barrier synchronization importance

The `barrier()` between reduction steps is critical for correctness. Here's why:

Without `barrier()`, race conditions occur:

```text
Initial shared memory: [0 1 4 9 16 25 36 49]

Step 1 (stride = 4):
Thread 0 reads: shared[0] = 0, shared[4] = 16
Thread 1 reads: shared[1] = 1, shared[5] = 25
Thread 2 reads: shared[2] = 4, shared[6] = 36
Thread 3 reads: shared[3] = 9, shared[7] = 49

Without barrier:
- Thread 0 writes: shared[0] = 0 + 16 = 16
- Thread 1 starts next step (stride = 2) before Thread 0 finishes
  and reads old value shared[0] = 0 instead of 16!
```

With `barrier()`:

```text
Step 1 (stride = 4):
All threads write their sums:
[16 26 40 58 16 25 36 49]
barrier() ensures ALL threads see these values

Step 2 (stride = 2):
Now threads safely read the updated values:
Thread 0: shared[0] = 16 + 40 = 56
Thread 1: shared[1] = 26 + 58 = 84
```

The `barrier()` ensures:

1. All writes from current step complete
2. All threads see updated values
3. No thread starts next iteration early
4. Consistent shared memory state

Without these synchronization points, we could get:

- Memory race conditions
- Threads reading stale values
- Non-deterministic results
- Incorrect final sum

</div>
</details>
