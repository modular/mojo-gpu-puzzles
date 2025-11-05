# `warp.shuffle_down()` One-to-One Communication

For warp-level neighbor communication we can use `shuffle_down()` to access data from adjacent lanes within a warp. This powerful primitive enables efficient finite differences, moving averages, and neighbor-based computations without shared memory or explicit synchronization.

**Key insight:** _The [shuffle_down()](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down) operation leverages SIMT execution to let each lane access data from its neighbors within the same warp, enabling efficient stencil patterns and sliding window operations._

> **What are stencil operations?** [Stencil](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops) operations are computations where each output element depends on a fixed pattern of neighboring input elements. Common examples include finite differences (derivatives), convolutions, and moving averages. The "stencil" refers to the pattern of neighbor access - like a 3-point stencil that reads `[i-1, i, i+1]` or a 5-point stencil that reads `[i-2, i-1, i, i+1, i+2]`.

## Key concepts

In this puzzle, you'll learn:

- **Warp-level data shuffling** with `shuffle_down()`
- **Neighbor access patterns** for stencil computations
- **Boundary handling** at warp edges
- **Multi-offset shuffling** for extended neighbor access
- **Cross-warp coordination** in multi-block scenarios

The `shuffle_down` operation enables each lane to access data from lanes at higher indices:
\\[\\Large \text{shuffle\_down}(\text{value}, \text{offset}) = \text{value_from_lane}(\text{lane\_id} + \text{offset})\\]

This transforms complex neighbor access patterns into simple warp-level operations, enabling efficient stencil computations without explicit memory indexing.

## 1. Basic neighbor difference

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### The shuffle_down concept

Traditional neighbor access requires complex indexing and bounds checking:

```mojo
# Traditional approach - complex and error-prone
if global_i < size - 1:
    next_value = input[global_i + 1]  # Potential out-of-bounds
    result = next_value - current_value
```

**Problems with traditional approach:**

- **Bounds checking**: Must manually verify array bounds
- **Memory access**: Requires separate memory loads
- **Synchronization**: May need barriers for shared memory patterns
- **Complex logic**: Handling edge cases becomes verbose

With `shuffle_down()`, neighbor access becomes elegant:

```mojo
# Warp shuffle approach - simple and safe
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # Get value from lane+1
if lane < WARP_SIZE - 1:
    result = next_val - current_val
```

**Benefits of shuffle_down:**

- **Zero memory overhead**: No additional memory accesses
- **Automatic bounds**: Hardware handles warp boundaries
- **No synchronization**: SIMT execution guarantees correctness
- **Composable**: Easy to combine with other warp operations

### Code to complete

Implement finite differences using `shuffle_down()` to access the next element.

**Mathematical operation:** Compute the discrete derivative (finite difference) for each element:
\\[\\Large \\text{output}[i] = \\text{input}[i+1] - \\text{input}[i]\\]

This transforms input data `[0, 1, 4, 9, 16, 25, ...]` (squares: `i * i`) into differences `[1, 3, 5, 7, 9, ...]` (odd numbers), effectively computing the discrete derivative of the quadratic function.

```mojo
{{#include ../../../problems/p25/p25.mojo:neighbor_difference}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p25/p25.mojo" class="filename">View full file: problems/p25/p25.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding shuffle_down**

The `shuffle_down(value, offset)` operation allows each lane to receive data from a lane at a higher index. Study how this can give you access to neighboring elements without explicit memory loads.

**What `shuffle_down(val, 1)` does:**

- Lane 0 gets value from Lane 1
- Lane 1 gets value from Lane 2
- ...
- Lane 30 gets value from Lane 31
- Lane 31 gets undefined value (handled by boundary check)

### 2. **Warp boundary considerations**

Consider what happens at the edges of a warp. Some lanes may not have valid neighbors to access via shuffle operations.

**Challenge:** Design your algorithm to handle cases where shuffle operations may return undefined data for lanes at warp boundaries.

For neighbor difference with `WARP_SIZE = 32`:

- **Valid difference** (`lane < WARP_SIZE - 1`): **Lanes 0-30** (31 lanes)
  - **When**: \\(\text{lane\_id}() \in \{0, 1, \cdots, 30\}\\)
  - **Why**: `shuffle_down(current_val, 1)` successfully gets next neighbor's value
  - **Result**: `output[i] = input[i+1] - input[i]` (finite difference)

- **Boundary case** (else): **Lane 31** (1 lane)
  - **When**: \\(\text{lane\_id}() = 31\\)
  - **Why**: `shuffle_down(current_val, 1)` returns undefined data (no lane 32)
  - **Result**: `output[i] = 0` (cannot compute difference)

### 3. **Lane identification**

```mojo
lane = lane_id()  # Returns 0 to WARP_SIZE-1
```

**Lane numbering:** Within each warp, lanes are numbered 0, 1, 2, ..., `WARP_SIZE-1`

</div>
</details>

**Test the neighbor difference:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --neighbor
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
expected: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
✅ Basic neighbor difference test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p25/p25.mojo:neighbor_difference_solution}}
```

<div class="solution-explanation">

This solution demonstrates how `shuffle_down()` transforms traditional array indexing into efficient warp-level communication.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]           # Each lane reads its element
    next_val = shuffle_down(current_val, 1) # Hardware shifts data right

    if lane < WARP_SIZE - 1:
        output[global_i] = next_val - current_val  # Compute difference
    else:
        output[global_i] = 0                       # Boundary handling
```

**SIMT execution deep dive:**

```
Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = 961

Cycle 2: shuffle_down(current_val, 1) executes on all lanes
  Lane 0: receives current_val from Lane 1 → next_val = 1
  Lane 1: receives current_val from Lane 2 → next_val = 4
  Lane 2: receives current_val from Lane 3 → next_val = 9
  ...
  Lane 30: receives current_val from Lane 31 → next_val = 961
  Lane 31: receives undefined (no Lane 32) → next_val = ?

Cycle 3: Difference computation (lanes 0-30 only)
  Lane 0: output[0] = 1 - 0 = 1
  Lane 1: output[1] = 4 - 1 = 3
  Lane 2: output[2] = 9 - 4 = 5
  ...
  Lane 31: output[31] = 0 (boundary condition)
```

**Mathematical insight:** This implements the discrete derivative operator \\(D\\):
\\[\\Large D[f](i) = f(i+1) - f(i)\\]

For our quadratic input \\(f(i) = i^2\\):
\\[\\Large D[i^2] = (i+1)^2 - i^2 = i^2 + 2i + 1 - i^2 = 2i + 1\\]

**Why shuffle_down is superior:**

1. **Memory efficiency**: Traditional approach requires `input[global_i + 1]` load, potentially causing cache misses
2. **Bounds safety**: No risk of out-of-bounds access; hardware handles warp boundaries
3. **SIMT optimization**: Single instruction processes all lanes simultaneously
4. **Register communication**: Data moves between registers, not through memory hierarchy

**Performance characteristics:**

- **Latency**: 1 cycle (vs 100+ cycles for memory access)
- **Bandwidth**: 0 bytes (vs 4 bytes per thread for traditional)
- **Parallelism**: All 32 lanes process simultaneously

</div>
</details>

## 2. Multi-offset moving average

### Configuration

- Vector size: `SIZE_2 = 64` (multi-block scenario)
- Grid configuration: `BLOCKS_PER_GRID = (2, 1)` blocks per grid
- Block configuration: `THREADS_PER_BLOCK = (WARP_SIZE, 1)` threads per block

### Code to complete

Implement a 3-point moving average using multiple `shuffle_down` operations.

**Mathematical operation:** Compute a sliding window average using three consecutive elements:
\\[\\Large \\text{output}[i] = \\frac{1}{3}\\left(\\text{input}[i] + \\text{input}[i+1] + \\text{input}[i+2]\\right)\\]

**Boundary handling:** The algorithm gracefully degrades at warp boundaries:

- **Full 3-point window**: \\(\\text{output}[i] = \\frac{1}{3}\\sum_{k=0}^{2} \\text{input}[i+k]\\) when all neighbors available
- **2-point window**: \\(\\text{output}[i] = \\frac{1}{2}\\sum_{k=0}^{1} \\text{input}[i+k]\\) when only next neighbor available
- **1-point window**: \\(\\text{output}[i] = \\text{input}[i]\\) when no neighbors available

This demonstrates how `shuffle_down()` enables efficient stencil operations with automatic boundary handling within warp limits.

```mojo
{{#include ../../../problems/p25/p25.mojo:moving_average_3}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Multi-offset shuffle patterns**

This puzzle requires accessing multiple neighbors simultaneously. You'll need to use shuffle operations with different offsets.

**Key questions:**

- How can you get both `input[i+1]` and `input[i+2]` using shuffle operations?
- What's the relationship between shuffle offset and neighbor distance?
- Can you perform multiple shuffles on the same source value?

**Visualization concept:**

```
Your lane needs:  current_val, next_val, next_next_val
Shuffle offsets:  0 (direct),  1,        2
```

**Think about:** How many shuffle operations do you need, and what offsets should you use?

### 2. **Tiered boundary handling**

Unlike the simple neighbor difference, this puzzle has multiple boundary scenarios because you need access to 2 neighbors.

**Boundary scenarios to consider:**

- **Full window:** Lane can access both neighbors → use all 3 values
- **Partial window:** Lane can access 1 neighbor → use 2 values
- **No window:** Lane can't access any neighbors → use 1 value

**Critical thinking:**

- Which lanes fall into each category?
- How should you weight the averages when you have fewer values?
- What boundary conditions should you check?

**Pattern to consider:**

```
if (can_access_both_neighbors):
    # 3-point average
elif (can_access_one_neighbor):
    # 2-point average
else:
    # 1-point (no averaging)
```

### 3. **Multi-block coordination**

This puzzle uses multiple blocks, each processing a different section of the data.

**Important considerations:**

- Each block has its own warp with lanes 0 to WARP_SIZE-1
- Boundary conditions apply within each warp independently
- Lane numbering resets for each block

**Questions to think about:**

- Does your boundary logic work correctly for both Block 0 and Block 1?
- Are you checking both lane boundaries AND global array boundaries?
- How does `global_i` relate to `lane_id()` in different blocks?

**Debugging tip:** Test your logic by tracing through what happens at the boundary lanes of each block.

</div>
</details>

**Test the moving average:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --average
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --average
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --average
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
expected: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
✅ Moving average test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p25/p25.mojo:moving_average_3_solution}}
```

<div class="solution-explanation">

This solution demonstrates advanced multi-offset shuffling for complex stencil operations.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    # Step 1: Acquire all needed data via multiple shuffles
    current_val = input[global_i]                   # Direct access
    next_val = shuffle_down(current_val, 1)         # Right neighbor
    next_next_val = shuffle_down(current_val, 2)    # Right+1 neighbor

    # Step 2: Adaptive computation based on available data
    if lane < WARP_SIZE - 2 and global_i < size - 2:
        # Full 3-point stencil available
        output[global_i] = (current_val + next_val + next_next_val) / 3.0
    elif lane < WARP_SIZE - 1 and global_i < size - 1:
        # Only 2-point stencil available (near warp boundary)
        output[global_i] = (current_val + next_val) / 2.0
    else:
        # No stencil possible (at warp boundary)
        output[global_i] = current_val
```

**Multi-offset execution trace (`WARP_SIZE = 32`):**

```
Initial state (Block 0, elements 0-31):
  Lane 0: current_val = input[0] = 1
  Lane 1: current_val = input[1] = 2
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = X

First shuffle: shuffle_down(current_val, 1)
  Lane 0: next_val = input[1] = 2
  Lane 1: next_val = input[2] = 4
  Lane 2: next_val = input[3] = 7
  ...
  Lane 30: next_val = input[31] = X
  Lane 31: next_val = undefined

Second shuffle: shuffle_down(current_val, 2)
  Lane 0: next_next_val = input[2] = 4
  Lane 1: next_next_val = input[3] = 7
  Lane 2: next_next_val = input[4] = 11
  ...
  Lane 29: next_next_val = input[31] = X
  Lane 30: next_next_val = undefined
  Lane 31: next_next_val = undefined

Computation phase:
  Lanes 0-29: Full 3-point average → (current + next + next_next) / 3
  Lane 30:    2-point average → (current + next) / 2
  Lane 31:    1-point average → current (passthrough)
```

**Mathematical foundation:** This implements a variable-width discrete convolution:
\\[\\Large h[i] = \\sum_{k=0}^{K(i)-1} w_k^{(i)} \\cdot f[i+k]\\]

Where the kernel adapts based on position:

- **Interior points**: \\(K(i) = 3\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{3}, \\frac{1}{3}, \\frac{1}{3}]\\)
- **Near boundary**: \\(K(i) = 2\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{2}, \\frac{1}{2}]\\)
- **At boundary**: \\(K(i) = 1\\), \\(\\mathbf{w}^{(i)} = [1]\\)

**Multi-block coordination:** With `SIZE_2 = 64` and 2 blocks:

```
Block 0 (global indices 0-31):
  Lane boundaries apply to global indices 29, 30, 31

Block 1 (global indices 32-63):
  Lane boundaries apply to global indices 61, 62, 63
  Lane numbers reset: global_i=32 → lane=0, global_i=63 → lane=31
```

**Performance optimizations:**

1. **Parallel data acquisition**: Both shuffle operations execute simultaneously
2. **Conditional branching**: GPU handles divergent lanes efficiently via predication
3. **Memory coalescing**: Sequential global memory access pattern optimal for GPU
4. **Register reuse**: All intermediate values stay in registers

**Signal processing perspective:** This is a causal FIR filter with impulse response \\(h[n] = \\frac{1}{3}[\\delta[n] + \\delta[n-1] + \\delta[n-2]]\\), providing smoothing with a cutoff frequency at \\(f_c \\approx 0.25f_s\\).

</div>
</details>

## Summary

Here is what the core pattern of this section looks like

```mojo
current_val = input[global_i]
neighbor_val = shuffle_down(current_val, offset)
if lane < WARP_SIZE - offset:
    result = compute(current_val, neighbor_val)
```

**Key benefits:**

- **Hardware efficiency**: Register-to-register communication
- **Boundary safety**: Automatic warp limit handling
- **SIMT optimization**: Single instruction, all lanes parallel

**Applications**: Finite differences, stencil operations, moving averages, convolutions.
