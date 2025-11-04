# `warp.shuffle_xor()` Butterfly Communication

For warp-level butterfly communication we can use `shuffle_xor()` to create sophisticated tree-based communication patterns within a warp. This powerful primitive enables efficient parallel reductions, sorting networks, and advanced coordination algorithms without shared memory or explicit synchronization.

**Key insight:** _The [shuffle_xor()](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor) operation leverages SIMT execution to create XOR-based communication trees, enabling efficient butterfly networks and parallel algorithms that scale with \\(O(\\log n)\\) complexity relative to warp size._

> **What are butterfly networks?** [Butterfly networks](https://en.wikipedia.org/wiki/Butterfly_network) are communication topologies where threads exchange data based on XOR patterns of their indices. The name comes from the visual pattern when drawn - connections that look like butterfly wings. These networks are fundamental to parallel algorithms like FFT, bitonic sort, and parallel reductions because they enable \\(O(\\log n)\\) communication complexity.

## Key concepts

In this puzzle, you'll learn:

- **XOR-based communication patterns** with `shuffle_xor()`
- **Butterfly network topologies** for parallel algorithms
- **Tree-based parallel reductions** with \\(O(\\log n)\\) complexity
- **Conditional butterfly operations** for advanced coordination
- **Hardware-optimized parallel primitives** replacing complex shared memory

The `shuffle_xor` operation enables each lane to exchange data with lanes based on [XOR](https://en.wikipedia.org/wiki/Exclusive_or) patterns:
\\[\\Large \text{shuffle\_xor}(\text{value}, \text{mask}) = \text{value_from_lane}(\text{lane\_id} \oplus \text{mask})\\]

This transforms complex parallel algorithms into elegant butterfly communication patterns, enabling efficient tree reductions and sorting networks without explicit coordination.

## 1. Basic butterfly pair swap

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### The `shuffle_xor` concept

Traditional pair swapping requires complex indexing and coordination:

```mojo
# Traditional approach - complex and requires synchronization
shared_memory[lane] = input[global_i]
barrier()
if lane % 2 == 0:
    partner = lane + 1
else:
    partner = lane - 1
if partner < WARP_SIZE:
    swapped_val = shared_memory[partner]
```

**Problems with traditional approach:**

- **Memory overhead**: Requires shared memory allocation
- **Synchronization**: Needs explicit barriers
- **Complex logic**: Manual partner calculation and bounds checking
- **Poor scaling**: Doesn't leverage hardware communication

With `shuffle_xor()`, pair swapping becomes elegant:

```mojo
# Butterfly XOR approach - simple and hardware-optimized
current_val = input[global_i]
swapped_val = shuffle_xor(current_val, 1)  # XOR with 1 creates pairs
output[global_i] = swapped_val
```

**Benefits of shuffle_xor:**

- **Zero memory overhead**: Direct register-to-register communication
- **No synchronization**: SIMT execution guarantees correctness
- **Hardware optimized**: Single instruction for all lanes
- **Butterfly foundation**: Building block for complex parallel algorithms

### Code to complete

Implement pair swapping using `shuffle_xor()` to exchange values between adjacent pairs.

**Mathematical operation:** Create adjacent pairs that exchange values using XOR pattern:
\\[\\Large \\text{output}[i] = \\text{input}[i \oplus 1]\\]

This transforms input data `[0, 1, 2, 3, 4, 5, 6, 7, ...]` into pairs `[1, 0, 3, 2, 5, 4, 7, 6, ...]`, where each pair `(i, i+1)` swaps values through XOR communication.

```mojo
{{#include ../../../problems/p26/p26.mojo:butterfly_pair_swap_solution}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p26/p26.mojo" class="filename">View full file: problems/p26/p26.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding shuffle_xor**

The `shuffle_xor(value, mask)` operation allows each lane to exchange data with a lane whose ID differs by the XOR mask. Think about what happens when you XOR a lane ID with different mask values.

**Key question to explore:**

- What partner does lane 0 get when you XOR with mask 1?
- What partner does lane 1 get when you XOR with mask 1?
- Do you see a pattern forming?

**Hint**: Try working out the XOR operation manually for the first few lane IDs to understand the pairing pattern.

### 2. **XOR pair pattern**

Think about the binary representation of lane IDs and what happens when you flip the least significant bit.

**Questions to consider:**

- What happens to even-numbered lanes when you XOR with 1?
- What happens to odd-numbered lanes when you XOR with 1?
- Why does this create perfect pairs?

### 3. **No boundary checking needed**

Unlike `shuffle_down()`, `shuffle_xor()` operations stay within warp boundaries. Consider why XOR with small masks never creates out-of-bounds lane IDs.

**Think about**: What's the maximum lane ID you can get when XORing any valid lane ID with 1?

</div>
</details>

**Test the butterfly pair swap:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --pair-swap
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
expected: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
✅ Butterfly pair swap test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p26/p26.mojo:butterfly_pair_swap_solution}}
```

<div class="solution-explanation">

This solution demonstrates how `shuffle_xor()` creates perfect pair exchanges through XOR communication patterns.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]              # Each lane reads its element
    swapped_val = shuffle_xor(current_val, 1)  # XOR creates pair exchange

    # For demonstration, store the swapped value
    output[global_i] = swapped_val
```

**SIMT execution deep dive:**

```
Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 2
  Lane 3: current_val = input[3] = 3
  ...
  Lane 31: current_val = input[31] = 31

Cycle 2: shuffle_xor(current_val, 1) executes on all lanes
  Lane 0: receives from Lane 1 (0⊕1=1) → swapped_val = 1
  Lane 1: receives from Lane 0 (1⊕1=0) → swapped_val = 0
  Lane 2: receives from Lane 3 (2⊕1=3) → swapped_val = 3
  Lane 3: receives from Lane 2 (3⊕1=2) → swapped_val = 2
  ...
  Lane 30: receives from Lane 31 (30⊕1=31) → swapped_val = 31
  Lane 31: receives from Lane 30 (31⊕1=30) → swapped_val = 30

Cycle 3: Store results
  Lane 0: output[0] = 1
  Lane 1: output[1] = 0
  Lane 2: output[2] = 3
  Lane 3: output[3] = 2
  ...
```

**Mathematical insight:** This implements perfect pair exchange using XOR properties:
\\[\\Large \\text{XOR}(i, 1) = \\begin{cases}
i + 1 & \\text{if } i \\bmod 2 = 0 \\\\
i - 1 & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Why shuffle_xor is superior:**

1. **Perfect symmetry**: Every lane participates in exactly one pair
2. **No coordination**: All pairs exchange simultaneously
3. **Hardware optimized**: Single instruction for entire warp
4. **Butterfly foundation**: Building block for complex parallel algorithms

**Performance characteristics:**

- **Latency**: 1 cycle (hardware register exchange)
- **Bandwidth**: 0 bytes (no memory traffic)
- **Parallelism**: All WARP_SIZE lanes exchange simultaneously
- **Scalability**: \\(O(1)\\) complexity regardless of data size

</div>
</details>

## 2. Butterfly parallel maximum

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

### Code to complete

Implement parallel maximum reduction using butterfly `shuffle_xor` with decreasing offsets.

**Mathematical operation:** Compute the maximum across all warp lanes using tree reduction:
\\[\\Large \\text{max\_result} = \\max_{i=0}^{\\small\\text{WARP\_SIZE}-1} \\text{input}[i]\\]

**Butterfly reduction pattern:** Use XOR offsets starting from `WARP_SIZE/2` down to `1` to create a binary tree where each step halves the active communication range:

- **Step 1**: Compare with lanes `WARP_SIZE/2` positions away (covers full warp)
- **Step 2**: Compare with lanes `WARP_SIZE/4` positions away (covers remaining range)
- **Step 3**: Compare with lanes `WARP_SIZE/8` positions away
- **Step 4**: Continue halving until `offset = 1`

After \\(\\log_2(\\text{WARP\_SIZE})\\) steps, all lanes have the global maximum. This works for any `WARP_SIZE` (32, 64, etc.).

```mojo
{{#include ../../../problems/p26/p26.mojo:butterfly_parallel_max}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding butterfly reduction**

The butterfly reduction creates a binary tree communication pattern. Think about how you can systematically reduce the problem size at each step.

**Key questions:**

- What should be your starting offset to cover the maximum range?
- How should the offset change between steps?
- When should you stop the reduction?

**Hint**: The name "butterfly" comes from the communication pattern - try sketching it out for a small example.

### 2. **XOR reduction properties**

XOR creates non-overlapping communication pairs at each step. Consider why this is important for parallel reductions.

**Think about:**

- How does XOR with different offsets create different communication patterns?
- Why don't lanes interfere with each other at the same step?
- What makes XOR particularly well-suited for tree reductions?

### 3. **Accumulating maximum values**

Each lane needs to progressively build up knowledge of the maximum value in its "region".

**Algorithm structure:**

- Start with your own value
- At each step, compare with a neighbor's value
- Keep the maximum and continue

**Key insight**: After each step, your "region of knowledge" doubles in size.

- After final step: Each lane knows global maximum

### 4. **Why this pattern works**

The butterfly reduction guarantees that after \\(\\log_2(\\text{WARP\\_SIZE})\\) steps:

- **Every lane** has seen **every other lane's** value indirectly
- **No redundant communication**: Each pair exchanges exactly once per step
- **Optimal complexity**: \\(O(\\log n)\\) steps instead of \\(O(n)\\) sequential comparison

**Trace example** (4 lanes, values [3, 1, 7, 2]):

```
Initial: Lane 0=3, Lane 1=1, Lane 2=7, Lane 3=2

Step 1 (offset=2): 0 ↔ 2, 1 ↔ 3
  Lane 0: max(3, 7) = 7
  Lane 1: max(1, 2) = 2
  Lane 2: max(7, 3) = 7
  Lane 3: max(2, 1) = 2

Step 2 (offset=1): 0 ↔ 1, 2 ↔ 3
  Lane 0: max(7, 2) = 7
  Lane 1: max(2, 7) = 7
  Lane 2: max(7, 2) = 7
  Lane 3: max(2, 7) = 7

Result: All lanes have global maximum = 7
```

</div>
</details>

**Test the butterfly parallel maximum:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --parallel-max
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --parallel-max
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --parallel-max
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
expected: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
✅ Butterfly parallel max test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p26/p26.mojo:butterfly_parallel_max_solution}}
```

<div class="solution-explanation">

This solution demonstrates how `shuffle_xor()` creates efficient parallel reduction trees with \\(O(\\log n)\\) complexity.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    max_val = input[global_i]  # Start with local value

    # Butterfly reduction tree: dynamic for any WARP_SIZE
    offset = WARP_SIZE // 2
    while offset > 0:
        max_val = max(max_val, shuffle_xor(max_val, offset))
        offset //= 2

    output[global_i] = max_val  # All lanes have global maximum
```

**Butterfly execution trace (8-lane example, values [0,2,4,6,8,10,12,1000]):**

```
Initial state:
  Lane 0: max_val = 0,    Lane 1: max_val = 2
  Lane 2: max_val = 4,    Lane 3: max_val = 6
  Lane 4: max_val = 8,    Lane 5: max_val = 10
  Lane 6: max_val = 12,   Lane 7: max_val = 1000

Step 1: shuffle_xor(max_val, 4) - Halves exchange
  Lane 0↔4: max(0,8)=8,     Lane 1↔5: max(2,10)=10
  Lane 2↔6: max(4,12)=12,   Lane 3↔7: max(6,1000)=1000
  Lane 4↔0: max(8,0)=8,     Lane 5↔1: max(10,2)=10
  Lane 6↔2: max(12,4)=12,   Lane 7↔3: max(1000,6)=1000

Step 2: shuffle_xor(max_val, 2) - Quarters exchange
  Lane 0↔2: max(8,12)=12,   Lane 1↔3: max(10,1000)=1000
  Lane 2↔0: max(12,8)=12,   Lane 3↔1: max(1000,10)=1000
  Lane 4↔6: max(8,12)=12,   Lane 5↔7: max(10,1000)=1000
  Lane 6↔4: max(12,8)=12,   Lane 7↔5: max(1000,10)=1000

Step 3: shuffle_xor(max_val, 1) - Pairs exchange
  Lane 0↔1: max(12,1000)=1000,  Lane 1↔0: max(1000,12)=1000
  Lane 2↔3: max(12,1000)=1000,  Lane 3↔2: max(1000,12)=1000
  Lane 4↔5: max(12,1000)=1000,  Lane 5↔4: max(1000,12)=1000
  Lane 6↔7: max(12,1000)=1000,  Lane 7↔6: max(1000,12)=1000

Final result: All lanes have max_val = 1000
```

**Mathematical insight:** This implements the parallel reduction operator with butterfly communication:
\\[\\Large \\text{Reduce}(\\oplus, [a_0, a_1, \\ldots, a_{n-1}]) = a_0 \\oplus a_1 \\oplus \\cdots \\oplus a_{n-1}\\]

Where \\(\\oplus\\) is the `max` operation and the butterfly pattern ensures optimal \\(O(\\log n)\\) complexity.

**Why butterfly reduction is superior:**

1. **Logarithmic complexity**: \\(O(\\log n)\\) vs \\(O(n)\\) for sequential reduction
2. **Perfect load balancing**: Every lane participates equally at each step
3. **No memory bottlenecks**: Pure register-to-register communication
4. **Hardware optimized**: Maps directly to GPU butterfly networks

**Performance characteristics:**

- **Steps**: \\(\\log_2(\\text{WARP\_SIZE})\\) (e.g., 5 for 32-thread, 6 for 64-thread warp)
- **Latency per step**: 1 cycle (register exchange + comparison)
- **Total latency**: \\(\\log_2(\\text{WARP\_SIZE})\\) cycles vs \\((\\text{WARP\_SIZE}-1)\\) cycles for sequential
- **Parallelism**: All lanes active throughout the algorithm

</div>
</details>

## 3. Butterfly conditional maximum

### Configuration

- Vector size: `SIZE_2 = 64` (multi-block scenario)
- Grid configuration: `BLOCKS_PER_GRID_2 = (2, 1)` blocks per grid
- Block configuration: `THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)` threads per block

### Code to complete

Implement conditional butterfly reduction where even lanes store the maximum and odd lanes store the minimum.

**Mathematical operation:** Perform butterfly reduction for both maximum and minimum, then conditionally output based on lane parity:
\\[\\Large \\text{output}[i] = \\begin{cases}
\\max_{j=0}^{\\text{WARP\_SIZE}-1} \\text{input}[j] & \\text{if } i \\bmod 2 = 0 \\\\
\\min_{j=0}^{\\text{WARP\_SIZE}-1} \\text{input}[j] & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Dual reduction pattern:** Simultaneously track both maximum and minimum values through the butterfly tree, then conditionally output based on lane ID parity. This demonstrates how butterfly patterns can be extended for complex multi-value reductions.

```mojo
{{#include ../../../problems/p26/p26.mojo:butterfly_conditional_max}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Dual-track butterfly reduction**

This puzzle requires tracking TWO different values simultaneously through the butterfly tree. Think about how you can run multiple reductions in parallel.

**Key questions:**

- How can you maintain both maximum and minimum values during the reduction?
- Can you use the same butterfly pattern for both operations?
- What variables do you need to track?

### 2. **Conditional output logic**

After completing the butterfly reduction, you need to output different values based on lane parity.

**Consider:**

- How do you determine if a lane is even or odd?
- Which lanes should output the maximum vs minimum?
- How do you access the lane ID?

### 3. **Butterfly reduction for both min and max**

The challenge is efficiently computing both min and max in parallel using the same butterfly communication pattern.

**Think about:**

- Do you need separate shuffle operations for min and max?
- Can you reuse the same neighbor values for both operations?
- How do you ensure both reductions complete correctly?

### 4. **Multi-block boundary considerations**

This puzzle uses multiple blocks. Consider how this affects the reduction scope.

**Important considerations:**

- What's the scope of each butterfly reduction?
- How does the block structure affect lane numbering?
- Are you computing global or per-block min/max values?

</div>
</details>

**Test the butterfly conditional maximum:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --conditional-max
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --conditional-max
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --conditional-max
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
expected: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
✅ Butterfly conditional max test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p26/p26.mojo:butterfly_conditional_max_solution}}
```

<div class="solution-explanation">

This solution demonstrates advanced butterfly reduction with dual tracking and conditional output.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    current_val = input[global_i]
    min_val = current_val  # Track minimum separately

    # Butterfly reduction for both max and min log_2(WARP_SIZE}) steps)
    offset = WARP_SIZE // 2
    while offset > 0:
        neighbor_val = shuffle_xor(current_val, offset)
        current_val = max(current_val, neighbor_val)    # Max reduction

        min_neighbor_val = shuffle_xor(min_val, offset)
        min_val = min(min_val, min_neighbor_val)        # Min reduction

        offset //= 2

    # Conditional output based on lane parity
    if lane % 2 == 0:
        output[global_i] = current_val  # Even lanes: maximum
    else:
        output[global_i] = min_val      # Odd lanes: minimum
```

**Dual reduction execution trace (4-lane example, values [3, 1, 7, 2]):**

```
Initial state:
  Lane 0: current_val=3, min_val=3
  Lane 1: current_val=1, min_val=1
  Lane 2: current_val=7, min_val=7
  Lane 3: current_val=2, min_val=2

Step 1: shuffle_xor(current_val, 2) and shuffle_xor(min_val, 2) - Halves exchange
  Lane 0↔2: max_neighbor=7, min_neighbor=7 → current_val=max(3,7)=7, min_val=min(3,7)=3
  Lane 1↔3: max_neighbor=2, min_neighbor=2 → current_val=max(1,2)=2, min_val=min(1,2)=1
  Lane 2↔0: max_neighbor=3, min_neighbor=3 → current_val=max(7,3)=7, min_val=min(7,3)=3
  Lane 3↔1: max_neighbor=1, min_neighbor=1 → current_val=max(2,1)=2, min_val=min(2,1)=1

Step 2: shuffle_xor(current_val, 1) and shuffle_xor(min_val, 1) - Pairs exchange
  Lane 0↔1: max_neighbor=2, min_neighbor=1 → current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 1↔0: max_neighbor=7, min_neighbor=3 → current_val=max(2,7)=7, min_val=min(1,3)=1
  Lane 2↔3: max_neighbor=2, min_neighbor=1 → current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 3↔2: max_neighbor=7, min_neighbor=3 → current_val=max(2,7)=7, min_val=min(1,3)=1

Final result: All lanes have current_val=7 (global max) and min_val=1 (global min)
```

**Dynamic algorithm** (works for any WARP_SIZE):

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = max(current_val, neighbor_val)

    min_neighbor_val = shuffle_xor(min_val, offset)
    min_val = min(min_val, min_neighbor_val)

    offset //= 2
```

**Mathematical insight:** This implements dual parallel reduction with conditional demultiplexing:
\\[\\Large \\begin{align}
\\text{max\_result} &= \\max_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{min\_result} &= \\min_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{output}[i] &= \\text{lane\_parity}(i) \\; \text{?} \\; \\text{min\_result} : \\text{max\_result}
\\end{align}\\]

**Why dual butterfly reduction works:**

1. **Independent reductions**: Max and min reductions are mathematically independent
2. **Parallel execution**: Both can use the same butterfly communication pattern
3. **Shared communication**: Same shuffle operations serve both reductions
4. **Conditional output**: Lane parity determines which result to output

**Performance characteristics:**

- **Communication steps**: \\(\\log_2(\\text{WARP\_SIZE})\\) (same as single reduction)
- **Computation per step**: 2 operations (max + min) vs 1 for single reduction
- **Memory efficiency**: 2 registers per thread vs complex shared memory approaches
- **Output flexibility**: Different lanes can output different reduction results

</div>
</details>

## Summary

The `shuffle_xor()` primitive enables powerful butterfly communication patterns that form the foundation of efficient parallel algorithms. Through these three problems, you've learned:

### **Core Butterfly Patterns**

1. **Pair Exchange** (`shuffle_xor(value, 1)`):
   - Creates perfect adjacent pairs: (0,1), (2,3), (4,5), ...
   - \\(O(1)\\) complexity with zero memory overhead
   - Foundation for sorting networks and data reorganization

2. **Tree Reduction** (dynamic offsets: `WARP_SIZE/2` → `1`):
   - Logarithmic parallel reduction: \\(O(\\log n)\\) vs \\(O(n)\\) sequential
   - Works for any associative operation (max, min, sum, etc.)
   - Optimal load balancing across all warp lanes

3. **Conditional Multi-Reduction** (dual tracking + lane parity):
   - Simultaneous multiple reductions in parallel
   - Conditional output based on thread characteristics
   - Advanced coordination without explicit synchronization

### **Key Algorithmic Insights**

**XOR Communication Properties:**

- `shuffle_xor(value, mask)` creates symmetric, non-overlapping pairs
- Each mask creates a unique communication topology
- Butterfly networks emerge naturally from binary XOR patterns

**Dynamic Algorithm Design:**

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = operation(current_val, neighbor_val)
    offset //= 2
```

**Performance Advantages:**

- **Hardware optimization**: Direct register-to-register communication
- **No synchronization**: SIMT execution guarantees correctness
- **Scalable complexity**: \\(O(\\log n)\\) for any WARP_SIZE (32, 64, etc.)
- **Memory efficiency**: Zero shared memory requirements

### **Practical Applications**

These butterfly patterns are fundamental to:

- **Parallel reductions**: Sum, max, min, logical operations
- **Prefix/scan operations**: Cumulative sums, parallel sorting
- **FFT algorithms**: Signal processing and convolution
- **Bitonic sorting**: Parallel sorting networks
- **Graph algorithms**: Tree traversals and connectivity

The `shuffle_xor()` primitive transforms complex parallel coordination into elegant, hardware-optimized communication patterns that scale efficiently across different GPU architectures.
