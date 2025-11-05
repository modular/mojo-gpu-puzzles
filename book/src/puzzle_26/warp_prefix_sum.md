# `warp.prefix_sum()` Hardware-Optimized Parallel Scan

For warp-level parallel scan operations we can use `prefix_sum()` to replace complex shared memory algorithms with hardware-optimized primitives. This powerful operation enables efficient cumulative computations, parallel partitioning, and advanced coordination algorithms that would otherwise require dozens of lines of shared memory and synchronization code.

**Key insight:** _The [prefix_sum()](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum) operation leverages hardware-accelerated parallel scan to compute cumulative operations across warp lanes with \\(O(\\log n)\\) complexity, replacing complex multi-phase algorithms with single function calls._

> **What is parallel scan?** [Parallel scan (prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum) is a fundamental parallel primitive that computes cumulative operations across data elements. For addition, it transforms `[a, b, c, d]` into `[a, a+b, a+b+c, a+b+c+d]`. This operation is essential for parallel algorithms like stream compaction, quicksort partitioning, and parallel sorting.

## Key concepts

In this puzzle, you'll learn:

- **Hardware-optimized parallel scan** with `prefix_sum()`
- **Inclusive vs exclusive prefix sum** patterns
- **Warp-level stream compaction** for data reorganization
- **Advanced parallel partitioning** combining multiple warp primitives
- **Single-warp algorithm optimization** replacing complex shared memory

This transforms multi-phase shared memory algorithms into elegant single-function calls, enabling efficient parallel scan operations without explicit synchronization.

## 1. Warp inclusive prefix sum

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### The `prefix_sum` advantage

Traditional prefix sum requires complex multi-phase shared memory algorithms. In [Puzzle 14](../puzzle_14/puzzle_14.md), we implemented this the hard way with explicit shared memory management:

```mojo
{{#include ../../../solutions/p14/p14.mojo:prefix_sum_simple_solution}}
```

**Problems with traditional approach:**

- **Memory overhead**: Requires shared memory allocation
- **Multiple barriers**: Complex multi-phase synchronization
- **Complex indexing**: Manual stride calculation and boundary checking
- **Poor scaling**: \\(O(\\log n)\\) phases with barriers between each

With `prefix_sum()`, parallel scan becomes trivial:

```mojo
# Hardware-optimized approach - single function call!
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)
output[global_i] = scan_result
```

**Benefits of prefix_sum:**

- **Zero memory overhead**: Hardware-accelerated computation
- **No synchronization**: Single atomic operation
- **Hardware optimized**: Leverages specialized scan units
- **Perfect scaling**: Works for any `WARP_SIZE` (32, 64, etc.)

### Code to complete

Implement inclusive prefix sum using the hardware-optimized `prefix_sum()` primitive.

**Mathematical operation:** Compute cumulative sum where each lane gets the sum of all elements up to and including its position:
\\[\\Large \\text{output}[i] = \\sum_{j=0}^{i} \\text{input}[j]\\]

This transforms input data `[1, 2, 3, 4, 5, ...]` into cumulative sums `[1, 3, 6, 10, 15, ...]`, where each position contains the sum of all previous elements plus itself.

```mojo
{{#include ../../../problems/p26/p26.mojo:warp_inclusive_prefix_sum}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p26/p26.mojo" class="filename">View full file: problems/p26/p26.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Understanding prefix_sum parameters**

The `prefix_sum()` function has an important template parameter that controls the scan type.

**Key questions:**

- What's the difference between inclusive and exclusive prefix sum?
- Which parameter controls this behavior?
- For inclusive scan, what should each lane output?

**Hint**: Look at the function signature and consider what "inclusive" means for cumulative operations.

### 2. **Single warp limitation**

This hardware primitive only works within a single warp. Consider the implications.

**Think about:**

- What happens if you have multiple warps?
- Why is this limitation important to understand?
- How would you extend this to multi-warp scenarios?

### 3. **Data type considerations**

The `prefix_sum` function may require specific data types for optimal performance.

**Consider:**

- What data type does your input use?
- Does `prefix_sum` expect a specific scalar type?
- How do you handle type conversions if needed?

</div>
</details>

**Test the warp inclusive prefix sum:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --prefix-sum
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
expected: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
✅ Warp inclusive prefix sum test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p26/p26.mojo:warp_inclusive_prefix_sum_solution}}
```

<div class="solution-explanation">

This solution demonstrates how `prefix_sum()` replaces complex multi-phase algorithms with a single hardware-optimized function call.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # This one call replaces ~30 lines of complex shared memory logic from Puzzle 14!
    # But it only works within the current warp (WARP_SIZE threads)
    scan_result = prefix_sum[exclusive=False](
        rebind[Scalar[dtype]](current_val)
    )

    output[global_i] = scan_result
```

**SIMT execution deep dive:**

```
Input: [1, 2, 3, 4, 5, 6, 7, 8, ...]

Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = 1
  Lane 1: current_val = 2
  Lane 2: current_val = 3
  Lane 3: current_val = 4
  ...
  Lane 31: current_val = 32

Cycle 2: prefix_sum[exclusive=False] executes (hardware-accelerated)
  Lane 0: scan_result = 1 (sum of elements 0 to 0)
  Lane 1: scan_result = 3 (sum of elements 0 to 1: 1+2)
  Lane 2: scan_result = 6 (sum of elements 0 to 2: 1+2+3)
  Lane 3: scan_result = 10 (sum of elements 0 to 3: 1+2+3+4)
  ...
  Lane 31: scan_result = 528 (sum of elements 0 to 31)

Cycle 3: Store results
  Lane 0: output[0] = 1
  Lane 1: output[1] = 3
  Lane 2: output[2] = 6
  Lane 3: output[3] = 10
  ...
```

**Mathematical insight:** This implements the inclusive prefix sum operation:
\\[\\Large \\text{output}[i] = \\sum_{j=0}^{i} \\text{input}[j]\\]

**Comparison with Puzzle 14's approach:**

- **[Puzzle 14](../puzzle_14/puzzle_14.md)**: ~30 lines of shared memory + multiple barriers + complex indexing
- **Warp primitive**: 1 function call with hardware acceleration
- **Performance**: Same \\(O(\\log n)\\) complexity, but implemented in specialized hardware
- **Memory**: Zero shared memory usage vs explicit allocation

**Evolution from Puzzle 12:** This demonstrates the power of modern GPU architectures - what required careful manual implementation in Puzzle 12 is now a single hardware-accelerated primitive. The warp-level `prefix_sum()` gives you the same algorithmic benefits with zero implementation complexity.

**Why prefix_sum is superior:**

1. **Hardware acceleration**: Dedicated scan units on modern GPUs
2. **Zero memory overhead**: No shared memory allocation required
3. **Automatic synchronization**: No explicit barriers needed
4. **Perfect scaling**: Works optimally for any `WARP_SIZE`

**Performance characteristics:**

- **Latency**: ~1-2 cycles (hardware scan units)
- **Bandwidth**: Zero memory traffic (register-only operation)
- **Parallelism**: All `WARP_SIZE` lanes participate simultaneously
- **Scalability**: \\(O(\\log n)\\) complexity with hardware optimization

**Important limitation**: This primitive only works within a single warp. For multi-warp scenarios, you would need additional coordination between warps.

</div>
</details>

## 2. Warp partition

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

### Code to complete

Implement single-warp parallel partitioning using BOTH `shuffle_xor` AND `prefix_sum` primitives.

**Mathematical operation:** Partition elements around a pivot value, placing elements `< pivot` on the left and elements `>= pivot` on the right:
\\[\\Large \\text{output} = [\\text{elements} < \\text{pivot}] \\,|\\, [\\text{elements} \\geq \\text{pivot}]\\]

**Advanced algorithm:** This combines two sophisticated warp primitives:

1. **`shuffle_xor()`**: Butterfly pattern for warp-level reduction (count left elements)
2. **`prefix_sum()`**: Exclusive scan for position calculation within partitions

This demonstrates the power of combining multiple warp primitives for complex parallel algorithms within a single warp.

```mojo
{{#include ../../../problems/p26/p26.mojo:warp_partition}}
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Multi-phase algorithm structure**

This algorithm requires several coordinated phases. Think about the logical steps needed for partitioning.

**Key phases to consider:**

- How do you identify which elements belong to which partition?
- How do you calculate positions within each partition?
- How do you determine the total size of the left partition?
- How do you write elements to their final positions?

### 2. **Predicate creation**

You need to create boolean predicates to identify partition membership.

**Think about:**

- How do you represent "this element belongs to the left partition"?
- How do you represent "this element belongs to the right partition"?
- What data type should you use for predicates that work with `prefix_sum`?

### 3. **Combining shuffle_xor and prefix_sum**

This algorithm uses both warp primitives for different purposes.

**Consider:**

- What is `shuffle_xor` used for in this context?
- What is `prefix_sum` used for in this context?
- How do these two operations work together?

### 4. **Position calculation**

The trickiest part is calculating where each element should be written in the output.

**Key insights:**

- Left partition elements: What determines their final position?
- Right partition elements: How do you offset them correctly?
- How do you combine local positions with partition boundaries?

</div>
</details>

**Test the warp partition:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p26 --partition
```

  </div>
  <div class="tab-content">

```bash
pixi run p26 --partition
```

  </div>
</div>

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
pivot: 5.0
✅ Warp partition test passed!
```

### Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p26/p26.mojo:warp_partition_solution}}
```

<div class="solution-explanation">

This solution demonstrates advanced coordination between multiple warp primitives to implement sophisticated parallel algorithms.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # Phase 1: Create warp-level predicates
    predicate_left = Float32(1.0) if current_val < pivot else Float32(0.0)
    predicate_right = Float32(1.0) if current_val >= pivot else Float32(0.0)

    # Phase 2: Warp-level prefix sum to get positions within warp
    warp_left_pos = prefix_sum[exclusive=True](predicate_left)
    warp_right_pos = prefix_sum[exclusive=True](predicate_right)

    # Phase 3: Get total left count using shuffle_xor reduction
    warp_left_total = predicate_left

    # Butterfly reduction to get total across the warp: dynamic for any WARP_SIZE
    offset = WARP_SIZE // 2
    while offset > 0:
        warp_left_total += shuffle_xor(warp_left_total, offset)
        offset //= 2

    # Phase 4: Write to output positions
    if current_val < pivot:
        # Left partition: use warp-level position
        output[Int(warp_left_pos)] = current_val
    else:
        # Right partition: offset by total left count + right position
        output[Int(warp_left_total + warp_right_pos)] = current_val
```

**Multi-phase execution trace (8-lane example, pivot=5, values [3,7,1,8,2,9,4,6]):**

```
Initial state:
  Lane 0: current_val=3 (< 5)  Lane 1: current_val=7 (>= 5)
  Lane 2: current_val=1 (< 5)  Lane 3: current_val=8 (>= 5)
  Lane 4: current_val=2 (< 5)  Lane 5: current_val=9 (>= 5)
  Lane 6: current_val=4 (< 5)  Lane 7: current_val=6 (>= 5)

Phase 1: Create predicates
  Lane 0: predicate_left=1.0, predicate_right=0.0
  Lane 1: predicate_left=0.0, predicate_right=1.0
  Lane 2: predicate_left=1.0, predicate_right=0.0
  Lane 3: predicate_left=0.0, predicate_right=1.0
  Lane 4: predicate_left=1.0, predicate_right=0.0
  Lane 5: predicate_left=0.0, predicate_right=1.0
  Lane 6: predicate_left=1.0, predicate_right=0.0
  Lane 7: predicate_left=0.0, predicate_right=1.0

Phase 2: Exclusive prefix sum for positions
  warp_left_pos:  [0, 0, 1, 1, 2, 2, 3, 3]
  warp_right_pos: [0, 0, 0, 1, 1, 2, 2, 3]

Phase 3: Butterfly reduction for left total
  Initial: [1, 0, 1, 0, 1, 0, 1, 0]
  After reduction: all lanes have warp_left_total = 4

Phase 4: Write to output positions
  Lane 0: current_val=3 < pivot → output[0] = 3
  Lane 1: current_val=7 >= pivot → output[4+0] = output[4] = 7
  Lane 2: current_val=1 < pivot → output[1] = 1
  Lane 3: current_val=8 >= pivot → output[4+1] = output[5] = 8
  Lane 4: current_val=2 < pivot → output[2] = 2
  Lane 5: current_val=9 >= pivot → output[4+2] = output[6] = 9
  Lane 6: current_val=4 < pivot → output[3] = 4
  Lane 7: current_val=6 >= pivot → output[4+3] = output[7] = 6

Final result: [3, 1, 2, 4, 7, 8, 9, 6] (< pivot | >= pivot)
```

**Mathematical insight:** This implements parallel partitioning with dual warp primitives:
\\[\\Large \\begin{align}
\\text{left\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_left}[i]) \\\\
\\text{right\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_right}[i]) \\\\
\\text{left\\_total} &= \\text{butterfly\\_reduce}(\\text{predicate\\_left}) \\\\
\\text{final\\_pos}[i] &= \\begin{cases}
\\text{left\\_pos}[i] & \\text{if } \\text{input}[i] < \\text{pivot} \\\\
\\text{left\\_total} + \\text{right\\_pos}[i] & \\text{if } \\text{input}[i] \\geq \\text{pivot}
\\end{cases}
\\end{align}\\]

**Why this multi-primitive approach works:**

1. **Predicate creation**: Identifies partition membership for each element
2. **Exclusive prefix sum**: Calculates relative positions within each partition
3. **Butterfly reduction**: Computes partition boundary (total left count)
4. **Coordinated write**: Combines local positions with global partition structure

**Algorithm complexity:**

- **Phase 1**: \\(O(1)\\) - Predicate creation
- **Phase 2**: \\(O(\\log n)\\) - Hardware-accelerated prefix sum
- **Phase 3**: \\(O(\\log n)\\) - Butterfly reduction with `shuffle_xor`
- **Phase 4**: \\(O(1)\\) - Coordinated write
- **Total**: \\(O(\\log n)\\) with excellent constants

**Performance characteristics:**

- **Communication steps**: \\(2 \\times \\log_2(\\text{WARP\_SIZE})\\) (prefix sum + butterfly reduction)
- **Memory efficiency**: Zero shared memory, all register-based
- **Parallelism**: All lanes active throughout algorithm
- **Scalability**: Works for any `WARP_SIZE` (32, 64, etc.)

**Practical applications:** This pattern is fundamental to:

- **Quicksort partitioning**: Core step in parallel sorting algorithms
- **Stream compaction**: Removing null/invalid elements from data streams
- **Parallel filtering**: Separating data based on complex predicates
- **Load balancing**: Redistributing work based on computational requirements

</div>
</details>

## Summary

The `prefix_sum()` primitive enables hardware-accelerated parallel scan operations that replace complex multi-phase algorithms with single function calls. Through these two problems, you've learned:

### **Core Prefix Sum Patterns**

1. **Inclusive Prefix Sum** (`prefix_sum[exclusive=False]`):
   - Hardware-accelerated cumulative operations
   - Replaces ~30 lines of shared memory code with single function call
   - \\(O(\\log n)\\) complexity with specialized hardware optimization

2. **Advanced Multi-Primitive Coordination** (combining `prefix_sum` + `shuffle_xor`):
   - Sophisticated parallel algorithms within single warp
   - Exclusive scan for position calculation + butterfly reduction for totals
   - Complex partitioning operations with optimal parallel efficiency

### **Key Algorithmic Insights**

**Hardware Acceleration Benefits:**

- `prefix_sum()` leverages dedicated scan units on modern GPUs
- Zero shared memory overhead compared to traditional approaches
- Automatic synchronization without explicit barriers

**Multi-Primitive Coordination:**

```mojo
# Phase 1: Create predicates for partition membership
predicate = 1.0 if condition else 0.0

# Phase 2: Use prefix_sum for local positions
local_pos = prefix_sum[exclusive=True](predicate)

# Phase 3: Use shuffle_xor for global totals
global_total = butterfly_reduce(predicate)

# Phase 4: Combine for final positioning
final_pos = local_pos + partition_offset
```

**Performance Advantages:**

- **Hardware optimization**: Specialized scan units vs software implementation
- **Memory efficiency**: Register-only operations vs shared memory allocation
- **Scalable complexity**: \\(O(\\log n)\\) with hardware acceleration
- **Single-warp optimization**: Perfect for algorithms within `WARP_SIZE` limits

### **Practical Applications**

These prefix sum patterns are fundamental to:

- **Parallel scan operations**: Cumulative sums, products, min/max scans
- **Stream compaction**: Parallel filtering and data reorganization
- **Quicksort partitioning**: Core parallel sorting algorithm building block
- **Parallel algorithms**: Load balancing, work distribution, data restructuring

The combination of `prefix_sum()` and `shuffle_xor()` demonstrates how modern GPU warp primitives can implement sophisticated parallel algorithms with minimal code complexity and optimal performance characteristics.
