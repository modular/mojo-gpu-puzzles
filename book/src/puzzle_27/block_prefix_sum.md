# block.prefix_sum() Parallel Histogram Binning

This puzzle implements parallel histogram binning using block-level [block.prefix_sum](https://docs.modular.com/mojo/stdlib/gpu/block/prefix_sum) operations for advanced parallel filtering and extraction. Each thread determines its element's target bin, then applies `block.prefix_sum()` to compute write positions for extracting elements from a specific bin, showing how prefix sum enables sophisticated parallel partitioning beyond simple reductions.

**Key insight:** _The [block.prefix_sum()](https://docs.modular.com/mojo/stdlib/gpu/block/prefix_sum) operation provides parallel filtering and extraction by computing cumulative write positions for matching elements across all threads in a block._

## Key concepts

This puzzle covers:

- **Block-level prefix sum** with `block.prefix_sum()`
- **Parallel filtering and extraction** using cumulative computations
- **Advanced parallel partitioning** algorithms
- **Histogram binning** with block-wide coordination
- **Exclusive vs inclusive** prefix sum patterns

The algorithm constructs histograms by extracting elements belonging to specific value ranges (bins):
\\[\Large \text{Bin}_k = \\{x_i : k/N \leq x_i < (k+1)/N\\}\\]

Each thread determines its element's bin assignment, with `block.prefix_sum()` coordinating parallel extraction.

## Configuration

- Vector size: `SIZE = 128` elements
- Data type: `DType.float32`
- Block configuration: `(128, 1)` threads per block (`TPB = 128`)
- Grid configuration: `(1, 1)` blocks per grid
- Number of bins: `NUM_BINS = 8` (ranges [0.0, 0.125), [0.125, 0.25), etc.)
- Layout: `Layout.row_major(SIZE)` (1D row-major)
- Warps per block: `128 / WARP_SIZE` (2 or 4 warps depending on GPU)

## The challenge: Parallel bin extraction

Traditional sequential histogram construction processes elements one by one:

```python
# Sequential approach - doesn't parallelize well
histogram = [[] for _ in range(NUM_BINS)]
for element in data:
    bin_id = int(element * NUM_BINS)  # Determine bin
    histogram[bin_id].append(element)  # Sequential append
```

**Problems with naive GPU parallelization:**

- **Race conditions**: Multiple threads writing to same bin simultaneously
- **Uncoalesced memory**: Threads access different memory locations
- **Load imbalance**: Some bins may have many more elements than others
- **Complex synchronization**: Need barriers and atomic operations

## The advanced approach: `block.prefix_sum()` coordination

Transform the complex parallel partitioning into coordinated extraction:

## Code to complete

### `block.prefix_sum()` approach

Implement parallel histogram binning using `block.prefix_sum()` for extraction:

```mojo
{{#include ../../../problems/p27/p27.mojo:block_histogram}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p27/p27.mojo" class="filename">View full file: problems/p27/p27.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Core algorithm structure (adapt from previous puzzles)**

Just like `block_sum_dot_product`, you need these key variables:

```mojo
global_i = block_dim.x * block_idx.x + thread_idx.x
local_i = thread_idx.x
```

Your function will have **5 main steps** (about 15-20 lines total):

1. Load element and determine its bin
2. Create binary predicate for target bin
3. Run `block.prefix_sum()` on the predicate
4. Conditionally write using computed offset
5. Final thread computes total count

### 2. **Bin calculation (use `math.floor`)**

To classify a `Float32` value into bins:

```mojo
my_value = input_data[global_i][0]  # Extract SIMD like in dot product
bin_number = Int(floor(my_value * num_bins))
```

**Edge case handling**: Values exactly 1.0 would go to bin `NUM_BINS`, but you only have bins 0 to `NUM_BINS-1`. Use an `if` statement to clamp the maximum bin.

### 3. **Binary predicate creation**

Create an integer variable (0 or 1) indicating if this thread's element belongs to target_bin:

```mojo
var belongs_to_target: Int = 0
if (thread_has_valid_element) and (my_bin == target_bin):
    belongs_to_target = 1
```

This is the key insight: prefix sum works on these binary flags to compute positions!

### 4. **`block.prefix_sum()` call pattern**

Following the documentation, the call looks like:

```mojo
offset = block.prefix_sum[
    dtype=DType.int32,         # Working with integer predicates
    block_size=tpb,            # Same as block.sum()
    exclusive=True             # Key: gives position BEFORE each thread
](val=SIMD[DType.int32, 1](my_predicate_value))
```

**Why exclusive?** Thread with predicate=1 at position 5 should write to output[4] if 4 elements came before it.

### 5. **Conditional writing pattern**

Only threads with `belongs_to_target == 1` should write:

```mojo
if belongs_to_target == 1:
    bin_output[Int(offset[0])] = my_value  # Convert SIMD to Int for indexing
```

This is just like the bounds checking pattern from [Puzzle 12](../puzzle_12/layout_tensor.md), but now the condition is "belongs to target bin."

### 6. **Final count computation**

The last thread (not thread 0!) computes the total count:

```mojo
if local_i == tpb - 1:  # Last thread in block
    total_count = offset[0] + belongs_to_target  # Inclusive = exclusive + own contribution
    count_output[0] = total_count
```

**Why last thread?** It has the highest `offset` value, so `offset + contribution` gives the total.

### 7. **Data types and conversions**

Remember the patterns from previous puzzles:

- `LayoutTensor` indexing returns SIMD: `input_data[i][0]`
- `block.prefix_sum()` returns SIMD: `offset[0]` to extract
- Array indexing needs `Int`: `Int(offset[0])` for `bin_output[...]`

</div>
</details>

**Test the block.prefix_sum() approach:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --histogram
```

  </div>
</div>

Expected output when solved:

```txt
SIZE: 128
TPB: 128
NUM_BINS: 8

Input sample: 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 ...

=== Processing Bin 0 (range [ 0.0 , 0.125 )) ===
Bin 0 count: 26
Bin 0 extracted elements: 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 ...

=== Processing Bin 1 (range [ 0.125 , 0.25 )) ===
Bin 1 count: 24
Bin 1 extracted elements: 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 ...

=== Processing Bin 2 (range [ 0.25 , 0.375 )) ===
Bin 2 count: 26
Bin 2 extracted elements: 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 ...

=== Processing Bin 3 (range [ 0.375 , 0.5 )) ===
Bin 3 count: 22
Bin 3 extracted elements: 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 ...

=== Processing Bin 4 (range [ 0.5 , 0.625 )) ===
Bin 4 count: 13
Bin 4 extracted elements: 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 ...

=== Processing Bin 5 (range [ 0.625 , 0.75 )) ===
Bin 5 count: 12
Bin 5 extracted elements: 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 ...

=== Processing Bin 6 (range [ 0.75 , 0.875 )) ===
Bin 6 count: 5
Bin 6 extracted elements: 0.75 0.76 0.77 0.78 0.79

=== Processing Bin 7 (range [ 0.875 , 1.0 )) ===
Bin 7 count: 0
Bin 7 extracted elements:
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p27/p27.mojo:block_histogram_solution}}
```

<div class="solution-explanation">

The `block.prefix_sum()` kernel demonstrates advanced parallel coordination patterns by building on concepts from previous puzzles:

## **Step-by-step algorithm walkthrough:**

### **Phase 1: Element processing (like [Puzzle 12](../puzzle_12/layout_tensor.md) dot product)**

```
Thread indexing (familiar pattern):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // Global element index
  local_i = thread_idx.x                              // Local thread index

Element loading (like LayoutTensor pattern):
  Thread 0:  my_value = input_data[0][0] = 0.00
  Thread 1:  my_value = input_data[1][0] = 0.01
  Thread 13: my_value = input_data[13][0] = 0.13
  Thread 25: my_value = input_data[25][0] = 0.25
  ...
```

### **Phase 2: Bin classification (new concept)**

```
Bin calculation using floor operation:
  Thread 0:  my_bin = Int(floor(0.00 * 8)) = 0  // Values [0.000, 0.125) → bin 0
  Thread 1:  my_bin = Int(floor(0.01 * 8)) = 0  // Values [0.000, 0.125) → bin 0
  Thread 13: my_bin = Int(floor(0.13 * 8)) = 1  // Values [0.125, 0.250) → bin 1
  Thread 25: my_bin = Int(floor(0.25 * 8)) = 2  // Values [0.250, 0.375) → bin 2
  ...
```

### **Phase 3: Binary predicate creation (filtering pattern)**

```
For target_bin=0, create extraction mask:
  Thread 0:  belongs_to_target = 1  (bin 0 == target 0)
  Thread 1:  belongs_to_target = 1  (bin 0 == target 0)
  Thread 13: belongs_to_target = 0  (bin 1 != target 0)
  Thread 25: belongs_to_target = 0  (bin 2 != target 0)
  ...

This creates binary array: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
```

### **Phase 4: Parallel prefix sum (the magic!)**

```
block.prefix_sum[exclusive=True] on predicates:
Input:     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
Exclusive: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, -, -, -, ...]
                                                      ^
                                                 doesn't matter

Key insight: Each thread gets its WRITE POSITION in the output array!
```

### **Phase 5: Coordinated extraction (conditional write)**

```
Only threads with belongs_to_target=1 write:
  Thread 0:  bin_output[0] = 0.00   // Uses write_offset[0] = 0
  Thread 1:  bin_output[1] = 0.01   // Uses write_offset[1] = 1
  Thread 12: bin_output[12] = 0.12  // Uses write_offset[12] = 12
  Thread 13: (no write)             // belongs_to_target = 0
  Thread 25: (no write)             // belongs_to_target = 0
  ...

Result: [0.00, 0.01, 0.02, ..., 0.12, ???, ???, ...] // Perfectly packed!
```

### **Phase 6: Count computation (like block.sum() pattern)**

```
Last thread computes total (not thread 0!):
  if local_i == tpb - 1:  // Thread 127 in our case
      total = write_offset[0] + belongs_to_target  // Inclusive sum formula
      count_output[0] = total
```

## **Why this advanced algorithm works:**

### **Connection to [Puzzle 12](../puzzle_12/layout_tensor.md) (Traditional dot product):**

- **Same thread indexing**: `global_i` and `local_i` patterns
- **Same bounds checking**: `if global_i < size` validation
- **Same data loading**: LayoutTensor SIMD extraction with `[0]`

### **Connection to [`block.sum()`](./block_sum.md) (earlier in this puzzle):**

- **Same block-wide operation**: All threads participate in block primitive
- **Same result handling**: Special thread (last instead of first) handles final result
- **Same SIMD conversion**: `Int(result[0])` pattern for array indexing

### **Advanced concepts unique to `block.prefix_sum()`:**

- **Every thread gets result**: Unlike `block.sum()` where only thread 0 matters
- **Coordinated write positions**: Prefix sum eliminates race conditions automatically
- **Parallel filtering**: Binary predicates enable sophisticated data reorganization

## **Performance advantages over naive approaches:**

### **vs. Atomic operations:**

- **No race conditions**: Prefix sum gives unique write positions
- **Coalesced memory**: Sequential writes improve cache performance
- **No serialization**: All writes happen in parallel

### **vs. Multi-pass algorithms:**

- **Single kernel**: Complete histogram extraction in one GPU launch
- **Full utilization**: All threads work regardless of data distribution
- **Optimal memory bandwidth**: Pattern optimized for GPU memory hierarchy

This demonstrates how `block.prefix_sum()` enables sophisticated parallel algorithms that would be complex or impossible with simpler primitives like `block.sum()`.

</div>
</details>

## Performance insights

**`block.prefix_sum()` vs Traditional:**

- **Algorithm sophistication**: Advanced parallel partitioning vs sequential processing
- **Memory efficiency**: Coalesced writes vs scattered random access
- **Synchronization**: Built-in coordination vs manual barriers and atomics
- **Scalability**: Works with any block size and bin count

**`block.prefix_sum()` vs `block.sum()`:**

- **Scope**: Every thread gets result vs only thread 0
- **Use case**: Complex partitioning vs simple aggregation
- **Algorithm type**: Parallel scan primitive vs reduction primitive
- **Output pattern**: Per-thread positions vs single total

**When to use `block.prefix_sum()`:**

- **Parallel filtering**: Extract elements matching criteria
- **Stream compaction**: Remove unwanted elements
- **Parallel partitioning**: Separate data into categories
- **Advanced algorithms**: Load balancing, sorting, graph algorithms

## Next steps

Once you've learned about `block.prefix_sum()` operations, you're ready for:

- **[Block Broadcast Operations](./block_broadcast.md)**: Sharing values across all threads in a block
- **Multi-block algorithms**: Coordinating multiple blocks for larger problems
- **Advanced parallel algorithms**: Sorting, graph traversal, dynamic load balancing
- **Complex memory patterns**: Combining block operations with sophisticated memory access

💡 **Key Takeaway**: Block prefix sum operations transform GPU programming from simple parallel computations to sophisticated parallel algorithms. While `block.sum()` simplified reductions, `block.prefix_sum()` enables advanced data reorganization patterns essential for high-performance parallel algorithms.
