# block.broadcast() Vector Normalization

Implement vector mean normalization by combining [block.sum](https://docs.modular.com/mojo/std/gpu/primitives/block/sum) and [block.broadcast](https://docs.modular.com/mojo/std/gpu/primitives/block/broadcast) operations to demonstrate the complete block-level communication workflow. Each thread will contribute to computing the mean, then receive the broadcast mean to normalize its element, showcasing how block operations work together to solve real parallel algorithms.

**Key insight:** _The [block.broadcast()](https://docs.modular.com/mojo/std/gpu/primitives/block/broadcast) operation enables one-to-all communication, completing the fundamental block communication patterns: reduction (all→one), scan (all→each), and broadcast (one→all)._

## Key concepts

In this puzzle, you'll learn:

- **Block-level broadcast** with `block.broadcast()`
- **One-to-all communication** patterns
- **Source thread specification** and parameter control
- **Complete block operations workflow** combining multiple operations
- **Real-world algorithm implementation** using coordinated block primitives

The algorithm demonstrates vector mean normalization:
\\[\Large \text{output}[i] = \frac{\text{input}[i]}{\frac{1}{N}\sum_{j=0}^{N-1} \text{input}[j]}\\]

Each thread contributes to the mean calculation, then receives the broadcast mean to normalize its element.

## Configuration

- Vector size: `SIZE = 128` elements
- Data type: `DType.float32`
- Block configuration: `(128, 1)` threads per block (`TPB = 128`)
- Grid configuration: `(1, 1)` blocks per grid
- Layout: `row_major[SIZE]()` (1D row-major for input and output)
- Test data: Values cycling 1-8, so mean = 4.5
- Expected output: Normalized vector with mean = 1.0

## The challenge: Coordinating block-wide computation and distribution

Traditional approaches to mean normalization require complex coordination:

```python
# Sequential approach - doesn't utilize parallelism
total = sum(input_array)
mean = total / len(input_array)
output_array = [x / mean for x in input_array]
```

**Problems with naive GPU parallelization:**

- **Multiple kernel launches**: One pass to compute mean, another to normalize
- **Global memory round-trip**: Store mean to global memory, read back later
- **Synchronization complexity**: Need barriers between computation phases
- **Thread divergence**: Different threads doing different tasks

**Traditional GPU solution complexity:**

```mojo
# Phase 1: Reduce to find sum (complex shared memory + barriers)
shared_sum[local_i] = my_value
barrier()
# Manual tree reduction with multiple barrier() calls...

# Phase 2: Thread 0 computes mean
if local_i == 0:
    mean = shared_sum[0] / size
    shared_mean[0] = mean

barrier()

# Phase 3: All threads read mean and normalize
mean = shared_mean[0]  # Everyone reads the same value
output[global_i] = my_value / mean
```

## The advanced approach: `block.sum()` + `block.broadcast()` coordination

Transform the multi-phase coordination into elegant block operations workflow:

## Code to complete

### Complete block operations workflow

Implement sophisticated vector mean normalization using the full block operations toolkit:

```mojo
{{#include ../../../problems/p27/p27.mojo:block_normalize}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p27/p27.mojo" class="filename">View full file: problems/p27/p27.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### 1. **Complete workflow structure (builds on all previous operations)**

The algorithm follows the perfect block operations pattern:

1. Each thread loads its element (familiar from all previous puzzles)
2. Use `block.sum()` to compute total (from earlier in this puzzle)
3. Thread 0 computes mean from the sum
4. Use `block.broadcast()` to share mean to all threads (NEW!)
5. Each thread normalizes using the broadcast mean

### 2. **Data loading and sum computation (familiar patterns)**

Load your element using the established TileTensor pattern:

```mojo
var my_value: Scalar[dtype] = 0.0
if global_i < size:
    my_value = input_data[global_i][0]  # SIMD extraction
```

Then use `block.sum()` exactly like the dot product earlier:

```mojo
total_sum = block.sum[block_size=tpb, broadcast=False](...)
```

### 3. **Mean computation (thread 0 only)**

Only thread 0 should compute the mean:

```mojo
var mean_value: Scalar[dtype] = 1.0  # Safe default
if local_i == 0:
    # Compute mean from total_sum and size
```

**Why thread 0?** Consistent with `block.sum()` pattern where thread 0 receives the result.

### 4. **[block.broadcast()](https://docs.modular.com/mojo/std/gpu/primitives/block/broadcast) API concepts**

Study the function signature - it needs:

- Template parameters: `dtype`, `width`, `block_size`
- Runtime parameters: `val` (SIMD value to broadcast), `src_thread` (default=0)

The call pattern follows the established template style:

```mojo
result = block.broadcast[
    dtype = DType.float32,
    width = 1,
    block_size = tpb
](val=SIMD[DType.float32, 1](value_to_broadcast), src_thread=UInt(0))
```

### 5. **Understanding the broadcast pattern**

**Key insight**: `block.broadcast()` takes a value from ONE thread and gives it to ALL threads:

- **Thread 0** has the computed mean value
- **All threads** need that same mean value
- **`block.broadcast()`** copies thread 0's value to everyone

This is the opposite of `block.sum()` (all→one) and different from `block.prefix_sum()` (all→each position).

### 6. **Final normalization step**

Once every thread has the broadcast mean, normalize your element:

```mojo
if global_i < size:
    normalized_value = my_value / broadcasted_mean[0]  # Extract SIMD
    output_data[global_i] = normalized_value
```

**SIMD extraction**: Remember that `block.broadcast()` returns SIMD, so use `[0]` to extract the scalar.

### 7. **Pattern recognition from previous puzzles**

- **Thread indexing**: Same `global_i`, `local_i` pattern as always
- **Bounds checking**: Same `if global_i < size` validation
- **SIMD handling**: Same `[0]` extraction patterns
- **Block operations**: Same template parameter style as `block.sum()`

The beauty is that each block operation follows consistent patterns!

</div>
</details>

**Test the block.broadcast() approach:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --normalize
```

  </div>
</div>

Expected output when solved:

```txt
SIZE: 128
TPB: 128

Input sample: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 ...
Sum value: 576.0
Mean value: 4.5

Mean Normalization Results:
Normalized sample: 0.22222222 0.44444445 0.6666667 0.8888889 1.1111112 1.3333334 1.5555556 1.7777778 ...

Output sum: 128.0
Output mean: 1.0
✅ Success: Output mean is 1.0 (should be close to 1.0)
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p27/p27.mojo:block_normalize_solution}}
```

<div class="solution-explanation">

The `block.broadcast()` kernel demonstrates the complete block operations workflow by combining all three fundamental communication patterns in a real algorithm that produces mathematically verifiable results:

## **Complete algorithm walkthrough with concrete execution:**

### **Phase 1: Parallel data loading (established patterns from all previous puzzles)**

```
Thread indexing (consistent across all puzzles):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // Maps to input array position
  local_i = thread_idx.x                              // Position within block (0-127)

Parallel element loading using TileTensor pattern:
  Thread 0:   my_value = input_data[0][0] = 1.0    // First cycle value
  Thread 1:   my_value = input_data[1][0] = 2.0    // Second cycle value
  Thread 7:   my_value = input_data[7][0] = 8.0    // Last cycle value
  Thread 8:   my_value = input_data[8][0] = 1.0    // Cycle repeats: 1,2,3,4,5,6,7,8,1,2...
  Thread 15:  my_value = input_data[15][0] = 8.0   // 15 % 8 = 7, so 8th value
  Thread 127: my_value = input_data[127][0] = 8.0  // 127 % 8 = 7, so 8th value

All 128 threads load simultaneously - perfect parallel efficiency!
```

### **Phase 2: Block-wide sum reduction (leveraging earlier block.sum() knowledge)**

```
block.sum() coordination across all 128 threads:
  Contribution analysis:
    - Values 1,2,3,4,5,6,7,8 repeat 16 times each (128/8 = 16)
    - Thread contributions: 16×1 + 16×2 + 16×3 + 16×4 + 16×5 + 16×6 + 16×7 + 16×8
    - Mathematical sum: 16 × (1+2+3+4+5+6+7+8) = 16 × 36 = 576.0

block.sum() hardware execution:
  All threads → [reduction tree] → Thread 0
  total_sum = SIMD[DType.float32, 1](576.0)  // Only thread 0 receives this

Threads 1-127: Have no access to total_sum (broadcast=False in block.sum)
```

### **Phase 3: Exclusive mean computation (single-thread processing)**

```
Thread 0 performs critical computation:
  Input: total_sum[0] = 576.0, size = 128
  Computation: mean_value = 576.0 / 128.0 = 4.5

  Verification: Expected mean = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5 ✓

All other threads (1-127):
  mean_value = 1.0 (default safety value)
  These values are irrelevant - will be overwritten by broadcast

Critical insight: Only thread 0 has the correct mean value at this point!
```

### **Phase 4: Block-wide broadcast distribution (one → all communication)**

```
block.broadcast() API execution:
  Source: src_thread = UInt(0) → Thread 0's mean_value = 4.5
  Target: All 128 threads in block

Before broadcast:
  Thread 0:   mean_value = 4.5  ← Source of truth
  Thread 1:   mean_value = 1.0  ← Will be overwritten
  Thread 2:   mean_value = 1.0  ← Will be overwritten
  ...
  Thread 127: mean_value = 1.0  ← Will be overwritten

After block.broadcast() execution:
  Thread 0:   broadcasted_mean[0] = 4.5  ← Receives own value back
  Thread 1:   broadcasted_mean[0] = 4.5  ← Now has correct value!
  Thread 2:   broadcasted_mean[0] = 4.5  ← Now has correct value!
  ...
  Thread 127: broadcasted_mean[0] = 4.5  ← Now has correct value!

Result: Perfect synchronization - all threads have identical mean value!
```

### **Phase 5: Parallel mean normalization (coordinated processing)**

```
Each thread independently normalizes using broadcast mean:
  Thread 0:   normalized = 1.0 / 4.5 = 0.22222222...
  Thread 1:   normalized = 2.0 / 4.5 = 0.44444444...
  Thread 2:   normalized = 3.0 / 4.5 = 0.66666666...
  Thread 7:   normalized = 8.0 / 4.5 = 1.77777777...
  Thread 8:   normalized = 1.0 / 4.5 = 0.22222222...  (pattern repeats)
  ...

Mathematical verification:
  Output sum = (0.222... + 0.444... + ... + 1.777...) × 16 = 4.5 × 16 × 2 = 128.0
  Output mean = 128.0 / 128 = 1.0  Perfect normalization!

Each value divided by original mean gives output with mean = 1.0
```

### **Phase 6: Verification of correctness**

```
Input analysis:
  - Sum: 576.0, Mean: 4.5
  - Max: 8.0, Min: 1.0
  - Range: [1.0, 8.0]

Output analysis:
  - Sum: 128.0, Mean: 1.0 ✓
  - Max: 1.777..., Min: 0.222...
  - Range: [0.222, 1.777] (all values scaled by factor 1/4.5)

Proportional relationships preserved:
  - Original 8:1 ratio becomes 1.777:0.222 = 8:1 ✓
  - All relative magnitudes maintained perfectly
```

## **Why this complete workflow is mathematically and computationally superior:**

### **Technical accuracy and verification:**

```
Mathematical proof of correctness:
  Input: x₁, x₂, ..., xₙ where n = 128
  Mean: μ = (∑xᵢ)/n = 576/128 = 4.5

  Normalization: yᵢ = xᵢ/μ
  Output mean: (∑yᵢ)/n = (∑xᵢ/μ)/n = (1/μ)(∑xᵢ)/n = (1/μ)μ = 1 ✓

Algorithm produces provably correct mathematical result.
```

### **Connection to [Puzzle 12](../puzzle_12/tile_tensor.md) (foundational patterns):**

- **Thread coordination evolution**: Same `global_i`, `local_i` patterns but with block primitives
- **Memory access patterns**: Same TileTensor SIMD extraction `[0]` but optimized workflow
- **Complexity elimination**: Replaces 20+ lines of manual barriers with 2 block operations
- **Educational progression**: Manual → automated, complex → simple, error-prone → reliable

### **Connection to [`block.sum()`](./block_sum.md) (perfect integration):**

- **API consistency**: Identical template structure `[block_size=tpb, broadcast=False]`
- **Result flow design**: Thread 0 receives sum, naturally computes derived parameter
- **Seamless composition**: Output of `block.sum()` becomes input for computation + broadcast
- **Performance optimization**: Single-kernel workflow vs multi-pass approaches

### **Connection to [`block.prefix_sum()`](./block_prefix_sum.md) (complementary communication):**

- **Distribution patterns**: `prefix_sum` gives unique positions, `broadcast` gives shared values
- **Usage scenarios**: `prefix_sum` for parallel partitioning, `broadcast` for parameter sharing
- **Template consistency**: Same `dtype`, `block_size` parameter patterns across all operations
- **SIMD handling uniformity**: All block operations return SIMD requiring `[0]` extraction

### **Advanced algorithmic insights:**

```
Communication pattern comparison:
  Traditional approach:
    1. Manual reduction:     O(log n) with explicit barriers
    2. Shared memory write:  O(1) with synchronization
    3. Shared memory read:   O(1) with potential bank conflicts
    Total: Multiple synchronization points, error-prone

  Block operations approach:
    1. block.sum():          O(log n) hardware-optimized, automatic barriers
    2. Computation:          O(1) single thread
    3. block.broadcast():    O(log n) hardware-optimized, automatic distribution
    Total: Two primitives, automatic synchronization, provably correct
```

### **Real-world algorithm patterns demonstrated:**

```
Common parallel algorithm structure:
  Phase 1: Parallel data processing      → All threads contribute
  Phase 2: Global parameter computation  → One thread computes
  Phase 3: Parameter distribution        → All threads receive
  Phase 4: Coordinated parallel output   → All threads process

This exact pattern appears in:
  - Batch normalization (deep learning)
  - Histogram equalization (image processing)
  - Iterative numerical methods (scientific computing)
  - Lighting calculations (computer graphics)

Mean normalization is the perfect educational example of this fundamental pattern.
```

## **Block operations trilogy completed:**

### **1. `block.sum()` - All to One (Reduction)**

- **Input**: All threads provide values
- **Output**: Thread 0 receives aggregated result
- **Use case**: Computing totals, finding maximums, etc.

### **2. `block.prefix_sum()` - All to Each (Scan)**

- **Input**: All threads provide values
- **Output**: Each thread receives cumulative position
- **Use case**: Computing write positions, parallel partitioning

### **3. `block.broadcast()` - One to All (Broadcast)**

- **Input**: One thread provides value (typically thread 0)
- **Output**: All threads receive the same value
- **Use case**: Sharing computed parameters, configuration values

</div>
</details>

**Complete block operations progression:**

1. **Manual coordination** ([Puzzle 12](../puzzle_12/tile_tensor.md)): Understand parallel fundamentals
2. **Warp primitives** ([Puzzle 24](../puzzle_24/warp_sum.md)): Learn hardware-accelerated patterns
3. **Block reduction** ([`block.sum()`](./block_sum.md)): Learn all→one communication
4. **Block scan** ([`block.prefix_sum()`](./block_prefix_sum.md)): Learn all→each communication
5. **Block broadcast** (`block.broadcast()`): Learn one→all communication

**The complete picture:** Block operations provide the fundamental communication building blocks for sophisticated parallel algorithms, replacing complex manual coordination with clean, composable primitives.

## Performance insights and technical analysis

### **Quantitative performance comparison:**

**`block.broadcast()` vs Traditional shared memory approach (for demonstration):**

**Traditional Manual Approach:**

```
Phase 1: Manual reduction
  • Shared memory allocation: ~5 cycles
  • Barrier synchronization: ~10 cycles
  • Tree reduction loop: ~15 cycles
  • Error-prone manual indexing

Phase 2: Mean computation: ~2 cycles

Phase 3: Shared memory broadcast
  • Manual write to shared: ~2 cycles
  • Barrier synchronization: ~10 cycles
  • All threads read: ~3 cycles

Total: ~47 cycles
  + synchronization overhead
  + potential race conditions
  + manual error debugging
```

**Block Operations Approach:**

```
Phase 1: block.sum()
  • Hardware-optimized: ~3 cycles
  • Automatic barriers: 0 explicit cost
  • Optimized reduction: ~8 cycles
  • Verified correct implementation

Phase 2: Mean computation: ~2 cycles

Phase 3: block.broadcast()
  • Hardware-optimized: ~4 cycles
  • Automatic distribution: 0 explicit cost
  • Verified correct implementation

Total: ~17 cycles
  + automatic optimization
  + guaranteed correctness
  + composable design
```

### **Memory hierarchy advantages:**

**Cache efficiency:**

- **block.sum()**: Optimized memory access patterns reduce cache misses
- **block.broadcast()**: Efficient distribution minimizes memory bandwidth usage
- **Combined workflow**: Single kernel reduces global memory round-trips by 100%

**Memory bandwidth utilization:**

```
Traditional multi-kernel approach:
  Kernel 1: Input → Reduction → Global memory write
  Kernel 2: Global memory read → Broadcast → Output
  Total global memory transfers: 3× array size

Block operations single-kernel:
  Input → block.sum() → block.broadcast() → Output
  Total global memory transfers: 2× array size (33% improvement)
```

### **When to use each block operation:**

**`block.sum()` optimal scenarios:**

- **Data aggregation**: Computing totals, averages, maximum/minimum values
- **Reduction patterns**: Any all-to-one communication requirement
- **Statistical computation**: Mean, variance, correlation calculations

**`block.prefix_sum()` optimal scenarios:**

- **Parallel partitioning**: Stream compaction, histogram binning
- **Write position calculation**: Parallel output generation
- **Parallel algorithms**: Sorting, searching, data reorganization

**`block.broadcast()` optimal scenarios:**

- **Parameter distribution**: Sharing computed values to all threads
- **Configuration propagation**: Mode flags, scaling factors, thresholds
- **Coordinated processing**: When all threads need the same computed parameter

### **Composition benefits:**

```
Individual operations: Good performance, limited scope
Combined operations:   Excellent performance, comprehensive algorithms

Example combinations seen in real applications:
• block.sum() + block.broadcast():       Normalization algorithms
• block.prefix_sum() + block.sum():      Advanced partitioning
• All three together:                    Complex parallel algorithms
• With traditional patterns:             Hybrid optimization strategies
```

## Next steps

Once you've learned about the complete block operations trilogy, you're ready for:

- **Multi-block algorithms**: Coordinating operations across multiple thread blocks
- **Advanced parallel patterns**: Combining block operations for complex algorithms
- **Memory hierarchy optimization**: Efficient data movement patterns
- **Algorithm design**: Structuring parallel algorithms using block operation building blocks
- **Performance optimization**: Choosing optimal block sizes and operation combinations

💡 **Key Takeaway**: The block operations trilogy (`sum`, `prefix_sum`, `broadcast`) provides complete communication primitives for block-level parallel programming. By composing these operations, you can implement sophisticated parallel algorithms with clean, maintainable code that leverages GPU hardware optimizations. Mean normalization demonstrates how these operations work together to solve real computational problems efficiently.
