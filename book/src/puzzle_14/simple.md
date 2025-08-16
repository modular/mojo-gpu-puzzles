# Simple Version

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:** _If the size of `a` is greater than the block size, only store the sum of each block._

## Configuration
- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Shared memory: `TPB` elements

Notes:
- **Data loading**: Each thread loads one element using LayoutTensor access
- **Memory pattern**: Shared memory for intermediate results using `LayoutTensorBuild`
- **Thread sync**: Coordination between computation phases
- **Access pattern**: Stride-based parallel computation
- **Type safety**: Leveraging LayoutTensor's type system

## Code to complete

```mojo
{{#include ../../../problems/p14/p14.mojo:prefix_sum_simple}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p14/p14.mojo" class="filename">View full file: problems/p14/p14.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Load data into `shared[local_i]`
2. Use `offset = 1` and double it each step
3. Add elements where `local_i >= offset`
4. Call `barrier()` between steps
</div>
</details>

### Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p14 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run p14 --simple
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:
```txt
out: DeviceBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p14/p14.mojo:prefix_sum_simple_solution}}
```

<div class="solution-explanation">

The parallel (inclusive) prefix-sum algorithm works as follows:

### Setup & Configuration
- `TPB` (Threads Per Block) = 8
- `SIZE` (Array Size) = 8

### Race Condition Prevention
The algorithm uses explicit synchronization to prevent read-write hazards:
- **Read Phase**: All threads first read the values they need into a local variable `current_val`
- **Synchronization**: `barrier()` ensures all reads complete before any writes begin
- **Write Phase**: All threads then safely write their computed values back to shared memory

This prevents the race condition that would occur if threads simultaneously read from and write to the same shared memory locations.

**Alternative approach**: Another solution to prevent race conditions is through _double buffering_, where you allocate twice the shared memory and alternate between reading from one buffer and writing to another. While this approach eliminates race conditions completely, it requires more shared memory and adds complexity. For educational purposes, we use the explicit synchronization approach as it's more straightforward to understand.

### Thread Mapping
- `thread_idx.x`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`local_i`)
- `block_idx.x`: \\([0, 0, 0, 0, 0, 0, 0, 0]\\)
- `global_i`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`block_idx.x * TPB + thread_idx.x`)

### Initial Load to Shared Memory
```txt
Threads:      T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
Input array:  [0    1    2    3    4    5    6    7]
shared:       [0    1    2    3    4    5    6    7]
               ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
              T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 1: First Parallel Step
Active threads: \\(T_1 \ldots T_7\\) (where `local_i ≥ 1`)

**Read Phase**: Each thread reads the value it needs:
```txt
T₁ reads shared[0] = 0    T₅ reads shared[4] = 4
T₂ reads shared[1] = 1    T₆ reads shared[5] = 5
T₃ reads shared[2] = 2    T₇ reads shared[6] = 6
T₄ reads shared[3] = 3
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value to its current position:
```txt
Before:      [0    1    2    3    4    5    6    7]
Add:              +0   +1   +2   +3   +4   +5   +6
                   |    |    |    |    |    |    |
Result:      [0    1    3    5    7    9    11   13]
                   ↑    ↑    ↑    ↑    ↑    ↑    ↑
                  T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 2: Second Parallel Step
Active threads: \\(T_2 \ldots T_7\\) (where `local_i ≥ 2`)

**Read Phase**: Each thread reads the value it needs:
```txt
T₂ reads shared[0] = 0    T₅ reads shared[3] = 5
T₃ reads shared[1] = 1    T₆ reads shared[4] = 7
T₄ reads shared[2] = 3    T₇ reads shared[5] = 9
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value:
```txt
Before:      [0    1    3    5    7    9    11   13]
Add:                   +0   +1   +3   +5   +7   +9
                        |    |    |    |    |    |
Result:      [0    1    3    6    10   14   18   22]
                        ↑    ↑    ↑    ↑    ↑    ↑
                       T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 4: Third Parallel Step
Active threads: \\(T_4 \ldots T_7\\) (where `local_i ≥ 4`)

**Read Phase**: Each thread reads the value it needs:
```txt
T₄ reads shared[0] = 0    T₆ reads shared[2] = 3
T₅ reads shared[1] = 1    T₇ reads shared[3] = 6
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value:
```txt
Before:      [0    1    3    6    10   14   18   22]
Add:                              +0   +1   +3   +6
                                  |    |    |    |
Result:      [0    1    3    6    10   15   21   28]
                                  ↑    ↑    ↑    ↑
                                  T₄   T₅   T₆   T₇
```

### Final Write to Output
```txt
Threads:       T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
global_i:      0    1    2    3    4    5    6    7
output:       [0    1    3    6    10   15   21   28]
               ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
               T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Key Implementation Details

**Synchronization Pattern**: Each iteration follows a strict read → sync → write pattern:
1. `var current_val: out.element_type = 0` - Initialize local variable
2. `current_val = shared[local_i - offset]` - Read phase (if conditions met)
3. `barrier()` - Explicit synchronization to prevent race conditions
4. `shared[local_i] += current_val` - Write phase (if conditions met)
5. `barrier()` - Standard synchronization before next iteration

**Race Condition Prevention**: Without the explicit read-write separation, multiple threads could simultaneously access the same shared memory location, leading to undefined behavior. The two-phase approach with explicit synchronization ensures correctness.

**Memory Safety**: The algorithm maintains memory safety through:
- Bounds checking with `if local_i >= offset and local_i < size`
- Proper initialization of the temporary variable
- Coordinated access patterns that prevent data races

The solution ensures correct synchronization between phases using `barrier()` and handles array bounds checking with `if global_i < size`. The final result produces the inclusive prefix sum where each element \\(i\\) contains \\(\sum_{j=0}^{i} a[j]\\).
</div>
</details>
