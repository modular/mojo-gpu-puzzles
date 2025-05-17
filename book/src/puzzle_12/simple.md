# Simple Version

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `out`.

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
{{#include ../../../problems/p12/p12.mojo:prefix_sum_simple}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p12/p12.mojo" class="filename">View full file: problems/p12/p12.mojo</a>

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
uv run poe p12 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run p12 --simple
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
{{#include ../../../solutions/p12/p12.mojo:prefix_sum_simple_solution}}
```

<div class="solution-explanation">

The parallel (inclusive) prefix-sum algorithm works as follows:

### Setup & Configuration
- `TPB` (Threads Per Block) = 8
- `SIZE` (Array Size) = 8

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
```txt
Before:      [0    1    3    6    10   14   18   22]
Add:                              +0   +1   +3   +7
                                  |    |    |    |
Result:      [0    1    3    6    10   15   21   28]
                                  ↑    ↑    ↑    ↑
                                  T₄   T₅   T₆   T₇
```

### Final Write to Output
```txt
Threads:      T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
global_i:     0    1    2    3    4    5    6    7
out[]:       [0    1    3    6    10   15   21   28]
              ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
              T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Thread-by-Thread Execution

**\\(T_0\\) (`local_i=0`):**
- Loads `shared[0] = 0`
- Never adds (`local_i < offset` always)
- Writes `out[0] = 0`

**\\(T_1\\) (`local_i=1`):**
- Loads `shared[1] = 1`
- `offset=1`: adds `shared[0]` → 1
- `offset=2,4`: no action (`local_i < offset`)
- Writes `out[1] = 1`

**\\(T_2\\) (`local_i=2`):**
- Loads `shared[2] = 2`
- `offset=1`: adds `shared[1]` → 3
- `offset=2`: adds `shared[0]` → 3
- `offset=4`: no action
- Writes `out[2] = 3`

**\\(T_3\\) (`local_i=3`):**
- Loads `shared[3] = 3`
- `offset=1`: adds `shared[2]` → 5
- `offset=2`: adds `shared[1]` → 6
- `offset=4`: no action
- Writes `out[3] = 7`

**\\(T_4\\) (`local_i=4`):**
- Loads `shared[4] = 4`
- `offset=1`: adds `shared[3]` → 7
- `offset=2`: adds `shared[2]` → 10
- `offset=4`: adds `shared[0]` → 10
- Writes `out[4] = 10`

**\\(T_5\\) (`local_i=5`):**
- Loads `shared[5] = 5`
- `offset=1`: adds `shared[4]` → 9
- `offset=2`: adds `shared[3]` → 14
- `offset=4`: adds `shared[1]` → 15
- Writes `out[5] = 16`

**\\(T_6\\) (`local_i=6`):**
- Loads `shared[6] = 6`
- `offset=1`: adds `shared[5]` → 11
- `offset=2`: adds `shared[4]` → 18
- `offset=4`: adds `shared[2]` → 21
- Writes `out[6] = 21`

**\\(T_7\\) (`local_i=7`):**
- Loads `shared[7] = 7`
- `offset=1`: adds `shared[6]` → 13
- `offset=2`: adds `shared[5]` → 22
- `offset=4`: adds `shared[3]` → 28
- Writes `out[7] = 28`

The solution ensures correct synchronization between phases using `barrier()` and handles array bounds checking with `if global_i < size`. The final result produces the inclusive prefix sum where each element \\(i\\) contains \\(\sum_{j=0}^{i} a[j]\\).
</div>
</details>
