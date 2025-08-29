# Tiled Matrix Multiplication

## Overview

Implement a kernel that multiplies square matrices \\(A\\) and \\(B\\) using tiled matrix multiplication with LayoutTensor. This approach handles large matrices by processing them in smaller chunks (tiles).

## Key concepts

- Matrix tiling with LayoutTensor for efficient computation
- Multi-block coordination with proper layouts
- Efficient shared memory usage through TensorBuilder
- Boundary handling for tiles with LayoutTensor indexing

## Configuration

- Matrix size: \\(\\text{SIZE\_TILED} = 9\\)
- Threads per block: \\(\\text{TPB} \times \\text{TPB} = 3 \times 3\\)
- Grid dimensions: \\(3 \times 3\\) blocks
- Shared memory: Two \\(\\text{TPB} \times \\text{TPB}\\) LayoutTensors per block

Layout configuration:

- Input A: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Input B: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Output: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Shared Memory: Two `TPB × TPB` LayoutTensors using TensorBuilder

## Tiling strategy

### Block organization

```txt
Grid Layout (3×3):           Thread Layout per Block (3×3):
[B00][B01][B02]               [T00 T01 T02]
[B10][B11][B12]               [T10 T11 T12]
[B20][B21][B22]               [T20 T21 T22]

Each block processes a tile using LayoutTensor indexing
```

### Tile processing steps

1. Calculate global and local indices for thread position
2. Allocate shared memory for A and B tiles
3. For each tile:
   - Load tile from matrix A and B
   - Compute partial products
   - Accumulate results in registers
4. Write final accumulated result

### Memory access pattern

```txt
Matrix A (8×8)                 Matrix B (8×8)               Matrix C (8×8)
+---+---+---+                  +---+---+---+                +---+---+---+
|T00|T01|T02| ...              |T00|T01|T02| ...            |T00|T01|T02| ...
+---+---+---+                  +---+---+---+                +---+---+---+
|T10|T11|T12|                  |T10|T11|T12|                |T10|T11|T12|
+---+---+---+                  +---+---+---+                +---+---+---+
|T20|T21|T22|                  |T20|T21|T22|                |T20|T21|T22|
+---+---+---+                  +---+---+---+                +---+---+---+
  ...                            ...                          ...

Tile Processing (for computing C[T11]):
1. Load tiles from A and B:
   +---+      +---+
   |A11| ×    |B11|     For each phase k:
   +---+      +---+     C[T11] += A[row, k] × B[k, col]

2. Tile movement:
   Phase 1     Phase 2     Phase 3
   A: [T10]    A: [T11]    A: [T12]
   B: [T01]    B: [T11]    B: [T21]

3. Each thread (i,j) in tile computes:
   C[i,j] = Σ (A[i,k] × B[k,j]) for k in tile width

Synchronization required:
* After loading tiles to shared memory
* After computing each phase
```

## Code to complete

```mojo
{{#include ../../../problems/p16/p16.mojo:matmul_tiled}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">View full file: problems/p16/p16.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Use the standard indexing convention: `local_row = thread_idx.y` and `local_col = thread_idx.x`
2. Calculate global positions:

   ```
   global_row = block_idx.y * TPB + local_row
   ```

   and

   ```
   global_col = block_idx.x * TPB + local_col
   ```

   **Understanding the global indexing formula:**
   - Each block processes a `TPB × TPB` tile of the matrix
   - `block_idx.y` tells us which row of blocks we're in (0, 1, 2...)
   - `block_idx.y * TPB` gives us the starting row of our block's tile
   - `local_row` (0 to TPB-1) is our thread's offset within the block
   - Adding them gives our thread's actual row in the full matrix

       **Example with TPB=3:**

    ```txt
    Block Layout:        Global Matrix (9×9):
    [B00][B01][B02]      [0 1 2 | 3 4 5 | 6 7 8]
    [B10][B11][B12]  →   [9 A B | C D E | F G H]
    [B20][B21][B22]      [I J K | L M N | O P Q]
                         ——————————————————————
                         [R S T | U V W | X Y Z]
                         [a b c | d e f | g h i]
                         [j k l | m n o | p q r]
                         ——————————————————————
                         [s t u | v w x | y z α]
                         [β γ δ | ε ζ η | θ ι κ]
                         [λ μ ν | ξ ο π | ρ σ τ]

    Thread(1,2) in Block(1,0):
    - block_idx.y = 1, local_row = 1
    - global_row = 1 * 3 + 1 = 4
    - This thread handles row 4 of the matrix
    ```

3. Allocate shared memory (now pre-initialized with `.fill(0)`)
4. With 9×9 perfect tiling, no bounds checking needed!
5. Accumulate results across tiles with proper synchronization

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p16 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run p16 --tiled
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([3672.0, 3744.0, 3816.0, 3888.0, 3960.0, 4032.0, 4104.0, 4176.0, 4248.0, 9504.0, 9738.0, 9972.0, 10206.0, 10440.0, 10674.0, 10908.0, 11142.0, 11376.0, 15336.0, 15732.0, 16128.0, 16524.0, 16920.0, 17316.0, 17712.0, 18108.0, 18504.0, 21168.0, 21726.0, 22284.0, 22842.0, 23400.0, 23958.0, 24516.0, 25074.0, 25632.0, 27000.0, 27720.0, 28440.0, 29160.0, 29880.0, 30600.0, 31320.0, 32040.0, 32760.0, 32832.0, 33714.0, 34596.0, 35478.0, 36360.0, 37242.0, 38124.0, 39006.0, 39888.0, 38664.0, 39708.0, 40752.0, 41796.0, 42840.0, 43884.0, 44928.0, 45972.0, 47016.0, 44496.0, 45702.0, 46908.0, 48114.0, 49320.0, 50526.0, 51732.0, 52938.0, 54144.0, 50328.0, 51696.0, 53064.0, 54432.0, 55800.0, 57168.0, 58536.0, 59904.0, 61272.0])
```

## Solution: Manual tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p16/p16.mojo:matmul_tiled_solution}}
```

<div class="solution-explanation">

The tiled matrix multiplication implementation demonstrates efficient handling of matrices \\((9 \times 9)\\) using small tiles \\((3 \times 3)\\). Here's how it works:

1. **Shared memory allocation**

   ```txt
   Input matrices (9×9) - Perfect fit for (3×3) tiling:
   A = [0  1  2  3  4  5  6  7  8 ]    B = [0  2  4  6  8  10 12 14 16]
       [9  10 11 12 13 14 15 16 17]        [18 20 22 24 26 28 30 32 34]
       [18 19 20 21 22 23 24 25 26]        [36 38 40 42 44 46 48 50 52]
       [27 28 29 30 31 32 33 34 35]        [54 56 58 60 62 64 66 68 70]
       [36 37 38 39 40 41 42 43 44]        [72 74 76 78 80 82 84 86 88]
       [45 46 47 48 49 50 51 52 53]        [90 92 94 96 98 100 102 104 106]
       [54 55 56 57 58 59 60 61 62]        [108 110 112 114 116 118 120 122 124]
       [63 64 65 66 67 68 69 70 71]        [126 128 130 132 134 136 138 140 142]
       [72 73 74 75 76 77 78 79 80]        [144 146 148 150 152 154 156 158 160]

   Shared memory per block (3×3):
   a_shared[TPB, TPB]  b_shared[TPB, TPB]
   ```

2. **Tile processing loop**

   ```txt
   Number of tiles = 9 // 3 = 3 tiles (perfect division!)

   For each tile:
   1. Load tile from A and B
   2. Compute partial products
   3. Accumulate in register
   ```

3. **Memory loading pattern**
   - With perfect \\((9 \times 9)\\) tiling, bounds check is technically unnecessary but included for defensive programming and consistency with other matrix sizes.

     ```mojo
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # Load B tile - row determined by tile, global col stays the same
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]
     ```

4. **Computation within tile**

   ```mojo
   for k in range(min(TPB, size - tile * TPB)):
       acc += a_shared[local_row, k] * b_shared[k, local_col]
   ```

   - Avoids shared memory bank conflicts:

     ```txt
     Bank Conflict Free (Good):        Bank Conflicts (Bad):
     Thread0: a_shared[0,k] b_shared[k,0]  Thread0: a_shared[k,0] b_shared[0,k]
     Thread1: a_shared[0,k] b_shared[k,1]  Thread1: a_shared[k,0] b_shared[1,k]
     Thread2: a_shared[0,k] b_shared[k,2]  Thread2: a_shared[k,0] b_shared[2,k]
     ↓                                     ↓
     Parallel access to different banks    Serialized access to same bank of b_shared
     (or broadcast for a_shared)           if shared memory was column-major
     ```

     **Shared memory bank conflicts explained:**
     - **Left (Good)**: `b_shared[k,threadIdx.x]` accesses different banks, `a_shared[0,k]` broadcasts to all threads
     - **Right (Bad)**: If b_shared were column-major, threads would access same bank simultaneously
     - **Key insight**: This is about shared memory access patterns, not global memory coalescing
     - **Bank structure**: Shared memory has 32 banks; conflicts occur when multiple threads access different addresses in the same bank simultaneously

5. **Synchronization points**

   ```txt
   barrier() after:
   1. Tile loading
   2. Tile computation
   ```

Key performance features:

- Processes \\((9 \times 9)\\) matrix using \\((3 \times 3)\\) tiles (perfect fit!)
- Uses shared memory for fast tile access
- Minimizes global memory transactions with coalesced memory access
- Optimized shared memory layout and access pattern to avoid shared memory bank conflicts

6. **Result writing**:

   ```mojo
   if tiled_row < size and tiled_col < size:
      output[tiled_row, tiled_col] = acc
   ```

   - Defensive bounds checking included for other matrix sizes and tiling strategies
   - Direct assignment to output matrix
   - All threads write valid results

### Key optimizations

1. **Layout optimization**:
   - Row-major layout for all tensors
   - Efficient 2D indexing

2. **Memory access**:
   - Coalesced global memory loads
   - Efficient shared memory usage

3. **Computation**:
   - Register-based accumulation i.e. `var acc: output.element_type = 0`
   - Compile-time loop unrolling via `@parameter`

This implementation achieves high performance through:

- Efficient use of LayoutTensor for memory access
- Optimal tiling strategy
- Proper thread synchronization
- Careful boundary handling

</div>
</details>

## Solution: Idiomatic LayoutTensor tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p16/p16.mojo:matmul_idiomatic_tiled_solution}}
```

<div class="solution-explanation">

The idiomatic tiled matrix multiplication leverages Mojo's LayoutTensor API and asynchronous memory operations for a beautifully clean implementation.

**🔑 Key Point: This implementation performs standard matrix multiplication A × B using coalesced loading for both matrices.**

**What this implementation does:**
- **Matrix operation**: Standard \\(A \times B\\) multiplication (not \\(A \times B^T\\))
- **Loading pattern**: Both matrices use `Layout.row_major(1, TPB)` for coalesced access
- **Computation**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
- **Data layout**: No transposition during loading - both matrices loaded in same orientation

**What this implementation does NOT do:**
- Does NOT perform \\(A \times B^T\\) multiplication
- Does NOT use transposed loading patterns
- Does NOT transpose data during copy operations

With the \\((9 \times 9)\\) matrix size, we get perfect tiling that eliminates all boundary checks:

1. **LayoutTensor tile API**

   ```mojo
   out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
   a_tile = a.tile[TPB, TPB](block_idx.y, idx)
   b_tile = b.tile[TPB, TPB](idx, block_idx.x)
   ```

   This directly expresses "get the tile at position (block_idx.y, block_idx.x)" without manual coordinate calculation. See the [documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile) for more details.

2. **Asynchronous memory operations**

   ```mojo
   copy_dram_to_sram_async[
      thread_layout = load_a_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   ](a_shared,a_tile)
   copy_dram_to_sram_async[
      thread_layout = load_b_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   ](b_shared, b_tile)
   async_copy_wait_all()
   ```

   These operations:
   - Use dedicated copy engines that bypass registers and enable compute-memory overlap via [copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/)
   - Use specialized thread layouts for optimal memory access patterns
   - Eliminate the need for manual memory initialization
   - **Important**: 
      - Standard GPU loads are already asynchronous; these provide better resource utilization and register bypass
      - `copy_dram_to_sram_async` assumes that you are using a 1d thread block (`block_dim.y == block_dim.z == 1`) and all the threads from a thread block participate in the copy unless you specify otherwise.  This behaviour in overridden by specifying:
         - `block_dim_count`: the dimensionality of the thread block (`2` for the 2d thread block `THREADS_PER_BLOCK_TILED = (TPB, TPB)`)
         - `num_threads`: the number of threads in the thread block (`TPB*TPB == 9`)

3. **Optimized memory access layouts**

   ```mojo
   alias load_a_layout = Layout.row_major(1, TPB)    # Coalesced loading
   alias load_b_layout = Layout.row_major(1, TPB)    # Coalesced loading
   # Note: Both matrices use the same layout for standard A × B multiplication
   ```

   **Memory Access Analysis for Current Implementation:**

   Both matrices use `Layout.row_major(1, TPB)` for coalesced loading from global memory:
   - `load_a_layout`: Threads cooperate to load consecutive elements from matrix A rows
   - `load_b_layout`: Threads cooperate to load consecutive elements from matrix B rows
   - **Key insight**: Thread layout determines how threads cooperate during copy, not the final data layout

   **Actual Computation Pattern (proves this is A × B):**

   ```mojo
   # This is the actual computation in the current implementation
   acc += a_shared[local_row, k] * b_shared[k, local_col]

   # This corresponds to: C[i,j] = Σ(A[i,k] * B[k,j])
   # Which is standard matrix multiplication A × B
   ```

   **Why both matrices use the same coalesced loading pattern:**

   ```txt
   Loading tiles from global memory:
   - Matrix A tile: threads load A[block_row, k], A[block_row, k+1], A[block_row, k+2]... (consecutive)
   - Matrix B tile: threads load B[k, block_col], B[k, block_col+1], B[k, block_col+2]... (consecutive)

   Both patterns are coalesced with Layout.row_major(1, TPB)
   ```

   **Three separate memory concerns:**
   1. **Global-to-shared coalescing**: `Layout.row_major(1, TPB)` ensures coalesced global memory access
   2. **Shared memory computation**: `a_shared[local_row, k] * b_shared[k, local_col]` avoids bank conflicts
   3. **Matrix operation**: The computation pattern determines this is A × B, not A × B^T

4. **Perfect tiling eliminates boundary checks**

   ```mojo
   @parameter
   for idx in range(size // TPB):  # Perfect division: 9 // 3 = 3
   ```

   With \\((9 \times 9)\\) matrices and \\((3 \times 3)\\) tiles, every tile is exactly full-sized. No boundary checking needed!

5. **Clean tile processing with defensive bounds checking**

   ```mojo
   # Defensive bounds checking included even with perfect tiling
   if tiled_row < size and tiled_col < size:
       out_tile[local_row, local_col] = acc
   ```

   With perfect \\((9 \times 9)\\) tiling, this bounds check is technically unnecessary but included for defensive programming and consistency with other matrix sizes.

### Performance considerations

The idiomatic implementation maintains the performance benefits of tiling while providing cleaner abstractions:

1. **Memory locality**: Exploits spatial and temporal locality through tiling
2. **Coalesced access**: Specialized load layouts ensure coalesced memory access patterns
3. **Compute-memory overlap**: Potential overlap through asynchronous memory operations
4. **Shared memory efficiency**: No redundant initialization of shared memory
5. **Register pressure**: Uses accumulation registers for optimal compute throughput

This implementation shows how high-level abstractions can express complex GPU algorithms without sacrificing performance. It's a prime example of Mojo's philosophy: combining high-level expressiveness with low-level performance control.

### Key differences from manual tiling

| Feature | Manual Tiling | Idiomatic Tiling |
|---------|--------------|------------------|
| Memory access | Direct indexing with bounds checks | LayoutTensor tile API |
| Tile loading | Explicit element-by-element copying | Dedicated copy engine bulk transfers |
| Shared memory | Manual initialization (defensive) | Managed by copy functions |
| Code complexity | More verbose with explicit indexing | More concise with higher-level APIs |
| Bounds checking | Multiple checks during loading and computing | Single defensive check at final write |
| Matrix orientation | Both A and B in same orientation (standard A × B) | Both A and B in same orientation (standard A × B) |
| Performance | Explicit control over memory patterns | Optimized layouts with register bypass |

The idiomatic approach is not just cleaner but also potentially more performant due to the use of specialized memory layouts and asynchronous operations.

### Educational: When would transposed loading be useful?

The current implementation does NOT use transposed loading. This section is purely educational to show what's possible with the layout system.

**Current implementation recap:**
- Uses `Layout.row_major(1, TPB)` for both matrices
- Performs standard A × B multiplication
- No data transposition during copy

**Educational scenarios where you WOULD use transposed loading:**

While this puzzle uses standard coalesced loading for both matrices, the layout system's flexibility enables powerful optimizations in other scenarios:

```mojo
# Example: Loading pre-transposed matrix B^T to compute A × B
# (This is NOT what the current implementation does)
alias load_b_layout = Layout.row_major(TPB, 1)   # Load B^T with coalesced access
alias store_b_layout = Layout.row_major(1, TPB)  # Store as B in shared memory
copy_dram_to_sram_async[src_thread_layout=load_b_layout, dst_thread_layout=store_b_layout](b_shared, b_tile)
```

**Use cases for transposed loading (not used in this puzzle):**
1. **Pre-transposed input matrices**: When \\(B\\) is already stored transposed in global memory
2. **Different algorithms**: Computing \\(A^T \times B\\), \\(A \times B^T\\), or \\(A^T \times B^T\\)
3. **Memory layout conversion**: Converting between row-major and column-major layouts
4. **Avoiding transpose operations**: Loading data directly in the required orientation

**Key distinction:**
- **Current implementation**: Both matrices use `Layout.row_major(1, TPB)` for standard \\(A \times B\\) multiplication
- **Transposed loading example**: Would use different layouts to handle pre-transposed data or different matrix operations

This demonstrates Mojo's philosophy: providing low-level control when needed while maintaining high-level abstractions for common cases.

---

## Summary: Key takeaways

**What the idiomatic tiled implementation actually does:**
1. **Matrix Operation**: Standard A × B multiplication
2. **Memory Loading**: Both matrices use `Layout.row_major(1, TPB)` for coalesced access
3. **Computation Pattern**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
4. **Data Layout**: No transposition during loading

**Why this is optimal:**
- **Coalesced global memory access**: `Layout.row_major(1, TPB)` ensures efficient loading
- **Bank conflict avoidance**: Shared memory access pattern avoids conflicts
- **Standard algorithm**: Implements the most common matrix multiplication pattern

</div>
</details>
