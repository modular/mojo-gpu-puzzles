# Embedding Kernels: Coaleasced vs non-Coaleasced

In this puzzle, you'll implement two different GPU kernels for embedding operations that produce identical results but use different memory access patterns, demonstrating the critical importance of memory coalescing in GPU performance.

## 1D coalesced kernel (optimized approach)

This kernel uses a simple 1D grid where each thread processes exactly one output element. The key insight is that consecutive threads will access consecutive memory locations, leading to optimal memory coalescing.

**Thread organization:**

- **Grid configuration**: `[total_elements // 256]` blocks, `256` threads per block
- **Thread mapping**: Each thread handles one `(batch, seq, embed)` position
- **Memory pattern**: Consecutive threads access consecutive embedding dimensions

**What you need to implement:**

1. Calculate the global thread index from block and thread indices
2. Convert the flat index to 3D coordinates `(batch_idx, seq_idx, embed_idx)`
3. Look up the token index from the indices tensor
4. Copy the appropriate embedding vector element to the output

### Code to complete

You need to complete the missing parts in both embedding kernels:

```mojo
{{#include ../../../problems/p21/op/embedding.mojo:embedding_kernel_coalesced}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p21/op/embedding.mojo" class="filename">View full file: problems/p21/op/embedding.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

- Start with `global_idx = block_idx.x * block_dim.x + thread_idx.x`
- Convert to 3D coordinates using division and modulo: `batch_idx = global_idx // (seq_len * embed_dim)`
- Use `remaining = global_idx % (seq_len * embed_dim)` to simplify further calculations
- Always check bounds: `if global_idx >= total_elements: return`
- Handle invalid token indices by setting output to 0
- The embedding lookup is: `output[batch_idx, seq_idx, embed_idx] = weights[token_idx, embed_idx]`

</div>
</details>

## 2D non-coalesced kernel (comparison approach)

This kernel uses a 2D grid where the X dimension spans `(batch × seq)` positions and the Y dimension spans embedding dimensions. This can lead to non-coalesced memory access patterns.

**Thread organization:**

- **Grid configuration**: `[batch x seq // 16, embed_dim // 16]` blocks, `16 x 16` threads per block
- **Thread mapping**: `thread_idx.x` maps to batch/sequence, `thread_idx.y` maps to embedding dimension
- **Memory pattern**: Threads in a warp may access scattered memory locations

**What you need to implement:**

1. Calculate both X and Y coordinates from the 2D grid
2. Convert the X coordinate to separate batch and sequence indices
3. Use the Y coordinate directly as the embedding dimension
4. Perform the same embedding lookup with bounds checking

### Code to complete

You need to complete the missing parts in both embedding kernels:

```mojo
{{#include ../../../problems/p21/op/embedding.mojo:embedding_kernel_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p21/op/embedding.mojo" class="filename">View full file: problems/p21/op/embedding.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

- Use both X and Y thread coordinates: `batch_seq_idx = block_idx.x * block_dim.x + thread_idx.x`
- And: `embed_idx = block_idx.y * block_dim.y + thread_idx.y`
- Convert `batch_seq_idx` to separate batch and sequence indices: `batch_idx = batch_seq_idx // seq_len`
- Remember to check bounds for both dimensions: `if batch_seq_idx >= total_positions or embed_idx >= embed_dim`
- The token lookup is the same as 1D, but you're only handling one embedding dimension per thread
- This kernel processes one embedding dimension per thread instead of entire vectors

</div>
</details>

## Custom ops registration

The kernels are wrapped in PyTorch custom operations for easy integration. The registration pattern is the same as MAX custom ops explained in [Understanding MAX Graph custom ops](../puzzle_17/puzzle_17.md#understanding-max-graph-custom-ops):

### 1D coalesced operation

This operation registers the optimized 1D embedding kernel as `"embedding"`:

```mojo
{{#include ../../../solutions/p21/op/embedding.mojo:embedding_custom_op_solution}}
```

**Key aspects of this registration:**

- **Simple grid configuration**: Uses a straightforward 1D grid with `ceildiv(total_elements, THREADS_PER_BLOCK)` blocks
- **Memory optimization**: Single `enqueue_memset` call to zero the output buffer efficiently
- **Compile-time parameters**: All tensor dimensions passed as compile-time parameters for optimal performance
- **Device abstraction**: Handles both GPU execution and CPU fallback seamlessly

### 2D non-coalesced operation

This operation registers the comparison 2D embedding kernel as `"embedding_2d"`:

```mojo
{{#include ../../../solutions/p21/op/embedding.mojo:embedding_2d_custom_op_solution}}
```

**Key differences from the 1D operation:**

- **Complex grid configuration**: Uses a 2D grid with separate calculations for `blocks_x` and `blocks_y`
- **Fixed block dimensions**: Hard-coded `BLOCK_X = 16` and `BLOCK_Y = 16` for 2D thread organization
- **Same memory management**: Identical memory initialization and CPU fallback logic
- **Different kernel call**: Passes 2D grid dimensions `(blocks_x, blocks_y)` and block dimensions `(BLOCK_X, BLOCK_Y)`

### Common wrapper functionality

Both custom operations provide essential infrastructure:

1. **Memory management**:
   - Zero-initialization of output tensors with `enqueue_memset`
   - Proper buffer creation and memory layout handling
   - Automatic cleanup and resource management

2. **Device abstraction**:
   - GPU execution with optimized kernels
   - CPU fallback for compatibility and debugging
   - Consistent interface regardless of execution target

3. **Parameter passing**:
   - Compile-time tensor dimensions for kernel optimization
   - Runtime tensor data through layout tensor conversion
   - Type-safe parameter validation

4. **Grid configuration**:
   - Automatic calculation of optimal grid dimensions
   - Different strategies optimized for each kernel's access pattern
   - Proper block dimension management

### Integration with PyTorch

These registered operations can be called from Python using the [CustomOpLibrary](https://docs.modular.com/max/api/python/torch/CustomOpLibrary/):

```python
# Load the custom operations
ops = CustomOpLibrary(mojo_kernels)

# Call the 1D coalesced version
result_1d = ops.embedding[{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}](
    indices, weights
)

# Call the 2D non-coalesced version
result_2d = ops.embedding_2d[{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}](
    indices, weights
)
```

The power of this approach is that the same kernel implementations can be used across different Python frameworks while maintaining optimal performance characteristics.

## Run the code

You can run the puzzle with:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p21
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p21
```

  </div>
  <div class="tab-content">

```bash
uv run poe p21
```

  </div>
</div>

When successful, you should see output similar to:

```
Puzzle 21: Mojo Embedding Kernel Comparison
======================================================================
Configuration: B=8, L=512, V=10000, E=512
------------------------------------------------------------

Testing Correctness...
   1D Coalesced - Max difference: 1.19e-07
   2D Non-coalesced - Max difference: 1.19e-07
   ✅ Both implementations CORRECT

Benchmarking Mojo Kernels...

Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D

Key Learning Points:
• Compare different GPU kernel implementations
• 1D vs 2D grid patterns have different memory access
• Coalesced memory access should be faster
• Grid configuration affects GPU utilization
```

## Solution

<details class="solution-details">
<summary></summary>

The solution involves implementing the coordinate transformations and memory operations for both kernels:

## 1D Coalesced Kernel

```mojo
{{#include ../../../solutions/p21/op/embedding.mojo:embedding_kernel_coalesced_solution}}
```

## 2D Non-Coalesced Kernel

```mojo
{{#include ../../../solutions/p21/op/embedding.mojo:embedding_kernel_2d_solution}}
```

<div class="solution-explanation">

Both solutions implement the same embedding lookup logic but with different thread organizations:

### Key differences

1. **Thread mapping**:
   - **1D kernel**: One thread per output element, simple flat indexing
   - **2D kernel**: 2D grid mapping to (batch×seq, embed_dim) coordinates

2. **Memory access patterns**:
   - **1D kernel**: Consecutive threads access consecutive embedding dimensions → coalesced
   - **2D kernel**: Thread access pattern depends on block configuration → potentially non-coalesced

3. **Indexing complexity**:
   - **1D kernel**: Single division/modulo chain to get 3D coordinates
   - **2D kernel**: Separate X/Y coordinate calculations

### Performance implications

The 1D kernel typically performs better because:

- **Memory coalescing**: Consecutive threads access consecutive memory addresses
- **Simple indexing**: Lower computational overhead for coordinate calculations
- **Better cache utilization**: Predictable memory access patterns

The 2D kernel may perform worse due to:

- **Scattered memory accesses**: Threads within a warp may access different embedding vectors
- **Complex grid configuration**: 16×16 blocks may not align optimally with memory layout
- **Warp divergence**: Different threads may follow different execution paths

</div>

</details>

## Key concepts

| Concept | 1D Coalesced | 2D Non-coalesced |
|---------|---------------|-------------------|
| **Thread organization** | 1D flat indexing | 2D grid (batch×seq, embed) |
| **Memory access** | Consecutive addresses | Potentially scattered |
| **Grid configuration** | Simple: `[total_elements // 256]` | Complex: `[batch×seq // 16, embed // 16]` |
| **Performance** | Optimized for memory bandwidth | Suboptimal memory pattern |
| **Use case** | Production kernels | Educational comparison |

The core lesson: **memory coalescing** can lead to 2-3x performance differences for memory-bound operations like embeddings.
