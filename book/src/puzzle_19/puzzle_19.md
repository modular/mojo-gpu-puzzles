# Puzzle 19: Attention Op

## Overview

In this puzzle, we'll implement the attention mechanism as a custom MAX Graph operation. Attention is a fundamental building block of modern neural networks, popularized particularly by [transformers](https://arxiv.org/abs/1706.03762), that allows models to focus on relevant parts of the input when making predictions.

Mathematically, the attention function is defined as:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

Where:

- \\(Q\\) is the **query vector** of shape \\((d,)~\\) - represents what we're looking for
- \\(K\\) is the **key matrix** of shape \\((\text{seq\_len}, d)~\\) - represents what's available to match against
- \\(V\\) is the **value matrix** of shape \\((\text{seq\_len}, d)~\\) - represents the information to retrieve
- The output is a **weighted combination** vector of shape \\((d,)\\)

The computation involves three main steps:

1. **Attention Scores**: Compute \\(Q \cdot K^T\\) to measure how well the query matches each key vector
2. **Attention Weights**: Apply softmax to convert scores into a probability distribution (weights sum to 1)
3. **Weighted Sum**: Combine value vectors using attention weights to produce the final output

## Understanding attention: a step-by-step breakdown

Think of attention as a **smart lookup mechanism**. Given a query (what you're looking for), attention finds the most relevant information from a collection of key-value pairs:

1. **Step 1 - Similarity Matching**: Compare your query \\(Q\\) against all keys \\(K\\) to get similarity scores
   - Compute \\(Q \cdot K^T\\) where each score measures how well \\(Q\\) matches each key vector
   - Higher scores = better matches

2. **Step 2 - Probability Distribution**: Convert raw scores into normalized weights
   - Apply softmax to ensure all weights sum to 1.0
   - This creates a probability distribution over which values to focus on

3. **Step 3 - Weighted Retrieval**: Combine values using the attention weights
   - Multiply each value vector by its corresponding weight
   - Sum everything up to get the final output

**Real-world analogy**: Imagine searching a library. Your query is what you want to find, the book titles are keys, and the book contents are values. Attention computes how relevant each book is to your query, then gives you a summary weighted by relevance.

### Visual computation flow

```
Input:  Q(16,)    K(16,16)    V(16,16)
         ↓           ↓           ↓
Step 1: Q(1,16) @ K^T(16,16) → Scores(1,16)
         ↓
Step 2: softmax(Scores) → Weights(1,16)  [sum = 1.0]
         ↓
Step 3: Weights(1,16) @ V(16,16) → Output(1,16) → reshape → Output(16,)
```

**Key insight**: We reshape the query vector \\(Q\\) from shape \\((16,)\\) to \\((1,16)\\) so we can use matrix multiplication instead of manual dot products. This allows us to leverage the highly optimized tiled matmul kernel from Puzzle 18!

Our GPU implementation **reuses and combines optimized kernels from previous puzzles**:

- **[Tiled matrix multiplication from Puzzle 16](../puzzle_16/puzzle_16.md)** for efficient \\(Q \cdot K^T\\) and \\(\text{weights} \cdot V\\) operations
- **Shared memory transpose** for computing \\(K^T\\) efficiently
- **[Parallel softmax from Puzzle 18](../puzzle_18/puzzle_18.md)** for numerically stable attention weight computation

> **🔄 Kernel Reuse Strategy**: This puzzle demonstrates how to build complex operations by combining proven, optimized kernels from previous puzzles. Rather than writing everything from scratch, we leverage the `matmul_idiomatic_tiled` from Puzzle 16 and `softmax_kernel` from Puzzle 18, showcasing the power of modular GPU kernel design.

## Key concepts

- Vector attention mechanism for sequence processing
- **Kernel reuse**: Leveraging proven implementations from [Puzzle 16](../puzzle_16/puzzle_16.md) and [Puzzle 18](../puzzle_18/puzzle_18.md)
- Efficient matrix multiplication using shared memory tiling
- Memory-optimized tensor reshaping to minimize buffer allocation
- Integration of multiple optimized kernels into a single operation
- Custom MAX Graph operation with multi-input support
- CPU fallback implementation for compatibility

## Configuration

- **Sequence length**: \\(\text{SEQ\_LEN} = 16~\\) - number of key/value vectors in our sequence
- **Model dimension**: \\(\text{D} = 16~\\) - dimensionality of each vector (query, keys, values)
- **Threads per block**: Individually optimized for each kernel
- **Grid dimensions**: Computed dynamically to handle different matrix sizes efficiently
- **Shared memory**: Utilized in transpose, matmul, and softmax kernels for performance

Layout configuration:

- Query tensor: `Layout.row_major(d)`
- Key tensor: `Layout.row_major(seq_len, d)`
- Value tensor: `Layout.row_major(seq_len, d)`
- Output tensor: `Layout.row_major(d)`
- Custom op parameters: `{"seq_len": seq_len, "d": d, "dtype": dtype}`

Key aspects of this puzzle include:

1. **Multi-kernel orchestration**: Combining transpose, matmul, and softmax operations
2. **Memory optimization**: Using reshape operations and buffer reuse to minimize allocations
3. **Numerical stability**: Leveraging the proven softmax implementation from [Puzzle 18](../puzzle_18/puzzle_18.md)
4. **Performance optimization**: Using tiled algorithms from [Puzzle 16](../puzzle_16/puzzle_16.md) for all matrix operations
5. **Multi-input operations**: Handling three input tensors (Q, K, V) in a single custom op

Our attention custom operation will:

- Accept query, key, and value tensors from Python
- Process them efficiently on GPU using optimized kernels
- Return the attention-weighted output vector
- Match the results of NumPy reference implementation

## Code to complete

To complete this puzzle, we'll leverage the tiled matmul kernel from [Puzzle 16](../puzzle_16/puzzle_16.md) and the softmax kernel from [Puzzle 18](../puzzle_18/puzzle_18.md). You only need to implement the transpose kernel in the Mojo file using shared memory.

### 1. Implement the transpose kernel

```mojo
{{#include ../../../problems/p19/op/attention.mojo:transpose_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p19/op/attention.mojo" class="filename">View full file: problems/p19/op/attention.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

**Transpose Kernel Implementation Guide:**

1. **Shared Memory Setup**: Use `LayoutTensor[dtype, Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()` to create a square `TRANSPOSE_BLOCK_DIM_XY` × `TRANSPOSE_BLOCK_DIM_XY` shared memory tile for efficient data exchange between threads

2. **Thread Indexing**: Map threads to matrix elements:
   - `local_row = thread_idx.y`, `local_col = thread_idx.x` (position within the block)
   - `global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row` (position in the full matrix)

3. **Two-Phase Operation**:
   - **Phase 1**: Load data from global memory into shared memory with normal indexing
   - **Phase 2**: Store data from shared memory to global memory with swapped indexing

4. **Critical Synchronization**: Call `barrier()` between loading and storing to ensure all threads have finished loading before any thread starts storing

5. **Transpose Magic**: The transpose happens through swapped indexing: `shared_tile[local_col, local_row]` instead of `shared_tile[local_row, local_col]`

6. **Boundary Handling**: Check bounds when accessing global memory to avoid out-of-bounds reads/writes for matrices that don't perfectly divide by `TRANSPOSE_BLOCK_DIM_XY` x `TRANSPOSE_BLOCK_DIM_XY`

7. **Memory Coalescing**: This pattern ensures both reads and writes are coalesced for optimal memory bandwidth utilization

</div>
</details>

### 2. Orchestrate the attention

```mojo
{{#include ../../../problems/p19/op/attention.mojo:attention_orchestration}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p19/op/attention.mojo" class="filename">View full file: problems/p19/op/attention.mojo</a>

### Test the kernels

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p19
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p19
```

  </div>
  <div class="tab-content">

```bash
uv run poe p19
```

  </div>
</div>

When successful, you should see output similar to on CPU and GPU:

```
Input shapes: Q=(16,), K=(16, 16), V=(16, 16)
Sample Q values: [ 0.04967142 -0.01382643  0.06476886  0.15230298 -0.02341534]
Sample K[0] values: [-0.10128311  0.03142473 -0.09080241 -0.14123037  0.14656489]
Sample V[0] values: [ 0.11631638  0.00102331 -0.09815087  0.04621035  0.01990597]

================================================================================
STEP-BY-STEP VECTOR ATTENTION COMPUTATION DEBUG
================================================================================

1. INPUT SHAPES:
   Q shape: (16,) (query vector)
   K shape: (16, 16) (key matrix)
   V shape: (16, 16) (value matrix)
   Q[:5]: [ 0.04967142 -0.01382643  0.06476886  0.15230298 -0.02341534]

2. ATTENTION SCORES (K[i] · Q):
   Scores shape: (16,)
   Scores[:5]: [-0.03479404 -0.01563787  0.04834607  0.06764711  0.04001468]
   Min: -0.061636, Max: 0.067647
   Manual verification:
     Q · K[0] = K[0] · Q = -0.034794 (computed: -0.034794)
     Q · K[1] = K[1] · Q = -0.015638 (computed: -0.015638)
     Q · K[2] = K[2] · Q = 0.048346 (computed: 0.048346)

3. SOFTMAX:
   Max score: 0.067647
   Attention weights shape: (16,)
   Attention weights[:5]: [0.05981331 0.06097015 0.06499878 0.0662655  0.06445949]
   Sum: 1.000000 (should be 1.0)

4. WEIGHTED SUM OF VALUES:
   Output shape: (16,)
   Output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
   Output norm: 0.092764
   Manual output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
   Match: True

================================================================================
TESTING INDIVIDUAL OPERATIONS
================================================================================

Test 1: Vector Dot Product
a · b = 3.000000

Test 2: Matrix-Vector Multiplication
M @ v = [ 3.  7. 11.]

Test 3: Softmax
Input: [1. 2. 3. 4.]
Softmax: [0.0320586  0.08714432 0.2368828  0.6439143 ]
Sum: 1.000000

================================================================================
TESTING FULL ATTENTION
================================================================================
Compiling attention graph on Device(type=cpu,id=0)
Executing attention on Device(type=cpu,id=0)
====================================================================================================

CPU attention output[:5]: [-0.00935538 -0.02434331  0.00306551  0.02346884  0.019306  ]
CPU matches NumPy: True
Compiling attention graph on Device(type=gpu,id=0)
Executing attention on Device(type=gpu,id=0)
====================================================================================================

GPU attention output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
Expected output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
GPU matches NumPy: True

================================================================================
FINAL VERIFICATION
================================================================================
✓ CPU implementation PASSED
✓ GPU implementation PASSED

Output vector norms:
  CPU: 0.092764
  GPU: 0.092764
  Expected: 0.092764
```

This indicates that your custom MAX Graph operation correctly implements the attention algorithm and produces results matching the NumPy reference implementation.

## Solution

<details class="solution-details">
<summary></summary>

To solve this puzzle, we need to implement the transpose kernel in Mojo and complete the Python graph definition for our attention custom operation. This puzzle builds upon concepts from previous puzzles, combining **tiled matrix multiplication from [Puzzle 16](../puzzle_16/puzzle_16.md)** and **softmax from [Puzzle 18](../puzzle_18/puzzle_18.md)** into a complete attention mechanism.

### Reused kernels

Our implementation directly incorporates these proven kernels:

1. **`matmul_idiomatic_tiled`** from [Puzzle 16](../puzzle_16/puzzle_16.md) - Powers both \\(Q \\times K^T\\) and \\(\\text{weights} \\times V\\) operations
2. **`softmax_kernel`** from [Puzzle 18](../puzzle_18/puzzle_18.md) - Provides numerically stable attention weight computation

This exemplifies **modular GPU architecture**: complex neural network operations built by orchestrating proven, optimized components rather than monolithic implementations.

The attention operation follows the canonical mathematical definition:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

**Breaking down the math**:

- \\(Q \cdot K^T~\\): Query-key similarity scores of shape: \\((1, \text{seq\_len})\\)
- \\(\text{softmax}(\cdot)~\\): Normalize scores to probabilities of shape: \\((1, \text{seq\_len})\\)
- \\(\text{weights} \cdot V~\\): Weighted combination of values of shape: \\((1, d)\\)

This involves several computational steps that we optimize using GPU kernels from previous puzzles.

### 1. Transpose kernel implementation

```mojo
{{#include ../../../solutions/p19/op/attention.mojo:transpose_kernel_solution}}
```

<div class="solution-explanation">

The transpose kernel uses **shared memory tiling** to achieve coalesced memory access patterns. Key implementation details:

#### Critical transpose pattern

```mojo
# Load with normal indexing
shared_tile[local_row, local_col] = inp[global_row, global_col]
barrier()
# Store with swapped indexing for transpose
output[out_row, out_col] = shared_tile[local_col, local_row]
```

The transpose happens through **swapped indexing** in shared memory access (`[local_col, local_row]` instead of `[local_row, local_col]`) and **swapped block coordinates** for output positioning. This ensures both reads and writes remain coalesced while achieving the transpose operation.
</div>

### 2. GPU kernel orchestration

```mojo
{{#include ../../../solutions/p19/op/attention.mojo:attention_orchestration_solution}}
```

<div class="solution-explanation">

The GPU orchestration demonstrates **sophisticated kernel chaining** and **zero-copy memory optimization**:

#### Advanced memory optimization strategies

```mojo
# Zero-copy reshaping - no data movement, just reinterpret tensor shape
q_2d = q_tensor.reshape[layout_q_2d]()
# Aggressive buffer reuse - same memory, different interpretations
weights = scores_2d.reshape[layout_scores]()
```

The implementation achieves **maximum memory efficiency** through:

- **Zero-copy reshaping**: Reinterpreting tensor shapes without moving data in memory
- **Intelligent buffer reuse**: The same `scores_weights_buf` serves dual purposes as both scores \\((1,\\text{seq_len})\\) and weights \\((\\text{seq_len},)\\)
- **Minimal allocations**: Only 2 temporary buffers power the entire attention operation
- **Memory coalescing**: All operations maintain optimal memory access patterns

#### Strategic kernel reuse pattern

- **Steps 3 & 7**: Both leverage `matmul_idiomatic_tiled` from [Puzzle 16](../puzzle_16/puzzle_16.md)
  - Step 3: \\(Q \\times K^T\\) → attention scores computation \\((1,d) \\times (d,\\text{seq_len}) \\rightarrow (1,\\text{seq_len})\\)
  - Step 7: \\(\\text{weights} \\times V\\) → final weighted output \\((1,\\text{seq_len}) \\times (\\text{seq_len},d) \\rightarrow (1,d)\\)
  - Both operations include bounds checking for robustness with variable matrix dimensions
- **Step 5**: Employs `softmax_kernel` from [Puzzle 18](../puzzle_18/puzzle_18.md)
  - Converts raw scores into normalized probability distribution
  - Ensures numerical stability through max subtraction and parallel reduction
  - Guarantees \\(\\sum_{i} \\text{weights}[i] = 1.0\\)

This exemplifies **modular GPU architecture**: complex neural network operations built by orchestrating proven, optimized kernels rather than monolithic implementations!
</div>

### Key implementation insights

<div class="solution-explanation">

#### Memory optimization strategy

The implementation achieves **minimal memory allocation** through aggressive buffer reuse:

```mojo
# Only 2 temporary buffers needed for the entire operation
k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](seq_len * d)
scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](seq_len)
```

**Key optimization insights**:

- The same `scores_weights_buf` is reused for both attention scores and weights through reshape operations
- Zero-copy tensor reshaping eliminates unnecessary data movement

#### Kernel reuse architecture

This puzzle showcases **modular kernel design** by combining three specialized kernels:

- **`matmul_idiomatic_tiled`** (used twice) - Powers both \\(Q \\times K^T\\) and \\(\\text{weights} \\times V\\) operations
- **`softmax_kernel`** - Provides numerically stable attention weight computation with parallel reduction
- **`transpose_kernel`** - Enables efficient \\(K^T\\) computation with coalesced memory access

**Architectural benefits**:

- **Composability**: Complex operations built from proven components
- **Maintainability**: Each kernel has a single, well-defined responsibility
- **Performance**: Leverages highly optimized implementations from previous puzzles
- **Scalability**: Modular design enables easy extension to larger attention mechanisms

The implementation demonstrates that **sophisticated neural network operations** can be built by orchestrating simpler, well-tested GPU kernels rather than writing monolithic implementations.
</div>

</details>
