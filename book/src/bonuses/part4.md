# Bonus Challenges

## Challenge I: advanced softmax implementations

*This challenge extends [Puzzle 18: Softmax Op](../puzzle_18/puzzle_18.md)*

Here are some advanced challenges to extend your softmax implementation:

### 1. Large-scale softmax: Handling `TPB < SIZE`

When the input size exceeds the number of threads per block (`TPB < SIZE`), our current implementation fails because a single block cannot process the entire array. Two approaches to solve this:

#### 1.1 Buffer reduction

- Store block-level results (max and sum) in device memory
- Use a second kernel to perform reduction across these intermediate results
- Implement a final normalization pass that uses the global max and sum

#### 1.2 Two-pass softmax

- First pass: Each block calculates its local max value
- Synchronize and compute global max
- Second pass: Calculate \\(e^{x-max}\\) and local sum
- Synchronize and compute global sum
- Final pass: Normalize using global sum

### 2. Batched softmax

Implement softmax for a batch of vectors (2D input tensor) with these variants:

- Row-wise softmax: Apply softmax independently to each row
- Column-wise softmax: Apply softmax independently to each column
- Compare performance differences between these implementations

## Challenge II: advanced attention mechanisms

*This challenge extends [Puzzle 19: Attention Op](../puzzle_19/puzzle_19.md)*

Building on the vector attention implementation, here are advanced challenges that push the boundaries of attention mechanisms:

### 1. Larger sequence lengths

Extend the attention mechanism to handle longer sequences using the existing kernels:

#### 1.1 Sequence length scaling

- Modify the attention implementation to handle `SEQ_LEN = 32` and `SEQ_LEN = 64`
- Update the `TPB` (threads per block) parameter accordingly
- Ensure the transpose kernel handles the larger matrix sizes correctly

#### 1.2 Dynamic sequence lengths

- Implement attention that can handle variable sequence lengths at runtime
- Add bounds checking in the kernels to handle sequences shorter than `SEQ_LEN`
- Compare performance with fixed vs. dynamic sequence length handling

### 2. Batched vector attention

Extend to process multiple attention computations simultaneously:

#### 2.1 Batch processing

- Modify the attention operation to handle multiple query vectors at once
- Input shapes: Q(batch_size, d), K(seq_len, d), V(seq_len, d)
- Output shape: (batch_size, d)
- Reuse the existing kernels with proper indexing

#### 2.2 Memory optimization for batches

- Minimize memory allocations by reusing buffers across batch elements
- Compare performance with different batch sizes (2, 4, 8)
- Analyze memory usage patterns
