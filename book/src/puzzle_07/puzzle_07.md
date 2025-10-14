# Puzzle 7: 2D Blocks

## Overview

Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`.

**Note:** _You have fewer threads per block than the size of `a` in both directions._

<img src="./media/07.png" alt="Blocks 2D visualization" class="light-mode-img">
<img src="./media/07d.png" alt="Blocks 2D visualization" class="dark-mode-img">

## Key concepts

- Block-based processing
- Grid-block coordination
- Multi-block indexing
- Memory access patterns

> 🔑 **2D thread indexing convention**
>
> We extend the block-based indexing from [puzzle 4](../puzzle_04/puzzle_04.md) to 2D:
>
> ```txt
> Global position calculation:
> row = block_dim.y * block_idx.y + thread_idx.y
> col = block_dim.x * block_idx.x + thread_idx.x
> ```
>
> For example, with 2×2 blocks in a 4×4 grid:
> ```txt
> Block (0,0):   Block (1,0):
> [0,0  0,1]     [0,2  0,3]
> [1,0  1,1]     [1,2  1,3]
>
> Block (0,1):   Block (1,1):
> [2,0  2,1]     [2,2  2,3]
> [3,0  3,1]     [3,2  3,3]
> ```
>
> Each position shows (row, col) for that thread's global index.
> The block dimensions and indices work together to ensure:
> - Continuous coverage of the 2D space
> - No overlap between blocks
> - Efficient memory access patterns

## Implementation approaches

### [🔰 Raw memory approach](./raw.md)
Learn how to handle multi-block operations with manual indexing.

### [📐 LayoutTensor Version](./layout_tensor.md)
Use LayoutTensor features to elegantly handle block-based processing.

💡 **Note**: See how LayoutTensor simplifies block coordination and memory access patterns.
