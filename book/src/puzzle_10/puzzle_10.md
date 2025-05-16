# Puzzle 10: Dot Product

## Overview

Implement a kernel that computes the dot-product of vector `a` and vector `b` and stores it in `out` (single number).

**Note:** _You have 1 thread per position. You only need 2 global reads per thread and 1 global write per thread block._

![Dot product visualization](./media/videos/720p30/puzzle_10_viz.gif)

## Implementation approaches

### [🔰 Raw memory approach](./raw.md)
Learn how to implement the reduction with manual memory management and synchronization.

### [📐 LayoutTensor Version](./layout_tensor.md)
Use LayoutTensor's features for efficient reduction and shared memory management.

💡 **Note**: See how LayoutTensor simplifies efficient memory access patterns.
