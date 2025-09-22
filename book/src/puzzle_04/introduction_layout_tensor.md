# Introduction to LayoutTensor

Let's take a quick break from solving puzzles to preview a powerful abstraction that will make our GPU programming journey more enjoyable:
🥁 ... the **[LayoutTensor](https://docs.modular.com/mojo/stdlib/layout/layout_tensor/LayoutTensor/)**.

> 💡 _This is a motivational overview of LayoutTensor's capabilities. Don't worry about understanding everything now - we'll explore each feature in depth as we progress through the puzzles_.

## The challenge: Growing complexity

Let's look at the challenges we've faced so far:

```mojo
# Puzzle 1: Simple indexing
output[i] = a[i] + 10.0

# Puzzle 2: Multiple array management
output[i] = a[i] + b[i]

# Puzzle 3: Bounds checking
if i < size:
    output[i] = a[i] + 10.0
```

As dimensions grow, code becomes more complex:

```mojo
# Traditional 2D indexing for row-major 2D matrix
idx = row * WIDTH + col
if row < height and col < width:
    output[idx] = a[idx] + 10.0
```

## The solution: A peek at LayoutTensor

LayoutTensor will help us tackle these challenges with elegant solutions. Here's a glimpse of what's coming:

1. **Natural Indexing**: Use `tensor[i, j]` instead of manual offset calculations
3. **Flexible Memory Layouts**: Support for row-major, column-major, and tiled organizations
4. **Performance Optimization**: Efficient memory access patterns for GPU

## A taste of what's ahead

Let's look at a few examples of what LayoutTensor can do. Don't worry about understanding all the details now - we'll cover each feature thoroughly in upcoming puzzles.

### Basic usage example

```mojo
from layout import Layout, LayoutTensor

# Define layout
alias HEIGHT = 2
alias WIDTH = 3
alias layout = Layout.row_major(HEIGHT, WIDTH)

# Create tensor
tensor = LayoutTensor[dtype, layout](buffer.unsafe_ptr())

# Access elements naturally
tensor[0, 0] = 1.0  # First element
tensor[1, 2] = 2.0  # Last element
```

### Preview of advanced features

As we progress through the puzzles, you'll learn about:

- Shared memory optimizations
- Efficient tiling strategies
- Vectorized operations
- Hardware acceleration
- Optimized memory access patterns

```mojo
# Column-major layout
layout_col = Layout.col_major(HEIGHT, WIDTH)

# Tiled layout (for better cache utilization)
layout_tiled = tensor.tiled[4, 4](HEIGHT, WIDTH)
```

Each layout has its advantages:

- **Row-major**: Elements in a row are contiguous

  ```mojo
  # [1 2 3]
  # [4 5 6] -> [1 2 3 4 5 6]
  layout_row = Layout.row_major(2, 3)
  ```

- **Column-major**: Elements in a column are contiguous

  ```mojo
  # [1 2 3]
  # [4 5 6] -> [1 4 2 5 3 6]
  layout_col = Layout.col_major(2, 3)
  ```

- **Tiled**: Elements grouped in tiles for cache efficiency

  ```mojo
  # [[1 2] [3 4]] in 2x2 tiles
  layout_tiled = Layout.tiled[2, 2](4, 4)
  ```

### Advanced GPU optimizations

As you progress, you'll discover LayoutTensor's powerful features for GPU programming:

1. **Memory hierarchy management**

```mojo
# Shared memory allocation
shared_mem = tb[dtype]().row_major[BM, BK]().shared().alloc()

# Register allocation
reg_tile = tb[dtype]().row_major[TM, TN]().local().alloc()
```

2. **Tiling strategies**

```mojo
# Block tiling
block_tile = tensor.tile[BM, BN](block_idx.y, block_idx.x)

# Register tiling
reg_tile = block_tile.tile[TM, TN](thread_row, thread_col)
```

3. **Memory access patterns**

```mojo
# Vectorized access
vec_tensor = tensor.vectorize[1, simd_width]()

# Asynchronous transfers
copy_dram_to_sram_async[thread_layout=layout](dst, src)
```

4. **Hardware acceleration**

```mojo
# Tensor Core operations (coming in later puzzles)
mma_op = TensorCore[dtype, out_type, Index(M, N, K)]()
result = mma_op.mma_op(a_reg, b_reg, c_reg)
```

💡 **Looking ahead**: Through these puzzles, you'll learn to:

- Optimize data access with shared memory
- Implement efficient tiling strategies
- Leverage vectorized operations
- Utilize hardware accelerators
- Learn memory access patterns

Each concept builds on the last, gradually taking you from basic tensor operations to advanced GPU programming. Ready to begin? Let's start with the fundamentals!

## Quick example

Let's put everything together with a simple example that demonstrates the basics of LayoutTensor:

```mojo
{{#include ./intro.mojo}}
```

When we run this code with:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
pixi run layout_tensor_intro
```

  </div>
</div>

```txt
Before:
0.0 0.0 0.0
0.0 0.0 0.0
After:
1.0 0.0 0.0
0.0 0.0 0.0
```

Let's break down what's happening:

1. We create a `2 x 3` tensor with row-major layout
2. Initially, all elements are zero
3. Using natural indexing, we modify a single element
4. The change is reflected in our output

This simple example demonstrates key LayoutTensor benefits:

- Clean syntax for tensor creation and access
- Automatic memory layout handling
- Natural multi-dimensional indexing

While this example is straightforward, the same patterns will scale to complex GPU operations in upcoming puzzles. You'll see how these basic concepts extend to:

- Multi-threaded GPU operations
- Shared memory optimizations
- Complex tiling strategies
- Hardware-accelerated computations

Ready to start your GPU programming journey with LayoutTensor? Let's dive into the puzzles!

💡 **Tip**: Keep this example in mind as we progress - we'll build upon these fundamental concepts to create increasingly sophisticated GPU programs.
