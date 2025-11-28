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
comptime HEIGHT = 2
comptime WIDTH = 3
comptime layout = Layout.row_major(HEIGHT, WIDTH)

# Create tensor
tensor = LayoutTensor[dtype, layout](buffer.unsafe_ptr())

# Access elements naturally
tensor[0, 0] = 1.0  # First element
tensor[1, 2] = 2.0  # Last element
```

To learn more about `Layout` and `LayoutTensor`, see these guides from the [Mojo manual](https://docs.modular.com/mojo/manual/)

- [Introduction to layouts](https://docs.modular.com/mojo/manual/layout/layouts)
- [Using LayoutTensor](https://docs.modular.com/mojo/manual/layout/tensors)

## Quick example

Let's put everything together with a simple example that demonstrates the basics of LayoutTensor:

```mojo
{{#include ./intro.mojo}}
```

When we run this code with:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
uv run poe layout_tensor_intro
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
