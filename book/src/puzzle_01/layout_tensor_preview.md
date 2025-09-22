## Why consider LayoutTensor?

Looking at our traditional implementation above, you might notice some potential issues:

### Current approach

```mojo
i = thread_idx.x
output[i] = a[i] + 10.0
```

This works for 1D arrays, but what happens when we need to:

- Handle 2D or 3D data?
- Deal with different memory layouts?
- Ensure coalesced memory access?

### Preview of future challenges

As we progress through the puzzles, array indexing will become more complex:

```mojo
# 2D indexing coming in later puzzles
idx = row * WIDTH + col

# 3D indexing
idx = (batch * HEIGHT + row) * WIDTH + col

# With padding
idx = (batch * padded_height + row) * padded_width + col
```

### LayoutTensor preview

[LayoutTensor](https://docs.modular.com/mojo/stdlib/layout/layout_tensor/LayoutTensor/) will help us handle these cases more elegantly:

```mojo
# Future preview - don't worry about this syntax yet!
output[i, j] = a[i, j] + 10.0  # 2D indexing
output[b, i, j] = a[b, i, j] + 10.0  # 3D indexing
```

We'll learn about LayoutTensor in detail in Puzzle 4, where these concepts become essential. For now, focus on understanding:

- Basic thread indexing
- Simple memory access patterns
- One-to-one mapping of threads to data

💡 **Key Takeaway**: While direct indexing works for simple cases, we'll soon need more sophisticated tools for complex GPU programming patterns.
