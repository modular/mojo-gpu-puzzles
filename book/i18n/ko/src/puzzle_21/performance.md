# Performance: Coalesced vs non-coalesced memory access

Understanding memory access patterns is crucial for GPU performance optimization. This section explains why coalesced memory access patterns typically outperform non-coalesced patterns, particularly for memory-bound operations like embedding lookups.

## Memory coalescing basics

**Memory coalescing** occurs when consecutive threads in a warp access consecutive memory addresses. GPUs can combine these individual memory requests into fewer, larger memory transactions, dramatically improving bandwidth utilization.

### Coalesced vs non-coalesced access

**Coalesced (efficient):**
```
- Thread 0 → Address 0x1000
- Thread 1 → Address 0x1004
- Thread 2 → Address 0x1008
- Thread 3 → Address 0x100C
- ...
```

**Result**: 1 memory transaction for entire warp (32 threads)

**Non-coalesced (inefficient):**
```
- Thread 0 → Address 0x1000
- Thread 1 → Address 0x2000
- Thread 2 → Address 0x3000
- Thread 3 → Address 0x4000
- ...
```

**Result**: Up to 32 separate memory transactions

## Why embedding operations are memory-bound

Embedding lookups are **memory-bound** because they involve:
- **Minimal computation**: Just copying data from input to output
- **Large memory footprint**: Embedding tables can be gigabytes in size
- **High memory bandwidth requirements**: Need to transfer large amounts of data

For such operations, **memory access efficiency** determines performance more than computational complexity.

## Kernel comparison

### 1D coalesced kernel
- **Thread organization**: `[total_elements // 256]` blocks, one thread per output element
- **Memory pattern**: Consecutive threads access consecutive embedding dimensions
- **Why it's coalesced**: `Thread 0: output[0,0,0]`, `Thread 1: output[0,0,1]` → consecutive addresses

### 2D non-coalesced kernel
- **Thread organization**: `[batch*seq // 16, embed_dim // 16]` blocks with 16×16 threads
- **Memory pattern**: Threads may access different embedding vectors
- **Why it's non-coalesced**: Thread access pattern can be scattered across memory

## Performance results

Typical benchmark results:
```
Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D
```

## Memory access visualization

### Coalesced pattern (1D kernel)

**Warp execution for output[0,0,0:32]:**

| Element | Thread ID | Memory Access | Address Pattern |
|---------|-----------|---------------|-----------------|
| `output[0,0,0]` | 0 | `[0,0]` | Base + 0 |
| `output[0,0,1]` | 1 | `[0,1]` | Base + 4 |
| `output[0,0,2]` | 2 | `[0,2]` | Base + 8 |
| `output[0,0,3]` | 3 | `[0,3]` | Base + 12 |
| ... | ... | ... | ... |
| `output[0,0,31]` | 31 | `[0,31]` | Base + 124 |

**Result**: Consecutive addresses → **1 memory transaction** for entire warp

### Non-coalesced pattern (2D kernel)

**Warp execution with 16×16 blocks:**

```
Block organization (16×16):
    X-dim: batch*seq positions (0-15)
    Y-dim: embed dimensions (0-15)

Warp threads might access:
    Thread 0:  batch=0, seq=0, embed=0  → Address A
    Thread 1:  batch=0, seq=1, embed=0  → Address B (different row)
    Thread 2:  batch=0, seq=2, embed=0  → Address C (different row)
    ...
    Thread 31: batch=1, seq=15, embed=0 → Address Z (scattered)
```

**Result**: Potentially scattered addresses → **Multiple memory transactions**

## Key optimization strategies

1. **Prefer 1D indexing** for memory-bound operations when possible
2. **Align data structures** to coalescing-friendly layouts
3. **Consider memory access patterns** during kernel design
4. **Profile memory bandwidth** to identify bottlenecks
5. **Use memory-bound benchmarks** to validate optimizations

The core insight: **memory access patterns** often determine GPU performance more than computational complexity, especially for memory-bound operations like embeddings.
