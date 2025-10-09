# When to Use Warp Programming

## Quick decision guide

**✅ Use warp operations when:**
- Reduction operations (`sum`, `max`, `min`) with 32+ elements
- Regular memory access patterns (adjacent lanes → adjacent addresses)
- Need cross-architecture portability (NVIDIA/RDNA 32 vs CDNA 64 threads)
- Want simpler, more maintainable code

**❌ Use traditional approaches when:**
- Complex cross-warp synchronization required
- Irregular/scattered memory access patterns
- Variable work per thread (causes warp divergence)
- Problem `size < WARP_SIZE`

## Performance characteristics

### Problem size scaling
| Elements | Warp Advantage | Notes |
|----------|---------------|-------|
| < 32 | None | Traditional better |
| 32-1K | 1.2-1.5× | Sweet spot begins |
| 1K-32K | 1.5-2.5× | **Warp operations excel** |
| > 32K | Memory-bound | Both approaches limited by bandwidth |

### Key warp advantages
- **No synchronization overhead**: Eliminates barrier costs
- **Minimal memory usage**: No shared memory allocation needed
- **Better scaling**: Performance improves with more warps
- **Simpler code**: Fewer lines, less error-prone

## Algorithm-specific guidance

| Algorithm | Recommendation | Reason |
|-----------|---------------|--------|
| **Dot product** | Warp ops (1K+ elements) | Single reduction, regular access |
| **Matrix row/col sum** | Warp ops | Natural reduction pattern |
| **Prefix sum** | Always warp `prefix_sum()` | Hardware-optimized primitive |
| **Pooling (max/min)** | Warp ops (regular windows) | Efficient window reductions |
| **Histogram with large number of bins** | Traditional | Irregular writes, atomic updates |

## Code examples

### ✅ Perfect for warps
```mojo
# Reduction operations
from gpu.warp import sum, max
var total = sum(partial_values)
var maximum = max(partial_values)

# Communication patterns
from gpu.warp import shuffle_idx, prefix_sum
var broadcast = shuffle_idx(my_value, 0)
var running_sum = prefix_sum(my_value)
```

### ❌ Better with traditional approaches
```mojo
# Complex multi-stage synchronization
stage1_compute()
barrier()  # Need ALL threads to finish
stage2_depends_on_stage1()

# Irregular memory access
var value = input[random_indices[global_i]]  # Scattered reads

# Data-dependent work
if input[global_i] > threshold:
    result = expensive_computation()  # Causes warp divergence
```

## Performance measurement

```bash
# Always benchmark both approaches
mojo p22.mojo --benchmark

# Look for scaling patterns:
# traditional_1x:  X.XX ms
# warp_1x:         Y.YY ms  # Should be faster
# warp_32x:        Z.ZZ ms  # Advantage should increase
```

## Summary

**Start with warp operations for:**
- Reductions with regular access patterns
- Problems ≥ 1 warp in size
- Cross-platform compatibility needs

**Use traditional approaches for:**
- Complex synchronization requirements
- Irregular memory patterns
- Small problems or heavy divergence

**When in doubt:** Implement both and benchmark. The performance difference will guide your decision.
