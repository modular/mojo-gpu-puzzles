# Puzzle 27: Block-Level Programming

## Overview

Welcome to **Puzzle 27: Block-Level Programming**! This puzzle introduces you to the fundamental building blocks of GPU parallel programming - **block-level communication primitives** that enable sophisticated parallel algorithms across entire thread blocks. You'll explore three essential communication patterns that replace complex manual synchronization with elegant, hardware-optimized operations.

**What you'll achieve:** Transform from complex shared memory + barriers + tree reduction patterns (Puzzle 12) to elegant single-function-call algorithms that leverage hardware-optimized block-wide communication primitives across multiple warps.

**Key insight:** _GPU thread blocks execute with sophisticated hardware coordination - Mojo's block operations harness cross-warp communication and dedicated hardware units to provide complete parallel programming building blocks: reduction (all→one), scan (all→each), and broadcast (one→all)._

## What you'll learn

### **Block-level communication model**
Understand the three fundamental communication patterns within GPU thread blocks:

```
GPU Thread Block (128 threads across 4 or 2 warps, hardware coordination)
All-to-One (Reduction):     All threads → Single result at thread 0
All-to-Each (Scan):         All threads → Each gets cumulative position
One-to-All (Broadcast):     Thread 0 → All threads get same value

Cross-warp coordination:
├── Warp 0 (threads 0-31)   ──block.sum()──┐
├── Warp 1 (threads 32-63)  ──block.sum()──┼→ Thread 0 result
├── Warp 2 (threads 64-95)  ──block.sum()──┤
└── Warp 3 (threads 96-127) ──block.sum()──┘
```

**Hardware reality:**
- **Cross-warp synchronization**: Automatic coordination across multiple warps within a block
- **Dedicated hardware units**: Specialized scan units and butterfly reduction networks
- **Zero explicit barriers**: Hardware manages all synchronization internally
- **Logarithmic complexity**: \\(O(\\log n)\\) algorithms with single-instruction simplicity

### **Block operations in Mojo**
Learn the complete parallel programming toolkit from `gpu.primitives.block`:

1. **[`block.sum(value)`](https://docs.modular.com/mojo/std/gpu/primitives/block/sum)**: All-to-one reduction for totals, averages, maximum/minimum values
2. **[`block.prefix_sum(value)`](https://docs.modular.com/mojo/std/gpu/primitives/block/prefix_sum)**: All-to-each scan for parallel filtering and extraction
3. **[`block.broadcast(value)`](https://docs.modular.com/mojo/std/gpu/primitives/block/broadcast)**: One-to-all distribution for parameter sharing and coordination

> **Note:** These primitives enable sophisticated parallel algorithms like statistical computations, histogram binning, and normalization workflows that would otherwise require dozens of lines of complex shared memory coordination code.

### **Performance transformation example**
```mojo
# Complex block-wide reduction (traditional approach - from Puzzle 12):
shared_memory[local_i] = my_value
barrier()
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
if local_i == 0:
    output[block_idx.x] = shared_memory[0]

# Block operations eliminate all this complexity:
my_partial = compute_local_contribution()
total = block.sum[block_size=128, broadcast=False](my_partial)  # Single call!
if local_i == 0:
    output[block_idx.x] = total[0]
```

### **When block operations excel**
Learn the performance characteristics:

| Algorithm Pattern | Traditional | Block Operations |
|-------------------|-------------|------------------|
| Block-wide reductions | Shared memory + barriers | Single `block.sum` call |
| Parallel filtering | Complex indexing | `block.prefix_sum` coordination |
| Parameter sharing | Manual synchronization | Single `block.broadcast` call |
| Cross-warp algorithms | Explicit barrier management | Hardware-managed coordination |

## The evolution of GPU programming patterns

### **Where we started: Manual coordination (Puzzle 12)**
Complex but educational - explicit shared memory, barriers, and tree reduction:
```mojo
# Manual approach: 15+ lines of complex synchronization
shared_memory[local_i] = my_value
barrier()
# Tree reduction with stride-based indexing...
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
```

### **The intermediate step: Warp programming (Puzzle 24)**
Hardware-accelerated but limited scope - `warp.sum()` within 32-thread warps:
```mojo
# Warp approach: 1 line but single warp only
total = warp.sum[warp_size=WARP_SIZE](val=partial_product)
```

### **The destination: Block programming (This puzzle)**
Complete toolkit - hardware-optimized primitives across entire blocks:
```mojo
# Block approach: 1 line across multiple warps (128+ threads)
total = block.sum[block_size=128, broadcast=False](val=partial_product)
```

## The three fundamental communication patterns

Block-level programming provides three essential primitives that cover all parallel communication needs:

### **1. All-to-One: Reduction (`block.sum()`)**
- **Pattern**: All threads contribute → One thread receives result
- **Use case**: Computing totals, averages, finding maximum/minimum values
- **Example**: Dot product, statistical aggregation
- **Hardware**: Cross-warp butterfly reduction with automatic barriers

### **2. All-to-Each: Scan (`block.prefix_sum()`)**
- **Pattern**: All threads contribute → Each thread receives cumulative position
- **Use case**: Parallel filtering, stream compaction, histogram binning
- **Example**: Computing write positions for parallel data extraction
- **Hardware**: Parallel scan with cross-warp coordination

### **3. One-to-All: Broadcast (`block.broadcast()`)**
- **Pattern**: One thread provides → All threads receive same value
- **Use case**: Parameter sharing, configuration distribution
- **Example**: Sharing computed mean for normalization algorithms
- **Hardware**: Optimized distribution across multiple warps


## Learning progression

Complete this puzzle in three parts, building from simple to sophisticated:

### **Part 1: [Block.sum() Essentials](./block_sum.md)**
**Transform complex reduction to simple function call**

Learn the foundational block reduction pattern by implementing dot product with `block.sum()`. This part shows how block operations replace 15+ lines of manual barriers with a single optimized call.

**Key concepts:**
- Block-wide synchronization across multiple warps
- Hardware-optimized reduction patterns
- Thread 0 result management
- Performance comparison with traditional approaches

**Expected outcome:** Understand how `block.sum()` provides warp.sum() simplicity at block scale.

---

### **Part 2: [Block.prefix_sum() Parallel Histogram](./block_prefix_sum.md)**
**Advanced parallel filtering and extraction**

Build sophisticated parallel algorithms using `block.prefix_sum()` for histogram binning. This part demonstrates how prefix sum enables complex data reorganization that would be difficult with simple reductions.

**Key concepts:**
- Parallel filtering with binary predicates
- Coordinated write position computation
- Advanced partitioning algorithms
- Cross-thread data extraction patterns

**Expected outcome:** Understand how `block.prefix_sum()` enables sophisticated parallel algorithms beyond simple aggregation.

---

### **Part 3: [Block.broadcast() Vector Normalization](./block_broadcast.md)**
**Complete workflow combining all patterns**

Implement vector mean normalization using the complete block operations toolkit. This part shows how all three primitives work together to solve real computational problems with mathematical correctness.

**Key concepts:**
- One-to-all communication patterns
- Coordinated multi-phase algorithms
- Complete block operations workflow
- Real-world algorithm implementation

**Expected outcome:** Understand how to compose block operations for sophisticated parallel algorithms.

## Why block operations matter

### **Code simplicity transformation:**
```
Traditional approach:  20+ lines of barriers, shared memory, complex indexing
Block operations:      3-5 lines of composable, hardware-optimized primitives
```

### **Performance advantages:**
- **Hardware optimization**: Leverages GPU architecture-specific optimizations
- **Automatic synchronization**: Eliminates manual barrier placement errors
- **Composability**: Operations work together seamlessly
- **Portability**: Same code works across different GPU architectures

### **Educational value:**
- **Conceptual clarity**: Each operation has a clear communication purpose
- **Progressive complexity**: Build from simple reductions to complex algorithms
- **Real applications**: Patterns used extensively in scientific computing, graphics, AI

## Prerequisites

Before starting this puzzle, you should have completed:
- **[Puzzle 12](../puzzle_12/puzzle_12.md)**: Understanding of manual GPU synchronization
- **[Puzzle 24](../puzzle_24/puzzle_24.md)**: Experience with warp-level programming

## Expected learning outcomes

After completing all three parts, you'll understand:

1. **When to use each block operation** for different parallel communication needs
2. **How to compose operations** to build sophisticated algorithms
3. **Performance trade-offs** between manual and automated approaches
4. **Real-world applications** of block-level programming patterns
5. **Architecture-independent programming** using hardware-optimized primitives

## Getting started

**Recommended approach:** Complete the three parts in sequence, as each builds on concepts from the previous parts. The progression from simple reduction → advanced partitioning → complete workflow provides the optimal learning path for understanding block-level GPU programming.

💡 **Key insight**: Block operations represent the sweet spot between programmer productivity and hardware performance - they provide the simplicity of high-level operations with the efficiency of carefully optimized low-level implementations. This puzzle teaches you to think at the right abstraction level for modern GPU programming.
