# Puzzle 26: Advanced Warp Patterns

## Overview

Welcome to **Puzzle 26: Advanced Warp Communication Primitives**! This puzzle introduces you to sophisticated GPU **warp-level butterfly communication and parallel scan operations** - hardware-accelerated primitives that enable efficient tree-based algorithms and parallel reductions within warps. You'll learn about using [shuffle_xor](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor) for butterfly networks and [prefix_sum](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum) for hardware-optimized parallel scan without complex multi-phase shared memory algorithms.

**What you'll achieve:** Transform from complex shared memory + barrier + multi-phase reduction patterns to elegant single-function-call algorithms that leverage hardware-optimized butterfly networks and parallel scan units.

**Key insight:** _GPU warps can perform sophisticated tree-based communication and parallel scan operations in hardware - Mojo's advanced warp primitives harness butterfly networks and dedicated scan units to provide \\(O(\\log n)\\) algorithms with single-instruction simplicity._

## What you'll learn

### **Advanced warp communication model**

Understand sophisticated communication patterns within GPU warps:

```
GPU Warp Butterfly Network (32 threads, XOR-based communication)
Offset 16: Lane 0 ↔ Lane 16, Lane 1 ↔ Lane 17, ..., Lane 15 ↔ Lane 31
Offset 8:  Lane 0 ↔ Lane 8,  Lane 1 ↔ Lane 9,  ..., Lane 23 ↔ Lane 31
Offset 4:  Lane 0 ↔ Lane 4,  Lane 1 ↔ Lane 5,  ..., Lane 27 ↔ Lane 31
Offset 2:  Lane 0 ↔ Lane 2,  Lane 1 ↔ Lane 3,  ..., Lane 29 ↔ Lane 31
Offset 1:  Lane 0 ↔ Lane 1,  Lane 2 ↔ Lane 3,  ..., Lane 30 ↔ Lane 31

Hardware Prefix Sum (parallel scan acceleration)
Input:  [1, 2, 3, 4, 5, 6, 7, 8, ...]
Output: [1, 3, 6, 10, 15, 21, 28, 36, ...] (inclusive scan)
```

**Hardware reality:**

- **Butterfly networks**: XOR-based communication creates optimal tree topologies
- **Dedicated scan units**: Hardware-accelerated parallel prefix operations
- **Logarithmic complexity**: \\(O(\\log n)\\) algorithms replace \\(O(n)\\) sequential patterns
- **Single-cycle operations**: Complex reductions happen in specialized hardware

### **Advanced warp operations in Mojo**

Learn the sophisticated communication primitives from `gpu.warp`:

1. **[`shuffle_xor(value, mask)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor)**: XOR-based butterfly communication for tree algorithms
2. **[`prefix_sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum)**: Hardware-accelerated parallel scan operations
3. **Advanced coordination patterns**: Combining multiple primitives for complex algorithms

> **Note:** These primitives enable sophisticated parallel algorithms like parallel reductions, stream compaction, quicksort partitioning, and FFT operations that would otherwise require dozens of lines of shared memory coordination code.

### **Performance transformation example**

```mojo
# Complex parallel reduction (traditional approach - from Puzzle 14):
shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
shared[local_i] = input[global_i]
barrier()
offset = 1
for i in range(Int(log2(Scalar[dtype](WARP_SIZE)))):
    var current_val: output.element_type = 0
    if local_i >= offset and local_i < WARP_SIZE:
        current_val = shared[local_i - offset]
    barrier()
    if local_i >= offset and local_i < WARP_SIZE:
        shared[local_i] += current_val
    barrier()
    offset *= 2

# Advanced warp primitives eliminate all this complexity:
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)  # Single call!
output[global_i] = scan_result
```

### **When advanced warp operations excel**

Learn the performance characteristics:

| Algorithm Pattern | Traditional | Advanced Warp Operations |
|------------------|-------------|-------------------------|
| Parallel reductions | Shared memory + barriers | Single `shuffle_xor` tree |
| Prefix/scan operations | Multi-phase algorithms | Hardware `prefix_sum` |
| Stream compaction | Complex indexing | `prefix_sum` + coordination |
| Quicksort partition | Manual position calculation | Combined primitives |
| Tree algorithms | Recursive shared memory | Butterfly communication |

## Prerequisites

Before diving into advanced warp communication, ensure you're comfortable with:

- **Part VII warp fundamentals**: Understanding SIMT execution and basic warp operations (see [Puzzle 24](../puzzle_24/puzzle_24.md) and [Puzzle 25](../puzzle_25/puzzle_25.md))
- **Parallel algorithm theory**: Tree reductions, parallel scan, and butterfly networks
- **GPU memory hierarchy**: Shared memory patterns and synchronization (see [Puzzle 14](../puzzle_14/puzzle_14.md))
- **Mathematical operations**: Understanding XOR operations and logarithmic complexity

## Learning path

### **1. Butterfly communication with shuffle_xor**

**→ [Warp Shuffle XOR](./warp_shuffle_xor.md)**

Learn XOR-based butterfly communication patterns for efficient tree algorithms and parallel reductions.

**What you'll learn:**

- Using `shuffle_xor()` for creating butterfly network topologies
- Implementing \\(O(\\log n)\\) parallel reductions with tree communication
- Understanding XOR-based lane pairing and communication patterns
- Advanced conditional butterfly operations for multi-value reductions

**Key pattern:**

```mojo
max_val = input[global_i]
offset = WARP_SIZE // 2
while offset > 0:
    max_val = max(max_val, shuffle_xor(max_val, offset))
    offset //= 2
# All lanes now have global maximum
```

### **2. Hardware-accelerated parallel scan with prefix_sum**

**→ [Warp Prefix Sum](./warp_prefix_sum.md)**

Learn hardware-optimized parallel scan operations that replace complex multi-phase algorithms with single function calls.

**What you'll learn:**

- Using `prefix_sum()` for hardware-accelerated cumulative operations
- Implementing stream compaction and parallel partitioning
- Combining `prefix_sum` with `shuffle_xor` for advanced coordination
- Understanding inclusive vs exclusive scan patterns

**Key pattern:**

```mojo
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)
output[global_i] = scan_result  # Hardware-optimized cumulative sum
```

## Key concepts

### **Butterfly network communication**

Understanding XOR-based communication topologies:

- **XOR pairing**: `lane_id ⊕ mask` creates symmetric communication pairs
- **Tree reduction**: Logarithmic complexity through hierarchical data exchange
- **Parallel coordination**: All lanes participate simultaneously in reduction
- **Dynamic algorithms**: Works for any power-of-2 `WARP_SIZE` (32, 64, etc.)

### **Hardware-accelerated parallel scan**

Recognizing dedicated scan unit capabilities:

- **Prefix sum operations**: Cumulative operations with hardware acceleration
- **Stream compaction**: Parallel filtering and data reorganization
- **Single-function simplicity**: Complex algorithms become single calls
- **Zero synchronization**: Hardware handles all coordination internally

### **Algorithm complexity transformation**

Converting traditional patterns to advanced warp operations:

- **Sequential reductions** (\\(O(n)\\)) → **Butterfly reductions** (\\(O(\\log n)\\))
- **Multi-phase scan algorithms** → **Single hardware prefix_sum**
- **Complex shared memory patterns** → **Register-only operations**
- **Explicit synchronization** → **Hardware-managed coordination**

### **Advanced coordination patterns**

Combining multiple primitives for sophisticated algorithms:

- **Dual reductions**: Simultaneous min/max tracking with butterfly patterns
- **Parallel partitioning**: `shuffle_xor` + `prefix_sum` for quicksort-style operations
- **Conditional operations**: Lane-based output selection with global coordination
- **Multi-primitive algorithms**: Complex parallel patterns with optimal performance

## Getting started

Ready to harness advanced GPU warp-level communication? Start with butterfly network operations to understand tree-based communication, then progress to hardware-accelerated parallel scan for optimal algorithm performance.

💡 **Success tip**: Think of advanced warp operations as **hardware-accelerated parallel algorithm building blocks**. These primitives replace entire categories of complex shared memory algorithms with single, optimized function calls.

**Learning objective**: By the end of Puzzle 24, you'll recognize when advanced warp primitives can replace complex multi-phase algorithms, enabling you to write dramatically simpler and faster tree-based reductions, parallel scans, and coordination patterns.

**Ready to begin?** Start with **[Warp Shuffle XOR Operations](./warp_shuffle_xor.md)** to learn butterfly communication, then advance to **[Warp Prefix Sum Operations](./warp_prefix_sum.md)** for hardware-accelerated parallel scan patterns!
