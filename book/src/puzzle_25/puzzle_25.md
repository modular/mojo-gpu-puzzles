# Puzzle 25: Warp Communication

## Overview

**Puzzle 25: Warp Communication Primitives** introduces advanced GPU **warp-level communication operations** - hardware-accelerated primitives that enable efficient data exchange and coordination patterns within warps. You'll learn about using [shuffle_down](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down) and [broadcast](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast) to implement neighbor communication and collective coordination without complex shared memory patterns.

**Part VII: GPU Warp Communication** introduces warp-level data movement operations within thread groups. You'll learn to replace complex shared memory + indexing + boundary checking patterns with efficient warp communication calls that leverage hardware-optimized data movement.

**Key insight:** _GPU warps execute in lockstep - Mojo's warp communication operations use this synchronization to provide efficient data exchange primitives with automatic boundary handling and zero explicit synchronization._

## What you'll learn

### **Warp communication model**

Understand the fundamental communication patterns within GPU warps:

```
GPU Warp (32 threads, SIMT lockstep execution)
├── Lane 0  ──shuffle_down──> Lane 1  ──shuffle_down──> Lane 2
├── Lane 1  ──shuffle_down──> Lane 2  ──shuffle_down──> Lane 3
├── Lane 2  ──shuffle_down──> Lane 3  ──shuffle_down──> Lane 4
│   ...
└── Lane 31 ──shuffle_down──> undefined (boundary)

Broadcast pattern:
Lane 0 ──broadcast──> All lanes (0, 1, 2, ..., 31)
```

**Hardware reality:**

- **Register-to-register communication**: Data moves directly between thread registers
- **Zero memory overhead**: No shared memory allocation required
- **Automatic boundary handling**: Hardware manages warp edge cases
- **Single-cycle operations**: Communication happens in one instruction cycle

### **Warp communication operations in Mojo**

Learn the core communication primitives from `gpu.warp`:

1. **[`shuffle_down(value, offset)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down)**: Get value from lane at higher index (neighbor access)
2. **[`broadcast(value)`](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast)**: Share lane 0's value with all other lanes (one-to-many)
3. **[`shuffle_idx(value, lane)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_idx)**: Get value from specific lane (random access)
4. **[`shuffle_up(value, offset)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_up)**: Get value from lane at lower index (reverse neighbor)

> **Note:** This puzzle focuses on `shuffle_down()` and `broadcast()` as the most commonly used communication patterns. For complete coverage of all warp operations, see the [Mojo GPU Warp Documentation](https://docs.modular.com/mojo/stdlib/gpu/warp/).

### **Performance transformation example**

```mojo
# Complex neighbor access pattern (traditional approach):
shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
shared[local_i] = input[global_i]
barrier()
if local_i < WARP_SIZE - 1:
    next_value = shared[local_i + 1]  # Neighbor access
    result = next_value - shared[local_i]
else:
    result = 0  # Boundary handling
barrier()

# Warp communication eliminates all this complexity:
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # Direct neighbor access
if lane < WARP_SIZE - 1:
    result = next_val - current_val
else:
    result = 0
```

### **When warp communication excels**

Learn the performance characteristics:

| Communication Pattern | Traditional | Warp Operations |
|----------------------|-------------|-----------------|
| Neighbor access | Shared memory | Register-to-register |
| Stencil operations | Complex indexing | Simple shuffle patterns |
| Block coordination | Barriers + shared | Single broadcast |
| Boundary handling | Manual checks | Hardware automatic |

## Prerequisites

Before diving into warp communication, ensure you're comfortable with:

- **Part VII warp fundamentals**: Understanding SIMT execution and basic warp operations (see [Puzzle 24](../puzzle_24/puzzle_24.md))
- **GPU thread hierarchy**: Blocks, warps, and lane numbering
- **LayoutTensor operations**: Loading, storing, and tensor manipulation
- **Boundary condition handling**: Managing edge cases in parallel algorithms

## Learning path

### **1. Neighbor communication with shuffle_down**

**→ [Warp Shuffle Down](./warp_shuffle_down.md)**

Learn neighbor-based communication patterns for stencil operations and finite differences.

**What you'll learn:**

- Using `shuffle_down()` for accessing adjacent lane data
- Implementing finite differences and moving averages
- Handling warp boundaries automatically
- Multi-offset shuffling for extended neighbor access

**Key pattern:**

```mojo
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)
if lane < WARP_SIZE - 1:
    result = compute_with_neighbors(current_val, next_val)
```

### **2. Collective coordination with broadcast**

**→ [Warp Broadcast](./warp_broadcast.md)**

Learn one-to-many communication patterns for block-level coordination and collective decision-making.

**What you'll learn:**

- Using `broadcast()` for sharing computed values across lanes
- Implementing block-level statistics and collective decisions
- Combining broadcast with conditional logic
- Advanced broadcast-shuffle coordination patterns

**Key pattern:**

```mojo
var shared_value = 0.0
if lane == 0:
    shared_value = compute_block_statistic()
shared_value = broadcast(shared_value)
result = use_shared_value(shared_value, local_data)
```

## Key concepts

### **Communication patterns**

Understanding fundamental warp communication paradigms:

- **Neighbor communication**: Lane-to-adjacent-lane data exchange
- **Collective coordination**: One-lane-to-all-lanes information sharing
- **Stencil operations**: Accessing fixed patterns of neighboring data
- **Boundary handling**: Managing communication at warp edges

### **Hardware optimization**

Recognizing how warp communication maps to GPU hardware:

- **Register file communication**: Direct inter-thread register access
- **SIMT execution**: All lanes execute communication simultaneously
- **Zero latency**: Communication happens within the execution unit
- **Automatic synchronization**: No explicit barriers needed

### **Algorithm transformation**

Converting traditional parallel patterns to warp communication:

- **Array neighbor access** → `shuffle_down()`
- **Shared memory coordination** → `broadcast()`
- **Complex boundary logic** → Hardware-handled edge cases
- **Multi-stage synchronization** → Single communication operations

## Getting started

Start with neighbor-based shuffle operations to understand the foundation, then progress to collective broadcast patterns for advanced coordination.

💡 **Success tip**: Think of warp communication as **hardware-accelerated message passing** between threads in the same warp. This mental model will guide you toward efficient communication patterns that leverage the GPU's SIMT architecture.

**Learning objective**: By the end of Puzzle 25, you'll recognize when warp communication can replace complex shared memory patterns, enabling you to write simpler, faster neighbor-based and coordination algorithms.

**Begin with**: **[Warp Shuffle Down Operations](./warp_shuffle_down.md)** to learn neighbor communication, then advance to **[Warp Broadcast Operations](./warp_broadcast.md)** for collective coordination patterns.
