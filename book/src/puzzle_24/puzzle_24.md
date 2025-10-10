# Puzzle 24: Warp Fundamentals

## Overview

**Part VI: GPU Warp Programming** introduces GPU **warp-level primitives** - hardware-accelerated operations that leverage synchronized thread execution within warps. You'll learn to use built-in warp operations to replace complex shared memory patterns with simple, efficient function calls.

**Goal:** Replace complex shared memory + barrier + tree reduction patterns with efficient warp primitive calls that leverage hardware synchronization.

**Key insight:** _GPU warps execute in lockstep - Mojo's warp operations use this synchronization to provide powerful parallel primitives with zero explicit synchronization._

## What you'll learn

### **GPU warp execution model**

Understand the fundamental hardware unit of GPU parallelism:

```
GPU Block (e.g., 256 threads)
├── Warp 0 (32 threads, SIMT lockstep execution)
│   ├── Lane 0  ─┐
│   ├── Lane 1   │ All execute same instruction
│   ├── Lane 2   │ at same time (SIMT)
│   │   ...      │
│   └── Lane 31 ─┘
├── Warp 1 (32 threads, independent)
├── Warp 2 (32 threads, independent)
└── ...
```

**Hardware reality:**

- **32 threads per warp** on NVIDIA GPUs (`WARP_SIZE=32`)
- **32 or 64 threads per warp** on AMD GPUs (`WARP_SIZE=32 or 64`)
- **Lockstep execution**: All threads in a warp execute the same instruction simultaneously
- **Zero synchronization cost**: Warp operations happen instantly within each warp

### **Warp operations available in Mojo**

Learn the core warp primitives from `gpu.warp`:

1. **`sum(value)`**: Sum all values across warp lanes
2. **`shuffle_idx(value, lane)`**: Get value from specific lane
3. **`shuffle_down(value, delta)`**: Get value from lane+delta
4. **`prefix_sum(value)`**: Compute prefix sum across lanes
5. **`lane_id()`**: Get current thread's lane number (0-31 or 0-63)

### **Performance transformation example**

```mojo
# 1. Reduction through shared memory
# Complex pattern we have seen earlier (from p12.mojo):
shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
shared[local_i] = partial_product
barrier()

# Safe tree reduction through shared memory requires a barrier after each reduction phase:
stride = WARP_SIZE // 2
while stride > 0:
    if local_i < stride:
        shared[local_i] += shared[local_i + stride]

    barrier()
    stride //= 2

# 2. Reduction using warp primatives
# Safe tree reduction using warp primatives does not require shared memory or a barrier
# after each reduction phase.
# Mojo's warp-level sum operation uses warp primatives under the hood and hides all this
# complexity:
total = sum(partial_product)  # Internally no barriers, no race conditions!
```

### **When warp operations excel**

Learn the performance characteristics:

```
Problem Scale         Traditional    Warp Operations
Single warp (32)      Fast          Fastest (no barriers)
Few warps (128)       Good          Excellent (minimal overhead)
Many warps (1024+)    Good          Outstanding (scales linearly)
Massive (16K+)        Bottlenecked  Memory-bandwidth limited
```

## Prerequisites

Before diving into warp programming, ensure you're comfortable with:

- **Part V functional patterns**: Elementwise, tiled, and vectorized approaches
- **GPU thread hierarchy**: Understanding blocks, warps, and threads
- **LayoutTensor operations**: Loading, storing, and tensor manipulation
- **Shared memory concepts**: Why barriers and tree reduction are complex

## Learning path

### **1. SIMT execution model**

**→ [Warp Lanes & SIMT Execution](./warp_simt.md)**

Understand the hardware foundation that makes warp operations possible.

**What you'll learn:**

- Single Instruction, Multiple Thread (SIMT) execution model
- Warp divergence and convergence patterns
- Lane synchronization within warps
- Hardware vs software thread management

**Key insight:** Warps are the fundamental unit of GPU execution - understanding SIMT unlocks warp programming.

### **2. Warp sum fundamentals**

**→ [warp.sum() Essentials](./warp_sum.md)**

Learn the most important warp operation through dot product implementation.

**What you'll learn:**

- Replacing shared memory + barriers with `sum()`
- Cross-GPU architecture compatibility (`WARP_SIZE`)
- Kernel vs functional programming patterns with warps
- Performance comparison with traditional approaches

**Key pattern:**

```mojo
partial_result = compute_per_lane_value()
total = sum(partial_result)  # Magic happens here!
if lane_id() == 0:
    output[0] = total
```

### **3. When to use warp programming**

**→ [When to Use Warp Programming](./warp_extra.md)**

Learn the decision framework for choosing warp operations over alternatives.

**What you'll learn:**

- Problem characteristics that favor warp operations
- Performance scaling patterns with warp count
- Memory bandwidth vs computation trade-offs
- Warp operation selection guidelines

**Decision framework:** When reduction operations become the bottleneck, warp primitives often provide the breakthrough.

## Key concepts to learn

### **Hardware-software alignment**

Understanding how Mojo's warp operations map to GPU hardware:

- **SIMT execution**: All lanes execute same instruction simultaneously
- **Built-in synchronization**: No explicit barriers needed within warps
- **Cross-architecture support**: `WARP_SIZE` handles NVIDIA vs AMD differences

### **Pattern transformation**

Converting complex parallel patterns to warp primitives:

- **Tree reduction** → `sum()`
- **Prefix computation** → `prefix_sum()`
- **Data shuffling** → `shuffle_idx()`, `shuffle_down()`

### **Performance characteristics**

Recognizing when warp operations provide advantages:

- **Small to medium problems**: Eliminates barrier overhead
- **Large problems**: Reduces memory traffic and improves cache utilization
- **Regular patterns**: Warp operations excel with predictable access patterns

## Getting started

Start with understanding the SIMT execution model, then dive into practical warp sum implementation, and finish with the strategic decision framework.

💡 **Success tip**: Think of warps as **synchronized vector units** rather than independent threads. This mental model will guide you toward effective warp programming patterns.

**Learning objective**: By the end of Part VI, you'll recognize when warp operations can replace complex synchronization patterns, enabling you to write simpler, faster GPU code.

**Ready to begin?** Start with **[SIMT Execution Model](./warp_simt.md)** and discover the power of warp-level programming!
