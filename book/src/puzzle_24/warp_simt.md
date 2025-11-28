# 🧠 Warp lanes & SIMT execution

## Mental model for warp programming vs SIMD

### What is a warp?

A **warp** is a group of 32 (or 64) GPU threads that execute **the same instruction at the same time** on different data. Think of it as a **synchronized vector unit** where each thread acts like a "lane" in a vector processor.

**Simple example:**

```mojo
from gpu.warp import sum
# All 32 threads in the warp execute this simultaneously:
var my_value = input[my_thread_id]     # Each gets different data
var warp_total = sum(my_value)         # All contribute to one sum
```

What just happened? Instead of 32 separate threads doing complex coordination, the **warp** automatically synchronized them to produce a single result. This is **SIMT (Single Instruction, Multiple Thread)** execution.

### SIMT vs SIMD comparison

If you're familiar with CPU vector programming (SIMD), GPU warps are similar but with key differences:

| Aspect | CPU SIMD (e.g., AVX) | GPU Warp (SIMT) |
|--------|---------------------|------------------|
| **Programming model** | Explicit vector operations | Thread-based programming |
| **Data width** | Fixed (256/512 bits) | Flexible (32/64 threads) |
| **Synchronization** | Implicit within instruction | Implicit within warp |
| **Communication** | Via memory/registers | Via shuffle operations |
| **Divergence handling** | Not applicable | Hardware masking |
| **Example** | `a + b` | `sum(thread_value)` |

**CPU SIMD approach (C++ intrinsics):**

```cpp
// Explicit vector operations - say 8 floats in parallel
__m256 result = _mm256_add_ps(a, b);   // Add 8 pairs simultaneously
```

**CPU SIMD approach (Mojo):**

```mojo
# SIMD in Mojo is first class citizen type so if a, b are of type SIMD then
# addition is performed in parallel
var result = a + b # Add 8 pairs simultaneously
```

**GPU SIMT approach (Mojo):**

```mojo
# Thread-based code that becomes vector operations
from gpu.warp import sum

var my_data = input[thread_id]         # Each thread gets its element
var partial = my_data * coefficient    # All threads compute simultaneously
var total = sum(partial)               # Hardware coordinates the sum
```

### Core concepts that make warps powerful

**1. Lane identity:** Each thread has a "lane ID" (0 to 31) that's essentially free to access

```mojo
var my_lane = lane_id()  # Just reading a hardware register
```

**2. Implicit synchronization:** No barriers needed within a warp

```mojo
# This just works - all threads automatically synchronized
var sum = sum(my_contribution)
```

**3. Efficient communication:** Threads can share data without memory

```mojo
# Get value from lane 0 to all other lanes
var broadcasted = shuffle_idx(my_value, 0)
```

**Key insight:** SIMT lets you write natural thread code that executes as efficient vector operations, combining the ease of thread programming with the performance of vector processing.

### Where warps fit in GPU execution hierarchy

For complete context on how warps relate to the overall GPU execution model, see [GPU Threading vs SIMD](../puzzle_23/gpu-thread-vs-simd.md). Here's where warps fit:

```
GPU Device
├── Grid (your entire problem)
│   ├── Block 1 (group of threads, shared memory)
│   │   ├── Warp 1 (32 threads, lockstep execution) ← This level
│   │   │   ├── Thread 1 → SIMD operations
│   │   │   ├── Thread 2 → SIMD operations
│   │   │   └── ... (32 threads total)
│   │   └── Warp 2 (32 threads)
│   └── Block 2 (independent group)
```

**Warp programming operates at the "Warp level"** - you work with operations that coordinate all 32 threads within a single warp, enabling powerful primitives like `sum()` that would otherwise require complex shared memory coordination.

This mental model supports recognizing when problems map naturally to warp operations versus requiring traditional shared memory approaches.

## The hardware foundation of warp programming

Understanding **Single Instruction, Multiple Thread (SIMT)** execution is crucial for effective warp programming. This isn't just a software abstraction - it's how GPU hardware actually works at the silicon level.

## What is SIMT execution?

**SIMT** means that within a warp, all threads execute the **same instruction** at the **same time** on **different data**. This is fundamentally different from CPU threads, which can execute completely different instructions independently.

### CPU vs GPU Execution Models

| Aspect | CPU (MIMD) | GPU Warp (SIMT) |
|--------|------------|------------------|
| **Instruction Model** | Multiple Instructions, Multiple Data | Single Instruction, Multiple Thread |
| **Core 1** | `add r1, r2` | `add r1, r2` |
| **Core 2** | `load r3, [mem]` | `add r1, r2` (same instruction) |
| **Core 3** | `branch loop` | `add r1, r2` (same instruction) |
| **... Core 32** | `different instruction` | `add r1, r2` (same instruction) |
| **Execution** | Independent, asynchronous | Synchronized, lockstep |
| **Scheduling** | Complex, OS-managed | Simple, hardware-managed |
| **Data** | Independent data sets | Different data, same operation |

**GPU Warp Execution Pattern:**

- **Instruction**: Same for all 32 lanes: `add r1, r2`
- **Lane 0**: Operates on `Data0` → `Result0`
- **Lane 1**: Operates on `Data1` → `Result1`
- **Lane 2**: Operates on `Data2` → `Result2`
- **... (all lanes execute simultaneously)**
- **Lane 31**: Operates on `Data31` → `Result31`

**Key insight:** All lanes execute the **same instruction** at the **same time** on **different data**.

### Why SIMT works for GPUs

GPUs are optimized for **throughput**, not latency. SIMT enables:

- **Hardware simplification**: One instruction decoder serves 32 or 64 threads
- **Execution efficiency**: No complex scheduling between warp threads
- **Memory bandwidth**: Coalesced memory access patterns
- **Power efficiency**: Shared control logic across lanes

## Warp execution mechanics

### Lane numbering and identity

Each thread within a warp has a **lane ID** from 0 to `WARP_SIZE-1`:

```mojo
from gpu import lane_id
from gpu.warp import WARP_SIZE

# Within a kernel function:
my_lane = lane_id()  # Returns 0-31 (NVIDIA/RDNA) or 0-63 (CDNA)
```

**Key insight:** `lane_id()` is **free** - it's just reading a hardware register, not computing a value.

### Synchronization within warps

The most powerful aspect of SIMT: **implicit synchronization**.

```mojo
# Example with thread_idx.x < WARP_SIZE

# 1. Traditional shared memory approach:
shared[thread_idx.x] = partial_result
barrier()  # Explicit synchronization required
var total = shared[0] + shared[1] + ... + shared[WARP_SIZE] # Sum reduction

# 2. Warp approach:
from gpu.warp import sum

var total = sum(partial_result)  # Implicit synchronization!
```

**Why no barriers needed?** All lanes execute each instruction at exactly the same time. When `sum()` starts, all lanes have already computed their `partial_result`.

## Warp divergence and convergence

### What happens with conditional code?

```mojo
if lane_id() % 2 == 0:
    # Even lanes execute this path
    result = compute_even()
else:
    # Odd lanes execute this path
    result = compute_odd()
# All lanes converge here
```

**Hardware behaviour steps:**

| Step | Phase | Active Lanes | Waiting Lanes | Efficiency | Performance Cost |
|------|-------|--------------|---------------|------------|------------------|
| **1** | Condition evaluation | All 32 lanes | None | 100% | Normal speed |
| **2** | Even lanes branch | Lanes 0,2,4...30 (16 lanes) | Lanes 1,3,5...31 (16 lanes) | 50% | **2× slower** |
| **3** | Odd lanes branch | Lanes 1,3,5...31 (16 lanes) | Lanes 0,2,4...30 (16 lanes) | 50% | **2× slower** |
| **4** | Convergence | All 32 lanes | None | 100% | Normal speed resumed |

**Example breakdown:**

- **Step 2**: Only even lanes execute `compute_even()` while odd lanes wait
- **Step 3**: Only odd lanes execute `compute_odd()` while even lanes wait
- **Total time**: `time(compute_even) + time(compute_odd)` (sequential execution)
- **Without divergence**: `max(time(compute_even), time(compute_odd))` (parallel execution)

**Performance impact:**

1. **Divergence**: Warp splits execution - some lanes active, others wait
2. **Serial execution**: Different paths run sequentially, not in parallel
3. **Convergence**: All lanes reunite and continue together
4. **Cost**: Divergent warps take 2× time (or more) vs unified execution

### Best practices for warp efficiency

### Warp efficiency patterns

**✅ EXCELLENT: Uniform execution (100% efficiency)**

```mojo
# All lanes do the same work - no divergence
var partial = a[global_i] * b[global_i]
var total = sum(partial)
```

*Performance: All 32 lanes active simultaneously*

**⚠️ ACCEPTABLE: Predictable divergence (~95% efficiency)**

```mojo
# Divergence based on lane_id() - hardware optimized
if lane_id() == 0:
    output[block_idx] = sum(partial)
```

*Performance: Brief single-lane operation, predictable pattern*

**🔶 CAUTION: Structured divergence (~50-75% efficiency)**

```mojo
# Regular patterns can be optimized by compiler
if (global_i / 4) % 2 == 0:
    result = method_a()
else:
    result = method_b()
```

*Performance: Predictable groups, some optimization possible*

**❌ AVOID: Data-dependent divergence (~25-50% efficiency)**

```mojo
# Different lanes may take different paths based on data
if input[global_i] > threshold:  # Unpredictable branching
    result = expensive_computation()
else:
    result = simple_computation()
```

*Performance: Random divergence kills warp efficiency*

**💀 TERRIBLE: Nested data-dependent divergence (~10-25% efficiency)**

```mojo
# Multiple levels of unpredictable branching
if input[global_i] > threshold1:
    if input[global_i] > threshold2:
        result = very_expensive()
    else:
        result = expensive()
else:
    result = simple()
```

*Performance: Warp efficiency destroyed*

## Cross-architecture compatibility

### NVIDIA vs AMD warp sizes

```mojo
from gpu.warp import WARP_SIZE

# NVIDIA GPUs:     WARP_SIZE = 32
# AMD RDNA GPUs:   WARP_SIZE = 32 (wavefront32 mode)
# AMD CDNA GPUs:   WARP_SIZE = 64 (traditional wavefront64)
```

**Why this matters:**

- **Memory patterns**: Coalesced access depends on warp size
- **Algorithm design**: Reduction trees must account for warp size
- **Performance scaling**: Twice as many lanes per warp on AMD

### Writing portable warp code

### Architecture adaptation strategies

**✅ PORTABLE: Always use `WARP_SIZE`**

```mojo
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)  # Adapts automatically
comptime ELEMENTS_PER_WARP = WARP_SIZE        # Scales with hardware
```

*Result: Code works optimally on NVIDIA/AMD (32) and AMD (64)*

**❌ BROKEN: Never hardcode warp size**

```mojo
comptime THREADS_PER_BLOCK = (32, 1)  # Breaks on AMD GPUs!
comptime REDUCTION_SIZE = 32           # Wrong on AMD!
```

*Result: Suboptimal on AMD, potential correctness issues*

### Real hardware impact

| GPU Architecture | WARP_SIZE | Memory per Warp | Reduction Steps | Lane Pattern |
|------------------|-----------|-----------------|-----------------|--------------|
| **NVIDIA/AMD RDNA** | 32 | 128 bytes (4×32) | 5 steps: 32→16→8→4→2→1 | Lanes 0-31 |
| **AMD CDNA** | 64 | 256 bytes (4×64) | 6 steps: 64→32→16→8→4→2→1 | Lanes 0-63 |

**Performance implications of 64 vs 32:**

- **CDNA advantage**: 2× memory bandwidth per warp
- **CDNA advantage**: 2× computation per warp
- **NVIDIA/RDNA advantage**: More warps per block (better occupancy)
- **Code portability**: Same source, optimal performance on both

## Memory access patterns with warps

### Coalesced memory access patterns

**✅ PERFECT: Coalesced access (100% bandwidth utilization)**

```mojo
# Adjacent lanes → adjacent memory addresses
var value = input[global_i]  # Lane 0→input[0], Lane 1→input[1], etc.
```

**Memory access patterns:**

| Access Pattern | NVIDIA/RDNA (32 lanes) | CDNA (64 lanes) | Bandwidth Utilization | Performance |
|----------------|-------------------|----------------|----------------------|-------------|
| **✅ Coalesced** | Lane N → Address 4×N | Lane N → Address 4×N | 100% | Optimal |
| | 1 transaction: 128 bytes | 1 transaction: 256 bytes | Full bus width | Fast |
| **❌ Scattered** | Lane N → Random address | Lane N → Random address | ~6% | Terrible |
| | 32 separate transactions | 64 separate transactions | Mostly idle bus | **32× slower** |

**Example addresses:**

- **Coalesced**: Lane 0→0, Lane 1→4, Lane 2→8, Lane 3→12, ...
- **Scattered**: Lane 0→1000, Lane 1→52, Lane 2→997, Lane 3→8, ...

### Shared memory bank conflicts

**What is a bank conflict?**

Assume that a GPU shared memory is divided into 32 independent **banks** that can be accessed simultaneously. A **bank conflict** occurs when multiple threads in a warp try to access different addresses within the same bank at the same time. When this happens, the hardware must **serialize** these accesses, turning what should be a single-cycle operation into multiple cycles.

**Key concepts:**

- **No conflict**: Each thread accesses a different bank → All accesses happen simultaneously (1 cycle)
- **Bank conflict**: Multiple threads access the same bank → Accesses happen sequentially (N cycles for N threads)
- **Broadcast**: All threads access the same address → Hardware optimizes this to 1 cycle

**Shared memory bank organization:**

| Bank | Addresses (byte offsets) | Example Data (float32) |
|------|--------------------------|------------------------|
| Bank 0 | 0, 128, 256, 384, ... | `shared[0]`, `shared[32]`, `shared[64]`, ... |
| Bank 1 | 4, 132, 260, 388, ... | `shared[1]`, `shared[33]`, `shared[65]`, ... |
| Bank 2 | 8, 136, 264, 392, ... | `shared[2]`, `shared[34]`, `shared[66]`, ... |
| ... | ... | ... |
| Bank 31 | 124, 252, 380, 508, ... | `shared[31]`, `shared[63]`, `shared[95]`, ... |

**Bank conflict examples:**

| Access Pattern | Bank Usage | Cycles | Performance | Explanation |
|----------------|------------|--------|-------------|-------------|
| **✅ Sequential** | `shared[thread_idx.x]` | 1 cycle | 100% | Each lane hits different bank |
| | Lane 0→Bank 0, Lane 1→Bank 1, ... | | Optimal | No conflicts |
| **✅ Same index** | `shared[0]`| 1 cycle | 100% | All lanes broadcast from same address |
| | All 32 lanes→Bank 0 (Same address) | | Optimal | No conflicts |
| **❌ Stride 2** | `shared[thread_idx.x * 2]` | 2 cycles | 50% | 2 lanes per bank |
| | Lane 0,16→Bank 0; Lane 1,17→Bank 1 | | **2× slower** | Serialized access |
| **💀 Stride 32** | `shared[thread_idx.x * 32]` | 32 cycles | 3% | All lanes hit same bank |
| | All 32 lanes→Bank 0 (Different address) | | **32× slower** | Completely serialized |

## Practical implications for warp programming

### When warp operations are most effective

1. **Reduction operations**: `sum()`, `max()`, etc.
2. **Broadcast operations**: `shuffle_idx()` to share values
3. **Neighbor communication**: `shuffle_down()` for sliding windows
4. **Prefix computations**: `prefix_sum()` for scan algorithms

### Performance characteristics

| Operation Type | Traditional | Warp Operations |
|----------------|------------|-----------------|
| **Reduction (32 elements)** | ~20 instructions | 10 instructions |
| **Memory traffic** | High | Minimal |
| **Synchronization cost** | Expensive | Free |
| **Code complexity** | High | Low |

## Next steps

Now that you understand the SIMT foundation, you're ready to see how these concepts enable powerful warp operations. The next section will show you how `sum()` transforms complex reduction patterns into simple, efficient function calls.

**→ Continue to [warp.sum() Essentials](./warp_sum.md)**
