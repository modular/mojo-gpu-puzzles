# Double-Buffered Stencil Computation

> **🔬 Fine-Grained Synchronization: mbarrier vs barrier()**
>
> This puzzle introduces **explicit memory barrier APIs** that provide significantly more control than the basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) function used in previous puzzles.
>
> **Basic `barrier()` limitations:**
>
> - **Fire-and-forget**: Single synchronization point with no state tracking
> - **Block-wide only**: All threads in the block must participate simultaneously
> - **No reusability**: Each barrier() call creates a new synchronization event
> - **Coarse-grained**: Limited control over memory ordering and timing
> - **Static coordination**: Cannot adapt to different thread participation patterns
>
> **Advanced [`mbarrier APIs`](https://docs.modular.com/mojo/stdlib/gpu/sync/) capabilities:**
>
> - **Precise control**: [`mbarrier_init()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init) sets up reusable barrier objects with specific thread counts
> - **State tracking**: [`mbarrier_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive) signals individual thread completion and maintains arrival count
> - **Flexible waiting**: [`mbarrier_test_wait()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait) allows threads to wait for specific completion states
> - **Reusable objects**: Same barrier can be reinitialized and reused across multiple iterations
> - **Multiple barriers**: Different barrier objects for different synchronization points (initialization, iteration, finalization)
> - **Hardware optimization**: Maps directly to GPU hardware synchronization primitives for better performance
> - **Memory semantics**: Explicit control over memory visibility and ordering guarantees
>
> **Why this matters for iterative algorithms:**
> In double-buffering patterns, you need **precise coordination** between buffer swap phases. Basic `barrier()` cannot provide the fine-grained control required for:
>
> - **Buffer role alternation**: Ensuring all writes to buffer_A complete before reading from buffer_A begins
> - **Iteration boundaries**: Coordinating multiple synchronization points within a single kernel
> - **State management**: Tracking which threads have completed which phase of processing
> - **Performance optimization**: Minimizing synchronization overhead through reusable barrier objects
>
> This puzzle demonstrates **synchronization patterns** used in real-world GPU computing applications like iterative solvers, simulation frameworks, and high-performance image processing pipelines.

## Overview

Implement a kernel that performs iterative stencil operations using double-buffered shared memory, coordinated with explicit memory barriers to ensure safe buffer swapping between iterations. A stencil operation is a computational pattern where the value of each element in an array is calculated based on a fixed pattern of its neighbors.

**Note:** _You have alternating buffer roles: `buffer_A` and `buffer_B` swap between read and write operations each iteration, with mbarrier synchronization ensuring all threads complete writes before buffer swaps._

**Algorithm architecture:** This puzzle implements a **double-buffering pattern** where two shared memory buffers alternate roles as read and write targets across multiple iterations. Unlike simple stencil operations that process data once, this approach performs iterative refinement with careful memory barrier coordination to prevent race conditions during buffer transitions.

**Pipeline concept:** The algorithm processes data through iterative stencil refinement, where each iteration reads from one buffer and writes to another. The buffers alternate roles each iteration, creating a ping-pong pattern that enables continuous processing without data corruption.

**Data dependencies and synchronization:** Each iteration depends on the complete results of the previous iteration:

- **Iteration N → Iteration N+1**: Current iteration produces refined data that next iteration consumes
- **Buffer coordination**: Read and write buffers swap roles each iteration
- **Memory barriers prevent race conditions** by ensuring all writes complete before any thread begins reading from the newly written buffer

Concretely, the double-buffered stencil implements an iterative smoothing algorithm with three mathematical operations:

**Iteration Pattern - Buffer Alternation:**

\\[\\text{Iteration } i: \\begin{cases}
\\text{Read from buffer\_A, Write to buffer\_B} & \\text{if } i \\bmod 2 = 0 \\\\
\\text{Read from buffer\_B, Write to buffer\_A} & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Stencil Operation - 3-Point Average:**

\\[S^{(i+1)}[j] = \\frac{1}{N_j} \\sum_{k=-1}^{1} S^{(i)}[j+k] \\quad \\text{where } j+k \\in [0, 255]\\]

where \\(S^{(i)}[j]\\) is the stencil value at position \\(j\\) after iteration \\(i\\), and \\(N_j\\) is the count of valid neighbors.

**Memory Barrier Coordination:**

\\[\\text{mbarrier\_arrive}() \\Rightarrow \\text{mbarrier\_test\_wait}() \\Rightarrow \\text{buffer swap} \\Rightarrow \\text{next iteration}\\]

**Final Output Selection:**

\\[\\text{Output}[j] = \\begin{cases}
\\text{buffer\_A}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 0 \\\\
\\text{buffer\_B}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 1
\\end{cases}\\]

## Key concepts

In this puzzle, you'll learn about:

- Implementing double-buffering patterns for iterative algorithms
- Coordinating explicit memory barriers using [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/)
- Managing alternating read/write buffer roles across iterations

The key insight is understanding how to safely coordinate buffer swapping in iterative algorithms where race conditions between read and write operations can corrupt data if not properly synchronized.

**Why this matters:** Most GPU tutorials show simple one-pass algorithms, but real-world applications often require **iterative refinement** with multiple passes over data. Double-buffering is essential for algorithms like iterative solvers, image processing filters, and simulation updates where each iteration depends on the complete results of the previous iteration.

**Previous vs. current synchronization:**

- **Previous puzzles ([P8](../puzzle_08/puzzle_08.md), [P12](../puzzle_12/puzzle_12.md), [P15](../puzzle_15/puzzle_15.md)):** Simple [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) calls for single-pass algorithms
- **This puzzle:** Explicit [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) for precise control over buffer swap timing

**Memory barrier specialization:** Unlike basic thread synchronization, this puzzle uses **explicit memory barriers** that provide fine-grained control over when memory operations complete, essential for complex memory access patterns.

## Configuration

**System parameters:**

- **Image size**: `SIZE = 1024` elements (1D for simplicity)
- **Threads per block**: `TPB = 256` threads organized as `(256, 1)` block dimension
- **Grid configuration**: `(4, 1)` blocks to process entire image in tiles (4 blocks total)
- **Data type**: `DType.float32` for all computations

**Iteration parameters:**

- **Stencil iterations**: `STENCIL_ITERATIONS = 3` refinement passes
- **Buffer count**: `BUFFER_COUNT = 2` (double-buffering)
- **Stencil kernel**: 3-point averaging with radius 1

**Buffer architecture:**

- **buffer_A**: Primary shared memory buffer (`[256]` elements)
- **buffer_B**: Secondary shared memory buffer (`[256]` elements)
- **Role alternation**: Buffers swap between read source and write target each iteration

**Processing requirements:**

**Initialization phase:**

- **Buffer setup**: Initialize buffer_A with input data, buffer_B with zeros
- **Barrier initialization**: Set up [mbarrier objects](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init) for synchronization points
- **Thread coordination**: All threads participate in initialization

**Iterative processing:**

- **Even iterations** (0, 2, 4...): Read from buffer_A, write to buffer_B
- **Odd iterations** (1, 3, 5...): Read from buffer_B, write to buffer_A
- **Stencil operation**: 3-point average \\((\\text{left} + \\text{center} + \\text{right}) / 3\\)
- **Boundary handling**: Use adaptive averaging for elements at buffer edges

**Memory barrier coordination:**

- **[mbarrier_arrive()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)**: Each thread signals completion of write phase
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)**: All threads wait until everyone completes writes
- **Buffer swap safety**: Prevents reading from buffer while others still writing
- **Barrier reinitialization**: Reset barrier state between iterations

**Output phase:**

- **Final buffer selection**: Choose active buffer based on iteration parity
- **Global memory write**: Copy final results to output array
- **Completion barrier**: Ensure all writes finish before block termination

## Code to complete

```mojo
{{#include ../../../problems/p29/p29.mojo:double_buffered_stencil}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p29/p29.mojo" class="filename">View full file: problems/p29/p29.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### **Buffer initialization**

- Initialize `buffer_A` with input data, `buffer_B` can start empty
- Use proper bounds checking with zero-padding for out-of-range elements
- Only thread 0 should initialize the mbarrier objects
- Set up separate barriers for different synchronization points

### **Iteration control**

- Use `@parameter for iteration in range(STENCIL_ITERATIONS)` for compile-time unrolling
- Determine buffer roles using `iteration % 2` to alternate read/write assignments
- Apply stencil operation only within valid bounds with neighbor checking

### **Stencil computation**

- Implement 3-point averaging: `(left + center + right) / 3`
- Handle boundary conditions by only including valid neighbors in average
- Use adaptive counting to handle edge cases gracefully

### **Memory barrier coordination**

- Call [`mbarrier_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive) after each thread completes its write operations
- Use [`mbarrier_test_wait()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait) to ensure all threads finish before buffer swap
- Reinitialize barriers between iterations for reuse: [`mbarrier_init()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init)
- Only thread 0 should reinitialize barriers to avoid race conditions

### **Output selection**

- Choose final active buffer based on `STENCIL_ITERATIONS % 2`
- Even iteration counts end with data in buffer_A
- Odd iteration counts end with data in buffer_B
- Write final results to global output with bounds checking

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p29 --double-buffer
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p29 --double-buffer
```

  </div>
  <div class="tab-content">

```bash
uv run poe p29 --double-buffer
```

  </div>
</div>

After completing the puzzle successfully, you should see output similar to:

```
Puzzle 29: GPU Synchronization Primitives
==================================================
TPB: 256
SIZE: 1024
STENCIL_ITERATIONS: 3
BUFFER_COUNT: 2

Testing Puzzle 29B: Double-Buffered Stencil Computation
============================================================
Double-buffered stencil completed
Input sample: 1.0 1.0 1.0
GPU output sample: 1.0 1.0 1.0
✅ Double-buffered stencil test PASSED!
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p29/p29.mojo:double_buffered_stencil_solution}}
```

<div class="solution-explanation">

The key insight is recognizing this as a **double-buffering architecture problem** with explicit memory barrier coordination:

1. **Design alternating buffer roles**: Swap read/write responsibilities each iteration
2. **Implement explicit memory barriers**: Use mbarrier APIs for precise synchronization control
3. **Coordinate iterative processing**: Ensure complete iteration results before buffer swaps
4. **Optimize memory access patterns**: Keep all processing in fast shared memory

<details class="solution-details">

<strong>Complete Solution with Detailed Explanation</strong>

The double-buffered stencil solution demonstrates sophisticated memory barrier coordination and iterative processing patterns. This approach enables safe iterative refinement algorithms that require precise control over memory access timing.

## **Double-buffering architecture design**

The fundamental breakthrough in this puzzle is **explicit memory barrier control** rather than simple thread synchronization:

**Traditional approach:** Use basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) for simple thread coordination

- All threads execute same operation on different data
- Single barrier call synchronizes thread completion
- No control over specific memory operation timing

**This puzzle's innovation:** Different buffer roles coordinated with explicit memory barriers

- buffer_A and buffer_B alternate between read source and write target
- [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) provide precise control over memory operation completion
- Explicit coordination prevents race conditions during buffer transitions

## **Iterative processing coordination**

Unlike single-pass algorithms, this establishes iterative refinement with careful buffer management:

- **Iteration 0**: Read from buffer_A (initialized with input), write to buffer_B
- **Iteration 1**: Read from buffer_B (previous results), write to buffer_A
- **Iteration 2**: Read from buffer_A (previous results), write to buffer_B
- **Continue alternating**: Each iteration refines results from previous iteration

## **Memory barrier API usage**

Understanding the mbarrier coordination pattern:

- **[mbarrier_init()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init)**: Initialize barrier for specific thread count (TPB)
- **[mbarrier_arrive()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)**: Signal individual thread completion of write phase
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)**: Block until all threads signal completion
- **Reinitialization**: Reset barrier state between iterations for reuse

**Critical timing sequence:**

1. **All threads write**: Each thread updates its assigned buffer element
2. **Signal completion**: Each thread calls [`mbarrier_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)
3. **Wait for all**: All threads call [`mbarrier_test_wait()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)
4. **Safe to proceed**: Now safe to swap buffer roles for next iteration

## **Stencil operation mechanics**

The 3-point stencil operation with adaptive boundary handling:

**Interior elements** (indices 1 to 254):

```mojo
# Average with left, center, and right neighbors
stencil_sum = buffer[i-1] + buffer[i] + buffer[i+1]
result[i] = stencil_sum / 3.0
```

**Boundary elements** (indices 0 and 255):

```mojo
# Only include valid neighbors in average
stencil_count = 0
for neighbor in valid_neighbors:
    stencil_sum += buffer[neighbor]
    stencil_count += 1
result[i] = stencil_sum / stencil_count
```

## **Buffer role alternation**

The ping-pong buffer pattern ensures data integrity:

**Even iterations** (0, 2, 4...):

- **Read source**: buffer_A contains current data
- **Write target**: buffer_B receives updated results
- **Memory flow**: buffer_A → stencil operation → buffer_B

**Odd iterations** (1, 3, 5...):

- **Read source**: buffer_B contains current data
- **Write target**: buffer_A receives updated results
- **Memory flow**: buffer_B → stencil operation → buffer_A

## **Race condition prevention**

Memory barriers eliminate multiple categories of race conditions:

**Without barriers (broken)**:

```mojo
# Thread A writes to buffer_B[10]
buffer_B[10] = stencil_result_A

# Thread B immediately reads buffer_B[10] for its stencil
# RACE CONDITION: Thread B might read old value before Thread A's write completes
stencil_input = buffer_B[10]  // Undefined behavior!
```

**With barriers (correct)**:

```mojo
# All threads write their results
buffer_B[local_i] = stencil_result

# Signal write completion
mbarrier_arrive(barrier)

# Wait for ALL threads to complete writes
mbarrier_test_wait(barrier, TPB)

# Now safe to read - all writes guaranteed complete
stencil_input = buffer_B[neighbor_index]  // Always sees correct values
```

## **Output buffer selection**

Final result location depends on iteration parity:

**Mathematical determination**:

- **STENCIL_ITERATIONS = 3** (odd number)
- **Final active buffer**: Iteration 2 writes to buffer_B
- **Output source**: Copy from buffer_B to global memory

**Implementation pattern**:

```mojo
@parameter
if STENCIL_ITERATIONS % 2 == 0:
    # Even total iterations end in buffer_A
    output[global_i] = buffer_A[local_i]
else:
    # Odd total iterations end in buffer_B
    output[global_i] = buffer_B[local_i]
```

## **Performance characteristics**

**Memory hierarchy optimization:**

- **Global memory**: Accessed only for input loading and final output
- **Shared memory**: All iterative processing uses fast shared memory
- **Register usage**: Minimal due to shared memory focus

**Synchronization overhead:**

- **mbarrier cost**: Higher than basic barrier() but provides essential control
- **Iteration scaling**: Overhead increases linearly with iteration count
- **Thread efficiency**: All threads remain active throughout processing

## **Real-world applications**

This double-buffering pattern is fundamental to:

**Iterative solvers:**

- Gauss-Seidel and Jacobi methods for linear systems
- Iterative refinement for numerical accuracy
- Multigrid methods with level-by-level processing

**Image processing:**

- Multi-pass filters (bilateral, guided, edge-preserving)
- Iterative denoising algorithms
- Heat diffusion and anisotropic smoothing

**Simulation algorithms:**

- Cellular automata with state evolution
- Particle systems with position updates
- Fluid dynamics with iterative pressure solving

## **Key technical insights**

**Memory barrier philosophy:**

- **Explicit control**: Precise timing control over memory operations vs automatic synchronization
- **Race prevention**: Essential for any algorithm with alternating read/write patterns
- **Performance trade-off**: Higher synchronization cost for guaranteed correctness

**Double-buffering benefits:**

- **Data integrity**: Eliminates read-while-write hazards
- **Algorithm clarity**: Clean separation between current and next iteration state
- **Memory efficiency**: No need for global memory intermediate storage

**Iteration management:**

- **Compile-time unrolling**: `@parameter for` enables optimization opportunities
- **State tracking**: Buffer role alternation must be deterministic
- **Boundary handling**: Adaptive stencil operations handle edge cases gracefully

This solution demonstrates how to design iterative GPU algorithms that require precise memory access control, moving beyond simple parallel loops to sophisticated memory management patterns used in production numerical software.

</details>
