# Multi-Stage Pipeline Coordination

## Overview

Implement a kernel that processes an image through a coordinated 3-stage pipeline where different thread groups handle specialized processing stages, synchronized with explicit barriers.

**Note:** _You have specialized thread roles: Stage 1 (threads 0-127) loads and preprocesses data, Stage 2 (threads 128-255) applies blur operations, and Stage 3 (all threads) performs final smoothing._

**Algorithm architecture:** This puzzle implements a **producer-consumer pipeline** where different thread groups execute completely different algorithms within a single GPU block. Unlike traditional GPU programming where all threads execute the same algorithm on different data, this approach divides threads by **functional specialization**.

**Pipeline concept:** The algorithm processes data through three distinct stages, where each stage has specialized thread groups that execute different algorithms. Each stage produces data that the next stage consumes, creating explicit **producer-consumer relationships** that must be carefully synchronized with barriers.

**Data dependencies and synchronization:** Each stage produces data that the next stage consumes:

- **Stage 1 → Stage 2**: First stage produces preprocessed data for blur processing
- **Stage 2 → Stage 3**: Second stage produces blur results for final smoothing
- **Barriers prevent race conditions** by ensuring complete stage completion before dependent stages begin

Concretely, the multi-stage pipeline implements a coordinated image processing algorithm with three mathematical operations:

**Stage 1 - Preprocessing Enhancement:**

\\[P[i] = I[i] \times 1.1\\]

where \\(P[i]\\) is the preprocessed data and \\(I[i]\\) is the input data.

**Stage 2 - Horizontal Blur Filter:**

\\[B[i] = \frac{1}{N_i} \sum_{k=-2}^{2} P[i+k] \quad \text{where } i+k \in [0, 255]\\]

where \\(B[i]\\) is the blur result, and \\(N_i\\) is the count of valid neighbors within the tile boundary.

**Stage 3 - Cascading Neighbor Smoothing:**

\\[F[i] = \begin{cases}
(B[i] + B[i+1]) \times 0.6 & \text{if } i = 0 \\\\
((B[i] + B[i-1]) \times 0.6 + B[i+1]) \times 0.6 & \text{if } 0 < i < 255 \\\\
(B[i] + B[i-1]) \times 0.6 & \text{if } i = 255
\end{cases}\\]

where \\(F[i]\\) is the final output with cascading smoothing applied.

**Thread Specialization:**

- **Threads 0-127**: Compute \\(P[i]\\) for \\(i \in \\{0, 1, 2, \ldots, 255\\}\\) (2 elements per thread)
- **Threads 128-255**: Compute \\(B[i]\\) for \\(i \in \\{0, 1, 2, \ldots, 255\\}\\) (2 elements per thread)
- **All 256 threads**: Compute \\(F[i]\\) for \\(i \in \\{0, 1, 2, \ldots, 255\\}\\) (1 element per thread)

**Synchronization Points:**

\\[\text{barrier}_1 \Rightarrow P[i] \text{ complete} \Rightarrow \text{barrier}_2 \Rightarrow B[i] \text{ complete} \Rightarrow \text{barrier}_3 \Rightarrow F[i] \text{ complete}\\]

## Key concepts

In this puzzle, you'll learn about:

- Implementing thread role specialization within a single GPU block
- Coordinating producer-consumer relationships between processing stages
- Using barriers to synchronize between different algorithms (not just within the same algorithm)

The key insight is understanding how to design multi-stage pipelines where different thread groups execute completely different algorithms, coordinated through strategic barrier placement.

**Why this matters:** Most GPU tutorials teach barrier usage within a single algorithm - synchronizing threads during reductions or shared memory operations. But real-world GPU algorithms often require **architectural complexity** with multiple distinct processing stages that must be carefully orchestrated. This puzzle demonstrates how to transform monolithic algorithms into specialized, coordinated processing pipelines.

**Previous vs. current barrier usage:**

- **Previous puzzles ([P8](../puzzle_08/puzzle_08.md), [P12](../puzzle_12/puzzle_12.md), [P15](../puzzle_15/puzzle_15.md)):** All threads execute the same algorithm, barriers sync within algorithm steps
- **This puzzle:** Different thread groups execute different algorithms, barriers coordinate between different algorithms

**Thread specialization architecture:** Unlike data parallelism where threads differ only in their data indices, this puzzle implements **algorithmic parallelism** where threads execute fundamentally different code paths based on their role in the pipeline.

## Configuration

**System parameters:**

- **Image size**: `SIZE = 1024` elements (1D for simplicity)
- **Threads per block**: `TPB = 256` threads organized as `(256, 1)` block dimension
- **Grid configuration**: `(4, 1)` blocks to process entire image in tiles (4 blocks total)
- **Data type**: `DType.float32` for all computations

**Thread specialization architecture:**

- **Stage 1 threads**: `STAGE1_THREADS = 128` (threads 0-127, first half of block)
  - **Responsibility**: Load input data from global memory and apply preprocessing
  - **Work distribution**: Each thread processes 2 elements for efficient load balancing
  - **Output**: Populates `input_shared[256]` with preprocessed data

- **Stage 2 threads**: `STAGE2_THREADS = 128` (threads 128-255, second half of block)
  - **Responsibility**: Apply horizontal blur filter on preprocessed data
  - **Work distribution**: Each thread processes 2 blur operations
  - **Output**: Populates `blur_shared[256]` with blur results

- **Stage 3 threads**: All 256 threads collaborate
  - **Responsibility**: Final smoothing and output to global memory
  - **Work distribution**: One-to-one mapping (thread `i` processes element `i`)
  - **Output**: Writes final results to global `output` array

## Code to complete

```mojo
{{#include ../../../problems/p29/p29.mojo:multi_stage_pipeline}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p29/p29.mojo" class="filename">View full file: problems/p29/p29.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

### **Thread role identification**

- Use thread index comparisons to determine which stage each thread should execute
- Stage 1: First half of threads (threads 0-127)
- Stage 2: Second half of threads (threads 128-255)
- Stage 3: All threads participate

### **Stage 1 approach**

- Identify Stage 1 threads using appropriate index comparison
- Each thread should handle multiple elements for load balancing
- Apply the preprocessing enhancement factor
- Implement proper boundary handling with zero-padding

### **Stage 2 approach**

- Identify Stage 2 threads and map their indices to processing range
- Implement the blur kernel by averaging neighboring elements
- Handle boundary conditions by only including valid neighbors
- Process multiple elements per thread for efficiency

### **Stage 3 approach**

- All threads participate in final processing
- Apply neighbor smoothing using the specified scaling factor
- Handle edge cases where neighbors may not exist
- Write results to global output with bounds checking

### **Synchronization strategy**

- Place barriers between stages to prevent race conditions
- Ensure each stage completes before dependent stages begin
- Use final barrier to guarantee completion before block exit

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
pixi run p29 --multi-stage
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p29 --multi-stage
```

  </div>
  <div class="tab-content">

```bash
uv run poe p29 --multi-stage
```

  </div>
</div>

After completing the puzzle successfully, you should see output similar to:

```
Puzzle 29: GPU Synchronization Primitives
==================================================
TPB: 256
SIZE: 1024
STAGE1_THREADS: 128
STAGE2_THREADS: 128
BLUR_RADIUS: 2

Testing Puzzle 29A: Multi-Stage Pipeline Coordination
============================================================
Multi-stage pipeline blur completed
Input sample: 0.0 1.01 2.02
Output sample: 1.6665002 2.3331003 3.3996604
✅ Multi-stage pipeline coordination test PASSED!
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p29/p29.mojo:multi_stage_pipeline_solution}}
```

<div class="solution-explanation">

The key insight is recognizing this as a **pipeline architecture problem** with thread role specialization:

1. **Design stage-specific thread groups**: Divide threads by function, not just by data
2. **Implement producer-consumer chains**: Stage 1 produces for Stage 2, Stage 2 produces for Stage 3
3. **Use strategic barrier placement**: Synchronize between different algorithms, not within the same algorithm
4. **Optimize memory access patterns**: Ensure coalesced reads and efficient shared memory usage

<strong>Complete Solution with Detailed Explanation</strong>

The multi-stage pipeline solution demonstrates sophisticated thread specialization and barrier coordination. This approach transforms a traditional monolithic GPU algorithm into a specialized, coordinated processing pipeline.

## **Pipeline architecture design**

The fundamental breakthrough in this puzzle is **thread specialization by role** rather than by data:

**Traditional approach:** All threads execute the same algorithm on different data

- Everyone performs identical operations (like reductions or matrix operations)
- Barriers synchronize threads within the same algorithm steps
- Thread roles differ only by data indices they process

**This puzzle's innovation:** Different thread groups execute completely different algorithms

- Threads 0-127 execute loading and preprocessing algorithms
- Threads 128-255 execute blur processing algorithms
- All threads collaborate in final smoothing algorithm
- Barriers coordinate between different algorithms, not within the same algorithm

## **Producer-consumer coordination**

Unlike previous puzzles where threads were peers in the same algorithm, this establishes explicit producer-consumer relationships:

- **Stage 1**: Producer (creates preprocessed data for Stage 2)
- **Stage 2**: Consumer (uses Stage 1 data) + Producer (creates blur data for Stage 3)
- **Stage 3**: Consumer (uses Stage 2 data)

## **Strategic barrier placement**

Understanding when barriers are necessary vs. wasteful:

- **Necessary**: Between dependent stages to prevent race conditions
- **Wasteful**: Within independent operations of the same stage
- **Performance insight**: Each barrier has a cost - use them strategically

**Critical synchronization points:**

1. **After Stage 1**: Prevent Stage 2 from reading incomplete preprocessed data
2. **After Stage 2**: Prevent Stage 3 from reading incomplete blur results
3. **After Stage 3**: Ensure all output writes complete before block termination

## **Thread utilization patterns**

- **Stage 1**: 50% utilization (128/256 threads active, 128 idle)
- **Stage 2**: 50% utilization (128 active, 128 idle)
- **Stage 3**: 100% utilization (all 256 threads active)

This demonstrates sophisticated **algorithmic parallelism** where different thread groups specialize in different computational tasks within a coordinated pipeline, moving beyond simple data parallelism to architectural thinking required for real-world GPU algorithms.

## **Memory hierarchy optimization**

**Shared memory architecture:**

- Two specialized buffers handle data flow between stages
- Global memory access minimized to boundary operations only
- All intermediate processing uses fast shared memory

**Access pattern benefits:**

- **Stage 1**: Coalesced global memory reads for input loading
- **Stage 2**: Fast shared memory reads for blur processing
- **Stage 3**: Coalesced global memory writes for output

## **Real-world applications**

This pipeline architecture pattern is fundamental to:

**Image processing pipelines:**

- Multi-stage filters (blur, sharpen, edge detection in sequence)
- Color space conversions (RGB → HSV → processing → RGB)
- Noise reduction with multiple algorithm passes

**Scientific computing:**

- Stencil computations with multi-stage finite difference methods
- Signal processing with filtering, transformation, and analysis pipelines
- Computational fluid dynamics with multi-stage solver iterations

**Machine learning:**

- Neural network layers with specialized thread groups for different operations
- Data preprocessing pipelines (load, normalize, augment in coordinated stages)
- Batch processing where different thread groups handle different operations

## **Key technical insights**

**Algorithmic vs. data parallelism:**

- **Data parallelism**: Threads execute identical code on different data elements
- **Algorithmic parallelism**: Threads execute fundamentally different algorithms based on their specialized roles

**Barrier usage philosophy:**

- **Strategic placement**: Barriers only where necessary to prevent race conditions between dependent stages
- **Performance consideration**: Each barrier incurs synchronization overhead - use sparingly but correctly
- **Correctness guarantee**: Proper barrier placement ensures deterministic results regardless of thread execution timing

**Thread specialization benefits:**

- **Algorithmic optimization**: Each stage can be optimized for its specific computational pattern
- **Memory access optimization**: Different stages can use different memory access strategies
- **Resource utilization**: Complex algorithms can be decomposed into specialized, efficient components

This solution demonstrates how to design sophisticated GPU algorithms that leverage thread specialization and strategic synchronization for complex multi-stage computations, moving beyond simple parallel loops to architectural approaches used in production GPU software.

</details>
