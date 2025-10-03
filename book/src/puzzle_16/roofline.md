# Understanding GPU Performance: The Roofline Model

Having implemented the naive matrix multiplication, you might be wondering: *How well is our kernel actually performing?* Is it limited by the GPU's computational power, or is something else holding it back?

The **roofline model** is your compass for GPU optimization—it reveals which hardware bottleneck limits your kernel's performance and guides you toward the most impactful optimizations. Rather than guessing at improvements, the roofline model shows you exactly where to focus your efforts.

## 1. Two ceilings for every GPU kernel

Every GPU kernel operates under two fundamental constraints:

- **Compute ceiling** – how quickly the cores can execute floating-point operations (peak FLOPs/s)
- **Memory ceiling** – how quickly the memory system can feed those cores with data (peak bytes/s)

Understanding which ceiling constrains your kernel is crucial for optimization strategy. The roofline model visualizes this relationship by plotting two key metrics:

**X-axis: Arithmetic Intensity** – How much computation you extract per byte of data

\\[\Large I = \frac{\text{Total FLOPs}}{\text{Total Bytes from Memory}} \quad [\text{FLOP/B}]\\]

**Y-axis: Sustained Performance** – How fast your kernel actually runs

\\[\Large P_{\text{sustained}} = \frac{\text{Total FLOPs}}{\text{Elapsed Time}} \quad [\text{GFLOP/s}]\\]

Two "roofs" bound all achievable performance:

| Roof             | Equation                         | Meaning                                            |
| ---------------- | -------------------------------- | -------------------------------------------------- |
| **Memory roof**  | \\(P = B_{\text{peak}} \cdot I\\) | Sloped line; performance limited by memory bandwidth |
| **Compute roof** | \\(P = P_{\text{peak}}\\)         | Horizontal line; performance limited by compute throughput |

The **critical intensity**

\\[\Large I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}\\]

marks where a kernel transitions from memory-bound (\\(I < I^* \\)) to compute-bound (\\(I > I^* \\)).

## 2. Hardware example: NVIDIA A100 specifications

Let's ground this theory in concrete numbers using the NVIDIA A100:

**Peak FP32 throughput**
\\[\Large P_{\text{peak}} = 19.5 \text{ TFLOP/s} = 19{,}500 \text{ GFLOP/s}\\]

**Peak HBM2 bandwidth**
\\[\Large B_{\text{peak}} = 1{,}555 \text{ GB/s}\\]

**Critical intensity**
\\[\Large I^* = \frac{19{,}500}{1{,}555} \approx 12.5 \text{ FLOP/B}\\]

*Source: [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)*

This means kernels with arithmetic intensity below 12.5 FLOP/B are memory-bound, while those above are compute-bound.

## 3. Visualizing our matrix multiplication implementations

The animation below shows how our puzzle implementations map onto the A100's roofline model:

![Roofline Model Visualization](media/videos/720p30/roofline_model_viz.gif)

The visualization demonstrates the optimization journey we'll take in this puzzle:

1. **Hardware constraints** – The red memory roof and blue compute roof define performance limits
2. **Our starting point** – The naive implementation (orange dot) sitting firmly on the memory roof
3. **Optimization target** – The shared memory version (teal dot) with improved arithmetic intensity
4. **Ultimate goal** – The golden arrow pointing toward the critical intensity where kernels become compute-bound

## 4. Analyzing our naive implementation

Let's examine why our naive kernel from the previous section performs as it does. For our \\(2 \times 2\\) matrix multiplication:

**Computation per output element**: \\(\text{SIZE} + (\text{SIZE}-1) = 3 \text{ FLOPs }\\)

 > Each element requires \\(\text{SIZE}\\) multiplications and \\(\text{SIZE} - 1\\) additions:
 > \\[C_{00} = A_{00} \cdot B_{00} + A_{01} \cdot B_{10}\\]
 > For \\(\text{SIZE} = 2\\) it is 2 multiplications + 1 addition = 3 FLOPs

**Memory accesses per output element**:
- Row from matrix A: \\(2 \times 4 = 8\\) bytes (FP32)
- Column from matrix B: \\(2 \times 4 = 8\\) bytes (FP32)
- Total: \\(16\\) bytes per output element

**Arithmetic intensity**:
\\[\Large I_{\text{naive}} = \frac{3 \text{ FLOPs}}{16 \text{ bytes}} = 0.1875 \text{ FLOP/B}\\]

This arithmetic intensity is far below the compute roof of an A100, indicating that our naive kernel is **severely memory-bound**.

\\[\Large I_{\text{naive}} = 0.1875 \ll I^* = 12.5\\]

**Expected performance**:
\\[\Large P \approx B_{\text{peak}} \times I_{\text{naive}} = 1{,}555 \times 0.1875 \approx 292 \text{ GFLOP/s}\\]

This represents only \\(\frac{292}{19{,}500} \approx 1.5\%\\) of the GPU's computational potential! The visualization clearly shows this as the yellow dot sitting squarely on the memory roof—we're nowhere near the compute ceiling.

## 5. The path forward: shared memory optimization

The roofline model reveals our optimization strategy: **increase arithmetic intensity** by reducing redundant memory accesses. This is exactly what the shared memory approach accomplishes:

**Shared memory benefits**:
- **Cooperative loading**: Threads work together to load matrix blocks into fast shared memory
- **Data reuse**: Each loaded element serves multiple computations
- **Reduced global memory traffic**: Fewer accesses to slow global memory

**Expected arithmetic intensity improvement**:
\\[\Large I_{\text{shared}} = \frac{12 \text{ FLOPs}}{32 \text{ bytes}} = 0.375 \text{ FLOP/B}\\]

While still memory-bound for our small \\(2 \times 2\\) case, this 2× improvement in arithmetic intensity scales dramatically for larger matrices where shared memory tiles can be reused many more times.

## 6. Optimization strategies revealed by the roofline

The roofline model not only diagnoses current performance but also illuminates optimization paths. Here are the key techniques we'll explore in later puzzles:

| Technique                       | Roofline effect                                               | Implementation approach                                        |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| **Shared memory tiling**        | ↑ Arithmetic intensity through data reuse                    | Cooperative loading, block-wise computation                    |
| **Register blocking**           | Reduce memory traffic with register accumulation             | Loop unrolling with register variables                         |
| **Kernel fusion**              | More FLOPs per byte by combining operations                   | Single kernel handling multiple computation stages             |
| **Memory coalescing**          | Maximize effective bandwidth utilization                      | Structured access patterns, proper thread organization         |
| **Asynchronous memory copies** | Dedicated copy engine enables compute-memory overlap          | `copy_dram_to_sram_async` with computation overlap            |
| **Mixed precision**            | Smaller data types reduce memory pressure                     | FP16/BF16 input with FP32 accumulation                        |

Each technique moves kernels along the roofline—either up the memory roof (better bandwidth utilization) or rightward toward the compute roof (higher arithmetic intensity).

**Note on asynchronous operations**: Standard GPU memory loads (`ld.global`) are already asynchronous - warps continue executing until they need the loaded data. Specialized async copy instructions like `cp.async` (CUDA) or [copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/) (Mojo) provide additional benefits by using dedicated copy engines, bypassing registers, and enabling better resource utilization rather than simply making synchronous operations asynchronous.

## 7. Beyond simple rooflines

**Multi-level memory**: Advanced rooflines include separate ceilings for L2 cache, shared memory, and register bandwidth to identify which memory hierarchy level constrains performance.

**Communication rooflines**: For multi-GPU applications, replace memory bandwidth with interconnect bandwidth (NVLink, InfiniBand) to analyze scaling efficiency.

**Specialized units**: Modern GPUs include tensor cores with their own performance characteristics, requiring specialized roofline analysis.

## 8. Using the roofline in practice

1. **Profile your kernel**: Use tools like Nsight Compute to measure actual FLOPs and memory traffic
2. **Plot the data point**: Calculate arithmetic intensity and sustained performance
3. **Identify the bottleneck**: Memory-bound kernels sit on the memory roof; compute-bound kernels approach the compute roof
4. **Choose optimizations**: Focus on bandwidth improvements for memory-bound kernels, algorithmic changes for compute-bound ones
5. **Measure and iterate**: Verify that optimizations move kernels in the expected direction

## Connection to our shared memory puzzle

In the next section, we'll implement the **shared memory optimization** that begins moving our kernel up the roofline. As the visualization shows, this takes us from the orange dot (naive) to the teal dot (shared memory)—a clear performance improvement through better data reuse.

While our \\(2 \times 2\\) example won't reach the compute roof, you'll see how the same principles scale to larger matrices where shared memory becomes crucial for performance. The roofline model provides the theoretical foundation for understanding **why** shared memory helps and **how much** improvement to expect.

Understanding the roofline model transforms GPU optimization from guesswork into systematic engineering. Every optimization technique in this book can be understood through its effect on this simple but powerful performance model.
