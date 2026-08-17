# Benchmark & Profile

> **Note: The profiling section is specific to NVIDIA GPUs**
>
> The `ld.global.nc.v4` instruction and the Nsight Compute (`ncu`) metrics
> below are NVIDIA CUDA concepts. The kernels run and verify on any supported
> GPU, but the codegen evidence is NVIDIA-specific.

## Step 1: Benchmark the three variants

```bash
pixi run mojo solutions/p35/p35.mojo --benchmark
```

This times all three kernels on a 1M-element buffer. Representative numbers from
a B200 (compute capability 10.0, driver 595.71.05, MAX 26.5.0 / Mojo 1.0.0):

```text
| name      | met (ms)  | iters |
| --------- | --------- | ----- |
| scalar    | 0.01332   |  100  |
| unaligned | 0.01220   |  100  |
| aligned   | 0.01137   |  100  |
```

Read the ratios rather than the absolute times — those move with the GPU, the
driver and the toolchain, but the ordering does not.

`aligned` is the fastest and `scalar` the slowest, in the expected order. But
look how small the gap is (~15% scalar→aligned) even though the aligned kernel
issues a quarter of the global memory instructions. Buffer allocation and
initialization sit outside the timed region, so this is the kernel plus its
launch and synchronization — and at this size that fixed overhead is a large
share of it. Measured with `ncu`, the kernels alone are 8.83 µs (scalar) versus
6.05 µs (aligned): a 31% gap, twice what wall-clock reports.

While the gap is small in this example, a 4× instruction-count inefficiency
will bite in a compute-mixed or instruction-issue-bound kernel. The wall-clock
test is too coarse to see the codegen difference but the profiler is not.

## Step 2: Build for profiling

```bash
mojo build --debug-level=full solutions/p35/p35.mojo -o solutions/p35/p35_profiler
```

`--debug-level=full` keeps source-line mapping so Nsight Compute can attribute
instructions back to your kernel.

## Step 3: Confirm the instruction-count difference

The clearest evidence is the number of global load/store instructions the SM
actually executed. The aligned kernel should issue roughly `SIMD_WIDTH`× fewer:

```bash
ncu --metrics \
  smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum \
  solutions/p35/p35_profiler --unaligned

ncu --metrics \
  smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum \
  solutions/p35/p35_profiler --aligned
```

The `--unaligned` run executes about four global loads and four global stores
per chunk (for `float32x4`); the `--aligned` run executes one of each. On a
B200 that reads as 32,768 against 8,192 for each metric — an exact 4:1.

The counts are per warp, not per thread, which is why they are smaller than
the element count: 1,048,576 elements ÷ `SIMD_WIDTH` 4 = 262,144 chunks ÷ 32
lanes = 8,192 warp-level instructions, one per chunk for the aligned kernel.
Unlike timings, these counts are the same in a debug and a release build.

> **Why not read the SASS directly?** `cuobjdump -sass` is the usual way to see
> that the machine instruction changed, but it does not work here: Mojo
> JIT-compiles GPU kernels and loads them through the driver at runtime, so an
> ahead-of-time binary embeds no device code and `cuobjdump` reports
> `does not contain device code`. The instruction counts above are the evidence
> to use instead.

## Step 4: Memory workload analysis (optional)

For the full bandwidth picture:

```bash
ncu --set roofline --section=MemoryWorkloadAnalysis \
  solutions/p35/p35_profiler --aligned
```

Compare the achieved memory throughput of `--aligned` vs `--unaligned`. The
aligned kernel moves closer to the memory roofline because it spends fewer
instructions per byte moved.

## What you've shown

1. Correctness is identical. All three kernels passed in the
   [previous section](./aligned_load_store.md).
2. The instruction mix is not. Under-stated alignment forces scalar global
   loads/stores; the correct alignment yields a single vectorized instruction
   per chunk.
3. Timing alone can hide it. The codegen change is unambiguous in the
   profiler even when wall-clock looks similar, which is exactly why a
   real-world kernel can ship with this bug unnoticed.

The practice to take away from this is: at every vectorized memory access, state
the alignment (`aligned_load`, or an explicit `align_of[SIMD[dtype, width]]()`).
It costs nothing, and it ensures your kernel takes the vectorized fast path.
