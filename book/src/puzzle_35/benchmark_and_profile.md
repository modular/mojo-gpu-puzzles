# Benchmark & Profile

> **Note: The profiling section is specific to NVIDIA GPUs**
>
> The `ld.global.nc.v4` instruction, the `LDG.E.128` SASS form, and the NSight
> Compute (`ncu`) metrics below are NVIDIA CUDA concepts. The kernels run and
> verify on any supported GPU, but the codegen evidence is NVIDIA-specific.

## Step 1: Benchmark the three variants

```bash
pixi run mojo solutions/p35/p35.mojo --benchmark
```

This times all three kernels on a 1M-element buffer. Representative numbers from
a B200:

```text
| name      | met (ms)  | iters |
| --------- | --------- | ----- |
| scalar    | 0.0248    |  100  |
| unaligned | 0.0238    |  100  |
| aligned   | 0.0228    |  100  |
```

`aligned` is the fastest and `scalar` the slowest, in the expected order. But
look how small the gap is (~8% scalar→aligned). The aligned kernel issues a
quarter of the global memory instructions, yet at this size the kernel is close
to DRAM-bandwidth-bound, so all three nearly saturate the same memory and the
wall-clock barely separates them.

While the gap is small in this example, leaving a 4× instruction-count
inefficiency in the kernel will impact performance bite in a compute-mixed or
instruction-issue-bound context. The wall-clock test is too coarse to see the
codegen difference but the profiler is not.

## Step 2: Build for profiling

```bash
mojo build --debug-level=full solutions/p35/p35.mojo -o solutions/p35/p35_profiler
```

`--debug-level=full` keeps source-line mapping so NSight Compute can attribute
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
per chunk (for `float32x4`); the `--aligned` run executes one of each.

## Step 4: See the vectorized instruction in the SASS

Confirm the actual machine instruction changed from scalar `LDG.E` to vectorized
`LDG.E.128`:

```bash
cuobjdump -sass solutions/p35/p35_profiler | grep -E 'LDG|STG'
```

The aligned kernel's SASS contains `LDG.E.128` / `STG.E.128` (the 128-bit
vectorized forms); the unaligned kernel's contains only the 32-bit `LDG.E` /
`STG.E`.

## Step 5: Memory workload analysis (optional)

For the full bandwidth picture:

```bash
ncu --set=@roofline --section=MemoryWorkloadAnalysis \
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
