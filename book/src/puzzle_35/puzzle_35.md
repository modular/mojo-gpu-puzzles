# Puzzle 35: Memory Alignment for Load/Store Performance

## Why this puzzle matters

You can write a GPU kernel that is mathematically correct, uses a sensible SIMD
width, and still makes inefficient use of memory bandwidth. The culprit is alignment:
whether the compiler *knows* a vectorized load or store lands on a properly aligned
address.

This is not a contrived concern. Picture a memory-bound kernel on recent NVIDIA
hardware that issues scalar global loads where it could have issued a single
128-bit vectorized load, purely because the access alignment is under-stated at
the API boundary. If the compiler hasn't been guaranteed that the data is aligned,
it conservatively emits the slow path. Explicitly stating alignment allows for more
efficient instructions and use of bandwidth.

In this puzzle, you'll write the same memory-bound kernel three ways, confirm all
three produce identical results, and then use benchmarking and NSight Compute to
see why only one of them saturates memory bandwidth.

## Overview

Modern GPUs move memory in wide transactions. A `float32x4` (128-bit) load maps
to a single `ld.global.nc.v4.f32` instruction, but only if the compiler can
prove the access is 16-byte aligned. When it can't, it falls back to four
separate scalar loads, quadrupling the instruction count on the memory pipeline.

**What you'll discover:**

- How alignment controls whether the compiler emits vectorized memory
  instructions
- Why three kernels with identical output can have very different bandwidth
- How to communicate alignment through the `LayoutTensor` API
  (`aligned_load`, `load`/`store` with explicit alignment)
- How to confirm the codegen change with NSight Compute

## Key concepts

- **Natural alignment**: `align_of[dtype]()` (4 bytes for `float32`) vs the
  alignment of a full vector, `align_of[SIMD[dtype, width]]()` (16 bytes for
  `float32x4`).
- **The under-stated-alignment trap**: passing the scalar alignment to a
  vectorized `load`/`store` so the compiler can't vectorize.
- **The aligned fast path**: `aligned_load` / explicit `store_alignment`, which
  lower to `ld.global.nc.v4` / `st.global.v4`.
- **Memory-bound vs compute-bound**: alignment matters most when load/store is
  the bottleneck.

## Puzzle structure

### **[📐 Why Alignment Matters](./alignment_basics.md)**

The hardware and compiler background: how wide memory transactions work, what
`align_of` tells you, and why a missing alignment hint forces scalar codegen.

### **[🔧 Aligned Load & Store](./aligned_load_store.md)**

The exercise. Implement three kernels (scalar, vectorized-but-under-aligned,
and vectorized-and-aligned) that all compute `out[i] = a[i] * 2 + 1`. Confirm
they agree, then look at the alignment that separates them.

### **[📊 Benchmark & Profile](./benchmark_and_profile.md)**

Measure the difference. Benchmark all three variants, then use NSight Compute to
prove the aligned kernel issues vectorized loads while the under-aligned one
does not.

## Getting started

**Prerequisites:**

- SIMD and vectorization from
  [Puzzle 23](../puzzle_23/puzzle_23.md)
- GPU profiling from [Puzzle 30](../puzzle_30/puzzle_30.md)
- `LayoutTensor` load/store from earlier puzzles

**Hardware requirements:**

- The kernels run and verify on any supported GPU (NVIDIA, AMD, Apple).
- The vectorized-codegen story and the NSight Compute profiling section are
  NVIDIA-specific: the `ld.global.nc.v4` instruction and `ncu` metrics are
  CUDA concepts.

**Learning outcome:** Treat alignment as a first-class performance lever: state
it deliberately at every vectorized memory access instead of hoping the compiler
infers it.
