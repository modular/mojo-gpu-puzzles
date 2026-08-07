# Aligned Load & Store

## The exercise

You'll implement three kernels that compute the same memory-bound map,
`out[i] = a[i] * 2 + 1`, over a 1M-element `float32` buffer. They differ only in
how they touch memory:

1. `scalar_kernel`: one element per thread with no vectorization. This is the baseline.
2. `unaligned_kernel`: vectorized by `SIMD_WIDTH`, but the access alignment
   is *under-stated* (scalar alignment), so the compiler emits scalar loads.
3. `aligned_kernel`: the same vectorized kernel, with the alignment
   communicated, so the compiler emits a single vectorized load/store per chunk.

All three produce identical output. The point of the puzzle is everything that
happens *after* you confirm that.

```mojo
{{#include ../../../problems/p35/p35.mojo:scalar_kernel}}
```

```mojo
{{#include ../../../problems/p35/p35.mojo:unaligned_kernel}}
```

```mojo
{{#include ../../../problems/p35/p35.mojo:aligned_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p35/p35.mojo" class="filename">View full file: problems/p35/p35.mojo</a>

## Key API

The file defines two alignment constants up front:

```mojo
comptime VEC_ALIGN = align_of[SIMD[dtype, SIMD_WIDTH]]()   # 16 bytes for float32x4
comptime SCALAR_ALIGN = align_of[dtype]()                  # 4 bytes
```

The vectorized kernels use `LayoutTensor`'s `load` and `store`, whose alignment
you set explicitly:

```mojo
# Under-stated: compiler can't prove 16-byte alignment -> scalar codegen
var v = a_lt.load[width=SIMD_WIDTH, load_alignment=SCALAR_ALIGN](Index(base))
out_lt.store[width=SIMD_WIDTH, store_alignment=SCALAR_ALIGN](Index(base), v * SCALE + BIAS)

# Aligned: 16-byte alignment -> ld.global.nc.v4 / st.global.v4
var v = a_lt.aligned_load[width=SIMD_WIDTH](Index(base))   # == load[..., load_alignment=VEC_ALIGN]
out_lt.store[width=SIMD_WIDTH, store_alignment=VEC_ALIGN](Index(base), v * SCALE + BIAS)
```

`aligned_load[w]` is the convenience wrapper: it picks
`align_of[SIMD[dtype, w]]()` for you. `aligned_store` exists too; here we pass
`store_alignment=VEC_ALIGN` explicitly so the contrast with the unaligned kernel
is a single changed value.

## Running it

```bash
pixi run mojo solutions/p35/p35.mojo --simple
pixi run mojo solutions/p35/p35.mojo --unaligned
pixi run mojo solutions/p35/p35.mojo --aligned
```

Each prints `<variant> kernel: passed`, confirming all three are correct. The
performance difference is the subject of the
[next section](./benchmark_and_profile.md).

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

- The data is already aligned. You are not moving the pointer or padding the
  buffer. Device allocations come back aligned, and each thread's `base` is a
  multiple of `SIMD_WIDTH`, so every access is on a 16-byte boundary. The only
  thing that changes between the unaligned and aligned kernels is the alignment
  *value you pass* to `load`/`store`.
- Guard the tail: each vectorized thread handles `SIMD_WIDTH` elements, so
  guard with `if base + SIMD_WIDTH <= size:` to avoid reading past the end.
- `aligned_load[w]` is the same as
  `load[w, load_alignment=align_of[SIMD[dtype, w]]()]`. Use whichever reads
  more clearly.
- Don't expect a wall-clock gap on a tiny input or on a non-NVIDIA GPU. The
  codegen difference instead shows up in NSight Compute's instruction/sector metrics.
  We'll cover this in the next section.

</div>
</details>

## Solution

<details class="solution-details">
<summary><strong>Complete solution</strong></summary>

The three kernels are identical in result and differ only in how they touch
memory.

```mojo
{{#include ../../../solutions/p35/p35.mojo:scalar_kernel_solution}}
```

```mojo
{{#include ../../../solutions/p35/p35.mojo:unaligned_kernel_solution}}
```

```mojo
{{#include ../../../solutions/p35/p35.mojo:aligned_kernel_solution}}
```

**What separates them:**

- The scalar kernel issues one scalar load and one scalar store per thread.
  Alignment is irrelevant because there is no vector to align.
- The unaligned kernel asks for a `SIMD_WIDTH`-wide load but declares only
  `SCALAR_ALIGN` (4 bytes). The compiler cannot prove the 16-byte alignment a
  `.v4` instruction requires, so it lowers the vector access to a sequence of
  scalar `ld.global.nc.f32` / `st.global.f32` instructions. Correct, but it
  issues `SIMD_WIDTH`× the memory instructions of the aligned kernel.
- The aligned kernel passes `VEC_ALIGN` (16 bytes). Now the compiler emits a
  single `ld.global.nc.v4.f32` load and `st.global.v4.f32` store per chunk,
  one quarter of the memory instructions for the same work.

The takeaway: state the alignment explicitly. The data was aligned in
all three kernels; only the aligned kernel informed the compiler, and only the
aligned kernel got the vectorized instruction. The
[benchmark and profile section](./benchmark_and_profile.md) demonstrates this in numbers.

</details>
