# Why Alignment Matters

## Wide memory transactions

GPUs do not move memory one scalar at a time. The memory subsystem is built
around wide transactions: a warp's loads are coalesced into cache-line-sized
transfers, and each thread can ask for up to 16 bytes (128 bits) in a single
instruction. A `float32x4` vector maps to one such instruction:

```text
ld.global.nc.v4.f32   {r0, r1, r2, r3}, [addr]   // one instruction, 16 bytes
```

The scalar equivalent does the same work in four instructions:

```text
ld.global.nc.f32   r0, [addr]
ld.global.nc.f32   r1, [addr+4]
ld.global.nc.f32   r2, [addr+8]
ld.global.nc.f32   r3, [addr+12]
```

Both read the same 16 bytes. But the scalar form issues 4× the instructions on
the load/store unit, and on a memory-bound kernel that directly limits how fast
you can stream data.

## The hardware requires alignment

A 128-bit load is only legal when the address is 16-byte aligned. If the
compiler cannot *prove* the access is aligned, it is not allowed to emit the
`.v4` instruction. It must fall back to the scalar sequence, which has no such
requirement. So the vectorized fast path is gated entirely on what the compiler
knows about alignment.

This is where the trap springs. Your data is almost always aligned in practice:
device allocations are returned 256-byte aligned, and indexing in multiples of
the vector width keeps every access on a 16-byte boundary. But the compiler
only knows what the *API* tells it. If you call a vectorized load and declare
only scalar alignment, the compiler assumes the worst and emits scalar loads,
even though the address would have been perfectly aligned at runtime.

## `align_of`: stating what you know

Mojo gives you the alignment of any type at compile time:

```mojo
align_of[DType.float32]()              # 4  (a single float)
align_of[SIMD[DType.float32, 4]]()     # 16 (a float32x4 vector)
```

The whole puzzle hinges on the gap between these two numbers:

- Tell a vectorized `load`/`store` that the alignment is 4 bytes
  (`align_of[dtype]()`) and the compiler cannot vectorize: scalar codegen.
- Tell it 16 bytes (`align_of[SIMD[dtype, width]]()`) and it emits the
  single `.v4` instruction.

Same data, same result, same SIMD width. The only difference is the number you
hand the API.

## Why this is easy to miss

Alignment bugs are invisible in three of the four ways you normally check a
kernel:

| Check | Catches a missing-alignment bug? |
| ----- | -------------------------------- |
| Correctness (does the output match?) | ❌ No: results are identical |
| Compiles cleanly? | ❌ No: both forms are valid |
| Wall-clock on a small input? | ❌ Often not: launch overhead hides it |
| Profiler (instruction mix / sectors) | ✅ Yes |

That is why the [next section](./aligned_load_store.md) builds three kernels
that are provably equivalent, and the
[profiling section](./benchmark_and_profile.md) reaches for NSight Compute
rather than trusting timing alone.
