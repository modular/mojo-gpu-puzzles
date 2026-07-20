# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from std.gpu import thread_idx, block_dim, block_idx
from std.gpu.host import DeviceContext
from std.gpu.host.compile import get_gpu_target
from layout import TileTensor
from layout.tile_layout import row_major
from std.utils import Index
from std.sys import argv, align_of, simd_width_of
from std.testing import assert_almost_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# 1M float32 elements: large enough to be memory-bandwidth bound, so the
# load/store path is what the benchmark actually measures.
comptime SIZE = 1024 * 1024
comptime TPB = 256
comptime dtype = DType.float32
# On NVIDIA, simd_width_of[float32] == 4, i.e. a 16-byte / 128-bit vector. That
# is exactly the width that lowers to a single `ld.global.nc.v4.f32` *when the
# compiler knows the access is 16-byte aligned*.
comptime SIMD_WIDTH = simd_width_of[dtype, target=get_gpu_target()]()
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)

# The kernels all compute the same memory-bound map: out[i] = a[i] * 2 + 1.
comptime SCALE = Scalar[dtype](2)
comptime BIAS = Scalar[dtype](1)

# The natural alignment of a full SIMD vector: 16 bytes for float32x4. Telling
# the compiler this is the whole game.
comptime VEC_ALIGN = align_of[SIMD[dtype, SIMD_WIDTH]]()
comptime SCALAR_ALIGN = align_of[dtype]()


# ANCHOR: scalar_kernel
def scalar_kernel(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    """One element per thread. No vectorization, so alignment is irrelevant.

    This is the baseline: each thread issues a scalar load and a scalar store.
    """
    var i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (1-2 lines): guard `i < size`, then `output[i] = a[i] * SCALE + BIAS`.


# ANCHOR_END: scalar_kernel


# ANCHOR: unaligned_kernel
def unaligned_kernel(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    """Vectorized by SIMD_WIDTH, but the access alignment is *under-stated*.

    The data is naturally 16-byte aligned, but `load`/`store` are told only the
    scalar alignment (`align_of[dtype]()` == 4 bytes). The compiler cannot prove
    the access is 16-byte aligned, so it falls back to scalar
    `ld.global.nc.f32` / `st.global.f32` instructions instead of the vectorized
    `.v4` form. The alignment trap: correct results, but lost bandwidth.
    """
    var a_lt = a.to_layout_tensor()
    var out_lt = output.to_layout_tensor()

    # Each thread owns one SIMD_WIDTH-wide chunk.
    var base = (block_dim.x * block_idx.x + thread_idx.x) * SIMD_WIDTH
    # FILL ME IN (~4 lines): guard `base + SIMD_WIDTH <= size`, then load a SIMD_WIDTH-wide vector with `load[width=SIMD_WIDTH, load_alignment=SCALAR_ALIGN](Index(base))` and store `v * SCALE + BIAS` with `store_alignment=SCALAR_ALIGN`.


# ANCHOR_END: unaligned_kernel


# ANCHOR: aligned_kernel
def aligned_kernel(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    """Same vectorized kernel, but the access alignment is communicated.

    Passing `VEC_ALIGN` (`align_of[SIMD[dtype, SIMD_WIDTH]]()` == 16 bytes for
    float32x4) lets the compiler emit a single vectorized `ld.global.nc.v4.f32`
    load and `st.global.v4.f32` store per chunk. `aligned_load` is the
    convenience wrapper that picks this alignment for you. Identical output to
    the unaligned kernel — only the codegen (and the bandwidth) changes.
    """
    var a_lt = a.to_layout_tensor()
    var out_lt = output.to_layout_tensor()

    var base = (block_dim.x * block_idx.x + thread_idx.x) * SIMD_WIDTH
    # FILL ME IN (~4 lines): guard `base + SIMD_WIDTH <= size`, then load with the *correct* alignment via `a_lt.aligned_load[width=SIMD_WIDTH](Index(base))` and store `v * SCALE + BIAS` with `store_alignment=VEC_ALIGN`.


# ANCHOR_END: aligned_kernel


def scalar_blocks(size: Int) -> Int:
    return (size + TPB - 1) // TPB


def vector_blocks(size: Int) -> Int:
    # One thread per SIMD_WIDTH-wide chunk.
    var threads = (size + SIMD_WIDTH - 1) // SIMD_WIDTH
    return (threads + TPB - 1) // TPB


# ---------------------------------------------------------------------------- #
# Correctness                                                                  #
# ---------------------------------------------------------------------------- #


def test_scalar() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i % 97)

        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[scalar_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(scalar_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        with out.map_to_host() as result:
            for i in range(SIZE):
                assert_almost_equal(
                    result[i], Scalar[dtype](i % 97) * SCALE + BIAS, atol=1e-5
                )
    print("scalar kernel: passed")


def test_unaligned() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i % 97)

        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[unaligned_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(vector_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        with out.map_to_host() as result:
            for i in range(SIZE):
                assert_almost_equal(
                    result[i], Scalar[dtype](i % 97) * SCALE + BIAS, atol=1e-5
                )
    print("unaligned kernel: passed")


def test_aligned() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i % 97)

        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[aligned_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(vector_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        with out.map_to_host() as result:
            for i in range(SIZE):
                assert_almost_equal(
                    result[i], Scalar[dtype](i % 97) * SCALE + BIAS, atol=1e-5
                )
    print("aligned kernel: passed")


# ---------------------------------------------------------------------------- #
# Benchmarks                                                                   #
# ---------------------------------------------------------------------------- #


@parameter
@always_inline
def benchmark_scalar(mut b: Bencher) raises:
    @parameter
    @always_inline
    def workflow(ctx: DeviceContext) raises:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[scalar_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(scalar_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[workflow](bench_ctx)


@parameter
@always_inline
def benchmark_unaligned(mut b: Bencher) raises:
    @parameter
    @always_inline
    def workflow(ctx: DeviceContext) raises:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[unaligned_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(vector_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[workflow](bench_ctx)


@parameter
@always_inline
def benchmark_aligned(mut b: Bencher) raises:
    @parameter
    @always_inline
    def workflow(ctx: DeviceContext) raises:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            a, layout
        )
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        ctx.enqueue_function[aligned_kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=(vector_blocks(SIZE), 1),
            block_dim=(TPB, 1),
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[workflow](bench_ctx)


def main() raises:
    if len(argv()) < 2:
        print(
            "Usage: mojo p35.mojo [--simple] [--unaligned] [--aligned]"
            " [--benchmark]"
        )
        return

    print("SIZE:", SIZE, "SIMD_WIDTH:", SIMD_WIDTH)

    if argv()[1] == "--simple":
        test_scalar()
        print("Puzzle 35 complete ✅")
    elif argv()[1] == "--unaligned":
        test_unaligned()
        print("Puzzle 35 complete ✅")
    elif argv()[1] == "--aligned":
        test_aligned()
        print("Puzzle 35 complete ✅")
    elif argv()[1] == "--benchmark":
        print("Benchmarking alignment impact on load/store...")
        print("-" * 60)
        var bench = Bench(BenchConfig(max_iters=100, num_warmup_iters=10))

        print("\nScalar (one element per thread):")
        bench.bench_function[benchmark_scalar](BenchId("scalar"))

        print("\nVectorized, under-stated alignment (scalar codegen):")
        bench.bench_function[benchmark_unaligned](BenchId("unaligned"))

        print("\nVectorized, aligned (ld.global.nc.v4 codegen):")
        bench.bench_function[benchmark_aligned](BenchId("aligned"))

        bench.dump_report()
        print("\nProfile with NSight Compute to confirm the codegen change!")
    else:
        print("Unknown argument:", argv()[1])
        print(
            "Usage: mojo p35.mojo [--simple] [--unaligned] [--aligned]"
            " [--benchmark]"
        )
