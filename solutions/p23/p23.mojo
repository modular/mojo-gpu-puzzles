from std.gpu import thread_idx, block_dim, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.host.compile import get_gpu_target
from layout import TileTensor, LayoutTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.utils import Index, IndexList
from std.math import log2
from std.algorithm.functional import elementwise, vectorize
from std.sys import simd_width_of, argv, align_of
from std.testing import assert_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

comptime SIZE = 1024
comptime rank = 1
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime dtype = DType.float32
comptime SIMD_WIDTH = simd_width_of[dtype, target=get_gpu_target()]()


# ANCHOR: elementwise_add_solution
def elementwise_add[
    LayoutT: TensorLayout, dtype: DType, simd_width: Int, rank: Int, size: Int
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    def add[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        var idx = indices[0]
        # Convert inside GPU kernel to avoid host-captured LayoutTensor issues
        var a_lt = a.to_layout_tensor()
        var b_lt = b.to_layout_tensor()
        var out_lt = output.to_layout_tensor()
        # Note: This is thread-local SIMD - each thread processes its own vector of data
        # we'll later better see this hierarchy in Mojo:
        # SIMD within threads, warp across threads, block across warps
        var a_simd = a_lt.aligned_load[width=simd_width](Index(idx))
        var b_simd = b_lt.aligned_load[width=simd_width](Index(idx))
        var ret = a_simd + b_simd
        out_lt.store[simd_width](Index(idx), ret)

    elementwise[add, SIMD_WIDTH, target="gpu"](size, ctx)


# ANCHOR_END: elementwise_add_solution


# ANCHOR: tiled_elementwise_add_solution
comptime TILE_SIZE = 32


def tiled_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    def process_tiles[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        var tile_id = indices[0]

        var output_tile = output.tile[tile_size](tile_id).to_layout_tensor()
        var a_tile = a.tile[tile_size](tile_id).to_layout_tensor()
        var b_tile = b.tile[tile_size](tile_id).to_layout_tensor()

        comptime for i in range(tile_size):
            var a_vec = a_tile.aligned_load[width=simd_width](Index(i))
            var b_vec = b_tile.aligned_load[width=simd_width](Index(i))
            var ret = a_vec + b_vec
            output_tile.store[simd_width](Index(i), ret)

    var num_tiles = (size + tile_size - 1) // tile_size
    elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)


# ANCHOR_END: tiled_elementwise_add_solution


# ANCHOR: manual_vectorized_tiled_elementwise_add_solution
def manual_vectorized_tiled_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size groups of simd_width elements
    comptime chunk_size = tile_size * simd_width

    @parameter
    @always_inline
    def process_manual_vectorized_tiles[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        var tile_id = indices[0]
        # Convert inside GPU kernel to avoid host-captured LayoutTensor issues
        var a_lt = a.to_layout_tensor()
        var b_lt = b.to_layout_tensor()
        var out_lt = output.to_layout_tensor()

        comptime for i in range(tile_size):
            var global_start = tile_id * chunk_size + i * simd_width

            var a_vec = a_lt.aligned_load[width=simd_width](Index(global_start))
            var b_vec = b_lt.aligned_load[width=simd_width](Index(global_start))
            var ret = a_vec + b_vec
            out_lt.store[simd_width](Index(global_start), ret)

    # Number of tiles needed: each tile processes chunk_size elements
    var num_tiles = (size + chunk_size - 1) // chunk_size
    elementwise[
        process_manual_vectorized_tiles, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: manual_vectorized_tiled_elementwise_add_solution


# ANCHOR: vectorize_within_tiles_elementwise_add_solution
def vectorize_within_tiles_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size elements (not SIMD groups)
    @parameter
    @always_inline
    def process_tile_with_vectorize[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        var tile_id = indices[0]
        var tile_start = tile_id * tile_size
        var tile_end = min(tile_start + tile_size, size)
        var actual_tile_size = tile_end - tile_start
        # Convert inside GPU kernel to avoid host-captured LayoutTensor issues
        var a_lt = a.to_layout_tensor()
        var b_lt = b.to_layout_tensor()
        var out_lt = output.to_layout_tensor()

        def vectorized_add[
            width: Int
        ](i: Int) unified {read tile_start, read a_lt, read b_lt, mut out_lt}:
            var global_idx = tile_start + i
            if global_idx + width <= size:
                var a_vec = a_lt.aligned_load[width](Index(global_idx))
                var b_vec = b_lt.aligned_load[width](Index(global_idx))
                var result = a_vec + b_vec
                out_lt.store[width](Index(global_idx), result)

        # Use vectorize within each tile
        vectorize[simd_width](actual_tile_size, vectorized_add)

    var num_tiles = (size + tile_size - 1) // tile_size
    elementwise[
        process_tile_with_vectorize, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: vectorize_within_tiles_elementwise_add_solution


@parameter
@always_inline
def benchmark_elementwise_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](a, bench_layout)
    var b_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](b_buf, bench_layout)
    var out_tensor = TileTensor[mut=True, dtype, BenchLayoutType, MutAnyOrigin](
        out, bench_layout
    )

    @parameter
    @always_inline
    def elementwise_workflow(ctx: DeviceContext) raises:
        elementwise_add[BenchLayoutType, dtype, SIMD_WIDTH, rank, test_size](
            out_tensor, a_tensor, b_tensor, ctx
        )

    b.iter_custom[elementwise_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
def benchmark_tiled_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](a, bench_layout)
    var b_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](b_buf, bench_layout)
    var out_tensor = TileTensor[mut=True, dtype, BenchLayoutType, MutAnyOrigin](
        out, bench_layout
    )

    @parameter
    @always_inline
    def tiled_workflow(ctx: DeviceContext) raises:
        tiled_elementwise_add[
            BenchLayoutType, dtype, SIMD_WIDTH, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[tiled_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
def benchmark_manual_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](a, bench_layout)
    var b_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](b_buf, bench_layout)
    var out_tensor = TileTensor[mut=True, dtype, BenchLayoutType, MutAnyOrigin](
        out, bench_layout
    )

    @parameter
    @always_inline
    def manual_vectorized_workflow(ctx: DeviceContext) raises:
        manual_vectorized_tiled_elementwise_add[
            BenchLayoutType, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[manual_vectorized_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
def benchmark_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](a, bench_layout)
    var b_tensor = TileTensor[
        mut=False, dtype, BenchLayoutType, ImmutAnyOrigin
    ](b_buf, bench_layout)
    var out_tensor = TileTensor[mut=True, dtype, BenchLayoutType, MutAnyOrigin](
        out, bench_layout
    )

    @parameter
    @always_inline
    def vectorized_workflow(ctx: DeviceContext) raises:
        vectorize_within_tiles_elementwise_add[
            BenchLayoutType, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    b.iter_custom[vectorized_workflow](bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


def main() raises:
    var ctx = DeviceContext()
    var out = ctx.enqueue_create_buffer[dtype](SIZE)
    out.enqueue_fill(0)
    var a = ctx.enqueue_create_buffer[dtype](SIZE)
    a.enqueue_fill(0)
    var b = ctx.enqueue_create_buffer[dtype](SIZE)
    b.enqueue_fill(0)
    var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
    expected.enqueue_fill(0)

    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i in range(SIZE):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)
            expected[i] = a_host[i] + b_host[i]

    var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
        a, layout
    )
    var b_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
        b, layout
    )

    ctx.synchronize()

    print("SIZE:", SIZE)
    print("simd_width:", SIMD_WIDTH)

    if argv()[1] == "--elementwise":
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        elementwise_add[LayoutType, dtype, SIMD_WIDTH, rank, SIZE](
            out_tensor, a_tensor, b_tensor, ctx
        )

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--tiled":
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        print("tile size:", TILE_SIZE)
        tiled_elementwise_add[
            LayoutType, dtype, SIMD_WIDTH, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--manual-vectorized":
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        print("tile size:", TILE_SIZE)
        manual_vectorized_tiled_elementwise_add[
            LayoutType, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--vectorized":
        var out_tensor = TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin](
            out, layout
        )
        print("tile size:", TILE_SIZE)
        vectorize_within_tiles_elementwise_add[
            LayoutType, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--benchmark":
        print("Running P21 GPU Benchmarks...")
        print("SIMD width:", SIMD_WIDTH)
        print("-" * 80)
        var bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
        var bench = Bench(bench_config.copy())

        print("Testing SIZE=16, TILE=4")
        bench.bench_function[benchmark_elementwise_parameterized[16, 4]](
            BenchId("elementwise_16_4")
        )
        bench.bench_function[benchmark_tiled_parameterized[16, 4]](
            BenchId("tiled_16_4")
        )
        bench.bench_function[benchmark_manual_vectorized_parameterized[16, 4]](
            BenchId("manual_vectorized_16_4")
        )
        bench.bench_function[benchmark_vectorized_parameterized[16, 4]](
            BenchId("vectorized_16_4")
        )

        print("-" * 80)
        print("Testing SIZE=128, TILE=16")
        bench.bench_function[benchmark_elementwise_parameterized[128, 16]](
            BenchId("elementwise_128_16")
        )
        bench.bench_function[benchmark_tiled_parameterized[128, 16]](
            BenchId("tiled_128_16")
        )
        bench.bench_function[
            benchmark_manual_vectorized_parameterized[128, 16]
        ](BenchId("manual_vectorized_128_16"))

        print("-" * 80)
        print("Testing SIZE=128, TILE=16, Vectorize within tiles")
        bench.bench_function[benchmark_vectorized_parameterized[128, 16]](
            BenchId("vectorized_128_16")
        )

        print("-" * 80)
        print("Testing SIZE=1048576 (1M), TILE=1024")
        bench.bench_function[
            benchmark_elementwise_parameterized[1048576, 1024]
        ](BenchId("elementwise_1M_1024"))
        bench.bench_function[benchmark_tiled_parameterized[1048576, 1024]](
            BenchId("tiled_1M_1024")
        )
        bench.bench_function[
            benchmark_manual_vectorized_parameterized[1048576, 1024]
        ](BenchId("manual_vectorized_1M_1024"))
        bench.bench_function[benchmark_vectorized_parameterized[1048576, 1024]](
            BenchId("vectorized_1M_1024")
        )

        print(bench)
        print("Benchmarks completed!")

    else:
        print(
            "Usage: --elementwise | --tiled | --manual-vectorized |"
            " --vectorized | --benchmark"
        )
