from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext
from gpu.host.compile import get_gpu_target
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from utils import IndexList
from math import log2
from algorithm.functional import elementwise, vectorize
from sys import simdwidthof, argv, alignof
from testing import assert_equal
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: elementwise_add
alias SIZE = 1024
alias rank = 1
alias layout = Layout.row_major(SIZE)
alias dtype = DType.float32
alias SIMD_WIDTH = simdwidthof[dtype, target = get_gpu_target()]()


fn elementwise_add[
    layout: Layout, dtype: DType, simd_width: Int, rank: Int, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn add[
        simd_width: Int, rank: Int, alignment: Int = alignof[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        idx = indices[0]
        print("idx:", idx)
        # FILL IN (2 to 4 lines)

    elementwise[add, SIMD_WIDTH, target="gpu"](a.size(), ctx)


# ANCHOR_END: elementwise_add


# ANCHOR: tiled_elementwise_add
alias TILE_SIZE = 32


fn tiled_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn process_tiles[
        simd_width: Int, rank: Int, alignment: Int = alignof[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        print("tile_id:", tile_id)
        out_tile = output.tile[tile_size](tile_id)
        a_tile = a.tile[tile_size](tile_id)
        b_tile = b.tile[tile_size](tile_id)

        # FILL IN (6 lines at most)

    num_tiles = (size + tile_size - 1) // tile_size
    elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)


# ANCHOR_END: tiled_elementwise_add


# ANCHOR: manual_vectorized_tiled_elementwise_add
fn manual_vectorized_tiled_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size groups of simd_width elements
    alias chunk_size = tile_size * simd_width

    @parameter
    @always_inline
    fn process_manual_vectorized_tiles[
        num_threads_per_tile: Int, rank: Int, alignment: Int = alignof[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        print("tile_id:", tile_id)
        out_tile = output.tile[chunk_size](tile_id)
        a_tile = a.tile[chunk_size](tile_id)
        b_tile = b.tile[chunk_size](tile_id)

        # FILL IN (7 lines at most)

    # Number of tiles needed: each tile processes chunk_size elements
    num_tiles = (size + chunk_size - 1) // chunk_size
    elementwise[
        process_manual_vectorized_tiles, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: manual_vectorized_tiled_elementwise_add


# ANCHOR: vectorize_within_tiles_elementwise_add
fn vectorize_within_tiles_elementwise_add[
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    # Each tile contains tile_size elements (not SIMD groups)
    @parameter
    @always_inline
    fn process_tile_with_vectorize[
        num_threads_per_tile: Int, rank: Int, alignment: Int = alignof[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        tile_start = tile_id * tile_size
        tile_end = min(tile_start + tile_size, size)
        actual_tile_size = tile_end - tile_start
        print(
            "tile_id:",
            tile_id,
            "tile_start:",
            tile_start,
            "tile_end:",
            tile_end,
            "actual_tile_size:",
            actual_tile_size,
        )

        # FILL IN (9 lines at most)

    num_tiles = (size + tile_size - 1) // tile_size
    elementwise[
        process_tile_with_vectorize, num_threads_per_tile, target="gpu"
    ](num_tiles, ctx)


# ANCHOR_END: vectorize_within_tiles_elementwise_add


@parameter
@always_inline
fn benchmark_elementwise_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn elementwise_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = 2 * i
                b_host[i] = 2 * i + 1

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

        elementwise_add[layout, dtype, SIMD_WIDTH, rank, test_size](
            out_tensor, a_tensor, b_tensor, ctx
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[elementwise_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_tiled_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn tiled_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = 2 * i
                b_host[i] = 2 * i + 1

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

        tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[tiled_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_manual_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn manual_vectorized_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = 2 * i
                b_host[i] = 2 * i + 1

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

        manual_vectorized_tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[manual_vectorized_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn vectorized_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = 2 * i
                b_host[i] = 2 * i + 1

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())

        vectorize_within_tiles_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, test_size, tile_size
        ](out_tensor, a_tensor, b_tensor, ctx)
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[vectorized_workflow](bench_ctx)


def main():
    ctx = DeviceContext()
    out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
    a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
    b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
    expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i in range(SIZE):
            a_host[i] = 2 * i
            b_host[i] = 2 * i + 1
            expected[i] = a_host[i] + b_host[i]

    a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
    b_tensor = LayoutTensor[mut=False, dtype, layout](b.unsafe_ptr())

    ctx.synchronize()

    print("SIZE:", SIZE)
    print("simd_width:", SIMD_WIDTH)

    if len(argv()) != 2 or argv()[1] not in [
        "--elementwise",
        "--tiled",
        "--manual-vectorized",
        "--vectorized",
        "--benchmark",
    ]:
        raise Error(
            "Usage: --elementwise | --tiled | --manual-vectorized |"
            " --vectorized | --benchmark"
        )

    if argv()[1] == "--elementwise":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        elementwise_add[layout, dtype, SIMD_WIDTH, rank, SIZE](
            out_tensor, a_tensor, b_tensor, ctx
        )

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--tiled":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        tiled_elementwise_add[layout, dtype, SIMD_WIDTH, rank, SIZE, TILE_SIZE](
            out_tensor, a_tensor, b_tensor, ctx
        )

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--manual-vectorized":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        manual_vectorized_tiled_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--vectorized":
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        print("tile size:", TILE_SIZE)
        vectorize_within_tiles_elementwise_add[
            layout, dtype, SIMD_WIDTH, 1, rank, SIZE, TILE_SIZE
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])

    elif argv()[1] == "--benchmark":
        print("Running P21 GPU Benchmarks...")
        print("SIMD width:", SIMD_WIDTH)
        print("-" * 80)
        bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
        bench = Bench(bench_config)

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
