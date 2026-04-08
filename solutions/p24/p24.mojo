from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, barrier, lane_id
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu.primitives.warp import sum as warp_sum, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.algorithm.functional import elementwise
from layout import TileTensor, LayoutTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.utils import Index, IndexList
from std.sys import argv, simd_width_of, align_of
from std.testing import assert_equal
from std.random import random_float64
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    keep,
    ThroughputMeasure,
    BenchMetric,
    BenchmarkInfo,
    run,
)

comptime SIZE = WARP_SIZE
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)  # optimal choice for warp kernel
comptime dtype = DType.float32
comptime SIMD_WIDTH = simd_width_of[dtype]()
comptime in_layout = row_major[SIZE]()
comptime out_layout = row_major[1]()
comptime InLayout = type_of(in_layout)
comptime OutLayout = type_of(out_layout)


# ANCHOR: traditional_approach_from_p12
def traditional_dot_product_p12_style[
    InLayoutT: TensorLayout, OutLayoutT: TensorLayout, size: Int
](
    output: TileTensor[mut=True, dtype, OutLayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
):
    """
    This is the complex approach from p12_layout_tensor.mojo - kept for comparison.
    """
    var a_lt = a.to_layout_tensor()
    var b_lt = b.to_layout_tensor()
    var out_lt = output.to_layout_tensor()
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[WARP_SIZE]())
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = rebind[Scalar[dtype]](a_lt[global_i]) * rebind[
            Scalar[dtype]
        ](b_lt[global_i])
    else:
        shared[local_i] = 0.0

    barrier()

    var stride = WARP_SIZE // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        out_lt.store[1](Index(global_i // WARP_SIZE), shared[0])


# ANCHOR_END: traditional_approach_from_p12


# ANCHOR: simple_warp_kernel_solution
def simple_warp_dot_product[
    InLayoutT: TensorLayout, OutLayoutT: TensorLayout, size: Int
](
    output: TileTensor[mut=True, dtype, OutLayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
):
    var a_lt = a.to_layout_tensor()
    var b_lt = b.to_layout_tensor()
    var out_lt = output.to_layout_tensor()
    var global_i = block_dim.x * block_idx.x + thread_idx.x

    # Each thread computes one partial product using vectorized approach as values in Mojo are SIMD based
    var partial_product: Scalar[dtype] = 0
    if global_i < size:
        partial_product = rebind[Scalar[dtype]](a_lt[global_i]) * rebind[
            Scalar[dtype]
        ](b_lt[global_i])

    # warp_sum() replaces all the shared memory + barriers + tree reduction
    var total = warp_sum(partial_product)

    # Only lane 0 writes the result (all lanes have the same total)
    if lane_id() == 0:
        out_lt.store[1](Index(global_i // WARP_SIZE), total)


# ANCHOR_END: simple_warp_kernel_solution


# ANCHOR: functional_warp_approach_solution
def functional_warp_dot_product[
    InLayoutT: TensorLayout,
    OutLayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
](
    output: TileTensor[mut=True, dtype, OutLayoutT, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, InLayoutT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    def compute_dot_product[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        var idx = indices[0]
        # Convert inside GPU kernel to avoid host-captured LayoutTensor issues
        var a_lt = a.to_layout_tensor()
        var b_lt = b.to_layout_tensor()
        var out_lt = output.to_layout_tensor()

        # Each thread computes one partial product
        var partial_product: Scalar[dtype] = 0.0
        if idx < size:
            var a_val = a_lt.load[1](Index(idx))
            var b_val = b_lt.load[1](Index(idx))
            partial_product = rebind[Scalar[dtype]](a_val) * rebind[
                Scalar[dtype]
            ](b_val)
        else:
            partial_product = 0.0

        # Warp magic - combines all WARP_SIZE partial products!
        var total = warp_sum(partial_product)

        # Only lane 0 writes the result (all lanes have the same total)
        if lane_id() == 0:
            out_lt.store[1](Index(idx // WARP_SIZE), total)

    # Launch exactly size == WARP_SIZE threads (one warp) to process all elements
    elementwise[compute_dot_product, 1, target="gpu"](size, ctx)


# ANCHOR_END: functional_warp_approach_solution


def expected_output[
    dtype: DType, n_warps: Int
](
    expected: HostBuffer[dtype],
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
) raises:
    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i_warp in range(n_warps):
            var i_warp_in_buff = WARP_SIZE * i_warp
            var warp_sum: Scalar[dtype] = 0
            for i in range(WARP_SIZE):
                warp_sum += (
                    a_host[i_warp_in_buff + i] * b_host[i_warp_in_buff + i]
                )
            expected[i_warp] = warp_sum


def rand_int[
    dtype: DType, size: Int
](buff: DeviceBuffer[dtype], min: Int = 0, max: Int = 100) raises:
    with buff.map_to_host() as buff_host:
        for i in range(size):
            buff_host[i] = Scalar[dtype](
                Int(random_float64(Float64(min), Float64(max)))
            )


def check_result[
    dtype: DType, size: Int, print_result: Bool = False
](actual: DeviceBuffer[dtype], expected: HostBuffer[dtype]) raises:
    with actual.map_to_host() as actual_host:
        if print_result:
            print("=== RESULT ===")
            print("actual:", actual_host)
            print("expected:", expected)
        for i in range(size):
            assert_equal(actual_host[i], expected[i])


@parameter
@always_inline
def benchmark_simple_warp_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    comptime n_warps = test_size // WARP_SIZE
    comptime bench_in_layout = row_major[test_size]()
    comptime bench_out_layout = row_major[n_warps]()
    comptime BenchInLayout = type_of(bench_in_layout)
    comptime BenchOutLayout = type_of(bench_out_layout)
    comptime n_threads = WARP_SIZE
    comptime n_blocks = (ceildiv(test_size, n_threads), 1)

    var bench_ctx = DeviceContext()

    var out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    var expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    var a_tensor = TileTensor[mut=False, dtype, BenchInLayout](
        a, bench_in_layout
    )
    var b_tensor = TileTensor[mut=False, dtype, BenchInLayout](
        b, bench_in_layout
    )
    var out_tensor = TileTensor[mut=True, dtype, BenchOutLayout](
        out, bench_out_layout
    )

    @parameter
    @always_inline
    def traditional_workflow(ctx: DeviceContext) raises:
        comptime kernel = simple_warp_dot_product[
            BenchInLayout, BenchOutLayout, test_size
        ]
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=n_blocks,
            block_dim=n_threads,
        )

    bencher.iter_custom[traditional_workflow](bench_ctx)
    check_result[dtype, n_warps](out, expected)
    keep(out.unsafe_ptr())
    keep(a.unsafe_ptr())
    keep(b.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
def benchmark_functional_warp_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    comptime n_warps = test_size // WARP_SIZE
    comptime bench_in_layout = row_major[test_size]()
    comptime bench_out_layout = row_major[n_warps]()
    comptime BenchInLayout = type_of(bench_in_layout)
    comptime BenchOutLayout = type_of(bench_out_layout)

    var bench_ctx = DeviceContext()

    var out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    var expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    var a_tensor = rebind[
        TileTensor[mut=False, dtype, BenchInLayout, ImmutAnyOrigin]
    ](TileTensor[mut=False, dtype, BenchInLayout](a, bench_in_layout))
    var b_tensor = rebind[
        TileTensor[mut=False, dtype, BenchInLayout, ImmutAnyOrigin]
    ](TileTensor[mut=False, dtype, BenchInLayout](b, bench_in_layout))
    var out_tensor = rebind[
        TileTensor[mut=True, dtype, BenchOutLayout, MutAnyOrigin]
    ](TileTensor[mut=True, dtype, BenchOutLayout](out, bench_out_layout))

    @parameter
    @always_inline
    def functional_warp_workflow(ctx: DeviceContext) raises:
        functional_warp_dot_product[
            BenchInLayout, BenchOutLayout, dtype, SIMD_WIDTH, 1, test_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher.iter_custom[functional_warp_workflow](bench_ctx)
    check_result[dtype, n_warps](out, expected)
    keep(out.unsafe_ptr())
    keep(a.unsafe_ptr())
    keep(b.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
def benchmark_traditional_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    comptime n_warps = test_size // WARP_SIZE
    comptime bench_in_layout = row_major[test_size]()
    comptime bench_out_layout = row_major[n_warps]()
    comptime BenchInLayout = type_of(bench_in_layout)
    comptime BenchOutLayout = type_of(bench_out_layout)
    comptime n_blocks = (ceildiv(test_size, WARP_SIZE), 1)

    var bench_ctx = DeviceContext()

    var out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    var expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    var a_tensor = TileTensor[mut=False, dtype, BenchInLayout](
        a, bench_in_layout
    )
    var b_tensor = TileTensor[mut=False, dtype, BenchInLayout](
        b, bench_in_layout
    )
    var out_tensor = TileTensor[mut=True, dtype, BenchOutLayout](
        out, bench_out_layout
    )

    @parameter
    @always_inline
    def traditional_workflow(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            traditional_dot_product_p12_style[
                BenchInLayout, BenchOutLayout, test_size
            ],
            traditional_dot_product_p12_style[
                BenchInLayout, BenchOutLayout, test_size
            ],
        ](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=n_blocks,
            block_dim=THREADS_PER_BLOCK,
        )

    bencher.iter_custom[traditional_workflow](bench_ctx)
    check_result[dtype, n_warps](out, expected)
    keep(out.unsafe_ptr())
    keep(a.unsafe_ptr())
    keep(b.unsafe_ptr())
    bench_ctx.synchronize()


def main() raises:
    if argv()[1] != "--benchmark":
        print("SIZE:", SIZE)
        print("WARP_SIZE:", WARP_SIZE)
        print("SIMD_WIDTH:", SIMD_WIDTH)
        comptime n_warps = SIZE // WARP_SIZE
        comptime main_out_layout = row_major[n_warps]()
        comptime MainOutLayout = type_of(main_out_layout)
        with DeviceContext() as ctx:
            var out = ctx.enqueue_create_buffer[dtype](n_warps)
            out.enqueue_fill(0)
            var a = ctx.enqueue_create_buffer[dtype](SIZE)
            a.enqueue_fill(0)
            var b = ctx.enqueue_create_buffer[dtype](SIZE)
            b.enqueue_fill(0)
            var expected = ctx.enqueue_create_host_buffer[dtype](n_warps)
            expected.enqueue_fill(0)

            var out_tensor = rebind[
                TileTensor[mut=True, dtype, MainOutLayout, MutAnyOrigin]
            ](TileTensor[mut=True, dtype, MainOutLayout](out, main_out_layout))
            var a_tensor = rebind[
                TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin]
            ](TileTensor[mut=False, dtype, InLayout](a, in_layout))
            var b_tensor = rebind[
                TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin]
            ](TileTensor[mut=False, dtype, InLayout](b, in_layout))

            with a.map_to_host() as a_host, b.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = Scalar[dtype](i)
                    b_host[i] = Scalar[dtype](i)

            if argv()[1] == "--traditional":
                ctx.enqueue_function[
                    traditional_dot_product_p12_style[
                        InLayout, MainOutLayout, SIZE
                    ],
                    traditional_dot_product_p12_style[
                        InLayout, MainOutLayout, SIZE
                    ],
                ](
                    out_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )
            elif argv()[1] == "--kernel":
                ctx.enqueue_function[
                    simple_warp_dot_product[InLayout, MainOutLayout, SIZE],
                    simple_warp_dot_product[InLayout, MainOutLayout, SIZE],
                ](
                    out_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )
            elif argv()[1] == "--functional":
                functional_warp_dot_product[
                    InLayout, MainOutLayout, dtype, SIMD_WIDTH, 1, SIZE
                ](out_tensor, a_tensor, b_tensor, ctx)
            expected_output[dtype, n_warps](expected, a, b)
            check_result[dtype, n_warps, True](out, expected)
            print("Puzzle 24 complete ✅")
            ctx.synchronize()
    elif argv()[1] == "--benchmark":
        print("-" * 80)
        var bench_config = BenchConfig(max_iters=100, num_warmup_iters=1)
        var bench = Bench(bench_config.copy())

        print("Testing SIZE=1 x WARP_SIZE, BLOCKS=1")
        bench.bench_function[benchmark_traditional_parameterized[WARP_SIZE]](
            BenchId("traditional_1x")
        )
        bench.bench_function[benchmark_simple_warp_parameterized[WARP_SIZE]](
            BenchId("simple_warp_1x")
        )
        bench.bench_function[
            benchmark_functional_warp_parameterized[WARP_SIZE]
        ](BenchId("functional_warp_1x"))

        print("-" * 80)
        print("Testing SIZE=4 x WARP_SIZE, BLOCKS=4")
        bench.bench_function[
            benchmark_traditional_parameterized[4 * WARP_SIZE]
        ](BenchId("traditional_4x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[4 * WARP_SIZE]
        ](BenchId("simple_warp_4x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[4 * WARP_SIZE]
        ](BenchId("functional_warp_4x"))

        print("-" * 80)
        print("Testing SIZE=32 x WARP_SIZE, BLOCKS=32")
        bench.bench_function[
            benchmark_traditional_parameterized[32 * WARP_SIZE]
        ](BenchId("traditional_32x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[32 * WARP_SIZE]
        ](BenchId("simple_warp_32x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[32 * WARP_SIZE]
        ](BenchId("functional_warp_32x"))

        print("-" * 80)
        print("Testing SIZE=256 x WARP_SIZE, BLOCKS=256")
        bench.bench_function[
            benchmark_traditional_parameterized[256 * WARP_SIZE]
        ](BenchId("traditional_256x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[256 * WARP_SIZE]
        ](BenchId("simple_warp_256x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[256 * WARP_SIZE]
        ](BenchId("functional_warp_256x"))

        print("-" * 80)
        print("Testing SIZE=2048 x WARP_SIZE, BLOCKS=2048")
        bench.bench_function[
            benchmark_traditional_parameterized[2048 * WARP_SIZE]
        ](BenchId("traditional_2048x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[2048 * WARP_SIZE]
        ](BenchId("simple_warp_2048x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[2048 * WARP_SIZE]
        ](BenchId("functional_warp_2048x"))

        print("-" * 80)
        print("Testing SIZE=16384 x WARP_SIZE, BLOCKS=16384 (Large Scale)")
        bench.bench_function[
            benchmark_traditional_parameterized[16384 * WARP_SIZE]
        ](BenchId("traditional_16384x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[16384 * WARP_SIZE]
        ](BenchId("simple_warp_16384x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[16384 * WARP_SIZE]
        ](BenchId("functional_warp_16384x"))

        print("-" * 80)
        print("Testing SIZE=65536 x WARP_SIZE, BLOCKS=65536 (Massive Scale)")
        bench.bench_function[
            benchmark_traditional_parameterized[65536 * WARP_SIZE]
        ](BenchId("traditional_65536x"))
        bench.bench_function[
            benchmark_simple_warp_parameterized[65536 * WARP_SIZE]
        ](BenchId("simple_warp_65536x"))
        bench.bench_function[
            benchmark_functional_warp_parameterized[65536 * WARP_SIZE]
        ](BenchId("functional_warp_65536x"))

        print(bench)
        print("Benchmarks completed!")
        print()
        print("WARP OPERATIONS PERFORMANCE ANALYSIS:")
        print(
            "   GPU Architecture: NVIDIA (WARP_SIZE=32) vs AMD (WARP_SIZE=64)"
        )
        print("   - 1,...,256 x WARP_SIZE: Grid size too small to benchmark")
        print("   - 2048 x WARP_SIZE: Warp primative benefits emerge")
        print("   - 16384 x WARP_SIZE: Large scale (512K-1M elements)")
        print("   - 65536 x WARP_SIZE: Massive scale (2M-4M elements)")
        print("   - Note: AMD GPUs process 2 x elements per warp vs NVIDIA!")
        print()
        print("   Expected Results at Large Scales:")
        print("   • Traditional: Slower due to more barrier overhead")
        print("   • Warp operations: Faster, scale better with problem size")
        print("   • Memory bandwidth becomes the limiting factor")
        return

    else:
        print("Usage: --traditional | --kernel | --functional | --benchmark")
        return
