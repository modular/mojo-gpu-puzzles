from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, barrier, lane_id
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.memory import AddressSpace
from algorithm.functional import elementwise
from layout import Layout, LayoutTensor
from utils import IndexList
from sys import argv, simd_width_of, size_of, align_of
from testing import assert_equal
from random import random_float64
from benchmark import (
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

alias SIZE = WARP_SIZE
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (WARP_SIZE, 1)  # optimal choice for warp kernel
alias dtype = DType.float32
alias SIMD_WIDTH = simd_width_of[dtype]()
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)


# ANCHOR: traditional_approach_from_p12
fn traditional_dot_product_p12_style[
    in_layout: Layout, out_layout: Layout, size: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
):
    """
    This is the complex approach from p12_layout_tensor.mojo - kept for comparison.
    """
    shared = LayoutTensor[
        dtype,
        Layout.row_major(WARP_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = (a[global_i] * b[global_i]).reduce_add()
    else:
        shared[local_i] = 0.0

    barrier()

    stride = WARP_SIZE // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        output[global_i // WARP_SIZE] = shared[0]


# ANCHOR_END: traditional_approach_from_p10


# ANCHOR: simple_warp_kernel_solution
fn simple_warp_dot_product[
    in_layout: Layout, out_layout: Layout, size: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Each thread computes one partial product using vectorized approach as values in Mojo are SIMD based
    var partial_product: Scalar[dtype] = 0
    if global_i < size:
        partial_product = (a[global_i] * b[global_i]).reduce_add()

    # warp_sum() replaces all the shared memory + barriers + tree reduction
    total = warp_sum(partial_product)

    # Only lane 0 writes the result (all lanes have the same total)
    if lane_id() == 0:
        output[global_i // WARP_SIZE] = total


# ANCHOR_END: simple_warp_kernel_solution


# ANCHOR: functional_warp_approach_solution
fn functional_warp_dot_product[
    layout: Layout,
    out_layout: Layout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn compute_dot_product[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        idx = indices[0]

        # Each thread computes one partial product
        var partial_product: Scalar[dtype] = 0.0
        if idx < size:
            a_val = a.load[1](idx, 0)
            b_val = b.load[1](idx, 0)
            partial_product = a_val * b_val
        else:
            partial_product = 0.0

        # Warp magic - combines all WARP_SIZE partial products!
        total = warp_sum(partial_product)

        # Only lane 0 writes the result (all lanes have the same total)
        if lane_id() == 0:
            output.store[1](idx // WARP_SIZE, 0, total)

    # Launch exactly size == WARP_SIZE threads (one warp) to process all elements
    elementwise[compute_dot_product, 1, target="gpu"](size, ctx)


# ANCHOR_END: functional_warp_approach_solution


fn expected_output[
    dtype: DType, n_warps: Int
](
    expected: HostBuffer[dtype],
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
) raises:
    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i_warp in range(n_warps):
            i_warp_in_buff = WARP_SIZE * i_warp
            var warp_sum: Scalar[dtype] = 0
            for i in range(WARP_SIZE):
                warp_sum += (
                    a_host[i_warp_in_buff + i] * b_host[i_warp_in_buff + i]
                )
            expected[i_warp] = warp_sum


fn rand_int[
    dtype: DType, size: Int
](buff: DeviceBuffer[dtype], min: Int = 0, max: Int = 100) raises:
    with buff.map_to_host() as buff_host:
        for i in range(size):
            buff_host[i] = Int(random_float64(min, max))


fn check_result[
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
fn benchmark_simple_warp_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    alias n_warps = test_size // WARP_SIZE
    alias in_layout = Layout.row_major(test_size)
    alias out_layout = Layout.row_major(n_warps)
    alias n_threads = WARP_SIZE
    alias n_blocks = (ceildiv(test_size, n_threads), 1)

    bench_ctx = DeviceContext()

    out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
    b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b)
    out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)

    @parameter
    @always_inline
    fn traditional_workflow(ctx: DeviceContext) raises:
        alias kernel = simple_warp_dot_product[in_layout, out_layout, test_size]
        ctx.enqueue_function_checked[kernel, kernel](
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
fn benchmark_functional_warp_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    alias n_warps = test_size // WARP_SIZE
    alias in_layout = Layout.row_major(test_size)
    alias out_layout = Layout.row_major(n_warps)

    bench_ctx = DeviceContext()

    out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
    b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b)
    out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)

    @parameter
    @always_inline
    fn functional_warp_workflow(ctx: DeviceContext) raises:
        functional_warp_dot_product[
            in_layout, out_layout, dtype, SIMD_WIDTH, 1, test_size
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher.iter_custom[functional_warp_workflow](bench_ctx)
    check_result[dtype, n_warps](out, expected)
    keep(out.unsafe_ptr())
    keep(a.unsafe_ptr())
    keep(b.unsafe_ptr())
    bench_ctx.synchronize()


@parameter
@always_inline
fn benchmark_traditional_parameterized[
    test_size: Int
](mut bencher: Bencher) raises:
    alias n_warps = test_size // WARP_SIZE
    alias in_layout = Layout.row_major(test_size)
    alias out_layout = Layout.row_major(n_warps)
    alias n_blocks = (ceildiv(test_size, WARP_SIZE), 1)

    bench_ctx = DeviceContext()

    out = bench_ctx.enqueue_create_buffer[dtype](n_warps)
    out.enqueue_fill(0)
    a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    b = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b.enqueue_fill(0)
    expected = bench_ctx.enqueue_create_host_buffer[dtype](n_warps)
    expected.enqueue_fill(0)

    rand_int[dtype, test_size](a)
    rand_int[dtype, test_size](b)
    expected_output[dtype, n_warps](expected, a, b)

    a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
    b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b)
    out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)

    @parameter
    @always_inline
    fn traditional_workflow(ctx: DeviceContext) raises:
        ctx.enqueue_function_checked[
            traditional_dot_product_p12_style[in_layout, out_layout, test_size],
            traditional_dot_product_p12_style[in_layout, out_layout, test_size],
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


def main():
    if argv()[1] != "--benchmark":
        print("SIZE:", SIZE)
        print("WARP_SIZE:", WARP_SIZE)
        print("SIMD_WIDTH:", SIMD_WIDTH)
        alias n_warps = SIZE // WARP_SIZE
        with DeviceContext() as ctx:
            out = ctx.enqueue_create_buffer[dtype](n_warps)
            out.enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](SIZE)
            a.enqueue_fill(0)
            b = ctx.enqueue_create_buffer[dtype](SIZE)
            b.enqueue_fill(0)
            expected = ctx.enqueue_create_host_buffer[dtype](n_warps)
            expected.enqueue_fill(0)

            out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
            a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
            b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b)

            with a.map_to_host() as a_host, b.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = i
                    b_host[i] = i

            if argv()[1] == "--traditional":
                ctx.enqueue_function_checked[
                    traditional_dot_product_p12_style[
                        in_layout, out_layout, SIZE
                    ],
                    traditional_dot_product_p12_style[
                        in_layout, out_layout, SIZE
                    ],
                ](
                    out_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )
            elif argv()[1] == "--kernel":
                ctx.enqueue_function_checked[
                    simple_warp_dot_product[in_layout, out_layout, SIZE],
                    simple_warp_dot_product[in_layout, out_layout, SIZE],
                ](
                    out_tensor,
                    a_tensor,
                    b_tensor,
                    grid_dim=BLOCKS_PER_GRID,
                    block_dim=THREADS_PER_BLOCK,
                )
            elif argv()[1] == "--functional":
                functional_warp_dot_product[
                    in_layout, out_layout, dtype, SIMD_WIDTH, 1, SIZE
                ](out_tensor, a_tensor, b_tensor, ctx)
            expected_output[dtype, n_warps](expected, a, b)
            check_result[dtype, n_warps, True](out, expected)
            ctx.synchronize()
    elif argv()[1] == "--benchmark":
        print("-" * 80)
        bench_config = BenchConfig(max_iters=100, num_warmup_iters=1)
        bench = Bench(bench_config.copy())

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
