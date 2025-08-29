from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, barrier, lane_id
from gpu.host import DeviceContext
from gpu.warp import sum as warp_sum, WARP_SIZE
from algorithm.functional import elementwise
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from utils import IndexList
from sys import argv, simdwidthof, sizeof, alignof
from testing import assert_equal
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

# ANCHOR: traditional_approach_from_p12
alias SIZE = WARP_SIZE
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (WARP_SIZE, 1)  # optimal choice for warp kernel
alias dtype = DType.float32
alias SIMD_WIDTH = simdwidthof[dtype]()
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)


fn traditional_dot_product_p12_style[
    in_layout: Layout, out_layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, in_layout],
):
    """
    This is the complex approach from p12_layout_tensor.mojo - kept for comparison.
    """
    shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = (a[global_i] * b[global_i]).reduce_add()
    else:
        shared[local_i] = 0.0

    barrier()

    stride = SIZE // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        output[0] = shared[0]


# ANCHOR_END: traditional_approach_from_p12

# ANCHOR: simple_warp_kernel
from gpu.warp import sum as warp_sum


fn simple_warp_dot_product[
    in_layout: Layout, out_layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, in_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL IN (6 lines at most)


# ANCHOR_END: simple_warp_kernel


# ANCHOR: functional_warp_approach
fn functional_warp_dot_product[
    layout: Layout, dtype: DType, simd_width: Int, rank: Int, size: Int
](
    output: LayoutTensor[
        mut=True, dtype, Layout.row_major(1), MutableAnyOrigin
    ],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    @always_inline
    fn compute_dot_product[
        simd_width: Int, rank: Int, alignment: Int = alignof[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        idx = indices[0]
        print("idx:", idx)
        # FILL IN (10 lines at most)

    # Launch exactly WARP_SIZE threads (one warp) to process all elements
    elementwise[compute_dot_product, 1, target="gpu"](WARP_SIZE, ctx)


# ANCHOR_END: functional_warp_approach


@parameter
@always_inline
fn benchmark_simple_warp_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn simple_warp_workflow(ctx: DeviceContext) raises:
        alias test_layout = Layout.row_major(test_size)
        alias test_blocks = (ceildiv(test_size, WARP_SIZE), 1)

        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = i
                b_host[i] = i

        out_tensor = LayoutTensor[dtype, out_layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[dtype, test_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, test_layout](b_buf.unsafe_ptr())

        ctx.enqueue_function[
            simple_warp_dot_product[test_layout, out_layout, test_size]
        ](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=test_blocks,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        keep(a.unsafe_ptr())
        keep(b_buf.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[simple_warp_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_functional_warp_parameterized[
    test_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn functional_warp_workflow(ctx: DeviceContext) raises:
        alias test_layout = Layout.row_major(test_size)

        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = i
                b_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, test_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, test_layout](
            b_buf.unsafe_ptr()
        )
        out_tensor = LayoutTensor[mut=True, dtype, Layout.row_major(1)](
            out.unsafe_ptr()
        )

        functional_warp_dot_product[
            test_layout, dtype, SIMD_WIDTH, 1, test_size
        ](out_tensor, a_tensor, b_tensor, ctx)
        keep(out.unsafe_ptr())
        keep(a.unsafe_ptr())
        keep(b_buf.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[functional_warp_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_traditional_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn traditional_workflow(ctx: DeviceContext) raises:
        alias test_layout = Layout.row_major(test_size)
        alias test_blocks = (ceildiv(test_size, WARP_SIZE), 1)

        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = i
                b_host[i] = i

        out_tensor = LayoutTensor[dtype, out_layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[dtype, test_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, test_layout](b_buf.unsafe_ptr())

        ctx.enqueue_function[
            traditional_dot_product_p12_style[
                test_layout, out_layout, test_size
            ]
        ](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=test_blocks,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        keep(a.unsafe_ptr())
        keep(b_buf.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[traditional_workflow](bench_ctx)


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i

        out_tensor = LayoutTensor[mut=True, dtype, out_layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, in_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, in_layout](b.unsafe_ptr())

        print("SIZE:", SIZE)
        print("WARP_SIZE:", WARP_SIZE)
        print("SIMD_WIDTH:", SIMD_WIDTH)
        if argv()[1] == "--traditional":
            ctx.enqueue_function[
                traditional_dot_product_p12_style[in_layout, out_layout, SIZE]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--kernel":
            ctx.enqueue_function[
                simple_warp_dot_product[in_layout, out_layout, SIZE]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

        elif argv()[1] == "--functional":
            functional_warp_dot_product[in_layout, dtype, SIMD_WIDTH, 1, SIZE](
                out_tensor, a_tensor, b_tensor, ctx
            )

        elif argv()[1] == "--benchmark":
            print("-" * 80)
            bench_config = BenchConfig(max_iters=100)
            bench = Bench(bench_config)

            print("Testing SIZE=1 x WARP_SIZE, BLOCKS=1")
            bench.bench_function[
                benchmark_traditional_parameterized[WARP_SIZE]
            ](BenchId("traditional_1x"))
            bench.bench_function[
                benchmark_simple_warp_parameterized[WARP_SIZE]
            ](BenchId("simple_warp_1x"))
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
            print(
                "Testing SIZE=65536 x WARP_SIZE, BLOCKS=65536 (Massive Scale)"
            )
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
            print("🚀 WARP OPERATIONS PERFORMANCE ANALYSIS:")
            print(
                "   GPU Architecture: NVIDIA (WARP_SIZE=32) vs AMD"
                " (WARP_SIZE=64)"
            )
            print("   - 1 x WARP_SIZE: Single warp baseline")
            print("   - 4 x WARP_SIZE: Few warps, warp overhead visible")
            print("   - 32 x WARP_SIZE: Medium scale, warp benefits emerge")
            print("   - 256 x WARP_SIZE: Large scale, dramatic warp advantages")
            print(
                "   - 2048 x WARP_SIZE: Massive scale, warp operations dominate"
            )
            print("   - 16384 x WARP_SIZE: Large scale (512K-1M elements)")
            print("   - 65536 x WARP_SIZE: Massive scale (2M-4M elements)")
            print(
                "   - Note: AMD GPUs process 2 x elements per warp vs NVIDIA!"
            )
            print()
            print("   Expected Results at Large Scales:")
            print("   • Traditional: Slower due to more barrier overhead")
            print(
                "   • Warp operations: Faster, scale better with problem size"
            )
            print("   • Memory bandwidth becomes the limiting factor")
            return

        else:
            print(
                "Usage: --traditional | --kernel | --functional | --benchmark"
            )
            return

        expected = ctx.enqueue_create_host_buffer[dtype](1).enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("=== RESULT ===")
            print("out:", out_host[0])
            print("expected:", expected[0])
            assert_equal(out_host[0], expected[0])

        if len(argv()) == 1 or argv()[1] == "--kernel":
            print()
            print(
                "🚀 Notice how simple the warp version is compared to p10.mojo!"
            )
            print(
                "   Same kernel structure, but warp_sum() replaces all the"
                " complexity!"
            )
        elif argv()[1] == "--functional":
            print()
            print(
                "🔧 Functional approach shows modern Mojo style with warp"
                " operations!"
            )
            print(
                "   Clean, composable, and still leverages warp hardware"
                " primitives!"
            )
