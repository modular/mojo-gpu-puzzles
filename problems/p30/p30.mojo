from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import argv
from testing import assert_almost_equal
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

alias SIZE = 16 * 1024 * 1024  # 16M elements - large enough to show memory patterns
alias THREADS_PER_BLOCK = (1024, 1)  # Max CUDA threads per block
alias BLOCKS_PER_GRID = (SIZE // 1024, 1)  # Enough blocks to cover all elements
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# ANCHOR: kernel1
fn kernel1[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[i] = a[i] + b[i]


# ANCHOR_END: kernel1


# ANCHOR: kernel2
fn kernel2[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    tid = block_idx.x * block_dim.x + thread_idx.x
    stride = 512

    i = tid
    while i < size:
        output[i] = a[i] + b[i]
        i += stride


# ANCHOR_END: kernel2


# ANCHOR: kernel3
fn kernel3[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    tid = block_idx.x * block_dim.x + thread_idx.x
    total_threads = (SIZE // 1024) * 1024

    for step in range(0, size, total_threads):
        forward_i = step + tid
        if forward_i < size:
            reverse_i = size - 1 - forward_i
            output[reverse_i] = a[reverse_i] + b[reverse_i]


# ANCHOR_END: kernel3


@parameter
@always_inline
fn benchmark_kernel1_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn kernel1_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())

        ctx.enqueue_function[kernel1[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[kernel1_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_kernel2_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn kernel2_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())

        ctx.enqueue_function[kernel2[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[kernel2_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_kernel3_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn kernel3_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)
        b_buf = ctx.enqueue_create_buffer[dtype](test_size).enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b_buf.unsafe_ptr())

        ctx.enqueue_function[kernel3[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[kernel3_workflow](bench_ctx)


def test_kernel1():
    """Test kernel 1."""
    print("Testing kernel 1...")
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        # Create LayoutTensors
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b.unsafe_ptr())

        ctx.enqueue_function[kernel1[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results
        with out.map_to_host() as out_host, a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(10):  # Check first 10
                expected = a_host[i] + b_host[i]
                actual = out_host[i]
                assert_almost_equal(expected, actual)

        print("✅ Kernel 1 test passed")


def test_kernel2():
    """Test kernel 2."""
    print("Testing kernel 2...")
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        # Create LayoutTensors
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b.unsafe_ptr())

        ctx.enqueue_function[kernel2[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results
        var processed = 0
        with out.map_to_host() as out_host, a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                if out_host[i] != 0:  # This element was processed
                    expected = a_host[i] + b_host[i]
                    actual = out_host[i]
                    assert_almost_equal(expected, actual)
                    processed += 1

        print("✅ Kernel 2 test passed,", processed, "elements processed")


def test_kernel3():
    """Test kernel 3."""
    print("Testing kernel 3...")
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i + 2)

        # Create LayoutTensors
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](b.unsafe_ptr())

        ctx.enqueue_function[kernel3[layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results
        with out.map_to_host() as out_host, a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected = a_host[i] + b_host[i]
                actual = out_host[i]
                assert_almost_equal(expected, actual)

        print("✅ Kernel 3 test passed")


def main():
    """Run the memory access pattern tests."""
    args = argv()
    if len(args) < 2:
        print("Usage: mojo p30.mojo <flags>")
        print("  Flags:")
        print("    --kernel1     Test kernel 1")
        print("    --kernel2     Test kernel 2")
        print("    --kernel3     Test kernel 3")
        print("    --all         Test all kernels")
        print("    --benchmark   Run benchmarks for all kernels")
        return

    # Parse flags
    run_kernel1 = False
    run_kernel2 = False
    run_kernel3 = False
    run_all = False
    run_benchmark = False

    for i in range(1, len(args)):
        arg = args[i]
        if arg == "--kernel1":
            run_kernel1 = True
        elif arg == "--kernel2":
            run_kernel2 = True
        elif arg == "--kernel3":
            run_kernel3 = True
        elif arg == "--all":
            run_all = True
        elif arg == "--benchmark":
            run_benchmark = True
        else:
            print("Unknown flag:", arg)
            print(
                "Valid flags: --kernel1, --kernel2, --kernel3, --all,"
                " --benchmark"
            )
            return

    print("MEMORY ACCESS PATTERN MYSTERY")
    print("================================")
    print("Vector size:", SIZE, "elements")
    print(
        "Grid config:",
        BLOCKS_PER_GRID[0],
        "blocks x",
        THREADS_PER_BLOCK[0],
        "threads",
    )

    if run_all:
        print("\nTesting all kernels...")
        test_kernel1()
        test_kernel2()
        test_kernel3()

    elif run_benchmark:
        print("\nRunning Kernel Performance Benchmarks...")
        print("Use nsys/ncu to profile these for detailed analysis!")
        print("-" * 50)

        bench = Bench()

        print("Benchmarking Kernel 1")
        bench.bench_function[benchmark_kernel1_parameterized[SIZE]](
            BenchId("kernel1")
        )

        print("Benchmarking Kernel 2")
        bench.bench_function[benchmark_kernel2_parameterized[SIZE]](
            BenchId("kernel2")
        )

        print("Benchmarking Kernel 3")
        bench.bench_function[benchmark_kernel3_parameterized[SIZE]](
            BenchId("kernel3")
        )

        bench.dump_report()
    else:
        # Run individual tests
        if run_kernel1:
            test_kernel1()
        if run_kernel2:
            test_kernel2()
        if run_kernel3:
            test_kernel3()
