from std.gpu import thread_idx, block_dim, block_idx
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.sys import argv
from std.testing import assert_almost_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

comptime SIZE = 16 * 1024 * 1024  # 16M elements - large enough to show memory patterns
comptime THREADS_PER_BLOCK = (1024, 1)  # Max CUDA threads per block
comptime BLOCKS_PER_GRID = (
    SIZE // 1024,
    1,
)  # Enough blocks to cover all elements
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: kernel1
def kernel1(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[i] = a[i] + b[i]


# ANCHOR_END: kernel1


# ANCHOR: kernel2
def kernel2(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var stride = 512

    var i = tid
    while i < size:
        output[i] = a[i] + b[i]
        i += stride


# ANCHOR_END: kernel2


# ANCHOR: kernel3
def kernel3(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total_threads = (SIZE // 1024) * 1024

    for step in range(0, size, total_threads):
        var forward_i = step + tid
        if forward_i < size:
            var reverse_i = size - 1 - forward_i
            output[reverse_i] = a[reverse_i] + b[reverse_i]


# ANCHOR_END: kernel3


@parameter
@always_inline
def benchmark_kernel1_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def kernel1_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](test_size)
        a.enqueue_fill(0)
        var b_buf = ctx.enqueue_create_buffer[dtype](test_size)
        b_buf.enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b_buf, layout)

        ctx.enqueue_function[kernel1, kernel1](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out)
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel1_workflow](bench_ctx)


@parameter
@always_inline
def benchmark_kernel2_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def kernel2_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](test_size)
        a.enqueue_fill(0)
        var b_buf = ctx.enqueue_create_buffer[dtype](test_size)
        b_buf.enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b_buf, layout)

        ctx.enqueue_function[kernel2, kernel2](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out)
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel2_workflow](bench_ctx)


@parameter
@always_inline
def benchmark_kernel3_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def kernel3_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](test_size)
        a.enqueue_fill(0)
        var b_buf = ctx.enqueue_create_buffer[dtype](test_size)
        b_buf.enqueue_fill(0)

        with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
            for i in range(test_size):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b_buf, layout)

        ctx.enqueue_function[kernel3, kernel3](
            out_tensor,
            a_tensor,
            b_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out)
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel3_workflow](bench_ctx)


def test_kernel1() raises:
    """Test kernel 1."""
    print("Testing kernel 1...")
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b, layout)

        ctx.enqueue_function[kernel1, kernel1](
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
                var expected = a_host[i] + b_host[i]
                var actual = out_host[i]
                assert_almost_equal(expected, actual)

        print("Kernel 1 test: passed")


def test_kernel2() raises:
    """Test kernel 2."""
    print("Testing kernel 2...")
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b, layout)

        ctx.enqueue_function[kernel2, kernel2](
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
                    var expected = a_host[i] + b_host[i]
                    var actual = out_host[i]
                    assert_almost_equal(expected, actual)
                    processed += 1

        print("Kernel 2 test: passed")


def test_kernel3() raises:
    """Test kernel 3."""
    print("Testing kernel 3...")
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)

        # Initialize test data
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](b, layout)

        ctx.enqueue_function[kernel3, kernel3](
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
                var expected = a_host[i] + b_host[i]
                var actual = out_host[i]
                assert_almost_equal(expected, actual)

        print("Kernel 3 test: passed")


def main() raises:
    """Run the memory access pattern tests."""
    var args = argv()
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
    var run_kernel1 = False
    var run_kernel2 = False
    var run_kernel3 = False
    var run_all = False
    var run_benchmark = False

    for i in range(1, len(args)):
        var arg = args[i]
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
        print("Puzzle 30 complete ✅")

    elif run_benchmark:
        print("\nRunning Kernel Performance Benchmarks...")
        print("Use nsys/ncu to profile these for detailed analysis!")
        print("-" * 50)

        var bench = Bench()

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
        print("Puzzle 30 complete ✅")
