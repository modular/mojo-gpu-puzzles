from std.gpu import thread_idx, block_dim, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_almost_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: minimal_kernel
comptime SIZE = 32 * 1024 * 1024  # 32M elements - larger workload to show occupancy effects
comptime THREADS_PER_BLOCK = (1024, 1)
comptime BLOCKS_PER_GRID = (SIZE // 1024, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime ALPHA = Scalar[dtype](2.5)  # SAXPY coefficient


def minimal_kernel(
    y: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    x: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Minimal SAXPY kernel - simple and register-light for high occupancy."""
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        # Direct computation: y[i] = alpha * x[i] + y[i]
        # Uses minimal registers (~8), no shared memory
        y[i] = alpha * x[i] + y[i]


# ANCHOR_END: minimal_kernel


# ANCHOR: sophisticated_kernel
def sophisticated_kernel(
    y: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    x: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Sophisticated SAXPY kernel - over-engineered with excessive resource usage.
    """
    # Maximum shared memory allocation (close to 48KB limit)
    var shared_cache = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](
        row_major[1024 * 12]()
    )  # 48KB

    var i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    if i < size:
        # REAL computational work that can't be optimized away - affects final result
        var base_x = x[i]
        var base_y = y[i]

        # Simulate "precision enhancement" - multiple small adjustments that add up
        # Each computation affects the final result so compiler can't eliminate them
        # But artificially increases register pressure
        var precision_x1 = base_x * 1.0001
        var precision_x2 = precision_x1 * 0.9999
        var precision_x3 = precision_x2 * 1.000001
        var precision_x4 = precision_x3 * 0.999999

        var precision_y1 = base_y * 1.000005
        var precision_y2 = precision_y1 * 0.999995
        var precision_y3 = precision_y2 * 1.0000001
        var precision_y4 = precision_y3 * 0.9999999

        # Multiple alpha computations for "stability" - should equal alpha
        var alpha1 = alpha * 1.00001 * 0.99999
        var alpha2 = alpha1 * 1.000001 * 0.999999
        var alpha3 = alpha2 * 1.0000001 * 0.9999999
        var alpha4 = alpha3 * 1.00000001 * 0.99999999

        # Complex polynomial "optimization" - creates register pressure
        var x_power2 = precision_x4 * precision_x4
        var x_power3 = x_power2 * precision_x4
        var x_power4 = x_power3 * precision_x4
        var x_power5 = x_power4 * precision_x4
        var x_power6 = x_power5 * precision_x4
        var x_power7 = x_power6 * precision_x4
        var x_power8 = x_power7 * precision_x4

        # "Advanced" mathematical series that contributes tiny amount to result
        var series_term1 = x_power2 * 0.0000001  # x^2/10M
        var series_term2 = x_power4 * 0.00000001  # x^4/100M
        var series_term3 = x_power6 * 0.000000001  # x^6/1B
        var series_term4 = x_power8 * 0.0000000001  # x^8/10B
        var series_correction = (
            series_term1 - series_term2 + series_term3 - series_term4
        )

        # Over-engineered shared memory usage with multiple caching strategies
        if local_i < 1024:
            shared_cache[local_i] = precision_x4
            shared_cache[local_i + 1024] = precision_y4
            shared_cache[local_i + 2048] = alpha4
            shared_cache[local_i + 3072] = series_correction
        barrier()

        # Load from shared memory for "optimization"
        var cached_x = shared_cache[local_i] if local_i < 1024 else precision_x4
        var cached_y = (
            shared_cache[local_i + 1024] if local_i < 1024 else precision_y4
        )
        var cached_alpha = (
            shared_cache[local_i + 2048] if local_i < 1024 else alpha4
        )
        var cached_correction = (
            shared_cache[local_i + 3072] if local_i
            < 1024 else series_correction
        )

        # Final "high precision" computation - all work contributes to result
        var high_precision_result = (
            cached_alpha * cached_x + cached_y + cached_correction
        )

        # Over-engineered result with massive resource usage but mathematically ~= alpha*x + y
        y[i] = high_precision_result


# ANCHOR_END: sophisticated_kernel


# ANCHOR: balanced_kernel
def balanced_kernel(
    y: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    x: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Balanced SAXPY kernel - efficient optimization with moderate resources.
    """
    # Reasonable shared memory usage for effective caching (16KB)
    var shared_cache = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](
        row_major[1024 * 4]()
    )  # 16KB total

    var i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    if i < size:
        # Moderate computational work that contributes to result
        var base_x = x[i]
        var base_y = y[i]

        # Light precision enhancement - less than sophisticated kernel
        var enhanced_x = base_x * 1.00001 * 0.99999
        var enhanced_y = base_y * 1.00001 * 0.99999
        var stable_alpha = alpha * 1.000001 * 0.999999

        # Moderate computational optimization
        var x_squared = enhanced_x * enhanced_x
        var optimization_hint = x_squared * 0.000001

        # Efficient shared memory caching - only what we actually need
        if local_i < 1024:
            shared_cache[local_i] = enhanced_x
            shared_cache[local_i + 1024] = enhanced_y
        barrier()

        # Use cached values efficiently
        var cached_x = shared_cache[local_i] if local_i < 1024 else enhanced_x
        var cached_y = (
            shared_cache[local_i + 1024] if local_i < 1024 else enhanced_y
        )

        # Balanced computation - moderate work, good efficiency
        var result = stable_alpha * cached_x + cached_y + optimization_hint

        # Balanced result with moderate resource usage (~15 registers, 16KB shared)
        y[i] = result


# ANCHOR_END: balanced_kernel


@parameter
@always_inline
def benchmark_minimal_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def minimal_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = minimal_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[minimal_workflow](bench_ctx)


@parameter
@always_inline
def benchmark_sophisticated_parameterized[
    test_size: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    def sophisticated_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = sophisticated_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[sophisticated_workflow](bench_ctx)


@parameter
@always_inline
def benchmark_balanced_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def balanced_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = balanced_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[balanced_workflow](bench_ctx)


def test_minimal() raises:
    """Test minimal kernel."""
    print("Testing minimal kernel...")
    with DeviceContext() as ctx:
        var y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = minimal_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results: y[i] = alpha * x[i] + original_y[i]
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(10):  # Check first 10
                var expected = ALPHA * x_host[i] + Scalar[dtype](
                    i + 2
                )  # original y[i] was (i + 2)
                var actual = y_host[i]
                assert_almost_equal(expected, actual)

        print("Minimal kernel test: passed")


def test_sophisticated() raises:
    """Test sophisticated kernel."""
    print("Testing sophisticated kernel...")
    with DeviceContext() as ctx:
        var y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = sophisticated_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results: y[i] = alpha * x[i] + original_y[i] (with precision tolerance)
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(10):  # Check first 10
                var expected = ALPHA * x_host[i] + Scalar[dtype](
                    i + 2
                )  # original y[i] was (i + 2)
                var actual = y_host[i]
                # Higher tolerance for sophisticated kernel's precision enhancements
                assert_almost_equal(expected, actual, rtol=1e-3, atol=1e-3)

        print("Sophisticated kernel test: passed")


def test_balanced() raises:
    """Test balanced kernel."""
    print("Testing balanced kernel...")
    with DeviceContext() as ctx:
        var y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Scalar[dtype](i + 1)
                y_host[i] = Scalar[dtype](i + 2)

        # Create TileTensors
        var y_tensor = TileTensor(y, layout)
        var x_tensor = TileTensor[mut=False, dtype, LayoutType](x, layout)

        comptime kernel = balanced_kernel
        ctx.enqueue_function[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Verify results: y[i] = alpha * x[i] + original_y[i] (with precision tolerance)
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(10):  # Check first 10
                var expected = ALPHA * x_host[i] + Scalar[dtype](
                    i + 2
                )  # original y[i] was (i + 2)
                var actual = y_host[i]
                # Higher tolerance for balanced kernel's precision enhancements
                assert_almost_equal(expected, actual, rtol=1e-4, atol=1e-4)

        print("Balanced kernel test: passed")


def main() raises:
    """Run the occupancy efficiency mystery tests."""
    var args = argv()
    if len(args) < 2:
        print("Usage: mojo p31.mojo <flags>")
        print("  Flags:")
        print("    --minimal       Test minimal kernel (high occupancy)")
        print("    --sophisticated Test sophisticated kernel (low occupancy)")
        print("    --balanced      Test balanced kernel (optimal occupancy)")
        print("    --all           Test all kernels")
        print("    --benchmark     Run benchmarks for all kernels")
        return

    # Parse flags
    var run_minimal = False
    var run_sophisticated = False
    var run_balanced = False
    var run_all = False
    var run_benchmark = False

    for i in range(1, len(args)):
        var arg = args[i]
        if arg == "--minimal":
            run_minimal = True
        elif arg == "--sophisticated":
            run_sophisticated = True
        elif arg == "--balanced":
            run_balanced = True
        elif arg == "--all":
            run_all = True
        elif arg == "--benchmark":
            run_benchmark = True
        else:
            print("Unknown flag:", arg)
            print(
                "Valid flags: --minimal, --sophisticated, --balanced, --all,"
                " --benchmark"
            )
            return

    print("============================")
    print("Vector size:", SIZE, "elements (32M - large workload)")
    print("Operation: SAXPY y[i] = alpha * x[i] + y[i], alpha =", ALPHA)
    print(
        "Grid config:",
        BLOCKS_PER_GRID[0],
        "blocks x",
        THREADS_PER_BLOCK[0],
        "threads",
    )

    if run_all:
        print("\nTesting all kernels...")
        test_minimal()
        test_sophisticated()
        test_balanced()
        print("Puzzle 31 complete ✅")

    elif run_benchmark:
        var bench = Bench()
        print("Benchmarking Minimal Kernel (High Occupancy)")
        bench.bench_function[benchmark_minimal_parameterized[SIZE]](
            BenchId("minimal")
        )

        print("Benchmarking Sophisticated Kernel (Low Occupancy)")
        bench.bench_function[benchmark_sophisticated_parameterized[SIZE]](
            BenchId("sophisticated")
        )

        print("Benchmarking Balanced Kernel (Optimal Occupancy)")
        bench.bench_function[benchmark_balanced_parameterized[SIZE]](
            BenchId("balanced")
        )

        bench.dump_report()
    else:
        if run_minimal:
            test_minimal()
        if run_sophisticated:
            test_sophisticated()
        if run_balanced:
            test_balanced()
        print("Puzzle 31 complete ✅")
