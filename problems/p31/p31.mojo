from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_almost_equal
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: minimal_kernel
alias SIZE = 32 * 1024 * 1024  # 32M elements - larger workload to show occupancy effects
alias THREADS_PER_BLOCK = (1024, 1)
alias BLOCKS_PER_GRID = (SIZE // 1024, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)
alias ALPHA = Float32(2.5)  # SAXPY coefficient


fn minimal_kernel[
    layout: Layout
](
    y: LayoutTensor[dtype, layout, MutAnyOrigin],
    x: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Minimal SAXPY kernel - simple and register-light for high occupancy."""
    i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        # Direct computation: y[i] = alpha * x[i] + y[i]
        # Uses minimal registers (~8), no shared memory
        y[i] = alpha * x[i] + y[i]


# ANCHOR_END: minimal_kernel


# ANCHOR: sophisticated_kernel
fn sophisticated_kernel[
    layout: Layout
](
    y: LayoutTensor[dtype, layout, MutAnyOrigin],
    x: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Sophisticated SAXPY kernel - over-engineered with excessive resource usage.
    """
    # Maximum shared memory allocation (close to 48KB limit)
    shared_cache = LayoutTensor[
        dtype,
        Layout.row_major(1024 * 12),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()  # 48KB

    i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    if i < size:
        # REAL computational work that can't be optimized away - affects final result
        base_x = x[i]
        base_y = y[i]

        # Simulate "precision enhancement" - multiple small adjustments that add up
        # Each computation affects the final result so compiler can't eliminate them
        # But artificially increases register pressure
        precision_x1 = base_x * 1.0001
        precision_x2 = precision_x1 * 0.9999
        precision_x3 = precision_x2 * 1.000001
        precision_x4 = precision_x3 * 0.999999

        precision_y1 = base_y * 1.000005
        precision_y2 = precision_y1 * 0.999995
        precision_y3 = precision_y2 * 1.0000001
        precision_y4 = precision_y3 * 0.9999999

        # Multiple alpha computations for "stability" - should equal alpha
        alpha1 = alpha * 1.00001 * 0.99999
        alpha2 = alpha1 * 1.000001 * 0.999999
        alpha3 = alpha2 * 1.0000001 * 0.9999999
        alpha4 = alpha3 * 1.00000001 * 0.99999999

        # Complex polynomial "optimization" - creates register pressure
        x_power2 = precision_x4 * precision_x4
        x_power3 = x_power2 * precision_x4
        x_power4 = x_power3 * precision_x4
        x_power5 = x_power4 * precision_x4
        x_power6 = x_power5 * precision_x4
        x_power7 = x_power6 * precision_x4
        x_power8 = x_power7 * precision_x4

        # "Advanced" mathematical series that contributes tiny amount to result
        series_term1 = x_power2 * 0.0000001  # x^2/10M
        series_term2 = x_power4 * 0.00000001  # x^4/100M
        series_term3 = x_power6 * 0.000000001  # x^6/1B
        series_term4 = x_power8 * 0.0000000001  # x^8/10B
        series_correction = (
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
        cached_x = shared_cache[local_i] if local_i < 1024 else precision_x4
        cached_y = (
            shared_cache[local_i + 1024] if local_i < 1024 else precision_y4
        )
        cached_alpha = (
            shared_cache[local_i + 2048] if local_i < 1024 else alpha4
        )
        cached_correction = (
            shared_cache[local_i + 3072] if local_i
            < 1024 else series_correction
        )

        # Final "high precision" computation - all work contributes to result
        high_precision_result = (
            cached_alpha * cached_x + cached_y + cached_correction
        )

        # Over-engineered result with massive resource usage but mathematically ~= alpha*x + y
        y[i] = high_precision_result


# ANCHOR_END: sophisticated_kernel


# ANCHOR: balanced_kernel
fn balanced_kernel[
    layout: Layout
](
    y: LayoutTensor[dtype, layout, MutAnyOrigin],
    x: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    alpha: Float32,
    size: Int,
):
    """Balanced SAXPY kernel - efficient optimization with moderate resources.
    """
    # Reasonable shared memory usage for effective caching (16KB)
    shared_cache = LayoutTensor[
        dtype,
        Layout.row_major(1024 * 4),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()  # 16KB total

    i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    if i < size:
        # Moderate computational work that contributes to result
        base_x = x[i]
        base_y = y[i]

        # Light precision enhancement - less than sophisticated kernel
        enhanced_x = base_x * 1.00001 * 0.99999
        enhanced_y = base_y * 1.00001 * 0.99999
        stable_alpha = alpha * 1.000001 * 0.999999

        # Moderate computational optimization
        x_squared = enhanced_x * enhanced_x
        optimization_hint = x_squared * 0.000001

        # Efficient shared memory caching - only what we actually need
        if local_i < 1024:
            shared_cache[local_i] = enhanced_x
            shared_cache[local_i + 1024] = enhanced_y
        barrier()

        # Use cached values efficiently
        cached_x = shared_cache[local_i] if local_i < 1024 else enhanced_x
        cached_y = (
            shared_cache[local_i + 1024] if local_i < 1024 else enhanced_y
        )

        # Balanced computation - moderate work, good efficiency
        result = stable_alpha * cached_x + cached_y + optimization_hint

        # Balanced result with moderate resource usage (~15 registers, 16KB shared)
        y[i] = result


# ANCHOR_END: balanced_kernel


@parameter
@always_inline
fn benchmark_minimal_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn minimal_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = minimal_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[minimal_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_sophisticated_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn sophisticated_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = sophisticated_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[sophisticated_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_balanced_parameterized[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn balanced_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        y = ctx.enqueue_create_buffer[dtype](test_size)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](test_size)
        x.enqueue_fill(0)

        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(test_size):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = balanced_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            y_tensor,
            x_tensor,
            ALPHA,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(y.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[balanced_workflow](bench_ctx)


def test_minimal():
    """Test minimal kernel."""
    print("Testing minimal kernel...")
    with DeviceContext() as ctx:
        y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        # Create LayoutTensors
        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = minimal_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
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
                expected = ALPHA * x_host[i] + Float32(
                    i + 2
                )  # original y[i] was (i + 2)
                actual = y_host[i]
                assert_almost_equal(expected, actual)

        print("✅ Minimal kernel test passed")


def test_sophisticated():
    """Test sophisticated kernel."""
    print("Testing sophisticated kernel...")
    with DeviceContext() as ctx:
        y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        # Create LayoutTensors
        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = sophisticated_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
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
                expected = ALPHA * x_host[i] + Float32(
                    i + 2
                )  # original y[i] was (i + 2)
                actual = y_host[i]
                # Higher tolerance for sophisticated kernel's precision enhancements
                assert_almost_equal(expected, actual, rtol=1e-3, atol=1e-3)

        print("✅ Sophisticated kernel test passed")


def test_balanced():
    """Test balanced kernel."""
    print("Testing balanced kernel...")
    with DeviceContext() as ctx:
        y = ctx.enqueue_create_buffer[dtype](SIZE)
        y.enqueue_fill(0)
        x = ctx.enqueue_create_buffer[dtype](SIZE)
        x.enqueue_fill(0)

        # Initialize test data
        with y.map_to_host() as y_host, x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = Float32(i + 1)
                y_host[i] = Float32(i + 2)

        # Create LayoutTensors
        y_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](y)
        x_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](x)

        alias kernel = balanced_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
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
                expected = ALPHA * x_host[i] + Float32(
                    i + 2
                )  # original y[i] was (i + 2)
                actual = y_host[i]
                # Higher tolerance for balanced kernel's precision enhancements
                assert_almost_equal(expected, actual, rtol=1e-4, atol=1e-4)

        print("✅ Balanced kernel test passed")


def main():
    """Run the occupancy efficiency mystery tests."""
    args = argv()
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
    run_minimal = False
    run_sophisticated = False
    run_balanced = False
    run_all = False
    run_benchmark = False

    for i in range(1, len(args)):
        arg = args[i]
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

    elif run_benchmark:
        bench = Bench()
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
