from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_almost_equal
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: no_conflict_kernel
alias SIZE = 8 * 1024  # 8K elements - small enough to focus on shared memory patterns
alias TPB = 256  # Threads per block - divisible by 32 (warp size)
alias THREADS_PER_BLOCK = (TPB, 1)
alias BLOCKS_PER_GRID = (SIZE // TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


fn no_conflict_kernel[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    """Perfect shared memory access - no bank conflicts.

    Each thread accesses a different bank: thread_idx.x maps to bank thread_idx.x % 32.
    This achieves optimal shared memory bandwidth utilization.
    """

    # Shared memory buffer - each thread loads one element
    shared_buf = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Load from global memory to shared memory - no conflicts
    if global_i < size:
        shared_buf[local_i] = (
            input[global_i] + 10.0
        )  # Add 10 as simple operation

    barrier()  # Synchronize shared memory writes

    # Read back from shared memory and write to output - no conflicts
    if global_i < size:
        output[global_i] = shared_buf[local_i] * 2.0  # Multiply by 2

    barrier()  # Ensure completion


# ANCHOR_END: no_conflict_kernel


# ANCHOR: two_way_conflict_kernel
fn two_way_conflict_kernel[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    """Stride-2 shared memory access - creates 2-way bank conflicts.

    Threads 0,16 → Bank 0, Threads 1,17 → Bank 1, etc.
    Each bank serves 2 threads, doubling access time.
    """

    # Shared memory buffer - stride-2 access pattern creates conflicts
    shared_buf = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # CONFLICT: stride-2 access creates 2-way bank conflicts
    conflict_index = (local_i * 2) % TPB

    # Load with bank conflicts
    if global_i < size:
        shared_buf[conflict_index] = (
            input[global_i] + 10.0
        )  # Same operation as no-conflict

    barrier()  # Synchronize shared memory writes

    # Read back with same conflicts
    if global_i < size:
        output[global_i] = (
            shared_buf[conflict_index] * 2.0
        )  # Same operation as no-conflict

    barrier()  # Ensure completion


# ANCHOR_END: two_way_conflict_kernel


@parameter
@always_inline
fn benchmark_no_conflict[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn kernel_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        input_buf = ctx.enqueue_create_buffer[dtype](test_size)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(test_size):
                input_host[i] = Float32(i + 1)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=False, dtype, layout](
            input_buf.unsafe_ptr()
        )

        alias kernel = no_conflict_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            input_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[kernel_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_two_way_conflict[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn kernel_workflow(ctx: DeviceContext) raises:
        alias layout = Layout.row_major(test_size)
        out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        input_buf = ctx.enqueue_create_buffer[dtype](test_size)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(test_size):
                input_host[i] = Float32(i + 1)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=False, dtype, layout](
            input_buf.unsafe_ptr()
        )

        alias kernel = two_way_conflict_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            input_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    bench_ctx = DeviceContext()
    b.iter_custom[kernel_workflow](bench_ctx)


fn test_no_conflict() raises:
    """Test that no-conflict kernel produces correct results."""
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Float32(i + 1)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=False, dtype, layout](
            input_buf.unsafe_ptr()
        )

        alias kernel = no_conflict_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            input_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        with out.map_to_host() as result:
            for i in range(min(SIZE, 10)):
                expected = Float32((i + 11) * 2)
                assert_almost_equal(result[i], expected, atol=1e-5)

        print("✅ No-conflict kernel: PASSED")


fn test_two_way_conflict() raises:
    """Test that 2-way conflict kernel produces identical results."""
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Float32(i + 1)

        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=False, dtype, layout](
            input_buf.unsafe_ptr()
        )

        alias kernel = two_way_conflict_kernel[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            input_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        with out.map_to_host() as result:
            for i in range(min(SIZE, 10)):
                expected = Float32((i + 11) * 2)
                assert_almost_equal(result[i], expected, atol=1e-5)

        print("✅ Two-way conflict kernel: PASSED")


fn main() raises:
    if len(argv()) < 2:
        print(
            "Usage: mojo p32.mojo [--test] [--benchmark] [--no-conflict]"
            " [--two-way]"
        )
        return

    var arg = argv()[1]

    if arg == "--test":
        print("Testing bank conflict kernels...")
        test_no_conflict()
        test_two_way_conflict()
        print("✅ Both kernels produce identical results")
        print("Now profile with NSight Compute to see performance differences!")

    elif arg == "--benchmark":
        print("Benchmarking bank conflict patterns...")
        print("-" * 50)

        bench = Bench()

        print("\nNo-conflict kernel (optimal):")
        bench.bench_function[benchmark_no_conflict[SIZE]](
            BenchId("no_conflict")
        )

        print("\nTwo-way conflict kernel:")
        bench.bench_function[benchmark_two_way_conflict[SIZE]](
            BenchId("two_way_conflict")
        )

        bench.dump_report()

    elif arg == "--no-conflict":
        test_no_conflict()
    elif arg == "--two-way":
        test_two_way_conflict()
    else:
        print("Unknown argument:", arg)
        print(
            "Usage: mojo p32.mojo [--test] [--benchmark] [--no-conflict]"
            " [--two-way]"
        )
