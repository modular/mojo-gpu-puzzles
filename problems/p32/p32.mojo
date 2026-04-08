from std.gpu import thread_idx, block_dim, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_almost_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

# ANCHOR: no_conflict_kernel
comptime SIZE = 8 * 1024  # 8K elements - small enough to focus on shared memory patterns
comptime TPB = 256  # Threads per block - divisible by 32 (warp size)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime BLOCKS_PER_GRID = (SIZE // TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def no_conflict_kernel(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    """Perfect shared memory access - no bank conflicts.

    Each thread accesses a different bank: thread_idx.x maps to bank thread_idx.x % 32.
    This achieves optimal shared memory bandwidth utilization.
    """

    # Shared memory buffer - each thread loads one element
    var shared_buf = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

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
def two_way_conflict_kernel(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    """Stride-2 shared memory access - creates 2-way bank conflicts.

    Threads 0,16 -> Bank 0, Threads 1,17 -> Bank 1, etc.
    Each bank serves 2 threads, doubling access time.
    """

    # Shared memory buffer - stride-2 access pattern creates conflicts
    var shared_buf = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # CONFLICT: stride-2 access creates 2-way bank conflicts
    var conflict_index = (local_i * 2) % TPB

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
def benchmark_no_conflict[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def kernel_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        var input_buf = ctx.enqueue_create_buffer[dtype](test_size)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(test_size):
                input_host[i] = Scalar[dtype](i + 1)

        var out_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[mut=False, dtype, LayoutType](
            input_buf, layout
        )

        comptime kernel = no_conflict_kernel
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            input_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel_workflow](bench_ctx)


@parameter
@always_inline
def benchmark_two_way_conflict[test_size: Int](mut b: Bencher) raises:
    @parameter
    @always_inline
    def kernel_workflow(ctx: DeviceContext) raises:
        comptime layout = row_major[test_size]()
        comptime LayoutType = type_of(layout)
        var out = ctx.enqueue_create_buffer[dtype](test_size)
        out.enqueue_fill(0)
        var input_buf = ctx.enqueue_create_buffer[dtype](test_size)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(test_size):
                input_host[i] = Scalar[dtype](i + 1)

        var out_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[mut=False, dtype, LayoutType](
            input_buf, layout
        )

        comptime kernel = two_way_conflict_kernel
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            input_tensor,
            test_size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        keep(out.unsafe_ptr())
        ctx.synchronize()

    var bench_ctx = DeviceContext()
    b.iter_custom[kernel_workflow](bench_ctx)


def test_no_conflict() raises:
    """Test that no-conflict kernel produces correct results."""
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Scalar[dtype](i + 1)

        var out_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[mut=False, dtype, LayoutType](
            input_buf, layout
        )

        comptime kernel = no_conflict_kernel
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            input_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        with out.map_to_host() as result:
            for i in range(min(SIZE, 10)):
                var expected = Scalar[dtype]((i + 11) * 2)
                assert_almost_equal(result[i], expected, atol=1e-5)

        print("No-conflict kernel test: passed")


def test_two_way_conflict() raises:
    """Test that 2-way conflict kernel produces identical results."""
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Scalar[dtype](i + 1)

        var out_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[mut=False, dtype, LayoutType](
            input_buf, layout
        )

        comptime kernel = two_way_conflict_kernel
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            input_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        with out.map_to_host() as result:
            for i in range(min(SIZE, 10)):
                var expected = Scalar[dtype]((i + 11) * 2)
                assert_almost_equal(result[i], expected, atol=1e-5)

        print("Two-way conflict kernel test: passed")


def main() raises:
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
        print("Puzzle 32 complete ✅")
        print("Now profile with NSight Compute to see performance differences!")

    elif arg == "--benchmark":
        print("Benchmarking bank conflict patterns...")
        print("-" * 50)

        var bench = Bench()

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
        print("Puzzle 32 complete ✅")
    elif arg == "--two-way":
        test_two_way_conflict()
        print("Puzzle 32 complete ✅")
    else:
        print("Unknown argument:", arg)
        print(
            "Usage: mojo p32.mojo [--test] [--benchmark] [--no-conflict]"
            " [--two-way]"
        )
