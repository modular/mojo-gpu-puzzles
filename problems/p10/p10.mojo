from std.gpu import thread_idx, block_dim, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal
from std.sys import argv

# ANCHOR: shared_memory_race

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)


def shared_memory_race(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var row = thread_idx.y
    var col = thread_idx.x

    var shared_sum = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[1]())

    if row < size and col < size:
        shared_sum[0] += a[row, col]

    barrier()

    if row < size and col < size:
        output[row, col] = shared_sum[0]


# ANCHOR_END: shared_memory_race


# ANCHOR: add_10_2d_no_guard
def add_10_2d(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var row = thread_idx.y
    var col = thread_idx.x
    output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_2d_no_guard


def main() raises:
    if len(argv()) != 2:
        print(
            "Expected one command-line argument: '--memory-bug' or"
            " '--race-condition'"
        )
        return

    var flag = argv()[1]

    with DeviceContext() as ctx:
        var out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_buf.enqueue_fill(0)
        var out_tensor = TileTensor(out_buf, layout)
        print("out shape:", out_tensor.dim[0](), "x", out_tensor.dim[1]())
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)

        var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE * SIZE):
                a_host[i] = Scalar[dtype](i)

        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)

        if flag == "--memory-bug":
            print("Running memory bug example (bounds checking issue)...")
            # Fill expected values directly since it's a HostBuffer
            for i in range(SIZE * SIZE):
                expected[i] = Scalar[dtype](i + 10)

            ctx.enqueue_function[add_10_2d, add_10_2d](
                out_tensor,
                a_tensor,
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with out_buf.map_to_host() as out_buf_host:
                print("out:", out_buf_host)
                print("expected:", expected)
                for i in range(SIZE * SIZE):
                    assert_equal(out_buf_host[i], expected[i])
                print("Memory bug test: passed")
                print("Puzzle 10 complete ✅")

        elif flag == "--race-condition":
            print("Running race condition example...")
            var total_sum = Scalar[dtype](0.0)
            with a.map_to_host() as a_host:
                for i in range(SIZE * SIZE):
                    total_sum += a_host[i]  # Sum: 0 + 1 + 2 + 3 = 6

            # All positions should contain the total sum
            for i in range(SIZE * SIZE):
                expected[i] = total_sum

            ctx.enqueue_function[shared_memory_race, shared_memory_race](
                out_tensor,
                a_tensor,
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with out_buf.map_to_host() as out_buf_host:
                print("out:", out_buf_host)
                print("expected:", expected)
                for i in range(SIZE * SIZE):
                    assert_equal(out_buf_host[i], expected[i])

                print("Race condition test: passed")
                print("Puzzle 10 complete ✅")

        else:
            print("Unknown flag:", flag)
            print("Available flags: --memory-bug, --race-condition")
