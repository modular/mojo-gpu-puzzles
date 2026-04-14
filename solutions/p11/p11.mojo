from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal

comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: pooling_solution
def pooling(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    # Allocate shared memory using stack_allocation
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Load data into shared memory
    if global_i < size:
        shared[local_i] = a[global_i]

    # Synchronize threads within block
    barrier()

    # Handle first two special cases
    if global_i == 0:
        output[0] = shared[0]
    elif global_i == 1:
        output[1] = shared[0] + shared[1]
    # Handle general case
    elif 1 < global_i < size:
        output[global_i] = (
            shared[local_i - 2] + shared[local_i - 1] + shared[local_i]
        )


# ANCHOR_END: pooling_solution


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)

        ctx.enqueue_function[pooling, pooling](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            var ptr = a_host
            for i in range(SIZE):
                var s = Scalar[dtype](0)
                for j in range(max(i - 2, 0), i + 1):
                    s += ptr[j]
                expected[i] = s

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 11 complete ✅")
