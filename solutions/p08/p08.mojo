from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal

comptime TPB = 4
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (2, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: add_10_shared_solution
def add_10_shared_tile_tensor(
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

    if global_i < size:
        shared[local_i] = a[global_i]

    # Note: barrier is not strictly needed here since each thread only accesses
    # its own shared memory location. However, it's included to teach proper
    # shared memory synchronization patterns for more complex scenarios where
    # threads need to coordinate access to shared data.
    barrier()

    if global_i < size:
        output[global_i] = shared[local_i] + 10


# ANCHOR_END: add_10_shared_solution


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)

        ctx.enqueue_function[
            add_10_shared_tile_tensor, add_10_shared_tile_tensor
        ](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(11)
        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 08 complete ✅")
