from std.memory import UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.testing import assert_equal

comptime SIZE = 9
comptime BLOCKS_PER_GRID = (3, 1)
comptime THREADS_PER_BLOCK = (4, 1)
comptime dtype = DType.float32


# ANCHOR: add_10_blocks_solution
def add_10_blocks(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0


# ANCHOR_END: add_10_blocks_solution


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i)

        ctx.enqueue_function[add_10_blocks, add_10_blocks](
            out,
            a,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)

        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = Scalar[dtype](i + 10)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 06 complete ✅")
