from std.memory import UnsafePointer
from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal

# ANCHOR: add_10_2d
comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32


def add_10_2d(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    var row = thread_idx.y
    var col = thread_idx.x
    # FILL ME IN (roughly 2 lines)


# ANCHOR_END: add_10_2d


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out.enqueue_fill(0)
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            # row-major
            for i in range(SIZE):
                for j in range(SIZE):
                    a_host[i * SIZE + j] = i * SIZE + j
                    expected[i * SIZE + j] = a_host[i * SIZE + j] + 10

        ctx.enqueue_function[add_10_2d, add_10_2d](
            out,
            a,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])
