from std.memory import UnsafePointer
from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32


# ANCHOR: broadcast_add_solution
def broadcast_add(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    var row = thread_idx.y
    var col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[col] + b[row]


# ANCHOR_END: broadcast_add_solution


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out.enqueue_fill(0)
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i + 1)
                b_host[i] = Float32(i * 10)

            for y in range(SIZE):
                for x in range(SIZE):
                    expected[y * SIZE + x] = a_host[x] + b_host[y]

        ctx.enqueue_function[broadcast_add, broadcast_add](
            out,
            a,
            b,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for y in range(SIZE):
                for x in range(SIZE):
                    assert_equal(out_host[y * SIZE + x], expected[y * SIZE + x])
            print("Puzzle 05 complete ✅")
