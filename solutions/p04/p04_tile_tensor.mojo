from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_equal

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: add_10_2d_tile_tensor_solution
def add_10_2d(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    size: Int,
):
    var row = thread_idx.y
    var col = thread_idx.x
    if col < size and row < size:
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_2d_tile_tensor_solution


def main() raises:
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
                expected[i] = a_host[i] + 10

        var a_tensor = TileTensor(a, layout)

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
            print("Puzzle 04 complete ✅")
