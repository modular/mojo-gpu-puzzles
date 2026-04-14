from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_equal

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime out_layout = row_major[SIZE, SIZE]()
comptime a_layout = row_major[1, SIZE]()
comptime b_layout = row_major[SIZE, 1]()
comptime OutLayout = type_of(out_layout)
comptime ALayout = type_of(a_layout)
comptime BLayout = type_of(b_layout)


# ANCHOR: broadcast_add_solution
def broadcast_add(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, ALayout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, BLayout, ImmutAnyOrigin],
    size: Int,
):
    var row = thread_idx.y
    var col = thread_idx.x
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0]


# ANCHOR_END: broadcast_add_solution


def main() raises:
    with DeviceContext() as ctx:
        var out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_buf.enqueue_fill(0)
        var out_tensor = TileTensor(out_buf, out_layout)
        print("out shape:", out_tensor.dim[0](), "x", out_tensor.dim[1]())

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected_buf.enqueue_fill(0)
        var expected_tensor = TileTensor(expected_buf, out_layout)

        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i + 1)
                b_host[i] = Scalar[dtype](i * 10)

            for i in range(SIZE):
                for j in range(SIZE):
                    expected_tensor[i, j] = a_host[j] + b_host[i]

        var a_tensor = TileTensor[mut=False, dtype, ALayout](a, a_layout)
        var b_tensor = TileTensor[mut=False, dtype, BLayout](b, b_layout)

        ctx.enqueue_function[broadcast_add, broadcast_add](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out_buf.map_to_host() as out_buf_host:
            print("out:", out_buf_host)
            print("expected:", expected_buf)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
            print("Puzzle 05 complete ✅")
