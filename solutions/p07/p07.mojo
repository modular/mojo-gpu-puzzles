from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_equal

comptime SIZE = 5
comptime BLOCKS_PER_GRID = (2, 2)
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime out_layout = row_major[SIZE, SIZE]()
comptime a_layout = row_major[SIZE, SIZE]()
comptime OutLayout = type_of(out_layout)
comptime ALayout = type_of(a_layout)


# ANCHOR: add_10_blocks_2d_solution
def add_10_blocks_2d(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, ALayout, ImmutAnyOrigin],
    size: Int,
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_blocks_2d_solution


def main() raises:
    with DeviceContext() as ctx:
        var out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_buf.enqueue_fill(0)
        var out_tensor = TileTensor(out_buf, out_layout)

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected_buf.enqueue_fill(1)

        var a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        a.enqueue_fill(1)

        with a.map_to_host() as a_host:
            for j in range(SIZE):
                for i in range(SIZE):
                    var k = j * SIZE + i
                    a_host[k] = Scalar[dtype](k)
                    expected_buf[k] = Scalar[dtype](k + 10)

        var a_tensor = TileTensor[mut=False, dtype, ALayout](a, a_layout)

        ctx.enqueue_function[add_10_blocks_2d, add_10_blocks_2d](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        var expected_tensor = TileTensor(expected_buf, out_layout)

        with out_buf.map_to_host() as out_buf_host:
            print(
                "out:",
                TileTensor(out_buf_host, out_layout),
            )
            print("expected:", expected_tensor)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
            print("Puzzle 07 complete ✅")
