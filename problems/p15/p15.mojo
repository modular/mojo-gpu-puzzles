from std.testing import assert_equal
from std.gpu.host import DeviceContext

# ANCHOR: axis_sum
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation


comptime TPB = 8
comptime BATCH = 4
comptime SIZE = 6
comptime BLOCKS_PER_GRID = (1, BATCH)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = row_major[BATCH, SIZE]()
comptime InLayout = type_of(in_layout)
comptime out_layout = row_major[BATCH, 1]()
comptime OutLayout = type_of(out_layout)


def axis_sum(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var batch = block_idx.y
    # FILL ME IN (roughly 15 lines)


# ANCHOR_END: axis_sum


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](BATCH)
        out.enqueue_fill(0)
        var inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE)
        inp.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = Scalar[dtype](row * SIZE + col)

        var out_tensor = TileTensor(out, out_layout)
        var inp_tensor = TileTensor[mut=False, dtype, InLayout](inp, in_layout)

        ctx.enqueue_function[axis_sum, axis_sum](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](BATCH)
        expected.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    expected[row] += inp_host[row * SIZE + col]

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out)
            print("expected:", expected)
            for i in range(BATCH):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 15 complete ✅")
