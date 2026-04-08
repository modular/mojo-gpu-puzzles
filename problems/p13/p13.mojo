from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal

# ANCHOR: conv_1d_simple
comptime TPB = 8
comptime SIZE = 6
comptime CONV = 3
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = row_major[SIZE]()
comptime InLayout = type_of(in_layout)
comptime out_layout = row_major[SIZE]()
comptime OutLayout = type_of(out_layout)
comptime conv_layout = row_major[CONV]()
comptime ConvLayout = type_of(conv_layout)


def conv_1d_simple(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, ConvLayout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # FILL ME IN (roughly 14 lines)


# ANCHOR_END: conv_1d_simple

# ANCHOR: conv_1d_block_boundary
comptime SIZE_2 = 15
comptime CONV_2 = 4
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime in_2_layout = row_major[SIZE_2]()
comptime In2Layout = type_of(in_2_layout)
comptime out_2_layout = row_major[SIZE_2]()
comptime Out2Layout = type_of(out_2_layout)
comptime conv_2_layout = row_major[CONV_2]()
comptime Conv2Layout = type_of(conv_2_layout)


def conv_1d_block_boundary(
    output: TileTensor[mut=True, dtype, Out2Layout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, In2Layout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, Conv2Layout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # FILL ME IN (roughly 18 lines)


# ANCHOR_END: conv_1d_block_boundary


def main() raises:
    with DeviceContext() as ctx:
        var size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        var conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        var out = ctx.enqueue_create_buffer[dtype](size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](conv)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = Scalar[dtype](i)

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = Scalar[dtype](i)

        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--block-boundary",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or"
                " '--block-boundary'"
            )

        if argv()[1] == "--simple":
            var out_tensor = TileTensor(out, out_layout)
            var a_tensor = TileTensor[mut=False, dtype, InLayout](a, in_layout)
            var b_tensor = TileTensor[mut=False, dtype, ConvLayout](
                b, conv_layout
            )
            ctx.enqueue_function[conv_1d_simple, conv_1d_simple](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var out_tensor = TileTensor(out, out_2_layout)
            var a_tensor = TileTensor[mut=False, dtype, In2Layout](
                a, in_2_layout
            )
            var b_tensor = TileTensor[mut=False, dtype, Conv2Layout](
                b, conv_2_layout
            )
            ctx.enqueue_function[
                conv_1d_block_boundary, conv_1d_block_boundary
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

        ctx.synchronize()
        var expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 13 complete ✅")
