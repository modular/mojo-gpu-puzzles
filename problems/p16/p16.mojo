from std.sys import argv
from std.testing import assert_equal
from std.gpu.host import DeviceContext

# ANCHOR: naive_matmul
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation


comptime TPB = 3
comptime SIZE = 2
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, TPB)
comptime dtype = DType.float32
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)


def naive_matmul(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 6 lines)


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
def single_block_matmul(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    # FILL ME IN (roughly 12 lines)


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
comptime SIZE_TILED = 9
comptime BLOCKS_PER_GRID_TILED = (3, 3)  # each block convers 3x3 elements
comptime THREADS_PER_BLOCK_TILED = (TPB, TPB)
comptime layout_tiled = row_major[SIZE_TILED, SIZE_TILED]()
comptime LayoutTiledType = type_of(layout_tiled)


def matmul_tiled(
    output: TileTensor[mut=True, dtype, LayoutTiledType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
):
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * TPB + thread_idx.y
    var tiled_col = block_idx.x * TPB + thread_idx.x
    # FILL ME IN (roughly 20 lines)


# ANCHOR_END: matmul_tiled


def main() raises:
    with DeviceContext() as ctx:
        if len(argv()) != 2 or argv()[1] not in [
            "--naive",
            "--single-block",
            "--tiled",
        ]:
            raise Error(
                "Expected one argument: '--naive', '--single-block', or"
                " '--tiled'"
            )
        var size = SIZE_TILED if argv()[1] == "--tiled" else SIZE
        var out = ctx.enqueue_create_buffer[dtype](size * size)
        out.enqueue_fill(0)
        var inp1 = ctx.enqueue_create_buffer[dtype](size * size)
        inp1.enqueue_fill(0)
        var inp2 = ctx.enqueue_create_buffer[dtype](size * size)
        inp2.enqueue_fill(0)
        var expected = ctx.enqueue_create_host_buffer[dtype](size * size)
        expected.enqueue_fill(0)

        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    var val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = Scalar[dtype](val)
                    inp2_host[row * size + col] = Scalar[dtype](2.0 * val)

            # inp1 @ inp2.T
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](inp1, layout)
        var b_tensor = TileTensor[mut=False, dtype, LayoutType](inp2, layout)

        if argv()[1] == "--naive":
            ctx.enqueue_function[naive_matmul, naive_matmul](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            ctx.enqueue_function[single_block_matmul, single_block_matmul](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            var out_tensor_tiled = TileTensor(out, layout_tiled)
            var a_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp1, layout_tiled
            )
            var b_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp2, layout_tiled
            )

            ctx.enqueue_function[matmul_tiled, matmul_tiled](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
            print("Puzzle 16 complete ✅")
