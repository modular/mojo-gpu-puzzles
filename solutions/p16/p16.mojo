from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal

comptime TPB = 3
comptime SIZE = 2
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, TPB)
comptime dtype = DType.float32
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)


# ANCHOR: naive_matmul_solution
def naive_matmul[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    if row < size and col < size:
        var acc: output.ElementType = 0

        comptime for k in range(size):
            acc += a[row, k] * b[k, col]

        output[row, col] = acc


# ANCHOR_END: naive_matmul_solution


# ANCHOR: single_block_matmul_solution
def single_block_matmul[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    var local_row = thread_idx.y
    var local_col = thread_idx.x

    var a_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())
    var b_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())

    if row < size and col < size:
        a_shared[local_row, local_col] = a[row, col]
        b_shared[local_row, local_col] = b[row, col]

    barrier()

    if row < size and col < size:
        var acc: output.ElementType = 0

        comptime for k in range(size):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        output[row, col] = acc


# ANCHOR_END: single_block_matmul_solution


comptime SIZE_TILED = 9
comptime BLOCKS_PER_GRID_TILED = (3, 3)  # each block covers 3x3 elements
comptime THREADS_PER_BLOCK_TILED = (TPB, TPB)
comptime layout_tiled = row_major[SIZE_TILED, SIZE_TILED]()
comptime LayoutTiledType = type_of(layout_tiled)


# ANCHOR: matmul_tiled_solution
def matmul_tiled[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutTiledType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
):
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * TPB + local_row
    var tiled_col = block_idx.x * TPB + local_col

    var a_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())
    var b_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())

    var acc: output.ElementType = 0

    # Iterate over tiles to compute matrix product
    comptime for tile in range((size + TPB - 1) // TPB):
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # Load B tile - row determined by tile, global col stays the same
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]

        barrier()

        # Matrix multiplication within the tile
        if tiled_row < size and tiled_col < size:
            comptime for k in range(min(Int(TPB), Int(size - tile * TPB))):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write out final result
    if tiled_row < size and tiled_col < size:
        output[tiled_row, tiled_col] = acc


# ANCHOR_END: matmul_tiled_solution

# ANCHOR: matmul_idiomatic_tiled_solution
from std.gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async
from layout import Layout as IntTupleLayout

comptime NUM_THREADS = TPB * TPB
comptime BLOCK_DIM_COUNT = 2


def matmul_idiomatic_tiled[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutTiledType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutTiledType, ImmutAnyOrigin],
):
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * TPB + local_row
    var tiled_col = block_idx.x * TPB + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    var out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
    var a_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())
    var b_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB, TPB]())

    var acc: output.ElementType = 0

    comptime load_a_layout = IntTupleLayout.row_major(
        1, TPB
    )  # Coalesced loading
    comptime load_b_layout = IntTupleLayout.row_major(
        1, TPB
    )  # Coalesced loading
    # Note: Both matrices stored in same orientation for correct matrix multiplication
    # Transposed loading would be useful if B were pre-transposed in global memory

    comptime for idx in range(
        size // TPB
    ):  # Perfect division: 9 // 3 = 3 tiles
        # Get tiles from A and B matrices
        var a_tile = a.tile[TPB, TPB](block_idx.y, Int(idx))
        var b_tile = b.tile[TPB, TPB](Int(idx), block_idx.x)

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async[
            thread_layout=load_a_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](a_shared.to_layout_tensor(), a_tile.to_layout_tensor())
        copy_dram_to_sram_async[
            thread_layout=load_b_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](b_shared.to_layout_tensor(), b_tile.to_layout_tensor())

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        comptime for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result to output tile
    if tiled_row < size and tiled_col < size:
        out_tile[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled_solution


def main() raises:
    with DeviceContext() as ctx:
        var size = (
            SIZE_TILED if argv()[1] == "--idiomatic-tiled"
            or argv()[1] == "--tiled" else SIZE
        )
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

            # inp1 @ inp2
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
            comptime kernel = naive_matmul[SIZE]
            ctx.enqueue_function[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            comptime kernel = single_block_matmul[SIZE]
            ctx.enqueue_function[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = TileTensor(out, layout_tiled)
            a_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp1, layout_tiled
            )
            b_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp2, layout_tiled
            )

            comptime kernel = matmul_tiled[SIZE_TILED]
            ctx.enqueue_function[kernel, kernel](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        elif argv()[1] == "--idiomatic-tiled":
            out_tensor_tiled = TileTensor(out, layout_tiled)
            a_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp1, layout_tiled
            )
            b_tensor_tiled = TileTensor[mut=False, dtype, LayoutTiledType](
                inp2, layout_tiled
            )

            comptime kernel = matmul_idiomatic_tiled[SIZE_TILED]
            ctx.enqueue_function[kernel, kernel](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        else:
            raise Error(
                "Invalid option. Choose among the available flags: --naive,"
                " --single-block, --tiled, --idiomatic-tiled"
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
