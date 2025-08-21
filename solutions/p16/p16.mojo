from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal

alias TPB = 3
alias SIZE = 2
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, TPB)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


# ANCHOR: naive_matmul_solution
fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    if row < size and col < size:
        var acc: output.element_type = 0

        @parameter
        for k in range(size):
            acc += a[row, k] * b[k, col]

        output[row, col] = acc


# ANCHOR_END: naive_matmul_solution


# ANCHOR: single_block_matmul_solution
fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x

    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    if row < size and col < size:
        a_shared[local_row, local_col] = a[row, col]
        b_shared[local_row, local_col] = b[row, col]

    barrier()

    if row < size and col < size:
        var acc: output.element_type = 0

        @parameter
        for k in range(size):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        output[row, col] = acc


# ANCHOR_END: single_block_matmul_solution


alias SIZE_TILED = 9
alias BLOCKS_PER_GRID_TILED = (3, 3)  # each block covers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)


# ANCHOR: matmul_tiled_solution
fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    var acc: output.element_type = 0

    # Iterate over tiles to compute matrix product
    @parameter
    for tile in range((size + TPB - 1) // TPB):
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

            @parameter
            for k in range(min(TPB, size - tile * TPB)):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write out final result
    if tiled_row < size and tiled_col < size:
        output[tiled_row, tiled_col] = acc


# ANCHOR_END: matmul_tiled_solution

# ANCHOR: matmul_idiomatic_tiled_solution
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

alias NUM_THREADS = TPB * TPB
alias BLOCK_DIM_COUNT = 2


fn matmul_idiomatic_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    var acc: output.element_type = 0

    alias load_a_layout = Layout.row_major(1, TPB)  # Coalesced loading
    alias load_b_layout = Layout.row_major(1, TPB)  # Coalesced loading
    # Note: Both matrices stored in same orientation for correct matrix multiplication
    # Transposed loading would be useful if B were pre-transposed in global memory

    @parameter
    for idx in range(size // TPB):  # Perfect division: 9 // 3 = 3 tiles
        # Get tiles from A and B matrices
        a_tile = a.tile[TPB, TPB](block_idx.y, idx)
        b_tile = b.tile[TPB, TPB](idx, block_idx.x)

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async[
            thread_layout=load_a_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](a_shared, a_tile)
        copy_dram_to_sram_async[
            thread_layout=load_b_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        ](b_shared, b_tile)

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result to output tile
    if tiled_row < size and tiled_col < size:
        out_tile[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled_solution


def main():
    with DeviceContext() as ctx:
        size = (
            SIZE_TILED if argv()[1] == "--idiomatic-tiled"
            or argv()[1] == "--tiled" else SIZE
        )
        out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            size * size
        ).enqueue_fill(0)
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # inp1 @ inp2
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr())

        if argv()[1] == "--naive":
            ctx.enqueue_function[naive_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            ctx.enqueue_function[single_block_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )

            ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
        elif argv()[1] == "--idiomatic-tiled":
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )
            ctx.enqueue_function[
                matmul_idiomatic_tiled[layout_tiled, SIZE_TILED]
            ](
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
