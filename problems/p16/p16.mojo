from sys import size_of, argv
from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: naive_matmul
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor


alias TPB = 3
alias SIZE = 2
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, TPB)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 6 lines)


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x
    # FILL ME IN (roughly 12 lines)


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
alias SIZE_TILED = 9
alias BLOCKS_PER_GRID_TILED = (3, 3)  # each block convers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)


fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout_tiled, MutAnyOrigin],
    a: LayoutTensor[dtype, layout_tiled, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout_tiled, ImmutAnyOrigin],
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + thread_idx.y
    tiled_col = block_idx.x * TPB + thread_idx.x
    # FILL ME IN (roughly 20 lines)


# ANCHOR_END: matmul_tiled


def main():
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
        size = SIZE_TILED if argv()[1] == "--tiled" else SIZE
        out = ctx.enqueue_create_buffer[dtype](size * size)
        out.enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size)
        inp1.enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size)
        inp2.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](size * size)
        expected.enqueue_fill(0)

        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # inp1 @ inp2.T
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](inp1)
        b_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](inp2)

        if argv()[1] == "--naive":
            alias kernel = naive_matmul[layout, SIZE]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            alias kernel = single_block_matmul[layout, SIZE]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = LayoutTensor[dtype, layout_tiled, MutAnyOrigin](
                out
            )
            a_tensor_tiled = LayoutTensor[dtype, layout_tiled, ImmutAnyOrigin](
                inp1
            )
            b_tensor_tiled = LayoutTensor[dtype, layout_tiled, ImmutAnyOrigin](
                inp2
            )

            alias kernel = matmul_tiled[layout_tiled, SIZE_TILED]
            ctx.enqueue_function_checked[kernel, kernel](
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
