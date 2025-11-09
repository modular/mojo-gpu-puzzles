from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: add_10_blocks_2d_layout_tensor
alias SIZE = 5
alias BLOCKS_PER_GRID = (2, 2)
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias out_layout = Layout.row_major(SIZE, SIZE)
alias a_layout = Layout.row_major(SIZE, SIZE)


fn add_10_blocks_2d[
    out_layout: Layout,
    a_layout: Layout,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    size: Int,
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 2 lines)


# ANCHOR_END: add_10_blocks_2d_layout_tensor


def main():
    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_buf.enqueue_fill(0)
        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out_buf)

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected_buf.enqueue_fill(1)

        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        a.enqueue_fill(1)

        with a.map_to_host() as a_host:
            for j in range(SIZE):
                for i in range(SIZE):
                    k = j * SIZE + i
                    a_host[k] = k
                    expected_buf[k] = k + 10

        a_tensor = LayoutTensor[dtype, a_layout, ImmutAnyOrigin](a)

        alias kernel = add_10_blocks_2d[out_layout, a_layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        expected_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](
            expected_buf
        )

        with out_buf.map_to_host() as out_buf_host:
            print(
                "out:",
                LayoutTensor[dtype, out_layout, MutAnyOrigin](out_buf_host),
            )
            print("expected:", expected_tensor)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
