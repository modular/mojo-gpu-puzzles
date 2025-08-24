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
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, a_layout],
    size: Int,
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_blocks_2d_layout_tensor


def main():
    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        out_tensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin](
            out_buf.unsafe_ptr()
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(1)

        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(1)
        a_tensor = LayoutTensor[dtype, a_layout, MutableAnyOrigin](
            a.unsafe_ptr()
        )

        ctx.enqueue_function[add_10_blocks_2d[out_layout, a_layout]](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        expected_tensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin](
            expected_buf.unsafe_ptr()
        )
        for i in range(SIZE):
            for j in range(SIZE):
                expected_tensor[i, j] += 10

        with out_buf.map_to_host() as out_buf_host:
            print(
                "out:",
                LayoutTensor[dtype, out_layout, MutableAnyOrigin](
                    out_buf_host.unsafe_ptr()
                ),
            )
            print("expected:", expected_tensor)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
