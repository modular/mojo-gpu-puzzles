from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext, HostBuffer
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: broadcast_add_layout_tensor
alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias out_layout = Layout.row_major(SIZE, SIZE)
alias a_layout = Layout.row_major(1, SIZE)
alias b_layout = Layout.row_major(SIZE, 1)


fn broadcast_add[
    out_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, a_layout],
    b: LayoutTensor[mut=False, dtype, b_layout],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0]


# ANCHOR_END: broadcast_add_layout_tensor
def main():
    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        out_tensor = LayoutTensor[mut=True, dtype, out_layout](
            out_buf.unsafe_ptr()
        )
        print("out shape:", out_tensor.shape[0](), "x", out_tensor.shape[1]())

        expected_buf = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        expected_tensor = LayoutTensor[mut=True, dtype, out_layout](
            expected_buf.unsafe_ptr()
        )

        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i

            for i in range(SIZE):
                for j in range(SIZE):
                    expected_tensor[i, j] = a_host[j] + b_host[i]

        a_tensor = LayoutTensor[dtype, a_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, b_layout](b.unsafe_ptr())

        ctx.enqueue_function[broadcast_add[out_layout, a_layout, b_layout]](
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
