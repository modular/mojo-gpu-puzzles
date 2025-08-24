from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: add_10_2d_layout_tensor
alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn add_10_2d(
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=True, dtype, layout],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_2d_layout_tensor


def main():
    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        out_tensor = LayoutTensor[mut=True, dtype, layout](out_buf.unsafe_ptr())
        print("out shape:", out_tensor.shape[0](), "x", out_tensor.shape[1]())

        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)

        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE * SIZE):
                a_host[i] = i
                expected[i] = a_host[i] + 10

        a_tensor = LayoutTensor[mut=True, dtype, layout](a.unsafe_ptr())

        ctx.enqueue_function[add_10_2d](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out_buf.map_to_host() as out_buf_host:
            print("out:", out_buf_host)
            print("expected:", expected)
            for i in range(SIZE * SIZE):
                assert_equal(out_buf_host[i], expected[i])
