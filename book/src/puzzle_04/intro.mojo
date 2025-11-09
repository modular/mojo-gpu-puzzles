from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias HEIGHT = 2
alias WIDTH = 3
alias dtype = DType.float32
alias layout = Layout.row_major(HEIGHT, WIDTH)


fn kernel[
    dtype: DType, layout: Layout
](tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
    print("Before:")
    print(tensor)
    tensor[0, 0] += 1
    print("After:")
    print(tensor)


def main():
    ctx = DeviceContext()

    a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
    a.enqueue_fill(0)
    tensor = LayoutTensor[dtype, layout, MutAnyOrigin](a)
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function_checked[kernel[dtype, layout], kernel[dtype, layout]](
        tensor, grid_dim=1, block_dim=1
    )

    ctx.synchronize()
