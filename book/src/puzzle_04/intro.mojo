from std.gpu.host import DeviceContext
from layout import Layout, TileTensor

comptime HEIGHT = 2
comptime WIDTH = 3
comptime dtype = DType.float32
comptime layout = Layout.row_major(HEIGHT, WIDTH)


def kernel[
    dtype: DType, layout: Layout
](tensor: TileTensor[dtype, layout, MutAnyOrigin]):
    print("Before:")
    print(tensor)
    tensor[0, 0] += 1
    print("After:")
    print(tensor)


def main() raises:
    ctx = DeviceContext()

    a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
    a.enqueue_fill(0)
    tensor = TileTensor[dtype, layout, MutAnyOrigin](a)
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function[kernel[dtype, layout], kernel[dtype, layout]](
        tensor, grid_dim=1, block_dim=1
    )

    ctx.synchronize()
