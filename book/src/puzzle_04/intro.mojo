from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major

comptime HEIGHT = 2
comptime WIDTH = 3
comptime dtype = DType.float32
comptime layout = row_major[HEIGHT, WIDTH]()
comptime LayoutType = type_of(layout)


def kernel(
    tensor: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    print("Before:")
    print(tensor)
    tensor[0, 0] += 1
    print("After:")
    print(tensor)


def main() raises:
    ctx = DeviceContext()

    a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
    a.enqueue_fill(0)
    tensor = TileTensor(a, layout)
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function[kernel, kernel](
        tensor, grid_dim=1, block_dim=1
    )

    ctx.synchronize()
