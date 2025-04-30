from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias HEIGHT = 2
alias WIDTH = 3
alias dtype = DType.float32
alias layout = Layout.row_major(HEIGHT, WIDTH)

fn kernel[dtype: DType, layout: Layout](tensor: LayoutTensor[mut=True, dtype, layout]):
    print("Before:")
    print(tensor)
    tensor[0, 0] += 1
    print("After:")
    print(tensor)

def main():
    ctx = DeviceContext()

    a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH).enqueue_fill(0)
    tensor = LayoutTensor[mut=True, dtype, layout](a.unsafe_ptr())
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function[kernel[dtype, layout]](tensor, grid_dim=1, block_dim=1)

    ctx.synchronize()
