"""Quick test to verify origin fix approaches."""
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout

comptime SIZE = 16
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime dtype = DType.float32

def fn_requires_mut_any[LayoutT: TensorLayout](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin],
):
    _ = output.to_layout_tensor()
    _ = input.to_layout_tensor()

def main() raises:
    with DeviceContext() as ctx:
        var out_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        var in_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        # Approach 3: explicit type annotation on variable declaration
        var out_tensor: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin] = TileTensor(out_buf, layout)
        var in_tensor: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin] = TileTensor[mut=False, dtype, LayoutType](in_buf, layout)
        fn_requires_mut_any[LayoutType](out_tensor, in_tensor)
        print("Approach 3 (explicit type annotation) works!")
