# same as p15/op/conv1d.mojo
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.utils import Index
from std.sys import argv
from std.testing import assert_equal

comptime TPB = 15
comptime BLOCKS_PER_GRID = (2, 1)


def conv1d_kernel[
    input_size: Int,
    conv_size: Int,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    ConvLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, InLayout, MutAnyOrigin],
    kernel: TileTensor[mut=True, dtype, ConvLayout, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var input_lt = input.to_layout_tensor()
    var kernel_lt = kernel.to_layout_tensor()
    var output_lt = output.to_layout_tensor()
    # first: need to account for padding
    var shared_a = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB + conv_size - 1]())
    var shared_b = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[conv_size]())
    if global_i < input_size:
        shared_a[local_i] = rebind[Scalar[dtype]](input_lt[global_i])

    # second: load elements needed for convolution at block boundary
    if local_i < conv_size - 1:
        # indices from next block
        var next_idx = global_i + TPB
        if next_idx < input_size:
            shared_a[TPB + local_i] = rebind[Scalar[dtype]](input_lt[next_idx])
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0

    if local_i < conv_size:
        shared_b[local_i] = rebind[Scalar[dtype]](kernel_lt[local_i])

    barrier()

    if global_i < input_size:
        var local_sum: Scalar[dtype] = 0

        comptime for j in range(conv_size):
            if local_i + j < TPB + conv_size - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output_lt.store[1](Index(global_i), local_sum)


import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceBuffer


@compiler.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
        input_size: Int,
        conv_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        input: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        kernel: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        comptime out_layout_val = row_major[input_size]()
        comptime OutLayout = type_of(out_layout_val)
        comptime conv_layout_val = row_major[conv_size]()
        comptime ConvLayout = type_of(conv_layout_val)

        var out_tensor = TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin](
            output.unsafe_ptr(), out_layout_val
        )
        var input_tensor = TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin](
            input.unsafe_ptr(), out_layout_val
        )
        var kernel_tensor = TileTensor[
            mut=True, dtype, ConvLayout, MutAnyOrigin
        ](kernel.unsafe_ptr(), conv_layout_val)

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    output.unsafe_ptr(),
                    input_size,
                    owning=False,
                ),
                0,
            )
            # ANCHOR: conv1d_custom_op_solution
            comptime kernel = conv1d_kernel[
                input_size, conv_size, OutLayout, OutLayout, ConvLayout
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                out_tensor,
                input_tensor,
                kernel_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )
            # ANCHOR_END: conv1d_custom_op_solution
        elif target == "cpu":
            # we can fallback to CPU
            pass
        else:
            raise Error("Unsupported target: " + target)
