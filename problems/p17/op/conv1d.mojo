from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation

# ANCHOR: conv1d_kernel
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
    input: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    kernel: TileTensor[mut=False, dtype, ConvLayout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # first: need to account for padding
    var shared_a = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB + conv_size - 1]())
    var shared_b = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[conv_size]())
    if global_i < input_size:
        shared_a[local_i] = input[global_i]

    # second: load elements needed for convolution at block boundary
    if local_i < conv_size - 1:
        # indices from next block
        var next_idx = global_i + TPB
        if next_idx < input_size:
            shared_a[TPB + local_i] = input[next_idx]
        else:
            shared_a[TPB + local_i] = 0

    if local_i < conv_size:
        shared_b[local_i] = kernel[local_i]

    barrier()

    if global_i < input_size:
        var local_sum: output.ElementType = 0

        comptime for j in range(conv_size):
            if local_i + j < TPB + conv_size - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum


# ANCHOR_END: conv1d_kernel


# ANCHOR: conv1d_custom_op
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
        output: OutputTensor[rank=1, static_spec=_],
        input: InputTensor[rank=output.rank, static_spec=_],
        kernel: InputTensor[rank=output.rank, static_spec=_],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        var output_tensor = output.to_layout_tensor()
        var input_tensor = input.to_layout_tensor()
        var kernel_tensor = kernel.to_layout_tensor()

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    output_tensor.ptr,
                    input_size,
                    owning=False,
                ),
                0,
            )

            # FILL ME IN with 1 line calling our conv1d_kernel

        elif target == "cpu":
            # we can fallback to CPU
            pass
        else:
            raise Error("Unsupported target: " + target)


# ANCHOR_END: conv1d_custom_op
