from std.memory import UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.math import exp
from std.bit import log2_ceil
from std.utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


# ANCHOR: softmax_gpu_kernel_solution
def softmax_gpu_kernel[
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    var shared_max = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[BLOCK_DIM_X]())
    var shared_sum = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[BLOCK_DIM_X]())
    var global_i = thread_idx.x

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[global_i])
    shared_max[global_i] = val

    barrier()

    # Parallel reduction to find max similar to reduction we saw before
    var stride = BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    var block_max = shared_max[0]

    # Initialize out-of-bounds (shared_max[global_i], global_i >= input_size) shared memory addresses to 0.0,
    # ensuring that if these elements are accessed in the parallel sum reduction below they
    # do not influence the result (adding 0.0 does not change the sum).
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[global_i] = exp_val
    barrier()

    # Parallel reduction for sum similar to reduction we saw before
    stride = BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride = stride // 2

    var block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < input_size:
        output[global_i] = exp_val / block_sum


# ANCHOR_END: softmax_gpu_kernel_solution


# ANCHOR: softmax_cpu_kernel_solution
def softmax_cpu_kernel[
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    var max_val: Scalar[dtype] = min_finite[dtype]()
    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    var sum_exp: Scalar[dtype] = 0.0
    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel_solution

import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        input: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        var output_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](output.unsafe_ptr(), layout)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](input.unsafe_ptr(), layout)

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[dtype](
                    gpu_ctx,
                    output.unsafe_ptr(),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[input_size, dtype]
            gpu_ctx.enqueue_function[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=1,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[input_size, dtype](output_tensor, input_tensor)
        else:
            raise Error("Unsupported target: " + target)
