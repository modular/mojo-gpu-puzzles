from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite


alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)


# ANCHOR: softmax_gpu_kernel_solution
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        thread_max = rebind[Scalar[dtype]](input[global_i])

    shared_max[local_i] = thread_max
    barrier()

    # Parallel reduction to find max similar to reduction we saw before
    # Note we need to avoid race conditions by reading the value first and then writing
    stride = TPB // 2
    while stride > 0:
        var temp_max: Scalar[dtype] = min_finite[dtype]()
        if local_i < stride:
            temp_max = rebind[Scalar[dtype]](shared_max[local_i + stride])
        barrier()
        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i], temp_max)
        barrier()
        stride = stride // 2

    block_max = shared_max[0]

    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_i] - block_max))
        output[global_i] = exp_val

    shared_sum[local_i] = exp_val
    barrier()

    # Parallel reduction for sum similar to reduction we saw before
    # Note we need to avoid race conditions by reading the value first and then writing
    stride = TPB // 2
    while stride > 0:
        var temp_sum: Scalar[dtype] = 0.0
        if local_i < stride:
            temp_sum = rebind[Scalar[dtype]](shared_sum[local_i + stride])
        barrier()
        if local_i < stride:
            shared_sum[local_i] += temp_sum
        barrier()
        stride = stride // 2

    block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < input_size:
        output[global_i] = output[global_i] / block_sum


# ANCHOR_END: softmax_gpu_kernel_solution


# ANCHOR: softmax_cpu_kernel_solution
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
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
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())

        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            # gpu_ctx.enqueue_memset(
            #     DeviceBuffer[output_tensor.type](
            #         gpu_ctx,
            #         rebind[UnsafePointer[Scalar[output_tensor.type]]](out_tensor.ptr),
            #         input_size,
            #         owning=False,
            #     ),
            #     0,
            # )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
