# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
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


# ANCHOR: softmax_gpu_kernel
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
    # FILL IN (roughly 31 lines)


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
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
    # FILL IN (roughly 10 lines)


# ANCHOR_END: softmax_cpu_kernel

import compiler

from extensibility import InputTensor, OutputTensor


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
        ctx: DeviceContext,
    ) raises:
        var output_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](output.unsafe_ptr(), layout)
        var input_tensor = TileTensor[
            mut=True, dtype, LayoutType, MutAnyOrigin
        ](input.unsafe_ptr(), layout)

        comptime if target == "gpu":
            var gpu_ctx = ctx
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
            gpu_ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                grid_dim=1,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[input_size, dtype](output_tensor, input_tensor)
        else:
            raise Error("Unsupported target: " + target)
