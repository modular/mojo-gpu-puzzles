# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
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
    # Single-block softmax: block max -> exp -> block sum -> normalize
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[BLOCK_DIM_X]())
    var tid = thread_idx.x

    # Phase 1: block-wide max (out-of-range threads contribute the identity)
    if tid < input_size:
        shared[tid] = input[tid]
    else:
        shared[tid] = min_finite[dtype]()
    barrier()

    var stride = BLOCK_DIM_X // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = max(shared[tid], shared[tid + stride])
        barrier()
        stride //= 2
    var block_max = shared[0]

    # Phase 2: exp(x - max), then block-wide sum
    if tid < input_size:
        shared[tid] = exp(input[tid] - block_max)
    else:
        shared[tid] = 0
    barrier()

    stride = BLOCK_DIM_X // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2
    var total = shared[0]

    # Phase 3: normalize
    if tid < input_size:
        output[tid] = exp(input[tid] - block_max) / total


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
    # Sequential softmax: max, then exp/sum, then normalize
    var m = min_finite[dtype]()
    for i in range(input_size):
        m = max(m, input[i])

    var total = Scalar[dtype](0)
    for i in range(input_size):
        var e = exp(input[i] - m)
        output[i] = e
        total += e

    for i in range(input_size):
        output[i] = output[i] / total


# ANCHOR_END: softmax_cpu_kernel

import extensibility

from extensibility import InputTensor, OutputTensor


@extensibility.register("softmax")
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
