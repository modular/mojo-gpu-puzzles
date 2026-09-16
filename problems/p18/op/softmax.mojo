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
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import TileTensor, Coord
from layout.tensor_engine import TensorEngine
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.math import exp
from std.bit import log2_ceil
from std.utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


# ANCHOR: softmax_gpu_kernel
def softmax_gpu_kernel[
    input_size: Int,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    Engine: TensorEngine,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    input: TileTensor[mut=True, dtype, InLayout, MutAnyOrigin, Engine=Engine],
) where (Engine.element_size == 1):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    # FILL IN (roughly 33 lines)


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
def softmax_cpu_kernel[
    input_size: Int,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    Engine: TensorEngine,
    dtype: DType = .float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    input: TileTensor[mut=True, dtype, InLayout, MutAnyOrigin, Engine=Engine],
) where (Engine.element_size == 1):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    # FILL IN (roughly 12 lines)


# ANCHOR_END: softmax_cpu_kernel

import extensibility

from extensibility import InputTensor, OutputTensor


@extensibility.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = .float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        input: InputTensor[dtype=dtype, rank=output.rank, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var output_tensor = output.to_tile_tensor().as_unsafe_any_origin()
        var input_tensor = input.to_tile_tensor().as_unsafe_any_origin()
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

            comptime kernel = softmax_gpu_kernel[
                input_size,
                output_tensor.LayoutType,
                input_tensor.LayoutType,
                output_tensor.Engine,
                dtype,
            ]
            gpu_ctx.enqueue_function[kernel](
                output_tensor,
                input_tensor,
                grid_dim=1,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[
                input_size,
                output_tensor.LayoutType,
                input_tensor.LayoutType,
                output_tensor.Engine,
                dtype,
            ](output_tensor, input_tensor)
        else:
            raise Error("Unsupported target: " + target)
