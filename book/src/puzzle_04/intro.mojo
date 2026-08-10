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
from max.gpu.host import DeviceContext
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
    var ctx = DeviceContext()

    var a = ctx.enqueue_create_buffer[dtype](HEIGHT * WIDTH)
    a.enqueue_fill(0)
    var tensor = TileTensor(a, layout)
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function[kernel](tensor, grid_dim=1, block_dim=1)

    ctx.synchronize()
