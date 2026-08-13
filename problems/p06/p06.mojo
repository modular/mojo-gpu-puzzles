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
from std.memory import Pointer
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.testing import assert_equal

# ANCHOR: add_10_blocks
comptime SIZE = 9
comptime BLOCKS_PER_GRID = (3, 1)
comptime THREADS_PER_BLOCK = (4, 1)
comptime dtype = DType.float32


def add_10_blocks(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0


# ANCHOR_END: add_10_blocks


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i)

        ctx.enqueue_function[add_10_blocks](
            out,
            a,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)

        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = Scalar[dtype](i + 10)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 06 complete ✅")
