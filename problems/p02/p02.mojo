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
from std.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

# ANCHOR: add
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32


def add(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = thread_idx.x
    output[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


# ANCHOR_END: add


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Scalar[dtype](i)
                b_host[i] = Scalar[dtype](i)
                expected[i] = a_host[i] + b_host[i]

        ctx.enqueue_function[add](
            out,
            a,
            b,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 02 complete ✅")
