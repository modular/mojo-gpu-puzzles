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
from max.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal

# ANCHOR: add_10_shared
comptime TPB = 4
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (2, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def add_10_shared_tile_tensor(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    # Allocate shared memory. Unlike most kernel variables, which are private to
    # each thread, shared memory (`AddressSpace.SHARED`) is scoped per-block.
    # So, each thread in a block can use the same "scratchpad" to store data.
    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = a[global_i]

    # Note: barrier is not strictly needed here since each thread only accesses
    # its own shared memory location. However, it's included to teach proper
    # shared memory synchronization patterns for more complex scenarios where
    # threads need to coordinate access to shared data.
    barrier()

    if global_i < size:
        output[global_i] = shared[local_i] + 10.0


# ANCHOR_END: add_10_shared


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)

        var out_tensor = TileTensor(out, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)

        ctx.enqueue_function[add_10_shared_tile_tensor](
            out_tensor,
            a_tensor,
            Int32(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(11)
        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 08 complete ✅")
