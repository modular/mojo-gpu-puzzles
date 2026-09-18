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
"""Tests that the canary stays quiet on a correct kernel and fires on a write
past either end of the output."""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer
from std.testing import assert_false, assert_true

from harness.canary import PuzzleMemory

comptime dtype = DType.float32
comptime SIZE = 8


def write_in_bounds(output: Pointer[Scalar[dtype], MutAnyOrigin]):
    output[unsafe_offset=Int(thread_idx.x)] = 1.0


def write_one_past_the_end(output: Pointer[Scalar[dtype], MutAnyOrigin]):
    output[unsafe_offset=Int(thread_idx.x)] = 1.0
    if thread_idx.x == 0:
        output[unsafe_offset=SIZE] = 1.0


def write_one_before_the_start(output: Pointer[Scalar[dtype], MutAnyOrigin]):
    output[unsafe_offset=Int(thread_idx.x)] = 1.0
    if thread_idx.x == 0:
        output[unsafe_offset=-1] = 1.0


def test_correct_kernel_leaves_margins_intact() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(SIZE)
        ctx.enqueue_function[write_in_bounds](out, grid_dim=1, block_dim=SIZE)
        ctx.synchronize()
        assert_true(mem.margins_intact())


def test_write_past_the_end_is_detected() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(SIZE)
        ctx.enqueue_function[write_one_past_the_end](
            out, grid_dim=1, block_dim=SIZE
        )
        ctx.synchronize()
        assert_false(mem.margins_intact())


def test_write_before_the_start_is_detected() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(SIZE)
        ctx.enqueue_function[write_one_before_the_start](
            out, grid_dim=1, block_dim=SIZE
        )
        ctx.synchronize()
        assert_false(mem.margins_intact())


def main() raises:
    test_correct_kernel_leaves_margins_intact()
    test_write_past_the_end_is_detected()
    test_write_before_the_start_is_detected()
    print("Canary harness tests passed ✅")
