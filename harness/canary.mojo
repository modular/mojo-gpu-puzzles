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
"""Guarded output buffers for the puzzles.

Each output buffer sits inside a larger allocation. The elements on either side
of it hold NaN, so a kernel that writes past the end of its output leaves a
real number behind and `verify` reports it. A kernel can write every correct
value into `output` and still write outside it, which comparing values cannot
detect.

Detection reaches `MARGIN` elements past each end. A write further out than
that misses the margins altogether and goes unnoticed.
"""

from std.math import isnan, nan
from std.sys import exit, size_of

from max.gpu.host import DeviceBuffer, DeviceContext


struct PuzzleMemory[dtype: DType]:
    """Allocates guarded output buffers and checks their margins.

    Parameters:
        dtype: Element type of the puzzle's buffers.
    """

    # A whole 256-byte block, so the kernel's buffer stays 256-byte aligned.
    comptime MARGIN = 256 // size_of[Scalar[Self.dtype]]()
    """Number of elements held on each side of an output buffer."""

    var _ctx: DeviceContext
    var _allocations: List[DeviceBuffer[Self.dtype]]

    def __init__(out self, ctx: DeviceContext):
        """Initializes the harness against a device context.

        Args:
            ctx: The context whose device the buffers are allocated on.
        """
        self._ctx = ctx
        self._allocations = []

    def output(mut self, size: Int) raises -> DeviceBuffer[Self.dtype]:
        """Allocates a zeroed output buffer with a margin on each side.

        Args:
            size: Number of elements the puzzle's kernel may write.

        Returns:
            A view of exactly `size` elements. The margins lie outside this
            view, so a kernel reaches them only by writing out of bounds.

        Raises:
            If the device allocation or either fill fails.
        """
        var allocation = self._ctx.enqueue_create_buffer[Self.dtype](
            size + 2 * Self.MARGIN
        )
        allocation.enqueue_fill(nan[Self.dtype]())
        var view = allocation.create_sub_buffer[Self.dtype](Self.MARGIN, size)
        view.enqueue_fill(0)
        self._allocations.append(allocation)
        return view

    def verify(self) raises:
        """Fails the puzzle if any output buffer's margin was written to.

        Prints a message and exits with a non-zero status.

        Raises:
            If reading a buffer back from the device fails.
        """
        if not self.margins_intact():
            print("❌ Write detected outside of output buffer")
            exit(1)

    def margins_intact(self) raises -> Bool:
        """Checks whether every output buffer's margins still hold NaN.

        Returns:
            True if no margin was written to.

        Raises:
            If reading a buffer back from the device fails.
        """
        for i in range(len(self._allocations)):
            var allocation = self._allocations[i]
            var size = len(allocation) - 2 * Self.MARGIN
            with allocation.map_to_host() as host:
                for j in range(Self.MARGIN):
                    if not isnan(host[j]):
                        return False
                    if not isnan(host[Self.MARGIN + size + j]):
                        return False
        return True
