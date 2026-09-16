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
"""Guarded output tensors for the puzzles that drive a kernel from Python.

This is the Python version of ``harness/canary.mojo``, for the puzzles that
allocate the output tensor here and pass it into the custom op. A puzzle whose
output comes from the graph runtime through ``out_types`` cannot use this,
because nothing in Python owns that allocation.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

import torch

MARGIN_BYTES = 256


def guarded_output(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, Callable[[], None]]:
    """Allocates an output tensor with a guarded margin on each side.

    Args:
        shape: Shape the kernel expects to write.
        dtype: Element type of the output.
        device: Device the output lives on.

    Returns:
        The tensor to hand to the kernel, and a check to call once the kernel
        has run. The check prints a message and exits with a non-zero status
        if either margin was written to.
    """
    margin = MARGIN_BYTES // torch.empty((), dtype=dtype).element_size()
    count = 1
    for dim in shape:
        count *= dim

    padded = torch.full(
        (count + 2 * margin,), float("nan"), dtype=dtype, device=device
    )
    view = padded[margin : margin + count].view(*shape)
    view.zero_()

    def check() -> None:
        intact = bool(
            padded[:margin].isnan().all()
            and padded[margin + count :].isnan().all()
        )
        if not intact:
            print("❌ Write detected outside of output buffer")
            sys.exit(1)

    return view, check
