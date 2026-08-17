#!/bin/bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##
# Refuse to run a puzzle the current GPU cannot support, and say why.
#
# Usage:  gate.sh <puzzle> <command> [args...]
#
# The compute-capability requirements live in config.sh and are shared with
# solutions/run.sh, so the suite and the per-puzzle tasks cannot disagree.
# Without this, a puzzle using an instruction the GPU lacks reaches LLVM
# instruction selection and aborts with a message asking the reader to file a
# compiler bug.
#
# Wraps the command rather than chaining before it: `poe` forwards trailing
# arguments only for single `cmd` tasks, and drops them silently for sequence
# and shell tasks, which would strip a puzzle's mode flag.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

puzzle="${1:-}"
if [ -z "$puzzle" ] || [ "$#" -lt 2 ]; then
    echo "usage: $(basename "$0") <puzzle> <command> [args...]" >&2
    exit 2
fi
shift

# Membership is a pure-shell test, so an ungated puzzle never pays for GPU
# detection — which shells out to python and pynvml.
if ! is_in_array "$puzzle" "${NVIDIA_COMPUTE_80_REQUIRED_PUZZLES[@]}" \
   && ! is_in_array "$puzzle" "${NVIDIA_COMPUTE_90_REQUIRED_PUZZLES[@]}"; then
    exec "$@"
fi

platform=$(detect_gpu_platform)

# Detection needs pynvml, which arrives with the `nvidia` extra. Without it
# `gpu_specs.py` reports "unknown" and exits 0, so the gate cannot tell an AMD
# or Apple machine from an NVIDIA one with a missing dependency. Running is the
# right default — refusing would block every non-NVIDIA reader — but say so,
# rather than letting a supported-looking run be an unchecked one.
if [ "$platform" = "unknown" ]; then
    echo "  Note: cannot identify this GPU, so ${puzzle}'s hardware" \
         "requirement was not checked." >&2
    echo "  If this is an NVIDIA GPU, install the nvidia extra to enable the" \
         "check." >&2
    exec "$@"
fi

# The requirements are NVIDIA compute capabilities, so they say nothing about
# AMD or Apple GPUs. Those have their own lists, applied by run.sh.
if [ "$platform" != "nvidia" ]; then
    exec "$@"
fi

capability=$(detect_gpu_compute_capability)
reason=$(should_skip_puzzle "$puzzle" "$capability")
if [ -z "$reason" ]; then
    exec "$@"
fi

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
needs_80=$(IFS=,; echo "${NVIDIA_COMPUTE_80_REQUIRED_PUZZLES[*]}")
needs_90=$(IFS=,; echo "${NVIDIA_COMPUTE_90_REQUIRED_PUZZLES[*]}")

cat >&2 <<MESSAGE

  ${puzzle} ${reason}.
  Detected: ${gpu_name:-unknown GPU}, compute capability ${capability:-unknown}.

  This puzzle has not been run.

  Needs compute 8.0 (Ampere) or newer: ${needs_80}
  Needs compute 9.0 (Hopper) or newer: ${needs_90}
  Every other puzzle runs on this GPU.

  Support matrix: book/src/howto.md
  To run it anyway: MOJO_PUZZLES_IGNORE_GPU_GATE=1 <your command>

MESSAGE

if [ "${MOJO_PUZZLES_IGNORE_GPU_GATE:-0}" != "1" ]; then
    exit 1
fi

echo "  MOJO_PUZZLES_IGNORE_GPU_GATE=1 is set, running anyway." >&2
exec "$@"
