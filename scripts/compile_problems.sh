#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# Compile every problem skeleton (the unfilled puzzle starting points).
#
# CI only ever runs the SOLUTIONS (solutions/run.sh), so a problem skeleton can
# silently rot: an API migration that updates the solutions but not the
# skeletons leaves learners with starter code that no longer compiles (see
# issue #254). This script closes that gap by compiling each skeleton:
#   - custom-op modules (problems/pNN/op/)  -> `mojo package`  (no main())
#   - standalone single-file puzzles        -> `mojo build`    (have main())
# An unfilled skeleton is expected to compile (it just produces wrong/zero
# output until the learner fills the `# FILL ME IN` gaps); it must never fail to
# parse or type-check.
#
# GPU-arch-gated puzzles are skipped using the same logic as the test runner
# (solutions/config.sh).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/solutions/config.sh"

cd "${REPO_ROOT}" || exit 1

PLATFORM="$(detect_gpu_platform)"
COMPUTE_CAP="$(detect_gpu_compute_capability)"
echo "Detected GPU platform: ${PLATFORM} (compute capability: ${COMPUTE_CAP:-n/a})"

pass=0
fail=0
skipped=0
failures=()

compile_target() {
    local label="$1"
    shift
    if mojo "$@" >/tmp/compile_problems_out 2>&1; then
        echo "  PASS  ${label}"
        pass=$((pass + 1))
    else
        echo "  FAIL  ${label}"
        sed 's/^/        /' /tmp/compile_problems_out | grep -E 'error:|Error' | head -5
        fail=$((fail + 1))
        failures+=("${label}")
    fi
}

# Only compile skeletons that have a matching solution (the drift candidates).
for sol_dir in solutions/p*/; do
    puzzle="$(basename "${sol_dir}")"
    prob_dir="problems/${puzzle}"
    [ -d "${prob_dir}" ] || continue

    # Skip puzzles unsupported on the detected platform, mirroring
    # solutions/run.sh (config.sh provides the arrays + is_in_array).
    if [ "${PLATFORM}" = "amd" ] && is_in_array "${puzzle}" "${AMD_UNSUPPORTED_PUZZLES[@]}"; then
        echo "SKIP  ${puzzle}: not supported on AMD GPU"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "${PLATFORM}" = "apple" ] && is_in_array "${puzzle}" "${APPLE_UNSUPPORTED_PUZZLES[@]}"; then
        echo "SKIP  ${puzzle}: not supported on Apple GPU"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "${PLATFORM}" = "nvidia" ]; then
        reason="$(should_skip_puzzle "${puzzle}" "${COMPUTE_CAP}")"
        if [ -n "${reason}" ]; then
            echo "SKIP  ${puzzle}: ${reason}"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    echo "${puzzle}:"
    if [ -d "${prob_dir}/op" ]; then
        # Custom-op module: compile the package (no main()).
        compile_target "${prob_dir}/op" package "${prob_dir}/op" -o /tmp/skel.mojopkg
        continue
    fi
    # Standalone puzzle files: compile each .mojo that defines a main().
    while IFS= read -r f; do
        if grep -qE '^def main\(|^fn main\(' "${f}"; then
            compile_target "${f}" build "${f}" -o /tmp/skel.o
        fi
    done < <(find "${prob_dir}" -maxdepth 1 -name '*.mojo' | sort)
done

echo
echo "=== compile-problems: ${pass} passed, ${fail} failed, ${skipped} skipped ==="
if [ "${fail}" -gt 0 ]; then
    echo "Failed skeletons:"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
