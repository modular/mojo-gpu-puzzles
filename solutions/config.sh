#!/bin/bash
# Shared configuration for test runner and sanitizer scripts

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Puzzles that require higher compute capability on NVIDIA
# >= 8.0 (Ampere): Tensor Cores, full async copy (RTX 30xx, A100+)
NVIDIA_COMPUTE_80_REQUIRED_PUZZLES=("p16" "p19" "p22" "p28" "p29" "p33")
# >= 9.0 (Hopper): SM90+ cluster programming (H100+)
NVIDIA_COMPUTE_90_REQUIRED_PUZZLES=("p34")

# Puzzles that are not supported on AMD GPUs
AMD_UNSUPPORTED_PUZZLES=("p09" "p10" "p30" "p31" "p32" "p33" "p34")

# Puzzles that are not supported on Apple GPUs
APPLE_UNSUPPORTED_PUZZLES=("p09" "p10" "p20" "p21" "p22" "p29" "p30" "p31" "p32" "p33" "p34")

# GPU detection functions
# These now use scripts/gpu_specs.py as the single source of truth
detect_gpu_platform() {
    # Detect GPU platform: nvidia, amd, apple, or unknown
    # Uses gpu_specs.py for accurate cross-platform detection

    # Try to find gpu_specs.py relative to this config file
    # BASH_SOURCE[0] in a sourced file points to the sourced file itself
    local config_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    local gpu_specs_path="${config_dir}/../scripts/gpu_specs.py"

    if [ -f "$gpu_specs_path" ]; then
        python3 "$gpu_specs_path" --platform 2>/dev/null || echo "unknown"
    else
        # Fallback: try from current working directory
        if [ -f "scripts/gpu_specs.py" ]; then
            python3 "scripts/gpu_specs.py" --platform 2>/dev/null || echo "unknown"
        else
            echo "unknown"
        fi
    fi
}

detect_gpu_compute_capability() {
    # Detect NVIDIA GPU compute capability (returns empty for non-NVIDIA)
    # Uses gpu_specs.py with pynvml for accurate detection

    # Try to find gpu_specs.py relative to this config file
    local config_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    local gpu_specs_path="${config_dir}/../scripts/gpu_specs.py"

    if [ -f "$gpu_specs_path" ]; then
        python3 "$gpu_specs_path" --compute-cap 2>/dev/null
    else
        # Fallback: try from current working directory
        if [ -f "scripts/gpu_specs.py" ]; then
            python3 "scripts/gpu_specs.py" --compute-cap 2>/dev/null
        else
            echo ""
        fi
    fi
}

# Check if a puzzle is in an array
is_in_array() {
    local element="$1"
    shift
    local arr=("$@")
    for item in "${arr[@]}"; do
        if [[ "$item" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

# Check if puzzle should be skipped based on compute capability
should_skip_puzzle() {
    local puzzle_name="$1"
    local compute_capability="$2"

    # Check compute 9.0 requirements
    if is_in_array "$puzzle_name" "${NVIDIA_COMPUTE_90_REQUIRED_PUZZLES[@]}"; then
        if [[ -z "$compute_capability" ]] || (( $(echo "$compute_capability < 9.0" | bc -l) )); then
            echo "requires compute capability >= 9.0 (Hopper)"
            return 0
        fi
    fi

    # Check compute 8.0 requirements
    if is_in_array "$puzzle_name" "${NVIDIA_COMPUTE_80_REQUIRED_PUZZLES[@]}"; then
        if [[ -z "$compute_capability" ]] || (( $(echo "$compute_capability < 8.0" | bc -l) )); then
            echo "requires compute capability >= 8.0 (Ampere)"
            return 0
        fi
    fi

    echo ""
    return 1
}
