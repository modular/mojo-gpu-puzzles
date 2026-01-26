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
detect_gpu_platform() {
    # Detect GPU platform: nvidia, amd, apple, or unknown
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_name" ]; then
            echo "nvidia"
            return
        fi
    fi

    # Check for AMD ROCm
    if command -v rocm-smi >/dev/null 2>&1; then
        if rocm-smi --showproductname >/dev/null 2>&1; then
            echo "amd"
            return
        fi
    fi

    # Check for Apple Silicon (macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple"; then
            echo "apple"
            return
        fi
    fi

    echo "unknown"
}

detect_gpu_compute_capability() {
    # Try to detect NVIDIA GPU compute capability
    local compute_capability=""

    # Method 1: Try nvidia-smi with expanded GPU list
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_name" ]; then
            # Check for known GPU families and their compute capabilities
            if echo "$gpu_name" | grep -qi "H100"; then
                compute_capability="9.0"
            elif echo "$gpu_name" | grep -qi "RTX 40[0-9][0-9]\|RTX 4090\|L40S\|L4\|RTX 2000 Ada Generation"; then
                compute_capability="8.9"
            elif echo "$gpu_name" | grep -qi "RTX 30[0-9][0-9]\|RTX 3090\|RTX 3080\|RTX 3070\|RTX 3060\|A40\|A30\|A10"; then
                compute_capability="8.6"
            elif echo "$gpu_name" | grep -qi "A100"; then
                compute_capability="8.0"
            elif echo "$gpu_name" | grep -qi "V100"; then
                compute_capability="7.0"
            elif echo "$gpu_name" | grep -qi "T4\|RTX 20[0-9][0-9]\|RTX 2080\|RTX 2070\|RTX 2060"; then
                compute_capability="7.5"
            fi
        fi
    fi

    echo "$compute_capability"
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
