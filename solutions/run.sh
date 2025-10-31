#!/bin/bash

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

# Unicode symbols
CHECK_MARK="✓"
CROSS_MARK="✗"
ARROW="→"
BULLET="•"

# Global counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Global options
VERBOSE_MODE=true
IGNORE_LOW_COMPUTE_FAILURES=false

# Puzzles that require higher compute capability (>= 8.0 on NVIDIA)
NVIDIA_HIGH_COMPUTE_REQUIRED_PUZZLES=("p16" "p19" "p28" "p29" "p33" "p34")

# Arrays to store results
declare -a FAILED_TESTS_LIST
declare -a PASSED_TESTS_LIST
declare -a SKIPPED_TESTS_LIST
declare -a IGNORED_LOW_COMPUTE_TESTS_LIST

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

    # Method 1: Try nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_name" ]; then
            # Check for H100 or other SM90+ GPUs
            if echo "$gpu_name" | grep -qi "H100\|A100\|RTX 40[0-9][0-9]\|RTX 4090\|L40S"; then
                if echo "$gpu_name" | grep -qi "H100"; then
                    compute_capability="9.0"
                elif echo "$gpu_name" | grep -qi "RTX 40[0-9][0-9]\|RTX 4090\|L40S"; then
                    compute_capability="8.9"
                elif echo "$gpu_name" | grep -qi "A100"; then
                    compute_capability="8.0"
                fi
            fi
        fi
    fi

    # Method 2: Try Python with GPU detection script if available
    if [ -z "$compute_capability" ] && [ -f "../scripts/gpu_specs.py" ]; then
        local gpu_info=$(python3 ../scripts/gpu_specs.py 2>/dev/null | grep -i "compute capability" | head -1)
        if [ -n "$gpu_info" ]; then
            compute_capability=$(echo "$gpu_info" | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        fi
    fi

    echo "$compute_capability"
}

has_high_compute_capability() {
    local compute_cap=$(detect_gpu_compute_capability)
    if [ -n "$compute_cap" ]; then
        # Convert to comparable format (e.g., 8.0 -> 80, 7.5 -> 75)
        local major=$(echo "$compute_cap" | cut -d'.' -f1)
        local minor=$(echo "$compute_cap" | cut -d'.' -f2)
        local numeric_cap=$((major * 10 + minor))

        # Compute capability 8.0+ is considered high compute
        [ "$numeric_cap" -ge 80 ]
    else
        # If we can't detect, assume it's low compute
        false
    fi
}

is_nvidia_high_compute_required_puzzle() {
    local puzzle_name="$1"
    local gpu_platform=$(detect_gpu_platform)

    # Only apply compute capability restrictions to NVIDIA GPUs
    if [ "$gpu_platform" != "nvidia" ]; then
        return 1  # Not restricted for non-NVIDIA platforms
    fi

    for required_puzzle in "${NVIDIA_HIGH_COMPUTE_REQUIRED_PUZZLES[@]}"; do
        if [[ "$puzzle_name" == *"$required_puzzle"* ]]; then
            return 0
        fi
    done
    return 1
}

# Usage function
usage() {
    echo -e "${BOLD}${CYAN}Mojo GPU Puzzles Test Runner${NC}"
    echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 [OPTIONS] [PUZZLE_NAME] [FLAG]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${YELLOW}-v, --verbose${NC}                     Show output for all tests (not just failures)"
    echo -e "  ${YELLOW}--ignore-low-compute-failures${NC}     Ignore failures from NVIDIA puzzles requiring compute >=8.0 (p16, p28, p29, p33, p34)"
    echo -e "  ${YELLOW}-h, --help${NC}                        Show this help message"
    echo ""
    echo -e "${BOLD}Parameters:${NC}"
    echo -e "  ${YELLOW}PUZZLE_NAME${NC}      Optional puzzle name (e.g., p23, p14, etc.)"
    echo -e "  ${YELLOW}FLAG${NC}             Optional flag to pass to puzzle files (e.g., --double-buffer)"
    echo ""
    echo -e "${BOLD}Behavior:${NC}"
    echo -e "  ${BULLET} If no puzzle specified, runs all puzzles"
    echo -e "  ${BULLET} If no flag specified, runs all detected flags or no flag if none found"
    echo -e "  ${BULLET} Failed tests always show actual vs expected output"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${GREEN}$0${NC}                                    ${GRAY}# Run all puzzles${NC}"
    echo -e "  ${GREEN}$0 -v${NC}                                 ${GRAY}# Run all puzzles with verbose output${NC}"
    echo -e "  ${GREEN}$0 --ignore-low-compute-failures${NC}      ${GRAY}# Run all puzzles, ignore compute <8.0 failures (for T4/CI)${NC}"
    echo -e "  ${GREEN}$0 p23${NC}                                ${GRAY}# Run only p23 tests with all flags${NC}"
    echo -e "  ${GREEN}$0 p26 --double-buffer${NC}                ${GRAY}# Run p26 with specific flag${NC}"
    echo -e "  ${GREEN}$0 -v p26 --double-buffer${NC}             ${GRAY}# Run p26 with specific flag (verbose)${NC}"
}

# Helper functions for better output
print_header() {
    local title="$1"
    echo ""
    echo -e "${BOLD}${BLUE}╭─────────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}${BLUE}│${NC} ${BOLD}${WHITE}$title${NC}$(printf "%*s" $((78 - ${#title} + 2)) "")${BOLD}${BLUE}│${NC}"
    echo -e "${BOLD}${BLUE}╰─────────────────────────────────────────────────────────────────────────────────╯${NC}"
}

print_test_start() {
    local test_name="$1"
    local flag="$2"
    if [ -n "$flag" ]; then
        echo -e "  ${CYAN}${ARROW}${NC} Running ${YELLOW}$test_name${NC} with flag ${PURPLE}$flag${NC}"
    else
        echo -e "  ${CYAN}${ARROW}${NC} Running ${YELLOW}$test_name${NC}"
    fi
}

print_test_result() {
    local test_name="$1"
    local flag="$2"
    local result="$3"
    local full_name="${test_name}$([ -n "$flag" ] && echo " ($flag)" || echo "")"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$result" = "PASS" ]; then
        echo -e "    ${GREEN}${CHECK_MARK}${NC} ${GREEN}PASSED${NC} ${GRAY}$full_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        PASSED_TESTS_LIST+=("$full_name")
    elif [ "$result" = "FAIL" ]; then
        echo -e "    ${RED}${CROSS_MARK}${NC} ${RED}FAILED${NC} ${GRAY}$full_name${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TESTS_LIST+=("$full_name")
    elif [ "$result" = "SKIP" ]; then
        echo -e "    ${YELLOW}${BULLET}${NC} ${YELLOW}SKIPPED${NC} ${GRAY}$full_name${NC}"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        SKIPPED_TESTS_LIST+=("$full_name")
    fi
}

print_progress() {
    local current="$1"
    local total="$2"
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 5))
    local empty=$((20 - filled))

    printf "\r  ${GRAY}Progress: [${NC}"
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "${GRAY}] %d%% (%d/%d)${NC}" $percentage $current $total
}

# Helper function to check if a test should be skipped and handle execution
execute_or_skip_test() {
    local test_name="$1"
    local flag="$2"
    local cmd="$3"
    local full_name="${test_name}$([ -n "$flag" ] && echo " ($flag)" || echo "")"

    # Check if this should be skipped due to low compute capability BEFORE running
    if [ "$IGNORE_LOW_COMPUTE_FAILURES" = "true" ] && is_nvidia_high_compute_required_puzzle "$test_name" && ! has_high_compute_capability; then
        print_test_start "$test_name" "$flag"
        echo -e "    ${YELLOW}${BULLET}${NC} ${YELLOW}SKIPPED${NC} ${GRAY}$full_name${NC} ${PURPLE}(NVIDIA requires compute >=8.0)${NC}"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        IGNORED_LOW_COMPUTE_TESTS_LIST+=("$full_name")
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        return 0  # Skipped successfully
    else
        # Run the test normally
        print_test_start "$test_name" "$flag"
        if capture_output "$cmd" "$VERBOSE_MODE"; then
            print_test_result "$test_name" "$flag" "PASS"
        else
            print_test_result "$test_name" "$flag" "FAIL"
        fi
        return 0
    fi
}

capture_output() {
    local cmd="$1"
    local show_output="$2"  # Optional parameter to force showing output
    local output_file=$(mktemp)
    local error_file=$(mktemp)

    if eval "$cmd" > "$output_file" 2> "$error_file"; then
        # Test passed - optionally show output if requested
        if [ "$show_output" = "true" ] && [ -s "$output_file" ]; then
            echo -e "    ${GREEN}Output:${NC}"
            sed 's/^/      /' "$output_file"
        fi
        rm "$output_file" "$error_file"
        return 0
    else
        # Test failed - show both stdout and stderr
        echo -e "    ${RED}${BOLD}Test Failed!${NC}"

        if [ -s "$output_file" ]; then
            echo -e "    ${CYAN}Program Output:${NC}"
            # Look for various output patterns
            if grep -q -E "out:|expected:|actual:|result:" "$output_file"; then
                # Parse and format the output nicely
                while IFS= read -r line; do
                    if [[ "$line" =~ ^out:.*$ ]] || [[ "$line" =~ ^actual:.*$ ]] || [[ "$line" =~ ^result:.*$ ]]; then
                        # Extract the value after the colon
                        value="${line#*: }"
                        echo -e "      ${YELLOW}${BOLD}Actual:${NC}   ${value}"
                    elif [[ "$line" =~ ^expected:.*$ ]]; then
                        # Extract the value after the colon
                        value="${line#*: }"
                        echo -e "      ${GREEN}${BOLD}Expected:${NC} ${value}"
                    elif [[ "$line" =~ ^.*shape:.*$ ]]; then
                        echo -e "      ${PURPLE}${BOLD}Shape:${NC}    ${line#*shape: }"
                    elif [[ "$line" =~ ^Error.*$ ]] || [[ "$line" =~ ^.*error.*$ ]]; then
                        echo -e "      ${RED}${BOLD}Error:${NC}    $line"
                    elif [[ -n "$line" ]]; then
                        echo -e "      ${GRAY}$line${NC}"
                    fi
                done < "$output_file"
            else
                # Show regular output with indentation
                sed 's/^/      /' "$output_file"
            fi
        fi

        if [ -s "$error_file" ]; then
            echo -e "    ${RED}Error Output:${NC}"
            sed 's/^/      /' "$error_file"
        fi

        rm "$output_file" "$error_file"
        return 1
    fi
}

run_mojo_files() {
  local path_prefix="$1"
  local specific_flag="$2"

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      # If specific flag is provided, use only that flag
      if [ -n "$specific_flag" ]; then
        # Check if the file supports this flag
        if grep -q "argv()\[1\] == \"$specific_flag\"" "$f" || grep -q "test_type == \"$specific_flag\"" "$f"; then
          execute_or_skip_test "${path_prefix}$f" "$specific_flag" "mojo \"$f\" \"$specific_flag\""
        else
          print_test_result "${path_prefix}$f" "$specific_flag" "SKIP"
        fi
      else
        # Original behavior - detect and run all flags or no flag
        flags=$(grep -o 'argv()\[1\] == "--[^"]*"\|test_type == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

        if [ -z "$flags" ]; then
          execute_or_skip_test "${path_prefix}$f" "" "mojo \"$f\""
        else
          for flag in $flags; do
            execute_or_skip_test "${path_prefix}$f" "$flag" "mojo \"$f\" \"$flag\""
          done
        fi
      fi
    fi
  done
}

run_python_files() {
  local path_prefix="$1"
  local specific_flag="$2"

  for f in *.py; do
    if [ -f "$f" ]; then
      # If specific flag is provided, use only that flag
      if [ -n "$specific_flag" ]; then
        # Check if the file supports this flag
        if grep -q "sys\.argv\[1\] == \"$specific_flag\"" "$f"; then
          execute_or_skip_test "${path_prefix}$f" "$specific_flag" "python \"$f\" \"$specific_flag\""
        else
          print_test_result "${path_prefix}$f" "$specific_flag" "SKIP"
        fi
      else
        # Original behavior - detect and run all flags or no flag
        flags=$(grep -o 'sys\.argv\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

        if [ -z "$flags" ]; then
          execute_or_skip_test "${path_prefix}$f" "" "python \"$f\""
        else
          for flag in $flags; do
            execute_or_skip_test "${path_prefix}$f" "$flag" "python \"$f\" \"$flag\""
          done
        fi
      fi
    fi
  done
}

process_directory() {
  local path_prefix="$1"
  local specific_flag="$2"

  run_mojo_files "$path_prefix" "$specific_flag"
  run_python_files "$path_prefix" "$specific_flag"
}

# Parse command line arguments
SPECIFIC_PUZZLE=""
SPECIFIC_FLAG=""

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --ignore-low-compute-failures)
            IGNORE_LOW_COMPUTE_FAILURES=true
            shift
            ;;
        -*)
            echo -e "${RED}${BOLD}Error:${NC} Unknown option $1"
            usage
            exit 1
            ;;
        *)
            if [ -z "$SPECIFIC_PUZZLE" ]; then
                SPECIFIC_PUZZLE="$1"
            elif [ -z "$SPECIFIC_FLAG" ]; then
                SPECIFIC_FLAG="$1"
            else
                echo -e "${RED}${BOLD}Error:${NC} Too many arguments"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Auto-detect and enable ignoring low compute failures BEFORE changing directory
if [ "$IGNORE_LOW_COMPUTE_FAILURES" = "false" ]; then
    gpu_platform=$(detect_gpu_platform)
    if [ "$gpu_platform" = "nvidia" ] && ! has_high_compute_capability; then
        IGNORE_LOW_COMPUTE_FAILURES=true
        echo -e "${YELLOW}${BOLD}Auto-detected:${NC} NVIDIA GPU with compute capability <8.0"
        echo -e "Automatically ignoring failures for high compute puzzles (p16, p19, p28, p29, p33, p34)"
        echo ""
    fi
fi

cd solutions || exit 1

# Function to test a specific directory
test_puzzle_directory() {
    local dir="$1"
    local specific_flag="$2"

    if [ -n "$specific_flag" ]; then
        print_header "Testing ${dir} with flag: $specific_flag"
    else
        print_header "Testing ${dir}"
    fi

    cd "$dir" || return 1

    process_directory "${dir}" "$specific_flag"

    # Check for test directory and run mojo run (only if no specific flag)
    if [ -z "$specific_flag" ] && ([ -d "test" ] || [ -d "tests" ]); then
        echo ""
        command="mojo run -I . test/*.mojo"
        echo -e "  ${CYAN}${ARROW}${NC} Running ${YELLOW}${command}${NC} in ${PURPLE}${dir}${NC}"
        if capture_output "$command" "$VERBOSE_MODE"; then
            print_test_result "$command" "" "PASS"
        else
            print_test_result "$command" "" "FAIL"
        fi
    fi

    cd ..
}

# Function to print final summary
print_summary() {
    echo ""
    echo ""
    echo -e "${BOLD}${CYAN}╭─────────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}${CYAN}│${NC} ${BOLD}${WHITE}TEST SUMMARY${NC}$(printf "%*s" $((63 - 12)) "")${BOLD}${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}╰─────────────────────────────────────────────────────────────────────────────────╯${NC}"
    echo ""

    # Overall statistics
    echo -e "  ${BOLD}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "  ${GREEN}${BOLD}Passed:${NC} $PASSED_TESTS"
    echo -e "  ${RED}${BOLD}Failed:${NC} $FAILED_TESTS"
    echo -e "  ${YELLOW}${BOLD}Skipped:${NC} $SKIPPED_TESTS"
    echo ""

    # Success rate (considering skipped tests as successful)
    if [ $TOTAL_TESTS -gt 0 ]; then
        local effective_passed=$((PASSED_TESTS + SKIPPED_TESTS))
        local success_rate=$((effective_passed * 100 / TOTAL_TESTS))
        echo -e "  ${BOLD}Success Rate:${NC} ${success_rate}%"

        # Progress bar for success rate
        local filled=$((success_rate / 5))
        local empty=$((20 - filled))
        echo -n "  "
        printf "%*s" $filled | tr ' ' '█'
        printf "%*s" $empty | tr ' ' '░'
        echo ""
        echo ""
    fi

    # Show failed tests if any
    if [ $FAILED_TESTS -gt 0 ]; then
        echo -e "${RED}${BOLD}Failed Tests:${NC}"
        for test in "${FAILED_TESTS_LIST[@]}"; do
            echo -e "  ${RED}${CROSS_MARK}${NC} $test"
        done
        echo ""
    fi

    # Show skipped tests if any
    if [ $SKIPPED_TESTS -gt 0 ]; then
        echo -e "${YELLOW}${BOLD}Skipped Tests:${NC}"
        for test in "${SKIPPED_TESTS_LIST[@]}"; do
            echo -e "  ${YELLOW}${BULLET}${NC} $test"
        done
        echo ""
    fi

    # Show ignored low compute tests if any
    if [ ${#IGNORED_LOW_COMPUTE_TESTS_LIST[@]} -gt 0 ]; then
        echo -e "${PURPLE}${BOLD}Ignored High Compute Required Tests:${NC}"
        for test in "${IGNORED_LOW_COMPUTE_TESTS_LIST[@]}"; do
            echo -e "  ${PURPLE}${BULLET}${NC} $test ${GRAY}(requires compute capability >=8.0)${NC}"
        done
        echo ""
    fi

    # Final status
    if [ $FAILED_TESTS -eq 0 ]; then
        if [ ${#IGNORED_LOW_COMPUTE_TESTS_LIST[@]} -gt 0 ]; then
            echo -e "${GREEN}${BOLD}${CHECK_MARK} All tests passed!${NC} ${GRAY}(${#IGNORED_LOW_COMPUTE_TESTS_LIST[@]} skipped due to low compute capability)${NC}"
        else
            echo -e "${GREEN}${BOLD}${CHECK_MARK} All tests passed!${NC}"
        fi
    else
        echo -e "${RED}${BOLD}${CROSS_MARK} Some tests failed.${NC}"
    fi
    echo ""
}

# Add startup banner
print_startup_banner() {
    echo -e "${BOLD}${CYAN}╭─────────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}${CYAN}│${NC} ${BOLD}${WHITE}🔥 MOJO GPU PUZZLES TEST RUNNER${NC}$(printf "%*s" $((78 - 29)) "")${BOLD}${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}╰─────────────────────────────────────────────────────────────────────────────────╯${NC}"
    echo ""

    # Display GPU information
    local gpu_platform=$(detect_gpu_platform)
    local compute_cap=$(detect_gpu_compute_capability)
    local gpu_name=""

    echo -e "${BOLD}GPU Information:${NC}"
    echo -e "  ${BULLET} Platform: ${CYAN}$(echo "$gpu_platform" | tr '[:lower:]' '[:upper:]')${NC}"

    case "$gpu_platform" in
        "nvidia")
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [ -n "$gpu_name" ]; then
                echo -e "  ${BULLET} Device: ${CYAN}$gpu_name${NC}"
            fi
            if [ -n "$compute_cap" ]; then
                echo -e "  ${BULLET} Compute Capability: ${PURPLE}$compute_cap${NC}"
                if has_high_compute_capability; then
                    echo -e "  ${BULLET} High Compute Support: ${GREEN}Yes${NC} ${GRAY}(>=8.0)${NC}"
                else
                    echo -e "  ${BULLET} High Compute Support: ${YELLOW}No${NC} ${GRAY}(<8.0)${NC}"
                fi
            fi
            ;;
        "amd")
            if command -v rocm-smi >/dev/null 2>&1; then
                gpu_name=$(rocm-smi --showproductname 2>/dev/null | grep -v "=" | head -1 | xargs)
                if [ -n "$gpu_name" ]; then
                    echo -e "  ${BULLET} Device: ${CYAN}$gpu_name${NC}"
                fi
            fi
            echo -e "  ${BULLET} ROCm Support: ${GREEN}Available${NC}"
            ;;
        "apple")
            echo -e "  ${BULLET} Metal Support: ${GREEN}Available${NC}"
            ;;
        *)
            echo -e "  ${BULLET} Status: ${YELLOW}Unknown GPU platform${NC}"
            ;;
    esac

    if [ "$IGNORE_LOW_COMPUTE_FAILURES" = "true" ]; then
        echo -e "  ${BULLET} Low Compute Failure Mode: ${YELLOW}IGNORED${NC} ${GRAY}(--ignore-low-compute-failures enabled)${NC}"
    fi
    echo ""
}

# Show startup banner
print_startup_banner

# Record start time
START_TIME=$(date +%s)

if [ -n "$SPECIFIC_PUZZLE" ]; then
    # Run specific puzzle
    if [ -d "${SPECIFIC_PUZZLE}/" ]; then
        test_puzzle_directory "${SPECIFIC_PUZZLE}/" "$SPECIFIC_FLAG"
    else
        echo -e "${RED}${BOLD}Error:${NC} Puzzle directory '${SPECIFIC_PUZZLE}' not found"
        echo ""
        echo -e "${BOLD}Available puzzles:${NC}"
        for puzzle in $(ls -d p*/ 2>/dev/null | tr -d '/' | sort); do
            echo -e "  ${BULLET} ${CYAN}$puzzle${NC}"
        done
        exit 1
    fi
else
    # Run all puzzles (original behavior)
    puzzle_dirs=($(ls -d p*/ 2>/dev/null | sort))
    total_puzzles=${#puzzle_dirs[@]}
    current_puzzle=0

    for dir in "${puzzle_dirs[@]}"; do
        if [ -d "$dir" ]; then
            current_puzzle=$((current_puzzle + 1))
            test_puzzle_directory "$dir" "$SPECIFIC_FLAG"
        fi
    done
fi

cd ..

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

# Print summary
print_summary

# Show execution time
echo -e "${GRAY}Execution time: ${EXECUTION_TIME}s${NC}"
echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi
