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

# Arrays to store results
declare -a FAILED_TESTS_LIST
declare -a PASSED_TESTS_LIST
declare -a SKIPPED_TESTS_LIST

# Usage function
usage() {
    echo -e "${BOLD}${CYAN}Mojo GPU Puzzles Test Runner${NC}"
    echo -e "${GRAY}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 [OPTIONS] [PUZZLE_NAME] [FLAG]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${YELLOW}-v, --verbose${NC}    Show output for all tests (not just failures)"
    echo -e "  ${YELLOW}-h, --help${NC}       Show this help message"
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
    echo -e "  ${GREEN}$0${NC}                              ${GRAY}# Run all puzzles${NC}"
    echo -e "  ${GREEN}$0 -v${NC}                           ${GRAY}# Run all puzzles with verbose output${NC}"
    echo -e "  ${GREEN}$0 p23${NC}                          ${GRAY}# Run only p23 tests with all flags${NC}"
    echo -e "  ${GREEN}$0 p26 --double-buffer${NC}          ${GRAY}# Run p26 with specific flag${NC}"
    echo -e "  ${GREEN}$0 -v p26 --double-buffer${NC}       ${GRAY}# Run p26 with specific flag (verbose)${NC}"
}

# Helper functions for better output
print_header() {
    local title="$1"
    echo ""
    echo -e "${BOLD}${BLUE}╭─────────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}${BLUE}│${NC} ${BOLD}${WHITE}$title${NC}$(printf "%*s" $((75 - ${#title})) "")${BOLD}${BLUE}│${NC}"
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
          print_test_start "${path_prefix}$f" "$specific_flag"
          if capture_output "mojo \"$f\" \"$specific_flag\"" "$VERBOSE_MODE"; then
            print_test_result "${path_prefix}$f" "$specific_flag" "PASS"
          else
            print_test_result "${path_prefix}$f" "$specific_flag" "FAIL"
          fi
        else
          print_test_result "${path_prefix}$f" "$specific_flag" "SKIP"
        fi
      else
        # Original behavior - detect and run all flags or no flag
        flags=$(grep -o 'argv()\[1\] == "--[^"]*"\|test_type == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

        if [ -z "$flags" ]; then
          print_test_start "${path_prefix}$f" ""
          if capture_output "mojo \"$f\"" "$VERBOSE_MODE"; then
            print_test_result "${path_prefix}$f" "" "PASS"
          else
            print_test_result "${path_prefix}$f" "" "FAIL"
          fi
        else
          for flag in $flags; do
            print_test_start "${path_prefix}$f" "$flag"
            if capture_output "mojo \"$f\" \"$flag\"" "$VERBOSE_MODE"; then
              print_test_result "${path_prefix}$f" "$flag" "PASS"
            else
              print_test_result "${path_prefix}$f" "$flag" "FAIL"
            fi
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
          print_test_start "${path_prefix}$f" "$specific_flag"
          if capture_output "python \"$f\" \"$specific_flag\"" "$VERBOSE_MODE"; then
            print_test_result "${path_prefix}$f" "$specific_flag" "PASS"
          else
            print_test_result "${path_prefix}$f" "$specific_flag" "FAIL"
          fi
        else
          print_test_result "${path_prefix}$f" "$specific_flag" "SKIP"
        fi
      else
        # Original behavior - detect and run all flags or no flag
        flags=$(grep -o 'sys\.argv\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')

        if [ -z "$flags" ]; then
          print_test_start "${path_prefix}$f" ""
          if capture_output "python \"$f\"" "$VERBOSE_MODE"; then
            print_test_result "${path_prefix}$f" "" "PASS"
          else
            print_test_result "${path_prefix}$f" "" "FAIL"
          fi
        else
          for flag in $flags; do
            print_test_start "${path_prefix}$f" "$flag"
            if capture_output "python \"$f\" \"$flag\"" "$VERBOSE_MODE"; then
              print_test_result "${path_prefix}$f" "$flag" "PASS"
            else
              print_test_result "${path_prefix}$f" "$flag" "FAIL"
            fi
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

    # Check for test directory and run mojo test (only if no specific flag)
    if [ -z "$specific_flag" ] && ([ -d "test" ] || [ -d "tests" ]); then
        echo ""
        echo -e "  ${CYAN}${ARROW}${NC} Running ${YELLOW}mojo test${NC} in ${PURPLE}${dir}${NC}"
        if capture_output "mojo test ." "$VERBOSE_MODE"; then
            print_test_result "mojo test" "" "PASS"
        else
            print_test_result "mojo test" "" "FAIL"
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

    # Success rate
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
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

    # Final status
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}${BOLD}${CHECK_MARK} All tests passed!${NC}"
    else
        echo -e "${RED}${BOLD}${CROSS_MARK} Some tests failed.${NC}"
    fi
    echo ""
}

# Add startup banner
print_startup_banner() {
    echo -e "${BOLD}${CYAN}╭─────────────────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}${CYAN}│${NC} ${BOLD}${WHITE}🔥 MOJO GPU PUZZLES TEST RUNNER${NC}$(printf "%*s" $((47 - 29)) "")${BOLD}${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}╰─────────────────────────────────────────────────────────────────────────────────╯${NC}"
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
