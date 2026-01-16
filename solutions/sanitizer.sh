#!/bin/bash

# Source shared configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Detect GPU platform and compute capability at startup
GPU_PLATFORM=$(detect_gpu_platform)
GPU_COMPUTE_CAP=$(detect_gpu_compute_capability)

if [ "$GPU_PLATFORM" != "nvidia" ]; then
    echo -e "${RED}Error: compute-sanitizer is only available for NVIDIA GPUs${NC}"
    echo "Detected platform: $GPU_PLATFORM"
    exit 1
fi

echo "Detected NVIDIA GPU with compute capability: ${GPU_COMPUTE_CAP:-unknown}"
echo ""

# Usage function
usage() {
    echo "Usage: $0 <tool> [PUZZLE_NAME] [FLAG]"
    echo "  tool: Required sanitizer tool (memcheck, racecheck, synccheck, initcheck, all)"
    echo "  PUZZLE_NAME: Optional puzzle name (e.g., p23, p14, etc.)"
    echo "  FLAG: Optional flag to pass to mojo files (e.g., --async-copy-overlap)"
    echo "  If no puzzle specified, runs tool on all puzzles"
    echo "  If no flag specified, runs all detected flags or no flag if none found"
    echo ""
    echo "Examples:"
    echo "  $0 racecheck                              # Run racecheck on all puzzles"
    echo "  $0 racecheck p25                          # Run racecheck on p25 with all flags"
    echo "  $0 racecheck p25 --async-copy-overlap     # Run racecheck on p25 with specific flag"
    echo "  $0 all                                    # Run all sanitizers on all puzzles"
    echo "  $0 all p25                                # Run all sanitizers on p25 with all flags"
    echo "  $0 all p25 --tma-coordination             # Run all sanitizers on p25 with specific flag"
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

TOOL="$1"
SPECIFIC_PUZZLE="$2"
SPECIFIC_FLAG="$3"

case "$TOOL" in
    memcheck|racecheck|synccheck|initcheck|all)
        ;;
    *)
        echo "Error: Invalid tool '$TOOL'"
        echo "Available tools: memcheck, racecheck, synccheck, initcheck, all"
        exit 1
        ;;
esac

# Handle "all" tool option - run all sanitizers on all puzzles or specific puzzle
if [ "$TOOL" = "all" ]; then
    TOOLS=("memcheck" "racecheck" "synccheck" "initcheck")

    if [ -z "$SPECIFIC_PUZZLE" ]; then
        # Run all sanitizers on all puzzles
        echo "Running all compute-sanitizer tools on ALL puzzles"
        echo "======================================================================="

        for CURRENT_TOOL in "${TOOLS[@]}"; do
            echo ""
            echo "Running $CURRENT_TOOL on all puzzles..."
            echo "-----------------------------------"
            bash "$0" "$CURRENT_TOOL"
            echo ""
        done

        echo "======================================================================="
        echo "All sanitizer tools completed for all puzzles"
        exit 0
    fi

    # Run all sanitizers on specific puzzle
    if [ -n "$SPECIFIC_FLAG" ]; then
        echo "Running all compute-sanitizer tools on $SPECIFIC_PUZZLE with flag: $SPECIFIC_FLAG"
    else
        echo "Running all compute-sanitizer tools on $SPECIFIC_PUZZLE (all detected flags)"
    fi
    echo "======================================================================="

    for CURRENT_TOOL in "${TOOLS[@]}"; do
        echo ""
        if [ -n "$SPECIFIC_FLAG" ]; then
            echo "Running $CURRENT_TOOL on $SPECIFIC_PUZZLE with flag: $SPECIFIC_FLAG..."
        else
            echo "Running $CURRENT_TOOL on $SPECIFIC_PUZZLE (all detected flags)..."
        fi
        echo "-----------------------------------"
        # Recursively call this script with individual tool and flag
        if [ -n "$SPECIFIC_FLAG" ]; then
            bash "$0" "$CURRENT_TOOL" "$SPECIFIC_PUZZLE" "$SPECIFIC_FLAG"
        else
            bash "$0" "$CURRENT_TOOL" "$SPECIFIC_PUZZLE"
        fi
        echo ""
    done

    echo "======================================================================="
    echo "All sanitizer tools completed for $SPECIFIC_PUZZLE"
    exit 0
fi

case "$TOOL" in
    memcheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Memory Error|========= Invalid|========= Out of bounds|out:|expected:)"
        ;;
    racecheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Race|========= Hazard|out:|expected:)"
        ;;
    synccheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Sync|========= Deadlock|out:|expected:)"
        ;;
    initcheck)
        GREP_PATTERN="(========= COMPUTE-SANITIZER|========= ERROR SUMMARY|========= Uninitialized|========= Unused|out:|expected:)"
        ;;
esac

TOTAL_ERRORS=0

run_mojo_files_with_sanitizer() {
  local path_prefix="$1"
  local tool="$2"
  local grep_pattern="$3"
  local specific_flag="$4"

  for f in *.mojo; do
    if [ -f "$f" ] && [ "$f" != "__init__.mojo" ]; then
      # If specific flag is provided, use only that flag
      if [ -n "$specific_flag" ]; then
        # Check if the file supports this flag (check both patterns)
        if grep -q "argv()\[1\] == \"$specific_flag\"" "$f" || grep -q "test_type == \"$specific_flag\"" "$f"; then
          echo "=== Running compute-sanitizer --tool $tool on ${path_prefix}$f with flag: $specific_flag ==="
          output=$(compute-sanitizer --tool "$tool" mojo "$f" "$specific_flag" 2>&1)
          filtered_output=$(echo "$output" | grep -E "$grep_pattern")

          error_count=$(echo "$output" | grep -E "========= ERROR SUMMARY: [0-9]+ error" | sed -n 's/.*ERROR SUMMARY: \([0-9]\+\) error.*/\1/p' | head -n1)

          if [ -n "$error_count" ] && [ "$error_count" -gt 0 ]; then
            echo -e "${RED}FOUND $error_count ERRORS!${NC}"
            TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
          fi

          if [ -n "$filtered_output" ]; then
            echo "$filtered_output"
          else
            echo "Failed: compute-sanitizer $tool ${path_prefix}$f with $specific_flag"
          fi
        else
          echo "Skipping ${path_prefix}$f - does not support flag: $specific_flag"
        fi
      else
        # Original behavior - detect and run all flags or no flag
        # Check both patterns: argv()[1] == "--flag" and test_type == "--flag"
        flags1=$(grep -o 'argv()\[1\] == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')
        flags2=$(grep -o 'test_type == "--[^"]*"' "$f" | cut -d'"' -f2 | grep -v '^--demo')
        flags=$(echo -e "$flags1\n$flags2" | sort -u | grep -v '^$')

        if [ -z "$flags" ]; then
          echo "No flags detected for ${path_prefix}$f"
          echo "=== Running compute-sanitizer --tool $tool on ${path_prefix}$f ==="
          output=$(compute-sanitizer --tool "$tool" mojo "$f" 2>&1)
          filtered_output=$(echo "$output" | grep -E "$grep_pattern")

          error_count=$(echo "$output" | grep -E "========= ERROR SUMMARY: [0-9]+ error" | sed -n 's/.*ERROR SUMMARY: \([0-9]\+\) error.*/\1/p' | head -n1)

          if [ -n "$error_count" ] && [ "$error_count" -gt 0 ]; then
            echo -e "${RED}FOUND $error_count ERRORS!${NC}"
            TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
          fi

          if [ -n "$filtered_output" ]; then
            echo "$filtered_output"
          else
            echo "Failed: compute-sanitizer $tool ${path_prefix}$f"
          fi
        else
          echo "Detected flags for ${path_prefix}$f: $flags"
          for flag in $flags; do
            echo "=== Running compute-sanitizer --tool $tool on ${path_prefix}$f with flag: $flag ==="
            output=$(compute-sanitizer --tool "$tool" mojo "$f" "$flag" 2>&1)
            filtered_output=$(echo "$output" | grep -E "$grep_pattern")

            error_count=$(echo "$output" | grep -E "========= ERROR SUMMARY: [0-9]+ error" | sed -n 's/.*ERROR SUMMARY: \([0-9]\+\) error.*/\1/p' | head -n1)

            if [ -n "$error_count" ] && [ "$error_count" -gt 0 ]; then
              echo -e "${RED}FOUND $error_count ERRORS!${NC}"
              TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
            fi

            if [ -n "$filtered_output" ]; then
              echo "$filtered_output"
            else
              echo "Failed: compute-sanitizer $tool ${path_prefix}$f with $flag"
            fi
          done
        fi
      fi
    fi
  done
}

cd solutions || exit 1

SKIPPED_PUZZLES=0

# Function to test a specific directory
test_puzzle_directory() {
    local dir="$1"
    # Extract puzzle name (remove trailing slash)
    local puzzle_name="${dir%/}"

    # Check compute capability requirements
    local skip_reason=$(should_skip_puzzle "$puzzle_name" "$GPU_COMPUTE_CAP")
    if [ -n "$skip_reason" ]; then
        echo -e "${YELLOW}=== SKIPPING ${puzzle_name}: ${skip_reason} ===${NC}"
        SKIPPED_PUZZLES=$((SKIPPED_PUZZLES + 1))
        return 0
    fi

    if [ -n "$SPECIFIC_FLAG" ]; then
        echo "=== Running compute-sanitizer $TOOL on solutions in ${dir} with flag: $SPECIFIC_FLAG ==="
    else
        echo "=== Running compute-sanitizer $TOOL on solutions in ${dir} (all detected flags) ==="
    fi
    cd "$dir" || return 1

    run_mojo_files_with_sanitizer "$dir" "$TOOL" "$GREP_PATTERN" "$SPECIFIC_FLAG"

    cd ..
}

if [ -n "$SPECIFIC_PUZZLE" ]; then
    # Run specific puzzle
    if [ -d "${SPECIFIC_PUZZLE}/" ]; then
        test_puzzle_directory "${SPECIFIC_PUZZLE}/"
    else
        echo "Error: Puzzle directory '${SPECIFIC_PUZZLE}' not found"
        echo "Available puzzles:"
        ls -d p*/ 2>/dev/null | tr -d '/' | sort
        exit 1
    fi
else
    # Run all puzzles (original behavior)
    for dir in p*/; do
        if [ -d "$dir" ]; then
            test_puzzle_directory "$dir"
        fi
    done
fi

cd ..

echo ""
echo "========================================"
if [ "$SKIPPED_PUZZLES" -gt 0 ]; then
  echo -e "${YELLOW}SKIPPED PUZZLES: $SKIPPED_PUZZLES (insufficient compute capability)${NC}"
fi
if [ "$TOTAL_ERRORS" -gt 0 ]; then
  echo -e "${RED}TOTAL ERRORS FOUND: $TOTAL_ERRORS${NC}"
  echo -e "${YELLOW}Please review the errors above and fix them.${NC}"
  exit 1
else
  echo -e "${GREEN}NO ERRORS FOUND! All tests passed clean.${NC}"
  exit 0
fi
