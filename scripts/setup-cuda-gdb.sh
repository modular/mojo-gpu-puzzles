#!/bin/bash
#
# Setup CUDA-GDB by linking system CUDA installation binaries to conda environment
#
# This script auto-detects CUDA installation from:
# - $CUDA_HOME environment variable
# - /usr/local/cuda (Ubuntu/Debian default)
# - /opt/cuda (ArchLinux and other distros)
# - System PATH (via `which cuda-gdb`)
#
# Usage: bash scripts/setup-cuda-gdb.sh
#

set -e

CUDA_BIN=""

echo "Detecting CUDA installation..."

# Try CUDA_HOME first
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/bin" ]; then
    CUDA_BIN="$CUDA_HOME/bin"
    echo "  ✓ Found CUDA via \$CUDA_HOME: $CUDA_BIN"
# Try common paths
elif [ -d "/usr/local/cuda/bin" ]; then
    CUDA_BIN="/usr/local/cuda/bin"
    echo "  ✓ Found CUDA at: $CUDA_BIN"
elif [ -d "/opt/cuda/bin" ]; then
    CUDA_BIN="/opt/cuda/bin"
    echo "  ✓ Found CUDA at: $CUDA_BIN"
# Try to find cuda-gdb in PATH
elif command -v cuda-gdb >/dev/null 2>&1; then
    CUDA_BIN=$(dirname "$(readlink -f "$(command -v cuda-gdb)")")
    echo "  ✓ Found CUDA via PATH: $CUDA_BIN"
fi

if [ -z "$CUDA_BIN" ]; then
    echo ""
    echo "Error: Could not find CUDA installation."
    echo ""
    echo "Please ensure CUDA is installed and try one of the following:"
    echo "  1. Set CUDA_HOME: export CUDA_HOME=/path/to/cuda"
    echo "  2. Install CUDA to /usr/local/cuda (Ubuntu/Debian)"
    echo "  3. Install CUDA to /opt/cuda (ArchLinux)"
    echo "  4. Add cuda-gdb to your PATH"
    echo ""
    exit 1
fi

if [ -z "$CONDA_PREFIX" ]; then
    echo ""
    echo "Error: \$CONDA_PREFIX not set. Are you in a pixi/conda environment?"
    echo "Run this via: pixi run setup-cuda-gdb"
    echo ""
    exit 1
fi

echo ""
echo "Linking CUDA-GDB binaries to conda environment..."
echo "  Target: $CONDA_PREFIX/bin/"

LINKED_COUNT=0

# Link cuda-gdb-minimal
if [ -f "$CUDA_BIN/cuda-gdb-minimal" ]; then
    ln -sf "$CUDA_BIN/cuda-gdb-minimal" "$CONDA_PREFIX/bin/cuda-gdb-minimal"
    echo "  ✓ cuda-gdb-minimal"
    LINKED_COUNT=$((LINKED_COUNT + 1))
else
    echo "  ⚠ cuda-gdb-minimal not found (optional)"
fi

# Link cuda-gdb with Python TUI support for different Python versions
for pyver in 3.12 3.11 3.10 3.9 3.8; do
    binary_name="cuda-gdb-python${pyver}-tui"
    if [ -f "$CUDA_BIN/$binary_name" ]; then
        ln -sf "$CUDA_BIN/$binary_name" "$CONDA_PREFIX/bin/$binary_name"
        echo "  ✓ $binary_name"
        LINKED_COUNT=$((LINKED_COUNT + 1))
    fi
done

echo ""
if [ $LINKED_COUNT -eq 0 ]; then
    echo "⚠ Warning: No cuda-gdb binaries were linked"
    echo "  Check that CUDA Toolkit is properly installed at: $CUDA_BIN"
    exit 1
else
    echo "✓ CUDA-GDB setup complete ($LINKED_COUNT binaries linked)"
    echo ""
    echo "You can now use: pixi run cuda-gdb --version"
fi
