## How to Use This Book

Each puzzle follows a consistent format designed to progressively build your skills:

- **Overview**: Clear problem statement and key concepts introduced in each puzzle
- **Configuration**: Setup parameters and memory organization specific to each challenge
- **Code to Complete**: Skeleton code with specific sections for you to implement
- **Tips**: Optional hints if you get stuck, without giving away complete solutions
- **Solution**: Detailed explanations of the implementation, performance considerations, and underlying concepts

The puzzles gradually increase in complexity, introducing new concepts while reinforcing fundamentals. We recommend solving them in order, as later puzzles build on skills developed in earlier ones.

## Running the code

All puzzles are designed to be run with the provided testing framework that verifies your implementation against expected results. Each puzzle includes instructions for running the code and validating your solution.

## Prerequisites

### System requirements

Make sure your system meets our [system requirements](https://docs.modular.com/max/packages#system-requirements).

### Compatible GPU

You'll need a [compatible GPU](https://docs.modular.com/max/faq#gpu-requirements) to run the puzzles.

#### macOS Apple Sillicon (Early preview)

For `osx-arm64` users, you'll need:

- **macOS 15.0 or later** for optimal compatibility
- **Xcode 16 or later** (minimum required). Use `xcodebuild -version` to check.

If `xcrun -sdk macosx metal` outputs `cannot execite tool 'metal' due to missing Metal toolchain` proceed by running

```bash
xcodebuild -downloadComponent MetalToolchain
```

and then `xcrun -sdk macosx metal`, should give you the `no input files error`.

> **Note**: Currently the puzzles 1-8 and 11-15 are working on macOS. We're working to enable more. Please stay tuned!

### Programming knowledge

Basic knowledge of:

- Programming fundamentals (variables, loops, conditionals, functions)
- Parallel computing concepts (threads, synchronization, race conditions)
- Basic familiarity with [Mojo](https://docs.modular.com/mojo/manual/) (language basics parts and [intro to pointers](https://docs.modular.com/mojo/manual/pointers/) section)
- [GPU programming fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals) is helpful!

No prior GPU programming experience is necessary! We'll build that knowledge through the puzzles.

Let's begin our journey into the exciting world of GPU computing with Mojo🔥!

## Setting up your environment

1. [Clone the GitHub repository](https://github.com/modular/mojo-gpu-puzzles) and navigate to the repository:

    ```bash
    # Clone the repository
    git clone https://github.com/modular/mojo-gpu-puzzles
    cd mojo-gpu-puzzles
    ```

2. Install a package manager to run the Mojo🔥 programs:

   #### **Option 1 (Hightly recommended)**: [pixi](https://pixi.sh/latest/#installation)

    `pixi` is the **recommended option** for this project because:
    - Easy access to Modular's MAX/Mojo packages
    - Handles GPU dependencies
    - Full conda + PyPI ecosystem support

    > **Note: Some puzzles only work with `pixi`**

    **Install:**

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    **Update:**

    ```bash
    pixi self-update
    ```

   #### Option 2: [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

    **Install:**

    ```bash
    curl -fsSL https://astral.sh/uv/install.sh | sh
    ```

    **Update:**

    ```bash
    uv self update
    ```

    **Create a virtual environment:**

    ```bash
    uv venv && source .venv/bin/activate
    ```

3. Run the puzzles via `pixi` or `uv` as follows:

    <div class="code-tabs" data-tab-group="package-manager">
      <div class="tab-buttons">
        <button class="tab-button">pixi</button>
        <button class="tab-button">uv</button>
      </div>
      <div class="tab-content">

    ```bash
    pixi run pXX  # Replace XX with the puzzle number
    ```

      </div>
      <div class="tab-content">

    ```bash
    uv run poe pXX  # Replace XX with the puzzle number
    ```

      </div>
    </div>

For example, to run puzzle 01:

- `pixi run p01` or
- `uv run poe p01`

## GPU support matrix

The following table shows GPU platform compatibility for each puzzle. Different puzzles require different GPU features and vendor-specific tools.

| Puzzle | NVIDIA GPU | AMD GPU | Apple GPU | Notes |
|--------|------------|---------|-----------|-------|
| **Part I: GPU Fundamentals** | | | | |
| 1 - Map | ✅ | ✅ | ✅ | Basic GPU kernels |
| 2 - Zip | ✅ | ✅ | ✅ | Basic GPU kernels |
| 3 - Guard | ✅ | ✅ | ✅ | Basic GPU kernels |
| 4 - Map 2D | ✅ | ✅ | ✅ | Basic GPU kernels |
| 5 - Broadcast | ✅ | ✅ | ✅ | Basic GPU kernels |
| 6 - Blocks | ✅ | ✅ | ✅ | Basic GPU kernels |
| 7 - Shared Memory | ✅ | ✅ | ✅ | Basic GPU kernels |
| 8 - Stencil | ✅ | ✅ | ✅ | Basic GPU kernels |
| **Part II: Debugging** | | | | |
| 9 - GPU Debugger | ✅ | ❌ | ❌ | NVIDIA-specific debugging tools |
| 10 - Sanitizer | ✅ | ❌ | ❌ | NVIDIA-specific debugging tools |
| **Part III: GPU Algorithms** | | | | |
| 11 - Reduction | ✅ | ✅ | ✅ | Basic GPU kernels |
| 12 - Scan | ✅ | ✅ | ✅ | Basic GPU kernels |
| 13 - Pool | ✅ | ✅ | ✅ | Basic GPU kernels |
| 14 - Conv | ✅ | ✅ | ✅ | Basic GPU kernels |
| 15 - Matmul | ✅ | ✅ | ✅ | Basic GPU kernels |
| 16 - Flashdot | ✅ | ✅ | ❌ | Advanced memory patterns |
| **Part IV: MAX Graph** | | | | |
| 17 - Custom Op | ✅ | ✅ | ❌ | MAX Graph integration |
| 18 - Softmax | ✅ | ✅ | ❌ | MAX Graph integration |
| 19 - Attention | ✅ | ✅ | ❌ | MAX Graph integration |
| **Part V: PyTorch Integration** | | | | |
| 20 - Torch Bridge | ✅ | ✅ | ❌ | PyTorch integration |
| 21 - Autograd | ✅ | ✅ | ❌ | PyTorch integration |
| 22 - Fusion | ✅ | ✅ | ❌ | PyTorch integration |
| **Part VI: Functional Patterns** | | | | |
| 23 - Functional | ✅ | ✅ | ❌ | Advanced Mojo patterns |
| **Part VII: Warp Programming** | | | | |
| 24 - Warp Sum | ✅ | ✅ | ❌ | Warp-level operations |
| 25 - Warp Communication | ✅ | ✅ | ❌ | Warp-level operations |
| 26 - Advanced Warp | ✅ | ✅ | ❌ | Warp-level operations |
| **Part VIII: Block Programming** | | | | |
| 27 - Block Operations | ✅ | ✅ | ❌ | Block-level patterns |
| **Part IX: Memory Systems** | | | | |
| 28 - Async Memory | ✅ | ✅ | ❌ | Advanced memory operations |
| 29 - Barriers | ✅ | ✅ | ❌ | Advanced synchronization |
| **Part X: Performance Analysis** | | | | |
| 30 - Profiling | ✅ | ❌ | ❌ | NVIDIA profiling tools (NSight) |
| 31 - Occupancy | ✅ | ❌ | ❌ | NVIDIA profiling tools |
| 32 - Bank Conflicts | ✅ | ❌ | ❌ | NVIDIA profiling tools |
| **Part XI: Modern GPU Features** | | | | |
| 33 - Tensor Cores | ✅ | ❌ | ❌ | NVIDIA Tensor Core specific |
| 34 - Cluster | ✅ | ❌ | ❌ | NVIDIA cluster programming |

### Legend

- ✅ **Supported**: Puzzle works on this platform
- ❌ **Not Supported**: Puzzle requires platform-specific features

### Platform Notes

**NVIDIA GPUs (Complete Support)**

- All puzzles (1-34) work on NVIDIA GPUs with CUDA support
- Requires CUDA toolkit and compatible drivers
- Best learning experience with access to all features

**AMD GPUs (Extensive Support)**

- Most puzzles (1-8, 11-29) work with ROCm support
- Missing only: Debugging tools (9-10), profiling (30-32), Tensor Cores (33-34)
- Excellent for learning GPU programming including advanced algorithms and memory patterns

**Apple GPUs (Basic Support)**

- Only fundamental puzzles (1-8, 11-15) supported
- Missing: All advanced features, debugging, profiling tools
- Suitable for learning basic GPU programming patterns

> **Future Support**: We're actively working to expand tooling and platform support for AMD and Apple GPUs. Missing features like debugging tools, profiling capabilities, and advanced GPU operations are planned for future releases. Check back for updates as we continue to broaden cross-platform compatibility.

### Recommendations

- **Complete Learning Path**: Use NVIDIA GPU for full curriculum access (all 34 puzzles)
- **Comprehensive Learning**: AMD GPUs excellent for most content (27 of 34 puzzles)
- **Basic Understanding**: Apple GPUs suitable for fundamental concepts (13 of 34 puzzles)
- **Debugging & Profiling**: NVIDIA GPU required for debugging tools and performance analysis
- **Modern GPU Features**: NVIDIA GPU required for Tensor Cores and cluster programming

## Development

Please see details in the [README](https://github.com/modular/mojo-gpu-puzzles#development).

## Join the community

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://www.modular.com/company/talk-to-us">
    <img src="https://img.shields.io/badge/Subscribe-Updates-00B5AD?logo=mail.ru" alt="Subscribe for Updates">
  </a>
  <a href="https://forum.modular.com/c/">
    <img src="https://img.shields.io/badge/Modular-Forum-9B59B6?logo=discourse" alt="Modular Forum">
  </a>
  <a href="https://discord.com/channels/1087530497313357884/1098713601386233997">
    <img src="https://img.shields.io/badge/Discord-Join_Chat-5865F2?logo=discord" alt="Discord">
  </a>
</p>

Join our vibrant community to discuss GPU programming, share solutions, and get help!
