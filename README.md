<p align="center">
  <img src="book/src/puzzles_images/puzzle-mark.svg" alt="Mojo GPU Puzzles Logo" width="150">
</p>

<p align="center">
  <h1 align="center">Mojo🔥 GPU Puzzles</h1>
</p>

<p align="center">
  <h3 align="center">Learn GPU Programming in Mojo🔥 Through Interactive Puzzles🧩</h3>
</p>

<p align="center">
  <a href="#overview"><strong>Overview</strong></a> •
  <a href="#why-mojo"><strong>Why Mojo</strong></a> •
  <a href="#getting-started"><strong>Getting Started</strong></a> •
  <a href="#development"><strong>Development</strong></a> •
  <a href="#community"><strong>Community</strong></a>
</p>

<p align="center">
  <a href="https://github.com/modular/mojo-gpu-puzzles/actions/workflows/ci.yml">
    <img src="https://github.com/modular/mojo-gpu-puzzles/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <a href="https://docs.modular.com/mojo">
    <img src="https://img.shields.io/badge/Powered%20by-Mojo-FF5F1F" alt="Powered by Mojo">
  </a>
  <a href="https://docs.modular.com/max/get-started/#stay-in-touch">
    <img src="https://img.shields.io/badge/Subscribe-Updates-00B5AD?logo=mail.ru" alt="Subscribe for Updates">
  </a>
  <a href="https://forum.modular.com/c/">
    <img src="https://img.shields.io/badge/Modular-Forum-9B59B6?logo=discourse" alt="Modular Forum">
  </a>
  <a href="https://discord.com/channels/1087530497313357884/1098713601386233997">
    <img src="https://img.shields.io/badge/Discord-Join_Chat-5865F2?logo=discord" alt="Discord">
  </a>
</p>

## Overview

> _"For the things we have to learn before we can do them, we learn by doing them."_
> — Aristotle, (Nicomachean Ethics)

Welcome to **Mojo🔥 GPU Puzzles, Edition 1** — an interactive approach to learning GPU programming through hands-on puzzle solving. Instead of traditional textbook learning, you'll immediately dive into writing real GPU code and seeing the results.

Start Learning Now 👉 [puzzles.modular.com](https://puzzles.modular.com/)

> 📬 [Subscribe to updates](https://www.modular.com/company/talk-to-us) to get notified when new puzzles are released!

## Why Mojo🔥

[Mojo](https://docs.modular.com/mojo/manual/) represents a revolutionary approach to GPU programming, making massive parallelism accessible while maintaining systems-level performance:

- 🐍 **Python-like Syntax** with systems programming capabilities
- ⚡ **Zero-cost Abstractions** that compile to efficient machine code
- 🛡️ **Strong Type System** catching errors at compile time
- 📊 **Built-in Tensor Support** with hardware-aware optimizations
- 🔧 **Direct Hardware Access** to CPU and GPU intrinsics
- 🔄 **Cross-Hardware Portability** for CPUs and GPUs
- 🎯 **Ergonomic Improvements** over traditional C/C++

## Getting Started

### Prerequisits

You'll need a [compatible GPU](https://docs.modular.com/max/faq#gpu-requirements) to run the examples.

1. Visit [puzzles.modular.com](https://puzzles.modular.com)
2. Clone this repository

   ```bash
   git clone https://github.com/modular/mojo-gpu-puzzles
   cd mojo-gpu-puzzles
   ```

3. Install a package manager to run the Mojo🔥 programs:

   ### **Option 1 (Hightly recommended)**: [pixi](https://pixi.sh/latest/#installation)

    `pixi` is the **recommended option** for this project because:
    - Easy access to Modular's MAX/Mojo packages
    - Handles GPU dependencies
    - Full conda + PyPI ecosystem support

    **Note: A few puzzles only work with `pixi`.**

    **Install:**

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    **Update:**

    ```bash
    pixi self-update
    ```

   ### Option 2: [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

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

4. Start solving puzzles!

## Development

We use `pixi` for development as it includes `uv` and also supports conda packages (like `mdbook` from the `conda-forge` channel) needed for development workflows.

```bash
# Build and serve the book
pixi run book

# Test solutions on GPU
pixi run tests
# Or a specific puzzle
pixi run tests pXX
# Or manually
pixi run mojo/python solutions/pXX/pXX.{mojo,py}

# Run GPU sanitizers for debugging on NVIDIA GPUs using `compute-sanitizer`
pixi run memcheck  <optional pXX>    # Detect memory errors
pixi run racecheck <optional pXX>    # Detect race conditions
pixi run synccheck <optional pXX>    # Detect synchronization errors
pixi run initcheck <optional pXX>    # Detect uninitialized memory access
# Or run all sanitizer tools
pixi run sanitizers pXX
# Or manually
# Note: ignore the mojo runtime error collision with the sanitizer. Look for `Error SUMMARY`
pixi run compute-sanitizer --tool {memcheck,racecheck,synccheck,initcheck} mojo solutions/pXX/pXX.mojo

# Format code
pixi run format
```

## Contributing

We welcome contributions! Whether it's:

- 📝 Improving explanations
- 🐛 Fixing bugs ([report bug](https://github.com/modular/mojo-gpu-puzzles/issues/new?template=bug_report.yml))
- 💡 Suggesting improvements ([request feature](https://github.com/modular/mojo-gpu-puzzles/issues/new?template=feature_request.yml))

Please feel free to:

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## Community

<p align="center">
  <a href="https://docs.modular.com/max/get-started/#stay-in-touch">
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

## Acknowledgments

- Thanks to all our [contributors](https://github.com/modular/mojo-gpu-puzzles/graphs/contributors)
- Initial puzzles are heavily inspired by [GPU Puzzles](https://github.com/srush/GPU-Puzzles)
- Built with [mdBook](https://rust-lang.github.io/mdBook/)

## License

This project is licensed under the LLVM License - see the [LICENSE](LICENSE) file for details.

<p align="center">
  <sub>Built with ❤️ by the Modular team</sub>
</p>
