# Puzzle 10: Memory Error Detection & Race Conditions with Sanitizers

> ⚠️ This puzzle works on compatible **NVIDIA GPU** only. We are working to enable tooling support for other GPU vendors.

## The moment every GPU developer dreads

You've written what looks like perfect GPU code. Your algorithm is sound, your memory management seems correct, and your thread coordination appears flawless. You run your tests with confidence and...

- **✅ ALL TESTS PASS**
- **✅ Performance looks great**
- **✅ Output matches expected results**

You ship your code to production, feeling proud of your work. Then weeks later, you get the call:

- **"The application crashed in production"**
- **"Results are inconsistent between runs"**
- **"Memory corruption detected"**

Welcome to the insidious world of **silent GPU bugs** - errors that hide in the shadows of massive parallelism, waiting to strike when you least expect them. These bugs can pass all your tests, produce correct results 99% of the time, and then catastrophically fail when it matters most.

**Important note**: This puzzle requires NVIDIA GPU hardware and is only available through `pixi`, as `compute-sanitizer` is part of NVIDIA's CUDA toolkit.

## Why GPU bugs are uniquely sinister

Unlike CPU programs where bugs usually announce themselves with immediate crashes or wrong results, GPU bugs are **experts at hiding**:

**Silent corruption patterns:**

- **Memory violations that don't crash**: Out-of-bounds access to "lucky" memory locations
- **Race conditions that work "most of the time"**: Timing-dependent bugs that appear random
- **Thread coordination failures**: Deadlocks that only trigger under specific load conditions

**Massive scale amplification:**

- **One thread's bug affects thousands**: A single memory violation can corrupt entire warps
- **Race conditions multiply exponentially**: More threads = more opportunities for corruption
- **Hardware variations mask problems**: Same bug behaves differently across GPU architectures

But here's the exciting part: **once you learn GPU sanitization tools, you'll catch these elusive bugs before they ever reach production**.

## Your sanitization toolkit: NVIDIA compute-sanitizer

**NVIDIA compute-sanitizer** is your specialized weapon against GPU bugs. It can detect:

- **Memory violations**: Out-of-bounds access, invalid pointers, memory leaks
- **Race conditions**: Shared memory hazards between threads
- **Synchronization bugs**: Deadlocks, barrier misuse, improper thread coordination
- **And more**: Check `pixi run compute-sanitizer --help`

📖 **Official documentation**: [NVIDIA Compute Sanitizer User Guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

Think of it as **X-ray vision for your GPU programs** - revealing hidden problems that normal testing can't see.

## What you'll learn in this puzzle

This puzzle teaches you to systematically find and fix the most elusive GPU bugs. You'll learn the detective skills that distinguish competent GPU developers from exceptional ones.

### **Critical skills you'll develop**

1. **Silent bug detection** - Find problems that tests don't catch
2. **Memory corruption investigation** - Track down undefined behavior before it strikes
3. **Race condition detection** - Identify and eliminate concurrency hazards
4. **Tool selection expertise** - Know exactly which sanitizer to use when
5. **Production debugging confidence** - Catch bugs before they reach users

### **Real-world bug hunting scenarios**

You'll investigate the two most dangerous classes of GPU bugs:

- **Memory violations** - The silent killers that corrupt data without warning
- **Race conditions** - The chaos creators that make results unpredictable

Each scenario teaches you to think like a GPU bug detective, following clues that are invisible to normal testing.

## Your bug hunting journey

This puzzle takes you through a carefully designed progression from discovering silent corruption to learning parallel debugging:

### 👮🏼‍♂️ [The Silent Corruption Mystery](./memcheck.md)

**Memory violation investigation** - When tests pass but memory lies

- Investigate programs that pass tests while committing memory crimes
- Learn to spot the telltale signs of undefined behavior (UB)
- Learn `memcheck` - your memory violation detector
- Understand why GPU hardware masks memory errors
- Practice systematic memory access validation

**Key outcome**: Ability to detect memory violations that would otherwise go unnoticed until production

### 🏁 [The Race Condition Hunt](./racecheck.md)

**Concurrency bug investigation** - When threads turn against each other

- Investigate programs that fail randomly due to thread timing
- Learn to identify shared memory hazards before they corrupt data
- Learn `racecheck` - your race condition detector
- Compare `racecheck` vs `synccheck` for different concurrency bugs
- Practice thread synchronization strategies

**Key outcome**: Advanced concurrency debugging - the ability to tame thousands of parallel threads

## The GPU detective mindset

GPU sanitization requires you to become a **parallel program detective** investigating crimes where:

- **The evidence is hidden** - Bugs occur in parallel execution you can't directly observe
- **Multiple suspects exist** - Thousands of threads, any combination could be guilty
- **The crime is intermittent** - Race conditions and timing-dependent failures
- **The tools are specialized** - Sanitizers that see what normal debugging can't

But like any good detective, you'll learn to:

- **Follow invisible clues** - Memory access patterns, thread timing, synchronization points
- **Think in parallel** - Consider how thousands of threads interact simultaneously
- **Prevent future crimes** - Build sanitization into your development workflow
- **Trust your tools** - Let sanitizers reveal what manual testing cannot

## Prerequisites and expectations

**What you need to know**:

- GPU programming concepts from Puzzles 1-8 (memory management, thread coordination, barriers)
- **[Compatible NVIDIA GPU hardware](https://docs.modular.com/max/faq#gpu-requirements)**
- Environment setup with `pixi` package manager for accessing `compute-sanitizer`
- **Prior puzzles**: Familiarity with [Puzzle 4](../puzzle_04/introduction_layout_tensor.md) and [Puzzle 8](../puzzle_08/layout_tensor.md) are recommended

**What you'll gain**:

- **Production-ready debugging skills** used by professional GPU development teams
- **Silent bug detection skills** that prevent costly production failures
- **Parallel debugging confidence** for the most challenging concurrency scenarios
- **Tool expertise** that will serve you throughout your GPU programming career
