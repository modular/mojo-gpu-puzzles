# 🧐 Detective Work: First Case

## Overview

This puzzle presents a crashing GPU program where your task is to identify the
issue using only `(cuda-gdb)` debugging tools, without examining the source
code. Apply your debugging skills to solve the mystery!

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](./essentials.md) to
understand CUDA-GDB setup and basic debugging commands. Make sure you've run:

```bash
pixi run -e nvidia setup-cuda-gdb
```

This auto-detects your CUDA installation and sets up the necessary links for GPU
debugging.

## Key concepts

In this debugging challenge, you'll learn about:

- **Systematic debugging**: Using error messages as clues to find root causes
- **Error analysis**: Reading crash messages and stack traces
- **Hypothesis formation**: Making educated guesses about the problem
- **Debugging workflow**: Step-by-step investigation process

## Running the code

First, examine the kernel without looking at the complete code:

```mojo
{{#include ../../../problems/p09/p09.mojo:first_crash}}
```

To experience the bug firsthand, run the following command in your terminal
(`pixi` only):

```bash
pixi run -e nvidia p09 --first-case
```

You'll see output like this when the program crashes:

```txt
First Case: Try to identify what's wrong without looking at the code!

CUDA call failed: CUDA_ERROR_MISALIGNED_ADDRESS (misaligned address)
To get more accurate error information, set MODULAR_DEBUG=device-sync-mode.
 #0 0x00007cc268dc4d1e (.../.pixi/envs/nvidia/lib/libKGENCompilerRTShared.so+0x112d1e)
...
#12 0x00007cc26a4fc328 AsyncRT_DeviceContext_release (.../.pixi/envs/nvidia/lib/libAsyncRTMojoBindings.so+0x45328)
#13 0x00007cc21280df9d
.../.pixi/envs/nvidia/bin/mojo: error: execution crashed
To get a symbolicated stack trace, compile your program using `mojo build` with debug info enabled (e.g., `-debug-level=line-tables`) and execute it separately.
```

## Your task: detective work

**Challenge**: Without looking at the code yet, what would be your debugging
strategy to investigate this crash?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Read the crash message carefully** - a `CUDA_ERROR_*_ADDRESS` error means
   the GPU tried to read or write through an address it can't use
2. **Check the breakpoint information** - Look at the function parameters shown
   when CUDA-GDB stops
3. **Inspect all pointers systematically** - Use `print` to examine each pointer
   parameter
4. **Compare the addresses** - Do `output` and `a` look like they point into
   the same kind of memory?
5. **Let the kernel run** - A value the debugger prints through a pointer
   doesn't prove the GPU can read it. Continue and see where the kernel stops
6. **Apply the systematic approach** - Like a detective, follow the evidence
   from symptom to root cause
7. **Compare valid vs invalid patterns** - If one pointer works and another
   doesn't, focus on the broken one

</div>
</details>

<details class="solution-details">
<summary><strong>💡 Investigation & Solution</strong></summary>

<div class="solution-explanation">

## Step-by-step investigation with CUDA-GDB

### Launch the debugger

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

### Examine the breakpoint information

When CUDA-GDB stops, it immediately shows valuable clues:

```text
(cuda-gdb) run
CUDA thread hit application kernel entry function breakpoint, p09_add_10_...
   <<<(1,1,1),(4,1,1)>>> (output=0x302000000, a=0x7ffff68575f0)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:38
38          var i = thread_idx.x
```

**🔍 First Clue**: The function signature shows
`(output=0x302000000, a=0x7ffff68575f0)`

- `output` points into the region where this program's GPU buffers live
- `a` points somewhere else entirely. On Linux, `0x7fff...` is the top of a
  process's CPU address space, where its stack and shared libraries live, not
  where GPU allocations come from

The exact value of `a` changes from run to run. What stays the same is that it
doesn't look like `output`.

### Systematic variable inspection

```text
(cuda-gdb) next
39          output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0
(cuda-gdb) print i
$1 = {0}
(cuda-gdb) print output
$2 = (!kgen.scalar<f32> * @register) 0x302000000
(cuda-gdb) print a
$3 = (!kgen.scalar<f32> * @register) 0x7ffff68575f0
```

**Evidence Gathering**:

- ✅ Thread index `i=0` is valid
- ✅ Result pointer `0x302000000` is a GPU buffer address
- ❌ Input pointer `0x7ffff68575f0` isn't

### A trap: the debugger can read what the GPU can't

```text
(cuda-gdb) print output[0]
$4 = {0}
(cuda-gdb) print a[0]
$5 = {-1.34493767e+33}
```

The debugger happily prints a value through `a`, because it reads memory from
the host side. That value is garbage, and it doesn't mean the GPU can read the
same address. Don't treat a printed value as proof that a pointer is good.

### Confirm the problem

Let the kernel run:

```text
(cuda-gdb) continue
Continuing.

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7fffc127d5f0  p09_add_10_...  (p09.mojo:39)

Thread 1 "mojo" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
```

**Smoking Gun**: The GPU faults on line 39, the line that reads through `a`.
`output` was never the problem.

## Root cause analysis

**The Problem**: Now if we look at the [code](../../../problems/p09/p09.mojo)
for `--first-case`, we see that the host code never allocates any GPU memory
for the input:

```mojo
# A `DeviceBuffer` with 0 elements. Nothing is allocated, so there's no
# GPU address to give the kernel.
var input_buf = ctx.enqueue_create_buffer[dtype](0)
```

**Why This Crashes**:

1. `ctx.enqueue_create_buffer[dtype](0)` creates a `DeviceBuffer` with zero (0)
   elements.
2. Since there are no elements for which to allocate memory, the buffer has no
   GPU address.
3. The kernel still receives something for `a`, but it isn't a GPU address. In
   this release it's a leftover CPU-side value like `0x7ffff68575f0`.
4. When the kernel evaluates `a[unsafe_offset=i]`, the GPU faults. Which CUDA
   error you see at the top depends on the bad address, which is why it can
   differ between Mojo versions.

## The fix

Replace the empty buffer with a proper allocation:

```mojo
# Wrong: allocates nothing, so the kernel gets no usable address
var input_buf = ctx.enqueue_create_buffer[dtype](0)

# Correct: allocate and initialize actual GPU memory for safe processing
var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
input_buf.enqueue_fill(0)
```

## Key debugging lessons

**Pattern Recognition**:

- Pointers into the same kind of memory look alike. Here the GPU buffer is
  `0x302000000`, and an address that looks nothing like it deserves suspicion
- `0x0` is a null pointer, and `0x7fff...` on Linux is CPU-side memory.
  Neither belongs in a kernel argument

**Debugging Strategy**:

1. **Read crash messages** - They often hint at the problem type
2. **Check function parameters** - CUDA-GDB shows them at breakpoint entry
3. **Inspect all pointers** - Compare addresses to identify invalid ones
4. **Let the kernel fault** - The line where it stops names the bad access;
   a value the debugger prints proves nothing about the GPU
5. **Trace back to allocation** - Find where the problematic pointer was created

**💡 Key Insight**: This type of invalid pointer bug is extremely common in
GPU programming. The systematic CUDA-GDB investigation approach you learned
here applies to debugging many other GPU memory issues, race conditions, and
kernel crashes.

</div>
</details>

## Next steps: from crashes to silent bugs

**You've learned crash debugging!** You can now:

- **Systematically investigate GPU crashes** using error messages as clues
- **Identify invalid pointer bugs** through pointer address inspection
- **Use CUDA-GDB effectively** for memory-related debugging

### Your next challenge: [Detective Work: Second Case](./second_case.md)

**But what if your program doesn't crash?** What if it runs perfectly but
produces **wrong results**?

The [Second Case](./second_case.md) presents a completely different debugging
challenge:

- **No crash messages** to guide you
- **No obvious pointer problems** to investigate
- **No stack traces** pointing to the issue
- **Just wrong results** that need systematic investigation

**New skills you'll develop:**

- **Logic bug detection** - Finding algorithmic errors without crashes
- **Pattern analysis** - Using incorrect output to trace back to root causes
- **Execution flow debugging** - When variable inspection fails due to
  optimizations

The systematic investigation approach you learned here - reading clues, forming
hypotheses, testing systematically - forms the foundation for debugging the more
subtle logic errors ahead.
