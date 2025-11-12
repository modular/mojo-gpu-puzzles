# 🧐 Detective Work: First Case

## Overview

This puzzle presents a crashing GPU program where your task is to identify the issue using only `(cuda-gdb)` debugging tools, without examining the source code. Apply your debugging skills to solve the mystery!

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](./essentials.md) to understand CUDA-GDB setup and basic debugging commands. Make sure you've run `pixi run setup-cuda-gdb` or similar symlink is available

```bash
ln -sf /usr/local/cuda/bin/cuda-gdb-minimal $CONDA_PREFIX/bin/cuda-gdb-minimal
ln -sf /usr/local/cuda/bin/cuda-gdb-python3.12-tui $CONDA_PREFIX/bin/cuda-gdb-python3.12-tui
```

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

To experience the bug firsthand, run the following command in your terminal (`pixi` only):

```bash
pixi run -e nvidia p09 --first-case
```

You'll see output like when the program crashes with this error:

```txt
CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)
[24326:24326:20250801,180816.333593:ERROR file_io_posix.cc:144] open /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq: No such file or directory (2)
[24326:24326:20250801,180816.333653:ERROR file_io_posix.cc:144] open /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq: No such file or directory (2)
Please submit a bug report to https://github.com/modular/modular/issues and include the crash backtrace along with all the relevant source codes.
Stack dump:
0.      Program arguments: /home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo problems/p09/p09.mojo
Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
0  mojo                      0x0000653a338d3d2b
1  mojo                      0x0000653a338d158a
2  mojo                      0x0000653a338d48d7
3  libc.so.6                 0x00007cbc08442520
4  libc.so.6                 0x00007cbc0851e88d syscall + 29
5  libAsyncRTMojoBindings.so 0x00007cbc0ab68653
6  libc.so.6                 0x00007cbc08442520
7  libc.so.6                 0x00007cbc084969fc pthread_kill + 300
8  libc.so.6                 0x00007cbc08442476 raise + 22
9  libc.so.6                 0x00007cbc084287f3 abort + 211
10 libAsyncRTMojoBindings.so 0x00007cbc097c7c7b
11 libAsyncRTMojoBindings.so 0x00007cbc097c7c9e
12 (error)                   0x00007cbb5c00600f
mojo crashed!
Please file a bug report.
```

## Your task: detective work

**Challenge**: Without looking at the code yet, what would be your debugging strategy to investigate this crash?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Read the crash message carefully** - `CUDA_ERROR_ILLEGAL_ADDRESS` means the GPU tried to access invalid memory
2. **Check the breakpoint information** - Look at the function parameters shown when CUDA-GDB stops
3. **Inspect all pointers systematically** - Use `print` to examine each pointer parameter
4. **Look for suspicious addresses** - Valid GPU addresses are typically large hex numbers (what does `0x0` mean?)
5. **Test memory access** - Try accessing the data through each pointer to see which one fails
6. **Apply the systematic approach** - Like a detective, follow the evidence from symptom to root cause
7. **Compare valid vs invalid patterns** - If one pointer works and another doesn't, focus on the broken one

</div>
</details>

<details class="solution-details">
<summary><strong>💡 Investigation & Solution</strong></summary>

<div class="solution-explanation">

## Step-by-Step Investigation with CUDA-GDB

### Launch the Debugger

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

### Examine the Breakpoint Information

When CUDA-GDB stops, it immediately shows valuable clues:

```
(cuda-gdb) run
CUDA thread hit breakpoint, p09_add_10_... (result=0x302000000, input=0x0)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:31
31          i = thread_idx.x
```

**🔍 First Clue**: The function signature shows `(result=0x302000000, input=0x0)`

- `result` has a valid GPU memory address
- `input` is `0x0` - this is a null pointer!

### Systematic variable inspection

```
(cuda-gdb) next
32          result[i] = input[i] + 10.0
(cuda-gdb) print i
$1 = 0
(cuda-gdb) print result
$2 = (!pop.scalar<f32> * @register) 0x302000000
(cuda-gdb) print input
$3 = (!pop.scalar<f32> * @register) 0x0
```

**Evidence Gathering**:

- ✅ Thread index `i=0` is valid
- ✅ Result pointer `0x302000000` is a proper GPU address
- ❌ Input pointer `0x0` is null

### Confirm the Problem

```
(cuda-gdb) print input[i]
Cannot access memory at address 0x0
```

**Smoking Gun**: Cannot access memory at null address - this confirms the crash cause!

## Root cause analysis

**The Problem**: Now if we look at the [code](../../../problems/p09/p09.mojo) for `--first-crash`, we see that the host code creates a null pointer instead of allocating proper GPU memory:

```mojo
input_ptr = {}  # Creates NULL pointer!
```

**Why This Crashes**:

1. `{}` creates an uninitialized pointer (null)
2. This null pointer gets passed to the GPU kernel
3. When kernel tries `input[i]`, it dereferences null → `CUDA_ERROR_ILLEGAL_ADDRESS`

## The fix

Replace null pointer creation with proper buffer allocation:

```mojo
# Wrong: Creates null pointer
input_ptr = {}

# Correct: Allocates and initialize actual GPU memory for safe processing
input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
input_bufenqueue_fill(0)
```

## Key debugging lessons

**Pattern Recognition**:

- `0x0` addresses are always null pointers
- Valid GPU addresses are large hex numbers (e.g., `0x302000000`)

**Debugging Strategy**:

1. **Read crash messages** - They often hint at the problem type
2. **Check function parameters** - CUDA-GDB shows them at breakpoint entry
3. **Inspect all pointers** - Compare addresses to identify null/invalid ones
4. **Test memory access** - Try dereferencing suspicious pointers
5. **Trace back to allocation** - Find where the problematic pointer was created

**💡 Key Insight**: This type of null pointer bug is extremely common in GPU programming. The systematic CUDA-GDB investigation approach you learned here applies to debugging many other GPU memory issues, race conditions, and kernel crashes.

</div>
</details>

## Next steps: from crashes to silent bugs

**You've learned crash debugging!** You can now:

- **Systematically investigate GPU crashes** using error messages as clues
- **Identify null pointer bugs** through pointer address inspection
- **Use CUDA-GDB effectively** for memory-related debugging

### Your next challenge: [Detective Work: Second Case](./second_case.md)

**But what if your program doesn't crash?** What if it runs perfectly but produces **wrong results**?

The [Second Case](./second_case.md) presents a completely different debugging challenge:

- **No crash messages** to guide you
- **No obvious pointer problems** to investigate
- **No stack traces** pointing to the issue
- **Just wrong results** that need systematic investigation

**New skills you'll develop:**

- **Logic bug detection** - Finding algorithmic errors without crashes
- **Pattern analysis** - Using incorrect output to trace back to root causes
- **Execution flow debugging** - When variable inspection fails due to optimizations

The systematic investigation approach you learned here - reading clues, forming hypotheses, testing systematically - forms the foundation for debugging the more subtle logic errors ahead.
