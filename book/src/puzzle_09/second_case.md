# 🔍 Detective Work: Second Case

## Overview

Building on your [crash debugging skills from the First Case](./first_case.md), you'll now face a completely different challenge: a **logic bug** that produces incorrect results without crashing.

**The debugging shift:**

- **First Case**: Clear crash signals (`CUDA_ERROR_ILLEGAL_ADDRESS`) guided your investigation
- **Second Case**: No crashes, no error messages - just subtly wrong results that require detective work

This intermediate-level debugging challenge covers investigating **algorithmic errors** using `LayoutTensor` operations, where the program runs successfully but produces wrong output - a much more common (and trickier) real-world debugging scenario.

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](./essentials.md) and [Detective Work: First Case](./first_case.md) to understand CUDA-GDB workflow and systematic debugging techniques. Make sure you run the setup:

```bash
pixi run -e nvidia setup-cuda-gdb
```

## Key concepts

In this debugging challenge, you'll learn about:

- **LayoutTensor debugging**: Investigating structured data access patterns
- **Logic bug detection**: Finding algorithmic errors that don't crash
- **Loop boundary analysis**: Understanding iteration count problems
- **Result pattern analysis**: Using output data to trace back to root causes

## Running the code

First, examine the kernel without looking at the complete code:

```mojo
{{#include ../../../problems/p09/p09.mojo:second_crash}}
```

To experience the bug firsthand, run the following command in your terminal (`pixi` only):

```bash
pixi run -e nvidia p09 --second-case
```

You'll see output like this - **no crash, but wrong results**:

```txt
This program computes sliding window sums for each position...

Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At open-source/max/mojo/stdlib/stdlib/gpu/host/device_context.mojo:2082:17: CUDA call failed: CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/nvidia/bin/mojo: error: execution exited with a non-zero result: 1
```

## Your task: detective work

**Challenge**: The program runs without crashing but produces consistently wrong results. Without looking at the code, what would be your systematic approach to investigate this logic bug?

**Think about:**

- What pattern do you see in the wrong results?
- How would you investigate a loop that might not be running correctly?
- What debugging strategy works when you can't inspect variables directly?
- How can you apply the systematic investigation approach from [First Case](./first_case.md) when there are no crash signals to guide you?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

### GDB command shortcuts (faster debugging)

**Use these abbreviations** to speed up your debugging session:

| Short | Full | Usage Example |
|-------|------|---------------|
| `r` | `run` | `(cuda-gdb) r` |
| `n` | `next` | `(cuda-gdb) n` |
| `c` | `continue` | `(cuda-gdb) c` |
| `b` | `break` | `(cuda-gdb) b 39` |
| `p` | `print` | `(cuda-gdb) p thread_id` |
| `q` | `quit` | `(cuda-gdb) q` |

**All debugging commands below use these shortcuts for efficiency!**

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. **Pattern analysis first** - Look at the relationship between expected and actual results (what's the mathematical pattern in the differences?)
2. **Focus on execution flow** - Count loop iterations when variables aren't accessible
3. **Use simple breakpoints** - Complex debugging commands often fail with optimized code
4. **Mathematical reasoning** - Work out what each thread should access vs what it actually accesses
5. **Missing data investigation** - If results are consistently smaller than expected, what might be missing?
6. **Host output verification** - The final results often reveal the pattern of the bug
7. **Algorithm boundary analysis** - Check if loops are processing the right number of elements
8. **Cross-validate with working cases** - Why does thread 3 work correctly but others don't?

</div>
</details>

<details class="solution-details">
<summary><strong>💡 Investigation & Solution</strong></summary>

<div class="solution-explanation">

## Step-by-step investigation with CUDA-GDB

### Phase 1: Launch and initial analysis

#### Step 1: Start the debugger

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

#### Step 2: analyze the symptoms first

Before diving into the debugger, examine what we know:

```txt
Actual result: [0.0, 1.0, 3.0, 5.0]
Expected: [1.0, 3.0, 6.0, 5.0]
```

**🔍 Pattern Recognition**:

- Thread 0: Got 0.0, Expected 1.0 → Missing 1.0
- Thread 1: Got 1.0, Expected 3.0 → Missing 2.0
- Thread 2: Got 3.0, Expected 6.0 → Missing 3.0
- Thread 3: Got 5.0, Expected 5.0 → ✅ Correct

**Initial Hypothesis**: Each thread is missing some data, but thread 3 works correctly.

### Phase 2: Entering the kernel

#### Step 3: Observe the breakpoint entry

Based on the real debugging session, here's what happens:

```bash
(cuda-gdb) r
Starting program: .../mojo run problems/p09/p09.mojo --second-case

This program computes sliding window sums for each position...
Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]

[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit application kernel entry function breakpoint, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:30
30          input: LayoutTensor[mut=False, dtype, vector_layout],
```

#### Step 4: Navigate to the main logic

```bash
(cuda-gdb) n
29          output: LayoutTensor[mut=True, dtype, vector_layout],
(cuda-gdb) n
32          thread_id = thread_idx.x
(cuda-gdb) n
38          for offset in range(ITER):
```

#### Step 5: Test variable accessibility - crucial discovery

```bash
(cuda-gdb) p thread_id
$1 = 0
```

**✅ Good**: Thread ID is accessible.

```bash
(cuda-gdb) p window_sum
Cannot access memory at address 0x0
```

**❌ Problem**: `window_sum` is not accessible.

```bash
(cuda-gdb) p a[0]
Attempt to take address of value not located in memory.
```

**❌ Problem**: Direct LayoutTensor indexing doesn't work.

```bash
(cuda-gdb) p a.ptr[0]
$2 = {0}
(cuda-gdb) p a.ptr[0]@4
$3 = {{0}, {1}, {2}, {3}}
```

**🎯 BREAKTHROUGH**: `a.ptr[0]@4` shows the full input array! This is how we can inspect LayoutTensor data.

### Phase 3: The critical loop investigation

#### Step 6: Set up loop monitoring

```bash
(cuda-gdb) b 42
Breakpoint 1 at 0x7fffd326ffd0: file problems/p09/p09.mojo, line 42.
(cuda-gdb) c
Continuing.

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:42
42              idx = thread_id + offset - 1
```

**🔍 We're now inside the loop body. Let's count iterations manually.**

#### Step 7: First loop iteration (offset = 0)

```bash
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**First iteration complete**: Loop went from line 42 → 43 → back to 41. The loop continues.

#### Step 8: Second loop iteration (offset = 1)

```bash
(cuda-gdb) n

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
42              idx = thread_id + offset - 1
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
44                  value = rebind[Scalar[dtype]](input[idx])
(cuda-gdb) n
45                  window_sum += value
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**Second iteration complete**: This time it went through the if-block (lines 44-45).

#### Step 9: testing for third iteration

```bash
(cuda-gdb) n
47          output[thread_id] = window_sum
```

**CRITICAL DISCOVERY**: The loop exited after only 2 iterations! It went directly to line 47 instead of hitting our breakpoint at line 42 again.

**Conclusion**: The loop ran exactly **2 iterations** and then exited.

#### Step 10: Complete kernel execution and context loss

```bash
(cuda-gdb) n
31      fn process_sliding_window(
(cuda-gdb) n
[Switching to Thread 0x7ffff7cc0e00 (LWP 110927)]
0x00007ffff064f84a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) p output.ptr[0]@4
No symbol "output" in current context.
(cuda-gdb) p offset
No symbol "offset" in current context.
```

**🔍 Context Lost**: After kernel completion, we lose access to kernel variables. This is normal behavior.

### Phase 4: Root cause analysis

#### Step 11: Algorithm analysis from observed execution

From our debugging session, we observed:

1. **Loop Iterations**: Only 2 iterations (offset = 0, offset = 1)
2. **Expected**: A sliding window of size 3 should require 3 iterations (offset = 0, 1, 2)
3. **Missing**: The third iteration (offset = 2)

Looking at what each thread should compute:

- **Thread 0**: window_sum = input[-1] + input[0] + input[1] = (boundary) + 0 + 1 = 1.0
- **Thread 1**: window_sum = input[0] + input[1] + input[2] = 0 + 1 + 2 = 3.0
- **Thread 2**: window_sum = input[1] + input[2] + input[3] = 1 + 2 + 3 = 6.0
- **Thread 3**: window_sum = input[2] + input[3] + input[4] = 2 + 3 + (boundary) = 5.0

#### Step 12: Trace the actual execution for thread 0

With only 2 iterations (offset = 0, 1):

**Iteration 1 (offset = 0)**:

- `idx = thread_id + offset - 1 = 0 + 0 - 1 = -1`
- `if 0 <= idx < SIZE:` → `if 0 <= -1 < 4:` → **False**
- Skip the sum operation

**Iteration 2 (offset = 1)**:

- `idx = thread_id + offset - 1 = 0 + 1 - 1 = 0`
- `if 0 <= idx < SIZE:` → `if 0 <= 0 < 4:` → **True**
- `window_sum += input[0]` → `window_sum += 0`

**Missing Iteration 3 (offset = 2)**:

- `idx = thread_id + offset - 1 = 0 + 2 - 1 = 1`
- `if 0 <= idx < SIZE:` → `if 0 <= 1 < 4:` → **True**
- `window_sum += input[1]` → `window_sum += 1` ← **THIS NEVER HAPPENS**

**Result**: Thread 0 gets `window_sum = 0` instead of `window_sum = 0 + 1 = 1`

### Phase 5: Bug confirmation

Looking at the problem code, we find:

```mojo
comptime ITER = 2                       # ← BUG: Should be 3!

for offset in range(ITER):           # ← Only 2 iterations: [0, 1]
    idx = Int(thread_id) + offset - 1     # ← Missing offset = 2
    if 0 <= idx < SIZE:
        value = rebind[Scalar[dtype]](a[idx])
        window_sum += value
```

**🎯 ROOT CAUSE IDENTIFIED**: `ITER = 2` should be `ITER = 3` for a sliding window of size 3.

**The Fix**: Change `comptime ITER = 2` to `comptime ITER = 3` in the source code.

## Key debugging lessons

**When Variables Are Inaccessible**:

1. **Focus on execution flow** - Count breakpoint hits and loop iterations
2. **Use mathematical reasoning** - Work out what should happen vs what does happen
3. **Pattern analysis** - Let the wrong results guide your investigation
4. **Cross-validation** - Test your hypothesis against multiple data points

**Professional GPU Debugging Reality**:

- **Variable inspection often fails** due to compiler optimizations
- **Execution flow analysis** is more reliable than data inspection
- **Host output patterns** provide crucial debugging clues
- **Source code reasoning** complements limited debugger capabilities

**LayoutTensor Debugging**:

- Even with LayoutTensor abstractions, underlying algorithmic bugs still manifest
- Focus on the algorithm logic rather than trying to inspect tensor contents
- Use systematic reasoning to trace what each thread should vs actually accesses

**Key Insight**: This type of off-by-one loop bug is extremely common in GPU programming. The systematic approach you learned here - combining limited debugger info with mathematical analysis and pattern recognition - is exactly how professional GPU developers debug when tools have limitations.

</div>
</details>

## Next steps: from logic bugs to coordination deadlocks

**You've learned logic bug debugging!** You can now:

- ✅ **Investigate algorithmic errors** without crashes or obvious symptoms
- ✅ **Use pattern analysis** to trace wrong results back to root causes
- ✅ **Debug with limited variable access** using execution flow analysis
- ✅ **Apply mathematical reasoning** when debugger tools have limitations

### Your final challenge: [Detective Work: Third Case](./third_case.md)

**But what if your program doesn't crash AND doesn't finish?** What if it just **hangs forever**?

The [Third Case](./third_case.md) presents the ultimate debugging challenge:

- ❌ **No crash messages** (like First Case)
- ❌ **No wrong results** (like Second Case)
- ❌ **No completion at all** - just infinite hanging
- ✅ **Silent deadlock** requiring advanced thread coordination analysis

**New skills you'll develop:**

- **Barrier deadlock detection** - Finding coordination failures in parallel threads
- **Multi-thread state analysis** - Examining all threads simultaneously
- **Synchronization debugging** - Understanding thread cooperation breakdowns

**The debugging evolution:**

1. **First Case**: Follow crash signals → Find memory bugs
2. **Second Case**: Analyze result patterns → Find logic bugs
3. **Third Case**: Investigate thread states → Find coordination bugs

The systematic investigation skills from both previous cases - hypothesis formation, evidence gathering, pattern analysis - become crucial when debugging the most challenging GPU issue: threads that coordinate incorrectly and wait forever.
