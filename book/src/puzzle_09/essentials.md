# 📚 Mojo GPU Debugging Essentials

Welcome to the world of GPU debugging! After learning GPU programming concepts through puzzles 1-8, you're now ready to learn the most critical skill for any GPU programmer: **how to debug when things go wrong**.

GPU debugging can seem intimidating at first - you're dealing with thousands of threads running in parallel, different memory spaces, and hardware-specific behaviors. But with the right tools and workflow, debugging GPU code becomes systematic and manageable.

In this guide, you'll learn to debug both the **CPU host code** (where you set up your GPU operations) and the **GPU kernel code** (where the parallel computation happens). We'll use real examples, actual debugger output, and step-by-step workflows that you can immediately apply to your own projects.

**Note**: The following content focuses on command-line debugging for universal IDE compatibility. If you prefer VS Code debugging, refer to the [Mojo debugging documentation](https://docs.modular.com/mojo/tools/debugging) for VS Code-specific setup and workflows.

## Why GPU debugging is different

Before diving into tools, consider what makes GPU debugging unique:

- **Traditional CPU debugging**: One thread, sequential execution, straightforward memory model
- **GPU debugging**: Thousands of threads, parallel execution, multiple memory spaces, race conditions

This means you need specialized tools that can:

- Switch between different GPU threads
- Inspect thread-specific variables and memory
- Handle the complexity of parallel execution
- Debug both CPU setup code and GPU kernel code

## Your debugging toolkit

Mojo's GPU debugging capabilities currently is limited to NVIDIA GPUs. The [Mojo debugging documentation](https://docs.modular.com/mojo/tools/debugging) explains that the Mojo package includes:

- **LLDB debugger** with Mojo plugin for CPU-side debugging
- **CUDA-GDB integration** for GPU kernel debugging
- **Command-line interface** via `mojo debug` for universal IDE compatibility

For GPU-specific debugging, the [Mojo GPU debugging guide](https://docs.modular.com/mojo/tools/gpu-debugging) provides additional technical details.

This architecture provides the best of both worlds: familiar debugging commands with GPU-specific capabilities.

## The debugging workflow: From problem to solution

When your GPU program crashes, produces wrong results, or behaves unexpectedly, follow this systematic approach:

1. **Prepare your code for debugging** (disable optimizations, add debug symbols)
2. **Choose the right debugger** (CPU host code vs GPU kernel debugging)
3. **Set strategic breakpoints** (where you suspect the problem lies)
4. **Execute and inspect** (step through code, examine variables)
5. **Analyze patterns** (memory access, thread behavior, race conditions)

This workflow works whether you're debugging a simple array operation from Puzzle 01 or complex shared memory code from Puzzle 08.

## Step 1: Preparing your code for debugging

**🥇 The golden rule**: Never debug _optimized_ code. Optimizations can reorder instructions, eliminate variables, and inline functions, making debugging nearly impossible.

### Building with debug information

When building Mojo programs for debugging, always include debug symbols:

```bash
# Build with full debug information
mojo build -O0 -g your_program.mojo -o your_program_debug
```

**What these flags do:**

- `-O0`: Disables all optimizations, preserving your original code structure
- `-g`: Includes debug symbols so the debugger can map machine code back to your Mojo source
- `-o`: Creates a named output file for easier identification

### Why this matters

Without debug symbols, your debugging session looks like this:

```
(lldb) print my_variable
error: use of undeclared identifier 'my_variable'
```

With debug symbols, you get:

```
(lldb) print my_variable
(int) $0 = 42
```

## Step 2: Choosing your debugging approach

Here's where GPU debugging gets interesting. You have **four different combinations** to choose from, and picking the right one saves you time:

### The four debugging combinations

**Quick reference:**

```bash
# 1. JIT + LLDB: Debug CPU host code directly from source
pixi run mojo debug your_gpu_program.mojo

# 2. JIT + CUDA-GDB: Debug GPU kernels directly from source
pixi run mojo debug --cuda-gdb --break-on-launch your_gpu_program.mojo

# 3. Binary + LLDB: Debug CPU host code from pre-compiled binary
pixi run mojo build -O0 -g your_gpu_program.mojo -o your_program_debug
pixi run mojo debug your_program_debug

# 4. Binary + CUDA-GDB: Debug GPU kernels from pre-compiled binary
pixi run mojo debug --cuda-gdb --break-on-launch your_program_debug
```

### When to use each approach

**For learning and quick experiments:**

- Use **JIT debugging** - no build step required, faster iteration

**For serious debugging sessions:**

- Use **binary debugging** - more predictable, cleaner debugger output

**For CPU-side issues** (buffer allocation, host memory, program logic):

- Use **LLDB mode** - perfect for debugging your `main()` function and setup code

**For GPU kernel issues** (thread behavior, GPU memory, kernel crashes):

- Use **CUDA-GDB mode** - the only way to inspect individual GPU threads

The beauty is that you can mix and match. Start with JIT + LLDB to debug your setup code, then switch to JIT + CUDA-GDB to debug the actual kernel.

---

## Understanding GPU kernel debugging with CUDA-GDB

Next comes GPU kernel debugging - the most powerful (and complex) part of your debugging toolkit.

When you use `--cuda-gdb`, Mojo integrates with NVIDIA's [CUDA-GDB debugger](https://docs.nvidia.com/cuda/cuda-gdb/index.html). This isn't just another debugger - it's specifically designed for the parallel, multi-threaded world of GPU computing.

### What makes CUDA-GDB special

**Regular GDB** debugs one thread at a time, stepping through sequential code.
**CUDA-GDB** debugs thousands of GPU threads simultaneously, each potentially executing different instructions.

This means you can:

- **Set breakpoints inside GPU kernels** - pause execution when any thread hits your breakpoint
- **Switch between GPU threads** - examine what different threads are doing at the same moment
- **Inspect thread-specific data** - see how the same variable has different values across threads
- **Debug memory access patterns** - catch out-of-bounds access, race conditions, and memory corruption (more on detecting such issues in the Puzzle 10)
- **Analyze parallel execution** - understand how your threads interact and synchronize

### Connecting to concepts from previous puzzles

Remember the GPU programming concepts you learned in puzzles 1-8? CUDA-GDB lets you inspect all of them at runtime:

#### Thread hierarchy debugging

Back in puzzles 1-8, you wrote code like this:

```mojo
# From puzzle 1: Basic thread indexing
i = thread_idx.x  # Each thread gets a unique index

# From puzzle 7: 2D thread indexing
row = thread_idx.y  # 2D grid of threads
col = thread_idx.x
```

With CUDA-GDB, you can **actually see these thread coordinates in action**:

```gdb
(cuda-gdb) info cuda threads
```

outputs

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffcf26fed0 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    13
```

and jump to a specific thread to see what it's doing

```gdb
(cuda-gdb) cuda thread (1,0,0)
```

shows

```
[Switching to CUDA thread (1,0,0)]
```

This is incredibly powerful - you can literally **watch your parallel algorithm execute across different threads**.

#### Memory space debugging

Remember puzzle 8 where you learned about different types of GPU memory? CUDA-GDB lets you inspect all of them:

```gdb
# Examine global memory (the arrays from puzzles 1-5)
(cuda-gdb) print input_array[0]@4
$1 = {{1}, {2}, {3}, {4}}   # Mojo scalar format

# Examine shared memory using local variables (thread_idx.x doesn't work)
(cuda-gdb) print shared_data[i]   # Use local variable 'i' instead
$2 = {42}
```

The debugger shows you exactly what each thread sees in memory - perfect for catching race conditions or memory access bugs.

#### Strategic breakpoint placement

CUDA-GDB breakpoints are much more powerful than regular breakpoints because they work with parallel execution:

```gdb
# Break when ANY thread enters your kernel
(cuda-gdb) break add_kernel

# Break only for specific threads (great for isolating issues)
(cuda-gdb) break add_kernel if thread_idx.x == 0

# Break on memory access violations
(cuda-gdb) watch input_array[thread_idx.x]

# Break on specific data conditions
(cuda-gdb) break add_kernel if input_array[thread_idx.x] > 100.0
```

This lets you focus on exactly the threads and conditions you care about, instead of drowning in output from thousands of threads.

---

## Getting your environment ready

Before you can start debugging, ensure your development environment is properly configured. If you've been working through the earlier puzzles, most of this is already set up!

**Note**: Without `pixi`, you would need to manually install CUDA Toolkit from [NVIDIA's official resources](https://developer.nvidia.com/cuda-toolkit), manage driver compatibility, configure environment variables, and handle version conflicts between components. `pixi` eliminates this complexity by automatically managing all CUDA dependencies, versions, and environment configuration for you.

### Why `pixi` matters for debugging

**The challenge**: GPU debugging requires precise coordination between CUDA toolkit, GPU drivers, Mojo compiler, and debugger components. Version mismatches can lead to frustrating "debugger not found" errors.

**The solution**: Using `pixi` ensures all these components work together harmoniously. When you run `pixi run mojo debug --cuda-gdb`, pixi automatically:

- Sets up CUDA toolkit paths
- Loads the correct GPU drivers
- Configures Mojo debugging plugins
- Manages environment variables consistently

### Verifying your setup

Let's check that everything is working:

```bash
# 1. Verify GPU hardware is accessible
pixi run nvidia-smi
# Should show your GPU(s) and driver version

# 2. Set up CUDA-GDB integration (required for GPU debugging)
pixi run setup-cuda-gdb
# Links system CUDA-GDB binaries to conda environment

# 3. Verify Mojo debugger is available
pixi run mojo debug --help
# Should show debugging options including --cuda-gdb

# 4. Test CUDA-GDB integration
pixi run cuda-gdb --version
# Should show NVIDIA CUDA-GDB version information
```

If any of these commands fail, double-check your `pixi.toml` configuration and ensure the CUDA toolkit feature is enabled.

**🚨Important**: The `pixi run setup-cuda-gdb` command is required because conda's `cuda-gdb` package only provides a wrapper script. This command links the actual CUDA-GDB binaries from your system CUDA installation (`/usr/local/cuda/`) to the conda environment, enabling full GPU debugging capabilities.

**What this command does under the hood:**

```bash
# Creates symlinks to system CUDA-GDB binaries
ln -sf /usr/local/cuda/bin/cuda-gdb-minimal $CONDA_PREFIX/bin/cuda-gdb-minimal
ln -sf /usr/local/cuda/bin/cuda-gdb-python3.12-tui $CONDA_PREFIX/bin/cuda-gdb-python3.12-tui
```

---

## Hands-on tutorial: Your first GPU debugging session

Theory is great, but nothing beats hands-on experience. Let's debug a real program using Puzzle 01 - the simple "add 10 to each array element" kernel you know well.

**Why Puzzle 01?** It's the perfect debugging tutorial because:

- **Simple enough** to understand what _should_ happen
- **Real GPU code** with actual kernel execution
- **Contains both** CPU setup code and GPU kernel code
- **Short execution time** so you can iterate quickly

By the end of this tutorial, you'll have debugged the same program using all four debugging approaches, seen real debugger output, and learned the essential debugging commands you'll use daily.

### Learning path through the debugging approaches

We'll explore the [four debugging combinations](#the-four-debugging-combinations) using Puzzle 01 as our example. **Learning path**: We'll start with JIT + LLDB (easiest), then progress to CUDA-GDB (most powerful).

**⚠️ Important for GPU debugging**:

- The `--break-on-launch` flag is **required** for CUDA-GDB approaches
- **Pre-compiled binaries** (Approaches 3 & 4) preserve local variables like `i` for debugging
- **JIT compilation** (Approaches 1 & 2) optimizes away most local variables
- For serious GPU debugging, use **Approach 4** (Binary + CUDA-GDB)

## Tutorial step 1: CPU debugging with LLDB

Let's begin with the most common debugging scenario: **your program crashes or behaves unexpectedly, and you need to see what's happening in your `main()` function**.

**The mission**: Debug the CPU-side setup code in Puzzle 01 to understand how Mojo initializes GPU memory and launches kernels.

### Launch the debugger

Fire up the LLDB debugger with JIT compilation:

```bash
# This compiles and debugs p01.mojo in one step
pixi run mojo debug solutions/p01/p01.mojo
```

You'll see the LLDB prompt: `(lldb)`. You're now inside the debugger, ready to inspect your program's execution!

### Your first debugging commands

Let's trace through what happens when Puzzle 01 runs. **Type these commands exactly as shown** and observe the output:

**Step 1: Set a breakpoint at the main function**

```bash
(lldb) br set -n main
```

Output:

```
Breakpoint 1: where = mojo`main, address = 0x00000000027d7530
```

The debugger found your main function and will pause execution there.

**Step 2: Start your program**

```bash
(lldb) run
```

Output:

```
Process 186951 launched: '/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo' (x86_64)
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.1
    frame #0: 0x0000555557d2b530 mojo`main
mojo`main:
->  0x555557d2b530 <+0>: pushq  %rbp
    0x555557d2b531 <+1>: movq   %rsp, %rbp
    ...
```

The program has stopped at your breakpoint. You're currently viewing **assembly code**, which is normal - the debugger starts at the low-level machine code before reaching your high-level Mojo source.

**Step 3: Navigate through the startup process**

```bash
# Try stepping through one instruction
(lldb) next
```

Output:

```
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = instruction step over
    frame #0: 0x0000555557d2b531 mojo`main + 1
mojo`main:
->  0x555557d2b531 <+1>: movq   %rsp, %rbp
    0x555557d2b534 <+4>: pushq  %r15
    ...
```

Stepping through assembly can be tedious. Let's proceed to the more relevant parts.

**Step 4: Continue to reach your Mojo source code**

```bash
# Skip through the startup assembly to get to your actual code
(lldb) continue
```

Output:

```
Process 186951 resuming
Process 186951 stopped and restarted: thread 1 received signal: SIGCHLD
2 locations added to breakpoint 1
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.3
    frame #0: 0x00007fff5c01e841 JIT(0x7fff5c075000)`stdlib::builtin::_startup::__mojo_main_prototype(argc=([0] = 1), argv=0x00007fffffffa858) at _startup.mojo:95:4
```

Mojo's runtime is initializing. The `_startup.mojo` indicates Mojo's internal startup code. The `SIGCHLD` signal is normal - it's how Mojo manages its internal processes.

**Step 5: Continue to your actual code**

```bash
# One more continue to reach your p01.mojo code!
(lldb) continue
```

Output:

```
Process 186951 resuming
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.2
    frame #0: 0x00007fff5c014040 JIT(0x7fff5c075000)`p01::main(__error__=<unavailable>) at p01.mojo:24:23
   21
   22
   23   def main():
-> 24       with DeviceContext() as ctx:
   25           out = ctx.enqueue_create_buffer[dtype](SIZE)
   26           out.enqueue_fill(0)
   27           a = ctx.enqueue_create_buffer[dtype](SIZE)
```

You can now view your actual Mojo source code. Notice:

- **Line numbers 21-27** from your p01.mojo file
- **Current line 24**: `with DeviceContext() as ctx:`
- **JIT compilation**: The `JIT(0x7fff5c075000)` indicates Mojo compiled your code just-in-time

**Step 6: Let the program complete**

```bash
# Let the program run to completion
(lldb) continue
```

Output:

```
Process 186951 resuming
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Process 186951 exited with status = 0 (0x00000000)
```

### What you just learned

🎓 **Congratulations!** You've just completed your first GPU program debugging session. Here's what happened:

**The debugging journey you took:**

1. **Started with assembly** - Normal for low-level debugging, shows how the debugger works at machine level
2. **Navigated through Mojo startup** - Learned that Mojo has internal initialization code
3. **Reached your source code** - Saw your actual p01.mojo lines 21-27 with syntax highlighting
4. **Watched JIT compilation** - Observed Mojo compiling your code on-the-fly
5. **Verified successful execution** - Confirmed your program produces the expected output

**LLDB debugging provides:**

- ✅ **CPU-side visibility**: See your `main()` function, buffer allocation, memory setup
- ✅ **Source code inspection**: View your actual Mojo code with line numbers
- ✅ **Variable examination**: Check values of host-side variables (CPU memory)
- ✅ **Program flow control**: Step through your setup logic line by line
- ✅ **Error investigation**: Debug crashes in device setup, memory allocation, etc.

**What LLDB cannot do:**

- ❌ **GPU kernel inspection**: Cannot step into `add_10` function execution
- ❌ **Thread-level debugging**: Cannot see individual GPU thread behavior
- ❌ **GPU memory access**: Cannot examine data as GPU threads see it
- ❌ **Parallel execution analysis**: Cannot debug race conditions or synchronization

**When to use LLDB debugging:**

- Your program crashes before the GPU code runs
- Buffer allocation or memory setup issues
- Understanding program initialization and flow
- Learning how Mojo applications start up
- Quick prototyping and experimenting with code changes

**Key insight**: LLDB is perfect for **host-side debugging** - everything that happens on your CPU before and after GPU execution. For the actual GPU kernel debugging, you need our next approach...

## Tutorial step 2: Binary debugging

You've learned JIT debugging - now let's explore the **professional approach** used in production environments.

**The scenario**: You're debugging a complex application with multiple files, or you need to debug the same program repeatedly. Building a binary first provides more control and faster debugging iterations.

### Build your debug binary

**Step 1: Compile with debug information**

```bash
# Create a debug build (notice the clear naming)
pixi run mojo build -O0 -g solutions/p01/p01.mojo -o solutions/p01/p01_debug
```

**What happens here:**

- 🔧 **`-O0`**: Disables optimizations (critical for accurate debugging)
- 🔍 **`-g`**: Includes debug symbols mapping machine code to source code
- 📁 **`-o p01_debug`**: Creates a clearly named debug binary

**Step 2: Debug the binary**

```bash
# Debug the pre-built binary
pixi run mojo debug solutions/p01/p01_debug
```

### What's different (and better)

**Startup comparison:**

| JIT Debugging | Binary Debugging |
|---------------|------------------|
| Compile + debug in one step | Build once, debug many times |
| Slower startup (compilation overhead) | Faster startup |
| Compilation messages mixed with debug output | Clean debugger output |
| Debug symbols generated during debugging | Fixed debug symbols |

**When you run the same LLDB commands** (`br set -n main`, `run`, `continue`), you'll notice:

- **Faster startup** - no compilation delay
- **Cleaner output** - no JIT compilation messages
- **More predictable** - debug symbols don't change between runs
- **Professional workflow** - this is how production debugging works

---

## Tutorial step 3: Debugging the GPU kernel

So far, you've debugged the **CPU host code** - the setup, memory allocation, and initialization. But what about the actual **GPU kernel** where the parallel computation happens?

**The challenge**: Your `add_10` kernel runs on the GPU with potentially thousands of threads executing simultaneously. LLDB can't reach into the GPU's parallel execution environment.

**The solution**: CUDA-GDB - a specialized debugger that understands GPU threads, GPU memory, and parallel execution.

### Why you need CUDA-GDB

Let's understand what makes GPU debugging fundamentally different:

**CPU debugging (LLDB):**

- One thread executing sequentially
- Single call stack to follow
- Straightforward memory model
- Variables have single values

**GPU debugging (CUDA-GDB):**

- Thousands of threads executing in parallel
- Multiple call stacks (one per thread)
- Complex memory hierarchy (global, shared, local, registers)
- Same variable has different values across threads

**Real example**: In your `add_10` kernel, the variable `thread_idx.x` has a **different value in every thread** - thread 0 sees `0`, thread 1 sees `1`, etc. Only CUDA-GDB can show you this parallel reality.

### Launch CUDA-GDB debugger

**Step 1: Start GPU kernel debugging**

Choose your approach:

```bash
# Make sure you've run this already (once is enough)
pixi run setup-cuda-gdb

# We'll use JIT + CUDA-GDB (Approach 2 from above)
pixi run mojo debug --cuda-gdb --break-on-launch solutions/p01/p01.mojo
```

We'll use the **JIT + CUDA-GDB approach** since it's perfect for learning and quick iterations.

**Step 2: Launch and automatically stop at GPU kernel entry**

The CUDA-GDB prompt looks like: `(cuda-gdb)`. Start the program:

```gdb
# Run the program - it automatically stops when the GPU kernel launches
(cuda-gdb) run
```

Output:

```
Starting program: /home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo...
[Thread debugging using libthread_db enabled]
...
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]

CUDA thread hit application kernel entry function breakpoint, p01_add_10_UnsafePointer...
   <<<(1,1,1),(4,1,1)>>> (output=0x302000000, a=0x302000200) at p01.mojo:16
16          i = thread_idx.x
```

**Success! You're automatically stopped inside the GPU kernel!** The `--break-on-launch` flag caught the kernel launch and you're now at line 16 where `i = thread_idx.x` executes.

**Important**: You **don't** need to manually set breakpoints like `break add_10` - the kernel entry breakpoint is automatic. GPU kernel functions have mangled names in CUDA-GDB (like `p01_add_10_UnsafePointer...`), but you're already inside the kernel and can start debugging immediately.

**Step 3: Explore the parallel execution**

```gdb
# See all the GPU threads that are paused at your breakpoint
(cuda-gdb) info cuda threads
```

Output:

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffd326fb70 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    16
```

Perfect! This shows you **all 4 parallel GPU threads** from Puzzle 01:

- **`*` marks your current thread**: `(0,0,0)` - the thread you're debugging
- **Thread range**: From `(0,0,0)` to `(3,0,0)` - all 4 threads in the block
- **Count**: `4` - matches `THREADS_PER_BLOCK = 4` from the code
- **Same location**: All threads are paused at line 16 in `p01.mojo`

**Step 4: Step through the kernel and examine variables**

```gdb
# Use 'next' to step through code (not 'step' which goes into internals)
(cuda-gdb) next
```

Output:

```
p01_add_10_UnsafePointer... at p01.mojo:17
17          output[i] = a[i] + 10.0
```

```gdb
# Local variables work with pre-compiled binaries!
(cuda-gdb) print i
```

Output:

```
$1 = 0                    # This thread's index (captures thread_idx.x value)
```

```gdb
# GPU built-ins don't work, but you don't need them
(cuda-gdb) print thread_idx.x
```

Output:

```
No symbol "thread_idx" in current context.
```

```gdb
# Access thread-specific data using local variables
(cuda-gdb) print a[i]     # This thread's input: a[0]
```

Output:

```
$2 = {0}                  # Input value (Mojo scalar format)
```

```gdb
(cuda-gdb) print output[i] # This thread's output BEFORE computation
```

Output:

```
$3 = {0}                  # Still zero - computation hasn't executed yet!
```

```gdb
# Execute the computation line
(cuda-gdb) next
```

Output:

```
13      fn add_10(         # Steps to function signature line after computation
```

```gdb
# Now check the result
(cuda-gdb) print output[i]
```

Output:

```
$4 = {10}                 # Now shows the computed result: 0 + 10 = 10
```

```gdb
# Function parameters are still available
(cuda-gdb) print a
```

Output:

```
$5 = (!pop.scalar<f32> * @register) 0x302000200
```

**Step 5: Navigate between parallel threads**

```gdb
# Switch to a different thread to see its execution
(cuda-gdb) cuda thread (1,0,0)
```

Output:

```
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
13      fn add_10(         # Thread 1 is also at function signature
```

```gdb
# Check the thread's local variable
(cuda-gdb) print i
```

Output:

```
$5 = 1                    # Thread 1's index (different from Thread 0!)
```

```gdb
# Examine what this thread processes
(cuda-gdb) print a[i]     # This thread's input: a[1]
```

Output:

```
$6 = {1}                  # Input value for thread 1
```

```gdb
# Thread 1's computation is already done (parallel execution!)
(cuda-gdb) print output[i] # This thread's output: output[1]
```

Output:

```
$7 = {11}                 # 1 + 10 = 11 (already computed)
```

```gdb
# BEST TECHNIQUE: View all thread results at once
(cuda-gdb) print output[0]@4
```

Output:

```
$8 = {{10}, {11}, {12}, {13}}     # All 4 threads' results in one command!
```

```gdb
(cuda-gdb) print a[0]@4
```

Output:

```
$9 = {{0}, {1}, {2}, {3}}         # All input values for comparison
```

```gdb
# Don't step too far or you'll lose CUDA context
(cuda-gdb) next
```

Output:

```
[Switching to Thread 0x7ffff7e25840 (LWP 306942)]  # Back to host thread
0x00007fffeca3f831 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
```

```gdb
(cuda-gdb) print output[i]
```

Output:

```
No symbol "output" in current context.  # Lost GPU context!
```

**Key insights from this debugging session:**

- 🤯 **Parallel execution is real** - when you switch to thread (1,0,0), its computation is already done!
- **Each thread has different data** - `i=0` vs `i=1`, `a[i]={0}` vs `a[i]={1}`, `output[i]={10}` vs `output[i]={11}`
- **Array inspection is powerful** - `print output[0]@4` shows all threads' results: `{{10}, {11}, {12}, {13}}`
- **GPU context is fragile** - stepping too far switches back to host thread and loses GPU variables

This demonstrates the fundamental nature of parallel computing: **same code, different data per thread, executing simultaneously.**

### What you've learned with CUDA-GDB

You've completed GPU kernel execution debugging with **pre-compiled binaries**. Here's what actually works:

**GPU debugging capabilities you gained:**

- ✅ **Debug GPU kernels automatically** - `--break-on-launch` stops at kernel entry
- ✅ **Navigate between GPU threads** - switch contexts with `cuda thread`
- ✅ **Access local variables** - `print i` works with `-O0 -g` compiled binaries
- ✅ **Inspect thread-specific data** - each thread shows different `i`, `a[i]`, `output[i]` values
- ✅ **View all thread results** - `print output[0]@4` shows `{{10}, {11}, {12}, {13}}` in one command
- ✅ **Step through GPU code** - `next` executes computation and shows results
- ✅ **See parallel execution** - threads execute simultaneously (other threads already computed when you switch)
- ✅ **Access function parameters** - examine `output` and `a` pointers
- ❌ **GPU built-ins unavailable** - `thread_idx.x`, `blockIdx.x` etc. don't work (but local variables do!)
- 📊 **Mojo scalar format** - values display as `{10}` instead of `10.0`
- ⚠️ **Fragile GPU context** - stepping too far loses access to GPU variables

**Key insights**:

- **Pre-compiled binaries** (`mojo build -O0 -g`) are essential - local variables preserved
- **Array inspection with `@N`** - most efficient way to see all parallel results at once
- **GPU built-ins are missing** - but local variables like `i` capture what you need
- **Mojo uses `{value}` format** - scalars display as `{10}` instead of `10.0`
- **Be careful with stepping** - easy to lose GPU context and return to host thread

**Real-world debugging techniques**

Now let's explore practical debugging scenarios you'll encounter in real GPU programming:

#### Technique 1: Verifying thread boundaries

```gdb
# Check if all 4 threads computed correctly
(cuda-gdb) print output[0]@4
```

Output:

```
$8 = {{10}, {11}, {12}, {13}}    # All 4 threads computed correctly
```

```gdb
# Check beyond valid range to detect out-of-bounds issues
(cuda-gdb) print output[0]@5
```

Output:

```
$9 = {{10}, {11}, {12}, {13}, {0}}  # Element 4 is uninitialized (good!)
```

```gdb
# Compare with input to verify computation
(cuda-gdb) print a[0]@4
```

Output:

```
$10 = {{0}, {1}, {2}, {3}}       # Input values: 0+10=10, 1+10=11, etc.
```

**Why this matters**: Out-of-bounds access is the #1 cause of GPU crashes. These debugging steps catch it early.

#### Technique 2: Understanding thread organization

```gdb
# See how your threads are organized into blocks
(cuda-gdb) info cuda blocks
```

Output:

```
  BlockIdx To BlockIdx Count   State
Kernel 0
*  (0,0,0)     (0,0,0)     1 running
```

```gdb
# See all threads in the current block
(cuda-gdb) info cuda threads
```

Output shows which threads are active, stopped, or have errors.

**Why this matters**: Understanding thread block organization helps debug synchronization and shared memory issues.

#### Technique 3: Memory access pattern analysis

```gdb
# Check GPU memory addresses:
(cuda-gdb) print a               # Input array GPU pointer
```

Output:

```
$9 = (!pop.scalar<f32> * @register) 0x302000200
```

```gdb
(cuda-gdb) print output          # Output array GPU pointer
```

Output:

```
$10 = (!pop.scalar<f32> * @register) 0x302000000
```

```gdb
# Verify memory access pattern using local variables:
(cuda-gdb) print a[i]            # Each thread accesses its own element using 'i'
```

Output:

```
$11 = {0}                        # Thread's input data
```

**Why this matters**: Memory access patterns affect performance and correctness. Wrong patterns cause race conditions or crashes.

#### Technique 4: Results verification and completion

```gdb
# After stepping through kernel execution, verify the final results
(cuda-gdb) print output[0]@4
```

Output:

```
$11 = {10.0, 11.0, 12.0, 13.0}    # Perfect! Each element increased by 10
```

```gdb
# Let the program complete normally
(cuda-gdb) continue
```

Output:

```
...Program output shows success...
```

```gdb
# Exit the debugger
(cuda-gdb) exit
```

You've completed debugging a GPU kernel execution from setup to results.

## Your GPU debugging progress: key insights

You've completed a comprehensive GPU debugging tutorial. Here's what you discovered about parallel computing:

### Deep insights about parallel execution

1. **Thread indexing in action**: You **saw** `thread_idx.x` have different values (0, 1, 2, 3...) across parallel threads - not just read about it in theory

2. **Memory access patterns revealed**: Each thread accesses `a[thread_idx.x]` and writes to `output[thread_idx.x]`, creating perfect data parallelism with no conflicts

3. **Parallel execution demystified**: Thousands of threads executing the **same kernel code** simultaneously, but each processing **different data elements**

4. **GPU memory hierarchy**: Arrays live in global GPU memory, accessible by all threads but with thread-specific indexing

### Debugging techniques that transfer to all puzzles

**From Puzzle 01 to Puzzle 08 and beyond**, you now have techniques that work universally:

- **Start with LLDB** for CPU-side issues (device setup, memory allocation)
- **Switch to CUDA-GDB** for GPU kernel issues (thread behavior, memory access)
- **Use conditional breakpoints** to focus on specific threads or data conditions
- **Navigate between threads** to understand parallel execution patterns
- **Verify memory access patterns** to catch race conditions and out-of-bounds errors

**Scalability**: These same techniques work whether you're debugging:

- **Puzzle 01**: 4-element arrays with simple addition
- **Puzzle 08**: Complex shared memory operations with thread synchronization
- **Production code**: Million-element arrays with sophisticated algorithms

---

## Essential debugging commands reference

Now that you've learned the debugging workflow, here's your **quick reference guide** for daily debugging sessions. Bookmark this section!

### GDB command abbreviations (save time!)

**Most commonly used shortcuts** for faster debugging:

| Abbreviation | Full Command | Function |
|-------------|-------------|----------|
| `r` | `run` | Start/launch the program |
| `c` | `continue` | Resume execution |
| `n` | `next` | Step over (same level) |
| `s` | `step` | Step into functions |
| `b` | `break` | Set breakpoint |
| `p` | `print` | Print variable value |
| `l` | `list` | Show source code |
| `q` | `quit` | Exit debugger |

**Examples:**

```bash
(cuda-gdb) r                    # Instead of 'run'
(cuda-gdb) b 39                 # Instead of 'break 39'
(cuda-gdb) p thread_id          # Instead of 'print thread_id'
(cuda-gdb) n                    # Instead of 'next'
(cuda-gdb) c                    # Instead of 'continue'
```

**⚡ Pro tip**: Use abbreviations for 3-5x faster debugging sessions!

## LLDB commands (CPU host code debugging)

**When to use**: Debugging device setup, memory allocation, program flow, host-side crashes

### Execution control

```bash
(lldb) run                    # Launch your program
(lldb) continue              # Resume execution (alias: c)
(lldb) step                  # Step into functions (source level)
(lldb) next                  # Step over functions (source level)
(lldb) finish                # Step out of current function
```

### Breakpoint management

```bash
(lldb) br set -n main        # Set breakpoint at main function
(lldb) br set -n function_name     # Set breakpoint at any function
(lldb) br list               # Show all breakpoints
(lldb) br delete 1           # Delete breakpoint #1
(lldb) br disable 1          # Temporarily disable breakpoint #1
```

### Variable inspection

```bash
(lldb) print variable_name   # Show variable value
(lldb) print pointer[offset]        # Dereference pointer
(lldb) print array[0]@4      # Show first 4 array elements
```

## CUDA-GDB commands (GPU kernel debugging)

**When to use**: Debugging GPU kernels, thread behavior, parallel execution, GPU memory issues

### GPU state inspection

```bash
(cuda-gdb) info cuda threads    # Show all GPU threads and their state
(cuda-gdb) info cuda blocks     # Show all thread blocks
(cuda-gdb) cuda kernel          # List active GPU kernels
```

### Thread navigation (The most powerful feature!)

```bash
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread coordinates
(cuda-gdb) cuda block (0,0)     # Switch to specific block
(cuda-gdb) cuda thread          # Show current thread coordinates
```

### Thread-specific variable inspection

```bash
# Local variables and function parameters:
(cuda-gdb) print i              # Local thread index variable
(cuda-gdb) print output         # Function parameter pointers
(cuda-gdb) print a              # Function parameter pointers
```

### GPU memory access

```bash
# Array inspection using local variables (what actually works):
(cuda-gdb) print array[i]       # Thread-specific array access using local variable
(cuda-gdb) print array[0]@4     # View multiple elements: {{val1}, {val2}, {val3}, {val4}}
```

### Advanced GPU debugging

```bash
# Memory watching
(cuda-gdb) watch array[i]     # Break on memory changes
(cuda-gdb) rwatch array[i]    # Break on memory reads
```

---

## Quick reference: Debugging decision tree

**🤔 What type of issue are you debugging?**

### Program crashes before GPU code runs

→ **Use LLDB debugging**

```bash
pixi run mojo debug your_program.mojo
```

### GPU kernel produces wrong results

→ **Use CUDA-GDB with conditional breakpoints**

```bash
pixi run mojo debug --cuda-gdb --break-on-launch your_program.mojo
```

### Performance issues or race conditions

→ **Use binary debugging for repeatability**

```bash
pixi run mojo build -O0 -g your_program.mojo -o debug_binary
pixi run mojo debug --cuda-gdb --break-on-launch debug_binary
```

---

## You've learned the essentials of GPU debugging

You've completed a comprehensive tutorial on GPU debugging fundamentals. Here's what you've accomplished:

### Skills you've learned

**Multi-level debugging knowledge**:

- ✅ **CPU host debugging** with LLDB - debug device setup, memory allocation, program flow
- ✅ **GPU kernel debugging** with CUDA-GDB - debug parallel threads, GPU memory, race conditions
- ✅ **JIT vs binary debugging** - choose the right approach for different scenarios
- ✅ **Environment management** with pixi - ensure consistent, reliable debugging setups

**Real parallel programming insights**:

- **Saw threads in action** - witnessed `thread_idx.x` having different values across parallel threads
- **Understood memory hierarchy** - debugged global GPU memory, shared memory, thread-local variables
- **Learned thread navigation** - jumped between thousands of parallel threads efficiently

### From theory to practice

You didn't just read about GPU debugging - you **experienced it**:

- **Debugged real code**: Puzzle 01's `add_10` kernel with actual GPU execution
- **Saw real debugger output**: LLDB assembly, CUDA-GDB thread states, memory addresses
- **Used professional tools**: The same CUDA-GDB used in production GPU development
- **Solved real scenarios**: Out-of-bounds access, race conditions, kernel launch failures

### Your debugging toolkit

**Quick decision guide** (keep this handy!):

| Problem Type | Tool | Command |
|-------------|------|---------|
| **Program crashes before GPU** | LLDB | `pixi run mojo debug program.mojo` |
| **GPU kernel issues** | CUDA-GDB | `pixi run mojo debug --cuda-gdb --break-on-launch program.mojo` |
| **Race conditions** | CUDA-GDB + thread nav | `(cuda-gdb) cuda thread (0,0,0)` |

**Essential commands** (for daily debugging):

```bash
# GPU thread inspection
(cuda-gdb) info cuda threads          # See all threads
(cuda-gdb) cuda thread (0,0,0)        # Switch threads
(cuda-gdb) print i                    # Local thread index (thread_idx.x equivalent)

# Smart breakpoints (using local variables since GPU built-ins don't work)
(cuda-gdb) break kernel if i == 0      # Focus on thread 0
(cuda-gdb) break kernel if array[i] > 100  # Focus on data conditions

# Memory debugging
(cuda-gdb) print array[i]              # Thread-specific data using local variable
(cuda-gdb) print array[0]@4            # Array segments: {{val1}, {val2}, {val3}, {val4}}
```

---

### Summary

GPU debugging involves thousands of parallel threads, complex memory hierarchies, and specialized tools. You now have:

- **Systematic workflows** that work for any GPU program
- **Professional tools** familiarity with LLDB and CUDA-GDB
- **Real experience** debugging actual parallel code
- **Practical strategies** for handling complex scenarios
- **Foundation** to tackle GPU debugging challenges

---

## Additional resources

- [Mojo Debugging Documentation](https://docs.modular.com/mojo/tools/debugging)
- [Mojo GPU Debugging Guide](https://docs.modular.com/mojo/tools/gpu-debugging)
- [NVIDIA CUDA-GDB User Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [CUDA-GDB Command Reference](https://docs.nvidia.com/cuda/cuda-gdb/index.html#command-reference)

**Note**: GPU debugging requires patience and systematic investigation. The workflow and commands in this puzzle provide the foundation for debugging complex GPU issues you'll encounter in real applications.
