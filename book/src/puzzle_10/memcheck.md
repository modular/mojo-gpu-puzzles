# 👮🏼‍♂️ The Silent Memory Corruption

## Overview

Learn how to detect memory violations that can silently corrupt GPU programs, even when tests appear to pass. Using NVIDIA's `compute-sanitizer` (avaible through `pixi`) with the `memcheck` tool, you'll discover hidden memory bugs that could cause unpredictable behavior in your GPU code.

**Key insight**: A GPU program can produce "correct" results while simultaneously performing illegal memory accesses.

**Prerequisites**: Understanding of [Puzzle 4 LayoutTensor](../puzzle_04/introduction_layout_tensor.md) and basic GPU memory concepts.

## The silent memory bug discovery

### Test passes, but is my code actually correct?

Let's start with a seemingly innocent program that appears to work perfectly (this is [Puzzle 04](../puzzle_04/layout_tensor.md) without guards):

```mojo
{{#include ../../../problems/p10/p10.mojo:add_10_2d_no_guard}}
```
<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p10/p10.mojo" class="filename">View full file: problems/p10/p10.mojo</a>

When you run this program normally, everything looks fine:

```bash
pixi run p10 --memory-bug
```

```txt
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
✅ Memory test PASSED! (memcheck may find bounds violations)
```

✅ **Test PASSED!** The output matches expected results perfectly. Case closed, right?

**Wrong!** Let's see what `compute-sanitizer` reveals:

```bash
MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

**Note**: `MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT=0` is a command-line environment variable setting that disables a device context's buffer cache. This setting can reveal memory issues, like bounds violations, that are otherwise masked by the normal caching behavior.

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,1,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (0,2,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (1,2,0) in block (0,0,0)
=========     Access at 0xe0c000214 is out of bounds
=========     and is 517 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,2,0) in block (0,0,0)
=========     Access at 0xe0c000218 is out of bounds
=========     and is 521 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuStreamSynchronize.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuEventCreate.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuMemFreeAsync.

========= ERROR SUMMARY: 7 errors
```

The program has **7 total errors** despite passing all tests:
- **4 memory violations** (Invalid __global__ read)
- **3 runtime errors** (caused by the memory violations)


## Understanding the hidden bug

### Root cause analysis

**The Problem:**
- **Tensor size**: 2×2 (valid indices: 0, 1)
- **Thread grid**: 3×3 (thread indices: 0, 1, 2)
- **Out-of-bounds threads**: `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)` access invalid memory
- **Missing bounds check**: No validation of `thread_idx` against tensor dimensions

### Understanding the 7 total errors

**4 Memory Violations:**
- Each out-of-bounds thread `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)` caused an "Invalid __global__ read"

**3 CUDA Runtime Errors:**
- `cuStreamSynchronize` failed due to kernel launch failure
- `cuEventCreate` failed during cleanup
- `cuMemFreeAsync` failed during memory deallocation

**Key Insight**: Memory violations have cascading effects - one bad memory access causes multiple downstream CUDA API failures.

**Why tests still passed:**
- Valid threads `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` wrote correct results
- Test only checked valid output locations
- Out-of-bounds accesses didn't immediately crash the program

## Understanding undefined behavior (UB)

### What is undefined behavior?

**Undefined Behavior (UB)** occurs when a program performs operations that have no defined meaning according to the language specification. Out-of-bounds memory access is a classic example of undefined behavior.

**Key characteristics of UB:**
- The program can do **literally anything**: crash, produce wrong results, appear to work, or corrupt memory
- **No guarantees**: Behavior may change between compilers, hardware, drivers, or even different runs

### Why undefined behavior is especially dangerous

**Correctness issues:**
- **Unpredictable results**: Your program may work during testing but fail in production
- **Non-deterministic behavior**: Same code can produce different results on different runs
- **Silent corruption**: UB can corrupt data without any visible errors
- **Compiler optimizations**: Compilers assume no UB occurs and may optimize in unexpected ways

**Security vulnerabilities:**
- **Buffer overflows**: Classic source of security exploits in systems programming
- **Memory corruption**: Can lead to privilege escalation and code injection attacks
- **Information leakage**: Out-of-bounds reads can expose sensitive data
- **Control flow hijacking**: UB can be exploited to redirect program execution

### GPU-specific undefined behavior dangers

**Massive scale impact:**
- **Thread divergence**: One thread's UB can affect entire warps (32 threads)
- **Memory coalescing**: Out-of-bounds access can corrupt neighboring threads' data
- **Kernel failures**: UB can cause entire GPU kernels to fail catastrophically

**Hardware variations:**
- **Different GPU architectures**: UB may manifest differently on different GPU models
- **Driver differences**: Same UB may behave differently across driver versions
- **Memory layout changes**: GPU memory allocation patterns can change UB manifestation

## Fixing the memory violation

### The solution

As we saw in [Puzzle 04](../puzzle_04/layout_tensor.md), we need to bound-check as follows:

```mojo
{{#include ../../../solutions/p04/p04_layout_tensor.mojo:add_10_2d_layout_tensor_solution}}
```

The fix is simple: **always validate thread indices against data dimensions** before accessing memory.

### Verification with compute-sanitizer

```bash
# Fix the bounds checking in your copy of p10.mojo, then run:
MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
✅ Memory test PASSED! (memcheck may find bounds violations)
========= ERROR SUMMARY: 0 errors
```

**✅ SUCCESS:** No memory violations detected!

## Key learning points

### Why manual bounds checking matters

1. **Clarity**: Makes the safety requirements explicit in the code
2. **Control**: You decide exactly what happens for out-of-bounds cases
3. **Debugging**: Easier to reason about when memory violations occur

### GPU memory safety rules

1. **Always validate thread indices** against data dimensions
2. **Avoid undefined behavior (UB) at all costs** - out-of-bounds access is UB and can break everything
3. **Use compute-sanitizer** during development and testing
4. **Never assume "it works" without memory checking**
5. **Test with different grid/block configurations** to catch undefined behavior (UB) that manifests inconsistently

### Compute-sanitizer best practices

```bash
MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo your_code.mojo
```

**Note**: You may see Mojo runtime warnings in the sanitizer output. Focus on the `========= Invalid` and `========= ERROR SUMMARY` lines for actual memory violations.
