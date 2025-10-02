# Puzzle 9: GPU Debugging Workflow

> ⚠️ This puzzle works on compatible **NVIDIA GPU** only. We are working to enable tooling support for other GPU vendors.

## When GPU programs fail

You've written GPU kernels, worked with shared memory, and coordinated thousands of parallel threads. Your code compiles. You run it expecting correct results, and then:

- **CRASH**
- **Wrong results**
- **Infinite hang**

This is GPU programming reality: **debugging parallel code running on thousands of threads simultaneously**. This is where theory meets practice, where algorithmic knowledge meets investigative skills.

## Why GPU debugging is challenging

Unlike traditional CPU debugging where you follow a single thread through sequential execution, GPU debugging requires you to:

- **Think in parallel**: Thousands of threads executing simultaneously, each potentially doing something different
- **Navigate multiple memory spaces**: Global memory, shared memory, registers, constant memory
- **Handle coordination failures**: Race conditions, barrier deadlocks, memory access violations
- **Debug optimized code**: JIT compilation, variable optimization, limited symbol information
- **Use specialized tools**: CUDA-GDB for kernel inspection, thread navigation, parallel state analysis

**GPU debugging skills provide deep understanding of parallel computing fundamentals**.

## What you'll learn in this puzzle

This puzzle teaches you to debug GPU code systematically. You'll learn the approaches, tools, and techniques that GPU developers use daily to solve complex parallel programming challenges.

### **Essential skills you'll develop**

1. **Professional debugging workflow** - The systematic approach professionals use
2. **Tool proficiency** - LLDB for host code, CUDA-GDB for GPU kernels
3. **Pattern recognition** - Common GPU bug types and symptoms
4. **Investigation techniques** - Finding root causes when variables are optimized out
5. **Thread coordination debugging** - Advanced GPU debugging skills

### **Real-world debugging scenarios**

You'll tackle the three most common GPU programming failures:

- **Memory crashes** - Null pointers, illegal memory access, segmentation faults
- **Logic bugs** - Correct execution with wrong results, algorithmic errors
- **Coordination deadlocks** - Barrier synchronization failures, infinite hangs

Each scenario teaches different investigation techniques and builds debugging intuition.

## Your debugging journey

This puzzle takes you through a carefully designed progression from basic debugging concepts to advanced parallel coordination failures:

### 📚 **Step 1: [Mojo GPU Debugging Essentials](./essentials.md)**

**Foundation building** - Learn the tools and workflow

- Set up your debugging environment with `pixi` and CUDA-GDB
- Learn the four debugging approaches: JIT vs binary, CPU vs GPU
- Learn essential CUDA-GDB commands for GPU kernel inspection
- Practice with hands-on examples using familiar code from previous puzzles
- Understand when to use each debugging approach

**Key outcome**: Professional debugging workflow and tool proficiency

### 🧐 **Step 2: [Detective Work: First Case](./first_case.md)**

**Memory crash investigation** - Debug a GPU program that crashes

- Investigate `CUDA_ERROR_ILLEGAL_ADDRESS` crashes
- Learn systematic pointer inspection techniques
- Learn null pointer detection and validation
- Practice professional crash analysis workflow
- Understand GPU memory access failures

**Key outcome**: Ability to debug GPU memory crashes and pointer issues

### 🔍 **Step 3: [Detective Work: Second Case](./second_case.md)**

**Logic bug investigation** - Debug a program with wrong results

- Investigate LayoutTensor-based algorithmic errors
- Learn execution flow analysis when variables are optimized out
- Learn loop boundary analysis and iteration counting
- Practice pattern recognition in incorrect results
- Debug without direct variable inspection

**Key outcome**: Ability to debug algorithmic errors and logic bugs in GPU kernels

### 🕵️ **Step 4: [Detective Work: Third Case](./third_case.md)**

**Barrier deadlock investigation** - Debug a program that hangs forever

- Investigate barrier synchronization failures
- Learn multi-thread state analysis across parallel execution
- Learn conditional execution path tracing
- Practice thread coordination debugging
- Understand the most challenging GPU debugging scenario

**Key outcome**: Advanced thread coordination debugging - the pinnacle of GPU debugging skills

## The detective mindset

GPU debugging requires a different mindset than traditional programming. You become a **detective** investigating a crime scene where:

- **The evidence is limited** - Variables are optimized out, symbols are mangled
- **Multiple suspects exist** - Thousands of threads, any could be the culprit
- **The timeline is complex** - Parallel execution, race conditions, timing dependencies
- **The tools are specialized** - CUDA-GDB, thread navigation, GPU memory inspection

But like any good detective, you'll learn to:

- **Follow the clues systematically** - Error messages, crash patterns, thread states
- **Form hypotheses** - What could cause this specific behavior?
- **Test theories** - Use debugging commands to verify or disprove ideas
- **Trace back to root causes** - From symptoms to the actual source of problems

## Prerequisites and expectations

**What you need to know**:

- GPU programming concepts from Puzzles 1-8 (thread indexing, memory management, barriers)
- Basic command-line comfort (you'll use terminal-based debugging tools)
- Patience and systematic thinking (GPU debugging requires methodical investigation)

**What you'll gain**:

- **Professional debugging skills** used in GPU development teams
- **Deep parallel computing understanding** that comes from seeing execution at the thread level
- **Problem-solving confidence** for the most challenging GPU programming scenarios
- **Tool proficiency** that will serve you throughout your GPU programming career

## Ready to begin?

GPU debugging is where you transition from *writing* GPU programs to *understanding* them deeply. Every professional GPU developer has spent countless hours debugging parallel code, learning to think in thousands of simultaneous threads, and developing the patience to investigate complex coordination failures.

This is your opportunity to join that elite group.

**Start your debugging journey**: [Mojo GPU Debugging Essentials](./essentials.md)

---

*"Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it."* - Brian Kernighan

*In GPU programming, this wisdom is amplified by a factor of thousands - the number of parallel threads you're debugging simultaneously.*
