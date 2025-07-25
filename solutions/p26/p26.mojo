from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.sync import (
    mbarrier_init,
    mbarrier_arrive,
    mbarrier_test_wait,
    async_copy_arrive,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from gpu.host import DeviceContext
from gpu.memory import async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout_tensor import copy_dram_to_sram_async
from sys import sizeof, argv, info
from testing import assert_true, assert_almost_equal

alias TPB = 256  # Threads per block for pipeline stages
alias SIZE = 1024  # Image size (1D for simplicity)
alias BLOCKS_PER_GRID = (4, 1)  # 4 blocks to process image in tiles
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)

# Multi-stage processing configuration
alias STAGE1_THREADS = TPB // 2  # First 128 threads: Load and preprocess
alias STAGE2_THREADS = TPB // 2  # Last 128 threads: Apply blur operation
alias BLUR_RADIUS = 2  # Simple blur kernel radius

# Double-buffered stencil configuration
alias STENCIL_ITERATIONS = 3  # Number of smoothing iterations
alias BUFFER_COUNT = 2  # Number of buffers

# Streaming matrix multiplication configuration (26C)
alias MATMUL_TPB = 16  # Tile size for matrix multiplication (16x16 tiles)
alias MATMUL_SIZE = 64  # Matrix size (64x64 for manageable testing)
alias MATMUL_BLOCKS_PER_GRID = (4, 4)  # 4x4 grid of blocks
alias MATMUL_THREADS_PER_BLOCK = (MATMUL_TPB, MATMUL_TPB)
alias matmul_layout = Layout.row_major(MATMUL_SIZE, MATMUL_SIZE)


# ANCHOR: multi_stage_pipeline_solution
fn multi_stage_image_blur_pipeline[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    """Multi-stage image blur pipeline demonstrating barrier coordination.

    This function implements a coordinated pipeline where different thread groups
    handle different processing stages, synchronized with explicit barriers.

    **Pipeline Architecture:**

    **Stage 1 - Load & Preprocess (Threads 0-127):**
    - First 128 threads load input data from global memory
    - Apply preprocessing enhancement (multiply by 1.1)
    - Each thread loads 2 elements: [local_i] and [local_i + 128]
    - Store preprocessed data in input_shared[TPB] array
    - Use bounds checking to handle edge cases with zero-padding

    **Barrier 1:** All threads wait until Stage 1 completes loading

    **Stage 2 - Horizontal Blur (Threads 128-255):**
    - Last 128 threads apply blur using BLUR_RADIUS=2 neighborhood
    - Each thread processes blur_idx = local_i - 128
    - For each element, average with neighbors: [blur_idx-2] to [blur_idx+2]
    - Store blurred results in blur_shared[TPB] array
    - Process both primary and secondary elements per thread

    **Barrier 2:** All threads wait until Stage 2 completes blurring

    **Stage 3 - Final Processing (All 256 threads):**
    - All threads participate in final smoothing operation
    - Apply neighbor smoothing: blend with [local_i-1] and [local_i+1]
    - Use formula: final_value = (value + neighbor) * 0.6 for smoothing
    - Write final results to global output memory

    **Final Barrier:** Ensure all threads complete before block exits

    **Memory Usage:**
    - input_shared[256]: Stores preprocessed input data from Stage 1
    - blur_shared[256]: Stores blur results from Stage 2
    - Both use LayoutTensorBuild for efficient shared memory allocation

    **Thread Coordination:**
    - STAGE1_THREADS = 128 (first half load data)
    - STAGE2_THREADS = 128 (second half blur data)
    - Stage 3 uses all 256 threads for final processing
    - Explicit barriers prevent race conditions between stages

    **Key Learning:**
    Understanding when barriers are necessary (between dependent stages)
    vs. wasteful (within independent operations) for optimal performance.
    """

    # Shared memory for pipeline stages using LayoutTensorBuild
    input_shared = tb[dtype]().row_major[TPB]().shared().alloc()
    blur_shared = tb[dtype]().row_major[TPB]().shared().alloc()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Stage 1: Load and preprocess (First half of threads)
    if local_i < STAGE1_THREADS:
        if global_i < size:
            # Load input data with simple preprocessing (add slight enhancement)
            input_shared[local_i] = input[global_i] * 1.1
            # Second element for this thread (processing 2 elements per thread)
            if local_i + STAGE1_THREADS < size:
                input_shared[local_i + STAGE1_THREADS] = (
                    input[global_i + STAGE1_THREADS] * 1.1
                )
        else:
            input_shared[local_i] = 0.0
            if local_i + STAGE1_THREADS < TPB:
                input_shared[local_i + STAGE1_THREADS] = 0.0

    # Barrier 1: Wait for Stage 1 (load/preprocess) to complete
    barrier()

    # Stage 2: Apply horizontal blur (Second half of threads)
    if local_i >= STAGE1_THREADS:
        var blur_idx = local_i - STAGE1_THREADS
        var blur_sum: Scalar[dtype] = 0.0
        var blur_count = 0

        # Simple horizontal blur - average with neighbors
        for offset in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
            var sample_idx = blur_idx + offset
            if sample_idx >= 0 and sample_idx < TPB:
                blur_sum += rebind[Scalar[dtype]](input_shared[sample_idx])
                blur_count += 1

        if blur_count > 0:
            blur_shared[blur_idx] = blur_sum / blur_count
        else:
            blur_shared[blur_idx] = 0.0

        # Process second element if within bounds
        var second_idx = blur_idx + STAGE1_THREADS
        if second_idx < TPB:
            blur_sum = 0.0
            blur_count = 0
            for offset in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
                sample_idx = second_idx + offset
                if sample_idx >= 0 and sample_idx < TPB:
                    blur_sum += rebind[Scalar[dtype]](input_shared[sample_idx])
                    blur_count += 1

            if blur_count > 0:
                blur_shared[second_idx] = blur_sum / blur_count
            else:
                blur_shared[second_idx] = 0.0

    # Barrier 2: Wait for Stage 2 (horizontal blur) to complete
    barrier()

    # Stage 3: Final processing (All threads participate)
    if global_i < size:
        # Apply final smoothing and write to output
        # All threads now work together for final stage
        var final_value = blur_shared[local_i]

        # Simple final processing - slight smoothing with immediate neighbors
        if local_i > 0:
            final_value = (final_value + blur_shared[local_i - 1]) * 0.6
        if local_i < TPB - 1:
            final_value = (final_value + blur_shared[local_i + 1]) * 0.6

        output[global_i] = final_value

    # Final barrier: Ensure all threads complete before block finishes
    barrier()


# ANCHOR_END: multi_stage_pipeline_solution


# ANCHOR: double_buffered_stencil_solution
fn double_buffered_stencil_computation[
    layout: Layout
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    """Double-buffered stencil computation demonstrating memory barrier coordination.

    This function implements an iterative stencil operation using double-buffering
    with explicit memory barriers for coordination between buffer swaps.

    **Double-Buffering Architecture:**

    **Buffer Management:**
    - buffer_A[256]: Active buffer for current iteration reads
    - buffer_B[256]: Target buffer for current iteration writes
    - Buffers alternate roles each iteration: A↔B, B↔A
    - mbarrier coordination ensures safe buffer switching

    **Iterative Stencil Processing:**
    - **Iteration 0**: Read from buffer_A → Write to buffer_B
    - **Memory Barrier**: Ensure all writes to buffer_B complete
    - **Iteration 1**: Read from buffer_B → Write to buffer_A
    - **Memory Barrier**: Ensure all writes to buffer_A complete
    - Continue alternating for STENCIL_ITERATIONS

    **Stencil Operation (3-point average):**
    - For each element i: result[i] = (data[i-1] + data[i] + data[i+1]) / 3
    - Boundary handling: Use zero-padding for out-of-bounds access
    - Each iteration refines the smoothing operation

    **Memory Barrier Coordination (using mbarrier APIs):**
      - **mbarrier_init(barrier_ptr, thread_count)**: Initialize shared memory barrier for TPB threads
      - **mbarrier_arrive(barrier_ptr)**: Each thread signals completion of write phase
      - **mbarrier_test_wait(barrier_ptr, expected_count)**: Wait for all threads before buffer swap
      - **Critical**: Prevents reading from buffer while others still writing to avoid race conditions

    **Thread Participation:**
    - All 256 threads participate in each iteration
    - Each thread processes one element of the stencil
    - Synchronous execution: all threads must complete before buffer swap

    **Educational Purpose:**
    This demonstrates essential double-buffering patterns:
    - **Memory consistency**: Explicit barrier management vs automatic sync
    - **Iterative algorithms**: Building complex operations from simple steps
    - **Buffer management**: Safe alternation between read/write buffers
    - **Race condition prevention**: Memory barriers ensure data integrity

    **Key Learning:**
    Understanding explicit memory barrier coordination for complex memory
    access patterns that require precise timing and consistency guarantees.
    """

    # Double-buffering: Two shared memory buffers
    buffer_A = tb[dtype]().row_major[TPB]().shared().alloc()
    buffer_B = tb[dtype]().row_major[TPB]().shared().alloc()

    # Initialize memory barriers for coordinating buffer swaps using mbarrier APIs
    # Need separate barriers for different synchronization points
    init_barrier = tb[DType.uint64]().row_major[1]().shared().alloc()
    iter_barrier = tb[DType.uint64]().row_major[1]().shared().alloc()
    final_barrier = tb[DType.uint64]().row_major[1]().shared().alloc()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Initialize shared memory barriers for TPB threads (only thread 0 initializes)
    if local_i == 0:
        mbarrier_init(init_barrier.ptr, TPB)
        mbarrier_init(iter_barrier.ptr, TPB)
        mbarrier_init(final_barrier.ptr, TPB)

    # Initialize buffer_A with input data
    if local_i < TPB and global_i < size:
        buffer_A[local_i] = input[global_i]
    else:
        buffer_A[local_i] = 0.0

    # Initial synchronization using mbarrier - ensure buffer_A is fully loaded
    _ = mbarrier_arrive(init_barrier.ptr)
    _ = mbarrier_test_wait(init_barrier.ptr, TPB)

    # Iterative stencil processing with double-buffering
    @parameter
    for iteration in range(STENCIL_ITERATIONS):

        @parameter
        if iteration % 2 == 0:
            # Even iteration: Read from A, Write to B
            if local_i < TPB:
                var stencil_sum: Scalar[dtype] = 0.0
                var stencil_count = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    var sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_A[sample_idx]
                        )
                        stencil_count += 1

                # Write averaged result to buffer_B
                if stencil_count > 0:
                    buffer_B[local_i] = stencil_sum / stencil_count
                else:
                    buffer_B[local_i] = buffer_A[local_i]

        else:
            # Odd iteration: Read from B, Write to A
            if local_i < TPB:
                var stencil_sum: Scalar[dtype] = 0.0
                var stencil_count = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    var sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_B[sample_idx]
                        )
                        stencil_count += 1

                # Write averaged result to buffer_A
                if stencil_count > 0:
                    buffer_A[local_i] = stencil_sum / stencil_count
                else:
                    buffer_A[local_i] = buffer_B[local_i]

        # Memory barrier coordination using mbarrier APIs
        # All threads must complete their work before buffer swap
        _ = mbarrier_arrive(iter_barrier.ptr)
        _ = mbarrier_test_wait(iter_barrier.ptr, TPB)

        # Reinitialize barrier for next iteration (only thread 0)
        if local_i == 0:
            mbarrier_init(iter_barrier.ptr, TPB)

    # Write final results to output (from last active buffer)
    if local_i < TPB and global_i < size:

        @parameter
        if STENCIL_ITERATIONS % 2 == 0:
            # Even iterations end in buffer_A
            output[global_i] = buffer_A[local_i]
        else:
            # Odd iterations end in buffer_B
            output[global_i] = buffer_B[local_i]

    # Final barrier: Ensure all output writes complete
    _ = mbarrier_arrive(final_barrier.ptr)
    _ = mbarrier_test_wait(final_barrier.ptr, TPB)


# ANCHOR_END: double_buffered_stencil_solution


# ANCHOR: streaming_matrix_multiplication_solution
fn streaming_matrix_multiplication[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    """Streaming matrix multiplication demonstrating advanced async copy coordination.

    This function implements a tiled matrix multiplication with streaming optimization
    that overlaps tile loading with computation using the required GPU sync primitives.

    **Streaming Architecture with Required APIs:**

    **async_copy_arrive() Usage:**
    - Signals completion of individual copy operations within each thread
    - Tracks when each thread has finished copying its portion of a tile
    - Enables fine-grained coordination of distributed copy work
    - Essential for overlapping computation with memory operations

    **cp_async_bulk_commit_group() Usage:**
    - Groups multiple tile copy operations into bulk transfer units
    - Allows hardware to optimize multiple transfers together
    - Creates coordination groups for A and B tile pairs
    - Enables efficient bandwidth utilization through batched operations

    **cp_async_bulk_wait_group() Usage:**
    - Waits selectively for specific bulk copy groups to complete
    - Avoids unnecessary synchronization on unrelated transfers
    - Enables precise timing control for compute-memory overlap
    - Supports ping-pong buffering between multiple copy groups

    **Memory Pipeline Architecture:**

    **Stage 1 - Distributed Async Copy Launch:**
    Each thread copies a portion of the tiles to shared memory using thread-level
    async coordination. Multiple threads work together to load each tile.

    **Stage 2 - Bulk Group Coordination:**
    Copy operations are grouped into bulk units for hardware optimization.
    Groups alternate to enable continuous streaming pipeline.

    **Stage 3 - Selective Synchronization:**
    Wait only for the specific bulk group containing the data we need,
    while other groups continue processing in parallel.

    **Stage 4 - Overlapped Computation:**
    Compute on ready data while next bulk group loads asynchronously.
    """

    local_row = thread_idx.y
    local_col = thread_idx.x
    block_row = block_idx.y
    block_col = block_idx.x

    # Global position of this thread's output element
    global_row = block_row * MATMUL_TPB + local_row
    global_col = block_col * MATMUL_TPB + local_col

    # Double-buffered shared memory for streaming pipeline
    a_shared_0 = (
        tb[dtype]().row_major[MATMUL_TPB, MATMUL_TPB]().shared().alloc()
    )
    b_shared_0 = (
        tb[dtype]().row_major[MATMUL_TPB, MATMUL_TPB]().shared().alloc()
    )
    a_shared_1 = (
        tb[dtype]().row_major[MATMUL_TPB, MATMUL_TPB]().shared().alloc()
    )
    b_shared_1 = (
        tb[dtype]().row_major[MATMUL_TPB, MATMUL_TPB]().shared().alloc()
    )

    var acc: output.element_type = 0
    alias num_tiles = (size + MATMUL_TPB - 1) // MATMUL_TPB

    # Streaming matrix multiplication pipeline using required APIs
    @parameter
    for tile in range(num_tiles):
        # Determine buffer set for current tile (ping-pong between 0 and 1)
        alias buffer_set = tile % 2
        alias bulk_group = tile % 2  # Use compile-time constant for bulk groups

        # Stage 1: Distributed Async Copy Launch
        # Each thread copies its portion of the A and B tiles
        if global_row < size and tile * MATMUL_TPB + local_col < size:
            # Copy A tile element: threads load row-wise portions
            @parameter
            if buffer_set == 0:
                a_shared_0[local_row, local_col] = a[
                    global_row, tile * MATMUL_TPB + local_col
                ]
                # Signal this thread's A copy operation has arrived/completed
                async_copy_arrive(
                    a_shared_0.ptr + local_row * MATMUL_TPB + local_col
                )
            else:
                a_shared_1[local_row, local_col] = a[
                    global_row, tile * MATMUL_TPB + local_col
                ]
                # Signal this thread's A copy operation has arrived/completed
                async_copy_arrive(
                    a_shared_1.ptr + local_row * MATMUL_TPB + local_col
                )

        if tile * MATMUL_TPB + local_row < size and global_col < size:
            # Copy B tile element: threads load column-wise portions
            @parameter
            if buffer_set == 0:
                b_shared_0[local_row, local_col] = b[
                    tile * MATMUL_TPB + local_row, global_col
                ]
                # Signal this thread's B copy operation has arrived/completed
                async_copy_arrive(
                    b_shared_0.ptr + local_row * MATMUL_TPB + local_col
                )
            else:
                b_shared_1[local_row, local_col] = b[
                    tile * MATMUL_TPB + local_row, global_col
                ]
                # Signal this thread's B copy operation has arrived/completed
                async_copy_arrive(
                    b_shared_1.ptr + local_row * MATMUL_TPB + local_col
                )

        # Stage 2: Bulk Group Coordination
        # Group the copy operations for this tile into a bulk transfer unit
        # This allows hardware to optimize multiple transfers together
        cp_async_bulk_commit_group()

        # Stage 3: Selective Synchronization
        # Wait for the current bulk group to complete before computation
        # This enables overlap: while we compute tile N, tile N+1 can be loading
        @parameter
        if bulk_group == 0:
            cp_async_bulk_wait_group[0]()
        else:
            cp_async_bulk_wait_group[1]()

        # Stage 4: Overlapped Computation
        # Compute matrix multiplication on the ready tile data
        if global_row < size and global_col < size:

            @parameter
            for k in range(min(MATMUL_TPB, size - tile * MATMUL_TPB)):

                @parameter
                if buffer_set == 0:
                    acc += a_shared_0[local_row, k] * b_shared_0[k, local_col]
                else:
                    acc += a_shared_1[local_row, k] * b_shared_1[k, local_col]

    # Write final accumulated result to global memory
    if global_row < size and global_col < size:
        output[global_row, global_col] = acc


# ANCHOR_END: streaming_matrix_multiplication_solution


def test_multi_stage_pipeline():
    """Test Puzzle 26A: Multi-Stage Pipeline Coordination."""
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialize input with a simple pattern
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a simple wave pattern for blurring
                inp_host[i] = Float32(i % 10) + Float32(i / 100.0)

        # Create LayoutTensors
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        inp_tensor = LayoutTensor[mut=False, dtype, layout](inp.unsafe_ptr())

        ctx.enqueue_function[multi_stage_image_blur_pipeline[layout]](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Simple verification - check that output differs from input and values are reasonable
        with out.map_to_host() as out_host, inp.map_to_host() as inp_host:
            print("Multi-stage pipeline blur completed")
            print("Input sample:", inp_host[0], inp_host[1], inp_host[2])
            print("Output sample:", out_host[0], out_host[1], out_host[2])

            # Basic verification - output should be different from input (pipeline processed them)
            assert_true(
                abs(out_host[0] - inp_host[0]) > 0.001,
                "Pipeline should modify values",
            )
            assert_true(
                abs(out_host[1] - inp_host[1]) > 0.001,
                "Pipeline should modify values",
            )
            assert_true(
                abs(out_host[2] - inp_host[2]) > 0.001,
                "Pipeline should modify values",
            )

            # Values should be reasonable (not NaN, not extreme)
            for i in range(10):
                assert_true(
                    out_host[i] >= 0.0, "Output values should be non-negative"
                )
                assert_true(
                    out_host[i] < 1000.0, "Output values should be reasonable"
                )

            print("✓ Multi-stage pipeline coordination test PASSED!")


def test_double_buffered_stencil():
    """Test Puzzle 26B: Double-Buffered Stencil Computation."""
    with DeviceContext() as ctx:
        # Test Puzzle 26B: Double-Buffered Stencil Computation
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # Initialize input with a different pattern for stencil testing
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a step pattern that will be smoothed by stencil
                inp_host[i] = Float32(1.0 if i % 20 < 10 else 0.0)

        # Create LayoutTensors for Puzzle 26B
        out_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        inp_tensor = LayoutTensor[mut=False, dtype, layout](inp.unsafe_ptr())

        ctx.enqueue_function[double_buffered_stencil_computation[layout]](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Compute expected values on CPU for verification
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

        with inp.map_to_host() as inp_host:
            # CPU simulation matching GPU's multi-block execution (BLOCKS_PER_GRID = (4, 1))
            for block_id in range(4):  # Simulate each GPU block
                var cpu_buffer_A = List[Float32](capacity=TPB)
                var cpu_buffer_B = List[Float32](capacity=TPB)

                # Initialize buffer_A per block (matching GPU initialization)
                for local_i in range(TPB):
                    var global_i = TPB * block_id + local_i
                    if local_i < TPB and global_i < SIZE:
                        cpu_buffer_A.append(inp_host[global_i])
                    else:
                        cpu_buffer_A.append(0.0)
                    cpu_buffer_B.append(0.0)

                # Simulate STENCIL_ITERATIONS iterations with double-buffering per block
                for iteration in range(STENCIL_ITERATIONS):
                    if iteration % 2 == 0:
                        # Even iteration: Read from A, Write to B
                        for local_i in range(TPB):
                            var stencil_sum: Float32 = 0.0
                            var stencil_count = 0

                            # 3-point stencil: [i-1, i, i+1]
                            for offset in range(-1, 2):
                                var sample_idx = local_i + offset
                                if sample_idx >= 0 and sample_idx < TPB:
                                    stencil_sum += cpu_buffer_A[sample_idx]
                                    stencil_count += 1

                            if stencil_count > 0:
                                cpu_buffer_B[local_i] = (
                                    stencil_sum / stencil_count
                                )
                            else:
                                cpu_buffer_B[local_i] = cpu_buffer_A[local_i]
                    else:
                        # Odd iteration: Read from B, Write to A
                        for local_i in range(TPB):
                            var stencil_sum: Float32 = 0.0
                            var stencil_count = 0

                            # 3-point stencil: [i-1, i, i+1]
                            for offset in range(-1, 2):
                                var sample_idx = local_i + offset
                                if sample_idx >= 0 and sample_idx < TPB:
                                    stencil_sum += cpu_buffer_B[sample_idx]
                                    stencil_count += 1

                            if stencil_count > 0:
                                cpu_buffer_A[local_i] = (
                                    stencil_sum / stencil_count
                                )
                            else:
                                cpu_buffer_A[local_i] = cpu_buffer_B[local_i]

                # Copy results from final active buffer to expected (per block)
                for local_i in range(TPB):
                    var global_i = TPB * block_id + local_i
                    if local_i < TPB and global_i < SIZE:
                        if STENCIL_ITERATIONS % 2 == 0:
                            # Even iterations end in buffer_A
                            expected[global_i] = cpu_buffer_A[local_i]
                        else:
                            # Odd iterations end in buffer_B
                            expected[global_i] = cpu_buffer_B[local_i]

        # Verification against computed expected values
        with inp.map_to_host() as inp_host, out.map_to_host() as out_host:
            print("Double-buffered stencil completed")
            print("Input sample:", inp_host[0], inp_host[1], inp_host[2])
            print("GPU output sample:", out_host[0], out_host[1], out_host[2])
            print("CPU expected sample:", expected[0], expected[1], expected[2])

            # Compare GPU output with CPU-computed expected values
            var max_error: Float32 = 0.0
            for i in range(SIZE):
                var error = abs(out_host[i] - expected[i])
                if error > max_error:
                    max_error = error

                # Assert values match within tolerance
                assert_true(
                    error < 0.0001,
                    "GPU output should match CPU expected values (error: "
                    + String(error)
                    + " at index "
                    + String(i)
                    + ")",
                )

            print("Maximum error between GPU and CPU:", max_error)
            print("✓ Double-buffered stencil test PASSED!")


def test_streaming_matrix_multiplication():
    """Test Puzzle 26C: Streaming Matrix Multiplication."""
    with DeviceContext() as ctx:
        # Test Puzzle 26C: Streaming Matrix Multiplication
        out_c = ctx.enqueue_create_buffer[dtype](
            MATMUL_SIZE * MATMUL_SIZE
        ).enqueue_fill(0)
        a_mat = ctx.enqueue_create_buffer[dtype](
            MATMUL_SIZE * MATMUL_SIZE
        ).enqueue_fill(0)
        b_mat = ctx.enqueue_create_buffer[dtype](
            MATMUL_SIZE * MATMUL_SIZE
        ).enqueue_fill(0)

        # Initialize matrices with simple patterns for verification
        with a_mat.map_to_host() as a_host, b_mat.map_to_host() as b_host:
            for row in range(MATMUL_SIZE):
                for col in range(MATMUL_SIZE):
                    # Matrix A: simple row pattern
                    a_host[row * MATMUL_SIZE + col] = (
                        Float32(row + col + 1) / 10.0
                    )
                    # Matrix B: identity-like pattern for easier verification
                    if row == col:
                        b_host[row * MATMUL_SIZE + col] = 1.0
                    else:
                        b_host[row * MATMUL_SIZE + col] = 0.1

        # Create LayoutTensors for Puzzle 26C
        out_tensor = LayoutTensor[mut=True, dtype, matmul_layout](
            out_c.unsafe_ptr()
        )
        a_tensor = LayoutTensor[mut=False, dtype, matmul_layout](
            a_mat.unsafe_ptr()
        )
        b_tensor = LayoutTensor[mut=False, dtype, matmul_layout](
            b_mat.unsafe_ptr()
        )

        ctx.enqueue_function[
            streaming_matrix_multiplication[matmul_layout, MATMUL_SIZE]
        ](
            out_tensor,
            a_tensor,
            b_tensor,
            grid_dim=MATMUL_BLOCKS_PER_GRID,
            block_dim=MATMUL_THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # CPU verification using naive matrix multiplication
        expected_c = ctx.enqueue_create_host_buffer[dtype](
            MATMUL_SIZE * MATMUL_SIZE
        ).enqueue_fill(0)

        with a_mat.map_to_host() as a_host, b_mat.map_to_host() as b_host:
            for row in range(MATMUL_SIZE):
                for col in range(MATMUL_SIZE):
                    var acc: Float32 = 0.0
                    for k in range(MATMUL_SIZE):
                        acc += (
                            a_host[row * MATMUL_SIZE + k]
                            * b_host[k * MATMUL_SIZE + col]
                        )
                    expected_c[row * MATMUL_SIZE + col] = acc

        # Verification against CPU results
        with out_c.map_to_host() as out_host:
            print("Streaming matrix multiplication completed")
            print("GPU output sample (0,0):", out_host[0])
            print("CPU expected sample (0,0):", expected_c[0])

            # Compare GPU output with CPU expected values
            var max_error: Float32 = 0.0
            for i in range(MATMUL_SIZE * MATMUL_SIZE):
                var error = abs(out_host[i] - expected_c[i])
                if error > max_error:
                    max_error = error

                # Assert values match within tolerance
                assert_almost_equal(
                    out_host[i],
                    expected_c[i],
                    rtol=0.01,
                    atol=0.01,
                )

            print("Maximum error between GPU and CPU:", max_error)
            print("✓ Streaming matrix multiplication test PASSED!")


def main():
    """Run GPU synchronization tests based on command line arguments."""
    print("Puzzle 26: GPU Synchronization Primitives")
    print("=" * 50)

    # Parse command line arguments
    if len(argv()) != 2:
        print(
            "Usage: p26.mojo [--multi-stage | --double-buffer |"
            " --streaming-matmul]"
        )
        print("  --multi-stage: Test multi-stage pipeline coordination (26A)")
        print(
            "  --double-buffer: Test double-buffered stencil computation (26B)"
        )
        print(
            "  --streaming-matmul: Test streaming matrix multiplication (26C)"
        )
        return

    if argv()[1] == "--multi-stage":
        print("TPB:", TPB)
        print("SIZE:", SIZE)
        print("STAGE1_THREADS:", STAGE1_THREADS)
        print("STAGE2_THREADS:", STAGE2_THREADS)
        print("BLUR_RADIUS:", BLUR_RADIUS)
        print("")
        print("Testing Puzzle 26A: Multi-Stage Pipeline Coordination")
        print("=" * 60)
        test_multi_stage_pipeline()
    elif argv()[1] == "--double-buffer":
        print("TPB:", TPB)
        print("SIZE:", SIZE)
        print("STENCIL_ITERATIONS:", STENCIL_ITERATIONS)
        # print("BUFFER_COUNT:", BUFFER_COUNT)
        print("")
        print("Testing Puzzle 26B: Double-Buffered Stencil Computation")
        print("=" * 60)
        test_double_buffered_stencil()
    elif argv()[1] == "--streaming-matmul":
        print("MATMUL_TPB:", MATMUL_TPB)
        print("MATMUL_SIZE:", MATMUL_SIZE)
        # print("MATMUL_BLOCKS_PER_GRID:", MATMUL_BLOCKS_PER_GRID)
        # print("MATMUL_THREADS_PER_BLOCK:", MATMUL_THREADS_PER_BLOCK)
        print("")
        print("Testing Puzzle 26C: Streaming Matrix Multiplication")
        print("=" * 60)
        test_streaming_matrix_multiplication()
    else:
        print(
            "Usage: p26.mojo [--multi-stage | --double-buffer |"
            " --streaming-matmul]"
        )
        print("  --multi-stage: Test multi-stage pipeline coordination (26A)")
        print(
            "  --double-buffer: Test double-buffered stencil computation (26B)"
        )
        print(
            "  --streaming-matmul: Test streaming matrix multiplication (26C)"
        )
        return
