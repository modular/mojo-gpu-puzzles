from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.sync import (
    mbarrier_init,
    mbarrier_arrive,
    mbarrier_test_wait,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from sys import size_of, argv, info
from testing import assert_true, assert_almost_equal

alias TPB = 256  # Threads per block for pipeline stages
alias SIZE = 1024  # Image size (1D for simplicity)
alias BLOCKS_PER_GRID = (4, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)

# Multi-stage processing configuration
alias STAGE1_THREADS = TPB // 2
alias STAGE2_THREADS = TPB // 2
alias BLUR_RADIUS = 2


# ANCHOR: multi_stage_pipeline_solution
fn multi_stage_image_blur_pipeline[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    """Multi-stage image blur pipeline with barrier coordination.

    Stage 1 (threads 0-127): Load input data and apply 1.1x preprocessing
    Stage 2 (threads 128-255): Apply 5-point blur with BLUR_RADIUS=2
    Stage 3 (all threads): Final neighbor smoothing and output
    """

    # Shared memory buffers for pipeline stages
    input_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    blur_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Stage 1: Load and preprocess (threads 0-127)
    if local_i < STAGE1_THREADS:
        if global_i < size:
            input_shared[local_i] = input[global_i] * 1.1
            # Each thread loads 2 elements
            if local_i + STAGE1_THREADS < size:
                input_shared[local_i + STAGE1_THREADS] = (
                    input[global_i + STAGE1_THREADS] * 1.1
                )
        else:
            # Zero-padding for out-of-bounds
            input_shared[local_i] = 0.0
            if local_i + STAGE1_THREADS < TPB:
                input_shared[local_i + STAGE1_THREADS] = 0.0

    barrier()  # Wait for Stage 1 completion

    # Stage 2: Apply blur (threads 128-255)
    if local_i >= STAGE1_THREADS:
        blur_idx = local_i - STAGE1_THREADS
        var blur_sum: Scalar[dtype] = 0.0
        blur_count = 0

        # 5-point blur kernel
        for offset in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
            sample_idx = blur_idx + offset
            if sample_idx >= 0 and sample_idx < TPB:
                blur_sum += rebind[Scalar[dtype]](input_shared[sample_idx])
                blur_count += 1

        if blur_count > 0:
            blur_shared[blur_idx] = blur_sum / blur_count
        else:
            blur_shared[blur_idx] = 0.0

        # Process second element
        second_idx = blur_idx + STAGE1_THREADS
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

    barrier()  # Wait for Stage 2 completion

    # Stage 3: Final smoothing (all threads)
    if global_i < size:
        final_value = blur_shared[local_i]

        # Neighbor smoothing with 0.6 scaling
        if local_i > 0:
            final_value = (final_value + blur_shared[local_i - 1]) * 0.6
        if local_i < TPB - 1:
            final_value = (final_value + blur_shared[local_i + 1]) * 0.6

        output[global_i] = final_value

    barrier()  # Ensure all writes complete


# ANCHOR_END: multi_stage_pipeline_solution


# Double-buffered stencil configuration
alias STENCIL_ITERATIONS = 3
alias BUFFER_COUNT = 2


# ANCHOR: double_buffered_stencil_solution
fn double_buffered_stencil_computation[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    """Double-buffered stencil computation with memory barrier coordination.

    Iteratively applies 3-point stencil using alternating buffers.
    Uses mbarrier APIs for precise buffer swap coordination.
    """

    # Double-buffering: Two shared memory buffers
    buffer_A = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    buffer_B = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Memory barriers for coordinating buffer swaps
    init_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    iter_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    final_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Initialize barriers (only thread 0)
    if local_i == 0:
        mbarrier_init(init_barrier.ptr, TPB)
        mbarrier_init(iter_barrier.ptr, TPB)
        mbarrier_init(final_barrier.ptr, TPB)

    # Initialize buffer_A with input data
    if local_i < TPB and global_i < size:
        buffer_A[local_i] = input[global_i]
    else:
        buffer_A[local_i] = 0.0

    # Wait for buffer_A initialization
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
                var stencil_count: Int = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_A[sample_idx]
                        )
                        stencil_count += 1

                if stencil_count > 0:
                    buffer_B[local_i] = stencil_sum / stencil_count
                else:
                    buffer_B[local_i] = buffer_A[local_i]

        else:
            # Odd iteration: Read from B, Write to A
            if local_i < TPB:
                var stencil_sum: Scalar[dtype] = 0.0
                var stencil_count: Int = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_B[sample_idx]
                        )
                        stencil_count += 1

                if stencil_count > 0:
                    buffer_A[local_i] = stencil_sum / stencil_count
                else:
                    buffer_A[local_i] = buffer_B[local_i]

        # Memory barrier: wait for all writes before buffer swap
        _ = mbarrier_arrive(iter_barrier.ptr)
        _ = mbarrier_test_wait(iter_barrier.ptr, TPB)

        # Reinitialize barrier for next iteration
        if local_i == 0:
            mbarrier_init(iter_barrier.ptr, TPB)

    # Write final results from active buffer
    if local_i < TPB and global_i < size:

        @parameter
        if STENCIL_ITERATIONS % 2 == 0:
            # Even iterations end in buffer_A
            output[global_i] = buffer_A[local_i]
        else:
            # Odd iterations end in buffer_B
            output[global_i] = buffer_B[local_i]

    # Final barrier
    _ = mbarrier_arrive(final_barrier.ptr)
    _ = mbarrier_test_wait(final_barrier.ptr, TPB)


# ANCHOR_END: double_buffered_stencil_solution


def test_multi_stage_pipeline():
    """Test Puzzle 26A: Multi-Stage Pipeline Coordination."""
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](SIZE)
        inp.enqueue_fill(0)

        # Initialize input with a simple pattern
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a simple wave pattern for blurring
                inp_host[i] = Float32(i % 10) + Float32(i / 100.0)

        # Create LayoutTensors
        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        inp_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](inp)

        alias kernel = multi_stage_image_blur_pipeline[layout]
        ctx.enqueue_function_checked[kernel, kernel](
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

            print("✅ Multi-stage pipeline coordination test PASSED!")


def test_double_buffered_stencil():
    """Test Puzzle 26B: Double-Buffered Stencil Computation."""
    with DeviceContext() as ctx:
        # Test Puzzle 26B: Double-Buffered Stencil Computation
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](SIZE)
        inp.enqueue_fill(0)

        # Initialize input with a different pattern for stencil testing
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a step pattern that will be smoothed by stencil
                inp_host[i] = Float32(1.0 if i % 20 < 10 else 0.0)

        # Create LayoutTensors for Puzzle 26B
        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        inp_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](inp)

        alias kernel = double_buffered_stencil_computation[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        # Simple verification - check that GPU implementation works correctly
        with inp.map_to_host() as inp_host, out.map_to_host() as out_host:
            print("Double-buffered stencil completed")
            print("Input sample:", inp_host[0], inp_host[1], inp_host[2])
            print("GPU output sample:", out_host[0], out_host[1], out_host[2])

            # Basic sanity checks
            var processing_occurred = False
            var all_values_valid = True

            for i in range(SIZE):
                # Check if processing occurred (output should differ from step pattern)
                if abs(out_host[i] - inp_host[i]) > 0.001:
                    processing_occurred = True

                # Check for invalid values (NaN, infinity, or out of reasonable range)
                if out_host[i] < 0.0 or out_host[i] > 1.0:
                    all_values_valid = False
                    break

            # Verify the stencil smoothed the step pattern
            assert_true(
                processing_occurred, "Stencil should modify the input values"
            )
            assert_true(
                all_values_valid,
                "All output values should be in valid range [0,1]",
            )

            # Check that values are smoothed (no sharp transitions)
            var smooth_transitions = True
            for i in range(1, SIZE - 1):
                # Check if transitions are reasonably smooth (not perfect step function)
                var left_diff = abs(out_host[i] - out_host[i - 1])
                var right_diff = abs(out_host[i + 1] - out_host[i])
                # After 3 stencil iterations, sharp 0->1 transitions should be smoothed
                if left_diff > 0.8 or right_diff > 0.8:
                    smooth_transitions = False
                    break

            assert_true(
                smooth_transitions, "Stencil should smooth sharp transitions"
            )

            print("✅ Double-buffered stencil test PASSED!")


def main():
    """Run GPU synchronization tests based on command line arguments."""
    print("Puzzle 26: GPU Synchronization Primitives")
    print("=" * 50)

    # Parse command line arguments
    if len(argv()) != 2:
        print("Usage: p26.mojo [--multi-stage | --double-buffer]")
        print("  --multi-stage: Test multi-stage pipeline coordination")
        print("  --double-buffer: Test double-buffered stencil computation")
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
        print("BUFFER_COUNT:", BUFFER_COUNT)
        print("")
        print("Testing Puzzle 26B: Double-Buffered Stencil Computation")
        print("=" * 60)
        test_double_buffered_stencil()
    else:
        print("Usage: p26.mojo [--multi-stage | --double-buffer]")
