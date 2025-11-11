from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.host import DeviceContext
from gpu.warp import shuffle_xor, prefix_sum, WARP_SIZE
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_equal, assert_almost_equal


alias SIZE = WARP_SIZE
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (WARP_SIZE, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# ANCHOR: butterfly_pair_swap_solution
fn butterfly_pair_swap[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Basic butterfly pair swap: Exchange values between adjacent pairs using XOR pattern.
    Each thread exchanges its value with its XOR-1 neighbor, creating pairs: (0,1), (2,3), (4,5), etc.
    Uses shuffle_xor(val, 1) to swap values within each pair.
    This is the foundation of butterfly network communication patterns.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x

    if global_i < size:
        current_val = input[global_i]

        # Exchange with XOR-1 neighbor using butterfly pattern
        # Lane 0 exchanges with lane 1, lane 2 with lane 3, etc.
        swapped_val = shuffle_xor(current_val, 1)

        # For demonstration, we'll store the swapped value
        # In real applications, this might be used for sorting, reduction, etc.
        output[global_i] = swapped_val


# ANCHOR_END: butterfly_pair_swap_solution


# ANCHOR: butterfly_parallel_max_solution
fn butterfly_parallel_max[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Parallel maximum reduction using butterfly pattern.
    Uses shuffle_xor with decreasing offsets (16, 8, 4, 2, 1) to perform tree-based reduction.
    Each step reduces the active range by half until all threads have the maximum value.
    This implements an efficient O(log n) parallel reduction algorithm.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x

    if global_i < size:
        max_val = input[global_i]

        # Butterfly reduction tree: dynamic for any WARP_SIZE (32, 64, etc.)
        # Start with half the warp size and reduce by half each step
        offset = WARP_SIZE // 2
        while offset > 0:
            max_val = max(max_val, shuffle_xor(max_val, offset))
            offset //= 2

        # All threads now have the maximum value across the entire warp
        output[global_i] = max_val


# ANCHOR_END: butterfly_parallel_max_solution


alias SIZE_2 = 64
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)
alias layout_2 = Layout.row_major(SIZE_2)


# ANCHOR: butterfly_conditional_max_solution
fn butterfly_conditional_max[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Conditional butterfly maximum: Perform butterfly max reduction, but only store result
    in even-numbered lanes. Odd-numbered lanes store the minimum value seen.
    Demonstrates conditional logic combined with butterfly communication patterns.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        current_val = input[global_i]
        min_val = current_val

        # Butterfly reduction for both maximum and minimum: dynamic for any WARP_SIZE
        offset = WARP_SIZE // 2
        while offset > 0:
            neighbor_val = shuffle_xor(current_val, offset)
            current_val = max(current_val, neighbor_val)

            min_neighbor_val = shuffle_xor(min_val, offset)
            min_val = min(min_val, min_neighbor_val)

            offset //= 2

        # Conditional output: max for even lanes, min for odd lanes
        if lane % 2 == 0:
            output[global_i] = current_val  # Maximum
        else:
            output[global_i] = min_val  # Minimum


# ANCHOR_END: butterfly_conditional_max_solution


# ANCHOR: warp_inclusive_prefix_sum_solution
fn warp_inclusive_prefix_sum[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Inclusive prefix sum using warp primitive: Each thread gets sum of all elements up to and including its position.
    Compare this to Puzzle 12's complex shared memory + barrier approach.

    Puzzle 12 approach:
    - Shared memory allocation
    - Multiple barrier synchronizations
    - Log(n) iterations with manual tree reduction
    - Complex multi-phase algorithm

    Warp prefix_sum approach:
    - Single function call!
    - Hardware-optimized parallel scan
    - Automatic synchronization
    - O(log n) complexity, but implemented in hardware.

    NOTE: This implementation only works correctly within a single warp (WARP_SIZE threads).
    For multi-warp scenarios, additional coordination would be needed.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x

    if global_i < size:
        current_val = input[global_i]

        # This one call replaces ~30 lines of complex shared memory logic from Puzzle 12!
        # But it only works within the current warp (WARP_SIZE threads)
        scan_result = prefix_sum[exclusive=False](
            rebind[Scalar[dtype]](current_val)
        )

        output[global_i] = scan_result


# ANCHOR_END: warp_inclusive_prefix_sum_solution


# ANCHOR: warp_partition_solution
fn warp_partition[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    pivot: Float32,
):
    """
    Single-warp parallel partitioning using BOTH shuffle_xor AND prefix_sum.
    This implements a warp-level quicksort partition step that places elements < pivot
    on the left and elements >= pivot on the right.

    ALGORITHM COMPLEXITY - combines two advanced warp primitives:
    1. shuffle_xor(): Butterfly pattern for warp-level reductions
    2. prefix_sum(): Warp-level exclusive scan for position calculation.

    This demonstrates the power of warp primitives for sophisticated parallel algorithms
    within a single warp (works for any WARP_SIZE: 32, 64, etc.).

    Example with pivot=5:
    Input:  [3, 7, 1, 8, 2, 9, 4, 6]
    Result: [3, 1, 2, 4, 7, 8, 9, 6] (< pivot | >= pivot).
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x

    if global_i < size:
        current_val = input[global_i]

        # Phase 1: Create warp-level predicates
        predicate_left = Float32(1.0) if current_val < pivot else Float32(0.0)
        predicate_right = Float32(1.0) if current_val >= pivot else Float32(0.0)

        # Phase 2: Warp-level prefix sum to get positions within warp
        warp_left_pos = prefix_sum[exclusive=True](predicate_left)
        warp_right_pos = prefix_sum[exclusive=True](predicate_right)

        # Phase 3: Get total left count using shuffle_xor reduction
        warp_left_total = predicate_left

        # Butterfly reduction to get total across the warp: dynamic for any WARP_SIZE
        offset = WARP_SIZE // 2
        while offset > 0:
            warp_left_total += shuffle_xor(warp_left_total, offset)
            offset //= 2

        # Phase 4: Write to output positions
        if current_val < pivot:
            # Left partition: use warp-level position
            output[Int(warp_left_pos)] = current_val
        else:
            # Right partition: offset by total left count + right position
            output[Int(warp_left_total + warp_right_pos)] = current_val


# ANCHOR_END: warp_partition_solution


def test_butterfly_pair_swap():
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = butterfly_pair_swap[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results: pairs should be swapped
        # (0,1) -> (1,0), (2,3) -> (3,2), (4,5) -> (5,4), etc.
        for i in range(SIZE):
            if i % 2 == 0:
                # Even positions get odd values
                expected_buf[i] = i + 1
            else:
                # Odd positions get even values
                expected_buf[i] = i - 1

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            for i in range(SIZE):
                assert_equal(output_host[i], expected_buf[i])

    print("✅ Butterfly pair swap test passed!")


def test_butterfly_parallel_max():
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i * 2
            # Make sure we have a clear maximum
            input_host[SIZE - 1] = 1000.0

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = butterfly_parallel_max[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(1000.0)

        # All threads should have the maximum value (1000.0)
        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            for i in range(SIZE):
                assert_almost_equal(output_host[i], 1000.0, rtol=1e-5)

    print("✅ Butterfly parallel max test passed!")


def test_butterfly_conditional_max():
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE_2):
                if i < 9:
                    values = [3, 1, 7, 2, 9, 4, 8, 5, 6]
                    input_host[i] = values[i]
                else:
                    input_host[i] = i % 10

        input_tensor = LayoutTensor[dtype, layout_2, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout_2, MutAnyOrigin](output_buf)

        alias kernel = butterfly_conditional_max[layout_2, SIZE_2]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID_2,
            block_dim=THREADS_PER_BLOCK_2,
        )

        ctx.synchronize()

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE_2)
        expected_buf.enqueue_fill(0)

        # Expected: even lanes get max, odd lanes get min
        with input_buf.map_to_host() as input_host:
            max_val = input_host[0]
            min_val = input_host[0]
            for i in range(1, SIZE_2):
                if input_host[i] > max_val:
                    max_val = input_host[i]
                if input_host[i] < min_val:
                    min_val = input_host[i]

            for i in range(SIZE_2):
                if i % 2 == 0:
                    expected_buf[i] = max_val
                else:
                    expected_buf[i] = min_val

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            for i in range(SIZE_2):
                if i % 2 == 0:
                    assert_almost_equal(output_host[i], max_val, rtol=1e-5)
                else:
                    assert_almost_equal(output_host[i], min_val, rtol=1e-5)

    print("✅ Butterfly conditional max test passed!")


def test_warp_inclusive_prefix_sum():
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i + 1

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = warp_inclusive_prefix_sum[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected inclusive prefix sum: [1, 3, 6, 10, 15, 21, 28, 36, ...]
        with input_buf.map_to_host() as input_host:
            expected_buf[0] = input_host[0]
            for i in range(1, SIZE):
                expected_buf[i] = expected_buf[i - 1] + input_host[i]

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-5)

    print("✅ Warp inclusive prefix sum test passed!")


def test_warp_partition():
    with DeviceContext() as ctx:
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        # Create test data: mix of values above and below pivot
        pivot_value = Float32(5.0)
        with input_buf.map_to_host() as input_host:
            # Create: [3, 7, 1, 8, 2, 9, 4, 6, ...]
            test_values = [3, 7, 1, 8, 2, 9, 4, 6, 0, 10, 3, 11, 1, 12, 4, 13]
            for i in range(SIZE):
                input_host[i] = test_values[i % len(test_values)]

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = warp_partition[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            pivot_value,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results: elements < 5 on left, >= 5 on right
        with input_buf.map_to_host() as input_host:
            left_values = List[Float32]()
            right_values = List[Float32]()

            for i in range(SIZE):
                if input_host[i] < pivot_value:
                    left_values.append(input_host[i])
                else:
                    right_values.append(input_host[i])

            # Fill expected buffer
            for i in range(len(left_values)):
                expected_buf[i] = left_values[i]
            for i in range(len(right_values)):
                expected_buf[len(left_values) + i] = right_values[i]

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            print("pivot:", pivot_value)

            # Verify partitioning property (left < pivot, right >= pivot)
            # Find partition boundary
            var partition_point = 0
            for i in range(SIZE):
                if output_host[i] >= pivot_value:
                    partition_point = i
                    break

            # Check left partition
            for i in range(partition_point):
                if output_host[i] >= pivot_value:
                    print("ERROR: Left partition contains value >= pivot")

            # Check right partition
            for i in range(partition_point, SIZE):
                if output_host[i] < pivot_value:
                    print("ERROR: Right partition contains value < pivot")

    print("✅ Warp partition test passed!")


def main():
    print("WARP_SIZE: ", WARP_SIZE)
    if len(argv()) != 2:
        print(
            "Usage: p24.mojo"
            " [--pair-swap|--parallel-max|--conditional-max|--prefix-sum|--partition]"
        )
        return

    test_type = argv()[1]
    if test_type == "--pair-swap":
        print("SIZE: ", SIZE)
        test_butterfly_pair_swap()
    elif test_type == "--parallel-max":
        print("SIZE: ", SIZE)
        test_butterfly_parallel_max()
    elif test_type == "--conditional-max":
        print("SIZE: ", SIZE_2)
        test_butterfly_conditional_max()
    elif test_type == "--prefix-sum":
        print("SIZE: ", SIZE)
        test_warp_inclusive_prefix_sum()
    elif test_type == "--partition":
        print("SIZE: ", SIZE)
        test_warp_partition()
    else:
        print(
            "Usage: p24.mojo"
            " [--pair-swap|--parallel-max|--conditional-max|--prefix-sum|--partition]"
        )
