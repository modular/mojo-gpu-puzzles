from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.host import DeviceContext
from gpu.warp import shuffle_down, broadcast, WARP_SIZE
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_equal, assert_almost_equal


alias SIZE = WARP_SIZE
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (WARP_SIZE, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# ANCHOR: neighbor_difference_solution
fn neighbor_difference[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Compute finite differences: output[i] = input[i+1] - input[i]
    Uses shuffle_down(val, 1) to get the next neighbor's value.
    Works across multiple blocks, each processing one warp worth of data.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        # Get current value
        current_val = input[global_i]

        # Get next neighbor's value using shuffle_down
        next_val = shuffle_down(current_val, 1)

        # Compute difference - valid within warp boundaries
        # Last lane of each warp has no valid neighbor within the warp
        # Note there's only one warp in this test, so we don't need to check global_i < size - 1
        # We'll see how this works with multiple blocks in the next tests
        if lane < WARP_SIZE - 1:
            output[global_i] = next_val - current_val
        else:
            # Last thread in warp or last thread overall, set to 0
            output[global_i] = 0


# ANCHOR_END: neighbor_difference_solution

# Advanced setup for multi-block patterns
alias SIZE_2 = 64
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)
alias layout_2 = Layout.row_major(SIZE_2)


# ANCHOR: moving_average_3_solution
fn moving_average_3[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Compute 3-point moving average: output[i] = (input[i] + input[i+1] + input[i+2]) / 3
    Uses shuffle_down with offsets 1 and 2 to access neighbors.
    Works within warp boundaries across multiple blocks.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        # Get current, next, and next+1 values
        current_val = input[global_i]
        next_val = shuffle_down(current_val, 1)
        next_next_val = shuffle_down(current_val, 2)

        # Compute 3-point average - valid within warp boundaries
        if lane < WARP_SIZE - 2 and global_i < size - 2:
            output[global_i] = (current_val + next_val + next_next_val) / 3.0
        elif lane < WARP_SIZE - 1 and global_i < size - 1:
            # Second-to-last in warp: only current + next available
            output[global_i] = (current_val + next_val) / 2.0
        else:
            # Last thread in warp or boundary cases: only current available
            output[global_i] = current_val


# ANCHOR_END: moving_average_3_solution


# ANCHOR: broadcast_shuffle_coordination_solution
fn broadcast_shuffle_coordination[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Combine broadcast() and shuffle_down() for advanced warp coordination.
    Lane 0 computes block-local scaling factor, broadcasts it to all lanes in the warp.
    Each lane uses shuffle_down() for neighbor access and applies broadcast factor.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        # Step 1: Lane 0 computes block-local scaling factor
        var scale_factor: output.element_type = 0.0
        if lane == 0:
            # Compute average of first 4 elements in this block's data
            block_start = block_idx.x * block_dim.x
            var sum: output.element_type = 0.0
            for i in range(4):
                if block_start + i < size:
                    sum += input[block_start + i]
            scale_factor = sum / 4.0

        # Step 2: Broadcast scaling factor to all lanes in this warp
        scale_factor = broadcast(scale_factor)

        # Step 3: Each lane gets current and next values
        current_val = input[global_i]
        next_val = shuffle_down(current_val, 1)

        # Step 4: Apply broadcast factor with neighbor coordination
        if lane < WARP_SIZE - 1 and global_i < size - 1:
            # Combine current + next, then scale by broadcast factor
            output[global_i] = (current_val + next_val) * scale_factor
        else:
            # Last lane in warp or last element: only current value, scaled by broadcast factor
            output[global_i] = current_val * scale_factor


# ANCHOR_END: broadcast_shuffle_coordination_solution


# ANCHOR: basic_broadcast_solution
fn basic_broadcast[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Basic broadcast: Lane 0 computes a block-local value, broadcasts it to all lanes.
    Each lane then uses this broadcast value in its own computation.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        # Step 1: Lane 0 computes special value (sum of first 4 elements in this block)
        var broadcast_value: output.element_type = 0.0
        if lane == 0:
            block_start = block_idx.x * block_dim.x
            var sum: output.element_type = 0.0
            for i in range(4):
                if block_start + i < size:
                    sum += input[block_start + i]
            broadcast_value = sum

        # Step 2: Broadcast lane 0's value to all lanes in this warp
        broadcast_value = broadcast(broadcast_value)

        # Step 3: All lanes use broadcast value in their computation
        output[global_i] = broadcast_value + input[global_i]


# ANCHOR_END: basic_broadcast_solution


# ANCHOR: conditional_broadcast_solution
fn conditional_broadcast[
    layout: Layout, size: Int
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    """
    Conditional broadcast: Lane 0 makes a decision based on block-local data, broadcasts it to all lanes.
    All lanes apply different logic based on the broadcast decision.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    lane = lane_id()

    if global_i < size:
        # Step 1: Lane 0 analyzes block-local data and makes decision (find max of first 8 in block)
        var decision_value: output.element_type = 0.0
        if lane == 0:
            block_start = block_idx.x * block_dim.x
            decision_value = input[block_start] if block_start < size else 0.0
            for i in range(1, min(8, min(WARP_SIZE, size - block_start))):
                if block_start + i < size:
                    current_val = input[block_start + i]
                    if current_val > decision_value:
                        decision_value = current_val

        # Step 2: Broadcast decision to all lanes in this warp
        decision_value = broadcast(decision_value)

        # Step 3: All lanes apply conditional logic based on broadcast decision
        current_input = input[global_i]
        threshold = decision_value / 2.0
        if current_input >= threshold:
            output[global_i] = current_input * 2.0  # Double if >= threshold
        else:
            output[global_i] = current_input / 2.0  # Halve if < threshold


# ANCHOR_END: conditional_broadcast_solution


def test_neighbor_difference():
    with DeviceContext() as ctx:
        # Create test data: [0, 1, 4, 9, 16, 25, ...] (squares)
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i * i

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = neighbor_difference[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results: differences of squares should be odd numbers
        for i in range(SIZE - 1):
            expected_buf[i] = (i + 1) * (i + 1) - i * i
        expected_buf[
            SIZE - 1
        ] = 0  # Last element should be 0 (no valid neighbor)

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            for i in range(SIZE):
                assert_equal(output_host[i], expected_buf[i])

    print("✅ Basic neighbor difference test passed!")


def test_moving_average():
    with DeviceContext() as ctx:
        # Create test data: [1, 2, 4, 7, 11, 16, 22, 29, ...]
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            input_host[0] = 1
            for i in range(1, SIZE_2):
                input_host[i] = input_host[i - 1] + i + 1

        input_tensor = LayoutTensor[dtype, layout_2, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout_2, MutAnyOrigin](output_buf)

        alias kernel = moving_average_3[layout_2, SIZE_2]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID_2,
            block_dim=THREADS_PER_BLOCK_2,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE_2)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            for block in range(BLOCKS_PER_GRID_2[0]):
                warp_start = block * WARP_SIZE
                warp_end = min(warp_start + WARP_SIZE, SIZE_2)

                for i in range(warp_start, warp_end):
                    lane = i % WARP_SIZE
                    if lane < WARP_SIZE - 2 and i < SIZE_2 - 2:
                        # 3-point average within warp
                        expected_buf[i] = (
                            input_host[i]
                            + input_host[i + 1]
                            + input_host[i + 2]
                        ) / 3.0
                    elif lane < WARP_SIZE - 1 and i < SIZE_2 - 1:
                        # 2-point average
                        expected_buf[i] = (
                            input_host[i] + input_host[i + 1]
                        ) / 2.0
                    else:
                        # Single value
                        expected_buf[i] = input_host[i]

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            # Verify results
            for i in range(SIZE_2):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-5)

    print("✅ Moving average test passed!")


def test_broadcast_shuffle_coordination():
    with DeviceContext() as ctx:
        # Create test data: [2, 4, 6, 8, 1, 3, 5, 7, ...]
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            # Create pattern: [2, 4, 6, 8, 1, 3, 5, 7, ...]
            for i in range(SIZE):
                if i < 4:
                    input_host[i] = (i + 1) * 2
                else:
                    input_host[i] = ((i - 4) % 4) * 2 + 1

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = broadcast_shuffle_coordination[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 computes scale_factor from first 4 elements in block: (2+4+6+8)/4 = 5.0
            expected_scale = Float32(5.0)

            for i in range(SIZE):
                if i < SIZE - 1:
                    expected_buf[i] = (
                        input_host[i] + input_host[i + 1]
                    ) * expected_scale
                else:
                    expected_buf[i] = input_host[i] * expected_scale

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            # Verify results
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-4)

    print("✅ Broadcast + Shuffle coordination test passed!")


def test_basic_broadcast():
    with DeviceContext() as ctx:
        # Create test data: [1, 2, 3, 4, 5, 6, 7, 8, ...]
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = i + 1

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = basic_broadcast[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 computes broadcast_value from first 4 elements: 1+2+3+4 = 10
            expected_broadcast = Float32(10.0)
            for i in range(SIZE):
                expected_buf[i] = expected_broadcast + input_host[i]

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            # Verify results
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-4)

    print("✅ Basic broadcast test passed!")


def test_conditional_broadcast():
    with DeviceContext() as ctx:
        # Create test data: [3, 1, 7, 2, 9, 4, 6, 8, ...]
        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            # Create pattern with known max
            test_values = [
                Float32(3.0),
                Float32(1.0),
                Float32(7.0),
                Float32(2.0),
                Float32(9.0),
                Float32(4.0),
                Float32(6.0),
                Float32(8.0),
            ]
            for i in range(SIZE):
                input_host[i] = test_values[i % len(test_values)]

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        alias kernel = conditional_broadcast[layout, SIZE]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)

        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 finds max of first 8 elements in block: max(3,1,7,2,9,4,6,8) = 9.0, threshold = 4.5
            expected_max = Float32(9.0)
            threshold = expected_max / 2.0
            for i in range(SIZE):
                if input_host[i] >= threshold:
                    expected_buf[i] = input_host[i] * 2.0
                else:
                    expected_buf[i] = input_host[i] / 2.0

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            # Verify results
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-4)

    print("✅ Conditional broadcast test passed!")


def main():
    print("WARP_SIZE: ", WARP_SIZE)
    if len(argv()) != 2:
        print(
            "Usage: p23.mojo"
            " [--neighbor|--average|--broadcast-basic|--broadcast-conditional|--broadcast-shuffle-coordination]"
        )
        return

    test_type = argv()[1]
    if test_type == "--neighbor":
        print("SIZE: ", SIZE)
        test_neighbor_difference()
    elif test_type == "--average":
        print("SIZE_2: ", SIZE_2)
        test_moving_average()
    elif test_type == "--broadcast-basic":
        print("SIZE: ", SIZE)
        test_basic_broadcast()
    elif test_type == "--broadcast-conditional":
        print("SIZE: ", SIZE)
        test_conditional_broadcast()
    elif test_type == "--broadcast-shuffle-coordination":
        print("SIZE: ", SIZE)
        test_broadcast_shuffle_coordination()
    else:
        print(
            "Usage: p23.mojo"
            " [--neighbor|--average|--broadcast-basic|--broadcast-conditional|--broadcast-shuffle-coordination]"
        )
