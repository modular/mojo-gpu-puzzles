from std.gpu import thread_idx, block_idx, block_dim, lane_id
from std.gpu.host import DeviceContext
from std.gpu.primitives.warp import shuffle_down, broadcast, WARP_SIZE
from layout import TileTensor
from layout.tile_layout import row_major
from std.sys import argv
from std.testing import assert_equal, assert_almost_equal

# ANCHOR: neighbor_difference
comptime SIZE = WARP_SIZE
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def neighbor_difference[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    """
    Compute finite differences: output[i] = input[i+1] - input[i]
    Uses shuffle_down(val, 1) to get the next neighbor's value.
    Works across multiple blocks, each processing one warp worth of data.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var lane = Int(lane_id())

    # FILL IN (roughly 7 lines)


# ANCHOR_END: neighbor_difference

# ANCHOR: moving_average_3
comptime SIZE_2 = 64
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)
comptime layout_2 = row_major[SIZE_2]()
comptime LayoutType_2 = type_of(layout_2)


def moving_average_3[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType_2, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType_2, ImmutAnyOrigin],
):
    """
    Compute 3-point moving average: output[i] = (input[i] + input[i+1] + input[i+2]) / 3
    Uses shuffle_down with offsets 1 and 2 to access neighbors.
    Works within warp boundaries across multiple blocks.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var lane = Int(lane_id())

    # FILL IN (roughly 10 lines)


# ANCHOR_END: moving_average_3


# ANCHOR: broadcast_shuffle_coordination
def broadcast_shuffle_coordination[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    """
    Combine broadcast() and shuffle_down() for advanced warp coordination.
    Lane 0 computes block-local scaling factor, broadcasts it to all lanes in the warp.
    Each lane uses shuffle_down() for neighbor access and applies broadcast factor.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var lane = Int(lane_id())
    if global_i < size:
        var scale_factor: output.ElementType = 0.0

        # FILL IN (roughly 14 lines)


# ANCHOR_END: broadcast_shuffle_coordination


# ANCHOR: basic_broadcast
def basic_broadcast[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    """
    Basic broadcast: Lane 0 computes a block-local value, broadcasts it to all lanes.
    Each lane then uses this broadcast value in its own computation.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var lane = Int(lane_id())
    if global_i < size:
        var broadcast_value: output.ElementType = 0.0

        # FILL IN (roughly 10 lines)


# ANCHOR_END: basic_broadcast


# ANCHOR: conditional_broadcast
def conditional_broadcast[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    """
    Conditional broadcast: Lane 0 makes a decision based on block-local data, broadcasts it to all lanes.
    All lanes apply different logic based on the broadcast decision.
    """
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var lane = Int(lane_id())
    if global_i < size:
        var decision_value: output.ElementType = 0.0

        # FILL IN (roughly 10 lines)

        var current_input = input[global_i]
        var threshold = decision_value / 2.0
        if current_input >= threshold:
            output[global_i] = current_input * 2.0  # Double if >= threshold
        else:
            output[global_i] = current_input / 2.0  # Halve if < threshold


# ANCHOR_END: conditional_broadcast


def test_neighbor_difference() raises:
    with DeviceContext() as ctx:
        # Create test data: [0, 1, 4, 9, 16, 25, ...] (squares)
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Scalar[dtype](i * i)

        var input_tensor = TileTensor[
            mut=False, dtype, LayoutType, ImmutAnyOrigin
        ](input_buf, layout)
        var output_tensor = TileTensor(output_buf, layout)

        comptime kernel = neighbor_difference[SIZE]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results: differences of squares should be odd numbers
        for i in range(SIZE - 1):
            expected_buf[i] = Scalar[dtype]((i + 1) * (i + 1) - i * i)
        expected_buf[
            SIZE - 1
        ] = 0  # Last element should be 0 (no valid neighbor)

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)
            for i in range(SIZE):
                assert_equal(output_host[i], expected_buf[i])

    print("Neighbor difference test: passed")


def test_moving_average() raises:
    with DeviceContext() as ctx:
        # Create test data: [1, 2, 4, 7, 11, 16, 22, 29, ...]
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        input_buf.enqueue_fill(0)
        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE_2)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            input_host[0] = 1
            for i in range(1, SIZE_2):
                input_host[i] = input_host[i - 1] + Scalar[dtype](i + 1)

        var input_tensor = TileTensor[
            mut=False, dtype, LayoutType_2, ImmutAnyOrigin
        ](input_buf, layout_2)
        var output_tensor = TileTensor(output_buf, layout_2)

        comptime kernel = moving_average_3[SIZE_2]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID_2,
            block_dim=THREADS_PER_BLOCK_2,
        )

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE_2)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            for block in range(BLOCKS_PER_GRID_2[0]):
                var warp_start = block * WARP_SIZE
                var warp_end = min(warp_start + WARP_SIZE, SIZE_2)

                for i in range(warp_start, warp_end):
                    var lane = i % WARP_SIZE
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

    print("Moving average test: passed")


def test_broadcast_shuffle_coordination() raises:
    with DeviceContext() as ctx:
        # Create test data: [2, 4, 6, 8, 1, 3, 5, 7, ...]
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            # Create pattern: [2, 4, 6, 8, 1, 3, 5, 7, ...]
            for i in range(SIZE):
                if i < 4:
                    input_host[i] = Scalar[dtype]((i + 1) * 2)
                else:
                    input_host[i] = Scalar[dtype](((i - 4) % 4) * 2 + 1)

        var input_tensor = TileTensor[
            mut=False, dtype, LayoutType, ImmutAnyOrigin
        ](input_buf, layout)
        var output_tensor = TileTensor(output_buf, layout)

        comptime kernel = broadcast_shuffle_coordination[SIZE]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 computes scale_factor from first 4 elements in block: (2+4+6+8)/4 = 5.0
            var expected_scale = Scalar[dtype](5.0)

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

    print("Broadcast + shuffle coordination test: passed")


def test_basic_broadcast() raises:
    with DeviceContext() as ctx:
        # Create test data: [1, 2, 3, 4, 5, 6, 7, 8, ...]
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = Scalar[dtype](i + 1)

        var input_tensor = TileTensor[
            mut=False, dtype, LayoutType, ImmutAnyOrigin
        ](input_buf, layout)
        var output_tensor = TileTensor(output_buf, layout)

        comptime kernel = basic_broadcast[SIZE]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 computes broadcast_value from first 4 elements: 1+2+3+4 = 10
            var expected_broadcast = Scalar[dtype](10.0)
            for i in range(SIZE):
                expected_buf[i] = expected_broadcast + input_host[i]

        with output_buf.map_to_host() as output_host:
            print("output:", output_host)
            print("expected:", expected_buf)

            # Verify results
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_buf[i], rtol=1e-4)

    print("Basic broadcast test: passed")


def test_conditional_broadcast() raises:
    with DeviceContext() as ctx:
        # Create test data: [3, 1, 7, 2, 9, 4, 6, 8, ...]
        var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        input_buf.enqueue_fill(0)
        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            # Create pattern with known max
            var test_values = [
                Scalar[dtype](3.0),
                Scalar[dtype](1.0),
                Scalar[dtype](7.0),
                Scalar[dtype](2.0),
                Scalar[dtype](9.0),
                Scalar[dtype](4.0),
                Scalar[dtype](6.0),
                Scalar[dtype](8.0),
            ]
            for i in range(SIZE):
                input_host[i] = test_values[i % len(test_values)]

        var input_tensor = TileTensor[
            mut=False, dtype, LayoutType, ImmutAnyOrigin
        ](input_buf, layout)
        var output_tensor = TileTensor(output_buf, layout)

        comptime kernel = conditional_broadcast[SIZE]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected_buf = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_buf.enqueue_fill(0)
        ctx.synchronize()

        # Create expected results
        with input_buf.map_to_host() as input_host:
            # Lane 0 finds max of first 8 elements in block: max(3,1,7,2,9,4,6,8) = 9.0, threshold = 4.5
            var expected_max = Scalar[dtype](9.0)
            var threshold = expected_max / 2.0
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

    print("Conditional broadcast test: passed")


def main() raises:
    print("WARP_SIZE: ", WARP_SIZE)
    if len(argv()) < 1 or len(argv()) > 2:
        print(
            "Usage: p23.mojo"
            " [--neighbor|--average|--broadcast-basic|--broadcast-conditional|--broadcast-shuffle-coordination]"
        )
        return

    var test_type = argv()[1]
    if test_type == "--neighbor":
        print("SIZE: ", SIZE)
        test_neighbor_difference()
        print("Puzzle 25 complete ✅")
    elif test_type == "--average":
        print("SIZE_2: ", SIZE_2)
        test_moving_average()
        print("Puzzle 25 complete ✅")
    elif test_type == "--broadcast-basic":
        print("SIZE: ", SIZE)
        test_basic_broadcast()
        print("Puzzle 25 complete ✅")
    elif test_type == "--broadcast-conditional":
        print("SIZE: ", SIZE)
        test_conditional_broadcast()
        print("Puzzle 25 complete ✅")
    elif test_type == "--broadcast-shuffle-coordination":
        print("SIZE: ", SIZE)
        test_broadcast_shuffle_coordination()
        print("Puzzle 25 complete ✅")
    else:
        print(
            "Usage: p23.mojo"
            " [--neighbor|--average|--broadcast-basic|--broadcast-conditional|--broadcast-shuffle-coordination]"
        )
