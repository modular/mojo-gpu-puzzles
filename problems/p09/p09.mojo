from std.memory import UnsafePointer
from std.gpu import thread_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal
from std.sys import argv

comptime SIZE = 4
comptime MATRIX_SIZE = 3
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32
comptime vector_layout = row_major[SIZE]()
comptime VectorLayout = type_of(vector_layout)
comptime ITER = 2


# ANCHOR: first_crash
def add_10(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var i = thread_idx.x
    output[i] = a[i] + 10.0


# ANCHOR_END: first_crash


# ANCHOR: second_crash
def process_sliding_window(
    output: TileTensor[mut=True, dtype, VectorLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, VectorLayout, ImmutAnyOrigin],
):
    var thread_id = thread_idx.x

    # Each thread processes a sliding window of 3 elements
    var window_sum = Scalar[dtype](0.0)

    # Sum elements in sliding window: [i-1, i, i+1]
    for offset in range(ITER):
        var idx = Int(thread_id) + offset - 1
        if 0 <= idx < SIZE:
            var value = rebind[Scalar[dtype]](a[idx])
            window_sum += value

    output[thread_id] = window_sum


# ANCHOR_END: second_crash


# ANCHOR: third_crash
def collaborative_filter(
    output: TileTensor[mut=True, dtype, VectorLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, VectorLayout, ImmutAnyOrigin],
):
    var thread_id = thread_idx.x

    # Shared memory workspace for collaborative processing
    var shared_workspace = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[SIZE - 1]())

    # Phase 1: Initialize shared workspace (all threads participate)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
    barrier()

    # Phase 2: Collaborative processing
    if thread_id < SIZE - 1:
        # Apply collaborative filter with neighbors
        if thread_id > 0:
            shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
        barrier()

    # Phase 3: Final synchronization and output
    barrier()

    # Write filtered results back to output
    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = rebind[Scalar[dtype]](a[thread_id])


# ANCHOR_END: third_crash


def main() raises:
    if len(argv()) != 2:
        print(
            "Usage: pixi run mojo p09 [--first-case | --second-case |"
            " --third-case]"
        )
        return

    if argv()[1] == "--first-case":
        print(
            "First Case: Try to identify what's wrong without looking at the"
            " code!"
        )
        print()

        with DeviceContext() as ctx:
            input_buf = ctx.enqueue_create_buffer[dtype](0)
            var result_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            result_buf.enqueue_fill(0)

            # Enqueue function
            ctx.enqueue_function[add_10, add_10](
                result_buf,
                input_buf,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with result_buf.map_to_host() as result_host:
                print("result:", result_host)

    elif argv()[1] == "--second-case":
        print("This program computes sliding window sums for each position...")
        print()

        with DeviceContext() as ctx:
            # Create buffers
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)

            # Initialize input [0, 1, 2, 3]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i)

            # Create TileTensors for structured access
            input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            output_tensor = TileTensor(output_buf, vector_layout)

            print("Input array: [0, 1, 2, 3]")
            print("Computing sliding window sums (window size = 3)...")
            print(
                "Each position should sum its neighbors: [left + center +"
                " right]"
            )

            ctx.enqueue_function[
                process_sliding_window, process_sliding_window
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with output_buf.map_to_host() as output_host:
                print("Actual result:", output_host)

                # Expected sliding window results
                var expected_0 = Scalar[dtype](1.0)
                var expected_1 = Scalar[dtype](3.0)
                var expected_2 = Scalar[dtype](6.0)
                var expected_3 = Scalar[dtype](5.0)
                print("Expected: [1.0, 3.0, 6.0, 5.0]")

                # Check if results match expected pattern
                matches = True
                if abs(output_host[0] - expected_0) > 0.001:
                    matches = False
                if abs(output_host[1] - expected_1) > 0.001:
                    matches = False
                if abs(output_host[2] - expected_2) > 0.001:
                    matches = False
                if abs(output_host[3] - expected_3) > 0.001:
                    matches = False

                if matches:
                    print(
                        "[PASS] Test PASSED - Sliding window sums are correct"
                    )
                else:
                    print(
                        "[FAIL] Test FAILED - Sliding window sums are"
                        " incorrect!"
                    )
                    print("Check the window indexing logic...")

    elif argv()[1] == "--third-case":
        print(
            "Third Case: Advanced collaborative filtering with shared memory..."
        )
        print("WARNING: This may hang - use Ctrl+C to stop if needed")
        print()

        with DeviceContext() as ctx:
            # Create input and output buffers
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)

            # Initialize input data [1, 2, 3, 4]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i + 1)

            # Create TileTensors
            input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            output_tensor = TileTensor(output_buf, vector_layout)

            print("Input array: [1, 2, 3, 4]")
            print("Applying collaborative filter using shared memory...")
            print("Each thread cooperates with neighbors for smoothing...")

            # This will likely hang due to barrier deadlock
            ctx.enqueue_function[collaborative_filter, collaborative_filter](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            print("Waiting for GPU computation to complete...")
            ctx.synchronize()

            with output_buf.map_to_host() as output_host:
                print("Result:", output_host)
                print(
                    "[SUCCESS] Collaborative filtering completed successfully!"
                )

    else:
        print(
            "Unsupported option. Choose between [--first-case, --second-case,"
            " --third-case]"
        )
