# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# Puzzle 09 solution — three debugging cases, all fixed:
#
#   Case 1 (add_10):             input buffer was created with size 0 while
#                                the kernel reads SIZE elements -> out-of-bounds
#                                read (illegal address). FIX: allocate SIZE.
#
#   Case 2 (process_sliding_window): ITER was 2 but the window has 3 positions
#                                (left, center, right). Thread 0 missed its
#                                right neighbour -> wrong results, no crash.
#                                FIX: ITER = 3 (offsets -1, 0, +1, bounds-guarded).
#
#   Case 3 (collaborative_filter): barrier() sat inside `if thread_id <
#                                SIZE-1` (Phase 2), so only 3 of 4 threads
#                                reached it -> barrier divergence -> deadlock.
#                                FIX: read the left neighbour's Phase-1 value
#                                (no race), then barrier() with all threads,
#                                then the conditional update, then barrier().
#
from std.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
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
# FIX (case 2): window has 3 positions (left, center, right)
comptime ITER = 3


# ANCHOR: first_crash
def add_10(
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = thread_idx.x
    output[unsafe_offset=i] = a[unsafe_offset=i] + 10.0


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
            var value = a[idx]
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
        shared_workspace[thread_id] = a[thread_id]
    barrier()

    # FIX (case 3): read the left neighbour's Phase-1 value BEFORE any
    # conditional barrier — reading shared[thread_id - 1] here is race-free
    # because Phase-1 writes are complete and no thread writes its own slot
    # until the barrier below.
    var left_neighbour = Scalar[dtype](0.0)
    if thread_id > 0:
        left_neighbour = shared_workspace[thread_id - 1]
    barrier()

    # Phase 2: Collaborative processing (all threads reached the barrier)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] += left_neighbour * 0.5
    barrier()

    # Phase 3: Final synchronization and output
    barrier()

    # Write filtered results back to output
    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = a[thread_id]


# ANCHOR_END: third_crash


def main() raises:
    if len(argv()) != 2:
        print(
            "Usage: mojo p09 [--first-case | --second-case | --third-case |"
            " --all]"
        )
        return

    if argv()[1] == "--first-case":
        print(
            "First Case: Try to identify what's wrong without looking at the"
            " code!"
        )
        print()

        with DeviceContext() as ctx:
            # FIX (case 1): input buffer must be SIZE elements, not 0
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var result_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            result_buf.enqueue_fill(0)

            # Enqueue function
            ctx.enqueue_function[add_10](
                result_buf,
                input_buf,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with result_buf.map_to_host() as result_host:
                print("result:", result_host)
                for i in range(SIZE):
                    assert_equal(result_host[i], Scalar[dtype](10.0))
                print("add_10: correct (no out-of-bounds read) ✅")

    elif argv()[1] == "--second-case":
        print("This program computes sliding window sums for each position...")
        print()

        with DeviceContext() as ctx:
            # Create buffers
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)

            # Initialize input [0, 1, 2, 3]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i)

            # Create TileTensors for structured access
            var input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            var output_tensor = TileTensor(output_buf, vector_layout)

            print("Input array: [0, 1, 2, 3]")
            print("Computing sliding window sums (window size = 3)...")
            print(
                "Each position should sum its neighbors: [left + center +"
                " right]"
            )

            ctx.enqueue_function[process_sliding_window](
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
                var matches = True
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
        print()

        with DeviceContext() as ctx:
            # Create input and output buffers
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)

            # Initialize input data [1, 2, 3, 4]
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i + 1)

            # Create TileTensors
            var input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            var output_tensor = TileTensor(output_buf, vector_layout)

            print("Input array: [1, 2, 3, 4]")
            print("Applying collaborative filter using shared memory...")
            print("Each thread cooperates with neighbors for smoothing...")

            # FIX (case 3): no longer hangs — barriers are reached by all
            # threads
            ctx.enqueue_function[collaborative_filter](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            print("Waiting for GPU computation to complete...")
            ctx.synchronize()

            with output_buf.map_to_host() as output_host:
                print("Result:", output_host)
                # Expected: y[i] = a[i] + 0.5*a[i-1] for i<3, y[3] = a[3]
                # = [1, 2.5, 4.0, 4.0] (thread 2 reads thread 1's PHASE-1
                # value 2, so 3 + 1.0 = 4.0)
                assert_equal(output_host[0], Scalar[dtype](1.0))
                assert_equal(output_host[1], Scalar[dtype](2.5))
                assert_equal(output_host[2], Scalar[dtype](4.0))
                assert_equal(output_host[3], Scalar[dtype](4.0))
                print(
                    "[SUCCESS] Collaborative filtering completed without"
                    " deadlock ✅"
                )

    elif argv()[1] == "--all":
        print("Puzzle 09 — all three cases:")
        # Reuse the three case bodies by invoking the fixed kernels directly.
        # Case 1: add_10 with a properly sized buffer
        with DeviceContext() as ctx:
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var result_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            result_buf.enqueue_fill(0)
            ctx.enqueue_function[add_10](
                result_buf,
                input_buf,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
            ctx.synchronize()
            with result_buf.map_to_host() as result_host:
                for i in range(SIZE):
                    assert_equal(result_host[i], Scalar[dtype](10.0))
            print("  case 1 (OOB read):            fixed ✅")

        # Case 2: sliding window sums
        with DeviceContext() as ctx:
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i)
            var input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            var output_tensor = TileTensor(output_buf, vector_layout)
            ctx.enqueue_function[process_sliding_window](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
            ctx.synchronize()
            with output_buf.map_to_host() as output_host:
                assert_equal(output_host[0], Scalar[dtype](1.0))
                assert_equal(output_host[1], Scalar[dtype](3.0))
                assert_equal(output_host[2], Scalar[dtype](6.0))
                assert_equal(output_host[3], Scalar[dtype](5.0))
            print("  case 2 (window logic bug):    fixed ✅")

        # Case 3: collaborative filter (no deadlock)
        with DeviceContext() as ctx:
            var input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            output_buf.enqueue_fill(0)
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Scalar[dtype](i + 1)
            var input_tensor = TileTensor[mut=False, dtype, VectorLayout](
                input_buf, vector_layout
            )
            var output_tensor = TileTensor(output_buf, vector_layout)
            ctx.enqueue_function[collaborative_filter](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
            ctx.synchronize()
            with output_buf.map_to_host() as output_host:
                # y[i] = a[i] + 0.5*a[i-1], y[3] = a[3]
                assert_equal(output_host[0], Scalar[dtype](1.0))
                assert_equal(output_host[1], Scalar[dtype](2.5))
                assert_equal(output_host[2], Scalar[dtype](4.0))
                assert_equal(output_host[3], Scalar[dtype](4.0))
            print("  case 3 (barrier deadlock):    fixed ✅")

        print("Puzzle 09 complete ✅")

    else:
        print(
            "Unsupported option. Choose between [--first-case, --second-case,"
            " --third-case, --all]"
        )
