from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_arrive,
    cluster_wait,
    elect_one_sync,
)
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_equal, assert_almost_equal, assert_true

alias SIZE = 1024
alias TPB = 256
alias CLUSTER_SIZE = 4
alias dtype = DType.float32
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)


# ANCHOR: cluster_coordination_basics_solution
fn cluster_coordination_basics[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """Real cluster coordination using SM90+ cluster APIs."""
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Check what's happening with cluster ranks
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    shared_data = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # FIX: Use block_idx.x for data distribution instead of cluster rank
    # Each block should process different portions of the data
    var data_scale = Float32(
        block_id + 1
    )  # Use block_idx instead of cluster rank

    # Phase 1: Each block processes its portion
    if global_i < size:
        shared_data[local_i] = input[global_i] * data_scale
    else:
        shared_data[local_i] = 0.0

    barrier()

    # Phase 2: Use cluster_arrive() for inter-block coordination
    cluster_arrive()  # Signal this block has completed processing

    # Block-level aggregation (only thread 0)
    if local_i == 0:
        var block_sum: Float32 = 0.0
        for i in range(tpb):
            block_sum += shared_data[i][0]
        # FIX: Store result at block_idx position (guaranteed unique per block)
        output[block_id] = block_sum

    # Wait for all blocks in cluster to complete
    cluster_wait()


# ANCHOR_END: cluster_coordination_basics_solution


# ANCHOR: cluster_collective_operations_solution
fn cluster_collective_operations[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    temp_storage: LayoutTensor[
        dtype, Layout.row_major(CLUSTER_SIZE), MutAnyOrigin
    ],
    size: Int,
):
    """Cluster-wide collective operations using real cluster APIs."""
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    # Each thread accumulates its data
    var my_value: Float32 = 0.0
    if global_i < size:
        my_value = input[global_i][0]

    # Block-level reduction using shared memory
    shared_mem = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_mem[local_i] = my_value
    barrier()

    # Tree reduction within block
    var stride = tpb // 2
    while stride > 0:
        if local_i < stride and local_i + stride < tpb:
            shared_mem[local_i] += shared_mem[local_i + stride]
        barrier()
        stride = stride // 2

    # FIX: Store block result using block_idx for reliable indexing
    if local_i == 0:
        temp_storage[block_id] = shared_mem[0]

    # Use cluster_sync() for full cluster synchronization
    cluster_sync()

    # Final cluster reduction (elect one thread to do the final work)
    if elect_one_sync() and my_block_rank == 0:
        var total: Float32 = 0.0
        for i in range(CLUSTER_SIZE):
            total += temp_storage[i][0]
        output[0] = total


# ANCHOR_END: cluster_collective_operations_solution


# ANCHOR: advanced_cluster_patterns_solution
fn advanced_cluster_patterns[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """Advanced cluster programming using cluster masks and relaxed synchronization.
    """
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    shared_data = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Compute cluster mask for advanced coordination
    # base_mask = cluster_mask_base()  # Requires cluster_shape parameter

    # FIX: Process data with block_idx-based scaling for guaranteed uniqueness
    var data_scale = Float32(block_id + 1)
    if global_i < size:
        shared_data[local_i] = input[global_i] * data_scale
    else:
        shared_data[local_i] = 0.0

    barrier()

    # Advanced pattern: Use elect_one_sync for efficient coordination
    if elect_one_sync():  # Only one thread per warp does this work
        var warp_sum: Float32 = 0.0
        var warp_start = (local_i // 32) * 32  # Get warp start index
        for i in range(32):  # Sum across warp
            if warp_start + i < tpb:
                warp_sum += shared_data[warp_start + i][0]
        shared_data[local_i] = warp_sum

    barrier()

    # Use cluster_arrive for staged synchronization in sm90+
    cluster_arrive()

    # Only first thread in each block stores result
    if local_i == 0:
        var block_total: Float32 = 0.0
        for i in range(0, tpb, 32):  # Sum warp results
            if i < tpb:
                block_total += shared_data[i][0]
        output[block_id] = block_total

    # Wait for all blocks to complete their calculations in sm90+
    cluster_wait()


# ANCHOR_END: advanced_cluster_patterns_solution


def main():
    """Test cluster programming concepts using proper Mojo GPU patterns."""
    if len(argv()) < 2:
        print("Usage: p34.mojo [--coordination | --reduction | --advanced]")
        return

    with DeviceContext() as ctx:
        if argv()[1] == "--coordination":
            print("Testing Multi-Block Coordination")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            output_buf.enqueue_fill(0)

            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Float32(i % 10) * 0.1

            input_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](
                input_buf
            )
            output_tensor = LayoutTensor[
                dtype, Layout.row_major(CLUSTER_SIZE), MutAnyOrigin
            ](output_buf)

            alias kernel = cluster_coordination_basics[
                in_layout, Layout.row_major(CLUSTER_SIZE), TPB
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                print("Block coordination results:")
                for i in range(CLUSTER_SIZE):
                    print("  Block", i, ":", result_host[i])

                # FIX: Verify each block produces NON-ZERO results using proper Mojo testing
                for i in range(CLUSTER_SIZE):
                    assert_true(
                        result_host[i] > 0.0
                    )  # All blocks SHOULD produce non-zero results
                    print("✅ Block", i, "produced result:", result_host[i])

                # FIX: Verify scaling pattern - each block should have DIFFERENT results
                # Due to scaling by block_id + 1 in the kernel
                assert_true(
                    result_host[1] > result_host[0]
                )  # Block 1 > Block 0
                assert_true(
                    result_host[2] > result_host[1]
                )  # Block 2 > Block 1
                assert_true(
                    result_host[3] > result_host[2]
                )  # Block 3 > Block 2
                print("✅ Multi-block coordination tests passed!")

        elif argv()[1] == "--reduction":
            print("Testing Cluster-Wide Reduction")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](1)
            output_buf.enqueue_fill(0)
            temp_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            temp_buf.enqueue_fill(0)

            var expected_sum: Float32 = 0.0
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = Float32(i)
                    expected_sum += input_host[i]

            print("Expected sum:", expected_sum)

            input_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](
                input_buf
            )
            var output_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](
                output_buf
            )
            temp_tensor = LayoutTensor[
                dtype, Layout.row_major(CLUSTER_SIZE), MutAnyOrigin
            ](temp_buf)

            alias kernel = cluster_collective_operations[
                in_layout, out_layout, TPB
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                temp_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                result = result_host[0]
                print("Cluster reduction result:", result)
                print("Expected:", expected_sum)
                print("Error:", abs(result - expected_sum))

                # Test cluster reduction accuracy with proper tolerance
                assert_almost_equal(
                    result, expected_sum, atol=10.0
                )  # Reasonable tolerance for cluster coordination
                print("✅ Passed: Cluster reduction accuracy test")
                print("✅ Cluster-wide collective operations tests passed!")

        elif argv()[1] == "--advanced":
            print("Testing Advanced Cluster Algorithms")
            print("SIZE:", SIZE, "TPB:", TPB, "CLUSTER_SIZE:", CLUSTER_SIZE)

            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](CLUSTER_SIZE)
            output_buf.enqueue_fill(0)

            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    input_host[i] = (
                        Float32(i % 50) * 0.02
                    )  # Pattern for testing

            input_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](
                input_buf
            )
            output_tensor = LayoutTensor[
                dtype, Layout.row_major(CLUSTER_SIZE), MutAnyOrigin
            ](output_buf)

            alias kernel = advanced_cluster_patterns[
                in_layout, Layout.row_major(CLUSTER_SIZE), TPB
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                SIZE,
                grid_dim=(CLUSTER_SIZE, 1),
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with output_buf.map_to_host() as result_host:
                print("Advanced cluster algorithm results:")
                for i in range(CLUSTER_SIZE):
                    print("  Block", i, ":", result_host[i])

                # FIX: Advanced pattern should produce NON-ZERO results
                for i in range(CLUSTER_SIZE):
                    assert_true(
                        result_host[i] > 0.0
                    )  # All blocks SHOULD produce non-zero results
                    print("✅ Advanced Block", i, "result:", result_host[i])

                # FIX: Advanced pattern should show DIFFERENT scaling per block
                assert_true(
                    result_host[1] > result_host[0]
                )  # Block 1 > Block 0
                assert_true(
                    result_host[2] > result_host[1]
                )  # Block 2 > Block 1
                assert_true(
                    result_host[3] > result_host[2]
                )  # Block 3 > Block 2

                print("✅ Advanced cluster patterns tests passed!")

        else:
            print(
                "Available options: [--coordination | --reduction | --advanced]"
            )
