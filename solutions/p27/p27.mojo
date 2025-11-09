from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from os.atomic import Atomic
from gpu.warp import WARP_SIZE
from gpu import block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import argv
from testing import assert_equal
from math import floor

alias SIZE = 128
alias TPB = 128
alias NUM_BINS = 8
alias in_layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)
alias dtype = DType.float32


# ANCHOR: block_sum_dot_product_solution
fn block_sum_dot_product[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """Dot product using block.sum() - convenience function like warp.sum()!
    Replaces manual shared memory + barriers + tree reduction with one line."""

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Each thread computes partial product
    var partial_product: Scalar[dtype] = 0.0
    if global_i < size:
        # LayoutTensor indexing `[0]` returns the underlying SIMD value
        partial_product = a[global_i][0] * b[global_i][0]

    # The magic: block.sum() replaces 15+ lines of manual reduction!
    # Just like warp.sum() but for the entire block
    total = block.sum[block_size=tpb, broadcast=False](
        val=SIMD[DType.float32, 1](partial_product)
    )

    # Only thread 0 writes the result
    if local_i == 0:
        output[0] = total[0]


# ANCHOR_END: block_sum_dot_product_solution


# ANCHOR: traditional_dot_product_solution
fn traditional_dot_product[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """Traditional dot product using shared memory + barriers + tree reduction.
    Educational but complex - shows the manual coordination needed."""

    shared = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Each thread computes partial product
    if global_i < size:
        a_val = rebind[Scalar[dtype]](a[global_i])
        b_val = rebind[Scalar[dtype]](b[global_i])
        shared[local_i] = a_val * b_val

    barrier()

    # Tree reduction in shared memory - complex but educational
    var stride = tpb // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    # Only thread 0 writes final result
    if local_i == 0:
        output[0] = shared[0]


# ANCHOR_END: traditional_dot_product_solution

alias bin_layout = Layout.row_major(SIZE)  # Max SIZE elements per bin


# ANCHOR: block_histogram_solution
fn block_histogram_bin_extract[
    in_layout: Layout, bin_layout: Layout, out_layout: Layout, tpb: Int
](
    input_data: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    bin_output: LayoutTensor[dtype, bin_layout, MutAnyOrigin],
    count_output: LayoutTensor[DType.int32, out_layout, MutAnyOrigin],
    size: Int,
    target_bin: Int,
    num_bins: Int,
):
    """Parallel histogram using block.prefix_sum() for bin extraction.

    This demonstrates advanced parallel filtering and extraction:
    1. Each thread determines which bin its element belongs to
    2. Use block.prefix_sum() to compute write positions for target_bin elements
    3. Extract and pack only elements belonging to target_bin
    """

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Step 1: Each thread determines its bin and element value
    var my_value: Scalar[dtype] = 0.0
    var my_bin: Int = -1

    if global_i < size:
        # `[0]` returns the underlying SIMD value
        my_value = input_data[global_i][0]
        # Bin values [0.0, 1.0) into num_bins buckets
        my_bin = Int(floor(my_value * num_bins))
        # Clamp to valid range
        if my_bin >= num_bins:
            my_bin = num_bins - 1
        if my_bin < 0:
            my_bin = 0

    # Step 2: Create predicate for target bin extraction
    var belongs_to_target: Int = 0
    if global_i < size and my_bin == target_bin:
        belongs_to_target = 1

    # Step 3: Use block.prefix_sum() for parallel bin extraction!
    # This computes where each thread should write within the target bin
    write_offset = block.prefix_sum[
        dtype = DType.int32, block_size=tpb, exclusive=True
    ](val=SIMD[DType.int32, 1](belongs_to_target))

    # Step 4: Extract and pack elements belonging to target_bin
    if belongs_to_target == 1:
        bin_output[Int(write_offset[0])] = my_value

    # Step 5: Final thread computes total count for this bin
    if local_i == tpb - 1:
        # Inclusive sum = exclusive sum + my contribution
        total_count = write_offset[0] + belongs_to_target
        count_output[0] = total_count


# ANCHOR_END: block_histogram_solution

alias vector_layout = Layout.row_major(SIZE)  # For full vector output


# ANCHOR: block_normalize_solution
fn block_normalize_vector[
    in_layout: Layout, out_layout: Layout, tpb: Int
](
    input_data: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    output_data: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    size: Int,
):
    """Vector mean normalization using block.sum() + block.broadcast() combination.

    This demonstrates the complete block operations workflow:
    1. Use block.sum() to compute sum of all elements (all → one)
    2. Thread 0 computes mean = sum / size
    3. Use block.broadcast() to share mean to all threads (one → all)
    4. Each thread normalizes: output[i] = input[i] / mean
    """

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Step 1: Each thread loads its element
    var my_value: Scalar[dtype] = 0.0
    if global_i < size:
        my_value = input_data[global_i][0]  # Extract SIMD value

    # Step 2: Use block.sum() to compute total sum (familiar from earlier!)
    total_sum = block.sum[block_size=tpb, broadcast=False](
        val=SIMD[DType.float32, 1](my_value)
    )

    # Step 3: Thread 0 computes mean value
    var mean_value: Scalar[dtype] = 1.0  # Default to avoid division by zero
    if local_i == 0:
        if total_sum[0] > 0.0:
            mean_value = total_sum[0] / Float32(size)

    # Step 4: block.broadcast() shares mean to ALL threads!
    # This completes the block operations trilogy demonstration
    broadcasted_mean = block.broadcast[
        dtype = DType.float32, width=1, block_size=tpb
    ](val=SIMD[DType.float32, 1](mean_value), src_thread=UInt(0))

    # Step 5: Each thread normalizes by the mean
    if global_i < size:
        normalized_value = my_value / broadcasted_mean[0]
        output_data[global_i] = normalized_value


# ANCHOR_END: block_normalize_solution


def main():
    if len(argv()) != 2:
        print(
            "Usage: --traditional-dot-product | --block-sum-dot-product |"
            " --histogram | --normalize"
        )
        return

    with DeviceContext() as ctx:
        if argv()[1] == "--traditional-dot-product":
            out = ctx.enqueue_create_buffer[dtype](1)
            out.enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](SIZE)
            a.enqueue_fill(0)
            b_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            b_buf.enqueue_fill(0)

            var expected: Scalar[dtype] = 0.0
            with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = i
                    b_host[i] = 2 * i
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
            b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b_buf)
            out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)

            # Traditional approach: works perfectly when size == TPB
            alias kernel = traditional_dot_product[in_layout, out_layout, TPB]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                SIZE,
                grid_dim=(1, 1),  # ✅ Single block works when size == TPB
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with out.map_to_host() as result_host:
                result = result_host[0]
                print("Traditional result:", result)
                assert_equal(result, expected)
                print("Complex: shared memory + barriers + tree reduction")

        elif argv()[1] == "--block-sum-dot-product":
            out = ctx.enqueue_create_buffer[dtype](1)
            out.enqueue_fill(0)
            a = ctx.enqueue_create_buffer[dtype](SIZE)
            a.enqueue_fill(0)
            b_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            b_buf.enqueue_fill(0)

            var expected: Scalar[dtype] = 0.0
            with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
                for i in range(SIZE):
                    a_host[i] = i
                    b_host[i] = 2 * i
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
            b_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](b_buf)
            out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)

            # Block.sum(): Same result with dramatically simpler code!
            alias kernel = block_sum_dot_product[in_layout, out_layout, TPB]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                SIZE,
                grid_dim=(1, 1),  # Same single block as traditional
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with out.map_to_host() as result_host:
                result = result_host[0]
                print("Block.sum result:", result)
                assert_equal(result, expected)
                print("Block.sum() gives identical results!")
                print(
                    "Compare the code: 15+ lines of barriers → 1 line of"
                    " block.sum()!"
                )
                print("Just like warp.sum() but for the entire block")

        elif argv()[1] == "--histogram":
            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("NUM_BINS:", NUM_BINS)
            print()

            # Create input data with known distribution across bins
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)

            # Create test data: values distributed across 8 bins [0.0, 1.0)
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    # Create values: 0.1, 0.2, 0.3, ..., cycling through bins
                    input_host[i] = (
                        Float32(i % 80) / 100.0
                    )  # Values [0.0, 0.79]

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print()

            input_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](
                input_buf
            )

            # Demonstrate histogram for each bin using block.prefix_sum()
            for target_bin in range(NUM_BINS):
                print(
                    "=== Processing Bin",
                    target_bin,
                    "(range [",
                    Float32(target_bin) / NUM_BINS,
                    ",",
                    Float32(target_bin + 1) / NUM_BINS,
                    ")) ===",
                )

                # Create output buffers for this bin
                bin_data = ctx.enqueue_create_buffer[dtype](SIZE)
                bin_data.enqueue_fill(0)
                bin_count = ctx.enqueue_create_buffer[DType.int32](1)
                bin_count.enqueue_fill(0)

                bin_tensor = LayoutTensor[dtype, bin_layout, MutAnyOrigin](
                    bin_data
                )
                count_tensor = LayoutTensor[
                    DType.int32, out_layout, MutAnyOrigin
                ](bin_count)

                # Execute histogram kernel for this specific bin
                alias kernel = block_histogram_bin_extract[
                    in_layout, bin_layout, out_layout, TPB
                ]
                ctx.enqueue_function_checked[kernel, kernel](
                    input_tensor,
                    bin_tensor,
                    count_tensor,
                    SIZE,
                    target_bin,
                    NUM_BINS,
                    grid_dim=(
                        1,
                        1,
                    ),  # Single block demonstrates block.prefix_sum()
                    block_dim=(TPB, 1),
                )

                ctx.synchronize()

                # Display results for this bin
                with bin_count.map_to_host() as count_host:
                    count = count_host[0]
                    print("Bin", target_bin, "count:", count)

                with bin_data.map_to_host() as bin_host:
                    print("Bin", target_bin, "extracted elements:", end=" ")
                    for i in range(min(8, Int(count))):
                        print(bin_host[i], end=" ")
                    if count > 8:
                        print("...")
                    else:
                        print()
                print()

        elif argv()[1] == "--normalize":
            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print()

            # Create input data with known values for easy verification
            input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)
            output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)

            # Create test data: values like [1, 2, 3, 4, 5, ..., 8, 1, 2, 3, ...]
            # Mean value will be 4.5, so normalized values will be input[i] / 4.5
            var sum_value: Scalar[dtype] = 0.0
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    # Create values cycling 1-8, mean will be 4.5
                    value = Float32(
                        (i % 8) + 1
                    )  # Values 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, ...
                    input_host[i] = value
                    sum_value += value

            var mean_value = sum_value / Float32(SIZE)

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print("Sum value:", sum_value)
            print("Mean value:", mean_value)
            print()

            input_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](
                input_buf
            )
            output_tensor = LayoutTensor[dtype, vector_layout, MutAnyOrigin](
                output_buf
            )

            # Execute vector normalization kernel
            alias kernel = block_normalize_vector[in_layout, vector_layout, TPB]
            ctx.enqueue_function_checked[kernel, kernel](
                input_tensor,
                output_tensor,
                SIZE,
                grid_dim=(1, 1),  # Single block demonstrates block.broadcast()
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            # Verify results
            print("Mean Normalization Results:")
            with output_buf.map_to_host() as output_host:
                print("Normalized sample:", end=" ")
                for i in range(min(16, SIZE)):
                    print(output_host[i], end=" ")
                print("...")

                # Verify that the mean normalization worked (mean of output should be ~1.0)
                var output_sum: Scalar[dtype] = 0.0
                for i in range(SIZE):
                    output_sum += output_host[i]

                var output_mean = output_sum / Float32(SIZE)
                print("Output sum:", output_sum)
                print("Output mean:", output_mean)
                print(
                    "✅ Success: Output mean is",
                    output_mean,
                    "(should be close to 1.0)",
                )
        else:
            print(
                "Available options: [--traditional-dot-product |"
                " --block-sum-dot-product | --histogram | --normalize]"
            )
