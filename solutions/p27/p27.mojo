from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.atomic import Atomic
from std.gpu.primitives.warp import WARP_SIZE
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal
from std.math import floor

comptime SIZE = 128
comptime TPB = 128
comptime NUM_BINS = 8
comptime in_layout = row_major[SIZE]()
comptime out_layout = row_major[1]()
comptime dtype = DType.float32
comptime InLayout = type_of(in_layout)
comptime OutLayout = type_of(out_layout)


# ANCHOR: block_sum_dot_product_solution
def block_sum_dot_product[
    tpb: Int
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    size: Int,
):
    """Dot product using block.sum() - convenience function like warp.sum()!
    Replaces manual shared memory + barriers + tree reduction with one line."""

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Each thread computes partial product
    var partial_product: Scalar[dtype] = 0.0
    if global_i < size:
        # TileTensor indexing `[0]` returns the underlying SIMD value
        partial_product = a[global_i][0] * b[global_i][0]

    # The magic: block.sum() replaces 15+ lines of manual reduction!
    # Just like warp.sum() but for the entire block
    var total = block.sum[block_size=tpb, broadcast=False](
        val=SIMD[DType.float32, 1](partial_product)
    )

    # Only thread 0 writes the result
    if local_i == 0:
        output[0] = total[0]


# ANCHOR_END: block_sum_dot_product_solution


# ANCHOR: traditional_dot_product_solution
def traditional_dot_product[
    tpb: Int
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    size: Int,
):
    """Traditional dot product using shared memory + barriers + tree reduction.
    Educational but complex - shows the manual coordination needed."""

    var shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[tpb]())
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Each thread computes partial product
    if global_i < size:
        var a_val = rebind[Scalar[dtype]](a[global_i])
        var b_val = rebind[Scalar[dtype]](b[global_i])
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

comptime bin_layout = row_major[SIZE]()  # Max SIZE elements per bin
comptime BinLayout = type_of(bin_layout)


# ANCHOR: block_histogram_solution
def block_histogram_bin_extract[
    tpb: Int
](
    input_data: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    bin_output: TileTensor[mut=True, dtype, BinLayout, MutAnyOrigin],
    count_output: TileTensor[mut=True, DType.int32, OutLayout, MutAnyOrigin],
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

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Step 1: Each thread determines its bin and element value
    var my_value: Scalar[dtype] = 0.0
    var my_bin: Int = -1

    if global_i < size:
        # `[0]` returns the underlying SIMD value
        my_value = input_data[global_i][0]
        # Bin values [0.0, 1.0) into num_bins buckets
        my_bin = Int(floor(my_value * Scalar[dtype](num_bins)))
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
    var write_offset = block.prefix_sum[
        dtype=DType.int32, block_size=tpb, exclusive=True
    ](val=SIMD[DType.int32, 1](belongs_to_target))

    # Step 4: Extract and pack elements belonging to target_bin
    if belongs_to_target == 1:
        bin_output[Int(write_offset[0])] = my_value

    # Step 5: Final thread computes total count for this bin
    if local_i == tpb - 1:
        # Inclusive sum = exclusive sum + my contribution
        var total_count = write_offset[0] + Int32(belongs_to_target)
        count_output[0] = total_count


# ANCHOR_END: block_histogram_solution

comptime vector_layout = row_major[SIZE]()  # For full vector output
comptime VectorLayout = type_of(vector_layout)


# ANCHOR: block_normalize_solution
def block_normalize_vector[
    tpb: Int
](
    input_data: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    output_data: TileTensor[mut=True, dtype, VectorLayout, MutAnyOrigin],
    size: Int,
):
    """Vector mean normalization using block.sum() + block.broadcast() combination.

    This demonstrates the complete block operations workflow:
    1. Use block.sum() to compute sum of all elements (all -> one)
    2. Thread 0 computes mean = sum / size
    3. Use block.broadcast() to share mean to all threads (one -> all)
    4. Each thread normalizes: output[i] = input[i] / mean
    """

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Step 1: Each thread loads its element
    var my_value: Scalar[dtype] = 0.0
    if global_i < size:
        my_value = input_data[global_i][0]  # Extract SIMD value

    # Step 2: Use block.sum() to compute total sum (familiar from earlier!)
    var total_sum = block.sum[block_size=tpb, broadcast=False](
        val=SIMD[DType.float32, 1](my_value)
    )

    # Step 3: Thread 0 computes mean value
    var mean_value: Scalar[dtype] = 1.0  # Default to avoid division by zero
    if local_i == 0:
        if total_sum[0] > 0.0:
            mean_value = total_sum[0] / Scalar[dtype](size)

    # Step 4: block.broadcast() shares mean to ALL threads!
    # This completes the block operations trilogy demonstration
    var broadcasted_mean = block.broadcast[
        dtype=DType.float32, width=1, block_size=tpb
    ](val=SIMD[DType.float32, 1](mean_value), src_thread=UInt(0))

    # Step 5: Each thread normalizes by the mean
    if global_i < size:
        var normalized_value = my_value / broadcasted_mean[0]
        output_data[global_i] = normalized_value


# ANCHOR_END: block_normalize_solution


def main() raises:
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
                    a_host[i] = Scalar[dtype](i)
                    b_host[i] = Scalar[dtype](2 * i)
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = TileTensor[mut=False, dtype, InLayout](a, in_layout)
            b_tensor = TileTensor[mut=False, dtype, InLayout](b_buf, in_layout)
            out_tensor = TileTensor(out, out_layout)

            # Traditional approach: works perfectly when size == TPB
            comptime kernel = traditional_dot_product[TPB]
            ctx.enqueue_function[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                SIZE,
                grid_dim=(1, 1),
                block_dim=(TPB, 1),
            )

            ctx.synchronize()

            with out.map_to_host() as result_host:
                result = result_host[0]
                print("Traditional result:", result)
                assert_equal(result, expected)
                print("Puzzle 27 complete ✅")
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
                    a_host[i] = Scalar[dtype](i)
                    b_host[i] = Scalar[dtype](2 * i)
                    expected += a_host[i] * b_host[i]

            print("SIZE:", SIZE)
            print("TPB:", TPB)
            print("Expected result:", expected)

            a_tensor = TileTensor[mut=False, dtype, InLayout](a, in_layout)
            b_tensor = TileTensor[mut=False, dtype, InLayout](b_buf, in_layout)
            out_tensor = TileTensor(out, out_layout)

            # Block.sum(): Same result with dramatically simpler code!
            comptime kernel = block_sum_dot_product[TPB]
            ctx.enqueue_function[kernel, kernel](
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
                print("Puzzle 27 complete ✅")
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
                        Scalar[dtype](i % 80) / 100.0
                    )  # Values [0.0, 0.79]

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print()

            input_tensor = TileTensor[mut=False, dtype, InLayout](
                input_buf, in_layout
            )

            # Demonstrate histogram for each bin using block.prefix_sum()
            for target_bin in range(NUM_BINS):
                print(
                    "=== Processing Bin",
                    target_bin,
                    "(range [",
                    Scalar[dtype](target_bin) / NUM_BINS,
                    ",",
                    Scalar[dtype](target_bin + 1) / NUM_BINS,
                    ")) ===",
                )

                # Create output buffers for this bin
                var bin_data = ctx.enqueue_create_buffer[dtype](SIZE)
                bin_data.enqueue_fill(0)
                var bin_count = ctx.enqueue_create_buffer[DType.int32](1)
                bin_count.enqueue_fill(0)

                var bin_tensor = TileTensor(bin_data, bin_layout)
                var count_tensor = TileTensor(bin_count, out_layout)

                # Execute histogram kernel for this specific bin
                comptime kernel = block_histogram_bin_extract[TPB]
                ctx.enqueue_function[kernel, kernel](
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
            var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
            input_buf.enqueue_fill(0)

            # Create test data: values like [1, 2, 3, 4, 5, ..., 8, 1, 2, 3, ...]
            # Mean value will be 4.5, so normalized values will be input[i] / 4.5
            var sum_value: Scalar[dtype] = 0.0
            with input_buf.map_to_host() as input_host:
                for i in range(SIZE):
                    # Create values cycling 1-8, mean will be 4.5
                    value = Scalar[dtype](
                        (i % 8) + 1
                    )  # Values 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, ...
                    input_host[i] = value
                    sum_value += value

            var mean_value = sum_value / Scalar[dtype](SIZE)

            print("Input sample:", end=" ")
            with input_buf.map_to_host() as input_host:
                for i in range(min(16, SIZE)):
                    print(input_host[i], end=" ")
            print("...")
            print("Sum value:", sum_value)
            print("Mean value:", mean_value)
            print()

            input_tensor = TileTensor[mut=False, dtype, InLayout](
                input_buf, in_layout
            )
            var output_tensor = TileTensor(output_buf, vector_layout)

            # Execute vector normalization kernel
            comptime kernel = block_normalize_vector[TPB]
            ctx.enqueue_function[kernel, kernel](
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

                var output_mean = output_sum / Scalar[dtype](SIZE)
                print("Output sum:", output_sum)
                print("Output mean:", output_mean)
                print(
                    "✅ Success: Output mean is",
                    output_mean,
                    "(should be close to 1.0)",
                )
                print("Puzzle 27 complete ✅")
        else:
            print(
                "Available options: [--traditional-dot-product |"
                " --block-sum-dot-product | --histogram | --normalize]"
            )
