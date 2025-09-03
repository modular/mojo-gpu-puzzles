from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import size_of, argv
from math import log2
from testing import assert_equal

alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# ANCHOR: prefix_sum_simple_solution
fn prefix_sum_simple[
    layout: Layout
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared = tb[dtype]().row_major[TPB]().shared().alloc()
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    offset = 1
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < size:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < size:
            shared[local_i] += current_val

        barrier()
        offset *= 2

    if global_i < size:
        output[global_i] = shared[local_i]


# ANCHOR_END: prefix_sum_simple_solution


alias SIZE_2 = 15
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
alias extended_layout = Layout.row_major(EXTENDED_SIZE)

# ANCHOR: prefix_sum_complete_solution


# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # Load data into shared memory
    # Example with SIZE_2=15, TPB=8, BLOCKS=2:
    # Block 0 shared mem: [0,1,2,3,4,5,6,7]
    # Block 1 shared mem: [8,9,10,11,12,13,14,uninitialized]
    # Note: The last position remains uninitialized since global_i >= size,
    # but this is safe because that thread doesn't participate in computation
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    # Compute local prefix sum using parallel reduction
    # This uses a tree-based algorithm with log(TPB) iterations
    # Iteration 1 (offset=1):
    #   Block 0: [0,0+1,2+1,3+2,4+3,5+4,6+5,7+6] = [0,1,3,5,7,9,11,13]
    # Iteration 2 (offset=2):
    #   Block 0: [0,1,3+0,5+1,7+3,9+5,11+7,13+9] = [0,1,3,6,10,14,18,22]
    # Iteration 3 (offset=4):
    #   Block 0: [0,1,3,6,10+0,14+1,18+3,22+6] = [0,1,3,6,10,15,21,28]
    #   Block 1 follows same pattern to get [8,17,27,38,50,63,77,???]
    offset = 1
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < TPB:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < TPB:
            shared[local_i] += current_val  # write

        barrier()
        offset *= 2

    # Write local results to output
    # Block 0 writes: [0,1,3,6,10,15,21,28]
    # Block 1 writes: [8,17,27,38,50,63,77,???]
    if global_i < size:
        output[global_i] = shared[local_i]

    # Store block sums in auxiliary space
    # Block 0: Thread 7 stores shared[7] == 28 at position size+0 (position 15)
    # Block 1: Thread 7 stores shared[7] == ??? at position size+1 (position 16).  This sum is not needed for the final output.
    # This gives us: [0,1,3,6,10,15,21,28, 8,17,27,38,50,63,77, 28,???]
    #                                                           ↑  ↑
    #                                                     Block sums here
    if local_i == TPB - 1:
        output[size + block_idx.x] = shared[local_i]


# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase[
    layout: Layout
](output: LayoutTensor[mut=False, dtype, layout], size: Int):
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Second pass: add previous block's sum to each element
    # Block 0: No change needed - already correct
    # Block 1: Add Block 0's sum (28) to each element
    #   Before: [8,17,27,38,50,63,77]
    #   After: [36,45,55,66,78,91,105]
    # Final result combines both blocks:
    # [0,1,3,6,10,15,21,28, 36,45,55,66,78,91,105]
    if block_idx.x > 0 and global_i < size:
        prev_block_sum = output[size + block_idx.x - 1]
        output[global_i] += prev_block_sum


# ANCHOR_END: prefix_sum_complete_solution


def main():
    with DeviceContext() as ctx:
        use_simple = argv()[1] == "--simple"
        size = SIZE if use_simple else SIZE_2
        num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = size if use_simple else EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())

        if use_simple:
            out_tensor = LayoutTensor[mut=False, dtype, layout](
                out.unsafe_ptr()
            )

            ctx.enqueue_function[prefix_sum_simple[layout]](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var out_tensor = LayoutTensor[mut=False, dtype, extended_layout](
                out.unsafe_ptr()
            )

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            ctx.enqueue_function[
                prefix_sum_local_phase[extended_layout, extended_layout]
            ](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Phase 2: Add block sums
            ctx.enqueue_function[prefix_sum_block_sum_phase[extended_layout]](
                out_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

        with out.map_to_host() as out_host:
            if not use_simple:
                print(
                    "Note: we print the extended buffer here, but we only need"
                    " to print the first `size` elements"
                )

            print("out:", out_host)
            print("expected:", expected)
            # Here we need to use the size of the original array, not the extended one
            size = size if use_simple else SIZE_2
            for i in range(size):
                assert_equal(out_host[i], expected[i])
