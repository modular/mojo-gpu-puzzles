from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import size_of, argv
from math import log2
from testing import assert_equal

# ANCHOR: prefix_sum_simple
alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


fn prefix_sum_simple[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 18 lines)


# ANCHOR_END: prefix_sum_simple

# ANCHOR: prefix_sum_complete
alias SIZE_2 = 15
alias BLOCKS_PER_GRID_2 = (2, 1)
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
alias layout_2 = Layout.row_major(SIZE_2)
alias extended_layout = Layout.row_major(EXTENDED_SIZE)


# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 20 lines)


# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase[
    layout: Layout
](output: LayoutTensor[dtype, layout, MutAnyOrigin], size: Int):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 3 lines)


# ANCHOR_END: prefix_sum_complete


def main():
    with DeviceContext() as ctx:
        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--complete",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or '--complete'"
            )

        use_simple = argv()[1] == "--simple"

        size = SIZE if use_simple else SIZE_2
        num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = size if use_simple else EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        if use_simple:
            a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)
            out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)

            alias kernel = prefix_sum_simple[layout]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var a_tensor = LayoutTensor[dtype, layout_2, ImmutAnyOrigin](a)
            var out_tensor = LayoutTensor[dtype, extended_layout, MutAnyOrigin](
                out
            )

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            alias kernel = prefix_sum_local_phase[extended_layout, layout_2]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Wait for all `blocks` to complete with using host `ctx.synchronize()`
            # Note this is in contrast with using `barrier()` in the kernel
            # which is a synchronization point for all threads in the same block and not across blocks.
            ctx.synchronize()

            # Phase 2: Add block sums
            alias kernel2 = prefix_sum_block_sum_phase[extended_layout]
            ctx.enqueue_function_checked[kernel2, kernel2](
                out_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)
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
