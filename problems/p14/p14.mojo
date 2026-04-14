from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.math import log2
from std.testing import assert_equal

# ANCHOR: prefix_sum_simple
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)


def prefix_sum_simple(
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # FILL ME IN (roughly 18 lines)


# ANCHOR_END: prefix_sum_simple

# ANCHOR: prefix_sum_complete
comptime SIZE_2 = 15
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
comptime layout_2 = row_major[SIZE_2]()
comptime Layout2Type = type_of(layout_2)
comptime extended_layout = row_major[EXTENDED_SIZE]()
comptime ExtendedLayoutType = type_of(extended_layout)


# Kernel 1: Compute local prefix sums and store block sums in out
def prefix_sum_local_phase(
    output: TileTensor[mut=True, dtype, ExtendedLayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, Layout2Type, ImmutAnyOrigin],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # FILL ME IN (roughly 20 lines)


# Kernel 2: Add block sums to their respective blocks
def prefix_sum_block_sum_phase(
    output: TileTensor[mut=True, dtype, ExtendedLayoutType, MutAnyOrigin],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 3 lines)


# ANCHOR_END: prefix_sum_complete


def main() raises:
    with DeviceContext() as ctx:
        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--complete",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or '--complete'"
            )

        var use_simple = argv()[1] == "--simple"

        var size = SIZE if use_simple else SIZE_2
        var num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        var buffer_size = size if use_simple else EXTENDED_SIZE
        var out = ctx.enqueue_create_buffer[dtype](buffer_size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = Scalar[dtype](i)

        if use_simple:
            a_tensor = TileTensor[mut=False, dtype, LayoutType](a, layout)
            out_tensor = TileTensor(out, layout)

            ctx.enqueue_function[prefix_sum_simple, prefix_sum_simple](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var a_tensor = TileTensor[mut=False, dtype, Layout2Type](
                a, layout_2
            )
            var out_tensor = TileTensor(out, extended_layout)

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            ctx.enqueue_function[
                prefix_sum_local_phase, prefix_sum_local_phase
            ](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Note: kernel2 starts when kernel1 is finished due to Mojo's DeviceContext using a single execution stream
            # No explicit ctx.synchronize() needed in this case.

            # Phase 2: Add block sums
            ctx.enqueue_function[
                prefix_sum_block_sum_phase, prefix_sum_block_sum_phase
            ](
                out_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        var expected = ctx.enqueue_create_host_buffer[dtype](size)
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
            print("Puzzle 14 complete ✅")
