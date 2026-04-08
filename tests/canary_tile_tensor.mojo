"""Canary test to validate TileTensor API availability in the current SDK.

This file exercises every TileTensor API surface needed for the migration
from LayoutTensor. Run this before migrating any production files.
"""

from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout import TileTensor, LayoutTensor, Layout as IntTupleLayout
from layout.tile_layout import Layout, row_major
from layout.tile_tensor import stack_allocation
from layout.layout_tensor import copy_dram_to_sram_async
from std.testing import assert_equal, assert_almost_equal

comptime SIZE = 4
comptime TPB = 4
comptime dtype = DType.float32
comptime layout_2d = row_major[SIZE, SIZE]()
comptime layout_1d = row_major[SIZE]()
comptime Layout2D = type_of(layout_2d)
comptime Layout1D = type_of(layout_1d)


# Test 1: Basic TileTensor construction, indexing
def test_basic_kernel(
    output: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, Layout2D, ImmutAnyOrigin],
    size: Int,
):
    var row = thread_idx.y
    var col = thread_idx.x
    if col < size and row < size:
        output[row, col] = input[row, col] + 10.0


# Test 2: Stack allocation for shared memory
def test_shared_memory_kernel(
    output: TileTensor[mut=True, dtype, Layout1D, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, Layout1D, ImmutAnyOrigin],
    size: Int,
):
    var shared = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](
        row_major[TPB]()
    )

    var i = thread_idx.x
    if i < size:
        shared[i] = input[i]

    barrier()

    if i < size:
        output[i] = shared[i] + 10.0


# Test 3: Tiling operations
def test_tiling_kernel(
    output: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, Layout2D, ImmutAnyOrigin],
):
    var row = thread_idx.y
    var col = thread_idx.x

    # Test .tile[] method
    var tile = input.tile[2, 2](0, 0)
    if row < 2 and col < 2:
        output[row, col] = tile[row, col] + 1.0


# Test 4: ElementType property
def test_element_type_kernel(
    output: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, Layout2D, ImmutAnyOrigin],
):
    var row = thread_idx.y
    var col = thread_idx.x
    var acc: output.ElementType = 0
    if row < SIZE and col < SIZE:
        acc = input[row, col] * 2.0
        output[row, col] = acc


# Test 5: Async copy bridge (TileTensor -> LayoutTensor for copy_dram_to_sram_async)
def test_async_copy_bridge_kernel(
    output: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
):
    var shared = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](
        row_major[TPB, TPB]()
    )

    var tile = input.tile[TPB, TPB](0, 0)

    # Bridge: convert to LayoutTensor for async copy
    copy_dram_to_sram_async[
        thread_layout=IntTupleLayout.row_major(1, TPB),
        num_threads=TPB * TPB,
        block_dim_count=2,
    ](shared.to_layout_tensor(), tile.to_layout_tensor())

    async_copy_wait_all()
    barrier()

    var row = thread_idx.y
    var col = thread_idx.x
    if row < SIZE and col < SIZE:
        output[row, col] = shared[row, col] + 1.0


# Test 6: dim access
def test_dim_kernel(
    output: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, Layout2D, MutAnyOrigin],
):
    var row = thread_idx.y
    var col = thread_idx.x
    # Test dim[] method
    if row == 0 and col == 0:
        output[0, 0] = Scalar[dtype](Int(input.dim[0]()))
        output[0, 1] = Scalar[dtype](Int(input.dim[1]()))


def main() raises:
    with DeviceContext() as ctx:
        print("=== TileTensor API Canary Test ===")

        # Setup buffers
        var out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var in_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)

        # Initialize input
        with in_buf.map_to_host() as in_host:
            for i in range(SIZE * SIZE):
                in_host[i] = Scalar[dtype](i)

        # --- Test 1: Basic construction + indexing ---
        print("\nTest 1: Basic TileTensor construction and indexing...")
        out_buf.enqueue_fill(0)
        var out_tt = TileTensor(out_buf, layout_2d)
        var in_tt = TileTensor[mut=False, dtype, Layout2D](in_buf, layout_2d)

        ctx.enqueue_function[test_basic_kernel, test_basic_kernel](
            out_tt, in_tt, SIZE,
            grid_dim=1, block_dim=(SIZE, SIZE),
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host, in_buf.map_to_host() as in_host2:
            for i in range(SIZE * SIZE):
                assert_almost_equal(out_host[i], in_host2[i] + 10.0)
        print("  PASSED")

        # --- Test 2: Shared memory (stack_allocation) ---
        print("\nTest 2: Shared memory via stack_allocation...")
        var out_1d = ctx.enqueue_create_buffer[dtype](SIZE)
        out_1d.enqueue_fill(0)
        var in_1d = ctx.enqueue_create_buffer[dtype](SIZE)
        with in_1d.map_to_host() as in_1d_host:
            for i in range(SIZE):
                in_1d_host[i] = Scalar[dtype](i)

        var out_tt_1d = TileTensor(out_1d, layout_1d)
        var in_tt_1d = TileTensor[mut=False, dtype, Layout1D](in_1d, layout_1d)

        ctx.enqueue_function[test_shared_memory_kernel, test_shared_memory_kernel](
            out_tt_1d, in_tt_1d, SIZE,
            grid_dim=1, block_dim=(SIZE, 1),
        )
        ctx.synchronize()

        with out_1d.map_to_host() as out_host:
            for i in range(SIZE):
                assert_almost_equal(out_host[i], Scalar[dtype](i) + 10.0)
        print("  PASSED")

        # --- Test 3: Tiling ---
        print("\nTest 3: Tiling operations...")
        out_buf.enqueue_fill(0)
        out_tt = TileTensor(out_buf, layout_2d)
        in_tt = TileTensor[mut=False, dtype, Layout2D](in_buf, layout_2d)

        ctx.enqueue_function[test_tiling_kernel, test_tiling_kernel](
            out_tt, in_tt,
            grid_dim=1, block_dim=(SIZE, SIZE),
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host, in_buf.map_to_host() as in_host3:
            for r in range(2):
                for c in range(2):
                    assert_almost_equal(
                        out_host[r * SIZE + c], in_host3[r * SIZE + c] + 1.0
                    )
        print("  PASSED")

        # --- Test 4: ElementType ---
        print("\nTest 4: ElementType property...")
        out_buf.enqueue_fill(0)
        out_tt = TileTensor(out_buf, layout_2d)
        in_tt = TileTensor[mut=False, dtype, Layout2D](in_buf, layout_2d)

        ctx.enqueue_function[test_element_type_kernel, test_element_type_kernel](
            out_tt, in_tt,
            grid_dim=1, block_dim=(SIZE, SIZE),
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host, in_buf.map_to_host() as in_host4:
            for i in range(SIZE * SIZE):
                assert_almost_equal(out_host[i], in_host4[i] * 2.0)
        print("  PASSED")

        # --- Test 5: Async copy bridge ---
        print("\nTest 5: Async copy bridge (to_layout_tensor)...")
        out_buf.enqueue_fill(0)
        out_tt = TileTensor(out_buf, layout_2d)
        var in_tt_mut = TileTensor(in_buf, layout_2d)

        ctx.enqueue_function[
            test_async_copy_bridge_kernel, test_async_copy_bridge_kernel
        ](
            out_tt, in_tt_mut,
            grid_dim=1, block_dim=(SIZE, SIZE),
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host, in_buf.map_to_host() as in_host5:
            for i in range(SIZE * SIZE):
                assert_almost_equal(out_host[i], in_host5[i] + 1.0)
        print("  PASSED")

        # --- Test 6: dim access ---
        print("\nTest 6: dim[] access...")
        out_buf.enqueue_fill(0)
        out_tt = TileTensor(out_buf, layout_2d)
        in_tt_mut = TileTensor(in_buf, layout_2d)

        ctx.enqueue_function[test_dim_kernel, test_dim_kernel](
            out_tt, in_tt_mut,
            grid_dim=1, block_dim=(SIZE, SIZE),
        )
        ctx.synchronize()

        with out_buf.map_to_host() as out_host:
            assert_almost_equal(out_host[0], Scalar[dtype](SIZE))  # dim[0]
            assert_almost_equal(out_host[1], Scalar[dtype](SIZE))  # dim[1]
        print("  PASSED")

        print("\n=== ALL CANARY TESTS PASSED ===")
        print("TileTensor API is available and working correctly.")
        print("Safe to proceed with migration.")
        print("")
        print("Validated APIs:")
        print("  - TileTensor(buffer, layout) construction")
        print("  - TileTensor[mut=False, ...](buffer, layout) immutable construction")
        print("  - row_major[N, M]() layout creation")
        print("  - type_of(layout) for concrete layout types")
        print("  - tensor[row, col] indexing")
        print("  - stack_allocation[dtype=..., address_space=...](layout)")
        print("  - tensor.tile[H, W](row, col)")
        print("  - tensor.ElementType")
        print("  - tensor.dim[i]()")
        print("  - tensor.to_layout_tensor() for async copy bridge")
        print("  - NOTE: lt_to_tt() available but produces opaque layout type (not directly indexable)")
        print("  - copy_dram_to_sram_async with to_layout_tensor() bridge")
