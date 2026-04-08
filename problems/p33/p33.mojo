from std.gpu import thread_idx, block_idx, block_dim, barrier, WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from layout.tensor_core import TensorCore
from layout.layout_tensor import copy_dram_to_sram_async
from std.utils import Index
from std.sys import argv
from std.testing import assert_equal, assert_almost_equal

comptime dtype = DType.float32
comptime SIZE = 1024
comptime layout = row_major[SIZE, SIZE]()
comptime LayoutType = type_of(layout)
comptime BLOCK_DIM_COUNT = 2

comptime TILE_SIZE = 32
comptime BLOCK_PER_GRID_TILED = (
    (SIZE + TILE_SIZE - 1) // TILE_SIZE,
    (SIZE + TILE_SIZE - 1) // TILE_SIZE,
)
comptime THREADS_PER_BLOCK_TILED = (TILE_SIZE, TILE_SIZE)


# ANCHOR: matmul_idiomatic_tiled_solution
def matmul_idiomatic_tiled[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    # Use block_dim to get actual tile size dynamically
    var tile_size_x = block_dim.x
    var tile_size_y = block_dim.y

    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * tile_size_y + local_row
    var tiled_col = block_idx.x * tile_size_x + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    var out_tile = output.tile[TILE_SIZE, TILE_SIZE](block_idx.y, block_idx.x)
    var a_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TILE_SIZE, TILE_SIZE]())
    var b_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TILE_SIZE, TILE_SIZE]())

    var acc: output.ElementType = 0

    comptime load_a_layout = row_major[1, TILE_SIZE]()  # Coalesced loading
    comptime load_b_layout = row_major[1, TILE_SIZE]()  # Coalesced loading
    # Note: Both matrices stored in same orientation for correct matrix multiplication
    # Transposed loading would be useful if B were pre-transposed in global memory

    for idx in range(size // TILE_SIZE):  # Iterate over K tiles
        # Get tiles from A and B matrices
        var a_tile = a.tile[TILE_SIZE, TILE_SIZE](block_idx.y, idx)
        var b_tile = b.tile[TILE_SIZE, TILE_SIZE](idx, block_idx.x)

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async[
            thread_layout=load_a_layout,
            num_threads=TILE_SIZE * TILE_SIZE,
            block_dim_count=BLOCK_DIM_COUNT,
        ](a_shared, a_tile)
        copy_dram_to_sram_async[
            thread_layout=load_b_layout,
            num_threads=TILE_SIZE * TILE_SIZE,
            block_dim_count=BLOCK_DIM_COUNT,
        ](b_shared, b_tile)

        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        for k in range(TILE_SIZE):
            if (
                local_row < TILE_SIZE
                and local_col < TILE_SIZE
                and k < TILE_SIZE
            ):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result to output tile
    if tiled_row < size and tiled_col < size:
        out_tile[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled_solution

# ANCHOR: tensor_core_matrix_multiplication
# Block and warp tiling sizes
comptime BM = 4 * WARP_SIZE  # Block tile M (4 warps along M)
comptime BN = 2 * WARP_SIZE  # Block tile N (2 warps along N)
comptime BK = WARP_SIZE  # Block tile K (stay within SMEM limit)
comptime WM = WARP_SIZE  # Warp tile M
comptime WN = WARP_SIZE  # Warp tile N

# MMA tile sizes for tensor cores
comptime MMA_M = 16
comptime MMA_N = 8
comptime MMA_K = 8

comptime THREADS_PER_BLOCK_TENSOR_CORE = (8 * WARP_SIZE, 1)  # 8 warps per block
# grid_dim is (x, y). We want x to sweep N (columns) and y to sweep M (rows)
comptime BLOCKS_PER_GRID_TENSOR_CORE = (
    (SIZE + BN - 1) // BN,
    (SIZE + BM - 1) // BM,
)


def tensor_core_matrix_multiplication[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    B: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    C: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime M = C.dim[0]()
    comptime N = C.dim[1]()
    comptime K = A.dim[1]()

    var warp_id = thread_idx.x // WARP_SIZE
    var warps_in_n = BN // WN
    var warps_in_m = BM // WM
    var warp_y = warp_id // warps_in_n
    var warp_x = warp_id % warps_in_n

    var warp_is_active = warp_y < warps_in_m

    var C_block_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
    var C_warp_tile = C_block_tile.tile[WM, WN](warp_y, warp_x)

    var mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    # Shared SRAM tiles (no padding to stay under shared memory limit)
    var A_sram_tile = stack_allocation[
        dtype=A.dtype, address_space=AddressSpace.SHARED
    ](row_major[BM, BK]())
    var B_sram_tile = stack_allocation[
        dtype=B.dtype, address_space=AddressSpace.SHARED
    ](row_major[BK, BN]())

    # One per-warp accumulator tile of shape [WM, WN]
    var C_warp_accum = stack_allocation[
        dtype=C.dtype, address_space=AddressSpace.GENERIC
    ](row_major[WM, WN]())

    # Zero initialize accumulator (only for active warps)
    if warp_is_active:
        comptime for i in range(WM):
            comptime for j in range(WN):
                C_warp_accum[i, j] = 0.0

    # Sweep across K in BK chunks (single-buffered)
    for k_i in range(K // BK):
        barrier()

        var A_dram_tile = A.tile[BM, BK](block_idx.y, k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, block_idx.x)

        copy_dram_to_sram_async[
            thread_layout=row_major[4, 8](),
            num_threads=256,
            block_dim_count=BLOCK_DIM_COUNT,
        ](A_sram_tile.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]())
        copy_dram_to_sram_async[
            thread_layout=row_major[4, 8](),
            num_threads=256,
            block_dim_count=BLOCK_DIM_COUNT,
        ](B_sram_tile.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]())

        async_copy_wait_all()
        barrier()

        if warp_is_active:
            var A_warp_tile = A_sram_tile.tile[WM, BK](warp_y, 0)
            var B_warp_tile = B_sram_tile.tile[BK, WN](0, warp_x)

            comptime for mma_k in range(BK // MMA_K):
                comptime for mma_m in range(WM // MMA_M):
                    comptime for mma_n in range(WN // MMA_N):
                        # FILL IN (roughly 8 lines)
                        ...

    # Store the final per-warp accumulation to the output warp tile
    if warp_is_active:
        comptime for mma_m in range(WM // MMA_M):
            comptime for mma_n in range(WN // MMA_N):
                var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
                var Acc_mma_tile = C_warp_accum.tile[MMA_M, MMA_N](mma_m, mma_n)
                var frag = mma_op.load_c(Acc_mma_tile)
                mma_op.store_d(C_mma_tile, frag)


# ANCHOR_END: tensor_core_matrix_multiplication


def main() raises:
    print("Puzzle 33: Tensor Core Operations")

    if len(argv()) < 2:
        print("\nUsage:")
        print("  --tensor-core      : Run ACTUAL tensor core matmul")
        print("  --tiled            : Run idiomatic tiled matmul")
        print(
            "  --test             : Run accuracy tests for all implementations"
            " (CPU, Tensor Core, Tiled)"
        )
        print("\nThis uses ACTUAL TensorCore API methods:")
        print("  - mma_op.load_a() - Load matrix A fragments")
        print("  - mma_op.load_b() - Load matrix B fragments")
        print("  - mma_op.load_c() - Load matrix C fragments")
        print("  - mma_op.mma_op() - Perform D = A * B + C operation")
        print("  - mma_op.store_d() - Store result matrix D")
        return

    var mode = argv()[1]

    with DeviceContext() as ctx:
        # Create buffers
        var out_tensor_core = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_tensor_core.enqueue_fill(0)
        var inp1 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var inp2 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)

        # Initialize data (like p16.mojo)
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(SIZE):
                for col in range(SIZE):
                    var val = row * SIZE + col
                    inp1_host[row * SIZE + col] = Scalar[dtype](val)
                    inp2_host[row * SIZE + col] = Scalar[dtype](2.0) * Scalar[
                        dtype
                    ](val)

            # Calculate expected CPU result: inp1 @ inp2
            for i in range(SIZE):
                for j in range(SIZE):
                    for k in range(SIZE):
                        expected[i * SIZE + j] += (
                            inp1_host[i * SIZE + k] * inp2_host[k * SIZE + j]
                        )
        # Create layout tensors
        var out_tensor_core_layout = TileTensor(out_tensor_core, layout)
        var a_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            inp1, layout
        )
        var b_tensor = TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin](
            inp2, layout
        )

        if mode == "--tensor-core":
            print("\n=== Running ACTUAL Tensor Core Matrix Multiplication ===")
            comptime kernel = tensor_core_matrix_multiplication[
                dtype,
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                MMA_K,
            ]
            ctx.enqueue_function[kernel, kernel](
                a_tensor,
                b_tensor,
                out_tensor_core_layout,
                grid_dim=BLOCKS_PER_GRID_TENSOR_CORE,
                block_dim=THREADS_PER_BLOCK_TENSOR_CORE,
            )
            ctx.synchronize()
            print("SUCCESS: Tensor core matmul completed!")
            print("Puzzle 33 complete ✅")

        elif mode == "--tiled":
            print("\n=== Running Idiomatic Tiled Matrix Multiplication ===")

            # Create separate buffer for tiled result
            out_tiled = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
            out_tiled.enqueue_fill(0)
            out_tiled_layout = TileTensor(out_tiled, layout)

            # Run idiomatic tiled version with proper 2D block configuration
            comptime kernel = matmul_idiomatic_tiled[SIZE]
            ctx.enqueue_function[kernel, kernel](
                out_tiled_layout,
                a_tensor,
                b_tensor,
                grid_dim=BLOCK_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
            ctx.synchronize()
            print("SUCCESS: Idiomatic tiled matmul completed!")
            print("Puzzle 33 complete ✅")

        elif mode == "--test":
            print("\n=== Running All Accuracy Tests ===")
            print(
                "Comparing CPU reference vs Tensor Core vs Idiomatic Tiled"
                " implementations"
            )

            # Test 1: Tensor Core vs CPU
            print("\n--- Test 1: Tensor Core vs CPU Reference ---")
            comptime kernel = tensor_core_matrix_multiplication[
                dtype,
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                MMA_K,
            ]
            ctx.enqueue_function[kernel, kernel](
                a_tensor,
                b_tensor,
                out_tensor_core_layout,
                grid_dim=BLOCKS_PER_GRID_TENSOR_CORE,
                block_dim=THREADS_PER_BLOCK_TENSOR_CORE,
            )
            ctx.synchronize()

            with out_tensor_core.map_to_host() as tc_host:
                print(
                    "Sample tensor core results:",
                    tc_host[0],
                    tc_host[1],
                    tc_host[SIZE * SIZE - 1],
                )
                print(
                    "Sample CPU reference:      ",
                    expected[0],
                    expected[1],
                    expected[SIZE * SIZE - 1],
                )

                tc_success = True
                var error_count = 0
                for i in range(SIZE * SIZE):
                    try:
                        assert_almost_equal(
                            tc_host[i], expected[i], atol=1e-3, rtol=2e-2
                        )
                    except:
                        if error_count < 10:  # Show first 10 failures
                            var row = i // SIZE
                            var col = i % SIZE
                            var diff = abs(tc_host[i] - expected[i])
                            print(
                                "FAIL[",
                                i,
                                "] (",
                                row,
                                ",",
                                col,
                                "): tc=",
                                tc_host[i],
                                ", expected=",
                                expected[i],
                                ", diff=",
                                diff,
                            )
                        error_count += 1
                        tc_success = False

                if tc_success:
                    print("Tensor core test: passed")
                else:
                    print(
                        "❌ TENSOR CORE ACCURACY TEST FAILED -",
                        error_count,
                        "mismatches out of",
                        SIZE * SIZE,
                        "elements",
                    )

            # Test 2: Idiomatic Tiled vs CPU
            print("\n--- Test 2: Idiomatic Tiled vs CPU Reference ---")
            out_tiled = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
            out_tiled.enqueue_fill(0)
            out_tiled_layout = TileTensor(out_tiled, layout)

            comptime kernel2 = matmul_idiomatic_tiled[SIZE]
            ctx.enqueue_function[kernel2, kernel2](
                out_tiled_layout,
                a_tensor,
                b_tensor,
                grid_dim=BLOCK_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )
            ctx.synchronize()

            with out_tiled.map_to_host() as tiled_host:
                print(
                    "Sample tiled results:",
                    tiled_host[0],
                    tiled_host[1],
                    tiled_host[SIZE * SIZE - 1],
                )
                print(
                    "Sample CPU reference:",
                    expected[0],
                    expected[1],
                    expected[SIZE * SIZE - 1],
                )

                try:
                    # Use assert_almost_equal for each element (exact FP32 precision)
                    for i in range(SIZE * SIZE):
                        assert_almost_equal(tiled_host[i], expected[i])
                    print("Idiomatic tiled test: passed")
                    tiled_success = True
                except:
                    print(
                        "❌ IDIOMATIC TILED ACCURACY TEST FAILED -"
                        " assert_almost_equal failed"
                    )
                    tiled_success = False

            print("\n=== ACCURACY TEST SUMMARY ===")
            if tc_success and tiled_success:
                print("ALL TESTS PASSED!")
                print("Puzzle 33 complete ✅")
            else:
                print("Some tests failed:")
                print("   - Tensor Core:", "✅" if tc_success else "❌")
                print("   - Idiomatic Tiled:", "✅" if tiled_success else "❌")

        else:
            print("ERROR: Unknown option:", mode)
            return

    print("\nACTUAL TensorCore API Implementation:")
    print("  - TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()")
    print("  - mma_op.load_a() - Load matrix A fragments from shared memory")
    print("  - mma_op.load_b() - Load matrix B fragments from shared memory")
    print("  - mma_op.load_c() - Load matrix C fragments from global memory")
    print("  - mma_op.mma_op() - Perform D = A * B + C using tensor cores")
    print("  - mma_op.store_d() - Store result matrix D to global memory")
    print("  - Warp organization and MMA tiling (16x8x8 for float32)")
    print("  - Asynchronous memory operations with barriers")
    print(
        "  - Reference:"
        " https://docs.modular.com/mojo/kernels/layout/tensor_core/TensorCore/"
    )

    print("\nPerformance Analysis:")
    print(
        "1. Build: pixi run mojo build solutions/p33/p33.mojo -o"
        " solutions/p33/p33_profiler"
    )
    print(
        "2. Profile: ncu --set full --metrics"
        " smspinst_executed_pipe_tensor_op_hmma.sum,smsp_pipe_tensor_op_hmma_cycles_active.sum"
        " ./solutions/p33/p33_profiler --tensor-core"
    )
