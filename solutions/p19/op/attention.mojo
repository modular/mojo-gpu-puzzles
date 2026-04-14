from std.memory import UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.math import exp
from std.bit import log2_ceil
from std.utils.numerics import max_finite, min_finite
import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

comptime SEQ_LEN = 16  # This must be equal to SEQ_LEN in p19.py
comptime D = 16  # This must be equal to D in p19.py

comptime TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output
comptime MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
comptime MATMUL_NUM_THREADS = MATMUL_BLOCK_DIM_XY * MATMUL_BLOCK_DIM_XY
comptime MATMUL_BLOCK_DIM_COUNT = 2
comptime SOFTMAX_BLOCK_DIM_X = 1 << log2_ceil(SEQ_LEN)


# Tiled matrix multiplication (from p16), updated to:
# 1) Support different layouts for input (a, b) and output TileTensors.
# 2) Handle cases where the inner dimension is not a multiple of MATMUL_BLOCK_DIM_XY.
# 3) Explicitly check for out-of-bounds elements.
# The approach still tiles all three TileTensors (a, b, and output) into identical square tiles
# of size (MATMUL_BLOCK_DIM_XY x MATMUL_BLOCK_DIM_XY) with each thread loading one element
# from a and b, and writing one element to output.
def matmul_idiomatic_tiled[
    rows: Int,
    cols: Int,
    inner: Int,
    OutLayout: TensorLayout,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=True, dtype, ALayout, MutAnyOrigin],
    b: TileTensor[mut=True, dtype, BLayout, MutAnyOrigin],
):
    """Updated idiomatic tiled matrix multiplication from p16."""
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var tiled_row = block_idx.y * MATMUL_BLOCK_DIM_XY + local_row
    var tiled_col = block_idx.x * MATMUL_BLOCK_DIM_XY + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    var out_tile = output.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
        block_idx.y, block_idx.x
    )
    comptime shared_layout = row_major[
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    ]()
    var a_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](shared_layout)
    var b_shared = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](shared_layout)
    var acc: output.ElementType = 0

    var a_lt = a.to_layout_tensor()
    var b_lt = b.to_layout_tensor()
    var out_tile_lt = out_tile.to_layout_tensor()
    var a_shared_lt = a_shared.to_layout_tensor()
    var b_shared_lt = b_shared.to_layout_tensor()

    comptime for idx in range(
        (inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY
    ):
        # Get tiles from A and B matrices
        var a_tile_row_start = block_idx.y * MATMUL_BLOCK_DIM_XY
        var a_tile_col_start = idx * MATMUL_BLOCK_DIM_XY
        var b_tile_row_start = idx * MATMUL_BLOCK_DIM_XY
        var b_tile_col_start = block_idx.x * MATMUL_BLOCK_DIM_XY

        # Synchronously load tiles to shared memory - each thread loads one element
        var a_global_row = a_tile_row_start + local_row
        var a_global_col = a_tile_col_start + local_col
        if a_global_row < rows and a_global_col < inner:
            a_shared_lt[local_row, local_col] = a_lt[a_global_row, a_global_col]
        else:
            a_shared_lt[local_row, local_col] = 0

        var b_global_row = b_tile_row_start + local_row
        var b_global_col = b_tile_col_start + local_col
        if b_global_row < inner and b_global_col < cols:
            b_shared_lt[local_row, local_col] = b_lt[b_global_row, b_global_col]
        else:
            b_shared_lt[local_row, local_col] = 0

        barrier()

        # Compute partial matrix multiplication for this tile
        comptime k_max = min(
            MATMUL_BLOCK_DIM_XY, inner - idx * MATMUL_BLOCK_DIM_XY
        )
        comptime for k in range(k_max):
            if tiled_row < rows and tiled_col < cols:
                acc += rebind[Scalar[dtype]](
                    a_shared_lt[local_row, k]
                ) * rebind[Scalar[dtype]](b_shared_lt[k, local_col])

        barrier()

    # Write final result with bounds checking (needed for attention's variable sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile_lt[local_row, local_col] = acc


# ANCHOR: transpose_kernel_solution
def transpose_kernel[
    rows: Int,
    cols: Int,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[mut=True, dtype, InLayout, MutAnyOrigin],
):
    """Transpose matrix using shared memory tiling for coalesced access."""
    comptime shared_layout = row_major[
        TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY
    ]()
    var shared_tile = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](shared_layout)

    var local_row = thread_idx.y
    var local_col = thread_idx.x

    var global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row
    var global_col = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_col

    var inp_lt = inp.to_layout_tensor()
    var output_lt = output.to_layout_tensor()
    var shared_tile_lt = shared_tile.to_layout_tensor()

    if global_row < rows and global_col < cols:
        shared_tile_lt[local_row, local_col] = inp_lt[global_row, global_col]

    barrier()

    var out_row = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_row
    var out_col = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output_lt[out_row, out_col] = shared_tile_lt[local_col, local_row]


# ANCHOR_END: transpose_kernel_solution


# Apply softmax to attention scores taken from p16
def softmax_gpu_kernel[
    input_size: Int,
    LayoutType: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
):
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating-point type"
    comptime softmax_layout = row_major[SOFTMAX_BLOCK_DIM_X]()
    var shared_max = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](softmax_layout)
    var shared_sum = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](softmax_layout)
    var global_i = thread_idx.x
    var input_lt = input.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input_lt[global_i])
    shared_max[global_i] = val

    barrier()

    # Parallel reduction to find max similar to reduction we saw before
    var stride = SOFTMAX_BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    var block_max = shared_max[0]

    # Initialize out-of-bounds (shared_max[global_i], global_i >= input_size) shared memory addresses to 0.0,
    # ensuring that if these elements are accessed in the parallel sum reduction below they
    # do not influence the result (adding 0.0 does not change the sum).
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[global_i] = exp_val
    barrier()

    # Parallel reduction for sum similar to reduction we saw before
    stride = SOFTMAX_BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride = stride // 2

    var block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < input_size:
        output_lt[global_i] = exp_val / block_sum


# CPU implementation for vector attention
def attention_cpu_kernel[
    seq_len: Int,
    d: Int,
    OutLayout: TensorLayout,
    QLayout: TensorLayout,
    KLayout: TensorLayout,
    VLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    q: TileTensor[mut=True, dtype, QLayout, MutAnyOrigin],
    k: TileTensor[mut=True, dtype, KLayout, MutAnyOrigin],
    v: TileTensor[mut=True, dtype, VLayout, MutAnyOrigin],
):
    """CPU implementation of vector attention."""
    var output_lt = output.to_layout_tensor()
    var q_lt = q.to_layout_tensor()
    var k_lt = k.to_layout_tensor()
    var v_lt = v.to_layout_tensor()
    var scores = List[Float32]()
    var weights = List[Float32]()
    for _ in range(seq_len):
        scores.append(0.0)
        weights.append(0.0)

    # Compute attention scores: Q · K[i] for each row i of K
    for i in range(seq_len):
        var score: Float32 = 0.0
        for dim in range(d):
            score = score + rebind[Float32](q_lt[dim]) * rebind[Float32](
                k_lt[i, dim]
            )
        scores[i] = score

    var max_score: Float32 = scores[0]
    for i in range(1, seq_len):
        if scores[i] > max_score:
            max_score = scores[i]

    var sum_exp: Float32 = 0.0
    for i in range(seq_len):
        weights[i] = exp(scores[i] - max_score)
        sum_exp = sum_exp + weights[i]

    for i in range(seq_len):
        weights[i] = weights[i] / sum_exp

    for dim in range(d):
        var weighted_sum: Float32 = 0.0
        for i in range(seq_len):
            weighted_sum = weighted_sum + weights[i] * rebind[Float32](
                v_lt[i, dim]
            )
        output_lt[dim] = rebind[Scalar[dtype]](weighted_sum)


@compiler.register("attention")
struct AttentionCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        seq_len: Int,
        d: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[
            dtype=dtype, rank=1, static_spec=_
        ],  # Output vector (d,)
        q: InputTensor[dtype=dtype, rank=1, static_spec=_],  # Query vector (d,)
        k: InputTensor[
            dtype=dtype, rank=2, static_spec=_
        ],  # Key matrix (seq_len, d)
        v: InputTensor[
            dtype=dtype, rank=2, static_spec=_
        ],  # Value matrix (seq_len, d)
        ctx: DeviceContextPtr,
    ) raises:
        # Define layouts
        comptime layout_q = row_major[d]()
        comptime layout_k = row_major[seq_len, d]()
        comptime layout_v = row_major[seq_len, d]()
        comptime layout_out = row_major[d]()
        comptime layout_scores = row_major[seq_len]()
        comptime QLayout = type_of(layout_q)
        comptime KLayout = type_of(layout_k)
        comptime VLayout = type_of(layout_v)
        comptime OutLayout = type_of(layout_out)
        comptime ScoresLayout = type_of(layout_scores)

        # Convert to layout tensors
        var output_tensor = TileTensor[
            mut=True, dtype, OutLayout, MutAnyOrigin
        ](output.unsafe_ptr(), layout_out)
        var q_tensor = TileTensor[mut=True, dtype, QLayout, MutAnyOrigin](
            q.unsafe_ptr(), layout_q
        )
        var k_tensor = TileTensor[mut=True, dtype, KLayout, MutAnyOrigin](
            k.unsafe_ptr(), layout_k
        )
        var v_tensor = TileTensor[mut=True, dtype, VLayout, MutAnyOrigin](
            v.unsafe_ptr(), layout_v
        )

        comptime if target == "gpu":
            var gpu_ctx = rebind[DeviceContext](ctx[])

            # Define layouts for matrix multiplication
            # Q reshaped to (1, d)
            comptime layout_q_2d = row_major[1, d]()
            comptime Q2DLayout = type_of(layout_q_2d)
            # K^T is (d, seq_len)
            comptime layout_k_t = row_major[d, seq_len]()
            comptime KTLayout = type_of(layout_k_t)
            # Scores as (1, seq_len)
            comptime layout_scores_2d = row_major[1, seq_len]()
            comptime Scores2DLayout = type_of(layout_scores_2d)
            # Weights as (1, seq_len)
            comptime layout_weights_2d = row_major[1, seq_len]()
            comptime Weights2DLayout = type_of(layout_weights_2d)
            # Result as (1, d)
            comptime layout_result_2d = row_major[1, d]()
            comptime Result2DLayout = type_of(layout_result_2d)

            # Transpose implementation limited to square (TRANSPOSE_BLOCK_DIM_XY x TRANSPOSE_BLOCK_DIM_XY) thread blocks
            comptime transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            # Tile over the K (seq_len, d) matrix
            comptime transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
            )
            # Matmul implementation limited to square (MATMUL_BLOCK_DIM_XY x MATMUL_BLOCK_DIM_XY) thread blocks
            comptime matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            # seq_len outputs ( Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len) ) with one thread per output
            comptime scores_blocks_per_grid = (
                seq_len + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY
            comptime softmax_threads = SOFTMAX_BLOCK_DIM_X
            comptime softmax_blocks_per_grid = 1
            # d outputs ( weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d) ) with one thread per output
            comptime result_blocks_per_grid = (
                d + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY

            # Allocate minimal temporary buffers - reuse same buffer for different shapes
            var k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len * d
            )  # K^T as (d, seq_len)
            var scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len
            )  # Reused for scores and weights

            var k_t = TileTensor(k_t_buf, layout_k_t)

            # ANCHOR: attention_orchestration_solution

            # Step 1: Reshape Q from (d,) to (1, d) - no buffer needed
            var q_2d = q_tensor.reshape(layout_q_2d)

            # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)\
            comptime kernel = transpose_kernel[
                seq_len, d, KTLayout, KLayout, dtype
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=transpose_threads_per_block,
            )

            # Step 3: Compute attention scores using matmul: Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len)
            # This computes Q · K^T[i] = Q · K[i] for each column i of K^T (which is row i of K)
            # Reuse scores_weights_buf as (1, seq_len) for scores
            var scores_2d = TileTensor(scores_weights_buf, layout_scores_2d)
            comptime kernel2 = matmul_idiomatic_tiled[
                1,
                seq_len,
                d,
                Scores2DLayout,
                Q2DLayout,
                KTLayout,
                dtype,
            ]
            gpu_ctx.enqueue_function[kernel2, kernel2](
                scores_2d,
                q_2d,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # Step 4: Reshape scores from (1, seq_len) to (seq_len,) for softmax
            var weights = scores_2d.reshape(layout_scores)

            # Step 5: Apply softmax to get attention weights (in-place)
            comptime ScoresLayout = type_of(layout_scores)
            comptime kernel3 = softmax_gpu_kernel[seq_len, ScoresLayout, dtype]
            # Create two TileTensor views from the underlying buffer to avoid aliasing error
            var weights_out = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            var weights_in = TileTensor[
                mut=True, dtype, ScoresLayout, MutAnyOrigin
            ](scores_weights_buf, layout_scores)
            gpu_ctx.enqueue_function[kernel3, kernel3](
                weights_out,
                weights_in,
                grid_dim=softmax_blocks_per_grid,
                block_dim=softmax_threads,
            )

            # Step 6: Reshape weights from (seq_len,) to (1, seq_len) for final matmul
            var weights_2d = weights.reshape(layout_weights_2d)

            # Step 7: Compute final result using matmul: weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d)
            # Reuse out_tensor reshaped as (1, d) for result
            var result_2d = output_tensor.reshape(layout_result_2d)
            comptime kernel4 = matmul_idiomatic_tiled[
                1,
                d,
                seq_len,
                Result2DLayout,
                Weights2DLayout,
                VLayout,
                dtype,
            ]
            gpu_ctx.enqueue_function[kernel4, kernel4](
                result_2d,
                weights_2d,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # ANCHOR_END: attention_orchestration_solution

        elif target == "cpu":
            attention_cpu_kernel[
                seq_len, d, OutLayout, QLayout, KLayout, VLayout, dtype
            ](output_tensor, q_tensor, k_tensor, v_tensor)

        else:
            raise Error("Unsupported target: " + target)
