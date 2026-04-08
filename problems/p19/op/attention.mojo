from std.memory import UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from layout.layout_tensor import copy_dram_to_sram_async
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
    a: TileTensor[mut=False, dtype, ALayout, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, BLayout, MutAnyOrigin],
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

    comptime load_a_layout = row_major[
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    ]()  # Coalesced loading
    comptime load_b_layout = row_major[
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    ]()  # Coalesced loading

    comptime for idx in range(
        (inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY
    ):
        # Get tiles from A and B matrices
        var a_tile = a.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
            block_idx.y, idx
        )
        var b_tile = b.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
            idx, block_idx.x
        )

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async[
            thread_layout=load_a_layout,
            num_threads=MATMUL_NUM_THREADS,
            block_dim_count=MATMUL_BLOCK_DIM_COUNT,
        ](a_shared, a_tile)
        copy_dram_to_sram_async[
            thread_layout=load_b_layout,
            num_threads=MATMUL_NUM_THREADS,
            block_dim_count=MATMUL_BLOCK_DIM_COUNT,
        ](b_shared, b_tile)

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        comptime for k in range(MATMUL_BLOCK_DIM_XY):
            if (
                tiled_row < rows and tiled_col < cols
            ):  # Only perform calculation for valid outputs
                if k < a_tile.dim(
                    1
                ):  # Only perform calculation on valid inputs
                    acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result with bounds checking (needed for attention's variable sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc


# ANCHOR: transpose_kernel
def transpose_kernel[
    rows: Int,
    cols: Int,
    OutLayout: TensorLayout,
    InLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    inp: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
):
    # FILL ME IN (roughly 18 lines)
    ...


# ANCHOR_END: transpose_kernel


# Apply softmax to attention scores taken from p16
def softmax_gpu_kernel[
    input_size: Int,
    LayoutType: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, LayoutType, MutAnyOrigin],
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

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[global_i])
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
        output[global_i] = exp_val / block_sum


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
    q: TileTensor[mut=False, dtype, QLayout, MutAnyOrigin],
    k: TileTensor[mut=False, dtype, KLayout, ImmutAnyOrigin],
    v: TileTensor[mut=False, dtype, VLayout, MutAnyOrigin],
):
    """CPU implementation of vector attention."""
    var scores = List[Float32]()
    var weights = List[Float32]()
    for _ in range(seq_len):
        scores.append(0.0)
        weights.append(0.0)

    # Compute attention scores: Q · K[i] for each row i of K
    for i in range(seq_len):
        var score: Float32 = 0.0
        for dim in range(d):
            score = score + rebind[Float32](q[dim]) * rebind[Float32](k[i, dim])
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
                v[i, dim]
            )
        output[dim] = rebind[Scalar[dtype]](weighted_sum)


@compiler.register("attention")
struct AttentionCustomOp:
    @staticmethod
    def execute[
        target: StaticString,  # "cpu" or "gpu"
        seq_len: Int,
        d: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1, static_spec=_],  # Output vector (d,)
        q: InputTensor[rank=1, static_spec=_],  # Query vector (d,)
        k: InputTensor[rank=2, static_spec=_],  # Key matrix (seq_len, d)
        v: InputTensor[rank=2, static_spec=_],  # Value matrix (seq_len, d)
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
        var output_tensor = rebind[
            TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin]
        ](output.to_layout_tensor())
        var q_tensor = rebind[
            TileTensor[mut=False, dtype, QLayout, MutAnyOrigin]
        ](q.to_layout_tensor())
        var k_tensor = rebind[
            TileTensor[mut=False, dtype, KLayout, ImmutAnyOrigin]
        ](k.to_layout_tensor())
        var v_tensor = rebind[
            TileTensor[mut=False, dtype, VLayout, MutAnyOrigin]
        ](v.to_layout_tensor())

        comptime if target == "gpu":
            # ANCHOR: attention_orchestration
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

            # Step 1: Reshape Q from (d,) to (1, d) - no buffer needed
            # FILL ME IN 1 line

            # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)
            # FILL ME IN 1 function call

            # Step 3: Compute attention scores using matmul: Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len)
            # This computes Q · K^T[i] = Q · K[i] for each column i of K^T (which is row i of K)
            # Reuse scores_weights_buf as (1, seq_len) for scores
            # FILL ME IN 2 lines

            # Step 4: Reshape scores from (1, seq_len) to (seq_len,) for softmax
            # FILL ME IN 1 line

            # Step 5: Apply softmax to get attention weights
            # FILL ME IN 1 function call

            # Step 6: Reshape weights from (seq_len,) to (1, seq_len) for final matmul
            # FILL ME IN 1 line

            # Step 7: Compute final result using matmul: weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d)
            # Reuse out_tensor reshaped as (1, d) for result
            # FILL ME IN 2 lines

            # ANCHOR_END: attention_orchestration

        elif target == "cpu":
            attention_cpu_kernel[seq_len, d, dtype](
                output_tensor, q_tensor, k_tensor, v_tensor
            )

        else:
            raise Error("Unsupported target: " + target)
