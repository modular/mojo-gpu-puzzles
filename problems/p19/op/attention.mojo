from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace, async_copy_wait_all
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

alias SEQ_LEN = 16  # This must be equal to SEQ_LEN in p19.py
alias D = 16  # This must be equal to D in p19.py

alias TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output
alias MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
alias MATMUL_NUM_THREADS = MATMUL_BLOCK_DIM_XY * MATMUL_BLOCK_DIM_XY
alias MATMUL_BLOCK_DIM_COUNT = 2
alias SOFTMAX_BLOCK_DIM_X = 1 << log2_ceil(SEQ_LEN)


# Tiled matrix multiplication (from p16), updated to:
# 1) Support different layouts for input (a, b) and output LayoutTensors.
# 2) Handle cases where the inner dimension is not a multiple of MATMUL_BLOCK_DIM_XY.
# 3) Explicitly check for out-of-bounds elements.
# The approach still tiles all three LayoutTensors (a, b, and output) into identical square tiles
# of size (MATMUL_BLOCK_DIM_XY x MATMUL_BLOCK_DIM_XY) with each thread loading one element
# from a and b, and writing one element to output.
fn matmul_idiomatic_tiled[
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, b_layout, MutAnyOrigin],
):
    """Updated idiomatic tiled matrix multiplication from p16."""
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * MATMUL_BLOCK_DIM_XY + local_row
    tiled_col = block_idx.x * MATMUL_BLOCK_DIM_XY + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
        block_idx.y, block_idx.x
    )
    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var acc: output.element_type = 0

    alias load_a_layout = Layout.row_major(
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    )  # Coalesced loading
    alias load_b_layout = Layout.row_major(
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    )  # Coalesced loading

    @parameter
    for idx in range((inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY):
        # Get tiles from A and B matrices
        a_tile = a.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
            block_idx.y, idx
        )
        b_tile = b.tile[MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY](
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
        @parameter
        for k in range(MATMUL_BLOCK_DIM_XY):
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
fn transpose_kernel[
    layout_in: Layout,  # Layout for input matrix (seq_len, d)
    layout_out: Layout,  # Layout for output matrix (d, seq_len)
    rows: Int,
    cols: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout_out, MutAnyOrigin],
    inp: LayoutTensor[mut=False, dtype, layout_in, MutAnyOrigin],
):
    # FILL ME IN (roughly 18 lines)
    ...


# ANCHOR_END: transpose_kernel


# Apply softmax to attention scores taken from p16
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(SOFTMAX_BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(SOFTMAX_BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = thread_idx.x

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[global_i])
    shared_max[global_i] = val

    barrier()

    # Parallel reduction to find max similar to reduction we saw before
    stride = SOFTMAX_BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    block_max = shared_max[0]

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

    block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < input_size:
        output[global_i] = exp_val / block_sum


# CPU implementation for vector attention
fn attention_cpu_kernel[
    layout_q: Layout,
    layout_k: Layout,
    layout_v: Layout,
    layout_out: Layout,
    seq_len: Int,
    d: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutAnyOrigin],
    q: LayoutTensor[dtype, layout_q, MutAnyOrigin],
    k: LayoutTensor[dtype, layout_k, MutAnyOrigin],
    v: LayoutTensor[dtype, layout_v, MutAnyOrigin],
):
    """CPU implementation of vector attention."""
    var scores = List[Float32]()
    var weights = List[Float32]()
    for _ in range(seq_len):
        scores.append(0.0)
        weights.append(0.0)

    # CPU: Compute attention scores K[i] · Q directly for each row i of K
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
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        seq_len: Int,
        d: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],  # Output vector (d,)
        q: InputTensor[rank=1],  # Query vector (d,)
        k: InputTensor[rank=2],  # Key matrix (seq_len, d)
        v: InputTensor[rank=2],  # Value matrix (seq_len, d)
        ctx: DeviceContextPtr,
    ) raises:
        # Define layouts
        alias layout_q = Layout.row_major(d)
        alias layout_k = Layout.row_major(seq_len, d)
        alias layout_v = Layout.row_major(seq_len, d)
        alias layout_out = Layout.row_major(d)
        alias layout_scores = Layout.row_major(seq_len)

        # Convert to layout tensors
        var output_tensor = rebind[
            LayoutTensor[dtype, layout_out, MutAnyOrigin]
        ](output.to_layout_tensor())
        var q_tensor = rebind[LayoutTensor[dtype, layout_q, MutAnyOrigin]](
            q.to_layout_tensor()
        )
        var k_tensor = rebind[LayoutTensor[dtype, layout_k, MutAnyOrigin]](
            k.to_layout_tensor()
        )
        var v_tensor = rebind[LayoutTensor[dtype, layout_v, MutAnyOrigin]](
            v.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            # ANCHOR: attention_orchestration
            var gpu_ctx = rebind[DeviceContext](ctx[])

            # Define layouts for matrix multiplication
            # Q reshaped to (1, d)
            alias layout_q_2d = Layout.row_major(1, d)
            # K^T is (d, seq_len)
            alias layout_k_t = Layout.row_major(d, seq_len)
            # Scores as (1, seq_len)
            alias layout_scores_2d = Layout.row_major(1, seq_len)
            # Weights as (1, seq_len)
            alias layout_weights_2d = Layout.row_major(1, seq_len)
            # Result as (1, d)
            alias layout_result_2d = Layout.row_major(1, d)

            # Transpose implementation limited to square (TRANSPOSE_BLOCK_DIM_XY x TRANSPOSE_BLOCK_DIM_XY) thread blocks
            alias transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            # Tile over the K (seq_len, d) matrix
            alias transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
            )
            # Matmul implementation limited to square (MATMUL_BLOCK_DIM_XY x MATMUL_BLOCK_DIM_XY) thread blocks
            alias matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            # seq_len outputs ( Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len) ) with one thread per output
            alias scores_blocks_per_grid = (
                seq_len + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY
            alias softmax_threads = SOFTMAX_BLOCK_DIM_X
            alias softmax_blocks_per_grid = 1
            # d outputs ( weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d) ) with one thread per output
            alias result_blocks_per_grid = (
                d + MATMUL_BLOCK_DIM_XY - 1
            ) // MATMUL_BLOCK_DIM_XY

            # Allocate minimal temporary buffers - reuse same buffer for different shapes
            k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len * d
            )  # K^T as (d, seq_len)
            scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len
            )  # Reused for scores and weights

            k_t = LayoutTensor[mut=True, dtype, layout_k_t, MutAnyOrigin](
                k_t_buf.unsafe_ptr()
            )

            # Step 1: Reshape Q from (d,) to (1, d) - no buffer needed
            # FILL ME IN 1 line

            # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)
            # FILL ME IN 1 function call

            # Step 3: Compute attention scores using matmul: Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len)
            # GPU: Uses matrix multiplication to compute all Q · K[i] scores in parallel
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
            attention_cpu_kernel[
                layout_q, layout_k, layout_v, layout_out, seq_len, d, dtype
            ](output_tensor, q_tensor, k_tensor, v_tensor)

        else:
            raise Error("Unsupported target: " + target)
