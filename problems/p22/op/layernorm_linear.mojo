from math import sqrt
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace, async_copy_wait_all
from os.atomic import Atomic
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils import StaticTuple

alias MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
alias MATMUL_NUM_THREADS = MATMUL_BLOCK_DIM_XY * MATMUL_BLOCK_DIM_XY
alias MATMUL_BLOCK_DIM_COUNT = 2
alias TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output
alias TPB = 16
alias dtype = DType.float32


# ANCHOR: matmul_idiomatic_tiled
# Idiomatic tiled matmul from p19.mojo
fn matmul_idiomatic_tiled[
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
):
    """Idiomatic tiled matrix multiplication from p19."""
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

    # Write final result with bounds checking (needed for variable matrix sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled


# ANCHOR: layernorm_kernel
fn layernorm_kernel[
    input_layout: Layout,
    ln_params_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, input_layout, ImmutAnyOrigin],
    ln_weight: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
    ln_bias: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
):
    batch_idx = block_idx.x
    seq_idx = block_idx.y
    hidden_idx = thread_idx.x

    if (
        batch_idx >= batch_size
        or seq_idx >= seq_len
        or hidden_idx >= hidden_dim
    ):
        return

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    # FILL ME IN (roughly 11 lines)


# ANCHOR_END: layernorm_kernel


# ANCHOR: transpose_kernel
fn transpose_kernel[
    layout_in: Layout,
    layout_out: Layout,
    rows: Int,
    cols: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutAnyOrigin],
    inp: LayoutTensor[dtype, layout_in, ImmutAnyOrigin],
):
    """Transpose matrix using shared memory tiling for coalesced access.
    We will learn more about coalesced access in the next part.
    """
    shared_tile = LayoutTensor[
        dtype,
        Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    local_row = thread_idx.y
    local_col = thread_idx.x

    global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row
    global_col = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_col

    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = inp[global_row, global_col]

    barrier()

    out_row = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_row
    out_col = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]


# ANCHOR_END: transpose_kernel


# ANCHOR: add_bias_kernel
fn add_bias_kernel[
    input_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    output_dim: Int,
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, input_layout, MutAnyOrigin],
    bias: LayoutTensor[dtype, bias_layout, ImmutAnyOrigin],
):
    """Simple bias addition."""
    batch_idx = block_idx.x
    seq_idx = block_idx.y
    out_idx = thread_idx.x

    if batch_idx >= batch_size or seq_idx >= seq_len or out_idx >= output_dim:
        return

    output[batch_idx, seq_idx, out_idx] = input[
        batch_idx, seq_idx, out_idx
    ] + rebind[Scalar[dtype]](bias[out_idx])


# ANCHOR_END: add_bias_kernel


# ANCHOR: minimal_fused_forward_kernel
fn minimal_fused_kernel[
    input_layout: Layout,
    ln_params_layout: Layout,
    weight_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
](
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, input_layout, ImmutAnyOrigin],
    ln_weight: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
    ln_bias: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
    linear_weight: LayoutTensor[dtype, weight_layout, ImmutAnyOrigin],
    linear_bias: LayoutTensor[dtype, bias_layout, ImmutAnyOrigin],
):
    """Minimal fused kernel - one thread per sequence position to avoid redundancy.
    """
    # Grid: (batch_size, seq_len) - one thread block per sequence position
    # Block: (1,) - single thread per sequence position to avoid redundant computation
    batch_idx = block_idx.x
    seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Step 1: Compute LayerNorm statistics once per sequence position

    # FILL IN roughly 10 lines

    # Step 2: Compute all outputs for this sequence position

    # FILL IN roughly 10 lines


# ANCHOR_END: minimal_fused_forward_kernel


# ANCHOR: minimal_fused_backward_kernel
fn minimal_fused_kernel_backward[
    grad_output_layout: Layout,
    input_layout: Layout,
    ln_params_layout: Layout,
    weight_layout: Layout,
    grad_input_layout: Layout,
    grad_ln_weight_layout: Layout,
    grad_ln_bias_layout: Layout,
    grad_weight_layout: Layout,
    grad_bias_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
](
    grad_input: LayoutTensor[dtype, grad_input_layout, MutAnyOrigin],
    grad_ln_weight: LayoutTensor[dtype, grad_ln_weight_layout, MutAnyOrigin],
    grad_ln_bias: LayoutTensor[dtype, grad_ln_bias_layout, MutAnyOrigin],
    grad_weight: LayoutTensor[dtype, grad_weight_layout, MutAnyOrigin],
    grad_bias: LayoutTensor[dtype, grad_bias_layout, MutAnyOrigin],
    grad_output: LayoutTensor[dtype, grad_output_layout, ImmutAnyOrigin],
    input: LayoutTensor[dtype, input_layout, ImmutAnyOrigin],
    ln_weight: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
    ln_bias: LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin],
    linear_weight: LayoutTensor[dtype, weight_layout, ImmutAnyOrigin],
):
    """Fused backward kernel using atomic operations for safe gradient accumulation.
    """
    # Grid: (batch_size, seq_len) - one thread per sequence position
    # Block: (1,) - single thread per sequence position
    batch_idx = block_idx.x
    seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Step 1: Recompute forward pass statistics (needed for gradients)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    # FILL IN roughly 8 lines

    # Step 2: Atomically accumulate gradients w.r.t. linear bias

    # FILL IN roughly 4 lines

    # Step 3: Atomically accumulate gradients w.r.t. linear weight
    # Make sure to use the correct atomic operation to avoid race conditions

    # FILL IN roughly 10 lines

    # Step 4: Atomically accumulate gradients w.r.t. LayerNorm parameters

    # FILL IN roughly 10 lines

    # Step 5: Compute gradients w.r.t. input (LayerNorm backward)
    # Compute sum terms needed for LayerNorm backward
    # Make sure to use the correct atomic operation to avoid race conditions

    # FILL IN roughly 12 lines

    # Compute actual input gradients (no race conditions here - each thread writes to different positions)

    # FILL IN roughly 10 lines


# ANCHOR_END: minimal_fused_backward_kernel


@compiler.register("layernorm_linear")
struct LayerNormLinearCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        algorithm: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
    ](
        output: OutputTensor[dtype = DType.float32, rank=3],
        input: InputTensor[dtype = DType.float32, rank=3],
        ln_weight: InputTensor[dtype = DType.float32, rank=1],
        ln_bias: InputTensor[dtype = DType.float32, rank=1],
        linear_weight: InputTensor[dtype = DType.float32, rank=2],
        linear_bias: InputTensor[dtype = DType.float32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        alias input_layout = input.static_spec.to_layout()
        alias ln_params_layout = ln_weight.static_spec.to_layout()
        alias weight_layout = linear_weight.static_spec.to_layout()
        alias bias_layout = linear_bias.static_spec.to_layout()
        alias output_layout = output.static_spec.to_layout()

        # Note: rebind is necessary now but it shouldn't be!
        output_tensor = rebind[
            LayoutTensor[dtype, output_layout, MutAnyOrigin]
        ](output.to_layout_tensor())
        input_tensor = rebind[
            LayoutTensor[dtype, input_layout, ImmutAnyOrigin]
        ](input.to_layout_tensor())
        ln_weight_tensor = rebind[
            LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin]
        ](ln_weight.to_layout_tensor())
        ln_bias_tensor = rebind[
            LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin]
        ](ln_bias.to_layout_tensor())
        linear_weight_tensor = rebind[
            LayoutTensor[dtype, weight_layout, ImmutAnyOrigin]
        ](linear_weight.to_layout_tensor())
        linear_bias_tensor = rebind[
            LayoutTensor[dtype, bias_layout, ImmutAnyOrigin]
        ](linear_bias.to_layout_tensor())

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # ANCHOR: layernorm_linear_custom_op
            @parameter
            if algorithm == "fused":
                # fused case - one thread per sequence position
                alias kernel = minimal_fused_kernel[
                    input_layout,
                    ln_params_layout,
                    weight_layout,
                    bias_layout,
                    output_layout,
                    batch_size,
                    seq_len,
                    hidden_dim,
                    output_dim,
                ]
                gpu_ctx.enqueue_function_checked[kernel, kernel](
                    output_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    linear_weight_tensor,
                    linear_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(1,),
                )
            elif algorithm == "unfused":
                # unfused case
                # Create intermediate normalized tensor
                normalized_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * hidden_dim
                )
                normalized_tensor = LayoutTensor[
                    dtype, input_layout, MutAnyOrigin
                ](normalized_buffer)

                # Step 1: LayerNorm kernel
                alias kernel = layernorm_kernel[
                    input_layout,
                    ln_params_layout,
                    input_layout,
                    batch_size,
                    seq_len,
                    hidden_dim,
                ]
                gpu_ctx.enqueue_function_checked[kernel, kernel](
                    normalized_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(min(hidden_dim, TPB),),
                )

                # Step 2: Matmul on normalized data
                total_rows = batch_size * seq_len
                blocks_x = (total_rows + TPB - 1) // TPB
                blocks_y = (output_dim + TPB - 1) // TPB

                # Create intermediate result without bias
                matmul_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * output_dim
                )
                matmul_tensor = LayoutTensor[
                    dtype, output_layout, MutAnyOrigin
                ](matmul_buffer)

                # Create transposed weight matrix: [output_dim, hidden_dim] -> [hidden_dim, output_dim]
                transposed_weight_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    hidden_dim * output_dim
                )
                transposed_weight_tensor = LayoutTensor[
                    dtype,
                    Layout.row_major(hidden_dim, output_dim),
                    MutAnyOrigin,
                ](transposed_weight_buffer)

                # Transpose the weight matrix
                transpose_blocks_x = (
                    hidden_dim + TRANSPOSE_BLOCK_DIM_XY - 1
                ) // TRANSPOSE_BLOCK_DIM_XY
                transpose_blocks_y = (
                    output_dim + TRANSPOSE_BLOCK_DIM_XY - 1
                ) // TRANSPOSE_BLOCK_DIM_XY
                alias kernel2 = transpose_kernel[
                    weight_layout,
                    transposed_weight_tensor.layout,
                    output_dim,
                    hidden_dim,
                ]
                gpu_ctx.enqueue_function_checked[kernel2, kernel2](
                    transposed_weight_tensor,
                    linear_weight_tensor,
                    grid_dim=(transpose_blocks_x, transpose_blocks_y),
                    block_dim=(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
                )

                # Reshape tensors for matmul: [batch*seq, hidden] @ [hidden, output] -> [batch*seq, output]
                flat_normalized = normalized_tensor.reshape[
                    Layout.row_major(batch_size * seq_len, hidden_dim)
                ]()
                flat_matmul = matmul_tensor.reshape[
                    Layout.row_major(batch_size * seq_len, output_dim)
                ]()

                alias kernel3 = matmul_idiomatic_tiled[
                    flat_normalized.layout,
                    transposed_weight_tensor.layout,
                    flat_matmul.layout,
                    batch_size * seq_len,
                    output_dim,
                    hidden_dim,
                ]
                gpu_ctx.enqueue_function_checked[kernel3, kernel3](
                    flat_matmul,
                    flat_normalized,
                    transposed_weight_tensor,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(TPB, TPB),
                )

                # Step 3: Add bias - reshape matmul result back to 3D for bias addition
                reshaped_matmul = matmul_tensor.reshape[
                    Layout.row_major(batch_size, seq_len, output_dim)
                ]()

                alias kernel4 = add_bias_kernel[
                    reshaped_matmul.layout,
                    bias_layout,
                    output_layout,
                    batch_size,
                    seq_len,
                    output_dim,
                ]
                gpu_ctx.enqueue_function_checked[kernel4, kernel4](
                    output_tensor,
                    reshaped_matmul,
                    linear_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(min(output_dim, TPB),),
                )
            # ANCHOR_END: layernorm_linear_custom_op

        elif target == "cpu":
            # CPU implementation - always fused (no separate kernels for CPU)
            # Note: CPU doesn't have separate fused vs unfused - both use the same implementation
            for batch in range(batch_size):
                for seq in range(seq_len):
                    # LayerNorm
                    var sum_val: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        sum_val += rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                    mean_val = sum_val / hidden_dim

                    var var_sum: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        diff = input_tensor[batch, seq, h] - mean_val
                        var_sum += rebind[Scalar[dtype]](diff * diff)
                    var_val = var_sum / hidden_dim
                    inv_std = 1.0 / sqrt(var_val + 1e-5)

                    # Apply LayerNorm and Linear in one step (truly fused)
                    for out_idx in range(output_dim):
                        var acc: Scalar[dtype] = 0
                        for h in range(hidden_dim):
                            input_val = input_tensor[batch, seq, h]
                            normalized = (
                                input_val - mean_val
                            ) * inv_std * ln_weight_tensor[h] + ln_bias_tensor[
                                h
                            ]
                            acc += rebind[Scalar[dtype]](
                                normalized * linear_weight_tensor[out_idx, h]
                            )
                        output_tensor[batch, seq, out_idx] = (
                            acc + linear_bias_tensor[out_idx]
                        )

        else:
            raise Error("Unsupported target: " + target)


# ANCHOR: layernorm_linear_backward_custom_op
@compiler.register("layernorm_linear_backward")
struct LayerNormLinearBackwardCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
    ](
        grad_input: OutputTensor[dtype = DType.float32, rank=3],
        grad_ln_weight: OutputTensor[dtype = DType.float32, rank=1],
        grad_ln_bias: OutputTensor[dtype = DType.float32, rank=1],
        grad_weight: OutputTensor[dtype = DType.float32, rank=2],
        grad_bias: OutputTensor[dtype = DType.float32, rank=1],
        grad_output: InputTensor[dtype = DType.float32, rank=3],
        input: InputTensor[dtype = DType.float32, rank=3],
        ln_weight: InputTensor[dtype = DType.float32, rank=1],
        ln_bias: InputTensor[dtype = DType.float32, rank=1],
        linear_weight: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        alias grad_output_layout = grad_output.static_spec.to_layout()
        alias input_layout = input.static_spec.to_layout()
        alias ln_params_layout = ln_weight.static_spec.to_layout()
        alias weight_layout = linear_weight.static_spec.to_layout()
        alias grad_input_layout = grad_input.static_spec.to_layout()
        alias grad_ln_weight_layout = grad_ln_weight.static_spec.to_layout()
        alias grad_ln_bias_layout = grad_ln_bias.static_spec.to_layout()
        alias grad_weight_layout = grad_weight.static_spec.to_layout()
        alias grad_bias_layout = grad_bias.static_spec.to_layout()

        grad_input_tensor = rebind[
            LayoutTensor[dtype, grad_input_layout, MutAnyOrigin]
        ](grad_input.to_layout_tensor())
        grad_ln_weight_tensor = rebind[
            LayoutTensor[dtype, grad_ln_weight_layout, MutAnyOrigin]
        ](grad_ln_weight.to_layout_tensor())
        grad_ln_bias_tensor = rebind[
            LayoutTensor[dtype, grad_ln_bias_layout, MutAnyOrigin]
        ](grad_ln_bias.to_layout_tensor())
        grad_weight_tensor = rebind[
            LayoutTensor[dtype, grad_weight_layout, MutAnyOrigin]
        ](grad_weight.to_layout_tensor())
        grad_bias_tensor = rebind[
            LayoutTensor[dtype, grad_bias_layout, MutAnyOrigin]
        ](grad_bias.to_layout_tensor())
        grad_output_tensor = rebind[
            LayoutTensor[dtype, grad_output_layout, ImmutAnyOrigin]
        ](grad_output.to_layout_tensor())
        input_tensor = rebind[
            LayoutTensor[dtype, input_layout, ImmutAnyOrigin]
        ](input.to_layout_tensor())
        ln_weight_tensor = rebind[
            LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin]
        ](ln_weight.to_layout_tensor())
        ln_bias_tensor = rebind[
            LayoutTensor[dtype, ln_params_layout, ImmutAnyOrigin]
        ](ln_bias.to_layout_tensor())
        linear_weight_tensor = rebind[
            LayoutTensor[dtype, weight_layout, ImmutAnyOrigin]
        ](linear_weight.to_layout_tensor())

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Launch backward kernel
            alias kernel = minimal_fused_kernel_backward[
                grad_output_layout,
                input_layout,
                ln_params_layout,
                weight_layout,
                grad_input_layout,
                grad_ln_weight_layout,
                grad_ln_bias_layout,
                grad_weight_layout,
                grad_bias_layout,
                batch_size,
                seq_len,
                hidden_dim,
                output_dim,
            ]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                grad_input_tensor,
                grad_ln_weight_tensor,
                grad_ln_bias_tensor,
                grad_weight_tensor,
                grad_bias_tensor,
                grad_output_tensor,
                input_tensor,
                ln_weight_tensor,
                ln_bias_tensor,
                linear_weight_tensor,
                grid_dim=(batch_size, seq_len),
                block_dim=(1,),
            )

            # Note: Parameter gradients (ln_weight, ln_bias, linear_weight, bias) are not computed in this kernel
            # This is a simplified version that only computes input gradients to avoid race conditions

        elif target == "cpu":
            # CPU implementation - same logic as GPU but in CPU loops
            # Initialize gradients to zero
            for batch in range(batch_size):
                for seq in range(seq_len):
                    for h in range(hidden_dim):
                        grad_input_tensor[batch, seq, h] = 0.0

            for h in range(hidden_dim):
                grad_ln_weight_tensor[h] = 0.0
                grad_ln_bias_tensor[h] = 0.0

            for out_idx in range(output_dim):
                grad_bias_tensor[out_idx] = 0.0
                for h in range(hidden_dim):
                    grad_weight_tensor[out_idx, h] = 0.0

            # Compute gradients - same algorithm as GPU kernel
            for batch in range(batch_size):
                for seq in range(seq_len):
                    # Recompute forward pass statistics
                    var sum_val: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        sum_val += rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                    mean_val = sum_val / hidden_dim

                    var var_sum: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        diff = input_tensor[batch, seq, h] - mean_val
                        var_sum += rebind[Scalar[dtype]](diff * diff)
                    var_val = var_sum / hidden_dim
                    inv_std = 1.0 / sqrt(var_val + 1e-5)

                    # Gradient w.r.t. linear bias
                    for out_idx in range(output_dim):
                        grad_bias_tensor[out_idx] = (
                            grad_bias_tensor[out_idx]
                            + grad_output_tensor[batch, seq, out_idx]
                        )

                    # Gradient w.r.t. linear weight
                    for out_idx in range(output_dim):
                        for h in range(hidden_dim):
                            input_val = rebind[Scalar[dtype]](
                                input_tensor[batch, seq, h]
                            )
                            normalized = (input_val - mean_val) * inv_std
                            ln_output_val = (
                                normalized * ln_weight_tensor[h]
                                + ln_bias_tensor[h]
                            )
                            grad_weight_tensor[out_idx, h] = (
                                grad_weight_tensor[out_idx, h]
                                + grad_output_tensor[batch, seq, out_idx]
                                * ln_output_val
                            )

                    # Gradient w.r.t. LayerNorm parameters
                    for h in range(hidden_dim):
                        input_val = rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                        normalized = (input_val - mean_val) * inv_std

                        var grad_ln_out: Scalar[dtype] = 0
                        for out_idx in range(output_dim):
                            grad_ln_out = grad_ln_out + rebind[Scalar[dtype]](
                                grad_output_tensor[batch, seq, out_idx]
                                * linear_weight_tensor[out_idx, h]
                            )

                        grad_ln_weight_tensor[h] = grad_ln_weight_tensor[
                            h
                        ] + rebind[Scalar[dtype]](grad_ln_out * normalized)
                        grad_ln_bias_tensor[h] = grad_ln_bias_tensor[
                            h
                        ] + rebind[Scalar[dtype]](grad_ln_out)

                    # Gradient w.r.t. input (LayerNorm backward)
                    var sum_grad_normalized: Scalar[dtype] = 0
                    var sum_grad_normalized_times_normalized: Scalar[dtype] = 0

                    for h in range(hidden_dim):
                        input_val = rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                        normalized = (input_val - mean_val) * inv_std

                        var grad_ln_out: Scalar[dtype] = 0
                        for out_idx in range(output_dim):
                            grad_ln_out = grad_ln_out + rebind[Scalar[dtype]](
                                grad_output_tensor[batch, seq, out_idx]
                                * linear_weight_tensor[out_idx, h]
                            )

                        grad_norm = grad_ln_out * ln_weight_tensor[h]
                        sum_grad_normalized = sum_grad_normalized + rebind[
                            Scalar[dtype]
                        ](grad_norm)
                        sum_grad_normalized_times_normalized = (
                            sum_grad_normalized_times_normalized
                            + rebind[Scalar[dtype]](grad_norm * normalized)
                        )

                    for h in range(hidden_dim):
                        input_val = rebind[Scalar[dtype]](
                            input_tensor[batch, seq, h]
                        )
                        normalized = (input_val - mean_val) * inv_std

                        var grad_ln_out: Scalar[dtype] = 0
                        for out_idx in range(output_dim):
                            grad_ln_out = grad_ln_out + rebind[Scalar[dtype]](
                                grad_output_tensor[batch, seq, out_idx]
                                * linear_weight_tensor[out_idx, h]
                            )

                        grad_norm = grad_ln_out * ln_weight_tensor[h]
                        grad_input_tensor[batch, seq, h] = inv_std * (
                            grad_norm
                            - (sum_grad_normalized / hidden_dim)
                            - (
                                normalized
                                * sum_grad_normalized_times_normalized
                                / hidden_dim
                            )
                        )

        else:
            raise Error("Unsupported target: " + target)


# ANCHOR_END: layernorm_linear_backward_custom_op
