from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.atomic import Atomic
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils import StaticTuple

comptime MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
comptime MATMUL_NUM_THREADS = MATMUL_BLOCK_DIM_XY * MATMUL_BLOCK_DIM_XY
comptime MATMUL_BLOCK_DIM_COUNT = 2
comptime TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output


# ANCHOR: matmul_idiomatic_tiled
# Idiomatic tiled matmul from p19.mojo
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
    """Idiomatic tiled matrix multiplication from p19."""
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
        # Synchronously load tiles to shared memory - each thread loads one element
        var a_tile_row_start = block_idx.y * MATMUL_BLOCK_DIM_XY
        var a_tile_col_start = idx * MATMUL_BLOCK_DIM_XY
        var b_tile_row_start = idx * MATMUL_BLOCK_DIM_XY
        var b_tile_col_start = block_idx.x * MATMUL_BLOCK_DIM_XY

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

    # Write final result with bounds checking (needed for variable matrix sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile_lt[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled


# ANCHOR: layernorm_kernel_solution
def layernorm_kernel[
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    OutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    LnParamsLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, InputLayout, MutAnyOrigin],
    ln_weight: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
    ln_bias: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
):
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y
    var hidden_idx = thread_idx.x

    if (
        batch_idx >= batch_size
        or seq_idx >= seq_len
        or hidden_idx >= hidden_dim
    ):
        return

    var output_lt = output.to_layout_tensor()
    var input_lt = input.to_layout_tensor()
    var ln_weight_lt = ln_weight.to_layout_tensor()
    var ln_bias_lt = ln_bias.to_layout_tensor()

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    comptime for h in range(hidden_dim):
        var val = input_lt[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    var mean_val = sum_val / hidden_dim
    var var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    var inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Apply LayerNorm to this element
    var input_val = input_lt[batch_idx, seq_idx, hidden_idx]
    var normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
        ln_weight_lt[hidden_idx]
    ) + rebind[Scalar[dtype]](ln_bias_lt[hidden_idx])
    output_lt[batch_idx, seq_idx, hidden_idx] = normalized


# ANCHOR_END: layernorm_kernel_solution


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
    """Transpose matrix using shared memory tiling for coalesced access.
    We will learn more about coalesced access in the next part.
    """
    comptime shared_layout = row_major[
        TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY
    ]()
    var shared_tile = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](shared_layout)

    var local_row = thread_idx.y
    var local_col = thread_idx.x

    var inp_lt = inp.to_layout_tensor()
    var output_lt = output.to_layout_tensor()
    var shared_tile_lt = shared_tile.to_layout_tensor()

    var global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row
    var global_col = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_col

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


# ANCHOR: add_bias_kernel
def add_bias_kernel[
    batch_size: Int,
    seq_len: Int,
    output_dim: Int,
    OutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    BiasLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, InputLayout, MutAnyOrigin],
    bias: TileTensor[mut=True, dtype, BiasLayout, MutAnyOrigin],
):
    """Simple bias addition."""
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y
    var out_idx = thread_idx.x

    if batch_idx >= batch_size or seq_idx >= seq_len or out_idx >= output_dim:
        return

    var output_lt = output.to_layout_tensor()
    var input_lt = input.to_layout_tensor()
    var bias_lt = bias.to_layout_tensor()

    output_lt[batch_idx, seq_idx, out_idx] = input_lt[
        batch_idx, seq_idx, out_idx
    ] + rebind[Scalar[dtype]](bias_lt[out_idx])


# ANCHOR_END: add_bias_kernel


# ANCHOR: minimal_fused_forward_kernel_solution
def minimal_fused_kernel[
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
    OutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    LnParamsLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, InputLayout, MutAnyOrigin],
    ln_weight: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
    ln_bias: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
    linear_weight: TileTensor[mut=True, dtype, WeightLayout, MutAnyOrigin],
    linear_bias: TileTensor[mut=True, dtype, BiasLayout, MutAnyOrigin],
):
    """Minimal fused kernel - one thread per sequence position to avoid redundancy.
    """
    # Grid: (batch_size, seq_len) - one thread block per sequence position
    # Block: (1,) - single thread per sequence position to avoid redundant computation
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    var output_lt = output.to_layout_tensor()
    var input_lt = input.to_layout_tensor()
    var ln_weight_lt = ln_weight.to_layout_tensor()
    var ln_bias_lt = ln_bias.to_layout_tensor()
    var linear_weight_lt = linear_weight.to_layout_tensor()
    var linear_bias_lt = linear_bias.to_layout_tensor()

    # Step 1: Compute LayerNorm statistics once per sequence position
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    comptime for h in range(hidden_dim):
        var val = input_lt[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    var mean_val = sum_val / hidden_dim
    var var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    var inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Step 2: Compute all outputs for this sequence position
    comptime for out_idx in range(output_dim):
        var acc: Scalar[dtype] = 0

        comptime for h in range(hidden_dim):
            var input_val = input_lt[batch_idx, seq_idx, h]
            var normalized = (input_val - mean_val) * inv_std * rebind[
                Scalar[dtype]
            ](ln_weight_lt[h]) + rebind[Scalar[dtype]](ln_bias_lt[h])
            acc += rebind[Scalar[dtype]](
                normalized * linear_weight_lt[out_idx, h]
            )

        output_lt[batch_idx, seq_idx, out_idx] = acc + rebind[Scalar[dtype]](
            linear_bias_lt[out_idx]
        )


# ANCHOR_END: minimal_fused_forward_kernel_solution


# ANCHOR: minimal_fused_backward_kernel_solution
def minimal_fused_kernel_backward[
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
    GradInputLayout: TensorLayout,
    GradLnWeightLayout: TensorLayout,
    GradLnBiasLayout: TensorLayout,
    GradWeightLayout: TensorLayout,
    GradBiasLayout: TensorLayout,
    GradOutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    LnParamsLayout: TensorLayout,
    WeightLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    grad_input: TileTensor[mut=True, dtype, GradInputLayout, MutAnyOrigin],
    grad_ln_weight: TileTensor[
        mut=True, dtype, GradLnWeightLayout, MutAnyOrigin
    ],
    grad_ln_bias: TileTensor[mut=True, dtype, GradLnBiasLayout, MutAnyOrigin],
    grad_weight: TileTensor[mut=True, dtype, GradWeightLayout, MutAnyOrigin],
    grad_bias: TileTensor[mut=True, dtype, GradBiasLayout, MutAnyOrigin],
    grad_output: TileTensor[mut=True, dtype, GradOutputLayout, MutAnyOrigin],
    input: TileTensor[mut=True, dtype, InputLayout, MutAnyOrigin],
    ln_weight: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
    ln_bias: TileTensor[mut=True, dtype, LnParamsLayout, MutAnyOrigin],
    linear_weight: TileTensor[mut=True, dtype, WeightLayout, MutAnyOrigin],
):
    """Fused backward kernel using atomic operations for safe gradient accumulation.
    """
    # Grid: (batch_size, seq_len) - one thread per sequence position
    # Block: (1,) - single thread per sequence position
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    var grad_input_lt = grad_input.to_layout_tensor()
    var grad_ln_weight_lt = grad_ln_weight.to_layout_tensor()
    var grad_ln_bias_lt = grad_ln_bias.to_layout_tensor()
    var grad_weight_lt = grad_weight.to_layout_tensor()
    var grad_bias_lt = grad_bias.to_layout_tensor()
    var grad_output_lt = grad_output.to_layout_tensor()
    var input_lt = input.to_layout_tensor()
    var ln_weight_lt = ln_weight.to_layout_tensor()
    var ln_bias_lt = ln_bias.to_layout_tensor()
    var linear_weight_lt = linear_weight.to_layout_tensor()

    # Initialize gradient tensors to zero (block 0,0 only to avoid UB with atomic ops)
    if batch_idx == 0 and seq_idx == 0:
        # Initialize grad_ln_weight and grad_ln_bias
        comptime for h in range(hidden_dim):
            (grad_ln_weight.ptr + h).init_pointee_copy(0)
            (grad_ln_bias.ptr + h).init_pointee_copy(0)

        # Initialize grad_weight and grad_bias
        comptime for out_idx in range(output_dim):
            (grad_bias.ptr + out_idx).init_pointee_copy(0)

            comptime for h in range(hidden_dim):
                (grad_weight.ptr + out_idx * hidden_dim + h).init_pointee_copy(
                    0
                )

    # Note: We cannot use barrier() here as it only synchronizes within a block.
    # The atomic operations will handle synchronization across blocks.

    # Step 1: Recompute forward pass statistics (needed for gradients)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    comptime for h in range(hidden_dim):
        var val = input_lt[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    var mean_val = sum_val / hidden_dim
    var var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    var inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Step 2: Atomically accumulate gradients w.r.t. linear bias
    comptime for out_idx in range(output_dim):
        var grad_bias_ptr = grad_bias.ptr + out_idx
        _ = Atomic[dtype].fetch_add(
            grad_bias_ptr,
            rebind[Scalar[dtype]](grad_output_lt[batch_idx, seq_idx, out_idx]),
        )

    # Step 3: Atomically accumulate gradients w.r.t. linear weight
    comptime for out_idx in range(output_dim):
        comptime for h in range(hidden_dim):
            var input_val = input_lt[batch_idx, seq_idx, h]
            var normalized = (input_val - mean_val) * inv_std
            var ln_output_val = normalized * rebind[Scalar[dtype]](
                ln_weight_lt[h]
            ) + rebind[Scalar[dtype]](ln_bias_lt[h])

            # Atomic gradient accumulation for linear weight
            var grad_w = (
                grad_output_lt[batch_idx, seq_idx, out_idx] * ln_output_val
            )
            var grad_weight_ptr = grad_weight.ptr + out_idx * hidden_dim + h
            _ = Atomic.fetch_add(grad_weight_ptr, rebind[Scalar[dtype]](grad_w))

    # Step 4: Atomically accumulate gradients w.r.t. LayerNorm parameters
    comptime for h in range(hidden_dim):
        input_val = input_lt[batch_idx, seq_idx, h]
        normalized = (input_val - mean_val) * inv_std

        # Compute gradient w.r.t. LayerNorm output for this h
        var grad_ln_out: Scalar[dtype] = 0

        comptime for out_idx in range(output_dim):
            grad_ln_out = grad_ln_out + rebind[Scalar[dtype]](
                grad_output_lt[batch_idx, seq_idx, out_idx]
                * linear_weight_lt[out_idx, h]
            )

        # Atomic accumulation of LayerNorm parameter gradients
        var grad_ln_weight_ptr = grad_ln_weight.ptr + h
        var grad_ln_bias_ptr = grad_ln_bias.ptr + h
        _ = Atomic[dtype].fetch_add(
            grad_ln_weight_ptr, rebind[Scalar[dtype]](grad_ln_out * normalized)
        )
        _ = Atomic[dtype].fetch_add(
            grad_ln_bias_ptr, rebind[Scalar[dtype]](grad_ln_out)
        )

    # Step 5: Compute gradients w.r.t. input (LayerNorm backward)
    # Compute sum terms needed for LayerNorm backward
    var sum_grad_normalized: Scalar[dtype] = 0
    var sum_grad_normalized_times_normalized: Scalar[dtype] = 0

    comptime for h in range(hidden_dim):
        h_input_val = input_lt[batch_idx, seq_idx, h]
        h_normalized = (h_input_val - mean_val) * inv_std

        var h_grad_ln_out: Scalar[dtype] = 0

        comptime for out_idx in range(output_dim):
            h_grad_ln_out = h_grad_ln_out + rebind[Scalar[dtype]](
                grad_output_lt[batch_idx, seq_idx, out_idx]
                * linear_weight_lt[out_idx, h]
            )

        h_grad_norm = h_grad_ln_out * rebind[Scalar[dtype]](ln_weight_lt[h])
        sum_grad_normalized = sum_grad_normalized + rebind[Scalar[dtype]](
            h_grad_norm
        )
        sum_grad_normalized_times_normalized = (
            sum_grad_normalized_times_normalized
            + rebind[Scalar[dtype]](h_grad_norm * h_normalized)
        )

    # Compute actual input gradients (no race conditions here - each thread writes to different positions)
    comptime for h in range(hidden_dim):
        h_input_val = input_lt[batch_idx, seq_idx, h]
        h_normalized = (h_input_val - mean_val) * inv_std

        var h_grad_ln_out: Scalar[dtype] = 0

        comptime for out_idx in range(output_dim):
            h_grad_ln_out = h_grad_ln_out + rebind[Scalar[dtype]](
                grad_output_lt[batch_idx, seq_idx, out_idx]
                * linear_weight_lt[out_idx, h]
            )

        h_grad_norm = h_grad_ln_out * rebind[Scalar[dtype]](ln_weight_lt[h])
        grad_input_lt[batch_idx, seq_idx, h] = inv_std * (
            h_grad_norm
            - (sum_grad_normalized / hidden_dim)
            - (h_normalized * sum_grad_normalized_times_normalized / hidden_dim)
        )


# ANCHOR_END: minimal_fused_backward_kernel_solution


@compiler.register("layernorm_linear")
struct LayerNormLinearCustomOp:
    @staticmethod
    def execute[
        target: StaticString,
        algorithm: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=3, static_spec=_],
        input: InputTensor[dtype=dtype, rank=3, static_spec=_],
        ln_weight: InputTensor[dtype=dtype, rank=1, static_spec=_],
        ln_bias: InputTensor[dtype=dtype, rank=1, static_spec=_],
        linear_weight: InputTensor[dtype=dtype, rank=2, static_spec=_],
        linear_bias: InputTensor[dtype=dtype, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        comptime input_layout_val = row_major[batch_size, seq_len, hidden_dim]()
        comptime ln_params_layout_val = row_major[hidden_dim]()
        comptime weight_layout_val = row_major[output_dim, hidden_dim]()
        comptime bias_layout_val = row_major[output_dim]()
        comptime output_layout_val = row_major[
            batch_size, seq_len, output_dim
        ]()
        comptime InputLayout = type_of(input_layout_val)
        comptime LnParamsLayout = type_of(ln_params_layout_val)
        comptime WeightLayout = type_of(weight_layout_val)
        comptime BiasLayout = type_of(bias_layout_val)
        comptime OutputLayout = type_of(output_layout_val)

        var output_tensor = TileTensor[
            mut=True, dtype, OutputLayout, MutAnyOrigin
        ](output.unsafe_ptr(), output_layout_val)
        var input_tensor = TileTensor[
            mut=True, dtype, InputLayout, MutAnyOrigin
        ](input.unsafe_ptr(), input_layout_val)
        var ln_weight_tensor = TileTensor[
            mut=True, dtype, LnParamsLayout, MutAnyOrigin
        ](ln_weight.unsafe_ptr(), ln_params_layout_val)
        var ln_bias_tensor = TileTensor[
            mut=True, dtype, LnParamsLayout, MutAnyOrigin
        ](ln_bias.unsafe_ptr(), ln_params_layout_val)
        var linear_weight_tensor = TileTensor[
            mut=True, dtype, WeightLayout, MutAnyOrigin
        ](linear_weight.unsafe_ptr(), weight_layout_val)
        var linear_bias_tensor = TileTensor[
            mut=True, dtype, BiasLayout, MutAnyOrigin
        ](linear_bias.unsafe_ptr(), bias_layout_val)

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            # ANCHOR: layernorm_linear_custom_op
            comptime if algorithm == "fused":
                # fused case - one thread per sequence position
                comptime kernel = minimal_fused_kernel[
                    batch_size,
                    seq_len,
                    hidden_dim,
                    output_dim,
                    OutputLayout,
                    InputLayout,
                    LnParamsLayout,
                    WeightLayout,
                    BiasLayout,
                ]
                gpu_ctx.enqueue_function[kernel, kernel](
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
                var normalized_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * hidden_dim
                )
                var normalized_tensor = TileTensor[
                    mut=True, dtype, InputLayout, MutAnyOrigin
                ](normalized_buffer, input_layout_val)

                # Step 1: LayerNorm kernel
                comptime kernel = layernorm_kernel[
                    batch_size,
                    seq_len,
                    hidden_dim,
                    InputLayout,
                    InputLayout,
                    LnParamsLayout,
                ]
                gpu_ctx.enqueue_function[kernel, kernel](
                    normalized_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=hidden_dim,
                )

                # Step 2: Matmul on normalized data
                # (batch_size*seq_len, output_dim) outputs from ((batch*seq, hidden) @ (hidden, output) -> (batch*seq, output) ) with one thread per output
                var total_rows = batch_size * seq_len
                var blocks_y = (
                    total_rows + MATMUL_BLOCK_DIM_XY - 1
                ) // MATMUL_BLOCK_DIM_XY
                var blocks_x = (
                    output_dim + MATMUL_BLOCK_DIM_XY - 1
                ) // MATMUL_BLOCK_DIM_XY

                # Create intermediate result without bias
                var matmul_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * output_dim
                )
                var matmul_tensor = TileTensor[
                    mut=True, dtype, OutputLayout, MutAnyOrigin
                ](matmul_buffer, output_layout_val)

                # Create transposed weight matrix: [output_dim, hidden_dim] -> [hidden_dim, output_dim]
                var transposed_weight_buffer = gpu_ctx.enqueue_create_buffer[
                    dtype
                ](hidden_dim * output_dim)
                comptime transposed_weight_layout = row_major[
                    hidden_dim, output_dim
                ]()
                comptime TransposedWeightLayout = type_of(
                    transposed_weight_layout
                )
                var transposed_weight_tensor = TileTensor[
                    mut=True,
                    dtype,
                    TransposedWeightLayout,
                    MutAnyOrigin,
                ](transposed_weight_buffer, transposed_weight_layout)

                # Transpose the weight matrix
                var transpose_blocks_x = (
                    hidden_dim + TRANSPOSE_BLOCK_DIM_XY - 1
                ) // TRANSPOSE_BLOCK_DIM_XY
                var transpose_blocks_y = (
                    output_dim + TRANSPOSE_BLOCK_DIM_XY - 1
                ) // TRANSPOSE_BLOCK_DIM_XY
                comptime kernel2 = transpose_kernel[
                    output_dim,
                    hidden_dim,
                    TransposedWeightLayout,
                    WeightLayout,
                ]
                gpu_ctx.enqueue_function[kernel2, kernel2](
                    transposed_weight_tensor,
                    linear_weight_tensor,
                    grid_dim=(transpose_blocks_x, transpose_blocks_y),
                    block_dim=(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
                )

                # Reshape tensors for matmul: [batch*seq, hidden] @ [hidden, output] -> [batch*seq, output]
                comptime flat_normalized_layout = row_major[
                    batch_size * seq_len, hidden_dim
                ]()
                comptime FlatNormalizedLayout = type_of(flat_normalized_layout)
                comptime flat_matmul_layout = row_major[
                    batch_size * seq_len, output_dim
                ]()
                comptime FlatMatmulLayout = type_of(flat_matmul_layout)
                var flat_normalized = normalized_tensor.reshape(
                    flat_normalized_layout
                )
                var flat_matmul = matmul_tensor.reshape(flat_matmul_layout)

                comptime kernel3 = matmul_idiomatic_tiled[
                    batch_size * seq_len,
                    output_dim,
                    hidden_dim,
                    FlatMatmulLayout,
                    FlatNormalizedLayout,
                    TransposedWeightLayout,
                ]
                gpu_ctx.enqueue_function[kernel3, kernel3](
                    flat_matmul,
                    flat_normalized,
                    transposed_weight_tensor,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
                )

                # Step 3: Add bias - reshape matmul result back to 3D for bias addition
                comptime reshaped_matmul_layout = row_major[
                    batch_size, seq_len, output_dim
                ]()
                comptime ReshapedMatmulLayout = type_of(reshaped_matmul_layout)
                var reshaped_matmul = matmul_tensor.reshape(
                    reshaped_matmul_layout
                )

                comptime kernel4 = add_bias_kernel[
                    batch_size,
                    seq_len,
                    output_dim,
                    OutputLayout,
                    ReshapedMatmulLayout,
                    BiasLayout,
                ]
                gpu_ctx.enqueue_function[kernel4, kernel4](
                    output_tensor,
                    reshaped_matmul,
                    linear_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=output_dim,
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
                    var mean_val = sum_val / hidden_dim

                    var var_sum: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        var diff = input_tensor[batch, seq, h] - mean_val
                        var_sum += rebind[Scalar[dtype]](diff * diff)
                    var var_val = var_sum / hidden_dim
                    var inv_std = 1.0 / sqrt(var_val + 1e-5)

                    # Apply LayerNorm and Linear in one step (truly fused)
                    for out_idx in range(output_dim):
                        var acc: Scalar[dtype] = 0
                        for h in range(hidden_dim):
                            var input_val = input_tensor[batch, seq, h]
                            var normalized = (
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
    def execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
        dtype: DType = DType.float32,
    ](
        grad_input: OutputTensor[dtype=dtype, rank=3, static_spec=_],
        grad_ln_weight: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        grad_ln_bias: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        grad_weight: OutputTensor[dtype=dtype, rank=2, static_spec=_],
        grad_bias: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        grad_output: InputTensor[dtype=dtype, rank=3, static_spec=_],
        input: InputTensor[dtype=dtype, rank=3, static_spec=_],
        ln_weight: InputTensor[dtype=dtype, rank=1, static_spec=_],
        ln_bias: InputTensor[dtype=dtype, rank=1, static_spec=_],
        linear_weight: InputTensor[dtype=dtype, rank=2, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        comptime input_layout_val = row_major[batch_size, seq_len, hidden_dim]()
        comptime ln_params_layout_val = row_major[hidden_dim]()
        comptime weight_layout_val = row_major[output_dim, hidden_dim]()
        comptime grad_input_layout_val = row_major[
            batch_size, seq_len, hidden_dim
        ]()
        comptime grad_ln_weight_layout_val = row_major[hidden_dim]()
        comptime grad_ln_bias_layout_val = row_major[hidden_dim]()
        comptime grad_weight_layout_val = row_major[output_dim, hidden_dim]()
        comptime grad_bias_layout_val = row_major[output_dim]()
        comptime grad_output_layout_val = row_major[
            batch_size, seq_len, output_dim
        ]()
        comptime GradOutputLayout = type_of(grad_output_layout_val)
        comptime InputLayout = type_of(input_layout_val)
        comptime LnParamsLayout = type_of(ln_params_layout_val)
        comptime WeightLayout = type_of(weight_layout_val)
        comptime GradInputLayout = type_of(grad_input_layout_val)
        comptime GradLnWeightLayout = type_of(grad_ln_weight_layout_val)
        comptime GradLnBiasLayout = type_of(grad_ln_bias_layout_val)
        comptime GradWeightLayout = type_of(grad_weight_layout_val)
        comptime GradBiasLayout = type_of(grad_bias_layout_val)

        var grad_input_tensor = TileTensor[
            mut=True, dtype, GradInputLayout, MutAnyOrigin
        ](grad_input.unsafe_ptr(), grad_input_layout_val)
        var grad_ln_weight_tensor = TileTensor[
            mut=True, dtype, GradLnWeightLayout, MutAnyOrigin
        ](grad_ln_weight.unsafe_ptr(), grad_ln_weight_layout_val)
        var grad_ln_bias_tensor = TileTensor[
            mut=True, dtype, GradLnBiasLayout, MutAnyOrigin
        ](grad_ln_bias.unsafe_ptr(), grad_ln_bias_layout_val)
        var grad_weight_tensor = TileTensor[
            mut=True, dtype, GradWeightLayout, MutAnyOrigin
        ](grad_weight.unsafe_ptr(), grad_weight_layout_val)
        var grad_bias_tensor = TileTensor[
            mut=True, dtype, GradBiasLayout, MutAnyOrigin
        ](grad_bias.unsafe_ptr(), grad_bias_layout_val)
        var grad_output_tensor = TileTensor[
            mut=True, dtype, GradOutputLayout, MutAnyOrigin
        ](grad_output.unsafe_ptr(), grad_output_layout_val)
        var input_tensor = TileTensor[
            mut=True, dtype, InputLayout, MutAnyOrigin
        ](input.unsafe_ptr(), input_layout_val)
        var ln_weight_tensor = TileTensor[
            mut=True, dtype, LnParamsLayout, MutAnyOrigin
        ](ln_weight.unsafe_ptr(), ln_params_layout_val)
        var ln_bias_tensor = TileTensor[
            mut=True, dtype, LnParamsLayout, MutAnyOrigin
        ](ln_bias.unsafe_ptr(), ln_params_layout_val)
        var linear_weight_tensor = TileTensor[
            mut=True, dtype, WeightLayout, MutAnyOrigin
        ](linear_weight.unsafe_ptr(), weight_layout_val)

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            # Launch backward kernel
            comptime kernel = minimal_fused_kernel_backward[
                batch_size,
                seq_len,
                hidden_dim,
                output_dim,
                GradInputLayout,
                GradLnWeightLayout,
                GradLnBiasLayout,
                GradWeightLayout,
                GradBiasLayout,
                GradOutputLayout,
                InputLayout,
                LnParamsLayout,
                WeightLayout,
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
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
                    var mean_val = sum_val / hidden_dim

                    var var_sum: Scalar[dtype] = 0
                    for h in range(hidden_dim):
                        var diff = input_tensor[batch, seq, h] - mean_val
                        var_sum += rebind[Scalar[dtype]](diff * diff)
                    var var_val = var_sum / hidden_dim
                    var inv_std = 1.0 / sqrt(var_val + 1e-5)

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
                            var ln_output_val = (
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
