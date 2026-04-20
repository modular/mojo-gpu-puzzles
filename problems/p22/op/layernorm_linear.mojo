from std.math import sqrt
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.atomic import Atomic
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from layout.layout_tensor import copy_dram_to_sram_async
import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils import StaticTuple

comptime MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
comptime MATMUL_NUM_THREADS = MATMUL_BLOCK_DIM_XY * MATMUL_BLOCK_DIM_XY
comptime MATMUL_BLOCK_DIM_COUNT = 2
comptime TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output
comptime TPB = 16
comptime dtype = DType.float32


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
    a: TileTensor[mut=False, dtype, ALayout, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, BLayout, MutAnyOrigin],
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

    # Write final result with bounds checking (needed for variable matrix sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc


# ANCHOR_END: matmul_idiomatic_tiled


# ANCHOR: layernorm_kernel
def layernorm_kernel[
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    OutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    LnParamsLayout: TensorLayout,
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InputLayout, ImmutAnyOrigin],
    ln_weight: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
    ln_bias: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
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

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    # FILL ME IN (roughly 11 lines)


# ANCHOR_END: layernorm_kernel


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

    var global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row
    var global_col = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_col

    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = inp[global_row, global_col]

    barrier()

    var out_row = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_row
    var out_col = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]


# ANCHOR_END: transpose_kernel


# ANCHOR: add_bias_kernel
def add_bias_kernel[
    batch_size: Int,
    seq_len: Int,
    output_dim: Int,
    OutputLayout: TensorLayout,
    InputLayout: TensorLayout,
    BiasLayout: TensorLayout,
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InputLayout, MutAnyOrigin],
    bias: TileTensor[mut=False, dtype, BiasLayout, ImmutAnyOrigin],
):
    """Simple bias addition."""
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y
    var out_idx = thread_idx.x

    if batch_idx >= batch_size or seq_idx >= seq_len or out_idx >= output_dim:
        return

    output[batch_idx, seq_idx, out_idx] = input[
        batch_idx, seq_idx, out_idx
    ] + rebind[Scalar[dtype]](bias[out_idx])


# ANCHOR_END: add_bias_kernel


# ANCHOR: minimal_fused_forward_kernel
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
](
    output: TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin],
    input: TileTensor[mut=False, dtype, InputLayout, ImmutAnyOrigin],
    ln_weight: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
    ln_bias: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
    linear_weight: TileTensor[mut=False, dtype, WeightLayout, ImmutAnyOrigin],
    linear_bias: TileTensor[mut=False, dtype, BiasLayout, ImmutAnyOrigin],
):
    """Minimal fused kernel - one thread per sequence position to avoid redundancy.
    """
    # Grid: (batch_size, seq_len) - one thread block per sequence position
    # Block: (1,) - single thread per sequence position to avoid redundant computation
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Step 1: Compute LayerNorm statistics once per sequence position

    # FILL IN roughly 10 lines

    # Step 2: Compute all outputs for this sequence position

    # FILL IN roughly 10 lines


# ANCHOR_END: minimal_fused_forward_kernel


# ANCHOR: minimal_fused_backward_kernel
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
](
    grad_input: TileTensor[mut=True, dtype, GradInputLayout, MutAnyOrigin],
    grad_ln_weight: TileTensor[
        mut=True, dtype, GradLnWeightLayout, MutAnyOrigin
    ],
    grad_ln_bias: TileTensor[mut=True, dtype, GradLnBiasLayout, MutAnyOrigin],
    grad_weight: TileTensor[mut=True, dtype, GradWeightLayout, MutAnyOrigin],
    grad_bias: TileTensor[mut=True, dtype, GradBiasLayout, MutAnyOrigin],
    grad_output: TileTensor[mut=False, dtype, GradOutputLayout, ImmutAnyOrigin],
    input: TileTensor[mut=False, dtype, InputLayout, ImmutAnyOrigin],
    ln_weight: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
    ln_bias: TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin],
    linear_weight: TileTensor[mut=False, dtype, WeightLayout, ImmutAnyOrigin],
):
    """Fused backward kernel using atomic operations for safe gradient accumulation.
    """
    # Grid: (batch_size, seq_len) - one thread per sequence position
    # Block: (1,) - single thread per sequence position
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

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
    def execute[
        target: StaticString,
        algorithm: StaticString,
        batch_size: Int,
        seq_len: Int,
        hidden_dim: Int,
        output_dim: Int,
    ](
        output: OutputTensor[dtype=DType.float32, rank=3, static_spec=_],
        input: InputTensor[dtype=DType.float32, rank=3, static_spec=_],
        ln_weight: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        ln_bias: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        linear_weight: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        linear_bias: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        comptime input_layout = input.static_spec.to_layout()
        comptime ln_params_layout = ln_weight.static_spec.to_layout()
        comptime weight_layout = linear_weight.static_spec.to_layout()
        comptime bias_layout = linear_bias.static_spec.to_layout()
        comptime output_layout = output.static_spec.to_layout()
        comptime InputLayout = type_of(input_layout)
        comptime LnParamsLayout = type_of(ln_params_layout)
        comptime WeightLayout = type_of(weight_layout)
        comptime BiasLayout = type_of(bias_layout)
        comptime OutputLayout = type_of(output_layout)

        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            TileTensor[mut=True, dtype, OutputLayout, MutAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            TileTensor[mut=False, dtype, InputLayout, ImmutAnyOrigin]
        ](input.to_layout_tensor())
        var ln_weight_tensor = rebind[
            TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin]
        ](ln_weight.to_layout_tensor())
        var ln_bias_tensor = rebind[
            TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin]
        ](ln_bias.to_layout_tensor())
        var linear_weight_tensor = rebind[
            TileTensor[mut=False, dtype, WeightLayout, ImmutAnyOrigin]
        ](linear_weight.to_layout_tensor())
        var linear_bias_tensor = rebind[
            TileTensor[mut=False, dtype, BiasLayout, ImmutAnyOrigin]
        ](linear_bias.to_layout_tensor())

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
                ](normalized_buffer, input_layout)

                # Step 1: LayerNorm kernel
                comptime kernel = layernorm_kernel[
                    batch_size,
                    seq_len,
                    hidden_dim,
                ]
                gpu_ctx.enqueue_function[kernel, kernel](
                    normalized_tensor,
                    input_tensor,
                    ln_weight_tensor,
                    ln_bias_tensor,
                    grid_dim=(batch_size, seq_len),
                    block_dim=(min(hidden_dim, TPB),),
                )

                # Step 2: Matmul on normalized data
                var total_rows = batch_size * seq_len
                var blocks_x = (total_rows + TPB - 1) // TPB
                var blocks_y = (output_dim + TPB - 1) // TPB

                # Create intermediate result without bias
                var matmul_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                    batch_size * seq_len * output_dim
                )
                var matmul_tensor = TileTensor[
                    mut=True, dtype, OutputLayout, MutAnyOrigin
                ](matmul_buffer, output_layout)

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
                var flat_normalized = normalized_tensor.reshape[
                    flat_normalized_layout
                ]()
                var flat_matmul = matmul_tensor.reshape[flat_matmul_layout]()

                comptime kernel3 = matmul_idiomatic_tiled[
                    batch_size * seq_len,
                    output_dim,
                    hidden_dim,
                ]
                gpu_ctx.enqueue_function[kernel3, kernel3](
                    flat_matmul,
                    flat_normalized,
                    transposed_weight_tensor,
                    grid_dim=(blocks_x, blocks_y),
                    block_dim=(TPB, TPB),
                )

                # Step 3: Add bias - reshape matmul result back to 3D for bias addition
                comptime reshaped_matmul_layout = row_major[
                    batch_size, seq_len, output_dim
                ]()
                comptime ReshapedMatmulLayout = type_of(reshaped_matmul_layout)
                var reshaped_matmul = matmul_tensor.reshape[
                    reshaped_matmul_layout
                ]()

                comptime kernel4 = add_bias_kernel[
                    batch_size,
                    seq_len,
                    output_dim,
                ]
                gpu_ctx.enqueue_function[kernel4, kernel4](
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
    ](
        grad_input: OutputTensor[dtype=DType.float32, rank=3, static_spec=_],
        grad_ln_weight: OutputTensor[
            dtype=DType.float32, rank=1, static_spec=_
        ],
        grad_ln_bias: OutputTensor[dtype=DType.float32, rank=1, static_spec=_],
        grad_weight: OutputTensor[dtype=DType.float32, rank=2, static_spec=_],
        grad_bias: OutputTensor[dtype=DType.float32, rank=1, static_spec=_],
        grad_output: InputTensor[dtype=DType.float32, rank=3, static_spec=_],
        input: InputTensor[dtype=DType.float32, rank=3, static_spec=_],
        ln_weight: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        ln_bias: InputTensor[dtype=DType.float32, rank=1, static_spec=_],
        linear_weight: InputTensor[dtype=DType.float32, rank=2, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        comptime grad_output_layout = grad_output.static_spec.to_layout()
        comptime input_layout = input.static_spec.to_layout()
        comptime ln_params_layout = ln_weight.static_spec.to_layout()
        comptime weight_layout = linear_weight.static_spec.to_layout()
        comptime grad_input_layout = grad_input.static_spec.to_layout()
        comptime grad_ln_weight_layout = grad_ln_weight.static_spec.to_layout()
        comptime grad_ln_bias_layout = grad_ln_bias.static_spec.to_layout()
        comptime grad_weight_layout = grad_weight.static_spec.to_layout()
        comptime grad_bias_layout = grad_bias.static_spec.to_layout()
        comptime GradOutputLayout = type_of(grad_output_layout)
        comptime InputLayout = type_of(input_layout)
        comptime LnParamsLayout = type_of(ln_params_layout)
        comptime WeightLayout = type_of(weight_layout)
        comptime GradInputLayout = type_of(grad_input_layout)
        comptime GradLnWeightLayout = type_of(grad_ln_weight_layout)
        comptime GradLnBiasLayout = type_of(grad_ln_bias_layout)
        comptime GradWeightLayout = type_of(grad_weight_layout)
        comptime GradBiasLayout = type_of(grad_bias_layout)

        var grad_input_tensor = rebind[
            TileTensor[mut=True, dtype, GradInputLayout, MutAnyOrigin]
        ](grad_input.to_layout_tensor())
        var grad_ln_weight_tensor = rebind[
            TileTensor[mut=True, dtype, GradLnWeightLayout, MutAnyOrigin]
        ](grad_ln_weight.to_layout_tensor())
        var grad_ln_bias_tensor = rebind[
            TileTensor[mut=True, dtype, GradLnBiasLayout, MutAnyOrigin]
        ](grad_ln_bias.to_layout_tensor())
        var grad_weight_tensor = rebind[
            TileTensor[mut=True, dtype, GradWeightLayout, MutAnyOrigin]
        ](grad_weight.to_layout_tensor())
        var grad_bias_tensor = rebind[
            TileTensor[mut=True, dtype, GradBiasLayout, MutAnyOrigin]
        ](grad_bias.to_layout_tensor())
        var grad_output_tensor = rebind[
            TileTensor[mut=False, dtype, GradOutputLayout, ImmutAnyOrigin]
        ](grad_output.to_layout_tensor())
        var input_tensor = rebind[
            TileTensor[mut=False, dtype, InputLayout, ImmutAnyOrigin]
        ](input.to_layout_tensor())
        var ln_weight_tensor = rebind[
            TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin]
        ](ln_weight.to_layout_tensor())
        var ln_bias_tensor = rebind[
            TileTensor[mut=False, dtype, LnParamsLayout, ImmutAnyOrigin]
        ](ln_bias.to_layout_tensor())
        var linear_weight_tensor = rebind[
            TileTensor[mut=False, dtype, WeightLayout, ImmutAnyOrigin]
        ](linear_weight.to_layout_tensor())

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            # Launch backward kernel
            comptime kernel = minimal_fused_kernel_backward[
                batch_size,
                seq_len,
                hidden_dim,
                output_dim,
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
