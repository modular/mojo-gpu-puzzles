from math import ceildiv
from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from sys import size_of, argv
from testing import assert_equal

alias THREADS_PER_BLOCK = 256


# ANCHOR: embedding_kernel_coalesced_solution
fn embedding_kernel_coalesced[
    indices_layout: Layout,
    weights_layout: Layout,
    out_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, indices_layout, MutAnyOrigin],
    weights: LayoutTensor[dtype, weights_layout, MutAnyOrigin],
):
    """
    Memory-coalescing focused embedding kernel.

    Key insight: The bottleneck is memory access patterns, not computation.
    - Each thread handles one (batch, seq, embed) position
    - Simple 1D grid for maximum simplicity and correctness
    - Focus on getting memory access right first
    """

    # Simple 1D indexing - each thread = one output element
    global_idx = block_idx.x * block_dim.x + thread_idx.x
    total_elements = batch_size * seq_len * embed_dim

    if global_idx >= total_elements:
        return

    # Convert to (batch, seq, embed) coordinates
    batch_idx = global_idx // (seq_len * embed_dim)
    remaining = global_idx % (seq_len * embed_dim)
    seq_idx = remaining // embed_dim
    embed_idx = remaining % embed_dim

    # Get token index
    token_idx_val = Int(indices[batch_idx, seq_idx])

    # Simple, correct assignment
    if token_idx_val >= 0 and token_idx_val < vocab_size:
        output[batch_idx, seq_idx, embed_idx] = weights[
            token_idx_val, embed_idx
        ]
    else:
        output[batch_idx, seq_idx, embed_idx] = 0


# ANCHOR_END: embedding_kernel_coalesced_solution


# ANCHOR: embedding_kernel_2d_solution
fn embedding_kernel_2d[
    indices_layout: Layout,
    weights_layout: Layout,
    out_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, indices_layout, MutAnyOrigin],
    weights: LayoutTensor[dtype, weights_layout, MutAnyOrigin],
):
    """
    2D grid non-coalesced embedding kernel.

    Non-optimal approach for comparison:
    - 2D grid: (batch*seq, embed_dim)
    - More complex indexing
    - Potentially worse memory access patterns
    """

    # 2D grid indexing
    batch_seq_idx = block_idx.x * block_dim.x + thread_idx.x
    embed_idx = block_idx.y * block_dim.y + thread_idx.y

    total_positions = batch_size * seq_len

    # Bounds check
    if batch_seq_idx >= total_positions or embed_idx >= embed_dim:
        return

    # Convert to (batch, seq) coordinates
    batch_idx = batch_seq_idx // seq_len
    seq_idx = batch_seq_idx % seq_len

    # Get token index
    token_idx_val = Int(indices[batch_idx, seq_idx])

    # Assignment with 2D grid pattern
    if token_idx_val >= 0 and token_idx_val < vocab_size:
        output[batch_idx, seq_idx, embed_idx] = weights[
            token_idx_val, embed_idx
        ]
    else:
        output[batch_idx, seq_idx, embed_idx] = 0


# ANCHOR_END: embedding_kernel_2d_solution

# ANCHOR: embedding_custom_op_solution
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer


@compiler.register("embedding")
struct EmbeddingCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
    ](
        output: OutputTensor[
            dtype = DType.float32, rank=3
        ],  # [batch_size, seq_len, embed_dim]
        indices: InputTensor[
            dtype = DType.int32, rank=2
        ],  # [batch_size, seq_len]
        weights: InputTensor[
            dtype = output.dtype, rank=2
        ],  # [vocab_size, embed_dim]
        ctx: DeviceContextPtr,
    ) raises:
        output_tensor = output.to_layout_tensor()
        indices_tensor = indices.to_layout_tensor()
        weights_tensor = weights.to_layout_tensor()

        alias indices_layout = indices_tensor.layout
        alias weights_layout = weights_tensor.layout
        alias out_layout = output_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output.dtype]]](
                        output_tensor.ptr
                    ),
                    batch_size * seq_len * embed_dim,
                    owning=False,
                ),
                0,
            )

            # Calculate 1D grid dimensions (matching kernel's flat indexing)
            total_elements = batch_size * seq_len * embed_dim
            blocks = max(1, ceildiv(total_elements, THREADS_PER_BLOCK))

            # Compile and launch optimized kernel
            alias kernel = embedding_kernel_coalesced[
                indices_layout,
                weights_layout,
                out_layout,
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]
            compiled_kernel = gpu_ctx.compile_function_checked[kernel, kernel]()

            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                output_tensor,
                indices_tensor,
                weights_tensor,
                grid_dim=(blocks,),
                block_dim=(THREADS_PER_BLOCK,),
            )

        elif target == "cpu":
            for batch in range(batch_size):
                for seq in range(seq_len):
                    token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)


# ANCHOR_END: embedding_custom_op_solution


# ANCHOR: embedding_2d_custom_op_solution
@compiler.register("embedding_2d")
struct Embedding2DCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
    ](
        output: OutputTensor[
            dtype = DType.float32, rank=3
        ],  # [batch_size, seq_len, embed_dim]
        indices: InputTensor[
            dtype = DType.int32, rank=2
        ],  # [batch_size, seq_len]
        weights: InputTensor[
            dtype = output.dtype, rank=2
        ],  # [vocab_size, embed_dim]
        ctx: DeviceContextPtr,
    ) raises:
        output_tensor = output.to_layout_tensor()
        indices_tensor = indices.to_layout_tensor()
        weights_tensor = weights.to_layout_tensor()

        alias indices_layout = indices_tensor.layout
        alias weights_layout = weights_tensor.layout
        alias out_layout = output_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output.dtype]]](
                        output_tensor.ptr
                    ),
                    batch_size * seq_len * embed_dim,
                    owning=False,
                ),
                0,
            )

            # Calculate 2D grid dimensions for non-coalesced access
            total_positions = batch_size * seq_len
            alias BLOCK_X = 16  # batch*seq dimension
            alias BLOCK_Y = 16  # embed dimension
            blocks_x = max(1, ceildiv(total_positions, BLOCK_X))
            blocks_y = max(1, ceildiv(embed_dim, BLOCK_Y))

            # Compile and launch 2D kernel
            alias kernel = embedding_kernel_2d[
                indices_layout,
                weights_layout,
                out_layout,
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]

            compiled_kernel = gpu_ctx.compile_function_checked[kernel, kernel]()

            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                output_tensor,
                indices_tensor,
                weights_tensor,
                grid_dim=(blocks_x, blocks_y),
                block_dim=(BLOCK_X, BLOCK_Y),
            )

        elif target == "cpu":
            # Same CPU fallback as 1D version
            for batch in range(batch_size):
                for seq in range(seq_len):
                    token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)


# ANCHOR_END: embedding_2d_custom_op_solution
