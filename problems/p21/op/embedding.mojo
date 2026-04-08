from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout
from std.sys import argv
from std.testing import assert_equal

# ANCHOR: embedding_kernel_coalesced
comptime THREADS_PER_BLOCK = 256


def embedding_kernel_coalesced[
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    OutLayout: TensorLayout,
    IndicesLayout: TensorLayout,
    WeightsLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    indices: TileTensor[mut=False, DType.int32, IndicesLayout, MutAnyOrigin],
    weights: TileTensor[mut=False, dtype, WeightsLayout, MutAnyOrigin],
):
    """
    Memory-coalescing focused embedding kernel.

    Key insight: The bottleneck is memory access patterns, not computation.
    - Each thread handles one (batch, seq, embed) position
    - Simple 1D grid for maximum simplicity and correctness
    - Focus on getting memory access right first
    """

    # Simple 1D indexing - each thread = one output element
    var global_idx = block_idx.x * block_dim.x + thread_idx.x
    var total_elements = batch_size * seq_len * embed_dim

    if global_idx >= total_elements:
        return

    # Convert to (batch, seq, embed) coordinates
    # FILL IN roughly 4 lines

    # Get token index
    # FILL IN 1 line

    # Simple, correct assignment
    # FILL IN 4 lines


# ANCHOR_END: embedding_kernel_coalesced


# ANCHOR: embedding_kernel_2d
def embedding_kernel_2d[
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    OutLayout: TensorLayout,
    IndicesLayout: TensorLayout,
    WeightsLayout: TensorLayout,
    dtype: DType = DType.float32,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    indices: TileTensor[mut=False, DType.int32, IndicesLayout, MutAnyOrigin],
    weights: TileTensor[mut=False, dtype, WeightsLayout, MutAnyOrigin],
):
    """
    2D grid non-coalesced embedding kernel.

    Non-optimal approach for comparison:
    - 2D grid: (batch*seq, embed_dim)
    - More complex indexing
    - Potentially worse memory access patterns
    """

    # 2D grid indexing
    var batch_seq_idx = block_idx.x * block_dim.x + thread_idx.x
    var embed_idx = block_idx.y * block_dim.y + thread_idx.y
    var total_positions = batch_size * seq_len

    if batch_seq_idx >= total_positions or embed_idx >= embed_dim:
        return

    # Convert to (batch, seq) coordinates
    # FILL IN 2 lines

    # Get token index
    # FILL IN 1 line

    # Assignment with 2D grid pattern
    # FILL IN 4 lines


# ANCHOR_END: embedding_kernel_2d

import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.memory import UnsafePointer
from std.gpu.host import DeviceBuffer


@compiler.register("embedding")
struct EmbeddingCustomOp:
    @staticmethod
    def execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
    ](
        output: OutputTensor[
            dtype=DType.float32, rank=3, static_spec=_
        ],  # [batch_size, seq_len, embed_dim]
        indices: InputTensor[
            dtype=DType.int32, rank=2, static_spec=_
        ],  # [batch_size, seq_len]
        weights: InputTensor[
            dtype=output.dtype, rank=2, static_spec=_
        ],  # [vocab_size, embed_dim]
        ctx: DeviceContextPtr,
    ) raises:
        var output_tensor = output.to_layout_tensor()
        var indices_tensor = indices.to_layout_tensor()
        var weights_tensor = weights.to_layout_tensor()

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    output_tensor.ptr,
                    batch_size * seq_len * embed_dim,
                    owning=False,
                ),
                0,
            )

            # Calculate 1D grid dimensions (matching kernel's flat indexing)
            var total_elements = batch_size * seq_len * embed_dim
            var blocks = max(1, ceildiv(total_elements, THREADS_PER_BLOCK))

            # Compile and launch optimized kernel
            comptime kernel = embedding_kernel_coalesced[
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]
            var compiled_kernel = gpu_ctx.compile_function[kernel, kernel]()

            gpu_ctx.enqueue_function(
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
                    var token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)


@compiler.register("embedding_2d")
struct Embedding2DCustomOp:
    @staticmethod
    def execute[
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
    ](
        output: OutputTensor[
            dtype=DType.float32, rank=3, static_spec=_
        ],  # [batch_size, seq_len, embed_dim]
        indices: InputTensor[
            dtype=DType.int32, rank=2, static_spec=_
        ],  # [batch_size, seq_len]
        weights: InputTensor[
            dtype=output.dtype, rank=2, static_spec=_
        ],  # [vocab_size, embed_dim]
        ctx: DeviceContextPtr,
    ) raises:
        var output_tensor = output.to_layout_tensor()
        var indices_tensor = indices.to_layout_tensor()
        var weights_tensor = weights.to_layout_tensor()

        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    output_tensor.ptr,
                    batch_size * seq_len * embed_dim,
                    owning=False,
                ),
                0,
            )

            # Calculate 2D grid dimensions for non-coalesced access
            var total_positions = batch_size * seq_len
            comptime BLOCK_X = 16  # batch*seq dimension
            comptime BLOCK_Y = 16  # embed dimension
            var blocks_x = max(1, ceildiv(total_positions, BLOCK_X))
            var blocks_y = max(1, ceildiv(embed_dim, BLOCK_Y))

            # Compile and launch 2D kernel
            comptime kernel = embedding_kernel_2d[
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]
            var compiled_kernel = gpu_ctx.compile_function[kernel, kernel]()

            gpu_ctx.enqueue_function(
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
                    var token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)
