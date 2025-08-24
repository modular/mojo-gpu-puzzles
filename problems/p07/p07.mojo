from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add_10_blocks_2d
alias SIZE = 5
alias BLOCKS_PER_GRID = (2, 2)
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32


fn add_10_blocks_2d(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    # bounds check
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


# ANCHOR_END: add_10_blocks_2d


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(1)
        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(1)
        ctx.enqueue_function[add_10_blocks_2d](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        for i in range(SIZE):
            for j in range(SIZE):
                expected[i * SIZE + j] += 10

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])
