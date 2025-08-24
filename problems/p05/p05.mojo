from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext, HostBuffer
from testing import assert_equal

# ANCHOR: broadcast_add
alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32


fn broadcast_add(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[col] + b[row]


# ANCHOR_END: broadcast_add
def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i

            for i in range(SIZE):
                for j in range(SIZE):
                    expected[i * SIZE + j] = a_host[j] + b_host[i]

        ctx.enqueue_function[broadcast_add](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])
