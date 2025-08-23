from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add_10_guard
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (8, 1)
alias dtype = DType.float32


fn add_10_guard(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    i = thread_idx.x
    if i < size: output[i] = a[i] + 10.0


# ANCHOR_END: add_10_guard


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i

        ctx.enqueue_function[add_10_guard](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)
        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = i + 10

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
