from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add_10
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32


fn add_10(
    output: UnsafePointer[Scalar[dtype]], a: UnsafePointer[Scalar[dtype]]
):
    i = thread_idx.x
    output[i] = a[i] + 10


# ANCHOR_END: add_10


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out = out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a = a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i

        ctx.enqueue_function[add_10](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected = expected.enqueue_fill(0)
        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = i + 10

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
