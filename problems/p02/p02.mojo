from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add
alias SIZE = 4
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = SIZE
alias dtype = DType.float32


fn add(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
):
    i = thread_idx.x
    output[i] = a[i] + b[i]


# ANCHOR_END: add


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i
                expected[i] = a_host[i] + b_host[i]

        ctx.enqueue_function[add](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
