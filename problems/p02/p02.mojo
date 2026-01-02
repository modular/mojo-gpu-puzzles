from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32


fn add(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    i = thread_idx.x
    output[i] = a[i] + b[i]


# ANCHOR_END: add


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i
                expected[i] = a_host[i] + b_host[i]

        ctx.enqueue_function_checked[add, add](
            out,
            a,
            b,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
