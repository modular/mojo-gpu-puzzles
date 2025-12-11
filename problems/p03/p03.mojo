from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add_10_guard
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (8, 1)
comptime dtype = DType.float32


fn add_10_guard(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    i = thread_idx.x
    # Interesting on Apple GPU this code worked even without the conditional
    # Not sure about other GPUs but the key learning here is that our thread block size is bigger than SIZE of the input
    # that is why we need to guard against out of bounds access
    # What about 2D/3D array boundaries?
    # How to handle different shapes efficiently?
    # What if we need padding or edge handling?
    if i < size:
        output[i] = a[i] + 10
    # FILL ME IN (roughly 2 lines)


# ANCHOR_END: add_10_guard


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i

        ctx.enqueue_function_checked[add_10_guard, add_10_guard](
            out,
            a,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = i + 10

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
