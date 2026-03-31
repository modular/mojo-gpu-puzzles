from std.memory import UnsafePointer, stack_allocation
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.testing import assert_equal

# ANCHOR: dot_product
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32


def dot_product(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    # FILL ME IN (roughly 13 lines)
    ...


# ANCHOR_END: dot_product


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i)
                b_host[i] = Float32(i)

        ctx.enqueue_function[dot_product, dot_product](
            out,
            a,
            b,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](1)
        expected.enqueue_fill(0)

        ctx.synchronize()

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])
            print("Puzzle 12 complete ✅")
