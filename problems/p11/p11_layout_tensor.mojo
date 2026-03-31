from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from std.testing import assert_equal

# ANCHOR: pooling_layout_tensor
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)


def pooling[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    # Allocate shared memory using tensor builder
    var shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # FIX ME IN (roughly 10 lines)


# ANCHOR_END: pooling_layout_tensor


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = Float32(i)

        var out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        var a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)

        ctx.enqueue_function[pooling[layout], pooling[layout]](
            out_tensor,
            a_tensor,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            var ptr = a_host
            for i in range(SIZE):
                var s = Scalar[dtype](0)
                for j in range(max(i - 2, 0), i + 1):
                    s += ptr[j]
                expected[i] = s

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 11 complete ✅")
