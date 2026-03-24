from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from std.testing import assert_equal

comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)
comptime out_layout = Layout.row_major(1)


# ANCHOR: dot_product_layout_tensor_solution
def dot_product[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: UInt,
):
    var shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x

    # Compute element-wise multiplication into shared memory
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    # Synchronize threads within block
    barrier()

    # Parallel reduction in shared memory
    var stride = UInt(TPB // 2)
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()
        stride //= 2

    # Only thread 0 writes the final result
    if local_i == 0:
        output[0] = shared[0]


# ANCHOR_END: dot_product_layout_tensor_solution


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
                a_host[i] = i
                b_host[i] = i

        var out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
        var a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)
        var b_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](b)

        comptime kernel = dot_product[layout, out_layout]
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            a_tensor,
            b_tensor,
            UInt(SIZE),
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
