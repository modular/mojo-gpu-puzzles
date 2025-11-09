from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from testing import assert_equal

alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)
alias out_layout = Layout.row_major(1)


# ANCHOR: dot_product_layout_tensor_solution
fn dot_product[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Compute element-wise multiplication into shared memory
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    # Synchronize threads within block
    barrier()

    # Parallel reduction in shared memory
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()
        stride //= 2

    # Only thread 0 writes the final result
    if local_i == 0:
        output[0] = shared[0]


# ANCHOR_END: dot_product_layout_tensor_solution


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i

        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
        a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)
        b_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](b)

        alias kernel = dot_product[layout, out_layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](1)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])
