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


# ANCHOR: pooling_layout_tensor_solution
fn pooling[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    # Allocate shared memory using tensor builder
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Load data into shared memory
    if global_i < size:
        shared[local_i] = a[global_i]

    # Synchronize threads within block
    barrier()

    # Handle first two special cases
    if global_i == 0:
        output[0] = shared[0]
    elif global_i == 1:
        output[1] = shared[0] + shared[1]
    # Handle general case
    elif 1 < global_i < size:
        output[global_i] = (
            shared[local_i - 2] + shared[local_i - 1] + shared[local_i]
        )


# ANCHOR_END: pooling_layout_tensor_solution


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i

        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)

        ctx.enqueue_function_checked[pooling[layout], pooling[layout]](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            ptr = a_host
            for i in range(SIZE):
                s = Scalar[dtype](0)
                for j in range(max(i - 2, 0), i + 1):
                    s += ptr[j]
                expected[i] = s

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
