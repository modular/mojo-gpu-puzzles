from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from testing import assert_equal

alias TPB = 4
alias SIZE = 8
alias BLOCKS_PER_GRID = (2, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# ANCHOR: add_10_shared_layout_tensor_solution
fn add_10_shared_layout_tensor[
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

    if global_i < size:
        shared[local_i] = a[global_i]

    # Note: barrier is not strictly needed here since each thread only accesses its own shared memory location.
    # However, it's included to teach proper shared memory synchronization patterns
    # for more complex scenarios where threads need to coordinate access to shared data.
    # For this specific puzzle, we can remove the barrier since each thread only accesses its own shared memory location.
    barrier()

    if global_i < size:
        output[global_i] = shared[local_i] + 10


# ANCHOR_END: add_10_shared_layout_tensor_solution


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(1)

        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)

        alias kernel = add_10_shared_layout_tensor[layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(11)
        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
