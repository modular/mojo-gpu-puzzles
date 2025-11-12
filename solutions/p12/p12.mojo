from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from sys import size_of
from testing import assert_equal

alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32


# ANCHOR: dot_product_solution
fn dot_product(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    shared = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    barrier()

    # The following causes race condition: all threads writing to the same location
    # out[0] += shared[local_i]

    # Instead can do parallel reduction in shared memory as opposed to
    # global memory which has no guarantee on synchronization.
    # Loops using global memory can cause thread divergence because
    # fundamentally GPUs execute threads in warps (groups of 32 threads typically)
    # and warps can be scheduled independently.
    # However, shared memory does not have such issues as long as we use `barrier()`
    # correctly when we're in the same thread block.
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()
        stride //= 2

    # only thread 0 writes the final result
    if local_i == 0:
        output[0] = shared[0]


# ANCHOR_END: dot_product_solution


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

        ctx.enqueue_function_checked[dot_product, dot_product](
            out,
            a,
            b,
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
