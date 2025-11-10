from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import size_of
from testing import assert_equal

alias TPB = 8
alias BATCH = 4
alias SIZE = 6
alias BLOCKS_PER_GRID = (1, BATCH)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias in_layout = Layout.row_major(BATCH, SIZE)
alias out_layout = Layout.row_major(BATCH, 1)


# ANCHOR: axis_sum_solution
fn axis_sum[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    batch = block_idx.y
    cache = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Visualize:
    # Block(0,0): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 0: [0,1,2,3,4,5]
    # Block(0,1): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 1: [6,7,8,9,10,11]
    # Block(0,2): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 2: [12,13,14,15,16,17]
    # Block(0,3): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 3: [18,19,20,21,22,23]

    # each row is handled by each block bc we have grid_dim=(1, BATCH)

    if local_i < size:
        cache[local_i] = a[batch, local_i]
    else:
        # Add zero-initialize padding elements for later reduction
        cache[local_i] = 0

    barrier()

    # do reduction sum per each block
    stride = TPB // 2
    while stride > 0:
        # Read phase: all threads read the values they need first to avoid race conditions
        var temp_val: output.element_type = 0
        if local_i < stride:
            temp_val = cache[local_i + stride]

        barrier()

        # Write phase: all threads safely write their computed values
        if local_i < stride:
            cache[local_i] += temp_val

        barrier()
        stride //= 2

    # writing with local thread = 0 that has the sum for each batch
    if local_i == 0:
        output[batch, 0] = cache[0]


# ANCHOR_END: axis_sum_solution


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](BATCH)
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE)
        inp.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = row * SIZE + col

        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
        inp_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](inp)

        alias kernel = axis_sum[in_layout, out_layout]
        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](BATCH)
        expected.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    expected[row] += inp_host[row * SIZE + col]

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out)
            print("expected:", expected)
            for i in range(BATCH):
                assert_equal(out_host[i], expected[i])
