from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from testing import assert_equal
from sys import argv

# ANCHOR: shared_memory_race

alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn shared_memory_race(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x

    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if row < size and col < size:
        shared_sum[0] += a[row, col]

    barrier()

    if row < size and col < size:
        output[row, col] = shared_sum[0]


# ANCHOR_END: shared_memory_race


# ANCHOR: add_10_2d_no_guard
fn add_10_2d(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_2d_no_guard


def main():
    if len(argv()) != 2:
        print(
            "Expected one command-line argument: '--memory-bug' or"
            " '--race-condition'"
        )
        return

    flag = argv()[1]

    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_buf.enqueue_fill(0)
        out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out_buf).reshape[
            layout
        ]()
        print("out shape:", out_tensor.shape[0](), "x", out_tensor.shape[1]())
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)

        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        a.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE * SIZE):
                a_host[i] = i

        a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a).reshape[
            layout
        ]()

        if flag == "--memory-bug":
            print("Running memory bug example (bounds checking issue)...")
            # Fill expected values directly since it's a HostBuffer
            for i in range(SIZE * SIZE):
                expected[i] = i + 10

            ctx.enqueue_function_checked[add_10_2d, add_10_2d](
                out_tensor,
                a_tensor,
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with out_buf.map_to_host() as out_buf_host:
                print("out:", out_buf_host)
                print("expected:", expected)
                for i in range(SIZE * SIZE):
                    assert_equal(out_buf_host[i], expected[i])
                print(
                    "✅ Memory test PASSED! (memcheck may find bounds"
                    " violations)"
                )

        elif flag == "--race-condition":
            print("Running race condition example...")
            total_sum = Scalar[dtype](0.0)
            with a.map_to_host() as a_host:
                for i in range(SIZE * SIZE):
                    total_sum += a_host[i]  # Sum: 0 + 1 + 2 + 3 = 6

            # All positions should contain the total sum
            for i in range(SIZE * SIZE):
                expected[i] = total_sum

            ctx.enqueue_function_checked[
                shared_memory_race, shared_memory_race
            ](
                out_tensor,
                a_tensor,
                SIZE,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            with out_buf.map_to_host() as out_buf_host:
                print("out:", out_buf_host)
                print("expected:", expected)
                for i in range(SIZE * SIZE):
                    assert_equal(out_buf_host[i], expected[i])

                print(
                    "✅ Race condition test PASSED! (racecheck will find"
                    " hazards)"
                )

        else:
            print("Unknown flag:", flag)
            print("Available flags: --memory-bug, --race-condition")
