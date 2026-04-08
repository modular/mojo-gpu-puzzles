from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal

comptime TPB = 8
comptime BATCH = 4
comptime SIZE = 6
comptime BLOCKS_PER_GRID = (1, BATCH)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = row_major[BATCH, SIZE]()
comptime out_layout = row_major[BATCH, 1]()
comptime InLayout = type_of(in_layout)
comptime OutLayout = type_of(out_layout)


# ANCHOR: axis_sum_solution
def axis_sum(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    size: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var batch = block_idx.y
    var cache = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB]())

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
    var stride = TPB // 2
    while stride > 0:
        # Read phase: all threads read the values they need first to avoid race conditions
        var temp_val: output.ElementType = 0
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


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[dtype](BATCH)
        out.enqueue_fill(0)
        var inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE)
        inp.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = Scalar[dtype](row * SIZE + col)

        var out_tensor = TileTensor(out, out_layout)
        var inp_tensor = TileTensor[mut=False, dtype, InLayout](inp, in_layout)

        ctx.enqueue_function[axis_sum, axis_sum](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        var expected = ctx.enqueue_create_host_buffer[dtype](BATCH)
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
            print("Puzzle 15 complete ✅")
