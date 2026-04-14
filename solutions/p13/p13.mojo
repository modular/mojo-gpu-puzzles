from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from std.sys import argv
from std.testing import assert_equal

comptime TPB = 8
comptime SIZE = 6
comptime CONV = 3
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = row_major[SIZE]()
comptime out_layout = row_major[SIZE]()
comptime conv_layout = row_major[CONV]()
comptime InLayout = type_of(in_layout)
comptime OutLayout = type_of(out_layout)
comptime ConvLayout = type_of(conv_layout)


# ANCHOR: conv_1d_simple_solution
def conv_1d_simple(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, InLayout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, ConvLayout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    var shared_a = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[SIZE]())
    var shared_b = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[CONV]())
    if global_i < SIZE:
        shared_a[local_i] = a[global_i]

    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    barrier()

    # Note: this is unsafe as it enforces no guard so could access `shared_a` beyond its bounds
    # local_sum = Scalar[dtype](0)
    # for j in range(CONV):
    #     if local_i + j < SIZE:
    #         local_sum += shared_a[local_i + j] * shared_b[j]

    # if global_i < SIZE:
    #     out[global_i] = local_sum

    # Safe and correct:
    if global_i < SIZE:
        # Note: using `var` allows us to include the type in the type inference
        # `out.ElementType` is available in TileTensor
        var local_sum: output.ElementType = 0

        # Note: `@parameter` decorator unrolls the loop at compile time given `CONV` is a compile-time constant
        # See: https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement
        comptime for j in range(CONV):
            # Bonus: do we need this check for this specific example with fixed SIZE, CONV
            if local_i + j < SIZE:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum


# ANCHOR_END: conv_1d_simple_solution

comptime SIZE_2 = 15
comptime CONV_2 = 4
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime in_2_layout = row_major[SIZE_2]()
comptime out_2_layout = row_major[SIZE_2]()
comptime conv_2_layout = row_major[CONV_2]()
comptime In2Layout = type_of(in_2_layout)
comptime Out2Layout = type_of(out_2_layout)
comptime Conv2Layout = type_of(conv_2_layout)


# ANCHOR: conv_1d_block_boundary_solution
def conv_1d_block_boundary(
    output: TileTensor[mut=True, dtype, Out2Layout, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, In2Layout, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, Conv2Layout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = thread_idx.x
    # first: need to account for padding
    var shared_a = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[TPB + CONV_2 - 1]())
    var shared_b = stack_allocation[
        dtype=dtype, address_space=AddressSpace.SHARED
    ](row_major[CONV_2]())
    if global_i < SIZE_2:
        shared_a[local_i] = a[global_i]
    else:
        shared_a[local_i] = 0

    # second: load elements needed for convolution at block boundary
    if local_i < CONV_2 - 1:
        # indices from next block
        var next_idx = global_i + TPB
        if next_idx < SIZE_2:
            shared_a[TPB + local_i] = a[next_idx]
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0

    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    barrier()

    if global_i < SIZE_2:
        var local_sum: output.ElementType = 0

        comptime for j in range(CONV_2):
            if global_i + j < SIZE_2:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum


# ANCHOR_END: conv_1d_block_boundary_solution


def main() raises:
    with DeviceContext() as ctx:
        var size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        var conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        var out = ctx.enqueue_create_buffer[dtype](size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](conv)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = Scalar[dtype](i)

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = Scalar[dtype](i)

        if argv()[1] == "--simple":
            var out_tensor = TileTensor(out, out_layout)
            var a_tensor = TileTensor[mut=False, dtype, InLayout](a, in_layout)
            var b_tensor = TileTensor[mut=False, dtype, ConvLayout](
                b, conv_layout
            )
            ctx.enqueue_function[conv_1d_simple, conv_1d_simple](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--block-boundary":
            var out_tensor = TileTensor(out, out_2_layout)
            var a_tensor = TileTensor[mut=False, dtype, In2Layout](
                a, in_2_layout
            )
            var b_tensor = TileTensor[mut=False, dtype, Conv2Layout](
                b, conv_2_layout
            )
            ctx.enqueue_function[
                conv_1d_block_boundary, conv_1d_block_boundary
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()
        var expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 13 complete ✅")
