from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_almost_equal
from std.bit import log2_ceil

from op import softmax_gpu_kernel, softmax_cpu_kernel

comptime SIZE = 128
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime GRID_DIM_X = 1
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)
comptime dtype = DType.float32


def test_softmax() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[DType.float32](SIZE)
        out.enqueue_fill(0)
        var inp = ctx.enqueue_create_buffer[DType.float32](SIZE)
        inp.enqueue_fill(0)
        # for CPU testing
        var expected = ctx.enqueue_create_host_buffer[DType.float32](SIZE)
        expected.enqueue_fill(0)
        var expected_tensor = TileTensor(expected, layout)
        # Initialize input with more reasonable values
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                inp_host[i] = Scalar[dtype](i)

            print("Input values:")
            for i in range(SIZE):
                print(inp_host[i], end=" ")
            print()
            # Create layout tensors for CPU calculation
            input_host_tensor = TileTensor[mut=False, dtype, LayoutType](
                inp_host, layout
            )

        # for GPU testing
        var output_tensor = TileTensor(out, layout)
        var input_tensor = TileTensor[mut=False, dtype, LayoutType](inp, layout)

        # Compute expected results using our CPU kernel
        softmax_cpu_kernel[SIZE, dtype](expected_tensor, input_host_tensor)

        # Run GPU kernel
        comptime kernel = softmax_gpu_kernel[SIZE, dtype]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=GRID_DIM_X,
            block_dim=BLOCK_DIM_X,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("GPU softmax results:")
            for i in range(SIZE):
                print(out_host[i], end=" ")
            print()

            print("Expected results:")
            for i in range(SIZE):
                print(expected[i], end=" ")
            print()

            var sum_gpu: Float32 = 0.0
            for i in range(SIZE):
                sum_gpu += out_host[i]
                assert_almost_equal(
                    out_host[i], expected[i], atol=1e-5, rtol=1e-5
                )

            print("Sum of probabilities:", sum_gpu)
            assert_almost_equal(sum_gpu, 1.0, atol=1e-5, rtol=1e-5)
            print("All tests passed 🎉")


def main() raises:
    test_softmax()
