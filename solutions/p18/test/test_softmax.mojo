from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_almost_equal
from bit import log2_ceil

from op import softmax_gpu_kernel, softmax_cpu_kernel

alias SIZE = 128
alias layout = Layout.row_major(SIZE)
alias GRID_DIM_X = 1
alias BLOCK_DIM_X = 1 << log2_ceil(SIZE)
alias dtype = DType.float32


def test_softmax():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[DType.float32](SIZE)
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[DType.float32](SIZE)
        inp.enqueue_fill(0)
        # for CPU testing
        expected = ctx.enqueue_create_host_buffer[DType.float32](SIZE)
        expected.enqueue_fill(0)
        expected_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](expected)
        # Initialize input with more reasonable values
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                inp_host[i] = Float32(i)

            print("Input values:")
            for i in range(SIZE):
                print(inp_host[i], end=" ")
            print()
            # Create layout tensors for CPU calculation
            input_host_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](
                inp_host
            )

        # for GPU testing
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)
        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](inp)

        # Compute expected results using our CPU kernel
        softmax_cpu_kernel[layout, SIZE, dtype](
            expected_tensor, input_host_tensor
        )

        # Run GPU kernel
        alias kernel = softmax_gpu_kernel[layout, SIZE, dtype]
        ctx.enqueue_function_checked[kernel, kernel](
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


def main():
    test_softmax()
