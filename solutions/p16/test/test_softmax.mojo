from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_almost_equal

from op import softmax_gpu_kernel, softmax_cpu_kernel

alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)
alias dtype = DType.float32


def test_softmax():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[DType.float32](SIZE).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[DType.float32](SIZE).enqueue_fill(0)
        # for CPU testing
        expected = ctx.enqueue_create_host_buffer[DType.float32](
            SIZE
        ).enqueue_fill(0)

        # Initialize input with more reasonable values
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                inp_host[i] = Float32(i)

            print("Input values:")
            for i in range(SIZE):
                print(inp_host[i], end=" ")
            print()

        # Create layout tensors for CPU calculation
        input_host_tensor = LayoutTensor[mut=True, dtype, layout](
            inp_host.unsafe_ptr()
        )
        expected_tensor = LayoutTensor[mut=True, dtype, layout](
            expected.unsafe_ptr()
        )
        # for GPU testing
        output_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=True, dtype, layout](inp.unsafe_ptr())

        # Compute expected results using our CPU kernel
        softmax_cpu_kernel[layout, SIZE, dtype](
            expected_tensor, input_host_tensor
        )

        # Run GPU kernel
        ctx.enqueue_function[softmax_gpu_kernel[layout, SIZE, dtype]](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
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


# def main():
#     test_softmax()
