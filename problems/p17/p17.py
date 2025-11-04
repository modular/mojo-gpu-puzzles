from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray


def conv_1d(
    input: NDArray[np.float32],
    kernel: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays and move them to the target device
    input_tensor = Tensor.from_numpy(input).to(device)
    kernel_tensor = Tensor.from_numpy(kernel).to(device)

    # Path to the directory containing our Mojo operations
    mojo_kernels = Path(__file__).parent / "op"

    # Configure our graph with the custom conv1d operation
    with Graph(
        "conv_1d_graph",
        input_types=[
            TensorType(
                dtype,
                shape=input_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=kernel_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Define inputs to the graph
        input_value, kernel_value = graph.inputs

        # The output shape is the same as the input for our 1D convolution implementation
        # Note: the name must match the name used in `@compiler.register("conv1d")` in op/conv1d.mojo
        output = ops.custom(
            name="conv1d",
            device=DeviceRef.from_device(device),
            values=[input_value, kernel_value],
            out_types=[
                TensorType(
                    dtype=input_value.tensor.dtype,
                    shape=input_value.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "input_size": input_tensor.shape[0],
                "conv_size": kernel_tensor.shape[0],
                "dtype": dtype,
            },
        )[0].tensor
        graph.output(output)

    # Compile the graph
    print("Compiling 1D convolution graph...")
    model = session.load(graph)

    # Execute the operation
    print("Executing 1D convolution...")
    result = model.execute(input_tensor, kernel_tensor)[0]

    # Copy values back to the CPU to be read
    assert isinstance(result, Tensor)
    return result.to(CPU())


if __name__ == "__main__":
    INPUT_SIZE = 15
    KERNEL_SIZE = 4

    # Place the graph on a GPU
    device = Accelerator()

    # Set up an inference session for running the graph
    session = InferenceSession(devices=[device])

    # Create test input and kernel with values that make verification easy
    input_array = np.arange(INPUT_SIZE, dtype=np.float32)
    kernel = np.arange(KERNEL_SIZE, dtype=np.float32)

    # Calculate expected result using NumPy
    expected_result = np.zeros_like(input_array, dtype=np.float32)
    for i in range(INPUT_SIZE):
        for j in range(KERNEL_SIZE):
            if i + j < INPUT_SIZE:
                expected_result[i] += input_array[i + j] * kernel[j]

    print(f"Input array: {input_array}")
    print(f"Convolution kernel: {kernel}")

    print(f"Expected result (NumPy calculation): {expected_result}")

    result = conv_1d(input_array, kernel, session, device)
    print(f"1D Convolution result (custom Mojo kernel): {result.to_numpy()}")

    # Verify results match
    np.testing.assert_allclose(result.to_numpy(), expected_result, rtol=1e-5)
    print("Verification passed: Custom kernel results match NumPy calculation")
