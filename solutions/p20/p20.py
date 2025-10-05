from typing import Optional
from pathlib import Path
import numpy as np
import torch
from max.torch import CustomOpLibrary


def conv1d_pytorch(
    input_tensor: torch.Tensor, kernel_tensor: torch.Tensor
) -> torch.Tensor:
    """
    1D convolution using our custom PyTorch operation.

    This demonstrates the transition from MAX Graph (p15) to PyTorch CustomOpLibrary.
    Uses the EXACT same Mojo kernel, but different Python integration!
    """
    # Load our custom operations
    mojo_kernels = Path(__file__).parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    # Create output tensor with same shape as input
    output_tensor = torch.empty_like(input_tensor)

    # ANCHOR: conv1d_pytorch_call
    # Call our custom conv1d operation with explicit output tensor
    # The Mojo signature expects: (out, input, kernel)
    conv1d = ops.conv1d[
        {
            "input_size": input_tensor.shape[0],
            "conv_size": kernel_tensor.shape[0],
        }
    ]
    torch.compile(conv1d)(output_tensor, input_tensor, kernel_tensor)
    # ANCHOR_END: conv1d_pytorch_call

    return output_tensor


def conv1d_max_graph_reference(
    input_array: np.ndarray,
    kernel_array: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Reference implementation using MAX Graph (like p15) for comparison.
    This shows the difference between MAX Graph and PyTorch approaches.
    """
    from max.driver import CPU, Accelerator, Tensor, accelerator_count
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, TensorType, ops

    # Use the same device logic as p15
    if device is None:
        device_obj = CPU() if accelerator_count() == 0 else Accelerator()
    else:
        device_obj = CPU() if device == "cpu" else Accelerator()

    session = InferenceSession(devices=[device_obj])

    # Convert to MAX Graph tensors
    input_tensor = Tensor.from_numpy(input_array).to(device_obj)
    kernel_tensor = Tensor.from_numpy(kernel_array).to(device_obj)

    # Same graph setup as p15
    with Graph(
        "conv_1d_reference_graph",
        input_types=[
            TensorType(
                DType.float32,
                shape=input_tensor.shape,
                device=DeviceRef.from_device(device_obj),
            ),
            TensorType(
                DType.float32,
                shape=kernel_tensor.shape,
                device=DeviceRef.from_device(device_obj),
            ),
        ],
        custom_extensions=[Path(__file__).parent / "op"],
    ) as graph:
        input_value, kernel_value = graph.inputs
        output = ops.custom(
            name="conv1d",
            values=[input_value, kernel_value],
            device=DeviceRef.from_device(device_obj),
            out_types=[
                TensorType(
                    dtype=input_value.tensor.dtype,
                    shape=input_value.tensor.shape,
                    device=DeviceRef.from_device(device_obj),
                )
            ],
            parameters={
                "input_size": input_tensor.shape[0],
                "conv_size": kernel_tensor.shape[0],
                "dtype": DType.float32,
            },
        )[0].tensor
        graph.output(output)

    model = session.load(graph)
    result = model.execute(input_tensor, kernel_tensor)[0]
    return result.to(CPU()).to_numpy()


def compute_numpy_reference(
    input_array: np.ndarray, kernel_array: np.ndarray
) -> np.ndarray:
    """NumPy reference implementation for verification."""
    INPUT_SIZE = len(input_array)
    KERNEL_SIZE = len(kernel_array)

    expected_result = np.zeros_like(input_array, dtype=np.float32)
    for i in range(INPUT_SIZE):
        for j in range(KERNEL_SIZE):
            if i + j < INPUT_SIZE:
                expected_result[i] += input_array[i + j] * kernel_array[j]
    return expected_result


if __name__ == "__main__":
    INPUT_SIZE = 15
    KERNEL_SIZE = 4

    # Create test data (same as p15 for easy comparison)
    input_array = np.arange(INPUT_SIZE, dtype=np.float32)
    kernel_array = np.arange(KERNEL_SIZE, dtype=np.float32)

    print("Puzzle 20: From MAX Graph to PyTorch Custom Ops")
    print("=" * 60)
    print(f"Input array: {input_array}")
    print(f"Convolution kernel: {kernel_array}")
    print()

    numpy_result = compute_numpy_reference(input_array, kernel_array)
    print(f"NumPy reference result: {numpy_result}")
    print()

    device = "cuda"
    input_tensor = torch.from_numpy(input_array).to(device)
    kernel_tensor = torch.from_numpy(kernel_array).to(device)

    print(f"Testing PyTorch Custom Op (device: {device})")
    print("-" * 40)

    try:
        pytorch_result = conv1d_pytorch(input_tensor, kernel_tensor)
        pytorch_result_cpu = pytorch_result.cpu().numpy()
        print(f"PyTorch custom op result: {pytorch_result_cpu}")

        # Verify PyTorch result
        np.testing.assert_allclose(pytorch_result_cpu, numpy_result, rtol=1e-5)
        print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"PyTorch custom op failed: {e}")
        pytorch_result_cpu = None

    print()

    # Compare with MAX Graph approach (like p15)
    print("Comparing with MAX Graph approach (like p15)")
    print("-" * 40)

    try:
        max_graph_result = conv1d_max_graph_reference(input_array, kernel_array)
        print(f"MAX Graph result: {max_graph_result}")

        # Verify MAX Graph result
        np.testing.assert_allclose(max_graph_result, numpy_result, rtol=1e-5)
        print("✅ MAX Graph verification PASSED")

        if pytorch_result_cpu is not None:
            np.testing.assert_allclose(
                pytorch_result_cpu, max_graph_result, rtol=1e-5
            )
            print("✅ PyTorch and MAX Graph results MATCH")

    except Exception as e:
        print(f"MAX Graph comparison failed: {e}")
