from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray
from scipy.special import softmax as scipy_softmax


def softmax(
    input: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32
    input_tensor = Tensor.from_numpy(input).to(device)
    mojo_kernels = Path(__file__).parent / "op"
    # ANCHOR: softmax_custom_op_graph_solution
    with Graph(
        "softmax_graph",
        input_types=[
            TensorType(
                dtype,
                shape=input_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        input_value = graph.inputs[0]

        # The output shape is the same as the input for softmax
        # Note: the name must match the name used in `@compiler.register("softmax")` in op/softmax.mojo
        output = ops.custom(
            name="softmax",
            values=[input_value],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(
                    dtype=input_value.tensor.dtype,
                    shape=input_value.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "target": "gpu" if device == Accelerator() else "cpu",
                "input_size": input_tensor.shape[0],
                "dtype": dtype,
            },
        )[0].tensor
        graph.output(output)

    # ANCHOR_END: softmax_custom_op_graph_solution

    print(f"Compiling softmax graph on {device}")
    model = session.load(graph)
    print(f"Executing softmax on {device}")
    print("="*100)
    result = model.execute(input_tensor)[0]
    assert isinstance(result, Tensor)
    return result.to(CPU()) if device == Accelerator() else result


if __name__ == "__main__":
    INPUT_SIZE = 128
    cpu_session = InferenceSession(devices=[CPU()])
    gpu_session = InferenceSession(devices=[Accelerator()])
    input_array = np.random.randn(INPUT_SIZE).astype(np.float32)
    expected_result = scipy_softmax(input_array)

    print(f"Input shape: {input_array.shape}")
    print(f"First few random input values: {input_array[:5]}")

    cpu_result = softmax(input_array, cpu_session, CPU())
    gpu_result = softmax(input_array, gpu_session, Accelerator())
    print(f"First few softmax results on CPU (custom Mojo kernel): {cpu_result.to_numpy()[:5]}")
    print(f"First few softmax results on GPU (custom Mojo kernel): {gpu_result.to_numpy()[:5]}")
    print(f"First few expected results (SciPy calculation): {expected_result[:5]}")

    np.testing.assert_allclose(cpu_result.to_numpy(), expected_result, rtol=1e-5)
    print("Verification passed: Custom kernel results match SciPy calculation")

    total_prob_cpu = np.round(np.sum(cpu_result.to_numpy()), 5)
    total_prob_gpu = np.round(np.sum(gpu_result.to_numpy()), 5)
    print(f"Sum of all probabilities on CPU: {total_prob_cpu}")
    print(f"Sum of all probabilities on GPU: {total_prob_gpu}")
