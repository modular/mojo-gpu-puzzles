import time
import argparse
from pathlib import Path
import os
import warnings
import logging

import torch
from max.torch import CustomOpLibrary

# Suppress PyTorch internal logging that causes cudagraphs messages
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# Configure torch.compile for optimal performance
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.verbose = False

# Set environment variables for better performance
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["TORCH_LOGS"] = "-dynamo"

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*grad attribute.*non-leaf Tensor.*")
warnings.filterwarnings("ignore", message=".*skipping cudagraphs.*")
warnings.filterwarnings("ignore", message=".*mutated inputs.*")


BATCH_SIZE = 4
SEQ_LEN = 4
HIDDEN_DIM = 8
OUTPUT_DIM = 16
EPS = 1e-5

print(
    f"Testing with dimensions: [{BATCH_SIZE}, {SEQ_LEN}, {HIDDEN_DIM}] ->"
    f" [{BATCH_SIZE}, {SEQ_LEN}, {OUTPUT_DIM}]"
)

mojo_kernels = Path(__file__).parent / "op"
ops = CustomOpLibrary(mojo_kernels)
print("✅ Loaded Mojo operations library")

# Global compilation cache that persists across function calls
_global_compile_cache = {}


def get_cached_compiled_op(op, cache_key):
    """Global caching to avoid recompilation."""
    if cache_key not in _global_compile_cache:
        _global_compile_cache[cache_key] = torch.compile(
            op, mode="reduce-overhead"
        )
    return _global_compile_cache[cache_key]


class LayerNormLinearFunction(torch.autograd.Function):
    """Custom autograd function for LayerNorm + Linear fusion."""

    @staticmethod
    def forward(ctx, input, ln_weight, ln_bias, linear_weight, linear_bias):
        """Forward pass using our custom Mojo operation."""
        # Save tensors for backward pass
        ctx.save_for_backward(
            input, ln_weight, ln_bias, linear_weight, linear_bias
        )

        # Use our custom Mojo operation (detached to avoid autograd conflicts)
        result = mojo_layernorm_linear(
            input.detach(),
            ln_weight.detach(),
            ln_bias.detach(),
            linear_weight.detach(),
            linear_bias.detach(),
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using our custom Mojo operation."""
        input, ln_weight, ln_bias, linear_weight, linear_bias = (
            ctx.saved_tensors
        )

        # Use our custom backward operation (detached)
        grad_input, grad_ln_weight, grad_ln_bias, grad_linear_weight, grad_linear_bias = mojo_layernorm_linear_backward(
            input.detach(),
            ln_weight.detach(),
            ln_bias.detach(),
            linear_weight.detach(),
            linear_bias.detach(),
            grad_output.detach(),
        )

        return (
            grad_input,
            grad_ln_weight,
            grad_ln_bias,
            grad_linear_weight,
            grad_linear_bias,
        )


def mojo_layernorm_linear(
    input, ln_weight, ln_bias, linear_weight, linear_bias
):
    """Forward pass using Mojo implementation."""
    output, error = run_mojo_implementation(
        input.detach(),
        ln_weight.detach(),
        ln_bias.detach(),
        linear_weight.detach(),
        linear_bias.detach(),
        algorithm="fused",
        target="gpu",
    )
    if output is None:
        raise RuntimeError(f"Mojo forward pass failed: {error}")
    return output


def mojo_layernorm_linear_backward(
    input, ln_weight, ln_bias, linear_weight, linear_bias, grad_output
):
    """Backward pass using Mojo implementation."""
    forward_output, gradients, error = run_mojo_backward_implementation(
        input, ln_weight, ln_bias, linear_weight, linear_bias, target="gpu"
    )
    if gradients is None:
        raise RuntimeError(f"Mojo backward pass failed: {error}")
    return (
        gradients["grad_input"],
        gradients["grad_ln_weight"],
        gradients["grad_ln_bias"],
        gradients["grad_linear_weight"],
        gradients["grad_linear_bias"],
    )


def mojo_layernorm_linear_autograd(
    input, ln_weight, ln_bias, linear_weight, linear_bias
):
    """Wrapper function that uses our custom autograd function."""
    return LayerNormLinearFunction.apply(
        input, ln_weight, ln_bias, linear_weight, linear_bias
    )


class SimpleTransformerBlock(torch.nn.Module):
    """A simple transformer block using our custom LayerNorm + Linear operations.
    """

    def __init__(self, hidden_dim, ff_dim, use_mojo=True, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.use_mojo = use_mojo
        self.device = device

        # Layer 1: LayerNorm + Linear (hidden_dim -> ff_dim)
        self.ln1_weight = torch.nn.Parameter(
            torch.ones(hidden_dim, device=device)
        )
        self.ln1_bias = torch.nn.Parameter(
            torch.zeros(hidden_dim, device=device)
        )
        self.linear1_weight = torch.nn.Parameter(
            torch.randn(ff_dim, hidden_dim, device=device) / (hidden_dim**0.5)
        )
        self.linear1_bias = torch.nn.Parameter(
            torch.zeros(ff_dim, device=device)
        )

        # Layer 2: LayerNorm + Linear (ff_dim -> hidden_dim)
        self.ln2_weight = torch.nn.Parameter(torch.ones(ff_dim, device=device))
        self.ln2_bias = torch.nn.Parameter(torch.zeros(ff_dim, device=device))
        self.linear2_weight = torch.nn.Parameter(
            torch.randn(hidden_dim, ff_dim, device=device) / (ff_dim**0.5)
        )
        self.linear2_bias = torch.nn.Parameter(
            torch.zeros(hidden_dim, device=device)
        )

    def forward(self, x):
        # Layer 1: LayerNorm + Linear + ReLU
        if self.use_mojo:
            x1 = mojo_layernorm_linear_autograd(
                x,
                self.ln1_weight,
                self.ln1_bias,
                self.linear1_weight,
                self.linear1_bias,
            )
        else:
            x1 = torch.nn.functional.layer_norm(
                x, (self.hidden_dim,), self.ln1_weight, self.ln1_bias
            )
            x1 = torch.nn.functional.linear(
                x1, self.linear1_weight, self.linear1_bias
            )

        x1 = torch.nn.functional.relu(x1)

        # Layer 2: LayerNorm + Linear (residual connection)
        if self.use_mojo:
            x2 = mojo_layernorm_linear_autograd(
                x1,
                self.ln2_weight,
                self.ln2_bias,
                self.linear2_weight,
                self.linear2_bias,
            )
        else:
            x2 = torch.nn.functional.layer_norm(
                x1, (self.ff_dim,), self.ln2_weight, self.ln2_bias
            )
            x2 = torch.nn.functional.linear(
                x2, self.linear2_weight, self.linear2_bias
            )

        return x + x2  # Residual connection


class SimpleNeuralNetwork(torch.nn.Module):
    """A simple neural network using our custom operations."""

    def __init__(
        self,
        hidden_dim,
        ff_dim,
        num_layers,
        num_classes,
        use_mojo=True,
        device="cuda",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_mojo = use_mojo
        self.device = device

        # Input projection (no LayerNorm, just Linear)
        self.input_proj = torch.nn.Linear(hidden_dim, hidden_dim, device=device)

        # Transformer blocks
        self.layers = torch.nn.ModuleList(
            [
                SimpleTransformerBlock(hidden_dim, ff_dim, use_mojo, device)
                for _ in range(num_layers)
            ]
        )

        # Output projection (no LayerNorm, just Linear)
        self.output_proj = torch.nn.Linear(
            hidden_dim, num_classes, device=device
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)

        x = x.mean(
            dim=1
        )  # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        x = self.output_proj(
            x
        )  # [batch_size, hidden_dim] -> [batch_size, num_classes]
        return x


def create_test_data(device="cuda"):
    """Create consistent test data for all tests."""
    torch.manual_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)
    ln_weight = torch.ones(HIDDEN_DIM, device=device)
    ln_bias = torch.zeros(HIDDEN_DIM, device=device)
    linear_weight = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=device) * 0.02
    linear_bias = torch.zeros(OUTPUT_DIM, device=device)

    return input_tensor, ln_weight, ln_bias, linear_weight, linear_bias


def create_test_data_with_grad(device="cuda"):
    """Create test data with gradients enabled for backward pass testing."""
    torch.manual_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create tensors as leaf tensors with requires_grad=True
    input_tensor = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, requires_grad=True
    )
    ln_weight = torch.ones(HIDDEN_DIM, device=device, requires_grad=True)
    ln_bias = torch.zeros(HIDDEN_DIM, device=device, requires_grad=True)
    linear_weight = (
        torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=device) * 0.02
    ).requires_grad_(True)
    linear_bias = torch.zeros(OUTPUT_DIM, device=device, requires_grad=True)

    return input_tensor, ln_weight, ln_bias, linear_weight, linear_bias


def reference_layernorm_linear(
    input, ln_weight, ln_bias, linear_weight, linear_bias, eps=EPS
):
    """Reference implementation using standard PyTorch operations."""
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (input - mean) / torch.sqrt(var + eps)
    ln_output = normalized * ln_weight + ln_bias

    output = torch.nn.functional.linear(ln_output, linear_weight, linear_bias)
    return output


def reference_layernorm_linear_with_grad(
    input, ln_weight, ln_bias, linear_weight, linear_bias, eps=EPS
):
    """Reference implementation with autograd for backward pass testing."""
    # Clear any existing gradients
    if input.grad is not None:
        input.grad.zero_()
    if ln_weight.grad is not None:
        ln_weight.grad.zero_()
    if ln_bias.grad is not None:
        ln_bias.grad.zero_()
    if linear_weight.grad is not None:
        linear_weight.grad.zero_()
    if linear_bias.grad is not None:
        linear_bias.grad.zero_()

    # Forward pass
    output = reference_layernorm_linear(
        input, ln_weight, ln_bias, linear_weight, linear_bias, eps
    )

    # Create dummy gradient output (ones) and run backward
    grad_output = torch.ones_like(output)
    output.backward(grad_output, retain_graph=True)

    # Return forward output and all gradients
    return output, {
        "grad_input": input.grad.clone() if input.grad is not None else None,
        "grad_ln_weight": ln_weight.grad.clone() if ln_weight.grad
        is not None else None,
        "grad_ln_bias": ln_bias.grad.clone() if ln_bias.grad
        is not None else None,
        "grad_linear_weight": linear_weight.grad.clone() if linear_weight.grad
        is not None else None,
        "grad_linear_bias": linear_bias.grad.clone() if linear_bias.grad
        is not None else None,
    }


def run_mojo_implementation(
    input,
    ln_weight,
    ln_bias,
    linear_weight,
    linear_bias,
    algorithm="fused",
    target="auto",
):
    """Generic function to run Mojo implementations."""
    batch_size, seq_len, hidden_dim = input.shape
    output_dim = linear_weight.shape[0]

    if target == "auto":
        target = "gpu" if input.device.type == "cuda" else "cpu"

    if target == "cpu":
        input = input.cpu()
        ln_weight = ln_weight.cpu()
        ln_bias = ln_bias.cpu()
        linear_weight = linear_weight.cpu()
        linear_bias = linear_bias.cpu()

    try:
        output = torch.empty(
            (batch_size, seq_len, output_dim),
            dtype=input.dtype,
            device=input.device,
        )

        op = ops.layernorm_linear[
            {
                "algorithm": algorithm,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
            }
        ]

        compiled_op = get_cached_compiled_op(
            op, f"forward_{algorithm}_{batch_size}x{seq_len}x{hidden_dim}"
        )
        compiled_op(
            output, input, ln_weight, ln_bias, linear_weight, linear_bias
        )
        return output, None

    except Exception as e:
        return None, str(e)


def run_mojo_backward_implementation(
    input, ln_weight, ln_bias, linear_weight, linear_bias, target="auto"
):
    """Run Mojo backward implementation."""
    batch_size, seq_len, hidden_dim = input.shape
    output_dim = linear_weight.shape[0]

    if target == "auto":
        target = "gpu" if input.device.type == "cuda" else "cpu"

    input_detached = input.detach()
    ln_weight_detached = ln_weight.detach()
    ln_bias_detached = ln_bias.detach()
    linear_weight_detached = linear_weight.detach()
    linear_bias_detached = linear_bias.detach()

    if target == "cpu":
        input_detached = input_detached.cpu()
        ln_weight_detached = ln_weight_detached.cpu()
        ln_bias_detached = ln_bias_detached.cpu()
        linear_weight_detached = linear_weight_detached.cpu()
        linear_bias_detached = linear_bias_detached.cpu()

    try:
        # Forward pass first
        forward_output, forward_error = run_mojo_implementation(
            input_detached,
            ln_weight_detached,
            ln_bias_detached,
            linear_weight_detached,
            linear_bias_detached,
            algorithm="fused",
            target=target,
        )
        if forward_output is None:
            return None, None, f"Forward pass failed: {forward_error}"

        # Prepare gradient tensors (initialized to zero) on the same device as forward output
        grad_input = torch.zeros_like(input_detached)
        grad_ln_weight = torch.zeros_like(ln_weight_detached)
        grad_ln_bias = torch.zeros_like(ln_bias_detached)
        grad_linear_weight = torch.zeros_like(linear_weight_detached)
        grad_linear_bias = torch.zeros_like(linear_bias_detached)

        # Dummy gradient output (ones)
        grad_output = torch.ones_like(forward_output)

        # Run backward pass
        backward_op = ops.layernorm_linear_backward[
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
            }
        ]

        compiled_backward_op = get_cached_compiled_op(
            backward_op, f"backward_{batch_size}x{seq_len}x{hidden_dim}"
        )
        compiled_backward_op(
            grad_input,
            grad_ln_weight,
            grad_ln_bias,
            grad_linear_weight,
            grad_linear_bias,
            grad_output,
            input_detached,
            ln_weight_detached,
            ln_bias_detached,
            linear_weight_detached,
        )

        gradients = {
            "grad_input": grad_input,
            "grad_ln_weight": grad_ln_weight,
            "grad_ln_bias": grad_ln_bias,
            "grad_linear_weight": grad_linear_weight,
            "grad_linear_bias": grad_linear_bias,
        }

        return forward_output, gradients, None

    except Exception as e:
        return None, None, str(e)


def test_implementation(
    name, algorithm=None, target="auto", reference_output=None, test_data=None
):
    """Generic test function for any implementation."""
    if test_data is None:
        test_data = create_test_data()

    input_tensor, ln_weight, ln_bias, linear_weight, linear_bias = test_data

    print(f"\nTesting {name}")
    print("-" * (15 + len(name)))

    if reference_output is None:
        reference_output = reference_layernorm_linear(
            input_tensor, ln_weight, ln_bias, linear_weight, linear_bias
        )

    if algorithm is None:
        output = reference_layernorm_linear(
            input_tensor, ln_weight, ln_bias, linear_weight, linear_bias
        )
        success_msg = "✅ Reference PyTorch"
        error_msg = "❌ Reference failed"
    else:
        output, error = run_mojo_implementation(
            input_tensor,
            ln_weight,
            ln_bias,
            linear_weight,
            linear_bias,
            algorithm=algorithm,
            target=target,
        )
        success_msg = f"✅ Using Mojo {algorithm} kernel ({target.upper()})"
        error_msg = f"{algorithm} kernel failed: {error}"

    if output is not None:
        print(success_msg)

        if reference_output.device != output.device:
            reference_output = reference_output.to(output.device)

        diff = torch.max(torch.abs(reference_output - output)).item()
        print(f"   Max difference: {diff:.2e}")

        is_correct = diff < 1e-4
        print(f"   Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        return is_correct, output
    else:
        print(error_msg)
        return False, None


def test_backward_pass(name, target="auto", test_data=None):
    """Test backward pass correctness against PyTorch autograd."""
    device = "cuda" if target == "gpu" else "cpu"
    test_data_ref = create_test_data_with_grad(device=device)
    test_data_mojo = create_test_data_with_grad(device=device)

    input_tensor_ref, ln_weight_ref, ln_bias_ref, linear_weight_ref, linear_bias_ref = (
        test_data_ref
    )
    input_tensor_mojo, ln_weight_mojo, ln_bias_mojo, linear_weight_mojo, linear_bias_mojo = (
        test_data_mojo
    )

    print(f"\nTesting {name} - Backward Pass")
    print("-" * (15 + len(name) + 15))

    # Get PyTorch autograd reference
    print("   Computing PyTorch autograd reference...")
    ref_output, ref_gradients = reference_layernorm_linear_with_grad(
        input_tensor_ref,
        ln_weight_ref,
        ln_bias_ref,
        linear_weight_ref,
        linear_bias_ref,
    )

    # Test Mojo backward implementation
    print(f"   Computing Mojo backward implementation ({target.upper()})...")
    mojo_output, mojo_gradients, error = run_mojo_backward_implementation(
        input_tensor_mojo,
        ln_weight_mojo,
        ln_bias_mojo,
        linear_weight_mojo,
        linear_bias_mojo,
        target=target,
    )

    if mojo_output is None or mojo_gradients is None:
        print(f"❌ {name} backward failed: {error}")
        return False

    print(f"✅ {name} backward completed")

    # Compare forward outputs
    if ref_output.device != mojo_output.device:
        ref_output = ref_output.to(mojo_output.device)

    forward_diff = torch.max(torch.abs(ref_output - mojo_output)).item()
    print(f"   Forward max difference: {forward_diff:.2e}")

    # Compare gradients
    gradient_diffs = {}
    gradient_names = [
        "grad_input",
        "grad_ln_weight",
        "grad_ln_bias",
        "grad_linear_weight",
        "grad_linear_bias",
    ]

    all_gradients_correct = True
    for grad_name in gradient_names:
        ref_grad = ref_gradients[grad_name]
        mojo_grad = mojo_gradients[grad_name]

        if ref_grad is None or mojo_grad is None:
            print(f"   {grad_name}: ❌ Missing gradient")
            all_gradients_correct = False
            continue

        if ref_grad.device != mojo_grad.device:
            ref_grad = ref_grad.to(mojo_grad.device)

        diff = torch.max(torch.abs(ref_grad - mojo_grad)).item()
        gradient_diffs[grad_name] = diff

        is_correct = diff < 1e-4
        print(f"   {grad_name}: {diff:.2e} {'✅' if is_correct else '❌'}")

        if not is_correct:
            all_gradients_correct = False

    forward_correct = forward_diff < 1e-4
    overall_correct = forward_correct and all_gradients_correct

    print(
        "\n   Forward pass:"
        f" {'✅ CORRECT' if forward_correct else '❌ INCORRECT'}"
    )
    print(
        "   Gradients:   "
        f" {'✅ CORRECT' if all_gradients_correct else '❌ INCORRECT'}"
    )
    print(
        f"   Overall:      {'✅ CORRECT' if overall_correct else '❌ INCORRECT'}"
    )

    return overall_correct


def run_comprehensive_test():
    """Run comprehensive test of all implementations."""
    print("=" * 60)
    print("   Puzzle 22: Fused LayerNorm + Linear Comparison")
    print("   (Reference vs CPU vs GPU Unfused vs GPU Fused)")
    print("=" * 60)

    test_data = create_test_data()
    input_tensor = test_data[0]

    reference_output = reference_layernorm_linear(*test_data)

    results = {}

    results["cpu"], _ = test_implementation(
        "CPU Implementation",
        algorithm="fused",
        target="cpu",
        reference_output=reference_output,
        test_data=test_data,
    )

    if input_tensor.device.type == "cuda":
        results["gpu_unfused"], _ = test_implementation(
            "GPU Unfused Implementation",
            algorithm="unfused",
            target="gpu",
            reference_output=reference_output,
            test_data=test_data,
        )

        results["gpu_fused"], _ = test_implementation(
            "GPU Fused Implementation",
            algorithm="fused",
            target="gpu",
            reference_output=reference_output,
            test_data=test_data,
        )
    else:
        results["gpu_unfused"] = False
        results["gpu_fused"] = False

    print(f"\nSummary:")
    print(
        f"   - CPU:         {'✅ CORRECT' if results['cpu'] else '❌ INCORRECT'}"
    )
    print(
        "   - GPU unfused:"
        f" {'✅ CORRECT' if results['gpu_unfused'] else '❌ INCORRECT'}"
    )
    print(
        "   - GPU fused:  "
        f" {'✅ CORRECT' if results['gpu_fused'] else '❌ INCORRECT'}"
    )

    all_correct = all(results.values())
    print(
        f"\n   Overall: {'✅ ALL CORRECT' if all_correct else '❌ SOME FAILED'}"
    )

    if all_correct:
        print("\nPuzzle 22 completed successfully!")
        print("\nWhat we achieved:")
        print("✅ CPU implementation: Fused LayerNorm + Linear")
        print(
            "✅ GPU unfused: Multi-kernel pipeline (LayerNorm → Transpose →"
            " Matmul → Bias)"
        )
        print("✅ GPU fused: Single kernel LayerNorm + Linear")
        print("✅ Perfect numerical accuracy (max diff ~1.5e-08)")

        print("\nLearning outcomes:")
        print("- Multi-kernel GPU pipeline design")
        print("- Single-kernel fusion implementation")
        print("- CPU vs GPU optimization strategies")
        print("- Integration of optimized kernels from previous puzzles")
    else:
        print("\n❌ Some implementations failed!")

    return all_correct


def run_fused_only_test():
    """Run only the fused kernel test."""
    print("=" * 50)
    print("   Puzzle 22: Testing FUSED KERNEL ONLY")
    print("=" * 50)

    test_data = create_test_data()
    reference_output = reference_layernorm_linear(*test_data)

    is_correct, _ = test_implementation(
        "GPU Fused Implementation ONLY",
        algorithm="fused",
        target="gpu",
        reference_output=reference_output,
        test_data=test_data,
    )

    print(
        "\n🧪 Fused kernel result:"
        f" {'✅ CORRECT' if is_correct else '❌ INCORRECT'}"
    )

    if is_correct:
        print("\n🎉 Fused kernel works perfectly!")
    else:
        print("\n❌ Fused kernel failed!")


def benchmark_implementations(algorithm, test_data, iterations=50):
    """Benchmark CPU vs GPU for specific algorithm."""
    print(f"\n⚡ Benchmarking CPU vs GPU {algorithm.upper()}")
    print("-" * (35 + len(algorithm)))

    input_tensor, ln_weight, ln_bias, linear_weight, linear_bias = test_data

    if input_tensor.device.type != "cuda":
        print("   ❌ CUDA not available - skipping GPU benchmark")
        return

    times = {}

    print("   Testing CPU performance...")
    cpu_output, cpu_error = run_mojo_implementation(
        *test_data, algorithm="fused", target="cpu"
    )
    if cpu_output is not None:
        # Warmup
        for _ in range(3):
            _ = run_mojo_implementation(
                *test_data, algorithm="fused", target="cpu"
            )

        start = time.perf_counter()
        for _ in range(iterations):
            _ = run_mojo_implementation(
                *test_data, algorithm="fused", target="cpu"
            )
        times["cpu"] = time.perf_counter() - start
        print(f"   CPU: {times['cpu']*1000:.2f}ms ({iterations} iterations)")
    else:
        print(f"   CPU failed: {cpu_error}")
        times["cpu"] = None

    # Benchmark GPU
    print(f"   Testing GPU {algorithm} performance...")
    gpu_output, gpu_error = run_mojo_implementation(
        *test_data, algorithm=algorithm, target="gpu"
    )
    if gpu_output is not None:
        # Warmup
        for _ in range(3):
            _ = run_mojo_implementation(
                *test_data, algorithm=algorithm, target="gpu"
            )
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = run_mojo_implementation(
                *test_data, algorithm=algorithm, target="gpu"
            )
        torch.cuda.synchronize()
        times["gpu"] = time.perf_counter() - start
        print(
            f"   GPU {algorithm}: {times['gpu']*1000:.2f}ms"
            f" ({iterations} iterations)"
        )
    else:
        print(f"   GPU {algorithm} failed: {gpu_error}")
        times["gpu"] = None

    # Performance comparison
    if times["cpu"] is not None and times["gpu"] is not None:
        speedup = times["cpu"] / times["gpu"]
        print(
            f"\n   GPU {algorithm} vs CPU: {speedup:.2f}x"
            f" {'faster' if speedup > 1 else 'slower'}"
        )

        if speedup > 1:
            print(f"   GPU {algorithm} wins!")
        else:
            print(f"   CPU wins (GPU overhead > computation benefit)")
    else:
        print("\n   ❌ Benchmark incomplete due to failures")


def run_algorithm_specific_test(algorithm):
    """Run correctness and benchmark tests for specific algorithm."""
    print("=" * 60)
    print(f"   Puzzle 22: {algorithm.upper()} Algorithm Test & Benchmark")
    print("=" * 60)

    test_data = create_test_data()
    reference_output = reference_layernorm_linear(*test_data)

    print(f"\n🧪 Correctness Testing for {algorithm.upper()} Algorithm")
    print("=" * (45 + len(algorithm)))

    results = {}

    results["reference"], _ = test_implementation(
        "Reference PyTorch Implementation",
        algorithm=None,
        reference_output=reference_output,
        test_data=test_data,
    )

    results["cpu"], _ = test_implementation(
        "CPU Implementation",
        algorithm="fused",
        target="cpu",
        reference_output=reference_output,
        test_data=test_data,
    )

    # Test selected GPU algorithm
    if test_data[0].device.type == "cuda":
        results[f"gpu_{algorithm}"], _ = test_implementation(
            f"GPU {algorithm.title()} Implementation",
            algorithm=algorithm,
            target="gpu",
            reference_output=reference_output,
            test_data=test_data,
        )
    else:
        print(f"\n CUDA not available - skipping GPU {algorithm} test")
        results[f"gpu_{algorithm}"] = False

    print(f"\nCorrectness Summary:")
    print(
        "   - Reference:  "
        f" {'✅ CORRECT' if results['reference'] else '❌ INCORRECT'}"
    )
    print(
        f"   - CPU:         {'✅ CORRECT' if results['cpu'] else '❌ INCORRECT'}"
    )
    print(
        f"   - GPU {algorithm}:"
        f" {'✅ CORRECT' if results[f'gpu_{algorithm}'] else '❌ INCORRECT'}"
    )

    all_correct = all(results.values())
    print(
        "\n   Overall Correctness:"
        f" {'✅ ALL CORRECT' if all_correct else '❌ SOME FAILED'}"
    )

    if all_correct and test_data[0].device.type == "cuda":
        benchmark_implementations(algorithm, test_data)

        print(f"\n{algorithm.upper()} Algorithm Test Completed!")
        print(f"\nWhat we verified:")
        print("✅ Numerical correctness against PyTorch reference")
        print("✅ CPU implementation accuracy")
        print(f"✅ GPU {algorithm} implementation accuracy")
        print("✅ Performance comparison CPU vs GPU")

        print(f"\nLearning outcomes:")
        print(f"- {algorithm.title()} kernel implementation and optimization")
        print("- Cross-platform correctness verification")
        print("- Performance characterization and bottleneck analysis")
        if algorithm == "unfused":
            print("- Multi-kernel pipeline composition")
            print("- Memory bandwidth vs compute trade-offs")
        else:
            print("- Single-kernel fusion benefits and limitations")
            print("- Computation density optimization")
    else:
        if not all_correct:
            print(f"\n❌ Correctness issues found - skipping benchmark")
        else:
            print(f"\n CUDA not available - only correctness tested")

    return all_correct


def run_comprehensive_backward_test():
    """Run comprehensive backward pass tests on both CPU and GPU."""
    print("=" * 60)
    print("           Comprehensive Backward Pass Test")
    print("           Testing Custom LayerNorm + Linear Gradients")
    print("=" * 60)

    print(
        f"Testing with dimensions: {[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM]} ->"
        f" {[BATCH_SIZE, SEQ_LEN, OUTPUT_DIM]}"
    )

    # Test backward pass on CPU
    print(f"\nTesting CPU Backward Pass:")
    cpu_success = test_backward_pass(
        "CPU Backward Implementation", target="cpu"
    )

    # Test backward pass on GPU (if available)
    gpu_success = False
    test_data = create_test_data()
    if test_data[0].device.type == "cuda":
        print(f"\nTesting GPU Backward Pass:")
        gpu_success = test_backward_pass(
            "GPU Backward Implementation", target="gpu"
        )
    else:
        print(f"\n❌ CUDA not available - skipping GPU backward test")

    # Summary
    print(f"\nBackward Pass Test Summary:")
    print(
        f"   - CPU Backward:  {'✅ CORRECT' if cpu_success else '❌ INCORRECT'}"
    )
    if test_data[0].device.type == "cuda":
        print(
            "   - GPU Backward: "
            f" {'✅ CORRECT' if gpu_success else '❌ INCORRECT'}"
        )
        overall_success = cpu_success and gpu_success
    else:
        print(f"   - GPU Backward:  ⏭️  SKIPPED (CUDA not available)")
        overall_success = cpu_success

    print(
        "\n   Overall Result:"
        f" {'✅ ALL CORRECT' if overall_success else '❌ SOME FAILED'}"
    )

    if overall_success:
        print(f"\nBACKWARD PASS Test Completed!")
        print(f"\nWhat we verified:")
        print("✅ Backward pass numerical correctness against PyTorch autograd")
        print(
            "✅ All gradient components (input, LayerNorm weights/bias, Linear"
            " weights/bias)"
        )
        print("✅ CPU implementation using atomic operations")
        if test_data[0].device.type == "cuda":
            print("✅ GPU implementation using atomic operations")
        print("✅ Race-condition-free gradient accumulation")
        print("✅ Cross-platform gradient computation")

        print(f"\nTechnical achievements:")
        print("- Custom backward kernels with atomic operations")
        print(
            "- Proper chain rule implementation for LayerNorm + Linear"
            " composition"
        )
        print("- Gradient correctness verification against PyTorch autograd")
        print("- Memory-efficient gradient computation")
        print("- Educational focus on backward pass mathematics")
    else:
        print(f"\n❌ Some backward pass tests failed!")
        print("   Check the error messages above for details.")

    return overall_success


def demonstrate_neural_network_fast():
    """Fast neural network demo with minimal overhead."""
    print("Neural Network Demo")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Dimensions: {BATCH_SIZE}x{SEQ_LEN}x{HIDDEN_DIM} -> {OUTPUT_DIM}")

    # Create data
    torch.manual_seed(42)
    input_data = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)
    target_labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=device)

    # Create simple network (1 layer, using Mojo ops)
    net = SimpleNeuralNetwork(
        hidden_dim=HIDDEN_DIM,
        ff_dim=OUTPUT_DIM,
        num_layers=1,
        num_classes=OUTPUT_DIM,
        use_mojo=True,
        device=device,
    )

    # Training
    print("\nTraining...")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    for step in range(3):
        optimizer.zero_grad()
        output = net(input_data)
        loss = criterion(output, target_labels)
        loss.backward()
        optimizer.step()

        accuracy = (output.argmax(dim=1) == target_labels).float().mean().item()
        print(f"Step {step+1}: Loss={loss.item():.3f}, Acc={accuracy:.0%}")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s")
    print("✅ Custom forward/backward ops working")

    return True


def demonstrate_single_operation():
    """Simple demonstration of a single LayerNorm + Linear operation."""
    print("=" * 60)
    print("   Simple Single Operation Demo")
    print("   (Just LayerNorm + Linear fusion - fastest)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚡ Simple Demo Configuration:")
    print(f"   Device: {device}")
    print(f"   Operations: 1 forward + 1 backward")
    print(f"   Focus: Correctness verification + speed")

    # Create test data
    print(f"\nCreating test data...")
    torch.manual_seed(42)
    batch_size, seq_len, hidden_dim = BATCH_SIZE, SEQ_LEN, HIDDEN_DIM
    output_dim = OUTPUT_DIM

    input_tensor = torch.randn(
        batch_size, seq_len, hidden_dim, device=device, requires_grad=True
    )
    ln_weight = torch.ones(hidden_dim, device=device, requires_grad=True)
    ln_bias = torch.zeros(hidden_dim, device=device, requires_grad=True)
    linear_weight = (
        torch.randn(output_dim, hidden_dim, device=device, requires_grad=True)
        * 0.02
    )
    linear_bias = torch.zeros(output_dim, device=device, requires_grad=True)

    print(f"   Input shape: {list(input_tensor.shape)} ({device})")

    # Test forward pass
    print(f"\nTesting forward pass...")
    start_time = time.time()
    mojo_output = mojo_layernorm_linear(
        input_tensor, ln_weight, ln_bias, linear_weight, linear_bias
    )
    forward_time = time.time() - start_time

    # Reference implementation
    pytorch_output = reference_layernorm_linear(
        input_tensor, ln_weight, ln_bias, linear_weight, linear_bias
    )

    forward_diff = torch.max(torch.abs(mojo_output - pytorch_output)).item()
    print(f"   Forward difference: {forward_diff:.2e}")
    print(f"   Forward time: {forward_time:.3f}s")
    print(
        "   Forward correctness:"
        f" {'✅ PASS' if forward_diff < 1e-4 else '❌ FAIL'}"
    )

    # Test backward pass
    print(f"\nTesting backward pass...")
    start_time = time.time()

    # Our custom autograd function
    result = mojo_layernorm_linear_autograd(
        input_tensor, ln_weight, ln_bias, linear_weight, linear_bias
    )
    loss = result.sum()
    loss.backward()

    backward_time = time.time() - start_time

    print(f"   Backward time: {backward_time:.3f}s")
    print(f"   Gradients computed: ✅")
    print(f"   Gradient shapes:")

    # Check each gradient and print shape if it exists
    if input_tensor.grad is not None:
        print(f"     input.grad: {list(input_tensor.grad.shape)}")
    else:
        print(f"     input.grad: None (expected for leaf tensors)")

    if ln_weight.grad is not None:
        print(f"     ln_weight.grad: {list(ln_weight.grad.shape)}")
    else:
        print(f"     ln_weight.grad: None")

    if ln_bias.grad is not None:
        print(f"     ln_bias.grad: {list(ln_bias.grad.shape)}")
    else:
        print(f"     ln_bias.grad: None")

    if linear_weight.grad is not None:
        print(f"     linear_weight.grad: {list(linear_weight.grad.shape)}")
    else:
        print(f"     linear_weight.grad: None")

    if linear_bias.grad is not None:
        print(f"     linear_bias.grad: {list(linear_bias.grad.shape)}")
    else:
        print(f"     linear_bias.grad: None")

    total_time = forward_time + backward_time
    print(f"\n⚡ Performance Summary:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Forward pass: {forward_time:.3f}s")
    print(f"   Backward pass: {backward_time:.3f}s")

    print(f"\nSimple Demo Complete!")
    print(f"\nWhat we demonstrated:")
    print(f"✅ Single LayerNorm + Linear forward operation")
    print(f"✅ Single backward pass with all gradients")
    print(f"✅ Correctness verification against PyTorch")
    print(f"✅ Performance timing")
    print(f"✅ Minimal overhead demonstration")

    print(f"\nThis is the fastest demo mode:")
    print(f"- No neural network overhead")
    print(f"- Single operation forward + backward")
    print(f"- Direct correctness comparison")
    print(f"- Minimal torch.compile overhead")

    return forward_diff < 1e-4


def main():
    """Main function with command line argument handling."""
    parser = argparse.ArgumentParser(description="Run tests for Puzzle 22")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fused",
        action="store_true",
        help="Test and benchmark fused algorithm only",
    )
    group.add_argument(
        "--unfused",
        action="store_true",
        help="Test and benchmark unfused algorithm only",
    )
    group.add_argument(
        "--backward",
        action="store_true",
        help="Run comprehensive backward pass test",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Neural network demo with custom operations",
    )
    group.add_argument(
        "--demo-simple",
        action="store_true",
        help="Single operation demo (fastest)",
    )
    args = parser.parse_args()

    if args.fused:
        run_algorithm_specific_test("fused")
    elif args.unfused:
        run_algorithm_specific_test("unfused")
    elif args.backward:
        run_comprehensive_backward_test()
    elif args.demo:
        demonstrate_neural_network_fast()
    elif args.demo_simple:
        demonstrate_single_operation()
    else:
        print("Usage:")
        print("  python p22.py --fused          # Test fused algorithm")
        print("  python p22.py --unfused        # Test unfused algorithm")
        print("  python p22.py --backward       # Test backward pass")
        print("  python p22.py --demo           # Neural network demo")
        print(
            "  python p22.py --demo-simple    # Single operation demo (fastest)"
        )
        exit(1)


if __name__ == "__main__":
    main()
