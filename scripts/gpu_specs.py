#!/usr/bin/env python3
"""
Cross-platform GPU Specification Detection Script

This script detects and reports GPU specifications across multiple vendors:
- NVIDIA GPUs (using CUDA/pynvml)
- AMD GPUs (using ROCm tools)
- Apple Silicon GPUs (using system APIs)

Features:
- Automatic platform detection
- Unified output format across all vendors
- Fallback mechanisms for missing dependencies
- Integration with PyTorch backends

Requirements:
- NVIDIA: pynvml, torch with CUDA support
- AMD: rocm-smi, torch with ROCm support
- Apple: system_profiler, sysctl (built-in macOS tools)

Usage:
    # With pixi (recommended - handles all dependencies):
    pixi run gpu-specs              # NVIDIA (default)
    pixi run -e amd gpu-specs       # AMD
    pixi run -e apple gpu-specs     # Apple

    # With uv (install GPU-specific deps if needed):
    uv pip install -e ".[nvidia]"   # For NVIDIA GPUs
    uv pip install -e ".[amd]"      # For AMD GPUs
    uv run poe gpu-specs
"""

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None


@dataclass
class GPUSpecs:
    """Unified GPU specification data structure"""

    vendor: str
    device_name: str
    architecture: str
    compute_capability: Optional[str] = None
    compute_units: Optional[int] = (
        None  # SM for NVIDIA, CU for AMD, GPU cores for Apple
    )
    registers_per_unit: Optional[int] = None
    shared_memory_per_unit_kb: Optional[int] = None
    max_threads_per_unit: Optional[int] = None
    max_blocks_per_unit: Optional[int] = None
    total_memory_gb: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None


def detect_platform() -> str:
    """Detect the current platform and available GPU backends"""
    system = platform.system()

    # Check for Apple Silicon
    if system == "Darwin" and platform.processor() == "arm":
        return "apple_silicon"

    # Check for NVIDIA GPU availability
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            return "nvidia"
    except (ImportError, Exception):
        pass

    # Check for AMD GPU availability
    try:
        # Check if ROCm is available
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Check PyTorch backends if available
    if torch:
        if torch.backends.mps.is_available():
            return "apple_silicon"
        elif torch.cuda.is_available():
            return "nvidia"

    return "unknown"


def get_nvidia_specs(device_index: int = 0) -> GPUSpecs:
    """Get NVIDIA GPU specifications using pynvml and PyTorch"""
    try:
        try:
            import pynvml
        except ImportError:
            raise RuntimeError(
                "pynvml not installed. Install with: uv pip install -e '.[nvidia]'\n"
                "Or use pixi which handles all dependencies automatically."
            )

        arch_map = {
            "5.0": "Maxwell",
            "5.2": "Maxwell",
            "5.3": "Maxwell",
            "6.0": "Pascal",
            "6.1": "Pascal",
            "6.2": "Pascal",
            "7.0": "Volta",
            "7.2": "Volta",
            "7.5": "Turing",
            "8.0": "Ampere",
            "8.6": "Ampere",
            "8.7": "Ampere",
            "8.9": "Ada",
            "9.0": "Hopper",
        }

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle)
        name_str = name if isinstance(name, str) else name.decode()
        cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_gb = mem_info.total / (1024**3)

        cap_str = f"{cap[0]}.{cap[1]}"
        arch = arch_map.get(cap_str, f"Compute {cap_str}")
        max_blocks = 32 if cap[0] >= 6 else 16

        # Get PyTorch device properties if available
        pytorch_shared_mem = None
        sm_count = None
        registers_per_sm = None
        max_threads_per_sm = None

        if torch and torch.cuda.is_available():
            d = torch.cuda.get_device_properties(device_index)
            pytorch_shared_mem = d.shared_memory_per_multiprocessor // 1024
            sm_count = d.multi_processor_count
            registers_per_sm = d.regs_per_multiprocessor
            max_threads_per_sm = d.max_threads_per_multi_processor

        # Architectural maximums (for occupancy calculations)
        if cap[0] >= 9:  # Hopper
            arch_shared_mem = 228
        elif cap[0] == 8:  # Ampere/Ada
            arch_shared_mem = 128 if cap[1] == 9 else 164  # Ada=128KB, Ampere=164KB
        elif cap[0] == 7:  # Volta/Turing
            arch_shared_mem = 96 if cap[1] >= 5 else 80
        elif cap[0] == 6:  # Pascal
            arch_shared_mem = 96
        else:  # Maxwell and older
            arch_shared_mem = 64

        return GPUSpecs(
            vendor="NVIDIA",
            device_name=name_str,
            architecture=arch,
            compute_capability=cap_str,
            compute_units=sm_count,
            registers_per_unit=registers_per_sm,
            shared_memory_per_unit_kb=arch_shared_mem,
            max_threads_per_unit=max_threads_per_sm,
            max_blocks_per_unit=max_blocks,
            total_memory_gb=total_memory_gb,
            additional_info={
                "pytorch_shared_mem_kb": pytorch_shared_mem,
                "unit_type": "Streaming Multiprocessor (SM)",
            },
        )

    except Exception as e:
        raise RuntimeError(f"Failed to get NVIDIA GPU specs: {e}")


def get_amd_specs(device_index: int = 0) -> GPUSpecs:
    """Get AMD GPU specifications using ROCm tools"""
    try:
        # Get basic info from rocm-smi
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("rocm-smi not available or failed")

        device_name = result.stdout.strip().split("\n")[-1].strip()

        # Get memory info
        mem_result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        total_memory_gb = None
        if mem_result.returncode == 0:
            lines = mem_result.stdout.strip().split("\n")
            for line in lines:
                if "Total" in line and "MB" in line:
                    try:
                        mb_value = int(line.split()[-2])
                        total_memory_gb = mb_value / 1024
                        break
                    except (ValueError, IndexError):
                        pass

        # Try to get compute units from PyTorch if available
        compute_units = None
        if torch and hasattr(torch, "version") and hasattr(torch.version, "hip"):
            try:
                if torch.cuda.is_available():  # ROCm uses torch.cuda namespace
                    props = torch.cuda.get_device_properties(device_index)
                    compute_units = props.multi_processor_count
            except Exception:
                pass

        # AMD architecture detection based on device name
        arch = "RDNA"
        if "RX 7" in device_name or "RX 6" in device_name:
            arch = "RDNA 2/3"
        elif "RX 5" in device_name:
            arch = "RDNA"
        elif "Vega" in device_name:
            arch = "GCN 5.0 (Vega)"
        elif "Polaris" in device_name or "RX 4" in device_name or "RX 5" in device_name:
            arch = "GCN 4.0 (Polaris)"

        return GPUSpecs(
            vendor="AMD",
            device_name=device_name,
            architecture=arch,
            compute_units=compute_units,
            total_memory_gb=total_memory_gb,
            additional_info={
                "unit_type": "Compute Unit (CU)",
                "note": "Detailed architectural specs require ROCm profiling tools",
            },
        )

    except Exception as e:
        raise RuntimeError(f"Failed to get AMD GPU specs: {e}")


def get_apple_silicon_specs() -> GPUSpecs:
    """Get Apple Silicon GPU specifications"""
    try:
        # Get system info (not currently used but available for future enhancements)
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Get more detailed system info
        hw_result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        chip_name = "Apple Silicon"
        total_memory_gb = None

        if hw_result.returncode == 0:
            lines = hw_result.stdout.split("\n")
            for line in lines:
                line = line.strip()
                if "Chip:" in line:
                    chip_name = line.split("Chip:")[1].strip()
                elif "Memory:" in line:
                    try:
                        mem_str = line.split("Memory:")[1].strip()
                        if "GB" in mem_str:
                            total_memory_gb = float(mem_str.split()[0])
                    except (ValueError, IndexError):
                        pass

        # Determine architecture based on chip
        arch = "Apple GPU"
        gpu_cores = None

        if "M1" in chip_name:
            if "Pro" in chip_name:
                arch = "Apple M1 Pro GPU"
                gpu_cores = 16
            elif "Max" in chip_name:
                arch = "Apple M1 Max GPU"
                gpu_cores = 32
            elif "Ultra" in chip_name:
                arch = "Apple M1 Ultra GPU"
                gpu_cores = 64
            else:
                arch = "Apple M1 GPU"
                gpu_cores = 8
        elif "M2" in chip_name:
            if "Pro" in chip_name:
                arch = "Apple M2 Pro GPU"
                gpu_cores = 19
            elif "Max" in chip_name:
                arch = "Apple M2 Max GPU"
                gpu_cores = 38
            elif "Ultra" in chip_name:
                arch = "Apple M2 Ultra GPU"
                gpu_cores = 76
            else:
                arch = "Apple M2 GPU"
                gpu_cores = 10
        elif "M3" in chip_name:
            if "Pro" in chip_name:
                arch = "Apple M3 Pro GPU"
                gpu_cores = 18
            elif "Max" in chip_name:
                arch = "Apple M3 Max GPU"
                gpu_cores = 40
            else:
                arch = "Apple M3 GPU"
                gpu_cores = 10
        elif "M4" in chip_name:
            if "Pro" in chip_name:
                arch = "Apple M4 Pro GPU"
                gpu_cores = 20
            elif "Max" in chip_name:
                arch = "Apple M4 Max GPU"
                gpu_cores = 40
            else:
                arch = "Apple M4 GPU"
                gpu_cores = 10

        return GPUSpecs(
            vendor="Apple",
            device_name=chip_name,
            architecture=arch,
            compute_units=gpu_cores,
            total_memory_gb=total_memory_gb,
            additional_info={
                "unit_type": "GPU Core",
                "unified_memory": True,
                "metal_support": True,
                "note": "Uses unified memory architecture",
            },
        )

    except Exception as e:
        raise RuntimeError(f"Failed to get Apple Silicon specs: {e}")


def print_gpu_specs(specs: GPUSpecs):
    """Print GPU specifications in a unified format"""
    print(f"Device: {specs.device_name}")
    print(f"Vendor: {specs.vendor}")
    print(f"Architecture: {specs.architecture}")

    if specs.compute_capability:
        print(f"Compute Capability: {specs.compute_capability}")

    if specs.compute_units:
        unit_type = (
            specs.additional_info.get("unit_type", "Compute Units")
            if specs.additional_info
            else "Compute Units"
        )
        print(f"{unit_type}: {specs.compute_units}")

    if specs.registers_per_unit:
        print(f"Registers per Unit: {specs.registers_per_unit:,}")

    if specs.shared_memory_per_unit_kb:
        print(
            f"Shared Memory per Unit: {specs.shared_memory_per_unit_kb}KB (architectural max)"
        )

    if specs.additional_info and "pytorch_shared_mem_kb" in specs.additional_info:
        pytorch_mem = specs.additional_info["pytorch_shared_mem_kb"]
        if pytorch_mem:
            print(f"Shared Memory per Unit: {pytorch_mem}KB (operational limit)")

    if specs.max_threads_per_unit:
        print(f"Max Threads per Unit: {specs.max_threads_per_unit:,}")

    if specs.max_blocks_per_unit:
        print(f"Max Blocks per Unit: {specs.max_blocks_per_unit}")

    if specs.total_memory_gb:
        print(f"Total Memory: {specs.total_memory_gb:.1f}GB")

    # Print additional info
    if specs.additional_info:
        for key, value in specs.additional_info.items():
            if key not in ["pytorch_shared_mem_kb", "unit_type"]:
                if isinstance(value, bool):
                    if value:
                        print(f'{key.replace("_", " ").title()}: Yes')
                else:
                    print(f'{key.replace("_", " ").title()}: {value}')


def main():
    """Main function to detect and display GPU specifications"""
    try:
        platform_type = detect_platform()
        print(f'Detected Platform: {platform_type.replace("_", " ").title()}\n')

        if platform_type == "nvidia":
            specs = get_nvidia_specs()
            print_gpu_specs(specs)

        elif platform_type == "amd":
            specs = get_amd_specs()
            print_gpu_specs(specs)

        elif platform_type == "apple_silicon":
            specs = get_apple_silicon_specs()
            print_gpu_specs(specs)

        else:
            print("No compatible GPU detected or unsupported platform.")
            print(
                "Supported platforms: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal)"
            )

            # Try to provide some fallback info
            if torch:
                print("\nPyTorch available: Yes")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if hasattr(torch.backends, "mps"):
                    print(f"MPS available: {torch.backends.mps.is_available()}")
            else:
                print(
                    "\nPyTorch not available - install PyTorch for better GPU detection"
                )

    except Exception as e:
        print(f"Error detecting GPU specifications: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
