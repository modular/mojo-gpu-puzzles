#!/usr/bin/env python3
"""
Unit tests for gpu_specs.py

Tests the new bash-friendly flags and mock support.
"""

import subprocess
import sys
import os


def run_command(cmd, env=None):
    """Run a command and return (returncode, stdout, stderr)"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def test_platform_flag():
    """Test --platform flag outputs valid platform name"""
    print("Testing --platform flag...")
    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"]
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout in ["nvidia", "amd", "apple", "unknown"], \
        f"Expected platform name (nvidia/amd/apple/unknown), got '{stdout}'"
    print(f"  ✓ Platform detected: {stdout}")


def test_compute_cap_flag():
    """Test --compute-cap flag outputs valid format or empty string"""
    print("Testing --compute-cap flag...")
    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--compute-cap"]
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"

    # Should be empty (non-NVIDIA) or a valid version like "8.6"
    if stdout:
        # Check format: should be digits.digits
        parts = stdout.split(".")
        assert len(parts) == 2, f"Expected format X.Y, got '{stdout}'"
        assert parts[0].isdigit() and parts[1].isdigit(), \
            f"Expected numeric format, got '{stdout}'"
        print(f"  ✓ Compute capability: {stdout}")
    else:
        print(f"  ✓ Compute capability: (empty - not NVIDIA)")


def test_check_platform_match():
    """Test --check-platform with matching platform"""
    print("Testing --check-platform (match)...")

    # First get actual platform
    _, platform, _ = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"]
    )

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--check-platform", platform]
    )

    assert returncode == 0, \
        f"Expected exit code 0 for matching platform '{platform}', got {returncode}"
    print(f"  ✓ Platform check passed for '{platform}'")


def test_check_platform_mismatch():
    """Test --check-platform with non-matching platform"""
    print("Testing --check-platform (mismatch)...")

    # First get actual platform
    _, platform, _ = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"]
    )

    # Pick a different platform to test mismatch
    test_platform = "nvidia" if platform != "nvidia" else "amd"

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--check-platform", test_platform]
    )

    assert returncode == 1, \
        f"Expected exit code 1 for non-matching platform '{test_platform}', got {returncode}"
    print(f"  ✓ Platform check correctly failed for '{test_platform}'")


def test_mock_nvidia_platform():
    """Test mocking NVIDIA platform"""
    print("Testing NVIDIA platform mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"],
        env={"MOCK_GPU_PLATFORM": "nvidia"}
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout == "nvidia", f"Expected 'nvidia', got '{stdout}'"
    print(f"  ✓ Mock NVIDIA platform works")


def test_mock_nvidia_compute_cap():
    """Test mocking NVIDIA compute capability"""
    print("Testing NVIDIA compute capability mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--compute-cap"],
        env={
            "MOCK_GPU_PLATFORM": "nvidia",
            "MOCK_COMPUTE_CAP": "8.6"
        }
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout == "8.6", f"Expected '8.6', got '{stdout}'"
    print(f"  ✓ Mock compute capability works")


def test_mock_amd_platform():
    """Test mocking AMD platform"""
    print("Testing AMD platform mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"],
        env={"MOCK_GPU_PLATFORM": "amd"}
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout == "amd", f"Expected 'amd', got '{stdout}'"
    print(f"  ✓ Mock AMD platform works")


def test_mock_apple_platform():
    """Test mocking Apple Silicon platform"""
    print("Testing Apple Silicon platform mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"],
        env={"MOCK_GPU_PLATFORM": "apple_silicon"}
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout == "apple", f"Expected 'apple', got '{stdout}'"
    print(f"  ✓ Mock Apple Silicon platform works")


def test_mock_unknown_platform():
    """Test mocking unknown platform"""
    print("Testing unknown platform mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--platform"],
        env={"MOCK_GPU_PLATFORM": "unknown"}
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert stdout == "unknown", f"Expected 'unknown', got '{stdout}'"
    print(f"  ✓ Mock unknown platform works")


def test_mock_check_platform():
    """Test --check-platform with mocked platform"""
    print("Testing --check-platform with mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--check-platform", "nvidia"],
        env={"MOCK_GPU_PLATFORM": "nvidia"}
    )

    assert returncode == 0, \
        f"Expected exit code 0 for mocked NVIDIA platform, got {returncode}"
    print(f"  ✓ Mock platform check works")


def test_default_output():
    """Test default (full specs) output"""
    print("Testing default full specs output...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py"]
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert "Detected Platform:" in stdout or "Error" in stderr, \
        "Expected either platform info or error message"
    print(f"  ✓ Default output works")


def test_summary_output():
    """Test --summary flag output"""
    print("Testing --summary flag...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--summary"]
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert "GPU:" in stdout, "Expected 'GPU:' in summary output"
    assert "Platform:" in stdout, "Expected 'Platform:' in summary output"
    print(f"  ✓ Summary output works")


def test_summary_nvidia_mock():
    """Test --summary with NVIDIA mock"""
    print("Testing --summary with NVIDIA mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--summary"],
        env={
            "MOCK_GPU_PLATFORM": "nvidia",
            "MOCK_COMPUTE_CAP": "8.6"
        }
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert "GPU:" in stdout, "Expected 'GPU:' in summary"
    assert "Compute 8.6" in stdout, "Expected compute capability in summary"
    assert "Ampere" in stdout, "Expected architecture in summary"
    assert "Platform: NVIDIA" in stdout, "Expected NVIDIA platform"
    print(f"  ✓ NVIDIA summary works")


def test_summary_amd_mock():
    """Test --summary with AMD mock"""
    print("Testing --summary with AMD mock...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/gpu_specs.py", "--summary"],
        env={"MOCK_GPU_PLATFORM": "amd"}
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert "GPU:" in stdout, "Expected 'GPU:' in summary"
    assert "Platform: AMD" in stdout, "Expected AMD platform"
    print(f"  ✓ AMD summary works")


def main():
    """Run all tests"""
    print("=" * 70)
    print("GPU Specs Unit Tests")
    print("=" * 70)
    print()

    tests = [
        test_platform_flag,
        test_compute_cap_flag,
        test_check_platform_match,
        test_check_platform_mismatch,
        test_mock_nvidia_platform,
        test_mock_nvidia_compute_cap,
        test_mock_amd_platform,
        test_mock_apple_platform,
        test_mock_unknown_platform,
        test_mock_check_platform,
        test_default_output,
        test_summary_output,
        test_summary_nvidia_mock,
        test_summary_amd_mock,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print()
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
