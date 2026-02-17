#!/usr/bin/env python3
"""
Unit tests for check_macos_env.py

Tests the macOS environment validation script with mocking support.
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


def test_default_run():
    """Test default run without mocks (should detect real environment)"""
    print("Testing default run...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"]
    )

    # Should exit with code 0 or 1 depending on environment
    assert returncode in [0, 1], f"Expected exit code 0 or 1, got {returncode}"
    assert "macOS Environment Check" in stdout, "Expected header in output"
    assert "macOS Version" in stdout, "Expected macOS version check"
    assert "Xcode Version" in stdout, "Expected Xcode version check"
    assert "Metal Toolchain" in stdout, "Expected Metal toolchain check"

    print(f"  ✓ Default run works (exit code: {returncode})")


def test_json_output():
    """Test JSON output mode"""
    print("Testing JSON output...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py", "--json"]
    )

    assert returncode in [0, 1], f"Expected exit code 0 or 1, got {returncode}"
    assert '"all_passed":' in stdout, "Expected JSON with all_passed field"
    assert '"checks":' in stdout, "Expected JSON with checks array"
    assert '"name":' in stdout, "Expected check names in JSON"

    print(f"  ✓ JSON output works")


def test_quiet_mode():
    """Test quiet mode (no output, only exit code)"""
    print("Testing quiet mode...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py", "--quiet"]
    )

    assert returncode in [0, 1], f"Expected exit code 0 or 1, got {returncode}"
    assert stdout == "", f"Expected no output in quiet mode, got: {stdout}"

    print(f"  ✓ Quiet mode works (exit code: {returncode})")


def test_mock_all_passing():
    """Test with all checks mocked to pass"""
    print("Testing all checks passing (mocked)...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"],
        env={
            "MOCK_MACOS_VERSION": "15.0",
            "MOCK_XCODE_VERSION": "16.0",
            "MOCK_METAL_AVAILABLE": "true"
        }
    )

    assert returncode == 0, f"Expected exit code 0 for passing checks, got {returncode}"
    assert "✓ PASS" in stdout, "Expected PASS status"
    assert "All checks passed" in stdout, "Expected success message"

    print(f"  ✓ All mocked checks pass correctly")


def test_mock_macos_fail():
    """Test with macOS version check failing"""
    print("Testing macOS version check failure...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"],
        env={
            "MOCK_MACOS_VERSION": "14.0",
            "MOCK_XCODE_VERSION": "16.0",
            "MOCK_METAL_AVAILABLE": "true"
        }
    )

    assert returncode == 1, f"Expected exit code 1 for failing check, got {returncode}"
    assert "✗ FAIL" in stdout, "Expected FAIL status"
    assert "macOS Version" in stdout, "Expected macOS version check"
    assert "does not meet" in stdout, "Expected failure message"
    assert "Upgrade to macOS" in stdout, "Expected fix instruction"

    print(f"  ✓ macOS version failure detected correctly")


def test_mock_xcode_fail():
    """Test with Xcode version check failing"""
    print("Testing Xcode version check failure...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"],
        env={
            "MOCK_MACOS_VERSION": "15.0",
            "MOCK_XCODE_VERSION": "15.0",
            "MOCK_METAL_AVAILABLE": "true"
        }
    )

    assert returncode == 1, f"Expected exit code 1 for failing check, got {returncode}"
    assert "✗ FAIL" in stdout, "Expected FAIL status"
    assert "Xcode Version" in stdout, "Expected Xcode version check"
    assert "does not meet" in stdout, "Expected failure message"
    assert "Upgrade to Xcode" in stdout or "App Store" in stdout, "Expected fix instruction"

    print(f"  ✓ Xcode version failure detected correctly")


def test_mock_metal_fail():
    """Test with Metal toolchain check failing"""
    print("Testing Metal toolchain check failure...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"],
        env={
            "MOCK_MACOS_VERSION": "15.0",
            "MOCK_XCODE_VERSION": "16.0",
            "MOCK_METAL_AVAILABLE": "false"
        }
    )

    assert returncode == 1, f"Expected exit code 1 for failing check, got {returncode}"
    assert "✗ FAIL" in stdout, "Expected FAIL status"
    assert "Metal Toolchain" in stdout, "Expected Metal toolchain check"
    assert "xcodebuild -downloadComponent MetalToolchain" in stdout, "Expected fix command"

    print(f"  ✓ Metal toolchain failure detected correctly")


def test_mock_multiple_failures():
    """Test with multiple checks failing"""
    print("Testing multiple check failures...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py"],
        env={
            "MOCK_MACOS_VERSION": "14.0",
            "MOCK_XCODE_VERSION": "15.0",
            "MOCK_METAL_AVAILABLE": "false"
        }
    )

    assert returncode == 1, f"Expected exit code 1 for failing checks, got {returncode}"

    # Count FAIL occurrences (should be 3)
    fail_count = stdout.count("✗ FAIL")
    assert fail_count == 3, f"Expected 3 FAIL statuses, got {fail_count}"

    assert "Some checks failed" in stdout, "Expected failure summary"

    print(f"  ✓ Multiple failures detected correctly")


def test_json_all_passing():
    """Test JSON output with all checks passing"""
    print("Testing JSON output with all passing...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py", "--json"],
        env={
            "MOCK_MACOS_VERSION": "15.0",
            "MOCK_XCODE_VERSION": "16.0",
            "MOCK_METAL_AVAILABLE": "true"
        }
    )

    assert returncode == 0, f"Expected exit code 0, got {returncode}"
    assert '"all_passed": true' in stdout, "Expected all_passed to be true"
    assert '"passed": true' in stdout, "Expected passed: true in checks"

    print(f"  ✓ JSON all passing works")


def test_json_with_failures():
    """Test JSON output with failures"""
    print("Testing JSON output with failures...")

    returncode, stdout, stderr = run_command(
        ["python3", "scripts/check_macos_env.py", "--json"],
        env={
            "MOCK_MACOS_VERSION": "14.0",
            "MOCK_XCODE_VERSION": "16.0",
            "MOCK_METAL_AVAILABLE": "true"
        }
    )

    assert returncode == 1, f"Expected exit code 1, got {returncode}"
    assert '"all_passed": false' in stdout, "Expected all_passed to be false"
    assert '"passed": false' in stdout, "Expected some passed: false in checks"
    assert '"fix_command"' in stdout, "Expected fix_command in output"

    print(f"  ✓ JSON with failures works")


def main():
    """Run all tests"""
    print("=" * 70)
    print("macOS Environment Check Unit Tests")
    print("=" * 70)
    print()

    tests = [
        test_default_run,
        test_json_output,
        test_quiet_mode,
        test_mock_all_passing,
        test_mock_macos_fail,
        test_mock_xcode_fail,
        test_mock_metal_fail,
        test_mock_multiple_failures,
        test_json_all_passing,
        test_json_with_failures,
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
