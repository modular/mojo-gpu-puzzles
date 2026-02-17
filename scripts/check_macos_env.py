#!/usr/bin/env python3
"""
macOS Environment Validation Script

Validates that macOS environment meets requirements for GPU programming:
- macOS 15.0 or later
- Xcode 16 or later
- Metal toolchain installed

Usage:
    python3 scripts/check_macos_env.py              # Run all checks (summary)
    python3 scripts/check_macos_env.py --json       # JSON output
    python3 scripts/check_macos_env.py --quiet      # Exit code only (for scripts)

Testing:
    MOCK_MACOS_VERSION=15.0 python3 scripts/check_macos_env.py
    MOCK_XCODE_VERSION=16.0 python3 scripts/check_macos_env.py
    MOCK_METAL_AVAILABLE=true python3 scripts/check_macos_env.py
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# ==============================================================================
# CONFIGURATION - Update required versions here
# ==============================================================================
# These values define the minimum required versions for the project.
# Update these when project requirements change.

REQUIRED_MACOS_VERSION = "15.0"  # Minimum macOS version for optimal compatibility
REQUIRED_XCODE_VERSION = "16.0"  # Minimum Xcode version required

# ==============================================================================


@dataclass
class CheckResult:
    """Result of an environment check"""
    name: str
    passed: bool
    version: Optional[str] = None
    message: str = ""
    fix_command: Optional[str] = None


def check_macos_version(required_version: str = REQUIRED_MACOS_VERSION) -> CheckResult:
    """Check if macOS version meets requirements

    Args:
        required_version: Minimum macOS version (default: from REQUIRED_MACOS_VERSION constant)

    Returns:
        CheckResult with pass/fail status and details
    """
    # Check for mock override
    mock_version = os.getenv("MOCK_MACOS_VERSION")
    if mock_version:
        try:
            major, minor = map(int, mock_version.split(".")[:2])
            current_version = f"{major}.{minor}"
            req_major, req_minor = map(int, required_version.split(".")[:2])

            passed = (major > req_major) or (major == req_major and minor >= req_minor)

            return CheckResult(
                name="macOS Version",
                passed=passed,
                version=current_version,
                message=f"macOS {current_version} {'meets' if passed else 'does not meet'} requirement (>= {required_version})",
                fix_command=f"Upgrade to macOS {required_version} or later" if not passed else None
            )
        except (ValueError, IndexError):
            pass

    # Check if running on macOS
    if platform.system() != "Darwin":
        return CheckResult(
            name="macOS Version",
            passed=False,
            message="Not running on macOS",
            fix_command="This check is only applicable on macOS systems"
        )

    try:
        # Get macOS version using sw_vers
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return CheckResult(
                name="macOS Version",
                passed=False,
                message="Failed to detect macOS version",
                fix_command="Ensure sw_vers command is available"
            )

        version_str = result.stdout.strip()

        # Parse version (handle formats like "15.0", "15.0.1", "14.5")
        version_match = re.match(r"(\d+)\.(\d+)", version_str)
        if not version_match:
            return CheckResult(
                name="macOS Version",
                passed=False,
                version=version_str,
                message=f"Could not parse macOS version: {version_str}"
            )

        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        current_version = f"{major}.{minor}"

        # Parse required version
        req_major, req_minor = map(int, required_version.split(".")[:2])

        # Check if version meets requirement
        passed = (major > req_major) or (major == req_major and minor >= req_minor)

        return CheckResult(
            name="macOS Version",
            passed=passed,
            version=current_version,
            message=f"macOS {current_version} {'meets' if passed else 'does not meet'} requirement (>= {required_version})",
            fix_command=f"Upgrade to macOS {required_version} or later" if not passed else None
        )

    except Exception as e:
        return CheckResult(
            name="macOS Version",
            passed=False,
            message=f"Error checking macOS version: {e}"
        )


def check_xcode_version(required_version: str = REQUIRED_XCODE_VERSION) -> CheckResult:
    """Check if Xcode version meets requirements

    Args:
        required_version: Minimum Xcode version (default: from REQUIRED_XCODE_VERSION constant)

    Returns:
        CheckResult with pass/fail status and details
    """
    # Check for mock override
    mock_version = os.getenv("MOCK_XCODE_VERSION")
    if mock_version:
        try:
            current_major = int(mock_version.split(".")[0])
            req_major = int(required_version.split(".")[0])

            passed = current_major >= req_major

            return CheckResult(
                name="Xcode Version",
                passed=passed,
                version=mock_version,
                message=f"Xcode {mock_version} {'meets' if passed else 'does not meet'} requirement (>= {required_version})",
                fix_command=f"Upgrade to Xcode {required_version} or later from the App Store" if not passed else None
            )
        except (ValueError, IndexError):
            pass

    try:
        # Try to get Xcode version
        result = subprocess.run(
            ["xcodebuild", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return CheckResult(
                name="Xcode Version",
                passed=False,
                message="Xcode not found or not properly installed",
                fix_command="Install Xcode 16 or later from the App Store"
            )

        # Parse version from output (format: "Xcode 16.0\nBuild version...")
        output = result.stdout.strip()
        version_match = re.search(r"Xcode\s+(\d+(?:\.\d+)?)", output)

        if not version_match:
            return CheckResult(
                name="Xcode Version",
                passed=False,
                message=f"Could not parse Xcode version from: {output}",
                fix_command="Ensure Xcode is properly installed"
            )

        version_str = version_match.group(1)
        current_major = int(version_str.split(".")[0])
        req_major = int(required_version.split(".")[0])

        passed = current_major >= req_major

        return CheckResult(
            name="Xcode Version",
            passed=passed,
            version=version_str,
            message=f"Xcode {version_str} {'meets' if passed else 'does not meet'} requirement (>= {required_version})",
            fix_command=f"Upgrade to Xcode {required_version} or later from the App Store" if not passed else None
        )

    except FileNotFoundError:
        return CheckResult(
            name="Xcode Version",
            passed=False,
            message="xcodebuild command not found",
            fix_command="Install Xcode 16 or later from the App Store"
        )
    except Exception as e:
        return CheckResult(
            name="Xcode Version",
            passed=False,
            message=f"Error checking Xcode version: {e}"
        )


def check_metal_toolchain() -> CheckResult:
    """Check if Metal toolchain is installed and available

    Returns:
        CheckResult with pass/fail status and details
    """
    # Check for mock override
    mock_available = os.getenv("MOCK_METAL_AVAILABLE")
    if mock_available:
        is_available = mock_available.lower() in ("true", "1", "yes")
        return CheckResult(
            name="Metal Toolchain",
            passed=is_available,
            message=f"Metal toolchain is {'available' if is_available else 'not available'} (mocked)",
            fix_command="xcodebuild -downloadComponent MetalToolchain" if not is_available else None
        )

    try:
        # Use the exact command from the tutorial: xcrun -sdk macosx metal
        # This matches what users are instructed to run manually
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Combine stdout and stderr for checking
        output = result.stdout + result.stderr

        # Check for the specific error message indicating Metal toolchain is missing
        if "cannot execute tool 'metal'" in output.lower():
            return CheckResult(
                name="Metal Toolchain",
                passed=False,
                message="Metal toolchain not installed",
                fix_command="xcodebuild -downloadComponent MetalToolchain"
            )

        # Check for "no input files" error, which indicates Metal is working correctly
        # This is the expected output when Metal toolchain is properly installed
        if "no input files" in output.lower():
            # Try to extract version if available
            version_match = re.search(r"metal version (\S+)", output)
            version_str = version_match.group(1) if version_match else "installed"

            return CheckResult(
                name="Metal Toolchain",
                passed=True,
                version=version_str if version_match else None,
                message=f"Metal toolchain is available{'(' + version_str + ')' if version_match else ''}"
            )

        # Some other error occurred
        return CheckResult(
            name="Metal Toolchain",
            passed=False,
            message=f"Unexpected Metal toolchain error: {output.strip()}",
            fix_command="xcodebuild -downloadComponent MetalToolchain"
        )

    except FileNotFoundError:
        return CheckResult(
            name="Metal Toolchain",
            passed=False,
            message="xcrun command not found (Xcode Command Line Tools not installed)",
            fix_command="Install Xcode Command Line Tools: xcode-select --install"
        )
    except Exception as e:
        return CheckResult(
            name="Metal Toolchain",
            passed=False,
            message=f"Error checking Metal toolchain: {e}",
            fix_command="xcodebuild -downloadComponent MetalToolchain"
        )


def print_summary(results: list[CheckResult]) -> None:
    """Print human-readable summary of check results"""
    print("macOS Environment Check")
    print("=" * 70)
    print()

    all_passed = True

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        status_color = "\033[32m" if result.passed else "\033[31m"  # Green or Red
        reset_color = "\033[0m"

        print(f"{status_color}{status}{reset_color} {result.name}")

        if result.version:
            print(f"     Version: {result.version}")

        if result.message:
            print(f"     {result.message}")

        if result.fix_command:
            print(f"     Fix: {result.fix_command}")
            all_passed = False

        print()

    print("=" * 70)
    if all_passed:
        print("\033[32m✓ All checks passed!\033[0m")
    else:
        print("\033[31m✗ Some checks failed. Please address the issues above.\033[0m")
    print()


def print_json(results: list[CheckResult]) -> None:
    """Print JSON output of check results"""
    output = {
        "all_passed": all(r.passed for r in results),
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "version": r.version,
                "message": r.message,
                "fix_command": r.fix_command
            }
            for r in results
        ]
    }
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Validate macOS environment for GPU programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/check_macos_env.py              # Human-readable summary
  python3 scripts/check_macos_env.py --json       # JSON output
  python3 scripts/check_macos_env.py --quiet      # No output, exit code only

Testing:
  MOCK_MACOS_VERSION=15.0 python3 scripts/check_macos_env.py
  MOCK_XCODE_VERSION=16.0 python3 scripts/check_macos_env.py
  MOCK_METAL_AVAILABLE=true python3 scripts/check_macos_env.py
        """
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="No output, only exit code (0 = pass, 1 = fail)"
    )

    args = parser.parse_args()

    # Run all checks
    results = [
        check_macos_version(),
        check_xcode_version(),
        check_metal_toolchain(),
    ]

    # Determine if all checks passed
    all_passed = all(r.passed for r in results)

    # Output results based on mode
    if args.quiet:
        # Quiet mode: no output, just exit code
        pass
    elif args.json:
        print_json(results)
    else:
        print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
