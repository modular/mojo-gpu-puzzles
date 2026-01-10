# Add Deprecation Scanner Tool

## Purpose

This PR contributes a TOML-driven deprecation scanner tool to help maintainers systematically identify deprecated Mojo API usage in the GPU Puzzles repository as new Mojo releases are published.

## Motivation

While working on [PR #193](https://github.com/modular/mojo-gpu-puzzles/pull/193) (LayoutTensor fixes) and the GPU import path fixes PR, it became clear that manually tracking deprecations across Mojo's frequent releases is time-consuming and error-prone. This tool automates the process of scanning for deprecated patterns based on Mojo's changelog.

## What's Included

### 1. `deprecation_patterns.toml` - Deprecation Database
A maintainable configuration file that defines 21+ deprecation patterns from Mojo 24.0+ releases:

```toml
[pointers.DTypePointer]
pattern = '\bDTypePointer\b'
description = "DTypePointer has been removed"
removed_in = "24.4"
replacement = "Use UnsafePointer instead"
severity = "high"
file_types = ["mojo", "md"]
changelog_url = "https://docs.modular.com/mojo/changelog/#v244-2024-08-08"
```

**Categories covered:**
- Pointer type removals (`DTypePointer`, `LegacyPointer`)
- Layout Tensor API changes
- GPU module reorganisation (8 import path patterns)
- GPU type consolidations
- Python interop changes
- Trait removals/renames
- Language keyword changes
- Type renames

### 2. `scan_deprecations.py` - Scanner Implementation
Python script (uv-compatible) that:
- Loads patterns from TOML configuration
- Scans specified file types (`.mojo`, `.md`)
- Generates detailed markdown reports
- Groups findings by category and severity
- Provides migration guidance

**Usage:**
```bash
# From repository root
uv run tools/deprecation-scanner/scan_deprecations.py --output report.md

# Custom config or repo path
uv run tools/deprecation-scanner/scan_deprecations.py \
  --config custom-patterns.toml \
  --repo /path/to/repo \
  --output scan-results.md
```

### 3. `README.md` - Complete Documentation
- Quick start guide
- Configuration format reference
- Usage examples
- CI/CD integration examples
- Pattern maintenance guidelines

## Real-World Results

This tool discovered the issues fixed in the companion PR:
- **18 deprecated GPU imports** across 13 files
- All instances verified against Mojo 25.7 changelog
- Identified both code and documentation examples

**Example scan output:**
```
============================================================
SCAN COMPLETE
============================================================

⚠️  Found 18 deprecated pattern instances

🟡 MEDIUM: 18

- 16 instances: gpu.warp → gpu.primitives.warp
- 2 instances: gpu.cluster → gpu
```

## Benefits

1. **Systematic**: Scans entire codebase consistently
2. **Maintainable**: Add new patterns by editing TOML (no code changes)
3. **Documented**: Each pattern includes changelog references
4. **Severity-based**: Prioritize high-impact deprecations
5. **Reusable**: Can be adapted for other Mojo projects
6. **CI-ready**: Can be integrated into GitHub Actions

## Usage Workflow

### For Maintainers

When a new Mojo release is published:

1. **Review changelog**: https://docs.modular.com/mojo/changelog/
2. **Add patterns**: Edit `deprecation_patterns.toml` with new deprecations:
   ```toml
   [category.new_pattern]
   pattern = '\bOldAPI\b'
   description = "OldAPI removed"
   removed_in = "25.8"
   replacement = "Use NewAPI instead"
   severity = "high"
   file_types = ["mojo", "md"]
   changelog_url = "https://docs.modular.com/mojo/changelog/#v258"
   ```
3. **Run scanner**: `uv run tools/deprecation-scanner/scan_deprecations.py`
4. **Review report**: Check findings and prioritize by severity
5. **Create PR**: Fix identified issues

### For Contributors

Run the scanner before submitting PRs to catch deprecated API usage:
```bash
uv run tools/deprecation-scanner/scan_deprecations.py --output my-scan.md
```

## Technical Details

**Requirements:**
- Python 3.9+
- `uv` (for dependency management)
- `tomli` package (Python < 3.11, auto-installed by uv)

**Design Decisions:**
- **TOML config**: Human-readable, version-controllable pattern database
- **Regex-based**: Simple, fast, works without parsing Mojo AST
- **uv integration**: Inline script dependencies (PEP 723)
- **Markdown output**: Easy to review, embed in PRs
- **Severity levels**: High/Medium/Low for prioritization

**Limitations:**
- Regex-based matching may have false positives
- Cannot detect semantic changes requiring type analysis
- Requires manual updates for each Mojo release

## Related Work

- **PR #193**: Fixed LayoutTensor deprecations (discovered by this tool)
- **GPU Import PR**: Fixed 18 GPU import path deprecations (discovered by this tool)

## Testing

```bash
# Test on current repository
cd mojo-gpu-puzzles
uv run tools/deprecation-scanner/scan_deprecations.py

# Should find 0 deprecations after related PRs are merged
```

## Integration Options

### Option 1: Manual Use
Maintainers run the tool periodically after Mojo releases.

### Option 2: CI Integration
Add to GitHub Actions workflow:
```yaml
name: Deprecation Scan
on: [push, pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install uv
      - run: uv run tools/deprecation-scanner/scan_deprecations.py --output report.md
      - uses: actions/upload-artifact@v3
        with:
          name: deprecation-report
          path: report.md
```

### Option 3: Pre-commit Hook
Contributors can run locally before committing.

## Maintainer Discretion

This tool is offered for maintainer consideration. Options:
- ✅ **Accept as-is**: Use the tool as proposed
- 🔧 **Modify**: Adapt to existing tooling/workflows
- 📋 **Reference**: Use patterns as documentation only
- ❌ **Decline**: No obligation to adopt

The primary goal is to **help maintain code quality** as Mojo evolves rapidly. The tool itself is less important than the systematic approach it enables.

## Files

```
tools/deprecation-scanner/
├── README.md                    # Complete documentation
├── scan_deprecations.py         # Scanner implementation (executable)
├── deprecation_patterns.toml    # Pattern database
└── PR_DESCRIPTION.md           # This file
```

## Future Enhancements

Potential improvements (not included in this PR):
- JSON output format for programmatic processing
- HTML report generation with filtering
- AST-based semantic analysis (more accurate, more complex)
- Automatic pattern generation from changelog parsing
- Integration with Mojo LSP/compiler diagnostics

---

**Co-Authored-By: Warp <agent@warp.dev>**
