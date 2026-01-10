# Mojo Deprecation Scanner

A TOML-driven configuration tool for scanning Mojo codebases for deprecated patterns.

## Features

- 📝 **TOML Configuration**: Easy-to-maintain deprecation pattern definitions
- 🔍 **Comprehensive Coverage**: Scans 25+ deprecation patterns from Mojo 24.0+
- 📊 **Detailed Reports**: Markdown reports with severity levels and context
- ⚡ **Efficient**: Smart file filtering and pattern grouping
- 🎯 **Extensible**: Add new patterns without modifying code

## Quick Start

### Installation

Requires Python 3.9+ and [uv](https://docs.astral.sh/uv/).

Dependencies are automatically managed by uv (no pip install needed).

### Usage

```bash
# Scan current directory
uv run scan_deprecations.py

# Scan specific repository
uv run scan_deprecations.py --repo /path/to/mojo-project

# Use custom config
uv run scan_deprecations.py --config my_patterns.toml

# Save report to file
uv run scan_deprecations.py --output report.md
```

### Example Output

```
Scanning repository: /Users/mjboothaus/code/mojo-gpu-puzzles-deprecation-audit
Using config: deprecation_patterns.toml
Loaded 25 deprecation patterns

Found 142 files to scan

============================================================
SCAN COMPLETE
============================================================

⚠️  Found 15 deprecated pattern instances

🔴 HIGH: 5
🟡 MEDIUM: 8
🟢 LOW: 2

See full report for details.
```

## Configuration

### Pattern Format

Each deprecation pattern is defined in `deprecation_patterns.toml`:

```toml
[category.pattern_name]
pattern = '\bDTypePointer\b'           # Regex pattern
description = "DTypePointer removed"    # What changed
removed_in = "24.4"                     # Mojo version
replacement = "Use UnsafePointer"       # Migration guidance
severity = "high"                       # high|medium|low
file_types = ["mojo", "md"]            # Extensions to scan
changelog_url = "https://..."          # Optional reference
fixed_in_pr = 123                      # Optional PR tracking
```

### Categories

Current categories in `deprecation_patterns.toml`:

- **pointers**: Pointer type removals (DTypePointer, LegacyPointer)
- **layout**: LayoutTensor API changes
- **gpu_imports**: GPU module reorganisation (25.7)
- **gpu_types**: GPU type consolidations
- **gpu_tma**: TMA module moves
- **python**: Python interop changes
- **traits**: Trait removals/renames
- **keywords**: Language keyword changes
- **types**: Type renames

### Scan Configuration

Control scanning behaviour in the `[scan_config]` section:

```toml
[scan_config]
exclude_dirs = [".git", "node_modules", "__pycache__"]
exclude_files = ["DEPRECATION_AUDIT.md"]
report_format = "markdown"
show_line_context = true
max_context_lines = 2
group_by_category = true
```

## Adding New Patterns

1. Find deprecation in [Mojo Changelog](https://docs.modular.com/mojo/changelog/)
2. Add entry to `deprecation_patterns.toml`:

```toml
[category.new_pattern]
pattern = '\bOldAPI\b'
description = "OldAPI was removed"
removed_in = "25.8"
replacement = "Use NewAPI instead"
severity = "high"
file_types = ["mojo", "md"]
changelog_url = "https://docs.modular.com/mojo/changelog/#v258"
```

3. Run scanner to test

## Report Structure

Generated markdown reports include:

1. **Metadata**: Scan date, patterns checked, total findings
2. **Summary by Category**: Grouped by deprecation category with severity counts
3. **Detailed Findings**: Per-pattern breakdown with:
   - File locations and line numbers
   - Code context
   - Migration guidance
   - Changelog references

## Examples

### Scan and Save Report

```bash
uv run scan_deprecations.py \
  --repo ~/code/my-mojo-project \
  --output deprecation-report.md
```

### Check Specific Config

```bash
uv run scan_deprecations.py \
  --config custom-patterns.toml \
  --repo ~/code/my-mojo-project
```

### CI Integration

```yaml
# .github/workflows/deprecation-scan.yml
name: Deprecation Scan
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install uv
        run: pip install uv
      - name: Run deprecation scan
        run: |
          uv run scan_deprecations.py --output report.md
      - uses: actions/upload-artifact@v3
        with:
          name: deprecation-report
          path: report.md
```

## Pattern Coverage

Currently scans for **25 deprecation patterns** across:

- ✅ Mojo 24.4 (Pointer changes)
- ✅ Mojo 25.0+ (Trait removals)
- ✅ Mojo 25.6 (@value decorator)
- ✅ Mojo 25.7 (GPU reorganisation, LayoutTensor)

## Exit Codes

- `0`: No deprecated patterns found
- `1`: Deprecated patterns found (or error)

Useful for CI/CD pipelines that should fail on deprecation usage.

## Limitations

- **Regex-based**: May have false positives for complex patterns
- **String matching**: Cannot detect semantic changes requiring type analysis
- **Manual updates**: Config must be updated for new Mojo releases

## Maintenance

To keep the scanner current:

1. Monitor [Mojo Changelog](https://docs.modular.com/mojo/changelog/)
2. Add new deprecations to `deprecation_patterns.toml`
3. Test against real codebases
4. Update version in `[metadata]`

## Related Files

- `deprecation_patterns.toml`: Pattern configuration
- `scan_deprecations.py`: Scanner implementation
- `DEPRECATION_AUDIT.md`: Initial manual audit results

## Contributing

To extend the scanner:

1. Add patterns to TOML config (prefer this)
2. For complex detection, modify Python scanner
3. Document new categories in this README
4. Test thoroughly to avoid false positives

## References

- [Mojo Changelog](https://docs.modular.com/mojo/changelog/)
- [Mojo Documentation](https://docs.modular.com/mojo/)
- [GPU Puzzles Repo](https://github.com/modular/mojo-gpu-puzzles)
