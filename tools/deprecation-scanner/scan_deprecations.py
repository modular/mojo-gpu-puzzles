#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "tomli>=2.0.0; python_version<'3.11'",
# ]
# ///
"""
Mojo Deprecation Scanner
Scans codebase for deprecated patterns defined in TOML configuration.

Usage:
    uv run scan_deprecations.py
    uv run scan_deprecations.py --repo /path/to/repo --output report.md
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set
from collections import defaultdict

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    import tomllib  # Python 3.11+


@dataclass
class DeprecationPattern:
    """Represents a single deprecation pattern."""
    
    name: str
    category: str
    pattern: str
    description: str
    removed_in: str
    replacement: str
    severity: str
    file_types: List[str]
    changelog_url: str = ""
    fixed_in_pr: int = None


@dataclass
class Finding:
    """Represents a found instance of a deprecated pattern."""
    
    pattern: DeprecationPattern
    file_path: Path
    line_number: int
    line_content: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


class DeprecationScanner:
    """Scans codebase for deprecated patterns."""
    
    def __init__(self, config_path: Path, repo_path: Path):
        self.config_path = config_path
        self.repo_path = repo_path
        self.patterns: List[DeprecationPattern] = []
        self.findings: List[Finding] = []
        self.scan_config: Dict = {}
        
        self._load_config()
    
    def _load_config(self):
        """Load TOML configuration file."""
        with open(self.config_path, 'rb') as f:
            config = tomllib.load(f)
        
        self.scan_config = config.get('scan_config', {})
        self.metadata = config.get('metadata', {})
        
        # Parse deprecation patterns
        for category, patterns in config.items():
            if category in ('metadata', 'scan_config'):
                continue
            
            for pattern_name, pattern_data in patterns.items():
                self.patterns.append(DeprecationPattern(
                    name=pattern_name,
                    category=category,
                    pattern=pattern_data['pattern'],
                    description=pattern_data['description'],
                    removed_in=pattern_data['removed_in'],
                    replacement=pattern_data['replacement'],
                    severity=pattern_data['severity'],
                    file_types=pattern_data['file_types'],
                    changelog_url=pattern_data.get('changelog_url', ''),
                    fixed_in_pr=pattern_data.get('fixed_in_pr')
                ))
    
    def _should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        exclude_dirs = self.scan_config.get('exclude_dirs', [])
        exclude_files = self.scan_config.get('exclude_files', [])
        
        # Check if any parent directory is in exclude list
        for part in path.parts:
            if part in exclude_dirs:
                return True
        
        # Check if filename is in exclude list
        if path.name in exclude_files:
            return True
        
        return False
    
    def _get_files_to_scan(self, file_types: List[str]) -> List[Path]:
        """Get all files matching the given extensions."""
        files = []
        for ext in file_types:
            pattern = f"**/*.{ext}"
            for file_path in self.repo_path.glob(pattern):
                if not self._should_exclude_path(file_path):
                    files.append(file_path)
        return files
    
    def scan_file(self, file_path: Path, pattern: DeprecationPattern) -> List[Finding]:
        """Scan a single file for a deprecation pattern."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            regex = re.compile(pattern.pattern)
            
            for i, line in enumerate(lines):
                if regex.search(line):
                    # Get context lines
                    context_lines = self.scan_config.get('max_context_lines', 2)
                    context_before = lines[max(0, i-context_lines):i]
                    context_after = lines[i+1:min(len(lines), i+1+context_lines)]
                    
                    finding = Finding(
                        pattern=pattern,
                        file_path=file_path.relative_to(self.repo_path),
                        line_number=i + 1,
                        line_content=line.rstrip(),
                        context_before=[l.rstrip() for l in context_before],
                        context_after=[l.rstrip() for l in context_after]
                    )
                    findings.append(finding)
        
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        
        return findings
    
    def scan_all(self):
        """Scan all files for all patterns."""
        print(f"Scanning repository: {self.repo_path}")
        print(f"Using config: {self.config_path}")
        print(f"Loaded {len(self.patterns)} deprecation patterns\n")
        
        # Group patterns by file types to minimize file reads
        patterns_by_filetype: Dict[str, List[DeprecationPattern]] = defaultdict(list)
        for pattern in self.patterns:
            for file_type in pattern.file_types:
                patterns_by_filetype[file_type].append(pattern)
        
        # Scan each file type
        all_file_types = set()
        for pattern in self.patterns:
            all_file_types.update(pattern.file_types)
        
        files_to_scan = self._get_files_to_scan(list(all_file_types))
        print(f"Found {len(files_to_scan)} files to scan")
        
        for file_path in files_to_scan:
            file_ext = file_path.suffix.lstrip('.')
            applicable_patterns = patterns_by_filetype.get(file_ext, [])
            
            for pattern in applicable_patterns:
                findings = self.scan_file(file_path, pattern)
                self.findings.extend(findings)
    
    def generate_report(self) -> str:
        """Generate a markdown report of findings."""
        report = []
        
        # Header
        report.append("# Mojo Deprecation Scan Report\n")
        report.append(f"**Repository**: {self.repo_path}")
        report.append(f"**Scan Date**: {self.metadata.get('last_updated', 'Unknown')}")
        report.append(f"**Patterns Checked**: {len(self.patterns)}")
        report.append(f"**Total Findings**: {len(self.findings)}\n")
        
        if not self.findings:
            report.append("✅ **No deprecated patterns found!**\n")
            return "\n".join(report)
        
        # Group findings by category
        findings_by_category: Dict[str, List[Finding]] = defaultdict(list)
        for finding in self.findings:
            findings_by_category[finding.pattern.category].append(finding)
        
        # Sort categories by number of findings (descending)
        sorted_categories = sorted(
            findings_by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Summary by category
        report.append("## Summary by Category\n")
        for category, findings in sorted_categories:
            severity_counts = defaultdict(int)
            for f in findings:
                severity_counts[f.pattern.severity] += 1
            
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append(f"- **Total**: {len(findings)} instances")
            for severity in ['high', 'medium', 'low']:
                if severity in severity_counts:
                    emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[severity]
                    report.append(f"- {emoji} **{severity.title()}**: {severity_counts[severity]}")
            report.append("")
        
        # Detailed findings
        report.append("## Detailed Findings\n")
        
        for category, findings in sorted_categories:
            report.append(f"### {category.replace('_', ' ').title()}\n")
            
            # Group by pattern within category
            findings_by_pattern = defaultdict(list)
            for finding in findings:
                findings_by_pattern[finding.pattern.name].append(finding)
            
            for pattern_name, pattern_findings in findings_by_pattern.items():
                pattern = pattern_findings[0].pattern
                
                severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[pattern.severity]
                report.append(f"#### {severity_emoji} {pattern.name}")
                report.append(f"**Description**: {pattern.description}")
                report.append(f"**Removed in**: Mojo {pattern.removed_in}")
                report.append(f"**Replacement**: {pattern.replacement}")
                
                if pattern.fixed_in_pr:
                    report.append(f"**Status**: ✅ Fixed in PR #{pattern.fixed_in_pr}")
                
                if pattern.changelog_url:
                    report.append(f"**Reference**: {pattern.changelog_url}")
                
                report.append(f"\n**Found {len(pattern_findings)} instance(s)**:\n")
                
                for finding in pattern_findings[:10]:  # Limit to first 10 per pattern
                    report.append(f"- `{finding.file_path}:{finding.line_number}`")
                    report.append(f"  ```")
                    report.append(f"  {finding.line_content}")
                    report.append(f"  ```")
                
                if len(pattern_findings) > 10:
                    report.append(f"\n_... and {len(pattern_findings) - 10} more instances_")
                
                report.append("")
        
        return "\n".join(report)
    
    def print_summary(self):
        """Print a brief summary to console."""
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)
        
        if not self.findings:
            print("✅ No deprecated patterns found!")
            return
        
        print(f"\n⚠️  Found {len(self.findings)} deprecated pattern instances\n")
        
        # Group by severity
        by_severity = defaultdict(int)
        for finding in self.findings:
            by_severity[finding.pattern.severity] += 1
        
        for severity in ['high', 'medium', 'low']:
            if severity in by_severity:
                emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[severity]
                print(f"{emoji} {severity.upper()}: {by_severity[severity]}")
        
        print(f"\nSee full report for details.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scan Mojo codebase for deprecated patterns"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'deprecation_patterns.toml',
        help='Path to TOML configuration file'
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help='Path to repository to scan'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for report (default: print to stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    if not args.repo.exists():
        print(f"Error: Repository path not found: {args.repo}", file=sys.stderr)
        sys.exit(1)
    
    # Run scanner
    scanner = DeprecationScanner(args.config, args.repo)
    scanner.scan_all()
    
    # Generate report
    if args.format == 'markdown':
        report = scanner.generate_report()
    else:
        print(f"Format {args.format} not yet implemented", file=sys.stderr)
        sys.exit(1)
    
    # Output report
    if args.output:
        args.output.write_text(report)
        print(f"\nReport written to: {args.output}")
    else:
        print("\n" + report)
    
    # Print summary
    scanner.print_summary()
    
    # Exit with error code if findings exist
    sys.exit(1 if scanner.findings else 0)


if __name__ == '__main__':
    main()
