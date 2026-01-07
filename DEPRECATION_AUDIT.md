# Mojo GPU Puzzles - Deprecation Audit Report

**Date**: 2026-01-07  
**Auditor**: Michael Booth (DataBooth)  
**Mojo Version Reference**: 25.7.0+ Changelog  
**Repository**: https://github.com/modular/mojo-gpu-puzzles

## Executive Summary

- **Total .mojo files scanned**: 91
- **Deprecated patterns found in code**: 0 ✅
- **Deprecated patterns found in documentation**: 5 ⚠️
- **Severity**: Documentation only - does not affect runnable code

## Findings

### Pattern: `tb[dtype]().row_major[SIZE]().shared().alloc()`

**Status**: DEPRECATED in Mojo 25.7.0  
**Changelog Reference**: https://docs.modular.com/mojo/changelog/#25-7-removed  
**Removal Note**: "LayoutTensorBuild type has been removed. Use LayoutTensor with parameters directly instead."

**Occurrences** (5 total, all in markdown documentation):

1. **puzzle_13/simple.md:44**
   ```
   1. Use `tb[dtype]().row_major[SIZE]().shared().alloc()` for shared memory allocation
   ```
   - **Context**: Tips section for Problem 13
   - **Impact**: Confuses learners with outdated API
   - **Suggested Fix**: Update to current LayoutTensor API syntax

2. **puzzle_09/third_case.md:296**
   ```
   shared_workspace = tb[dtype]().row_major[SIZE-1]().shared().alloc()
   ```
   - **Context**: Code example in walkthrough
   - **Impact**: Copy-paste will fail
   - **Suggested Fix**: Update example code

3. **puzzle_24/puzzle_24.md:52**
   ```
   shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
   ```
   - **Context**: Problem description/hints
   - **Impact**: Outdated guidance
   - **Suggested Fix**: Update to current syntax

4. **puzzle_25/puzzle_25.md:51**
   ```
   shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
   ```
   - **Context**: Problem description/hints
   - **Impact**: Outdated guidance
   - **Suggested Fix**: Update to current syntax

5. **puzzle_26/puzzle_26.md:51**
   ```
   shared = tb[dtype]().row_major[WARP_SIZE]().shared().alloc()
   ```
   - **Context**: Problem description/hints
   - **Impact**: Outdated guidance
   - **Suggested Fix**: Update to current syntax

## Additional Patterns Checked

The following deprecated patterns were NOT found:
- ✅ `LayoutTensorBuild` (direct type usage)

## Impact Assessment

### Severity: MEDIUM

- **Code files**: Clean ✅ (no deprecated code in actual .mojo files)
- **Documentation**: 5 instances of outdated API in hints/examples
- **User Impact**: Confusion for learners, copy-paste errors
- **Breaking**: No (documentation only)

### Affected Puzzles

- Puzzle 9 (third_case walkthrough)
- Puzzle 13 (simple hints)  
- Puzzle 24, 25, 26 (problem descriptions)

## Recommendations

### Option 1: Quick Fix (Recommended)
- Create PR updating 5 markdown files with correct LayoutTensor syntax
- Add note about Mojo version requirements
- **Effort**: 1-2 hours
- **Impact**: Immediate fix for learners

### Option 2: Systematic Audit
- Review all 91 .mojo files for runtime deprecations
- Check for other deprecated patterns from 25.0+ changelogs
- **Effort**: 4-6 hours
- **Impact**: Comprehensive modernization

### Option 3: Automated Tool
- Build deprecation scanner for future maintenance
- Could run on CI to catch future deprecations
- **Effort**: 8-12 hours
- **Impact**: Long-term maintenance

## Next Steps

1. **Immediate**: Determine correct LayoutTensor syntax from current docs
2. **Create PR**: Fix 5 markdown files with updated examples
3. **Follow-up**: Review changelog for other post-25.0 deprecations

## Appendix: Search Commands Used

```bash
# Scan for deprecated LayoutTensorBuild type
grep -rn "LayoutTensorBuild" --include="*.mojo" --include="*.html" --include="*.md" .

# Scan for builder pattern usage
grep -rn "tb\[dtype\]()\.row_major\[" --include="*.mojo" --include="*.html" --include="*.md" .

# Count mojo files
find . -name "*.mojo" -type f | wc -l
```

## References

- [Mojo 25.7 Changelog - Removed Features](https://docs.modular.com/mojo/changelog/#25-7-removed)
- [GPU Puzzles Repository](https://github.com/modular/mojo-gpu-puzzles)
- [Original Issue Report](https://puzzles.modular.com/puzzle_13/simple.html)
