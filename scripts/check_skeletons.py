#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Check that problem skeletons stay in sync with their solutions.

The repository's teaching contract is: a file under ``problems/pNN/`` must be
identical to its ``solutions/pNN/`` counterpart *except* that

1. each ``# ANCHOR: NAME`` / ``# ANCHOR_END: NAME`` marker gains a ``_solution``
   suffix on the solution side (the book includes both via mdBook
   ``{{#include ...:NAME}}`` / ``...:NAME_solution}}`` directives), and
2. the implementation inside each pedagogical region is replaced, on the problem
   side, by one or more ``# FILL ...`` hint comments.

In other words, solving a puzzle should only ever *add* lines. If a problem
contains real (non-``# FILL``) code that does not appear in the solution, the
two have drifted -- which is exactly the bug tracked in issue #254, where the
LayoutTensor -> TileTensor migration updated solutions but left the skeletons on
the old API.

This script flags every drifted pair. It is read-only and exits non-zero when
any pair is out of sync, so it can gate CI.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Locate the directory containing both `problems/` and `solutions/`.

    Walks up from this file rather than using ``Path.resolve()``: under Bazel
    the script and its data run from a runfiles tree of symlinks, and resolving
    them would escape that tree and miss the co-located puzzle sources.
    """
    here = Path(__file__).absolute()
    for d in (here.parent, *here.parents):
        if (d / "problems").is_dir() and (d / "solutions").is_dir():
            return d
    return here.parent.parent


REPO_ROOT = _find_repo_root()
PROBLEMS_DIR = REPO_ROOT / "problems"
SOLUTIONS_DIR = REPO_ROOT / "solutions"

# Files whose problem/solution divergence is INTENTIONAL and must not be
# "fixed". Paths are relative to the repo root.
#
# - p10 is the sanitizer puzzle ("Memory Error Detection & Race Conditions"):
#   the problem deliberately ships buggy kernels (a shared-memory race and an
#   unguarded out-of-bounds write, anchors `shared_memory_race` /
#   `add_10_2d_no_guard`) so the learner can catch them with memcheck/racecheck.
#   The solution fixes both. Forcing them into sync would destroy the lesson.
EXCLUDED = {
    "problems/p10/p10.mojo",
}


def _is_anchor(line: str) -> bool:
    s = line.strip()
    return s.startswith("# ANCHOR:") or s.startswith("# ANCHOR_END:")


def _is_fill(line: str) -> bool:
    # A line that legitimately appears in a problem but not its solution: either
    # a fill-in hint comment ("# FILL ME IN ...", "# FILL IN ...") or a body
    # placeholder (`...` / `pass`) that keeps an otherwise-empty gap body
    # syntactically valid until the learner fills it in.
    s = line.strip()
    if s in ("...", "pass"):
        return True
    return s.lower().startswith("# fill")


def _normalize(text: str) -> list[str]:
    """Reduce a source file to the lines that must match across problem/solution.

    Drops blank lines (placement of blanks shifts when anchors move) and anchor
    markers (their name and position legitimately differ between the two sides),
    and strips trailing whitespace from the rest.
    """
    out: list[str] = []
    for raw in text.splitlines():
        if _is_anchor(raw):
            continue
        stripped = raw.rstrip()
        if stripped.strip() == "":
            continue
        out.append(stripped)
    return out


def check_pair(problem: Path, solution: Path) -> list[str]:
    """Return the list of drifted problem lines (empty => in sync)."""
    p_lines = _normalize(problem.read_text())
    s_lines = _normalize(solution.read_text())

    drift: list[str] = []
    matcher = difflib.SequenceMatcher(a=p_lines, b=s_lines, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            # Lines present in the problem but absent from the solution. These
            # are only legitimate when they are FILL hint comments.
            for line in p_lines[i1:i2]:
                if not _is_fill(line):
                    drift.append(line)
    return drift


def main() -> int:
    pairs: list[tuple[Path, Path]] = []
    for solution in sorted(SOLUTIONS_DIR.rglob("*.mojo")):
        rel = solution.relative_to(SOLUTIONS_DIR)
        problem = PROBLEMS_DIR / rel
        if (
            problem.exists()
            and str(problem.relative_to(REPO_ROOT)) not in EXCLUDED
        ):
            pairs.append((problem, solution))

    # Guard against a vacuous pass (e.g. mis-wired Bazel runfiles): if we found
    # no pairs at all, the sources are not where we expected them.
    if not pairs:
        print(f"error: found no problem/solution pairs under {REPO_ROOT}")
        return 1

    diverged: list[tuple[Path, list[str]]] = []
    for problem, solution in pairs:
        drift = check_pair(problem, solution)
        rel = problem.relative_to(REPO_ROOT)
        if drift:
            diverged.append((problem, drift))
            print(f"DIVERGED  {rel}  ({len(drift)} unexpected line(s))")
        else:
            print(f"IN-SYNC   {rel}")

    print()
    if diverged:
        print(
            f"{len(diverged)} of {len(pairs)} problem file(s) drifted from "
            f"their solutions:\n"
        )
        for problem, drift in diverged:
            rel = problem.relative_to(REPO_ROOT)
            print(f"--- {rel}")
            for line in drift[:25]:
                print(f"    {line}")
            if len(drift) > 25:
                print(f"    ... and {len(drift) - 25} more")
            print()
        print(
            "Problem skeletons must equal their solution outside the "
            "'# FILL ...' regions. Regenerate the skeleton from the solution "
            "(strip '_solution' from anchors, blank the fill-in regions)."
        )
        return 1

    print(f"All {len(pairs)} problem/solution pairs are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
