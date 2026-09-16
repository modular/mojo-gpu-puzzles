# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Check that every puzzle where the learner writes a kernel uses the canary.

A puzzle's value assertions can pass even when the kernel wrote outside
`output`, so a puzzle without the canary silently loses that check.

A puzzle is in scope when its problem file contains a ``# FILL`` marker. That
excludes p09, p30, p31 and p32, which are debugging and profiling exercises
with no kernel for the learner to write, and p10, which contains an
out-of-bounds write on purpose for the ``compute-sanitizer`` lesson.

It also excludes p17 to p22, which are driven from Python and keep their
kernels in ``problems/pNN/op/``, so the glob below does not reach them.

Each puzzle in scope must use the canary in both its problem and solution
files, and its `pixi` task must pass ``-I .`` so the `harness` import resolves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).absolute()
    for d in (here.parent, *here.parents):
        if (d / "problems").is_dir() and (d / "solutions").is_dir():
            return d
    return here.parent.parent


REPO_ROOT = _find_repo_root()
FILL_MARKER = re.compile(r"#\s*FILL")
REQUIRED = (
    ("from harness.canary import PuzzleMemory", "the harness import"),
    ("mem.output(", "an output buffer from the harness"),
    ("mem.verify()", "a call to mem.verify()"),
)


def _pixi_tasks() -> dict[str, str]:
    # Only the [tasks] table. Reading the whole manifest would also pick up
    # keys from [dependencies] and friends, and would silently match a task
    # that had moved into a per-platform override.
    text = (REPO_ROOT / "pixi.toml").read_text()
    table = re.search(
        r"^\[tasks\]$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    if table is None:
        return {}
    return dict(
        re.findall(r"^([A-Za-z0-9_]+) = (.*)$", table.group(1), re.MULTILINE)
    )


def main() -> None:
    tasks = _pixi_tasks()
    problems = sorted((REPO_ROOT / "problems").glob("p*/p*.mojo"))
    failures: list[str] = []
    checked = 0

    for problem in problems:
        if not FILL_MARKER.search(problem.read_text()):
            continue
        checked += 1
        rel = problem.relative_to(REPO_ROOT)
        solution = REPO_ROOT / "solutions" / rel.relative_to("problems")

        for path in (problem, solution):
            if not path.is_file():
                failures.append(
                    f"{rel}: no matching {path.parent.parent.name} file"
                )
                continue
            text = path.read_text()
            for needle, description in REQUIRED:
                if needle not in text:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}: missing {description}"
                    )

        task_name = problem.stem
        task = tasks.get(task_name)
        if task is None:
            failures.append(f"{rel}: no `{task_name}` task in pixi.toml")
        elif "-I ." not in task:
            failures.append(
                f"pixi.toml: task `{task_name}` must pass `-I .` so the"
                " harness import resolves"
            )

    if failures:
        print(f"Checked {checked} learner-authored puzzles.\n")
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} problem(s) found.")
        sys.exit(1)

    print(f"All {checked} learner-authored puzzles use the canary.")


if __name__ == "__main__":
    main()
