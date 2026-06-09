#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Dependency-free link/asset checker for the mojo-gpu-puzzles book.

mdbook / mdbook-linkcheck are not guaranteed to be available, so this is a
stdlib-only validator (Python 3.10+) that walks the book's Markdown sources and
verifies that every local link and image target resolves to a file that exists.

Resolution rules (derived from the verified build topology):

  * English book builds ``book/src`` -> ``book/html`` with assets copied
    adjacent to their sources, so English pages use relative paths.
  * Korean book builds ``book/i18n/ko/src`` -> ``book/html/ko``, but the ko
    source tree contains ZERO media files. Korean pages therefore reference
    assets via site-absolute paths (``/puzzle_NN/media/...``) that resolve at
    the site root, which the ENGLISH source tree mirrors once built.

Two kinds of references are extracted from each ``*.md`` file:
  (a) Markdown links/images ``[text](target)`` and ``![alt](target)``.
  (b) HTML ``<img src="target">`` tags.

Each non-skipped target is resolved as follows:
  * ``http(s)://`` / ``mailto:`` / pure ``#anchor`` targets are skipped.
  * ``{{#include ...}}`` directives are skipped (different resolution model).
  * Fenced code blocks (``` ``` ```) and inline code spans (`` `...` ``) are
    stripped first: per CommonMark, links/images do not parse inside code, and
    these pages contain code like ``out.tile[size](id)`` that otherwise looks
    like a Markdown link.
  * A site-absolute target (starts with ``/``) resolves against ``book/src``
    (the English source tree, which mirrors the built site root) for BOTH
    English and Korean files.
  * A relative target resolves against the directory of the ``.md`` file within
    its own source tree (en -> book/src, ko -> book/i18n/ko/src).
  * ``#fragments`` and ``?query`` suffixes are stripped before resolution.

Exits nonzero and lists ``file:line  ->  target`` for every unresolved
reference; exits 0 with a summary count when everything resolves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

# Markdown link/image: capture the target inside (...) of [text](target).
MD_REF = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# HTML <img src="..."> (single or double quoted).
IMG_REF = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
INCLUDE = re.compile(r"\{\{#include")
# Opening/closing fence for code blocks (``` or ~~~, optional indent + info).
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
# Inline code spans (`...` / ``...``) — stripped so code is not parsed as links.
INLINE_CODE = re.compile(r"(`+)(?:.+?)\1")

# Targets we deliberately do not resolve. Keep empty unless a genuinely
# ambiguous reference surfaces; document any entry here and in the PR body.
ALLOWLIST: set[str] = set()


def is_external(target: str) -> bool:
    return target.startswith(
        ("http://", "https://", "mailto:", "#")
    ) or target.startswith("data:")


def resolve(target: str, md_file: Path, en_root: Path) -> Path:
    """Map a reference target to the filesystem path it should point at."""
    clean = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if clean.startswith("/"):
        # Site-absolute -> English source tree (mirrors the built site
        # root) for en and ko alike.
        return (en_root / clean.lstrip("/")).resolve()
    return (md_file.parent / clean).resolve()


def check_tree(src_root: Path, en_root: Path) -> list[tuple[Path, int, str]]:
    misses: list[tuple[Path, int, str]] = []
    for md_file in sorted(src_root.rglob("*.md")):
        in_fence = False
        fence_marker = ""
        for lineno, line in enumerate(
            md_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fence = FENCE.match(line)
            if fence:
                marker = fence.group(1)[0]  # ` or ~
                if not in_fence:
                    in_fence, fence_marker = True, marker
                elif marker == fence_marker:
                    in_fence = False
                continue
            if in_fence or INCLUDE.search(line):
                continue
            line = INLINE_CODE.sub("", line)  # drop inline code spans
            for target in MD_REF.findall(line) + IMG_REF.findall(line):
                if is_external(target) or not target or target in ALLOWLIST:
                    continue
                if not resolve(target, md_file, en_root).exists():
                    misses.append((md_file, lineno, target))
    return misses


def main() -> int:
    book = Path(__file__).resolve().parent.parent / "book"
    en_root = book / "src"
    ko_root = book / "i18n" / "ko" / "src"
    trees = [(en_root, en_root), (ko_root, en_root)]

    all_misses: list[tuple[Path, int, str]] = []
    checked = 0
    for src_root, site_root in trees:
        if not src_root.is_dir():
            continue
        checked += sum(1 for _ in src_root.rglob("*.md"))
        all_misses.extend(check_tree(src_root, site_root))

    repo = book.parent
    if all_misses:
        print(
            f"FAIL: {len(all_misses)} unresolved reference(s):", file=sys.stderr
        )
        for md_file, lineno, target in all_misses:
            rel = md_file.relative_to(repo)
            print(f"  {rel}:{lineno}  ->  {target}", file=sys.stderr)
        return 1

    print(f"OK: all references resolved across {checked} markdown file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
