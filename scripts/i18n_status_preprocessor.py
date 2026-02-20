#!/usr/bin/env python3
"""
mdBook preprocessor for checking translation status.

Compares the commit hash stored in translation files with the
current commit of the English source file. If they differ, injects
a warning banner at the top of the page.

Supports multiple languages via WARNING_MESSAGES dictionary.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# Warning messages by language code
WARNING_MESSAGES: Dict[str, str] = {
    "ko": "이 번역은 원본 문서보다 오래되었을 수 있습니다.",
}
DEFAULT_WARNING = "This translation may be outdated."


def get_warning_banner(lang: str) -> str:
    """Get warning banner HTML for the specified language."""
    message = WARNING_MESSAGES.get(lang, DEFAULT_WARNING)
    return f"""
<div class="i18n-outdated-warning">
  <span class="i18n-warning-icon">⚠️</span>
  <span class="i18n-warning-text">
    {message}
  </span>
</div>
"""


def get_source_commit(source_file: str, book_root: Path) -> str | None:
    """Get the latest commit hash for the English source file."""
    # Path relative to book_root for git command
    relative_path = f"src/{source_file}"
    try:
        result = subprocess.run(
            ["git", "log", "main", "-1", "--format=%H", "--", relative_path],
            capture_output=True,
            text=True,
            cwd=book_root,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def extract_source_commit(content: str) -> str | None:
    """Extract i18n source commit from markdown content."""
    commit_match = re.search(r"<!--\s*i18n-source-commit:\s*([a-f0-9]+)\s*-->", content)
    return commit_match.group(1) if commit_match else None


def process_content(content: str, source_file: str, book_root: Path, lang: str) -> str:
    """Process content and inject warning banner if translation is outdated."""
    source_commit = extract_source_commit(content)

    if not source_commit:
        return content

    current_commit = get_source_commit(source_file, book_root)

    if not current_commit:
        return content

    if source_commit != current_commit:
        warning_banner = get_warning_banner(lang)
        # Insert after first heading
        heading_match = re.search(r"^(#[^\n]+\n)", content, re.MULTILINE)
        if heading_match:
            insert_pos = heading_match.end()
            content = (
                content[:insert_pos]
                + "\n"
                + warning_banner
                + "\n"
                + content[insert_pos:]
            )
        else:
            content = warning_banner + "\n" + content

    return content


def process_items(items: List, book_root: Path, lang: str) -> None:
    """Recursively process chapter items."""
    for item in items:
        if "Chapter" in item:
            chapter = item["Chapter"]
            source_file = chapter.get("path", "")
            if source_file:
                chapter["content"] = process_content(
                    chapter["content"], source_file, book_root, lang
                )
            process_items(chapter.get("sub_items", []), book_root, lang)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        renderer = sys.argv[2] if len(sys.argv) > 2 else ""
        sys.exit(0 if renderer == "html" else 1)

    # Read and parse input
    raw_input = sys.stdin.read()
    try:
        data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        print(raw_input[:500], file=sys.stderr)
        sys.exit(1)

    # mdbook sends [context, book] as a JSON array
    if isinstance(data, list) and len(data) >= 2:
        context, book = data[0], data[1]
    else:
        print(f"Unexpected data format: {type(data)}", file=sys.stderr)
        print(json.dumps(data))
        sys.exit(0)

    # Get language from config (book/i18n/<lang>/)
    config = context.get("config", {})
    lang = config.get("book", {}).get("language", "en")

    # Translation book root is book/i18n/<lang>/, English is at book/
    translation_root = Path(context.get("root", "."))
    english_book_root = (
        translation_root.parent.parent
    )  # book/i18n/<lang> -> book/i18n -> book

    # mdbook 0.5.x uses "items", older versions use "sections"
    items_key = "items" if "items" in book else "sections"
    for section in book.get(items_key, []):
        if "Chapter" in section:
            chapter = section["Chapter"]
            source_file = chapter.get("path", "")
            if source_file:
                chapter["content"] = process_content(
                    chapter["content"], source_file, english_book_root, lang
                )
            process_items(chapter.get("sub_items", []), english_book_root, lang)

    print(json.dumps(book))


if __name__ == "__main__":
    main()
