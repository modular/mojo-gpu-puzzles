#!/bin/bash
# Serve English + all translated mdbook languages with live reload.
#
# - Auto-discovers languages from book/i18n/*/book.toml
# - English: mdbook serve (WebSocket live reload)
# - Translations: mdbook watch + polling live reload (livereload-poll.js)
# - Symlinks book/html/{lang} -> ../html-{lang} bridge the outputs.
#
# Usage: pixi run book

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOK_DIR="$(cd "$SCRIPT_DIR/../book" && pwd)"

# Discover translation languages from book/i18n/*/book.toml
LANGS=()
for toml in "$BOOK_DIR"/i18n/*/book.toml; do
    [ -f "$toml" ] || continue
    LANGS+=("$(basename "$(dirname "$toml")")")
done

PIDS=()
cleanup() {
    kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null
    for lang in "${LANGS[@]}"; do rm -f "$BOOK_DIR/html/$lang"; done
}

trap cleanup INT TERM EXIT

ensure_symlinks() {
    for lang in "${LANGS[@]}"; do
        local link="$BOOK_DIR/html/$lang"
        [ -L "$link" ] || ln -sfn "../html-$lang" "$link" 2>/dev/null || true
    done
}

# Build both (translations to separate directories)
bash "$SCRIPT_DIR/build_book.sh" --serve
ensure_symlinks

# Background: translation watchers + symlink restorer
for lang in "${LANGS[@]}"; do
    mdbook watch "$BOOK_DIR/i18n/$lang" --dest-dir "../../html-$lang" & PIDS+=($!)
done
# Restore symlinks after English rebuilds clean book/html/.
# Each iteration is just an lstat check per language (~0 CPU cost); negligible
# compared to the filesystem I/O mdbook serve already does for file-change detection.
(while true; do ensure_symlinks; sleep 0.2; done) & PIDS+=($!)

# Foreground: English server
echo ""
echo "  English: http://localhost:3000/"
for lang in "${LANGS[@]}"; do
    printf "  %-7s  http://localhost:3000/%s/\n" "$lang:" "$lang"
done
echo ""
cd "$BOOK_DIR" && mdbook serve --open & PIDS+=($!)
wait
