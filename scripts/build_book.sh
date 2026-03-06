#!/bin/bash
# Build English + all translated mdbook languages.
# Auto-discovers languages from book/i18n/*/book.toml
#
# Usage:
#   bash scripts/build_book.sh          # CI: translations → html/{lang}/
#   bash scripts/build_book.sh --serve  # Dev: translations → html-{lang}/

set -euo pipefail

BOOK_DIR="$(cd "$(dirname "$0")/../book" && pwd)"
SERVE_MODE=false
[[ "${1:-}" == "--serve" ]] && SERVE_MODE=true

# Discover translation languages
LANGS=()
for toml in "$BOOK_DIR"/i18n/*/book.toml; do
    [ -f "$toml" ] || continue
    LANGS+=("$(basename "$(dirname "$toml")")")
done

if $SERVE_MODE; then
    # Serve mode: build translations to separate directories (html-{lang}/)
    # so English rebuilds don't destroy them.
    # Translations first: English build cleans html/, so building English
    # after translations avoids immediately destroying the symlinks.
    for lang in "${LANGS[@]}"; do
        echo "[$lang] Building..."
        mdbook build "$BOOK_DIR/i18n/$lang" --dest-dir "../../html-$lang"
    done

    echo "[en] Building..."
    (cd "$BOOK_DIR" && mdbook build)

    for lang in "${LANGS[@]}"; do
        cp -r "$BOOK_DIR/html/theme" "$BOOK_DIR/html-$lang/" 2>/dev/null || true
    done
else
    # CI mode: everything under html/ for single-directory deployment.
    echo "[en] Building..."
    (cd "$BOOK_DIR" && mdbook build)

    for lang in "${LANGS[@]}"; do
        echo "[$lang] Building..."
        mdbook build "$BOOK_DIR/i18n/$lang"
        cp -r "$BOOK_DIR/html/theme" "$BOOK_DIR/html/$lang/"
    done
fi
