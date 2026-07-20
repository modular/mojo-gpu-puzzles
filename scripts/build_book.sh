#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
# Build English + all translated mdbook languages.
# Auto-discovers languages from book/i18n/*/book.toml
#
# Fonts + lottie-player are fetched from CDN and fall back to a locally
# vendored copy at runtime if that fails (see book/theme/head.hbs and
# book/theme/index.hbs) — this script never needs network access itself.
#
# Usage:
#   bash scripts/build_book.sh                     # CI: translations → html/{lang}/
#   bash scripts/build_book.sh --serve              # Dev: translations → html-{lang}/
#   bash scripts/build_book.sh --offline            # also pre-fetch the runtime
#                                                     # fallback assets, so the book
#                                                     # is ready to browse with no
#                                                     # network access right away

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOK_DIR="$(cd "$SCRIPT_DIR/../book" && pwd)"
SERVE_MODE=false
OFFLINE_MODE=false
for arg in "$@"; do
    case "$arg" in
        --serve) SERVE_MODE=true ;;
        --offline) OFFLINE_MODE=true ;;
    esac
done

$OFFLINE_MODE && bash "$SCRIPT_DIR/fetch_offline_assets.sh"

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
