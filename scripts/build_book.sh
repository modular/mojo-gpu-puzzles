#!/bin/bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##
# Build the mdbook.
#
# Fonts + lottie-player are fetched from CDN and fall back to a locally
# vendored copy at runtime if that fails (see book/theme/head.hbs and
# book/theme/index.hbs) — this script never needs network access itself.
#
# Usage:
#   bash scripts/build_book.sh              # build to book/html/
#   bash scripts/build_book.sh --offline    # also pre-fetch the runtime fallback
#                                           # assets, so the book is ready to
#                                           # browse with no network access
#                                           # right away

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOK_DIR="$(cd "$SCRIPT_DIR/../book" && pwd)"
OFFLINE_MODE=false
for arg in "$@"; do
    case "$arg" in
        --offline) OFFLINE_MODE=true ;;
    esac
done

$OFFLINE_MODE && bash "$SCRIPT_DIR/fetch_offline_assets.sh"

cd "$BOOK_DIR" && mdbook build
