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
# Serve the mdbook with live reload.
#
# Usage: pixi run book
#        pixi run book --offline   # also pre-fetch the runtime fallback assets,
#                                   # so the book is ready to browse with no
#                                   # network access right away

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOK_DIR="$(cd "$SCRIPT_DIR/../book" && pwd)"

for arg in "$@"; do
    [[ "$arg" == "--offline" ]] && bash "$SCRIPT_DIR/fetch_offline_assets.sh"
done

cd "$BOOK_DIR" && mdbook serve --open
