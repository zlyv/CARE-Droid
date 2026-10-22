#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ECA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
PYTHON="${ECA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
export ECA_PROJECT="$PROJECT"
"$PYTHON" "$SCRIPT_DIR/validate_reviews.py"
OUTPUT="${ECA_OUTPUT:-$PROJECT/outputs/care-direct-c43-error-attribution-audit-v1}"
cat "$OUTPUT/manual_review_validation.json"
