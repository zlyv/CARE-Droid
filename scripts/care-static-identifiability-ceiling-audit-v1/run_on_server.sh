#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${SICA_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
OUTPUT="${SICA_OUTPUT:-$PROJECT/outputs/care-static-identifiability-ceiling-audit-v1}"
PYTHON="${SICA_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || true)"; fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then echo "BLOCKED_BY_CONTRACT: no Python" >&2; exit 1; fi
case "$(realpath -m "$OUTPUT")" in
  "$(realpath "$PROJECT")"/outputs/care-static-identifiability-ceiling-audit-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe output=$OUTPUT" >&2; exit 1 ;;
esac
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoints"
export SICA_PROJECT="$PROJECT" SICA_OUTPUT="$OUTPUT" SICA_PYTHON="$PYTHON"
phase() {
  local name="$1"; shift
  echo "SICA_PHASE_START name=$name epoch=$(date +%s)"
  if "$@"; then echo "SICA_PHASE_DONE name=$name epoch=$(date +%s)";
  else echo "SICA_PHASE_FAIL name=$name epoch=$(date +%s)"; return 1; fi
}
echo "CARE_STATIC_IDENTIFIABILITY_CEILING_START output=$OUTPUT python=$PYTHON"
phase syntax "$PYTHON" -m py_compile "$HERE"/*.py
phase selftest "$PYTHON" "$HERE/selftest.py"
phase preflight "$PYTHON" "$HERE/preflight.py"
phase audit "$PYTHON" "$HERE/run_audit.py"
echo "CARE_STATIC_IDENTIFIABILITY_CEILING_DONE"
