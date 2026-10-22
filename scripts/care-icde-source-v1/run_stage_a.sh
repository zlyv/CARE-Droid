#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ICDE_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUTPUT="${ICDE_OUTPUT:-$PROJECT/outputs/care-icde-source-v1}"

case "$(realpath -m "$OUTPUT")" in
  "$(realpath -m "$PROJECT")"/outputs/care-icde-source-v*) ;;
  *) echo "BLOCKED_BY_CONTRACT: unsafe ICDE_OUTPUT=$OUTPUT"; exit 2 ;;
esac

if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "BLOCKED_BY_CONTRACT: non-empty output exists: $OUTPUT"; exit 2
fi
mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/stage_a.log") 2>&1

if [[ -n "${ICDE_PYTHON:-}" ]]; then PY="$ICDE_PYTHON"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then PY=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"
else echo "BLOCKED_BY_CONTRACT: no Python interpreter"; exit 2; fi

export ICDE_PROJECT="$PROJECT" ICDE_OUTPUT="$OUTPUT" ICDE_PYTHON="$PY"
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1

echo "CARE_ICDE_STAGE_A_START project=$PROJECT output=$OUTPUT python=$PY"
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" "$SCRIPT_DIR/selftest.py"
"$PY" "$SCRIPT_DIR/audit_stage_a.py"
echo "CARE_ICDE_STAGE_A_DONE"
echo "decision=$OUTPUT/decision.json"
echo "report=$OUTPUT/report.md"

