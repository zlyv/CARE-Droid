#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FULLGRAPH_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUTPUT="${FULLGRAPH_OUTPUT:-$PROJECT/outputs/care-fullgraph-trcbe-v2}"
PROJECT_REAL="$(realpath -m "$PROJECT")"
OUTPUT_REAL="$(realpath -m "$OUTPUT")"
case "$OUTPUT_REAL" in "$PROJECT_REAL"/outputs/care-fullgraph-trcbe-v*) ;; *) echo "BLOCKED_BY_CONTRACT unsafe output=$OUTPUT_REAL"; exit 2;; esac
mkdir -p "$OUTPUT_REAL/logs"
OWNER="$OUTPUT_REAL/.CARE_FULLGRAPH_TRCBE_OWNER"
if [[ -d "$OUTPUT_REAL" && -n "$(find "$OUTPUT_REAL" -mindepth 1 -maxdepth 1 ! -name logs ! -name .CARE_FULLGRAPH_TRCBE_OWNER -print -quit 2>/dev/null)" && ! -f "$OWNER" ]]; then
  echo "BLOCKED_BY_CONTRACT non-empty foreign output=$OUTPUT_REAL"; exit 2
fi
printf '%s\n' 'care-fullgraph-trcbe-v1-r2' > "$OWNER"
exec > >(tee -a "$OUTPUT_REAL/logs/run.log") 2>&1
if [[ -n "${FULLGRAPH_PYTHON:-}" ]]; then PY="$FULLGRAPH_PYTHON"
elif [[ -x /home/boyuanjie/envs/tif/bin/python ]]; then PY=/home/boyuanjie/envs/tif/bin/python
elif command -v python3 >/dev/null; then PY="$(command -v python3)"
else echo "BLOCKED_BY_CONTRACT no Python interpreter"; exit 2; fi
export FULLGRAPH_PROJECT="$PROJECT_REAL" FULLGRAPH_OUTPUT="$OUTPUT_REAL"
export PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1
echo "CARE_FULLGRAPH_TRCBE_PIPELINE_START project=$PROJECT_REAL output=$OUTPUT_REAL python=$PY"
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" "$SCRIPT_DIR/selftest.py"
"$PY" "$SCRIPT_DIR/run_pipeline.py"
echo "CARE_FULLGRAPH_TRCBE_PIPELINE_DONE"
echo "decision=$OUTPUT_REAL/final_decision.json"
echo "report=$OUTPUT_REAL/report.md"
