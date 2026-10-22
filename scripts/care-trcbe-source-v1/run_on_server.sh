#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${TRCBE_PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUTPUT="${TRCBE_OUTPUT:-$PROJECT/outputs/care-trcbe-source-v1}"
case "$(realpath -m "$OUTPUT")" in "$(realpath -m "$PROJECT")"/outputs/care-trcbe-source-v*) ;; *) echo "BLOCKED_BY_CONTRACT unsafe output=$OUTPUT";exit 2;; esac
if [[ -d "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]];then echo "BLOCKED_BY_CONTRACT non-empty output=$OUTPUT";exit 2;fi
mkdir -p "$OUTPUT/logs";exec > >(tee -a "$OUTPUT/logs/run.log") 2>&1
if [[ -n "${TRCBE_PYTHON:-}" ]];then PY="$TRCBE_PYTHON";elif [[ -x /home/boyuanjie/envs/tif/bin/python ]];then PY=/home/boyuanjie/envs/tif/bin/python;elif command -v python3 >/dev/null;then PY="$(command -v python3)";else echo "BLOCKED no python";exit 2;fi
export TRCBE_PROJECT="$PROJECT" TRCBE_OUTPUT="$OUTPUT" PYTHONWARNINGS=ignore LOGURU_LEVEL=ERROR TQDM_DISABLE=1
echo "CARE_TRCBE_LABEL_FREE_START project=$PROJECT output=$OUTPUT"
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" "$SCRIPT_DIR/selftest.py"
"$PY" "$SCRIPT_DIR/run_coverage.py"
echo "CARE_TRCBE_LABEL_FREE_DONE"
echo "decision=$OUTPUT/feasibility_decision.json"
echo "report=$OUTPUT/report.md"

