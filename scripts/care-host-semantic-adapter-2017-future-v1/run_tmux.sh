#!/usr/bin/env bash
# Execute the complete frozen S1/S3 Adapter experiment with resumable stages.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
PY=/home/boyuanjie/envs/tif/bin/python
DIR=scripts/care-host-semantic-adapter-2017-future-v1
OUT=artifacts/care_host_semantic_adapter_2017_future_v1
LOG=$OUT/logs/experiment.log
mkdir -p "$OUT/logs"
trap 'code=$?; "$PY" - <<PY
import json,time
from pathlib import Path
p=Path("$OUT/logs/run_status.json"); p.write_text(json.dumps({"status":"FAILED","exit_code":$code,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")},indent=2)+"\n")
PY
exit $code' ERR
exec > >(tee -a "$LOG") 2>&1
echo "SEMANTIC_ADAPTER_START $(date -Is)"
$PY "$DIR/preflight.py" --resume
$PY "$DIR/train.py" --smoke --device cuda
$PY "$DIR/train.py" --source --device cuda --resume
$PY "$DIR/train.py" --full --device cuda --resume
$PY "$DIR/evaluate.py" --source
$PY "$DIR/future_extract.py"
$PY "$DIR/evaluate.py" --future --device cuda
$PY "$DIR/finalize.py"
echo "SEMANTIC_ADAPTER_END $(date -Is)"
