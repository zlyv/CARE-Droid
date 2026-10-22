#!/usr/bin/env bash
# Run the full read-only CARE API/Manifest fusion experiment in an independent tmux session.
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=1
PY=/home/boyuanjie/envs/tif/bin/python
HERE=scripts/care-api-manifest-fusion-2017-future-v1
OUT=artifacts/care_api_manifest_fusion_2017_future_v1
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/experiment.log") 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then "$PY" -c "import json,time,pathlib; p=pathlib.Path(\"$OUT/logs/run_status.json\"); p.write_text(json.dumps({\"status\":\"FAILED\",\"stage\":\"launcher\",\"exit_code\":$code,\"updated_at\":time.strftime(\"%Y-%m-%dT%H:%M:%S%z\")},indent=2))"; fi; exit "$code"' EXIT
printf 'API_MANIFEST_RUN_START %s\n' "$(date --iso-8601=seconds)"
"$PY" "$HERE/run_experiment.py" preflight --resume
"$PY" "$HERE/api_extract.py" --workers 2
"$PY" "$HERE/run_experiment.py" w0 --device cuda
"$PY" "$HERE/run_experiment.py" train-source --device cuda
"$PY" "$HERE/run_experiment.py" evaluate-source
"$PY" "$HERE/run_experiment.py" train-full --device cuda
"$PY" "$HERE/run_experiment.py" evaluate-future
"$PY" "$HERE/run_experiment.py" integrity
printf 'API_MANIFEST_RUN_END %s\n' "$(date --iso-8601=seconds)"
