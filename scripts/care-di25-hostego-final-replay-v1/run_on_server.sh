#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
"$PY" -c 'import numpy,scipy,sklearn,torch; print("Dependencies:",numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__,"cuda="+str(torch.cuda.is_available()))'
"$PY" scripts/care-di25-hostego-final-replay-v1/selftest.py
"$PY" scripts/care-di25-hostego-final-replay-v1/preflight.py
"$PY" scripts/care-di25-hostego-final-replay-v1/run_final_replay.py --mode dryrun
"$PY" scripts/care-di25-hostego-final-replay-v1/run_final_replay.py --mode full
"$PY" scripts/care-di25-hostego-final-replay-v1/finalize_replay.py
cat artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/final_replay/FINAL_SEED_RELIABILITY_REPORT.txt
