#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

PYTHON_BIN="${CARE_ML_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
ROOT="artifacts/care-di25-host-complement-neural-v1"

"$PYTHON_BIN" -c 'import numpy,sklearn,torch; print("Dependencies:",numpy.__version__,sklearn.__version__,torch.__version__,"cuda="+str(torch.cuda.is_available()))'
"$PYTHON_BIN" scripts/care-di25-host-complement-neural-v1/run_host_complement_experiment.py --mode full --arm all
"$PYTHON_BIN" scripts/care-di25-host-complement-neural-v1/summarize_host_complement.py

test -f "$ROOT/reports/HOST_COMPLEMENT_NEURAL_REPORT.txt"
grep -q '^HOST_COMPLEMENT_NEURAL_EXPERIMENT_DONE$' "$ROOT/reports/HOST_COMPLEMENT_NEURAL_REPORT.txt"

