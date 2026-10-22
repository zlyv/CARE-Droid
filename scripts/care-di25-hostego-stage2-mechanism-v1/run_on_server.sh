#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid

ML_PY=/home/boyuanjie/envs/tif/bin/python
"$ML_PY" -c 'import numpy, sklearn, torch; print("Dependencies:", numpy.__version__, sklearn.__version__, torch.__version__, "cuda="+str(torch.cuda.is_available()))'
"$ML_PY" scripts/care-di25-hostego-stage2-mechanism-v1/selftest.py
"$ML_PY" scripts/care-di25-hostego-stage2-mechanism-v1/preflight.py
"$ML_PY" scripts/care-di25-hostego-stage2-mechanism-v1/run_instrumented_m1.py
"$ML_PY" scripts/care-di25-hostego-stage2-mechanism-v1/analyze_mechanism.py

cat artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/FINAL_STAGE2_MECHANISM_REPORT.txt
