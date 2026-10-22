#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PYTHON_BIN="${PYTHON_BIN:-/home/boyuanjie/envs/tif/bin/python}"
CHECK="scripts/care-source-recovery-stage2only-v1/check_complete.py"
"$PYTHON_BIN" "$CHECK" csv artifacts/care-malben-host-v2/source/oof_predictions.csv --expected 799
"$PYTHON_BIN" "$CHECK" csv artifacts/care-stage2only-malben-v1/source/oof_predictions.csv --expected 799
"$PYTHON_BIN" scripts/care-malben-host-v2/finalize_main_table.py
cat artifacts/care-malben-host-v2/source/main_comparison.txt
