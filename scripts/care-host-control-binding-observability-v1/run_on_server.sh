#!/usr/bin/env bash
set -euo pipefail
PROJECT="${HCB_PROJECT:-/home/boyuanjie/projects/tasa-droid}"
SCRIPT="$PROJECT/scripts/care-host-control-binding-observability-v1"
OUTPUT="$PROJECT/outputs/care-host-control-binding-observability-v1"
if [ -e "$OUTPUT" ]; then
  echo "Experiment output exists: do not overwrite; use a new version."; exit 1
fi
export HCB_PROJECT="$PROJECT" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT"
PYTHON="${HCB_PYTHON:-/home/boyuanjie/envs/tif/bin/python}"
"$PYTHON" -B -c 'import ast, pathlib, sys; [ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in pathlib.Path(sys.argv[1]).glob("*.py")]; print("HCB_SYNTAX_PASS")' "$SCRIPT"
"$PYTHON" -B "$SCRIPT/selftest.py"
"$PYTHON" -B "$SCRIPT/run_audit.py"
