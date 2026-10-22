#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
DIR=scripts/care-p31rf-c43-dependency-sufficiency-source-v1
$PY "$DIR/selftest.py"
P71=artifacts/p71/verified_dependency_features_2017_v3.csv
if [[ ! -s "$P71" ]]; then
  echo "[build] verified dependency evidence is absent; starting resumable two-worker extraction"
  $PY "$DIR/prepare_dependency_shards.py"
  $PY "$DIR/extract_worker.py" --worker 0 --workers 2 > artifacts/care-p31rf-c43-dependency-sufficiency-source-v1/worker0.log 2>&1 &
  W0=$!
  $PY "$DIR/extract_worker.py" --worker 1 --workers 2 > artifacts/care-p31rf-c43-dependency-sufficiency-source-v1/worker1.log 2>&1 &
  W1=$!
  FAIL=0
  wait "$W0" || FAIL=1
  wait "$W1" || FAIL=1
  if [[ "$FAIL" -ne 0 ]]; then
    echo "BLOCKED_BY_CONTRACT: dependency extraction worker failed; inspect worker0.log and worker1.log"
    exit 1
  fi
  mkdir -p artifacts/p71
  $PY "$DIR/vendor/scripts/build_p71_verified_dependency_features_v3.py" \
    --input artifacts/care-p31rf-c43-dependency-sufficiency-source-v1/dependency_raw \
    --output "$P71" \
    --schema-out artifacts/p71/verified_dependency_features_2017_v3.schema.json
fi
$PY "$DIR/preflight.py"
$PY "$DIR/run_probe.py"
echo P31RF_DEPENDENCY_SUFFICIENCY_FULL_RUN_DONE

