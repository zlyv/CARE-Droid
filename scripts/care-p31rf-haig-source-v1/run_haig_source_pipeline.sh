#!/usr/bin/env bash
set -euo pipefail
cd /home/boyuanjie/projects/tasa-droid
PY=/home/boyuanjie/envs/tif/bin/python
S=scripts/care-p31rf-haig-source-v1
A=artifacts/care-p31rf-haig-source-v1
mkdir -p "$A/logs" "$A/graphs/source" "$A/graphs/future" "$A/future" "$A/models"

run_workers() {
  local scope="$1" lineage="$2" output="$3" limit="${4:-0}"
  local extra=(); if [[ "$limit" != 0 ]]; then extra=(--limit "$limit"); fi
  "$PY" "$S/haig_extract.py" --lineage "$lineage" --output "$output" --worker 0 --workers 2 "${extra[@]}" 2>&1 | tee "$A/logs/${scope}.worker0.log" & p0=$!
  "$PY" "$S/haig_extract.py" --lineage "$lineage" --output "$output" --worker 1 --workers 2 "${extra[@]}" 2>&1 | tee "$A/logs/${scope}.worker1.log" & p1=$!
  local s0=0 s1=0
  wait "$p0" || s0=$?
  wait "$p1" || s1=$?
  if [[ "$s0" -ne 0 || "$s1" -ne 0 ]]; then
    echo "BLOCKED_BY_CONTRACT: extraction workers failed scope=$scope worker0=$s0 worker1=$s1"
    return 1
  fi
}

"$PY" -m py_compile "$S"/*.py
"$PY" "$S/selftest.py"
"$PY" "$S/preflight.py"
"$PY" "$S/prepare.py" --scope source
"$PY" "$S/schemas.py"
"$PY" "$S/feasibility.py"

echo HAIG_EXTRACTION_STARTED
"$PY" "$S/haig_extract.py" --lineage "$A/lineage/source_2017.csv" --output "$A/graphs/source" --worker 0 --workers 1 --limit 5 --progress-every 1
"$PY" "$S/validate_extraction.py" --lineage "$A/lineage/source_2017.csv" --graphs "$A/graphs/source" --limit 5 --phase five_apk
run_workers audit25 "$A/lineage/source_2017.csv" "$A/graphs/source" 25
"$PY" "$S/validate_extraction.py" --lineage "$A/lineage/source_2017.csv" --graphs "$A/graphs/source" --limit 25 --phase twentyfive_apk
run_workers source "$A/lineage/source_2017.csv" "$A/graphs/source"
"$PY" "$S/collect_extraction.py" --lineage "$A/lineage/source_2017.csv" --graphs "$A/graphs/source" --scope source
echo HAIG_EXTRACTION_DONE

echo SOURCE_PROBE_STARTED
"$PY" "$S/run_source_probe.py"
echo SOURCE_PROBE_DONE

# Source choice is now frozen. Future extraction and both diagnostics start only here.
echo HAIG_FUTURE_DIAGNOSTICS_STARTED
"$PY" "$S/preflight_future.py"
"$PY" "$S/prepare.py" --scope future
run_workers future "$A/lineage/future_2018_2021.csv" "$A/graphs/future"
"$PY" "$S/collect_extraction.py" --lineage "$A/lineage/future_2018_2021.csv" --graphs "$A/graphs/future" --scope future
"$PY" "$S/run_future_diagnostics.py"
"$PY" "$S/run_within_year.py" --years 2018,2019 --output "$A/future/within_2018_2019" 2>&1 | tee "$A/logs/within_2018_2019.log" & w0=$!
"$PY" "$S/run_within_year.py" --years 2020,2021 --output "$A/future/within_2020_2021" 2>&1 | tee "$A/logs/within_2020_2021.log" & w1=$!
s0=0; s1=0; wait "$w0" || s0=$?; wait "$w1" || s1=$?
if [[ "$s0" -ne 0 || "$s1" -ne 0 ]]; then echo "BLOCKED_BY_CONTRACT: within-year workers failed $s0/$s1"; exit 1; fi
"$PY" "$S/merge_within_year.py"
"$PY" "$S/summarize.py"
echo CARE_HAIG_SOURCE_PIPELINE_DONE
echo "artifacts=$A"
echo "log=$A.console.log"
echo "final_summary=$A/final_summary.json"
