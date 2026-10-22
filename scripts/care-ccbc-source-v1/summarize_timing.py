#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import OUTPUT, dump_json, read_csv, read_json


def distribution(values):
    values = np.asarray([float(value) for value in values if np.isfinite(float(value))], float)
    if not len(values):
        return {"status": "unavailable", "n": 0}
    return {
        "status": "measured", "n": int(len(values)), "median_seconds": float(np.median(values)),
        "p90_seconds": float(np.quantile(values, .90)), "p95_seconds": float(np.quantile(values, .95)),
        "total_seconds": float(values.sum()),
    }


def main() -> None:
    extraction = read_json(OUTPUT / "contract_extraction_summary.json")
    mapping_rows = read_csv(OUTPUT / "component_mapping_by_apk.csv")
    phases_path = OUTPUT / "phase_timing.jsonl"
    phases = []
    if phases_path.is_file():
        for line in phases_path.read_text(encoding="utf-8").splitlines():
            if line.strip(): phases.append(json.loads(line))
    contract_shards = []
    for path in (OUTPUT / "cache/contracts").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") == "OK": contract_shards.append(float(row.get("extraction_seconds", float("nan"))))
        except Exception:
            pass
    result = {
        "manifest_contract_extraction_per_apk": distribution(contract_shards),
        "component_scope_per_apk": distribution([row.get("scope_seconds", "nan") for row in mapping_rows]),
        "pipeline_phases": phases,
        "cold_cache": "not separately measured; no inferred speedup claim",
        "warm_cache": "resume behavior measured only by phase wall clock where present",
        "peak_memory": "unavailable unless /usr/bin/time -v is present in launcher log",
        "P31_and_formal_C43": "reused frozen artifacts; not re-attributed to CCBC",
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "timing.json", result)
    print("CCBC_TIMING_SUMMARY_DONE")


if __name__ == "__main__":
    main()
