#!/usr/bin/env python3
from __future__ import annotations

from common import (ARMS, FEATURE_DIR, FORMAL, OUTPUT, SOURCE_OUTPUT, dump_json,
                    file_record, parse_formal, read_gzip_json, read_json)


def main() -> None:
    rows = parse_formal(); coverage_path = SOURCE_OUTPUT / "coverage_gate.json"
    decision_path = SOURCE_OUTPUT / "final_decision.json"
    coverage = read_json(coverage_path); decision = read_json(decision_path)
    if coverage.get("status") != "FAIL" or decision.get("stopped_at") != "LABEL_FREE_COVERAGE_GATE":
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected frozen v1 coverage-gate failure")
    expected = {row["sha256"] for row in rows}
    files = {path.name.removesuffix(".json.gz") for path in FEATURE_DIR.glob("*.json.gz")}
    if files != expected:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: feature cohort mismatch files={len(files)} expected={len(expected)}")
    first = read_gzip_json(FEATURE_DIR / f"{rows[0]['sha256']}.json.gz")
    absent = {name for name in ARMS if name not in {"COUNT_ONLY", "PATH_ONLY"} and name not in first.get("variants", {})}
    if absent:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing frozen variants {sorted(absent)}")
    folds = [sum(row["fold"] == fold for row in rows) for fold in range(1, 6)]
    if folds != [160, 160, 160, 160, 159] or sum(row["routed"] for row in rows) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: formal protocol changed folds={folds}")
    assets = [file_record(FORMAL, "frozen P31-RF/DI25/A0 source predictions"),
              file_record(coverage_path, "frozen failed label-free coverage gate"),
              file_record(decision_path, "frozen v1 decision")]
    dump_json(OUTPUT / "input_asset_manifest_before.json", {"assets": assets})
    result = {"status": "PASS", "source_rows": 799, "feature_rows": 799, "fold_sizes": folds,
              "route": 200, "future_rows_loaded": 0, "v1_coverage": coverage,
              "scope": "post-gate source-only information audit; cannot revise v1 decision"}
    dump_json(OUTPUT / "preflight.json", result)
    print("CMW_POSTGATE_PREFLIGHT_PASS"); print(result, flush=True)


if __name__ == "__main__":
    main()
