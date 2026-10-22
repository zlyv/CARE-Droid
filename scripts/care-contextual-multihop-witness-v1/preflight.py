#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from common import (OUTPUT, dump_json, file_record, load_apk_paths, load_attention,
                    metrics, parse_source_formal, read_endpoint_rows, source_assets)


def main() -> None:
    assets = source_assets(); rows = parse_source_formal(assets["formal"]); wanted = {row["sha256"] for row in rows}
    if [sum(row["fold"] == fold for row in rows) for fold in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fixed folds changed")
    if sum(row["routed"] for row in rows) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: DI25 route changed")
    y = np.asarray([row["label"] for row in rows], int)
    base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool)
    formal_probability = np.asarray([row["formal_prob"] for row in rows], float)
    stage1 = metrics(y, base, base, np.zeros(len(rows), bool))
    formal = metrics(y, base, formal_probability, route)
    stage1_error = (base >= .5) != y
    captured_errors = int(np.sum(stage1_error & route))
    if abs(stage1["BA"] - 0.927380952381) > 1e-9 or sum(stage1[key] for key in ("FP", "FN")) != 58:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: Stage-I baseline changed {stage1}")
    if abs(formal["BA"] - 0.932384085213) > 1e-9 or (formal["fixed"], formal["broken"], formal["net"]) != (11, 7, 4):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: formal A0 baseline changed {formal}")
    if captured_errors != 45:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: DI25 capture changed {captured_errors}/58")
    endpoint = read_endpoint_rows(assets["endpoint"], wanted); attention = load_attention(assets["attention"])
    paths, path_audit = load_apk_paths(wanted, future=False)
    if set(endpoint) != wanted or not wanted.issubset(attention) or set(paths) != wanted:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: source coverage endpoint={len(endpoint)} attention={len(wanted & set(attention))} apk={len(paths)}")
    records = [file_record(assets[key], f"protected source {key}") for key in ("formal", "endpoint", "attention")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"git_used": False, "assets": records})
    result = {"status": "PASS", "source_rows": 799, "route": 200, "captured_stage1_errors": captured_errors,
              "error_recall_at_25pct": captured_errors / 58, "route_precision": captured_errors / 200,
              "stage1": stage1, "formal_A0": formal,
              "endpoint_rows": len(endpoint),
              "attention_rows": len(wanted & set(attention)), "apk_paths": len(paths), "future_rows_loaded": 0,
              "assets": {key: str(value) for key, value in assets.items()}, "apk_path_audit": path_audit}
    dump_json(OUTPUT / "preflight.json", result)
    print("CMW_PREFLIGHT_PASS"); print(result, flush=True)


if __name__ == "__main__":
    main()
