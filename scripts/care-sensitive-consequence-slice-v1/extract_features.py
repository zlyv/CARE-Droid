#!/usr/bin/env python3
from __future__ import annotations

import time

from common import OUTPUT, assets, load_attention, parse_formal, read_endpoint_rows, read_gzip_json, write_csv, write_gzip_json
from effect_slice import extract


def main() -> None:
    started = time.monotonic(); found = assets(); rows = parse_formal(found["formal"])
    wanted = {row["sha256"] for row in rows}; endpoints = read_endpoint_rows(found["endpoint"], wanted)
    attention = load_attention(found["attention"]); folder = OUTPUT / "cache/features/source"; folder.mkdir(parents=True, exist_ok=True)
    audit_rows = []; failures = []
    for index, row in enumerate(rows, 1):
        sid = row["sha256"]; target = folder / f"{sid}.json.gz"
        try:
            if target.is_file():
                record = read_gzip_json(target)
            else:
                graph = read_gzip_json(found["graphs"] / f"{sid}.json.gz")
                record = extract(graph, endpoints[sid], attention[sid]); write_gzip_json(target, record)
            audit_rows.append({"sha256": sid, "fold": row["fold"], "routed": row["routed"], **record["audit"]})
        except Exception as exc:
            failures.append({"sha256": sid, "error_type": type(exc).__name__, "error": str(exc)})
        if index % 25 == 0 or index == len(rows):
            elapsed = time.monotonic() - started; rate = index / max(elapsed, 1e-9)
            print(f"SCS_FEATURES {index}/{len(rows)} failed={len(failures)} rate={rate:.2f}/s ETA={(len(rows)-index)/max(rate,1e-9)/60:.1f}min", flush=True)
    write_csv(OUTPUT / "slice_coverage_rows.csv", audit_rows)
    from common import dump_json
    dump_json(OUTPUT / "feature_extraction.json", {"rows": len(rows), "completed": len(audit_rows), "failures": failures,
        "wall_seconds": time.monotonic() - started, "future_rows_loaded": 0})
    if failures or len(audit_rows) != len(rows):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: feature extraction completed={len(audit_rows)} failures={len(failures)}")
    print("SCS_SOURCE_FEATURES_DONE rows=799", flush=True)


if __name__ == "__main__":
    main()
