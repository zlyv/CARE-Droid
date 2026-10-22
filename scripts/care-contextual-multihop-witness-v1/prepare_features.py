#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (OUTPUT, diagnostic_assets, dump_json, load_apk_paths, load_attention,
                    parse_diagnostic_rows, parse_source_formal, read_endpoint_rows,
                    read_gzip_json, source_assets, write_gzip_json)
from witness import extract

SCRIPT = Path(__file__).resolve().parent
GRAPH_DIR = OUTPUT / "cache/callsite_graphs"
STATUS_DIR = OUTPUT / "checkpoints/graph_status"


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def cohort_inputs(name: str):
    if name == "source":
        assets = source_assets(); rows = parse_source_formal(assets["formal"])
        endpoint, attention = assets["endpoint"], assets["attention"]
        apk_map, audit = load_apk_paths({row["sha256"] for row in rows}, future=False)
    else:
        assets = diagnostic_assets(required=True)
        rows = parse_diagnostic_rows(assets["frozen"])
        endpoint = assets["endpoint"]
        attention = assets["frozen_attention"] if name == "future_frozen" else assets["within_attention"]
        apk_map, audit = load_apk_paths({row["sha256"] for row in rows}, future=True)
    return rows, endpoint, attention, apk_map, audit


def build_one(sid: str, apk: Path, timeout: int) -> dict:
    graph = GRAPH_DIR / f"{sid}.json.gz"; status = STATUS_DIR / f"{sid}.json"
    if graph.is_file() and status.is_file():
        saved = json.loads(status.read_text(encoding="utf-8"))
        if saved.get("status") == "OK":
            return saved
    command = [sys.executable, str(SCRIPT / "graph_builder.py"), "--apk", str(apk), "--sha256", sid,
               "--output", str(graph), "--status", str(status)]
    try:
        subprocess.run(command, check=False, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        dump_json(status, {"sha256": sid, "status": "TIMEOUT", "timeout_seconds": timeout})
    if not status.is_file():
        return {"sha256": sid, "status": "ERROR", "error": "worker produced no status"}
    return json.loads(status.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--cohort", choices=("source", "future_frozen", "future_within"), required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CMW_GRAPH_WORKERS", "2")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("CMW_GRAPH_TIMEOUT", "1800")))
    args = parser.parse_args(); started = time.monotonic()
    rows, endpoint_path, attention_path, apk_map, path_audit = cohort_inputs(args.cohort)
    wanted = {row["sha256"] for row in rows}
    dump_json(OUTPUT / f"audit/{args.cohort}_apk_paths.json", {"wanted": len(wanted), "resolved": len(apk_map), "audit": path_audit})
    if set(apk_map) != wanted:
        missing = sorted(wanted - set(apk_map))
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK coverage {len(apk_map)}/{len(wanted)} missing_first={missing[:10]}")
    endpoints = read_endpoint_rows(endpoint_path, wanted)
    if set(endpoints) != wanted:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint coverage {len(endpoints)}/{len(wanted)}")
    attention = load_attention(attention_path)
    if not wanted.issubset(attention):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: attention coverage {len(wanted & set(attention))}/{len(wanted)}")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True); STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(wanted); statuses = []; complete = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(build_one, sid, apk_map[sid], args.timeout): sid for sid in ordered}
        for future in as_completed(futures):
            statuses.append(future.result()); complete += 1
            if complete % 10 == 0 or complete == len(ordered):
                elapsed = time.monotonic() - started; rate = complete / max(elapsed, 1e-6)
                failed = sum(row.get("status") != "OK" for row in statuses)
                print(f"CMW_GRAPH_BUILD cohort={args.cohort} {complete}/{len(ordered)} failed={failed} rate={rate:.3f}/s ETA={(len(ordered)-complete)/max(rate,1e-9)/60:.1f}min", flush=True)
    failed = [row for row in statuses if row.get("status") != "OK"]
    graph_times = [row["wall_seconds"] for row in statuses if row.get("status") == "OK" and row.get("wall_seconds") is not None]
    build_elapsed = time.monotonic() - started
    dump_json(OUTPUT / f"audit/{args.cohort}_graph_build.json", {"rows": len(ordered), "ok": len(ordered) - len(failed), "failed": failed,
              "observed_worker_seconds_sum": sum(graph_times), "per_apk_seconds_median": percentile(graph_times, .5),
              "per_apk_seconds_p90": percentile(graph_times, .9), "per_apk_seconds_p95": percentile(graph_times, .95),
              "phase_wall_seconds": build_elapsed})
    if len(failed) > .10 * len(ordered):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: graph failures {len(failed)}/{len(ordered)}")

    feature_dir = OUTPUT / "cache/features" / args.cohort; feature_dir.mkdir(parents=True, exist_ok=True)
    feature_started = time.monotonic(); completed, feature_failures = 0, []
    for sid in ordered:
        target = feature_dir / f"{sid}.json.gz"
        if target.is_file():
            completed += 1; continue
        graph_path = GRAPH_DIR / f"{sid}.json.gz"
        if not graph_path.is_file():
            feature_failures.append({"sha256": sid, "error": "missing graph"}); continue
        try:
            evidence = extract(read_gzip_json(graph_path), endpoints[sid], attention[sid])
            write_gzip_json(target, evidence); completed += 1
        except Exception as exc:
            feature_failures.append({"sha256": sid, "error_type": type(exc).__name__, "error": str(exc)})
        if (completed + len(feature_failures)) % 10 == 0 or completed + len(feature_failures) == len(ordered):
            print(f"CMW_FEATURE_EXTRACT cohort={args.cohort} completed={completed}/{len(ordered)} failed={len(feature_failures)}", flush=True)
    dump_json(OUTPUT / f"audit/{args.cohort}_feature_build.json", {"rows": len(ordered), "completed": completed,
              "failures": feature_failures, "endpoint": str(endpoint_path), "attention": str(attention_path),
              "phase_wall_seconds": time.monotonic() - feature_started,
              "total_prepare_wall_seconds": time.monotonic() - started})
    if feature_failures or completed != len(ordered):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: feature extraction incomplete completed={completed} failed={len(feature_failures)}")
    print(f"CMW_PREPARE_DONE cohort={args.cohort} rows={completed}", flush=True)


if __name__ == "__main__":
    main()
