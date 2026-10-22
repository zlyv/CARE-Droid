#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    OUTPUT, PROJECT, discover_assets, dump_json, endpoint_hosts, ensure_output_boundary,
    file_record, load_apk_map, load_endpoint_rows, read_csv, read_gzip_json,
    sha256_file, svm_route_rows, write_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = OUTPUT / "cache/guard_scan"
STATUS = OUTPUT / "checkpoints/scan_status"
PILOT_SIZE_PER_FOLD = 6
TIMEOUT_SECONDS = int(os.environ.get("SGCA_APK_TIMEOUT_SECONDS", "900"))


def select_pilot(routed: list[dict[str, Any]], endpoints: dict[str, dict[str, Any]], apk_map: dict[str, Path]) -> list[dict[str, Any]]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in routed:
        sid = row["sha256"]
        by_fold[row["fold"]].append({"sha256": sid, "fold": row["fold"],
                                     "host_count": len(endpoint_hosts(endpoints[sid])),
                                     "apk_size_bytes": apk_map[sid].stat().st_size})
    selected = []
    for fold in range(1, 6):
        rows = sorted(by_fold[fold], key=lambda row: (row["host_count"], row["apk_size_bytes"], row["sha256"]))
        indexes = []
        for rank in range(PILOT_SIZE_PER_FOLD):
            index = round(rank * (len(rows) - 1) / (PILOT_SIZE_PER_FOLD - 1))
            if index not in indexes:
                indexes.append(index)
        if len(indexes) != PILOT_SIZE_PER_FOLD:
            raise RuntimeError("BLOCKED_BY_CONTRACT: pilot quantiles not unique")
        for rank, index in enumerate(indexes):
            selected.append({**rows[index], "quantile_rank": rank,
                             "selection_reason": "label-free SVM-routed fold-stratified Host-count sextile"})
    return selected


def hash_manifest(paths: list[tuple[Path, str]], phase: str) -> list[dict[str, Any]]:
    rows = []
    for index, (path, purpose) in enumerate(paths, 1):
        rows.append(file_record(path, purpose))
        if index % 10 == 0 or index == len(paths):
            print(f"SGCA_ASSET_HASH phase={phase} {index}/{len(paths)}", flush=True)
    return rows


def scan_one(row: dict[str, Any], apk: Path, mapping: Path) -> dict[str, Any]:
    sid = row["sha256"]
    output = CACHE / f"{sid}.json.gz"
    status_path = STATUS / f"{sid}.json"
    if output.is_file() and status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "OK":
            return existing
    command = [sys.executable, str(SCRIPT_DIR / "scan_one.py"), "--apk", str(apk), "--sha256", sid,
               "--mapping", str(mapping), "--output", str(output), "--status", str(status_path)]
    try:
        subprocess.run(command, check=False, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        dump_json(status_path, {"sha256": sid, "status": "TIMEOUT", "timeout_seconds": TIMEOUT_SECONDS})
    if not status_path.is_file():
        return {"sha256": sid, "status": "ERROR", "error": "worker produced no status"}
    return json.loads(status_path.read_text(encoding="utf-8"))


def report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    return f"""# CARE-Droid sensitive guard-context pilot audit v1

## Decision

**{decision['status']}**

This is a label-free 2017 SVM-routed pilot. It adds no classifier, threshold, gate, correction rule or future-year feedback.

## Coverage

- Selected APKs: {aggregate['selected_apks']} (six Host-count strata per fold).
- Successful scans: {aggregate['successful_scans']} ({aggregate['scan_success_rate']:.3f}).
- APKs with Role31-sensitive events: {aggregate['sensitive_event_apks']} ({aggregate['sensitive_event_apk_rate']:.3f}).
- Endpoint events aligned to frozen Hosts: {aggregate['aligned_events']}/{aggregate['sensitive_events']} ({aggregate['endpoint_host_alignment_rate']:.3f}).
- APKs with any condition-bearing sensitive event: {aggregate['condition_bearing_apks']} ({aggregate['condition_bearing_apk_rate']:.3f}).
- APKs with framework-stable directional callback context: {aggregate['directional_callback_apks']} ({aggregate['directional_callback_apk_rate']:.3f}).
- Folds containing directional callback context: {aggregate['directional_callback_folds']} / 5.

## Evidence boundary

`USER_CALLBACK` and `BACKGROUND_CALLBACK` are framework-stable callback-name observations. Branch, permission, consent and scheduling facts are only co-located static context; they are not claimed as control dependence or runtime causality. Exact APIs are kept only in the audit trace and mapped to Role31 before any future probe.

Only `GO_FULL_200_GUARD_AUDIT` authorizes a full routed coverage audit. It does not authorize model integration.
"""


def main() -> None:
    ensure_output_boundary()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    STATUS.mkdir(parents=True, exist_ok=True)
    assets = discover_assets()
    source = svm_route_rows(assets["source_member"])
    routed = [row for row in source if row["routed"]]
    wanted = {row["sha256"] for row in routed}
    apk_map, apk_audit = load_apk_map(wanted)
    endpoints = load_endpoint_rows(assets["endpoint"], wanted)
    if len(apk_map) != 200 or len(endpoints) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: routed assets apk={len(apk_map)} endpoint={len(endpoints)}")

    selection_path = OUTPUT / "pilot_selection.csv"
    if selection_path.is_file():
        selected = [{**row, "fold": int(row["fold"]), "host_count": int(row["host_count"]),
                     "apk_size_bytes": int(row["apk_size_bytes"]), "quantile_rank": int(row["quantile_rank"])}
                    for row in read_csv(selection_path)]
    else:
        selected = select_pilot(routed, endpoints, apk_map)
        write_csv(selection_path, selected)
    if len(selected) != 30 or {int(row["fold"]) for row in selected} != {1, 2, 3, 4, 5}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: pilot selection mismatch")

    protected = [(assets[key], key) for key in ("source_member", "endpoint", "mapping")]
    protected += [(apk_map[row["sha256"]], "selected source APK " + row["sha256"]) for row in selected]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.is_file() else hash_manifest(protected, "before")
    if not before_path.is_file():
        dump_json(before_path, before)

    protocol = {
        "schema_version": "1.0.0", "labels_used": False, "future_rows_loaded": 0,
        "stage1": "P31-SVM", "route": "SVM-anchored four-member MeanOpposition Top25% per fold",
        "pilot_selection": "six Host-count strata per fold; APK size and SHA deterministic tie-break",
        "observations": ["framework user callback", "framework background callback", "local branch within 12 instructions",
                         "co-located permission check", "co-located consent UI", "co-located background scheduling"],
        "semantic_boundary": "co-location is not dominance or runtime causality; exact API is audit-only",
        "go_full_200_criteria": {
            "scan_success_rate>=0.95": .95, "sensitive_event_apk_rate>=0.80": .80,
            "endpoint_host_alignment_rate>=0.80": .80, "condition_bearing_apk_rate>=0.50": .50,
            "directional_callback_apk_rate>=0.15": .15, "directional_callback_folds>=4": 4,
            "formal_integrity=PASS": True,
        },
        "mapping_sha256": sha256_file(assets["mapping"]),
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "environment_inventory.json", {"python": sys.executable, "project": str(PROJECT),
              "output": str(OUTPUT), "assets": {key: str(value) for key, value in assets.items()},
              "apk_path_audit": apk_audit, "network_required": False, "gpu_required": False})

    statuses = []
    for index, row in enumerate(selected, 1):
        status = scan_one(row, apk_map[row["sha256"]], assets["mapping"])
        statuses.append(status)
        print(f"SGCA_PILOT_SCAN {index}/30 status={status.get('status')} sensitive={status.get('sensitive_event_count', 0)} condition={status.get('condition_bearing_event_count', 0)}", flush=True)

    apk_rows = []
    event_rows = []
    context_counts = Counter()
    success = sensitive_event_apks = condition_bearing_apks = directional_callback_apks = 0
    sensitive_events = aligned_events = condition_events = directional_events = 0
    directional_folds = set()
    for row, status in zip(selected, statuses):
        sid = row["sha256"]
        record = read_gzip_json(CACHE / f"{sid}.json.gz") if status.get("status") == "OK" else None
        frozen_hosts = {item["node_id"]: item for item in endpoint_hosts(endpoints[sid])}
        events = [] if record is None else record["events"]
        if record is not None:
            success += 1
        if events:
            sensitive_event_apks += 1
        if any(event["condition_bearing"] for event in events):
            condition_bearing_apks += 1
        if any(event["directional_callback"] for event in events):
            directional_callback_apks += 1
            directional_folds.add(int(row["fold"]))
        local_aligned = 0
        for event in events:
            aligned = int(event["host_node_id"] in frozen_hosts)
            local_aligned += aligned
            context_counts[event["context"]] += 1
            event_rows.append({"sha256": sid, "fold": int(row["fold"]), **event,
                               "frozen_host_aligned": aligned,
                               "frozen_host_roles": "|".join(map(str, frozen_hosts.get(event["host_node_id"], {}).get("roles", [])))})
        sensitive_events += len(events)
        aligned_events += local_aligned
        condition_events += sum(bool(event["condition_bearing"]) for event in events)
        directional_events += sum(bool(event["directional_callback"]) for event in events)
        apk_rows.append({"sha256": sid, "fold": int(row["fold"]), "status": status.get("status"),
                         "host_count": int(row["host_count"]), "method_count": 0 if record is None else record["method_count"],
                         "sensitive_event_count": len(events), "aligned_event_count": local_aligned,
                         "condition_bearing_event_count": sum(bool(event["condition_bearing"]) for event in events),
                         "directional_callback_event_count": sum(bool(event["directional_callback"]) for event in events),
                         "wall_seconds": status.get("wall_seconds", 0), "error": status.get("error", "")})
    write_csv(OUTPUT / "pilot_apk_summary.csv", apk_rows)
    write_csv(OUTPUT / "pilot_sensitive_events.csv", event_rows)
    aggregate = {
        "selected_apks": 30, "successful_scans": success, "scan_success_rate": success / 30,
        "sensitive_event_apks": sensitive_event_apks, "sensitive_event_apk_rate": sensitive_event_apks / 30,
        "sensitive_events": sensitive_events, "aligned_events": aligned_events,
        "endpoint_host_alignment_rate": aligned_events / max(sensitive_events, 1),
        "condition_bearing_apks": condition_bearing_apks, "condition_bearing_apk_rate": condition_bearing_apks / 30,
        "condition_bearing_events": condition_events,
        "directional_callback_apks": directional_callback_apks,
        "directional_callback_apk_rate": directional_callback_apks / 30,
        "directional_callback_events": directional_events,
        "directional_callback_folds": len(directional_folds), "directional_callback_fold_ids": sorted(directional_folds),
        "context_counts": dict(context_counts), "labels_used": False, "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "pilot_coverage.json", aggregate)

    after = hash_manifest(protected, "after")
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    before_by = {item["absolute_path"]: item for item in before}
    after_by = {item["absolute_path"]: item for item in after}
    changed = [path for path in sorted(before_by) if path not in after_by or before_by[path]["sha256"] != after_by[path]["sha256"]]
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(before), "changed": changed}
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)

    criteria = {
        "scan_success_rate>=0.95": aggregate["scan_success_rate"] >= .95,
        "sensitive_event_apk_rate>=0.80": aggregate["sensitive_event_apk_rate"] >= .80,
        "endpoint_host_alignment_rate>=0.80": aggregate["endpoint_host_alignment_rate"] >= .80,
        "condition_bearing_apk_rate>=0.50": aggregate["condition_bearing_apk_rate"] >= .50,
        "directional_callback_apk_rate>=0.15": aggregate["directional_callback_apk_rate"] >= .15,
        "directional_callback_folds>=4": aggregate["directional_callback_folds"] >= 4,
        "formal_integrity=PASS": integrity["status"] == "PASS",
    }
    status = "GO_FULL_200_GUARD_AUDIT" if all(criteria.values()) else "PILOT_COVERAGE_GATE_FAILED"
    decision = {"status": status, "stopped_at": "LABEL_FREE_GUARD_CONTEXT_PILOT", "criteria": criteria,
                "next_action": "run the same frozen observation over all 200 routed source APKs; still no model integration" if status.startswith("GO_") else "stop; control-flow condition observation lacks sufficient operational coverage",
                "model_training_authorized": False, "future_rows_loaded": 0, "labels_used": False,
                "formal_integrity": integrity["status"]}
    dump_json(OUTPUT / "feasibility_decision.json", decision)
    dump_json(OUTPUT / "final_decision.json", decision)
    rendered = report_text(decision, aggregate)
    (OUTPUT / "REPORT.md").write_text(rendered, encoding="utf-8")
    (OUTPUT / "report.md").write_text(rendered, encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        f"# Reproduce commands\n\n```bash\ncd {PROJECT}\nSGCA_PROJECT={PROJECT} SGCA_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'selftest.py'}\nSGCA_PROJECT={PROJECT} SGCA_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'preflight.py'}\nSGCA_PROJECT={PROJECT} SGCA_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'audit.py'}\n```\n",
        encoding="utf-8")
    print(f"SENSITIVE_GUARD_CONTEXT_FINAL_STATUS={status}", flush=True)
    print(f"report={OUTPUT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
