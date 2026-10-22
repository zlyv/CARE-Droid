#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    OUTPUT, PROJECT, discover_assets, dump_json, endpoint_hosts, ensure_output_boundary,
    file_record, formal_rows, load_apk_map, load_attention, load_endpoint_rows,
    read_gzip_json, sha256_file, write_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = OUTPUT / "cache/wrapper_scan"
STATUS = OUTPUT / "checkpoints/scan_status"
PILOT_SIZE_PER_FOLD = 6
TIMEOUT_SECONDS = int(os.environ.get("TW_APK_TIMEOUT_SECONDS", "900"))


def phase(name: str, status: str) -> None:
    print(f"TW_PHASE_{status} name={name}", flush=True)


def select_pilot(routed: list[dict[str, Any]], endpoints: dict[str, dict[str, Any]], apk_map: dict[str, Path]) -> list[dict[str, Any]]:
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in routed:
        sid = row["sha256"]
        by_fold[row["fold"]].append({
            "sha256": sid, "fold": row["fold"],
            "host_count": len(endpoint_hosts(endpoints[sid])),
            "apk_size_bytes": apk_map[sid].stat().st_size,
        })
    selected = []
    for fold in range(1, 6):
        rows = sorted(by_fold[fold], key=lambda row: (row["host_count"], row["apk_size_bytes"], row["sha256"]))
        indexes = []
        for rank in range(PILOT_SIZE_PER_FOLD):
            index = round(rank * (len(rows) - 1) / (PILOT_SIZE_PER_FOLD - 1))
            if index not in indexes:
                indexes.append(index)
        if len(indexes) != PILOT_SIZE_PER_FOLD:
            raise RuntimeError("pilot quantile selection did not produce six rows")
        for rank, index in enumerate(indexes):
            selected.append({
                **rows[index], "quantile_rank": rank,
                "selection_reason": "label-free fold-stratified Host-count sextile; APK size and SHA deterministic tie-break",
            })
    return selected


def hash_manifest(paths: list[tuple[Path, str]], phase_name: str) -> list[dict[str, Any]]:
    rows = []
    for index, (path, purpose) in enumerate(paths, 1):
        rows.append(file_record(path, purpose))
        if index % 10 == 0 or index == len(paths):
            print(f"TW_ASSET_HASH phase={phase_name} {index}/{len(paths)}", flush=True)
    return rows


def scan_one(row: dict[str, Any], apk: Path, mapping: Path) -> dict[str, Any]:
    sid = row["sha256"]
    output = CACHE / f"{sid}.json.gz"
    status_path = STATUS / f"{sid}.json"
    if output.is_file() and status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "OK":
            return existing
    command = [
        sys.executable, str(SCRIPT_DIR / "scan_one.py"), "--apk", str(apk), "--sha256", sid,
        "--mapping", str(mapping), "--output", str(output), "--status", str(status_path), "--max-chain", "2",
    ]
    try:
        subprocess.run(command, check=False, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        dump_json(status_path, {"sha256": sid, "status": "TIMEOUT", "timeout_seconds": TIMEOUT_SECONDS})
    if not status_path.is_file():
        return {"sha256": sid, "status": "ERROR", "error": "worker produced no status"}
    return json.loads(status_path.read_text(encoding="utf-8"))


def report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    return f"""# CARE-Droid DEX transparent-wrapper pilot audit v1

## Decision

**{decision['status']}**

This is a source-2017, label-free, routed-APK feasibility audit. It does not train or modify CARE-Droid, rewrite APKs, load future cohorts, or claim whole-program semantic equivalence.

## Pilot

- Selected APKs: {aggregate['selected_apks']} (six Host-count strata per fold).
- Successful DEX scans: {aggregate['successful_scans']}.
- APKs with strict verified wrappers: {aggregate['strict_affected_apks']}.
- APKs whose semantic Host candidate changes: {aggregate['host_changing_apks']}.
- Strict verified wrappers: {aggregate['verified_wrapper_count']}.
- Declared-target-only candidates (excluded from strict evidence): {aggregate['declared_target_only_count']}.
- Endpoint-Host aligned strict wrappers: {aggregate['host_aligned_wrapper_count']}.
- Critical Hosts affected: {aggregate['critical_host_wrapper_count']}.
- Routed attention mass on affected Hosts: {aggregate['affected_attention_mass']:.6f}.
- Affected folds: {aggregate['affected_folds']} / 5.

## Boundary

`VERIFIED_STRICT` means a restricted DEX-body forwarding contract: one exact-dispatch invoke, exact parameter-word forwarding, direct return, and no other opcode beyond register moves/nop. Branches, exceptions, writes, allocation, reflection, JNI, synchronization, async behavior and dynamic dispatch are rejected or placed outside strict evidence.

The next stage is authorized only by `GO_FULL_200_AUDIT`. Even then it authorizes coverage expansion, not model training.
"""


def main() -> None:
    ensure_output_boundary()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    STATUS.mkdir(parents=True, exist_ok=True)
    assets = discover_assets()
    formal = formal_rows(assets["formal"])
    routed = [row for row in formal if row["routed"]]
    wanted = {row["sha256"] for row in routed}
    apk_map, apk_audit = load_apk_map(wanted)
    endpoints = load_endpoint_rows(assets["endpoint"], wanted)
    attention = load_attention(assets["attention"])

    selection_path = OUTPUT / "pilot_selection.csv"
    if selection_path.is_file():
        from common import read_csv
        selected = [{**row, "fold": int(row["fold"]), "host_count": int(row["host_count"]), "apk_size_bytes": int(row["apk_size_bytes"]), "quantile_rank": int(row["quantile_rank"])} for row in read_csv(selection_path)]
    else:
        selected = select_pilot(routed, endpoints, apk_map)
        write_csv(selection_path, selected, ["sha256", "fold", "quantile_rank", "host_count", "apk_size_bytes", "selection_reason"])
    if len(selected) != 30 or {row["fold"] for row in selected} != {1, 2, 3, 4, 5}:
        raise RuntimeError("pilot selection contract mismatch")

    protected = [(assets[key], key) for key in ("formal", "endpoint", "attention", "mapping")]
    protected += [(apk_map[row["sha256"]], "selected source APK " + row["sha256"]) for row in selected]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    if before_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))
    else:
        before = hash_manifest(protected, "before")
        dump_json(before_path, before)

    protocol = {
        "schema_version": "1.0.0", "labels_used": False, "future_rows_loaded": 0,
        "pilot_size": 30, "per_fold": 6, "max_wrapper_chain": 2,
        "strict_dispatch": ["invoke-static", "invoke-direct", "invoke-super"],
        "dynamic_dispatch_policy": "record as DECLARED_TARGET_ONLY; never include in strict gate",
        "allowed_noninvoke_opcodes": ["nop", "move*", "move-result*", "return*"],
        "selection": "fold-stratified Host-count sextiles without labels",
        "go_full_200_criteria": {
            "scan_success_rate>=0.95": 0.95,
            "strict_affected_apk_rate>=0.15": 0.15,
            "host_changing_apk_rate>=0.10": 0.10,
            "affected_folds>=4": 4,
            "affected_attention_mass>=0.02": 0.02,
            "formal_integrity=PASS": True,
        },
        "mapping_sha256": sha256_file(assets["mapping"]),
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "environment_inventory.json", {
        "python": sys.executable, "project": str(PROJECT), "output": str(OUTPUT),
        "assets": {key: str(value) for key, value in assets.items()},
        "apk_path_audit": apk_audit, "network_required": False, "gpu_required": False,
    })

    phase("pilot_scan", "START")
    statuses = []
    for index, row in enumerate(selected, 1):
        status = scan_one(row, apk_map[row["sha256"]], assets["mapping"])
        statuses.append(status)
        print(f"TW_PILOT_SCAN {index}/30 status={status.get('status')} verified={status.get('verified_wrapper_count', 0)}", flush=True)
    phase("pilot_scan", "DONE")

    summary_rows = []
    wrapper_rows = []
    rejection = Counter()
    verified_total = declared_total = aligned_total = critical_total = host_changing_total = 0
    affected_attention_mass = 0.0
    total_attention_mass = 0.0
    affected_folds = set()
    success = 0
    for row, status in zip(selected, statuses):
        sid = row["sha256"]
        record = None
        if status.get("status") == "OK":
            success += 1
            record = read_gzip_json(CACHE / f"{sid}.json.gz")
        hosts = endpoint_hosts(endpoints[sid])
        host_index = {item["node_id"]: index for index, item in enumerate(hosts)}
        att = attention[sid]["attention"]
        total_attention_mass += float(sum(att))
        wrapper_indices = set()
        verified = [] if record is None else record["verified_wrappers"]
        for wrapper in verified:
            if wrapper["node_id"] in host_index:
                wrapper_indices.add(host_index[wrapper["node_id"]])
            wrapper_rows.append({
                "sha256": sid, "fold": row["fold"], "signature": wrapper["signature"],
                "node_id": wrapper["node_id"], "depth": wrapper["depth"],
                "role": wrapper["role"], "rule_id": wrapper["rule_id"],
                "sensitive_endpoint": wrapper["sensitive_endpoint"],
                "invoke_opcode": wrapper["invoke_opcode"],
                "wrapper_chain": " | ".join(wrapper["wrapper_chain"]),
                "direct_callers": " | ".join(wrapper["direct_callers"]),
                "effective_host_candidates": " | ".join(wrapper["effective_host_candidates"]),
                "semantic_view_changes_host": int(wrapper["semantic_view_changes_host"]),
                "endpoint_host_aligned": int(wrapper["node_id"] in host_index),
            })
        affected_mass = sum(float(att[index]) for index in wrapper_indices if index < len(att))
        critical = int(attention[sid]["critical"])
        critical_hit = int(critical in wrapper_indices)
        strict_count = len(verified)
        host_changing = sum(bool(item["semantic_view_changes_host"]) for item in verified)
        declared = 0 if record is None else int(record["declared_target_only_count"])
        if strict_count:
            affected_folds.add(int(row["fold"]))
        verified_total += strict_count
        declared_total += declared
        aligned_total += len(wrapper_indices)
        critical_total += critical_hit
        host_changing_total += host_changing
        affected_attention_mass += affected_mass
        if record:
            rejection.update(record.get("rejection_reason_counts", {}))
        summary_rows.append({
            "sha256": sid, "fold": row["fold"], "status": status.get("status"),
            "method_count": 0 if record is None else record["method_count"],
            "verified_wrapper_count": strict_count,
            "host_changing_wrapper_count": host_changing,
            "declared_target_only_count": declared,
            "host_aligned_wrapper_count": len(wrapper_indices),
            "critical_host_affected": critical_hit,
            "affected_attention_mass": affected_mass,
            "wall_seconds": status.get("wall_seconds", 0),
            "error": status.get("error", ""),
        })
    write_csv(OUTPUT / "pilot_apk_summary.csv", summary_rows, list(summary_rows[0]))
    wrapper_fields = [
        "sha256", "fold", "signature", "node_id", "depth", "role", "rule_id",
        "sensitive_endpoint", "invoke_opcode", "wrapper_chain", "direct_callers",
        "effective_host_candidates", "semantic_view_changes_host", "endpoint_host_aligned",
    ]
    write_csv(OUTPUT / "verified_wrappers.csv", wrapper_rows, wrapper_fields)
    dump_json(OUTPUT / "rejection_reason_counts.json", dict(rejection.most_common()))

    aggregate = {
        "selected_apks": 30, "successful_scans": success, "scan_success_rate": success / 30,
        "strict_affected_apks": sum(row["verified_wrapper_count"] > 0 for row in summary_rows),
        "strict_affected_apk_rate": sum(row["verified_wrapper_count"] > 0 for row in summary_rows) / 30,
        "host_changing_apks": sum(row["host_changing_wrapper_count"] > 0 for row in summary_rows),
        "host_changing_apk_rate": sum(row["host_changing_wrapper_count"] > 0 for row in summary_rows) / 30,
        "verified_wrapper_count": verified_total, "declared_target_only_count": declared_total,
        "host_aligned_wrapper_count": aligned_total, "critical_host_wrapper_count": critical_total,
        "affected_attention_mass": affected_attention_mass / max(total_attention_mass, 1e-12),
        "affected_folds": len(affected_folds), "affected_fold_ids": sorted(affected_folds),
        "future_rows_loaded": 0, "labels_used": False,
    }
    dump_json(OUTPUT / "pilot_coverage.json", aggregate)

    after = hash_manifest(protected, "after")
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    before_by = {row["absolute_path"]: row for row in before}
    after_by = {row["absolute_path"]: row for row in after}
    changed = [path for path in sorted(before_by) if path not in after_by or before_by[path]["sha256"] != after_by[path]["sha256"]]
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(before), "changed": changed}
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)

    criteria = {
        "scan_success_rate>=0.95": aggregate["scan_success_rate"] >= 0.95,
        "strict_affected_apk_rate>=0.15": aggregate["strict_affected_apk_rate"] >= 0.15,
        "host_changing_apk_rate>=0.10": aggregate["host_changing_apk_rate"] >= 0.10,
        "affected_folds>=4": aggregate["affected_folds"] >= 4,
        "affected_attention_mass>=0.02": aggregate["affected_attention_mass"] >= 0.02,
        "formal_integrity=PASS": integrity["status"] == "PASS",
    }
    status = "GO_FULL_200_AUDIT" if all(criteria.values()) else "PILOT_COVERAGE_GATE_FAILED"
    decision = {
        "status": status, "stopped_at": "DEX_WRAPPER_PILOT", "criteria": criteria,
        "next_action": "expand the same frozen label-free audit to all 200 routed APKs" if status == "GO_FULL_200_AUDIT" else "stop; do not train or integrate wrapper normalization",
        "model_training_authorized": False, "apk_rewrite_performed": False,
        "future_rows_loaded": 0, "labels_used": False,
        "formal_integrity": integrity["status"],
    }
    dump_json(OUTPUT / "feasibility_decision.json", decision)
    dump_json(OUTPUT / "final_decision.json", decision)
    rendered_report = report_text(decision, aggregate)
    (OUTPUT / "REPORT.md").write_text(rendered_report, encoding="utf-8")
    (OUTPUT / "report.md").write_text(rendered_report, encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduce commands\n\n"
        f"```bash\ncd {PROJECT}\n"
        f"TW_PROJECT={PROJECT} TW_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'selftest.py'}\n"
        f"TW_PROJECT={PROJECT} TW_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'preflight.py'}\n"
        f"TW_PROJECT={PROJECT} TW_OUTPUT={OUTPUT} {sys.executable} {SCRIPT_DIR / 'audit.py'}\n```\n",
        encoding="utf-8",
    )
    print(f"DEX_TRANSPARENT_WRAPPER_FINAL_STATUS={status}", flush=True)
    print(f"report={OUTPUT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
