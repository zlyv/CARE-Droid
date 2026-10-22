#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (OUTPUT, PROJECT, discover_assets, dump_json, ensure_output_boundary, exact_mcnemar_p,
                    load_apk_paths, read_csv, record, write_csv)
from select_pairs import select

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = OUTPUT / "cache/apk_audit"
STATUS = OUTPUT / "checkpoints/scan_status"
TIMEOUT_SECONDS = int(os.environ.get("HCOA_APK_TIMEOUT_SECONDS", "1200"))
BINARY_FEATURES = (
    "has_native", "has_reflection", "has_dynamic_loading", "has_embedded_payload", "known_packer",
    "has_multidex", "high_sdk_dominance", "low_component_mapping", "high_short_identifier", "visibility_gap_any",
)
CONTINUOUS_FEATURES = (
    "native_method_fraction", "native_library_count", "reflection_call_rate", "dynamic_loading_call_rate",
    "embedded_payload_fraction", "known_sdk_class_fraction", "component_mapping_rate", "short_identifier_fraction",
    "dex_count", "apk_size_bytes",
)


def hash_manifest(paths: list[tuple[Path, str]], phase: str) -> list[dict[str, Any]]:
    rows = []
    for index, (path, purpose) in enumerate(paths, 1):
        rows.append(record(path, purpose))
        if index % 10 == 0 or index == len(paths):
            print(f"HCOA_ASSET_HASH phase={phase} {index}/{len(paths)}", flush=True)
    return rows


def scan_one(sid: str, apk: Path) -> dict[str, Any]:
    output = CACHE / f"{sid}.json"; status_path = STATUS / f"{sid}.json"
    if output.is_file() and status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "OK": return existing
    command = [sys.executable, str(SCRIPT_DIR / "scan_one.py"), "--apk", str(apk), "--sha256", sid,
               "--output", str(output), "--status", str(status_path)]
    try: subprocess.run(command, check=False, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        dump_json(status_path, {"sha256": sid, "status": "TIMEOUT", "timeout_seconds": TIMEOUT_SECONDS})
    if not status_path.is_file(): return {"sha256": sid, "status": "ERROR", "error": "worker produced no status"}
    return json.loads(status_path.read_text(encoding="utf-8"))


def feature_rows(selection, features):
    rows = []
    for pair in selection:
        for kind, key in (("ERROR", "error_sha256"), ("CONTROL", "control_sha256")):
            sid = pair[key]; row = features.get(sid)
            base = {"pair_id": pair["pair_id"], "year": pair["year"], "error_type": pair["error_type"],
                    "kind": kind, "sha256": sid}
            rows.append({**base, **({} if row is None else row)})
    return rows


def binary_stats(selection, features):
    results = []
    for scope, predicate in (("ALL", lambda p: True), ("FN", lambda p: p["error_type"] == "FN"),
                             ("FP", lambda p: p["error_type"] == "FP")):
        local = [pair for pair in selection if predicate(pair)
                 and pair["error_sha256"] in features and pair["control_sha256"] in features]
        for feature in BINARY_FEATURES:
            both1 = error1 = control1 = both0 = 0
            per_year = defaultdict(list)
            for pair in local:
                e = int(features[pair["error_sha256"]][feature]); c = int(features[pair["control_sha256"]][feature])
                if e and c: both1 += 1
                elif e and not c: error1 += 1
                elif not e and c: control1 += 1
                else: both0 += 1
                per_year[int(pair["year"])].append(e - c)
            year_directions = {str(year): (sum(values) / len(values) if values else 0.0) for year, values in sorted(per_year.items())}
            diff = (error1 - control1) / max(len(local), 1)
            results.append({"scope": scope, "feature": feature, "pairs": len(local),
                            "error_prevalence": (both1 + error1) / max(len(local), 1),
                            "control_prevalence": (both1 + control1) / max(len(local), 1),
                            "paired_difference": diff, "error_only": error1, "control_only": control1,
                            "both_present": both1, "both_absent": both0, "discordant_pairs": error1 + control1,
                            "mcnemar_exact_p": exact_mcnemar_p(error1, control1),
                            "years_positive": sum(value > 0 for value in year_directions.values()),
                            "years_negative": sum(value < 0 for value in year_directions.values()),
                            "year_differences": json.dumps(year_directions, sort_keys=True)})
    return results


def continuous_stats(selection, features):
    results = []
    for scope, predicate in (("ALL", lambda p: True), ("FN", lambda p: p["error_type"] == "FN"),
                             ("FP", lambda p: p["error_type"] == "FP")):
        local = [pair for pair in selection if predicate(pair)
                 and pair["error_sha256"] in features and pair["control_sha256"] in features]
        for feature in CONTINUOUS_FEATURES:
            diffs = [float(features[p["error_sha256"]][feature]) - float(features[p["control_sha256"]][feature]) for p in local]
            results.append({"scope": scope, "feature": feature, "pairs": len(diffs),
                            "mean_paired_difference": sum(diffs) / max(len(diffs), 1),
                            "positive_pairs": sum(value > 0 for value in diffs),
                            "negative_pairs": sum(value < 0 for value in diffs),
                            "tied_pairs": sum(value == 0 for value in diffs)})
    return results


def main() -> None:
    ensure_output_boundary(); OUTPUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True); STATUS.mkdir(parents=True, exist_ok=True)
    assets = discover_assets()
    selection_path = OUTPUT / "matched_selection.csv"
    selection = read_csv(selection_path) if selection_path.is_file() else select()
    for row in selection:
        row["pair_id"] = int(row["pair_id"]); row["year"] = int(row["year"])
    wanted = {row[key] for row in selection for key in ("error_sha256", "control_sha256")}
    apk_map, path_audit = load_apk_paths(wanted)
    missing = sorted(wanted - set(apk_map))
    if missing:
        dump_json(OUTPUT / "apk_path_audit.json", {"wanted": len(wanted), "resolved": len(apk_map), "missing": missing, "sources": path_audit})
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: selected APK paths resolved {len(apk_map)}/{len(wanted)}")
    dump_json(OUTPUT / "apk_path_audit.json", {"wanted": len(wanted), "resolved": len(apk_map), "missing": [], "sources": path_audit})

    protected = [(assets[key], key) for key in ("sample_audit", "source_predictions", "future_predictions")]
    protected += [(apk_map[sid], "selected diagnostic APK " + sid) for sid in sorted(wanted)]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.is_file() else hash_manifest(protected, "before")
    if not before_path.is_file(): dump_json(before_path, before)
    dump_json(OUTPUT / "protocol.json", {
        "schema_version": "1.0.0", "task": "matched high-confidence error observability audit",
        "labels_used_for_diagnostic_selection": True, "future_labels_used_for_diagnostic_only": True,
        "model_training": False, "threshold_tuning": False, "formal_selection": False,
        "per_year_per_error_type_limit": 4, "binary_features": BINARY_FEATURES,
        "candidate_rule": "descriptive only: abs paired difference>=0.20, discordant>=10, exact p<0.05, same sign in >=3 years",
        "candidate_cannot_authorize_model_integration": True,
    })

    statuses = {}; ordered = sorted(wanted)
    for index, sid in enumerate(ordered, 1):
        status = scan_one(sid, apk_map[sid]); statuses[sid] = status
        print(f"HCOA_SCAN {index}/{len(ordered)} status={status.get('status')} gap={status.get('visibility_gap_any', 0)}", flush=True)
    features = {}
    for sid, status in statuses.items():
        path = CACHE / f"{sid}.json"
        if status.get("status") == "OK" and path.is_file(): features[sid] = json.loads(path.read_text(encoding="utf-8"))
    rows = feature_rows(selection, features); write_csv(OUTPUT / "sample_observability.csv", rows)
    binary = binary_stats(selection, features); continuous = continuous_stats(selection, features)
    write_csv(OUTPUT / "paired_binary_results.csv", binary); write_csv(OUTPUT / "paired_continuous_results.csv", continuous)

    candidates = [row for row in binary if row["scope"] == "ALL" and abs(float(row["paired_difference"])) >= .20
                  and int(row["discordant_pairs"]) >= 10 and float(row["mcnemar_exact_p"]) < .05
                  and max(int(row["years_positive"]), int(row["years_negative"])) >= 3]
    scan_rate = len(features) / max(len(wanted), 1)
    after = hash_manifest(protected, "after"); dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    before_by = {row["absolute_path"]: row for row in before}; after_by = {row["absolute_path"]: row for row in after}
    changed = [path for path in sorted(before_by) if path not in after_by or before_by[path]["sha256"] != after_by[path]["sha256"]]
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(before), "changed": changed}
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS": raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")
    if scan_rate < .95:
        status = "BLOCKED_BY_EXTRACTION_COVERAGE"
    elif candidates:
        status = "OBSERVABILITY_GAP_DESCRIPTIVELY_SUPPORTED"
    else:
        status = "NO_DOMINANT_NEW_STATIC_OBSERVABILITY_GAP"
    decision = {"status": status, "selected_pairs": len(selection), "selected_apks": len(wanted),
                "successful_scans": len(features), "scan_success_rate": scan_rate,
                "candidate_mechanisms": candidates, "future_labels_used_for_diagnostic_only": True,
                "model_integration_authorized": False, "future_used_for_formal_selection": False,
                "formal_integrity": integrity["status"],
                "next_action": "if a mechanism is descriptive, require a separate source-only or untouched-external confirmation; otherwise stop incremental C43 feature engineering"}
    dump_json(OUTPUT / "final_decision.json", decision)
    lines = ["# CARE-Droid high-confidence error observability audit v1", "", f"Final status: **{status}**.", "",
             f"Matched pairs: {len(selection)}; unique APKs: {len(wanted)}; scan success: {scan_rate:.3f}.", "",
             "This audit uses labels, including future labels, only to construct diagnostic error-control pairs. It does not train, tune, route, or authorize a model change.", "",
             "## Candidate mechanisms", ""]
    if candidates:
        lines += ["| Feature | Error prevalence | Control prevalence | Difference | p |", "|---|---:|---:|---:|---:|"]
        for row in candidates:
            lines.append(f"| {row['feature']} | {float(row['error_prevalence']):.3f} | {float(row['control_prevalence']):.3f} | {float(row['paired_difference']):+.3f} | {float(row['mcnemar_exact_p']):.4g} |")
    else:
        lines.append("No pre-registered dominant observability mechanism passed the descriptive gate.")
    lines += ["", "Any descriptive candidate still requires source-only or untouched-external confirmation. Future diagnostics cannot select a CARE-Droid feature.", "", "Formal asset integrity: PASS.", ""]
    rendered = "\n".join(lines); (OUTPUT / "REPORT.md").write_text(rendered, encoding="utf-8"); (OUTPUT / "report.md").write_text(rendered, encoding="utf-8")
    print("HIGH_CONFIDENCE_OBSERVABILITY_FINAL_STATUS=" + status, flush=True)
    print("report=" + str(OUTPUT / "REPORT.md"), flush=True)


if __name__ == "__main__": main()
