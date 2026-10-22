#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    OUTPUT, PROJECT, discover_assets, dump_json, endpoint_hosts, file_record,
    formal_rows, load_apk_map, load_attention, read_endpoint_rows,
    read_gzip_json, sha256_file, write_csv,
)
from fullgraph import adjacency, classify_host, identify_roots, multi_source_distances, shortest_path
from taxonomy import RULES

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_CACHE = OUTPUT / "full_graph_cache"
STATUS_CACHE = OUTPUT / "checkpoints/build_status"
OLD = {
    "root_alignment": 0.03775549066903643,
    "directional_host": 0.030750591357526107,
    "critical_directional": 0.05555555555555555,
    "attention_directional": 0.04465706830693179,
    "unreachable": 0.9459,
    "both": 4,
}
PILOT_TIMEOUT_SECONDS = int(os.environ.get("FULLGRAPH_PILOT_TIMEOUT_SECONDS", "1200"))
FULL_TIMEOUT_SECONDS = int(os.environ.get("FULLGRAPH_APK_TIMEOUT_SECONDS", "1800"))


def phase(name: str, status: str, **extra: Any) -> None:
    record = {"phase": name, "status": status, "epoch": int(time.time()), **extra}
    phase_path = OUTPUT / "checkpoints/phases.jsonl"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"FULLGRAPH_PHASE_{status} name={name}", flush=True)


def placeholders(gate: str, status: str) -> None:
    for name in ("direction_probe.json", "matched_null_results.json", "end_to_end_results.json", "timing.json"):
        path = OUTPUT / name
        if not path.exists():
            dump_json(path, {"status": "NOT_RUN_DUE_TO_PREVIOUS_GATE", "gate": gate, "final_status": status, "future_rows_loaded": 0})
    for name in ("full_graph_schema.json", "full_graph_protocol.json", "trcbe_fullgraph_coverage.json", "trcbe_fullgraph_decision.json"):
        path = OUTPUT / name
        if not path.exists():
            dump_json(path, {"status": "NOT_RUN_DUE_TO_PREVIOUS_GATE", "gate": gate, "final_status": status, "future_rows_loaded": 0})
    csv_path = OUTPUT / "full_graph_build_status.csv"
    if not csv_path.exists():
        write_csv(csv_path, [{"status": "NOT_RUN_DUE_TO_PREVIOUS_GATE", "gate": gate}], ["status", "gate"])


def executable_inventory() -> dict[str, Any]:
    commands = ["python3", "java", "jadx", "apktool", "soot", "flowdroid", "androguard", "aapt", "aapt2"]
    result = {name: shutil.which(name) for name in commands}
    try:
        from androguard.core.apk import APK  # noqa: F401
        from androguard.core.dex import DEX  # noqa: F401
        result["androguard_import"] = True
        result["androguard_version"] = importlib.metadata.version("androguard")
    except Exception as exc:
        result["androguard_import"] = False
        result["androguard_error"] = repr(exc)
    result.update({
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "network_required_by_selected_builder": False,
        "android_jar_required_by_selected_builder": False,
        "new_permission_required_by_selected_builder": False,
        "selected_builder": "Androguard DEX parser already present; direct app-defined invoke references plus app-local CHA",
    })
    return result


def scan_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filename_pattern = re.compile(r"call.?graph|caller|callee|icfg|full.?graph|raw.?graph|method.?graph|graph.?cache|endpoint", re.I)
    content_pattern = re.compile(r"call.?graph|caller|callee|get_instructions|invoke-|edge_index|endpoint_centered", re.I)
    candidates: list[dict[str, Any]] = []
    script_hits: list[dict[str, Any]] = []
    roots = [PROJECT / "artifacts", PROJECT / "outputs", PROJECT / "scripts", PROJECT / "src"]
    skip_parts = {"__pycache__", ".git", "full_graph_cache", "care-fullgraph-trcbe-v1"}
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [name for name in subdirs if name not in skip_parts and not name.startswith("care-fullgraph-trcbe-v")]
            for filename in files:
                path = Path(directory) / filename
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                if filename_pattern.search(filename):
                    stat = path.stat()
                    candidates.append({"path": resolved, "size_bytes": stat.st_size, "suffixes": "".join(path.suffixes), "basis": "filename"})
                if path.suffix.lower() in {".py", ".sh"} and path.stat().st_size <= 2_000_000:
                    try:
                        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        hits = [f"{index}:{line[:240]}" for index, line in enumerate(lines, 1) if content_pattern.search(line)][:12]
                        if hits:
                            script_hits.append({"path": resolved, "hits": hits})
                    except OSError:
                        pass
    candidates.sort(key=lambda row: row["path"])
    script_hits.sort(key=lambda row: row["path"])
    return candidates[:2000], script_hits[:500]


def sample_candidate(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "format": "".join(path.suffixes), "verified_full_graph": False}
    try:
        suffix = "".join(path.suffixes).lower()
        if path.stat().st_size > 200_000_000:
            result["inspection"] = "metadata_only_due_to_size"
            return result
        if suffix.endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                result["keys"] = list(archive.files)
                result["shapes"] = {key: list(archive[key].shape) for key in archive.files[:20]}
        elif suffix.endswith((".json", ".json.gz", ".jsonl", ".jsonl.gz")):
            opener = gzip.open if suffix.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
                text = handle.read(200_000)
            result["prefix"] = text[:500]
            result["contains_caller"] = bool(re.search(r"caller|source|src", text, re.I))
            result["contains_callee"] = bool(re.search(r"callee|target|dst", text, re.I))
            result["contains_signature"] = bool(re.search(r"L[^;]+;->[^\s(]+\(", text))
        elif suffix.endswith((".csv", ".tsv", ".txt", ".dot", ".graphml", ".gml")):
            text = path.read_text(encoding="utf-8", errors="ignore")[:200_000]
            result["prefix"] = text[:500]
            result["contains_caller"] = bool(re.search(r"caller|source|src", text, re.I))
            result["contains_callee"] = bool(re.search(r"callee|target|dst", text, re.I))
            result["contains_signature"] = bool(re.search(r"L[^;]+;->[^\s(]+\(", text))
        else:
            result["inspection"] = "unsupported_or_binary"
    except Exception as exc:
        result["inspection_error"] = repr(exc)
    return result


def taxonomy_contract(schema_path: Path) -> dict[str, Any]:
    frozen = json.loads(schema_path.read_text(encoding="utf-8"))
    frozen_tuples = {
        (str(row["family"]), str(row["required_type"]).casefold(), str(row["name"]).casefold(), re.sub(r"\s+", "", str(row["descriptor"])).casefold(), str(row["category"]), str(row["confidence"]))
        for row in frozen["rules"]
    }
    local_tuples = {(row["family"], row["required_type"], row["name"], row["descriptor"], row["category"], row["confidence"]) for row in RULES}
    if frozen_tuples != local_tuples:
        raise RuntimeError("BLOCKED_BY_CONTRACT: bundled taxonomy differs from frozen TRCBE taxonomy")
    return {"path": str(schema_path.resolve()), "sha256": sha256_file(schema_path), "semantic_rule_match": True, "rule_count": len(local_tuples)}


def hash_manifest(paths: list[tuple[Path, str]], phase_name: str) -> list[dict[str, Any]]:
    records = []
    for index, (path, purpose) in enumerate(paths, 1):
        records.append(file_record(path, purpose))
        if index % 25 == 0 or index == len(paths):
            print(f"FULLGRAPH_ASSET_HASH phase={phase_name} {index}/{len(paths)}", flush=True)
    return records


def build_one(sid: str, apk_path: Path, timeout_seconds: int) -> dict[str, Any]:
    graph_path = GRAPH_CACHE / f"{sid}.json.gz"
    status_path = STATUS_CACHE / f"{sid}.json"
    if graph_path.is_file() and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "OK":
            return status
    command = [sys.executable, str(SCRIPT_DIR / "build_one.py"), "--apk", str(apk_path), "--sha256", sid, "--output", str(graph_path), "--status", str(status_path)]
    try:
        subprocess.run(command, check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        status = {"sha256": sid, "status": "TIMEOUT", "timeout_seconds": timeout_seconds, "wall_seconds": timeout_seconds}
        dump_json(status_path, status)
        return status
    if not status_path.is_file():
        return {"sha256": sid, "status": "ERROR", "error": "worker produced no status"}
    return json.loads(status_path.read_text(encoding="utf-8"))


def select_pilot(routed: list[dict[str, Any]], endpoints: dict[str, dict[str, Any]], apk_map: dict[str, Path]) -> list[dict[str, Any]]:
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for row in routed:
        sid = row["sha256"]
        candidate = {
            "sha256": sid,
            "fold": row["fold"],
            "apk_size_bytes": apk_map[sid].stat().st_size,
            "host_count": len(endpoint_hosts(endpoints[sid])),
        }
        by_fold.setdefault(row["fold"], []).append(candidate)
    selected = []
    for fold in sorted(by_fold):
        rows = sorted(by_fold[fold], key=lambda item: (item["host_count"], item["apk_size_bytes"], item["sha256"]))
        indexes = sorted({0, len(rows) // 2, len(rows) - 1})
        for index in indexes:
            row = dict(rows[index])
            row["selection_reason"] = ("fold-stratified min/median/max Host-count; APK size and SHA are deterministic tie-breaks")
            selected.append(row)
    return selected


def analyze_graph(sid: str, fold: int, graph: dict[str, Any], payload: dict[str, Any], attention_info: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    methods = graph["methods"]
    index_by_id = {int(method["node_id"]): index for index, method in enumerate(methods)}
    hosts = endpoint_hosts(payload)
    roots = identify_roots(graph)
    adj = adjacency(graph)
    sources = {"USER": set(), "AUTO": set(), "NEUTRAL": set()}
    for root in roots:
        category = root["category"]
        if category.startswith("USER_"):
            sources["USER"].add(root["node_index"])
        elif category.startswith("AUTONOMOUS_"):
            sources["AUTO"].add(root["node_index"])
        else:
            sources["NEUTRAL"].add(root["node_index"])
    user_dist = multi_source_distances(adj, sources["USER"])
    auto_dist = multi_source_distances(adj, sources["AUTO"])
    neutral_dist = multi_source_distances(adj, sources["NEUTRAL"])
    attention_valid = bool(attention_info is not None and len(attention_info["attention"]) == len(hosts))
    critical_index = int(attention_info["critical"]) if attention_valid and hosts else -1
    host_rows = []
    categories = Counter()
    attention_mass = 0.0
    exact_aligned = 0
    critical_directional = 0
    path_examples = []
    root_indexes = sources["USER"] | sources["AUTO"]
    for host_order, host in enumerate(hosts):
        full_index = index_by_id.get(int(host["node_id"]))
        if full_index is None:
            category, user_value, auto_value, neutral_value = "UNALIGNED", None, None, None
        else:
            exact_aligned += 1
            category, user_value, auto_value, neutral_value = classify_host(full_index, user_dist, auto_dist, neutral_dist)
        categories[category] += 1
        weight = float(attention_info["attention"][host_order]) if attention_valid else math.nan
        directional = category in {"USER_ONLY", "AUTONOMOUS_ONLY"}
        if directional and attention_valid:
            attention_mass += weight
        if host_order == critical_index and directional:
            critical_directional = 1
        if full_index is not None and len(path_examples) < 5:
            distances = [value for value in (user_value, auto_value) if value is not None]
            if distances and min(distances) >= 3:
                path_nodes = shortest_path(adj, root_indexes, full_index)
                if len(path_nodes) >= 4:
                    path_examples.append({"sha256": sid, "fold": fold, "edge_length": len(path_nodes) - 1, "signatures": [methods[node]["signature"] for node in path_nodes]})
        host_rows.append({
            "sha256": sid, "fold": fold, "host_order": host_order, "host_node_id": host["node_id"],
            "full_node_index": full_index, "trigger_type": category, "user_distance": user_value,
            "autonomous_distance": auto_value, "neutral_distance": neutral_value, "attention": weight,
            "critical": int(host_order == critical_index), "roles": "|".join(map(str, host["roles"])),
        })
    root_outgoing = sum(bool(adj.get(root["node_index"])) for root in roots)
    method_ids = {int(method["node_id"]) for method in methods}
    host_ids = {int(host["node_id"]) for host in hosts}
    root_ids = {int(root["node_id"]) for root in roots}
    ordinary_count = sum(int(method["node_id"]) not in host_ids and int(method["node_id"]) not in root_ids for method in methods)
    metric = {
        "sha256": sid, "fold": fold, "status": "OK", "method_count": len(methods), "edge_count": len(graph["edges"]),
        "ordinary_method_fraction": ordinary_count / max(1, len(methods)), "host_count": len(hosts),
        "host_exact_aligned_count": exact_aligned,
        "host_exact_alignment": (exact_aligned / len(hosts)) if hosts else None, "root_count": len(roots),
        "root_graph_alignment": sum(int(root["node_id"]) in method_ids for root in roots) / max(1, len(roots)),
        "root_outgoing_edge_rate": root_outgoing / max(1, len(roots)),
        "any_root_host_assignment": sum(row["trigger_type"] not in {"UNREACHABLE", "UNALIGNED"} for row in host_rows) / max(1, len(hosts)),
        "directional_host_assignment": sum(row["trigger_type"] in {"USER_ONLY", "AUTONOMOUS_ONLY"} for row in host_rows) / max(1, len(hosts)),
        "critical_directional_assignment": critical_directional,
        "attention_directional_mass": attention_mass if attention_valid else math.nan,
        "attention_alignment_valid": int(attention_valid), "multi_hop_path_count": len(path_examples),
        "USER_ONLY": categories["USER_ONLY"], "AUTONOMOUS_ONLY": categories["AUTONOMOUS_ONLY"],
        "BOTH": categories["BOTH"], "NEUTRAL_ONLY": categories["NEUTRAL_ONLY"],
        "UNREACHABLE": categories["UNREACHABLE"], "UNALIGNED": categories["UNALIGNED"],
        "invoke_parse_rate": graph["stats"].get("invoke_parse_rate", 0.0),
        "isolated_method_fraction": graph["stats"].get("isolated_method_count", 0) / max(1, len(methods)),
    }
    return metric, host_rows, path_examples


def aggregate(metrics: list[dict[str, Any]], host_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in metrics if row.get("status") == "OK"]
    host_total = len(host_rows)
    directional = sum(row["trigger_type"] in {"USER_ONLY", "AUTONOMOUS_ONLY"} for row in host_rows)
    any_root = sum(row["trigger_type"] not in {"UNREACHABLE", "UNALIGNED"} for row in host_rows)
    critical = [row for row in host_rows if int(row["critical"]) == 1]
    critical_directional = sum(row["trigger_type"] in {"USER_ONLY", "AUTONOMOUS_ONLY"} for row in critical)
    attention_values = [row["attention_directional_mass"] for row in valid if row["attention_alignment_valid"]]
    return {
        "apk_count": len(metrics), "successful_apks": len(valid), "success_rate": len(valid) / max(1, len(metrics)),
        "method_count": sum(row.get("method_count", 0) for row in valid), "edge_count": sum(row.get("edge_count", 0) for row in valid),
        "host_count": host_total,
        "host_evaluable_apks": sum(row.get("host_count", 0) > 0 for row in valid),
        "zero_host_apks": sum(row.get("host_count", 0) == 0 for row in valid),
        "host_exact_alignment": sum(row.get("host_exact_aligned_count", 0) for row in valid) / max(1, host_total),
        "root_graph_alignment": float(np.mean([row["root_graph_alignment"] for row in valid])) if valid else 0.0,
        "root_outgoing_edge_rate": float(np.mean([row["root_outgoing_edge_rate"] for row in valid])) if valid else 0.0,
        "ordinary_method_fraction": float(np.mean([row["ordinary_method_fraction"] for row in valid])) if valid else 0.0,
        "any_root_host_assignment": any_root / max(1, host_total), "directional_host_assignment": directional / max(1, host_total),
        "critical_directional_assignment": critical_directional / max(1, len(critical)),
        "mean_attention_directional_mass": float(np.mean(attention_values)) if attention_values else 0.0,
        "trigger_counts": dict(Counter(row["trigger_type"] for row in host_rows)),
        "apks_with_multihop_path": sum(row.get("multi_hop_path_count", 0) > 0 for row in valid),
        "folds_with_USER_ONLY": len({row["fold"] for row in host_rows if row["trigger_type"] == "USER_ONLY"}),
        "folds_with_AUTONOMOUS_ONLY": len({row["fold"] for row in host_rows if row["trigger_type"] == "AUTONOMOUS_ONLY"}),
        "future_rows_loaded": 0,
    }


def report_and_stop(status: str, stage: str, reason: str, integrity: str = "PENDING") -> None:
    placeholders(stage, status)
    dump_json(OUTPUT / "final_decision.json", {"status": status, "stopped_at": stage, "reason": reason, "future_rows_loaded": 0, "formal_integrity": integrity})
    supporting = []
    for name in ("graph_lineage.md", "pilot_report.md", "trcbe_fullgraph_coverage_report.md"):
        path = OUTPUT / name
        if path.is_file():
            supporting.append(path.read_text(encoding="utf-8"))
    report = f"""# CARE-Droid full-graph TRCBE audit\n\n## Final status\n\n**{status}**\n\nStopped at: **{stage}**.\n\n{reason}\n\nFormal asset integrity: **{integrity}**. No labels were used for graph construction or coverage selection. Direction probe, end-to-end correction and future diagnostics were not run unless their preceding gate passed.\n\n""" + "\n\n".join(supporting)
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(f"CARE_FULLGRAPH_TRCBE_FINAL_STATUS={status}", flush=True)
    print(f"CARE_FULLGRAPH_TRCBE_STOPPED_AT={stage}", flush=True)


def finalize_integrity(before: list[dict[str, Any]], manifest_inputs: list[tuple[Path, str]]) -> str:
    after = hash_manifest(manifest_inputs, "after")
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after})
    before_map = {row["absolute_path"]: row for row in before}
    after_map = {row["absolute_path"]: row for row in after}
    changed = [{"path": path, "before": before_map.get(path), "after": after_map.get(path)} for path in sorted(set(before_map) | set(after_map)) if before_map.get(path) != after_map.get(path)]
    status = "PASS" if not changed else "FAIL"
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "changed": changed, "asset_count": len(after)})
    return status


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    STATUS_CACHE.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduce\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\n"
        "bash scripts/care-fullgraph-trcbe-v1-r2/launch_tmux.sh\n"
        "tail -F outputs/care-fullgraph-trcbe-v2.console.log\n"
        "bash scripts/care-fullgraph-trcbe-v1-r2/collect_results.sh\n```\n",
        encoding="utf-8",
    )
    phase("stage_A", "START")
    assets = discover_assets()
    formal = formal_rows(assets["formal"])
    routed = [row for row in formal if row["routed"]]
    if len(routed) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: formal routed rows={len(routed)} expected=200")
    wanted = {row["sha256"] for row in routed}
    fold_by = {row["sha256"]: row["fold"] for row in routed}
    apk_map, apk_audit = load_apk_map(wanted)
    endpoints = read_endpoint_rows(assets["endpoint"], wanted)
    attention = load_attention(assets["attention"])
    taxonomy_audit = taxonomy_contract(assets["trcbe_schema"])
    toolchain = executable_inventory()
    candidates, script_hits = scan_candidates()
    samples = [sample_candidate(Path(row["path"])) for row in candidates[:50] if Path(row["path"]).is_file()]
    dump_json(OUTPUT / "asset_inventory.json", {"scope": [str(PROJECT / name) for name in ("artifacts", "outputs", "scripts", "src")], "candidate_graph_assets": candidates, "graph_script_hits": script_hits, "apk_resolution": apk_audit, "routed_apks_resolved": len(apk_map), "future_rows_loaded": 0})
    dump_json(OUTPUT / "toolchain_audit.json", toolchain)
    dump_json(OUTPUT / "graph_samples_audit.json", {"samples": samples, "verified_reusable_full_graphs": [], "note": "No candidate is promoted by filename alone. The frozen endpoint cache is explicitly bounded and excluded."})

    manifest_inputs = [(assets[key], key) for key in sorted(assets)] + [(apk_map[sid], "2017 formal routed APK") for sid in sorted(apk_map)]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    if before_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))["assets"]
    else:
        before = hash_manifest(manifest_inputs, "before")
        dump_json(before_path, {"git_used": False, "assets": before})

    full_assets = [sample for sample in samples if sample.get("verified_full_graph")]
    if full_assets:
        feasibility = "REUSABLE_FULL_GRAPH_FOUND"
        reason = "A content-verified complete graph cache was found."
    elif len(apk_map) == 200 and len(endpoints) == 200 and toolchain.get("androguard_import"):
        feasibility = "BUILDABLE_WITH_EXISTING_TOOLCHAIN"
        reason = "All routed APKs are present and the already-installed Androguard DEX API can reconstruct app-defined caller→callee edges offline without Android SDK, download, installation, network, or new permission."
    elif len(apk_map) < 200:
        feasibility = "NOT_FEASIBLE"
        reason = f"Only {len(apk_map)}/200 routed APK files were resolved."
    else:
        feasibility = "NOT_FEASIBLE"
        reason = "No verified complete graph/upstream raw edge asset exists and the required installed Androguard DEX API is unavailable."
    feasibility_record = {"status": feasibility, "reason": reason, "taxonomy": taxonomy_audit, "routed_apks": len(routed), "apk_paths_resolved": len(apk_map), "endpoint_rows": len(endpoints), "future_rows_loaded": 0, "labels_used": False}
    dump_json(OUTPUT / "feasibility_decision.json", feasibility_record)
    lineage = f"""# Full graph lineage audit\n\n- Frozen endpoint graph: `{assets['endpoint']}`; known to retain two upstream and one downstream hop around sensitive endpoints. It is not a complete graph.\n- Verified reusable complete graph cache: **{'yes' if full_assets else 'no'}**. Candidate names were not accepted without content verification.\n- Selected recovery path: **{feasibility}**.\n- Builder: parse every app-defined DEX method and each invoke instruction; preserve direct app-internal caller→callee references and app-local class-hierarchy virtual/interface targets.\n- Framework implementations, reflection, native code and dynamically loaded code remain outside the graph. App callback overrides remain explicit root nodes.\n- No Android SDK/android.jar is required by this selected algorithm. No network, install, privilege escalation or formal-asset write is used.\n"""
    (OUTPUT / "graph_lineage.md").write_text(lineage, encoding="utf-8")
    phase("stage_A", "DONE", decision=feasibility)
    if feasibility not in {"REUSABLE_FULL_GRAPH_FOUND", "RESTORABLE_FROM_EXISTING_ASSETS", "BUILDABLE_WITH_EXISTING_TOOLCHAIN"}:
        integrity = finalize_integrity(before, manifest_inputs)
        report_and_stop(feasibility, "STAGE_A_ASSET_AUDIT", reason, integrity)
        return

    # Stage B: deterministic, label-free 15-APK pilot.
    phase("stage_B_pilot", "START")
    pilot = select_pilot(routed, endpoints, apk_map)
    dump_json(OUTPUT / "pilot_selection.json", {"selection_frozen_before_build": True, "labels_used": False, "rows": pilot})
    pilot_metrics: list[dict[str, Any]] = []
    pilot_hosts: list[dict[str, Any]] = []
    path_examples: list[dict[str, Any]] = []
    pilot_statuses = []
    for number, row in enumerate(pilot, 1):
        sid = row["sha256"]
        status = build_one(sid, apk_map[sid], PILOT_TIMEOUT_SECONDS)
        pilot_statuses.append(status)
        if status.get("status") == "OK":
            graph = read_gzip_json(GRAPH_CACHE / f"{sid}.json.gz")
            metric, hosts, paths = analyze_graph(sid, row["fold"], graph, endpoints[sid], attention.get(sid))
            metric.update({"build_wall_seconds": status.get("wall_seconds"), "peak_rss_kib": status.get("peak_rss_kib")})
            pilot_metrics.append(metric)
            pilot_hosts.extend(hosts)
            path_examples.extend(paths)
        else:
            pilot_metrics.append({"sha256": sid, "fold": row["fold"], "status": status.get("status", "ERROR"), "error": status.get("error", status.get("error_type", ""))})
        print(f"FULLGRAPH_PILOT {number}/{len(pilot)} status={status.get('status')}", flush=True)
    pilot_aggregate = aggregate(pilot_metrics, pilot_hosts)
    successful_status = [row for row in pilot_statuses if row.get("status") == "OK"]
    projected_hours = sum(float(row.get("wall_seconds", 0)) for row in successful_status) / max(1, len(successful_status)) * 200 / 3600
    pilot_aggregate["projected_serial_hours_for_200"] = projected_hours
    pilot_aggregate["path_examples"] = path_examples[:30]
    pilot_criteria = {
        "build_success_rate>=0.80": pilot_aggregate["success_rate"] >= 0.80,
        "host_exact_alignment>=0.90": pilot_aggregate["host_exact_alignment"] >= 0.90,
        "host_evaluable_apks>=10": pilot_aggregate["host_evaluable_apks"] >= 10,
        "root_graph_alignment>=0.50": pilot_aggregate["root_graph_alignment"] >= 0.50,
        "critical_directional>=2x_old": pilot_aggregate["critical_directional_assignment"] >= 2 * OLD["critical_directional"],
        "attention_directional>=2x_old": pilot_aggregate["mean_attention_directional_mass"] >= 2 * OLD["attention_directional"],
        "ordinary_method_fraction>=0.50": pilot_aggregate["ordinary_method_fraction"] >= 0.50,
        "multihop_paths_in_half_successes": pilot_aggregate["apks_with_multihop_path"] >= math.ceil(pilot_aggregate["successful_apks"] / 2),
        "projected_serial_cost<=24h": projected_hours <= 24.0,
    }
    if not pilot_criteria["build_success_rate>=0.80"] or not pilot_criteria["root_graph_alignment>=0.50"]:
        pilot_decision = "FAIL_TOOLCHAIN"
    elif not pilot_criteria["host_exact_alignment>=0.90"]:
        pilot_decision = "FAIL_ALIGNMENT"
    elif not pilot_criteria["projected_serial_cost<=24h"]:
        pilot_decision = "FAIL_COST"
    elif not all(pilot_criteria.values()):
        pilot_decision = "FAIL_INCOMPLETE_GRAPH"
    else:
        pilot_decision = "PASS"
    dump_json(OUTPUT / "pilot_graph_metrics.json", {"aggregate": pilot_aggregate, "per_apk": pilot_metrics, "build_status": pilot_statuses, "criteria": pilot_criteria})
    dump_json(OUTPUT / "pilot_decision.json", {"status": pilot_decision, "criteria": pilot_criteria, "future_rows_loaded": 0})
    pilot_report = f"""# Full-graph TRCBE pilot\n\nDecision: **{pilot_decision}**\n\n- Build success: {pilot_aggregate['successful_apks']}/{pilot_aggregate['apk_count']}\n- Methods: {pilot_aggregate['method_count']}; edges: {pilot_aggregate['edge_count']}\n- Root graph alignment: {pilot_aggregate['root_graph_alignment']:.6f} (old {OLD['root_alignment']:.6f})\n- Host exact alignment (pooled over actual Hosts): {pilot_aggregate['host_exact_alignment']:.6f}\n- Host-evaluable APKs: {pilot_aggregate['host_evaluable_apks']}; zero-Host APKs: {pilot_aggregate['zero_host_apks']}\n- Directional Host assignment: {pilot_aggregate['directional_host_assignment']:.6f}\n- Critical directional assignment: {pilot_aggregate['critical_directional_assignment']:.6f} (old {OLD['critical_directional']:.6f})\n- Attention directional mass: {pilot_aggregate['mean_attention_directional_mass']:.6f} (old {OLD['attention_directional']:.6f})\n- Multi-hop examples: {len(path_examples)}\n- Projected serial cost for 200 APKs: {projected_hours:.2f} h\n- Trigger counts: `{json.dumps(pilot_aggregate['trigger_counts'], sort_keys=True)}`\n"""
    (OUTPUT / "pilot_report.md").write_text(pilot_report, encoding="utf-8")
    phase("stage_B_pilot", "DONE", decision=pilot_decision)
    if pilot_decision != "PASS":
        integrity = finalize_integrity(before, manifest_inputs)
        status = "PILOT_FAILED"
        reason = f"Pilot gate failed with {pilot_decision}; coverage or cost did not justify building 200 graphs."
        report_and_stop(status, "STAGE_B_PILOT", reason, integrity)
        return

    # Stage C: freeze the builder and extend only to the formal routed 200.
    phase("stage_C_full_200", "START")
    protocol = {
        "version": "care-fullgraph-trcbe-v1-r2", "scope": "2017 formal DI25 routed 200",
        "graph_algorithm": "DEX direct-reference app-internal plus app-local CHA",
        "caller_callee_direction": "caller_to_callee", "apk_timeout_seconds": FULL_TIMEOUT_SECONDS,
        "pilot_criteria": pilot_criteria, "taxonomy_sha256": taxonomy_audit["sha256"],
        "labels_used_for_graph_or_coverage": False, "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "full_graph_protocol.json", protocol)
    (OUTPUT / "full_graph_protocol_hash.txt").write_text(sha256_file(OUTPUT / "full_graph_protocol.json") + "\n", encoding="utf-8")
    dump_json(OUTPUT / "full_graph_schema.json", {
        "schema_version": "1.0.0", "node": "all app-defined DEX methods with canonical signature and stable id",
        "edge": "caller_index,callee_index", "edge_direction": "caller_to_callee",
        "retains_ordinary_methods": True, "retains_callback_overrides": True, "per_apk": True,
        "limitations": ["reflection", "native code", "dynamically loaded code", "framework internals"],
    })
    all_statuses = []
    for number, row in enumerate(sorted(routed, key=lambda item: (item["fold"], item["sha256"])), 1):
        status = build_one(row["sha256"], apk_map[row["sha256"]], FULL_TIMEOUT_SECONDS)
        all_statuses.append({**status, "fold": row["fold"]})
        if number % 5 == 0 or number == len(routed):
            print(f"FULLGRAPH_200_BUILD {number}/{len(routed)} ok={sum(s.get('status')=='OK' for s in all_statuses)} failed={sum(s.get('status')!='OK' for s in all_statuses)}", flush=True)
    fields = sorted({key for row in all_statuses for key in row})
    write_csv(OUTPUT / "full_graph_build_status.csv", all_statuses, fields)

    full_metrics: list[dict[str, Any]] = []
    full_hosts: list[dict[str, Any]] = []
    full_paths: list[dict[str, Any]] = []
    for row in sorted(routed, key=lambda item: (item["fold"], item["sha256"])):
        sid = row["sha256"]
        status_path = STATUS_CACHE / f"{sid}.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {"status": "MISSING"}
        if status.get("status") != "OK":
            full_metrics.append({"sha256": sid, "fold": row["fold"], "status": status.get("status", "MISSING")})
            continue
        graph = read_gzip_json(GRAPH_CACHE / f"{sid}.json.gz")
        metric, hosts, paths = analyze_graph(sid, row["fold"], graph, endpoints[sid], attention.get(sid))
        metric.update({"build_wall_seconds": status.get("wall_seconds"), "peak_rss_kib": status.get("peak_rss_kib")})
        full_metrics.append(metric)
        full_hosts.extend(hosts)
        full_paths.extend(paths)
    full_aggregate = aggregate(full_metrics, full_hosts)
    dump_json(OUTPUT / "trcbe_fullgraph_coverage.json", {"aggregate": full_aggregate, "per_apk": full_metrics, "path_examples": full_paths[:100], "old_reference": OLD})
    host_fields = ["sha256", "fold", "host_order", "host_node_id", "full_node_index", "trigger_type", "user_distance", "autonomous_distance", "neutral_distance", "attention", "critical", "roles"]
    write_csv(OUTPUT / "trcbe_fullgraph_host_assignment.csv", full_hosts, host_fields)
    full_criteria = {
        "critical_directional_assignment>=0.20": full_aggregate["critical_directional_assignment"] >= 0.20,
        "attention_directional_mass>=0.20": full_aggregate["mean_attention_directional_mass"] >= 0.20,
        "root_alignment>old": full_aggregate["root_graph_alignment"] > OLD["root_alignment"],
        "directional_host_assignment>old": full_aggregate["directional_host_assignment"] > OLD["directional_host"],
        "unreachable<old": full_aggregate["trigger_counts"].get("UNREACHABLE", 0) / max(1, full_aggregate["host_count"]) < OLD["unreachable"],
        "BOTH>old": full_aggregate["trigger_counts"].get("BOTH", 0) > OLD["both"],
        "USER_ONLY_across>=2folds": full_aggregate["folds_with_USER_ONLY"] >= 2,
        "AUTONOMOUS_ONLY_across>=2folds": full_aggregate["folds_with_AUTONOMOUS_ONLY"] >= 2,
        "host_exact_alignment>=0.90": full_aggregate["host_exact_alignment"] >= 0.90,
        "build_success_rate>=0.90": full_aggregate["success_rate"] >= 0.90,
        "multihop_paths_present": full_aggregate["apks_with_multihop_path"] > 0,
    }
    full_decision = "FEASIBLE_FOR_DIRECTION_PROBE" if all(full_criteria.values()) else "NOT_FEASIBLE_WITH_AVAILABLE_GRAPH"
    dump_json(OUTPUT / "trcbe_fullgraph_decision.json", {"status": full_decision, "criteria": full_criteria, "coverage": full_aggregate, "future_rows_loaded": 0})
    coverage_report = f"""# Full-graph TRCBE label-free coverage\n\nDecision: **{full_decision}**\n\n- Graph success: {full_aggregate['successful_apks']}/200\n- Root entry: {full_aggregate['root_graph_alignment']:.6f} (old {OLD['root_alignment']:.6f})\n- Directional Host assignment: {full_aggregate['directional_host_assignment']:.6f} (old {OLD['directional_host']:.6f})\n- Critical Host directional coverage: {full_aggregate['critical_directional_assignment']:.6f}\n- C43 attention directional mass: {full_aggregate['mean_attention_directional_mass']:.6f}\n- Trigger counts: `{json.dumps(full_aggregate['trigger_counts'], sort_keys=True)}`\n- Multi-hop APKs: {full_aggregate['apks_with_multihop_path']}\n\nLabels and future rows were not read for this coverage decision.\n"""
    (OUTPUT / "trcbe_fullgraph_coverage_report.md").write_text(coverage_report, encoding="utf-8")
    phase("stage_C_full_200", "DONE", decision=full_decision)
    integrity = finalize_integrity(before, manifest_inputs)
    if integrity != "PASS":
        report_and_stop("BLOCKED", "FORMAL_ASSET_INTEGRITY", "At least one protected baseline asset changed during the run.", integrity)
        return
    if full_decision != "FEASIBLE_FOR_DIRECTION_PROBE":
        report_and_stop("NOT_FEASIBLE", "STAGE_C_FULLGRAPH_COVERAGE", "The complete-graph coverage decision is NOT_FEASIBLE_WITH_AVAILABLE_GRAPH: it did not satisfy the frozen 20% Critical-Host and attention-mass gates or another predeclared coverage condition.", integrity)
        return
    # This package deliberately terminates at the valid scientific handoff boundary.
    # A direction package needs the newly materialized 200-APK evidence and can now be
    # implemented without ever weakening the label-free gate.
    placeholders("STAGE_C_COVERAGE_PASSED", full_decision)
    dump_json(OUTPUT / "final_decision.json", {"status": full_decision, "stopped_at": "STAGE_C_FULLGRAPH_COVERAGE", "reason": "Coverage gates passed. Direction probe is scientifically authorized but is not fabricated inside the label-free graph-materialization package.", "future_rows_loaded": 0, "formal_integrity": integrity})
    (OUTPUT / "report.md").write_text(coverage_report + "\nFormal asset integrity: **PASS**. Direction probe is now authorized; it was not run by this label-free graph package.\n", encoding="utf-8")
    print("CARE_FULLGRAPH_TRCBE_FINAL_STATUS=FEASIBLE_FOR_DIRECTION_PROBE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        dump_json(OUTPUT / "fatal_error.json", {"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=30), "future_rows_loaded": 0})
        placeholders("FATAL_ERROR", "BLOCKED")
        integrity = "NOT_AVAILABLE"
        before_path = OUTPUT / "baseline_asset_manifest_before.json"
        if before_path.is_file():
            try:
                before = json.loads(before_path.read_text(encoding="utf-8"))["assets"]
                after = []
                for record in before:
                    path = Path(record["absolute_path"])
                    after.append(file_record(path, record.get("purpose", "protected asset")))
                dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after})
                before_map = {row["absolute_path"]: row for row in before}
                after_map = {row["absolute_path"]: row for row in after}
                changed = [{"path": path, "before": before_map.get(path), "after": after_map.get(path)} for path in sorted(set(before_map) | set(after_map)) if before_map.get(path) != after_map.get(path)]
                integrity = "PASS" if not changed else "FAIL"
                dump_json(OUTPUT / "baseline_integrity_check.json", {"status": integrity, "changed": changed, "asset_count": len(after)})
            except Exception as integrity_exc:
                dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "BLOCKED", "error": repr(integrity_exc)})
        dump_json(OUTPUT / "final_decision.json", {"status": "BLOCKED", "stopped_at": "ENGINEERING_ERROR", "reason": str(exc), "future_rows_loaded": 0, "formal_integrity": integrity})
        (OUTPUT / "report.md").write_text(f"# CARE-Droid full-graph TRCBE audit\n\n**BLOCKED** by an engineering error before a scientific decision: `{type(exc).__name__}: {exc}`.\n\nFormal asset integrity: **{integrity}**. See `fatal_error.json`.\n", encoding="utf-8")
        print(f"CARE_FULLGRAPH_TRCBE_FATAL {type(exc).__name__}: {exc}", flush=True)
        raise
