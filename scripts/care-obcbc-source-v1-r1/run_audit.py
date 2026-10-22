#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from common import (
    GRAPH_ROOT, OUTPUT, PROJECT, SCRIPT_DIR, compare_manifest, discover_assets,
    dump_json, endpoint_hosts, formal_rows, hash_manifest, load_apk_map,
    load_attention, read_csv, read_endpoint_rows, read_gzip_json, sha256_file,
    stable_hash,
)
from context import graph_context, instruction_context, nearest_boundaries, resource_callbacks, root_category
from ownership import CorpusStats, assign_graph, build_corpus_stats, manifest_metadata

PACKAGE_ID = "care-obcbc-source-v1-r1-stage-b-static-only"
OWNERSHIP_FIELDS = [
    "sha256", "fold", "class_id", "ownership", "method_count",
    "manifest_component_exact", "manifest_namespace_match", "component_namespace_match",
    "structure_reuse_apks", "shape_reuse_apks", "class_name_reuse_apks",
    "structure_id", "shape_id", "evidence", "raw_name_emitted",
]
BOUNDARY_FIELDS = [
    "sha256", "fold", "host_order", "host_node_id", "host_ownership", "trcbe_trigger_type",
    "attention", "critical", "roles", "boundary_source_id", "boundary_target_id",
    "source_class_id", "target_class_id", "boundary_to_host_distance", "root_category",
    "user_root_distance", "autonomous_root_distance", "method_code_available",
    "branch_context_proxy", "branch_count", "switch_count", "ui_callback_root",
    "resource_callback_link", "exact_control_dependency", "identity_feature_emitted",
]


def write_phase(name: str, status: str, **extra: Any) -> None:
    path = OUTPUT / "checkpoints/phases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"phase": name, "status": status, "epoch": int(time.time()), **extra}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"OBCBC_PHASE_{status} name={name}", flush=True)


def copy_frozen_literature() -> None:
    source = SCRIPT_DIR / "frozen_literature"
    for name in ("literature_search_strategy.md", "literature_inventory.csv", "mechanism_matrix.md", "novelty_audit.md", "literature_decision.json"):
        target = OUTPUT / name
        if not target.exists():
            shutil.copy2(source / name, target)
    decision = json.loads((OUTPUT / "literature_decision.json").read_text(encoding="utf-8"))
    if decision.get("status") != "NOVELTY_PLAUSIBLE":
        raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-A literature gate is not NOVELTY_PLAUSIBLE")


def trcbe_map(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    result = {}
    for row in read_csv(path):
        sid = str(row.get("sha256", "")).strip().casefold()
        # Stable Host IDs are uint64 values. Never parse them through float,
        # which silently loses identity above 2**53.
        node = int(Decimal(str(row["host_node_id"]).strip()))
        result[(sid, node)] = row
    return result


def class_id(method: dict[str, Any]) -> str:
    return stable_hash(str(method["class"]).casefold())


def fold_counter_template() -> dict[str, float]:
    return defaultdict(float)


def finalize_later_placeholders(gate: str, final_status: str) -> None:
    payload = {
        "status": "NOT_RUN_DUE_TO_PREVIOUS_GATE",
        "gate": gate,
        "stage_b_final_status": final_status,
        "note": "This Stage-B package never reads labels or runs a direction model. A separate authorized package is required after a STATICALLY_FEASIBLE decision.",
        "future_rows_loaded": 0,
    }
    for name in ("sufficiency_probe.json", "matched_null_results.json", "end_to_end_results.json"):
        dump_json(OUTPUT / name, payload)


def dominance(values: dict[str, float], fraction: float = 0.10) -> float:
    if not values:
        return 0.0
    ordered = sorted(values.values(), reverse=True)
    take = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:take]) / max(1e-12, sum(ordered))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    identity_path = OUTPUT / "run_identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("package_id") != PACKAGE_ID:
            raise RuntimeError("BLOCKED_BY_CONTRACT: output belongs to another package")
    else:
        allowed_existing = {
            "logs", "checkpoints", "literature_search_strategy.md", "literature_inventory.csv",
            "mechanism_matrix.md", "novelty_audit.md", "literature_decision.json",
            "report.md", "reproduce_commands.md",
        }
        unexpected = sorted(path.name for path in OUTPUT.iterdir() if path.name not in allowed_existing)
        if unexpected:
            raise RuntimeError(
                "BLOCKED_BY_CONTRACT: output directory contains files from another or later run; "
                f"refusing overwrite: {unexpected[:20]}"
            )
        dump_json(identity_path, {"package_id": PACKAGE_ID, "created_epoch": int(time.time()), "static_only": True})
    copy_frozen_literature()

    write_phase("asset_preflight", "START")
    assets = discover_assets()
    formal = formal_rows(assets["formal"])
    routed = [row for row in formal if row["routed"]]
    if len(routed) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: routed={len(routed)} expected=200")
    wanted = {row["sha256"] for row in routed}
    fold_by = {row["sha256"]: row["fold"] for row in routed}
    apk_map, apk_audit = load_apk_map(wanted)
    if len(apk_map) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths={len(apk_map)}/200")
    graph_paths = {sid: GRAPH_ROOT / f"{sid}.json.gz" for sid in wanted}
    missing_graphs = [sid for sid, path in graph_paths.items() if not path.is_file()]
    if missing_graphs:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: complete graphs missing={len(missing_graphs)} first={missing_graphs[:3]}")
    endpoints = read_endpoint_rows(assets["endpoint"], wanted)
    if len(endpoints) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint rows={len(endpoints)}/200")
    attention = load_attention(assets["attention"])
    trigger_by_host = trcbe_map(assets["trcbe_hosts"])
    dump_json(OUTPUT / "asset_preflight.json", {
        "status": "PASS", "routed_rows": len(routed), "apk_paths": len(apk_map),
        "full_graphs": len(graph_paths), "endpoint_rows": len(endpoints),
        "attention_apks": sum(sid in attention for sid in wanted),
        "trcbe_host_rows": len(trigger_by_host), "apk_resolution": apk_audit,
        "labels_read": False, "future_rows_loaded": 0, "dynamic_analysis_used": False,
    })
    protected = [(assets[key], f"formal static input: {key}") for key in sorted(assets)]
    protected += [(graph_paths[sid], "frozen complete method graph") for sid in sorted(wanted)]
    protected += [(apk_map[sid], "2017 formal routed APK; read-only static parse") for sid in sorted(wanted)]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    if before_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))["assets"]
    else:
        before = hash_manifest(protected, "before")
        dump_json(before_path, {"git_used": False, "assets": before})
    write_phase("asset_preflight", "DONE")

    protocol = {
        "package_id": PACKAGE_ID,
        "scope": "2017 formal DI25 routed 200; label-free Stage-B static feasibility only",
        "ownership_unit": "class; methods inherit class assignment",
        "first_party_rules": [
            "exact Manifest component class and no conflicting cross-APK reuse",
            "Manifest/component namespace plus APK-specific structural fingerprint and at least two methods",
        ],
        "third_party_rules": [
            "cross-APK structural reuse in >=3 APKs plus independent name/shape reuse and outside app namespace",
            "cross-APK class-name and structure reuse in >=5 APKs with >=3 methods",
        ],
        "unknown_policy": "conflict or insufficient evidence => UNKNOWN",
        "boundary": "nearest FIRST_PARTY->THIRD_PARTY edge on reverse paths that remain entirely inside high-confidence THIRD_PARTY nodes",
        "raw_identity_as_feature": False,
        "labels_used": False,
        "future_rows_loaded": 0,
        "gates": {
            "host_ownership_coverage>=0.35": 0.35,
            "critical_host_ownership_coverage>=0.35": 0.35,
            "both_third_hosts_with_multiple_boundaries>=20": 20,
            "both_third_host_decomposition_ratio>=0.10": 0.10,
            "accountable_attention_mass>=0.20": 0.20,
            "user_boundaries>=20_and_3folds": [20, 3],
            "autonomous_boundaries>=20_and_3folds": [20, 3],
            "unknown_critical_ratio<=0.65": 0.65,
            "branch_or_resource_context_boundaries>=20_and_ratio>=0.05": [20, 0.05],
            "top10pct_apk_boundary_share<=0.60": 0.60,
        },
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "protocol_hash.txt").write_text(sha256_file(OUTPUT / "protocol.json") + "\n", encoding="utf-8")
    dump_json(OUTPUT / "ownership_schema.json", {
        "schema_version": "1.0.0", "classes": ["FIRST_PARTY", "THIRD_PARTY", "UNKNOWN"],
        "assignment_is_label_free": True,
        "signals": ["Manifest component seed", "Manifest/component namespace as one weak signal", "cross-APK class structural reuse", "cross-APK name-hash reuse", "method-shape reuse"],
        "forbidden_as_single_truth": ["Manifest namespace", "different package prefix", "vendor name", "library whitelist", "short obfuscated package"],
        "emitted_identifiers": "SHA256 hashes only; raw package/class names are not written",
        "model_feature_status": "No model is run in Stage B; ownership is a path-decomposition variable only",
    })
    dump_json(OUTPUT / "shortcut_null_protocol.json", {
        "status": "FROZEN_NOT_RUN_STAGE_B_ONLY", "future_rows_loaded": 0,
        "controls": ["COUNT_ONLY", "OWNERSHIP_SHUFFLE", "BOUNDARY_SHUFFLE", "PACKAGE_ONLY", "LIBRARY_DELETION", "OWNERSHIP_CONCATENATION", "ROOT_ONLY", "UI_TEXT_SHUFFLE", "CONTROL_CONTEXT_SHUFFLE", "SDK_IDENTITY_ZERO"],
        "identity_zero_assertion": "raw package, SDK/vendor, library id and class names are absent from boundary_inventory and all later model contracts",
    })

    write_phase("corpus_fingerprints", "START")
    corpus = build_corpus_stats(graph_paths, read_gzip_json)
    dump_json(OUTPUT / "corpus_fingerprint_audit.json", {
        "class_instances": corpus.class_count,
        "unique_structure_fingerprints": len(corpus.structure_count),
        "unique_shape_fingerprints": len(corpus.shape_count),
        "unique_class_name_hashes": len(corpus.class_name_count),
        "labels_used": False, "future_rows_loaded": 0,
    })
    write_phase("corpus_fingerprints", "DONE")

    write_phase("ownership_and_boundaries", "START")
    ownership_totals = Counter()
    host_ownership_totals = Counter()
    critical_ownership_totals = Counter()
    attention_by_ownership = Counter()
    fold_metrics: dict[int, Counter] = defaultdict(Counter)
    global_counts = Counter()
    per_apk_boundaries = Counter()
    per_structure_boundaries = Counter()
    manifest_records = []

    ownership_path = OUTPUT / "ownership_assignment.csv"
    boundary_path = OUTPUT / "boundary_inventory.csv"
    with ownership_path.open("w", encoding="utf-8", newline="") as owner_handle, boundary_path.open("w", encoding="utf-8", newline="") as boundary_handle:
        owner_writer = csv.DictWriter(owner_handle, fieldnames=OWNERSHIP_FIELDS)
        boundary_writer = csv.DictWriter(boundary_handle, fieldnames=BOUNDARY_FIELDS)
        owner_writer.writeheader()
        boundary_writer.writeheader()
        for number, row in enumerate(sorted(routed, key=lambda item: (item["fold"], item["sha256"])), 1):
            sid, fold = row["sha256"], row["fold"]
            graph = read_gzip_json(graph_paths[sid])
            manifest = manifest_metadata(apk_map[sid])
            manifest_records.append({"sha256": sid, "fold": fold, "package_hash": manifest["package_hash"], "component_count": manifest["component_count"]})
            method_ownership, class_rows, class_map = assign_graph(graph, manifest, corpus)
            for record in class_rows:
                owner_writer.writerow({"sha256": sid, "fold": fold, **record})
                ownership_totals[record["ownership"]] += int(record["method_count"])
            ctx = graph_context(graph)
            index_by_id = {int(method["node_id"]): index for index, method in enumerate(graph["methods"])}
            hosts = endpoint_hosts(endpoints[sid])
            att = attention.get(sid)
            attention_valid = bool(att is not None and len(att["attention"]) == len(hosts))
            host_results = []
            all_boundaries = []
            source_signatures = set()
            for host in hosts:
                host_index = index_by_id.get(int(host["node_id"]))
                ownership = method_ownership[host_index] if host_index is not None else "UNKNOWN"
                host_ownership_totals[ownership] += 1
                weight = float(att["attention"][host["host_order"]]) if attention_valid else 0.0
                attention_by_ownership[ownership] += weight
                critical = int(attention_valid and host["host_order"] == int(att["critical"]))
                if critical:
                    critical_ownership_totals[ownership] += 1
                trigger = trigger_by_host.get((sid, int(host["node_id"])), {}).get("trigger_type", "MISSING")
                boundaries = nearest_boundaries(host_index, method_ownership, ctx["reverse"]) if host_index is not None else []
                for boundary in boundaries:
                    source_signatures.add(str(graph["methods"][boundary["source"]]["signature"]).casefold())
                host_results.append({"host": host, "index": host_index, "ownership": ownership, "weight": weight, "critical": critical, "trigger": trigger, "boundaries": boundaries})
            static_context = instruction_context(apk_map[sid], source_signatures)
            resources = resource_callbacks(apk_map[sid])
            global_counts["layout_files"] += resources["layout_files"]
            global_counts["parsed_layout_files"] += resources["parsed_layout_files"]
            for host_record in host_results:
                host = host_record["host"]
                boundaries = host_record["boundaries"]
                if boundaries:
                    global_counts["hosts_with_boundary"] += 1
                    global_counts["attention_accountable"] += host_record["weight"]
                    fold_metrics[fold]["attention_total_accountable"] += host_record["weight"]
                global_counts["host_total"] += 1
                global_counts["attention_total"] += host_record["weight"]
                fold_metrics[fold]["host_total"] += 1
                fold_metrics[fold][f"host_{host_record['ownership']}"] += 1
                if host_record["trigger"] == "BOTH":
                    global_counts["both_hosts"] += 1
                    if host_record["ownership"] == "THIRD_PARTY":
                        global_counts["both_third_hosts"] += 1
                        distinct_sources = {item["source"] for item in boundaries}
                        if len(distinct_sources) >= 2:
                            global_counts["both_third_multiboundary_hosts"] += 1
                            fold_metrics[fold]["both_third_multiboundary_hosts"] += 1
                for boundary in boundaries:
                    source = graph["methods"][boundary["source"]]
                    target = graph["methods"][boundary["target"]]
                    signature = str(source["signature"]).casefold()
                    static = static_context.get(signature, {})
                    category, user_distance, auto_distance = root_category(boundary["source"], ctx)
                    ui_root = int(category == "USER_ONLY")
                    resource_link = int(str(source["name"]).casefold() in resources["callback_names"])
                    source_class = class_map[str(source["class"]).casefold()]
                    target_class = class_map[str(target["class"]).casefold()]
                    record = {
                        "sha256": sid, "fold": fold, "host_order": host["host_order"], "host_node_id": host["node_id"],
                        "host_ownership": host_record["ownership"], "trcbe_trigger_type": host_record["trigger"],
                        "attention": host_record["weight"], "critical": host_record["critical"], "roles": "|".join(map(str, host["roles"])),
                        "boundary_source_id": int(source["node_id"]), "boundary_target_id": int(target["node_id"]),
                        "source_class_id": source_class["class_id"], "target_class_id": target_class["class_id"],
                        "boundary_to_host_distance": boundary["distance"], "root_category": category,
                        "user_root_distance": "" if user_distance is None else user_distance,
                        "autonomous_root_distance": "" if auto_distance is None else auto_distance,
                        "method_code_available": static.get("method_code_available", 0),
                        "branch_context_proxy": static.get("branch_context_proxy", 0),
                        "branch_count": static.get("branch_count", 0), "switch_count": static.get("switch_count", 0),
                        "ui_callback_root": ui_root, "resource_callback_link": resource_link,
                        "exact_control_dependency": 0, "identity_feature_emitted": 0,
                    }
                    boundary_writer.writerow(record)
                    all_boundaries.append(record)
                    global_counts["boundary_total"] += 1
                    per_apk_boundaries[sid] += 1
                    per_structure_boundaries[target_class["structure_id"]] += 1
                    fold_metrics[fold]["boundary_total"] += 1
                    if category == "USER_ONLY":
                        global_counts["user_boundary"] += 1
                        fold_metrics[fold]["user_boundary"] += 1
                    elif category == "AUTONOMOUS_ONLY":
                        global_counts["auto_boundary"] += 1
                        fold_metrics[fold]["auto_boundary"] += 1
                    elif category == "BOTH":
                        global_counts["both_boundary"] += 1
                    if static.get("branch_context_proxy", 0) or resource_link:
                        global_counts["control_or_resource_context"] += 1
                        fold_metrics[fold]["control_or_resource_context"] += 1
                    if resource_link:
                        global_counts["resource_link"] += 1
            for source_index, target_index in graph["edges"]:
                source_owner, target_owner = method_ownership[int(source_index)], method_ownership[int(target_index)]
                if source_owner == "FIRST_PARTY" and target_owner == "THIRD_PARTY":
                    global_counts["all_first_to_third_edges"] += 1
                elif source_owner == "THIRD_PARTY" and target_owner == "FIRST_PARTY":
                    global_counts["all_third_to_first_edges"] += 1
            if number % 5 == 0 or number == len(routed):
                print(f"OBCBC_STATIC_AUDIT {number}/{len(routed)} boundaries={global_counts['boundary_total']}", flush=True)

    method_total = sum(ownership_totals.values())
    host_total = sum(host_ownership_totals.values())
    critical_total = sum(critical_ownership_totals.values())
    method_coverage = 1.0 - ownership_totals["UNKNOWN"] / max(1, method_total)
    host_coverage = 1.0 - host_ownership_totals["UNKNOWN"] / max(1, host_total)
    critical_coverage = 1.0 - critical_ownership_totals["UNKNOWN"] / max(1, critical_total)
    unknown_critical_ratio = critical_ownership_totals["UNKNOWN"] / max(1, critical_total)
    accountable_attention = global_counts["attention_accountable"] / max(1e-12, global_counts["attention_total"])
    decomposition_ratio = global_counts["both_third_multiboundary_hosts"] / max(1, global_counts["both_third_hosts"])
    context_ratio = global_counts["control_or_resource_context"] / max(1, global_counts["boundary_total"])
    user_folds = sum(fold_metrics[fold]["user_boundary"] > 0 for fold in fold_metrics)
    auto_folds = sum(fold_metrics[fold]["auto_boundary"] > 0 for fold in fold_metrics)
    apk_dominance = dominance(per_apk_boundaries)
    structure_dominance = max(per_structure_boundaries.values(), default=0) / max(1, sum(per_structure_boundaries.values()))

    ownership_audit = {
        "method_total": method_total, "method_ownership": dict(ownership_totals), "method_ownership_coverage": method_coverage,
        "host_total": host_total, "host_ownership": dict(host_ownership_totals), "host_ownership_coverage": host_coverage,
        "critical_host_total": critical_total, "critical_host_ownership": dict(critical_ownership_totals),
        "critical_host_ownership_coverage": critical_coverage, "attention_by_ownership": dict(attention_by_ownership),
        "ownership_uses_labels": False, "raw_identity_emitted": False, "package_namespace_used_as_single_truth": False,
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "ownership_audit.json", ownership_audit)
    context_coverage = {
        "host_with_responsibility_boundary": int(global_counts["hosts_with_boundary"]),
        "first_to_third_edges_all_graphs": int(global_counts["all_first_to_third_edges"]),
        "third_to_first_callback_edges_all_graphs": int(global_counts["all_third_to_first_edges"]),
        "nearest_boundary_instances": int(global_counts["boundary_total"]),
        "both_hosts": int(global_counts["both_hosts"]), "both_third_party_hosts": int(global_counts["both_third_hosts"]),
        "both_third_multiboundary_hosts": int(global_counts["both_third_multiboundary_hosts"]),
        "both_third_decomposition_ratio": decomposition_ratio,
        "accountable_attention_mass": accountable_attention,
        "user_only_boundary_instances": int(global_counts["user_boundary"]), "user_boundary_folds": user_folds,
        "autonomous_only_boundary_instances": int(global_counts["auto_boundary"]), "autonomous_boundary_folds": auto_folds,
        "both_root_boundary_instances": int(global_counts["both_boundary"]),
        "branch_or_resource_context_instances": int(global_counts["control_or_resource_context"]),
        "branch_or_resource_context_ratio": context_ratio,
        "exact_resource_callback_links": int(global_counts["resource_link"]),
        "layout_files": int(global_counts["layout_files"]), "parsed_layout_files": int(global_counts["parsed_layout_files"]),
        "exact_control_dependency_available": False,
        "control_context_definition": "method-local conditional/switch presence is a conservative proxy, not exact control dependence",
        "top10pct_apk_boundary_share": apk_dominance,
        "largest_target_structure_boundary_share": structure_dominance,
        "per_fold": {str(fold): dict(values) for fold, values in sorted(fold_metrics.items())},
        "labels_used": False, "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "context_coverage.json", context_coverage)

    criteria = {
        "host_ownership_coverage>=0.35": host_coverage >= 0.35,
        "critical_host_ownership_coverage>=0.35": critical_coverage >= 0.35,
        "both_third_hosts_with_multiple_boundaries>=20": global_counts["both_third_multiboundary_hosts"] >= 20,
        "both_third_host_decomposition_ratio>=0.10": decomposition_ratio >= 0.10,
        "accountable_attention_mass>=0.20": accountable_attention >= 0.20,
        "user_boundaries>=20_and_3folds": global_counts["user_boundary"] >= 20 and user_folds >= 3,
        "autonomous_boundaries>=20_and_3folds": global_counts["auto_boundary"] >= 20 and auto_folds >= 3,
        "unknown_critical_ratio<=0.65": unknown_critical_ratio <= 0.65,
        "identity_free_ownership": True,
        "branch_or_resource_context>=20_and_ratio>=0.05": global_counts["control_or_resource_context"] >= 20 and context_ratio >= 0.05,
        "top10pct_apk_boundary_share<=0.60": apk_dominance <= 0.60,
    }
    if not criteria["host_ownership_coverage>=0.35"] or not criteria["critical_host_ownership_coverage>=0.35"] or not criteria["unknown_critical_ratio<=0.65"]:
        decision = "OWNERSHIP_UNRELIABLE"
    elif not criteria["branch_or_resource_context>=20_and_ratio>=0.05"]:
        decision = "CONTEXT_UNOBSERVABLE"
    elif all(criteria.values()):
        decision = "STATICALLY_FEASIBLE"
    else:
        decision = "STATIC_ASSET_INSUFFICIENT"
    dump_json(OUTPUT / "feasibility_decision.json", {
        "status": decision, "criteria": criteria, "ownership": ownership_audit,
        "coverage": context_coverage, "labels_used": False, "future_rows_loaded": 0,
    })
    audit_md = f"""# OBCBC Stage-B static feasibility audit

## Decision

**{decision}**

This package is label-free and static-only. It reuses the frozen 200-APK complete app-defined method graphs and never executes APKs.

## Ownership

- Method ownership coverage: {method_coverage:.6f}
- Host ownership coverage: {host_coverage:.6f}
- Critical Host ownership coverage: {critical_coverage:.6f}
- UNKNOWN Critical Host ratio: {unknown_critical_ratio:.6f}
- FIRST/THIRD/UNKNOWN methods: {dict(ownership_totals)}

Raw package, vendor, SDK, library and class names are not emitted and are not model features. Manifest namespace is never sufficient by itself.

## Boundary deconvolution

- FIRST→THIRD edges in complete graphs: {int(global_counts['all_first_to_third_edges'])}
- Sensitive Hosts with a nearest boundary: {int(global_counts['hosts_with_boundary'])}/{int(global_counts['host_total'])}
- BOTH third-party Hosts: {int(global_counts['both_third_hosts'])}
- BOTH third-party Hosts with >=2 first-party boundaries: {int(global_counts['both_third_multiboundary_hosts'])} ({decomposition_ratio:.6f})
- Accountable C43 attention mass: {accountable_attention:.6f}
- USER_ONLY boundary instances/folds: {int(global_counts['user_boundary'])}/{user_folds}
- AUTONOMOUS_ONLY boundary instances/folds: {int(global_counts['auto_boundary'])}/{auto_folds}

## Static context

- Boundary instances: {int(global_counts['boundary_total'])}
- Branch/resource context instances: {int(global_counts['control_or_resource_context'])} ({context_ratio:.6f})
- Exact XML onClick links: {int(global_counts['resource_link'])}
- Layout XML parsed: {int(global_counts['parsed_layout_files'])}/{int(global_counts['layout_files'])}
- Exact control dependence: unavailable in the frozen graph. The reported branch context is an explicit conservative method-local proxy and is not described as control dependence.

## Shortcut risk

- Top 10% APK share of boundary instances: {apk_dominance:.6f}
- Largest target structural fingerprint share: {structure_dominance:.6f}
- No label, future row, package string, SDK/vendor identity or library ID was used for this decision.

## Frozen gates

```json
{json.dumps(criteria, indent=2, sort_keys=True)}
```

If this status is not STATICALLY_FEASIBLE, no source direction probe is authorized. If it is STATICALLY_FEASIBLE, a separate package must implement the predeclared nulls before reading labels.
"""
    (OUTPUT / "static_feasibility_audit.md").write_text(audit_md, encoding="utf-8")
    write_phase("ownership_and_boundaries", "DONE", decision=decision)

    after = hash_manifest(protected, "after")
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after})
    integrity = compare_manifest(before, after)
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS":
        final_status = "BLOCKED"
        reason = "At least one protected formal asset changed during the read-only audit."
    elif decision == "STATICALLY_FEASIBLE":
        final_status = "FEASIBLE_FOR_SOURCE_PROBE"
        reason = "Stage-A novelty and Stage-B frozen label-free static feasibility gates passed. Stage-C null materialization and the source direction probe require a separate package."
    else:
        final_status = decision
        reason = "The predeclared Stage-B static feasibility gate did not pass; no label probe or CARE integration is authorized."
    finalize_later_placeholders("STAGE_B_STATIC_FEASIBILITY", final_status)
    dump_json(OUTPUT / "final_decision.json", {
        "status": final_status, "stopped_at": "STAGE_B_STATIC_FEASIBILITY", "stage_b_status": decision,
        "reason": reason, "formal_integrity": integrity["status"], "static_only": True,
        "labels_used": False, "future_rows_loaded": 0,
    })
    literature_summary = (OUTPUT / "report.md").read_text(encoding="utf-8") if (OUTPUT / "report.md").is_file() else ""
    combined = f"""# CARE-Droid OBCBC Stage-A/B report

## Final status

**{final_status}**

Stopped at: **STAGE_B_STATIC_FEASIBILITY**.

{reason}

- Pure static boundary: PASS
- Labels used in Stage B: false
- Future rows loaded: 0
- Formal asset integrity: {integrity['status']}

""" + audit_md + "\n\n## Frozen Stage-A literature record\n\n" + literature_summary
    (OUTPUT / "report.md").write_text(combined, encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduce OBCBC Stage B\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\n"
        "bash scripts/care-obcbc-source-v1-r1/launch_tmux.sh\n"
        "tail -F outputs/care-obcbc-source-v1.console.log\n"
        "bash scripts/care-obcbc-source-v1-r1/collect_results.sh\n```\n",
        encoding="utf-8",
    )
    print(f"CARE_OBCBC_STAGE_B_STATUS={decision}", flush=True)
    print(f"CARE_OBCBC_FINAL_STATUS={final_status}", flush=True)
    print(f"report={OUTPUT / 'report.md'}", flush=True)
    print(f"decision={OUTPUT / 'final_decision.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        integrity_status = "NOT_AVAILABLE"
        before_path = OUTPUT / "baseline_asset_manifest_before.json"
        if before_path.is_file():
            try:
                before_records = json.loads(before_path.read_text(encoding="utf-8"))["assets"]
                after_records = []
                for record in before_records:
                    path = Path(record["absolute_path"])
                    stat = path.stat()
                    after_records.append({
                        "absolute_path": str(path.resolve()), "purpose": record.get("purpose", "protected asset"),
                        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path),
                    })
                dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after_records})
                integrity = compare_manifest(before_records, after_records)
                dump_json(OUTPUT / "baseline_integrity_check.json", integrity)
                integrity_status = integrity["status"]
            except Exception as integrity_exc:
                dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "BLOCKED", "error": repr(integrity_exc)})
                integrity_status = "BLOCKED"
        dump_json(OUTPUT / "fatal_error.json", {
            "status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=40), "formal_integrity": integrity_status,
            "future_rows_loaded": 0,
        })
        dump_json(OUTPUT / "final_decision.json", {
            "status": "BLOCKED", "stopped_at": "ENGINEERING_ERROR", "reason": str(exc),
            "static_only": True, "labels_used": False, "formal_integrity": integrity_status,
            "future_rows_loaded": 0,
        })
        print(f"CARE_OBCBC_FATAL {type(exc).__name__}: {exc}", flush=True)
        raise
