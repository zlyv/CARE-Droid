#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    OUTPUT, ROLE_NAMES, SEED, discover_assets, dump, endpoint_hosts, file_record, formal_rows,
    load_apk_map, load_attention, norm, read_csv, read_endpoint_rows, read_gzip_json,
    sha256_file, sketch_jaccard, write_csv,
)
from fingerprints import fingerprint_apk, host_direct_invokes

CHECKPOINTS = OUTPUT / "checkpoints/fingerprints"
REVIEW_COUNT_PER_ERROR = 1


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def phase(name: str, status: str, **extra) -> None:
    row = {"phase": name, "status": status, "epoch": int(time.time()), **extra}
    path = OUTPUT / "checkpoints/phases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"P31_LINEAGE_PHASE_{status} name={name}", flush=True)


def manifest_records(assets: dict[str, Path], apk_map: dict[str, Path], wanted: set[str]) -> list[dict[str, Any]]:
    result = [file_record(path, "protected input " + name) for name, path in sorted(assets.items()) if path.is_file()]
    for number, sid in enumerate(sorted(wanted), 1):
        record = file_record(apk_map[sid], "protected 2017 source APK")
        if record["sha256"] != sid:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK content SHA mismatch expected={sid} actual={record['sha256']} path={apk_map[sid]}")
        result.append(record)
        if number % 50 == 0 or number == len(wanted):
            print(f"P31_LINEAGE_ASSET_HASH {number}/{len(wanted)}", flush=True)
    graph_files = sorted(assets["full_graph_dir"].glob("*.json.gz"))
    for number, path in enumerate(graph_files, 1):
        result.append(file_record(path, "protected routed full graph cache"))
        if number % 50 == 0 or number == len(graph_files):
            print(f"P31_LINEAGE_GRAPH_HASH {number}/{len(graph_files)}", flush=True)
    return result


def load_fingerprint(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_fingerprints(selected: list[str], apk_map: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    result, failures = {}, []
    for number, sid in enumerate(selected, 1):
        path = CHECKPOINTS / f"{sid}.json.gz"
        if path.is_file():
            try:
                record = load_fingerprint(path)
                if record.get("status") == "OK" and record.get("sha256") == sid:
                    result[sid] = record
                    if number % 10 == 0 or number == len(selected):
                        print(f"P31_LINEAGE_FINGERPRINT {number}/{len(selected)} ok={len(result)} failed={len(failures)} resume=1", flush=True)
                    continue
            except Exception:
                pass
        try:
            record = fingerprint_apk(apk_map[sid], sid)
            write_gzip_json(path, record)
            result[sid] = record
        except Exception as exc:
            failures.append({"sha256": sid, "error_type": type(exc).__name__, "error": str(exc)})
        if number % 10 == 0 or number == len(selected):
            print(f"P31_LINEAGE_FINGERPRINT {number}/{len(selected)} ok={len(result)} failed={len(failures)}", flush=True)
    return result, failures


def pair_record(left: str, right: str, records: dict[str, dict[str, Any]], pair_type: str,
                group_left: int, group_right: int) -> dict[str, Any]:
    a, b = records[left], records[right]
    row = {
        "pair_type": pair_type, "left_sha256": left, "right_sha256": right,
        "left_group": group_left, "right_group": group_right,
        "same_package_hash": int(a.get("package_sha256") == b.get("package_sha256") and bool(a.get("package_sha256"))),
        "shared_certificate": int(bool(set(a.get("certificate_sha256", [])) & set(b.get("certificate_sha256", [])))),
        "same_dex_hash_list": int(a.get("dex_sha256") == b.get("dex_sha256")),
        "method_count_ratio": min(a["method_count"], b["method_count"]) / max(1, max(a["method_count"], b["method_count"])),
    }
    for name in ("manifest", "classes", "method_exact", "method_shape", "internal_edges", "external_api", "zip_layout"):
        row[name + "_jaccard"] = sketch_jaccard(a["sketches"][name], b["sketches"][name])
        row[name + "_exact"] = int(a["sketches"][name]["set_sha256"] == b["sketches"][name]["set_sha256"])
    return row


def summarize_pairs(rows: list[dict[str, Any]], pair_type: str) -> dict[str, Any]:
    block = [row for row in rows if row["pair_type"] == pair_type]
    result = {"pair_type": pair_type, "pairs": len(block)}
    fields = [key for key in block[0] if key.endswith("_jaccard")] if block else []
    for field in fields:
        values = np.asarray([float(row[field]) for row in block])
        result[field + "_mean"] = float(np.mean(values))
        result[field + "_median"] = float(np.median(values))
        result[field + "_p10"] = float(np.quantile(values, .1))
        result[field + "_p90"] = float(np.quantile(values, .9))
    for field in ("same_package_hash", "shared_certificate", "same_dex_hash_list"):
        result[field + "_rate"] = float(np.mean([row[field] for row in block])) if block else None
    return result


def bootstrap_pair_difference(within: list[dict[str, Any]], null: list[dict[str, Any]], field: str, seed: int) -> dict[str, Any]:
    a = np.asarray([row[field] for row in within], float)
    b = np.asarray([row[field] for row in null], float)
    if not len(a) or not len(b):
        return {"field": field, "status": "INSUFFICIENT"}
    rng = np.random.RandomState(seed)
    differences = []
    for _ in range(2000):
        differences.append(float(np.mean(rng.choice(a, len(a), replace=True)) - np.mean(rng.choice(b, len(b), replace=True))))
    return {"field": field, "within_mean": float(np.mean(a)), "matched_null_mean": float(np.mean(b)),
            "difference": float(np.mean(a) - np.mean(b)), "ci_low": float(np.quantile(differences, .025)),
            "ci_high": float(np.quantile(differences, .975)), "note": "descriptive pair bootstrap; pairs are not independent"}


def lineage_analysis(assignments: list[dict[str, str]], formal: list[dict[str, Any]],
                     fingerprints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group_by = {norm(row["sha256"]): int(row["group"]) for row in assignments}
    size_by = {int(row["group"]): int(row["group_size"]) for row in assignments}
    label_by = {row["sha256"]: row["label"] for row in formal}
    members: dict[int, list[str]] = defaultdict(list)
    for sid in fingerprints:
        members[group_by[sid]].append(sid)
    within = []
    for group, ids in sorted(members.items()):
        if size_by[group] <= 1:
            continue
        for left, right in itertools.combinations(sorted(ids), 2):
            within.append(pair_record(left, right, fingerprints, "WITHIN_P31_GROUP", group, group))
    rng = np.random.RandomState(SEED)
    ids = sorted(fingerprints)
    null = []
    seen_null = set()
    attempts = 0
    target = len(within)
    while len(null) < target and attempts < target * 100 + 10000:
        attempts += 1
        left, right = rng.choice(ids, 2, replace=False)
        pair_key = tuple(sorted((str(left), str(right))))
        if pair_key in seen_null or group_by[left] == group_by[right] or label_by[left] != label_by[right]:
            continue
        a, b = fingerprints[left], fingerprints[right]
        ratio = min(a["method_count"], b["method_count"]) / max(1, max(a["method_count"], b["method_count"]))
        if ratio < .67:
            continue
        seen_null.add(pair_key)
        null.append(pair_record(left, right, fingerprints, "MATCHED_DIFFERENT_P31_GROUP", group_by[left], group_by[right]))
    pair_rows = within + null
    write_csv(OUTPUT / "lineage_pair_audit.csv", pair_rows)
    summaries = [summarize_pairs(pair_rows, value) for value in ("WITHIN_P31_GROUP", "MATCHED_DIFFERENT_P31_GROUP")]
    boot = [bootstrap_pair_difference(within, null, field, SEED + ordinal) for ordinal, field in enumerate(
        ("manifest_jaccard", "method_exact_jaccard", "method_shape_jaccard", "internal_edges_jaccard", "external_api_jaccard"))]
    largest_group = max(size_by, key=size_by.get)
    largest_rows = [row for row in within if row["left_group"] == largest_group]
    largest_summary = summarize_pairs(largest_rows, "WITHIN_P31_GROUP") if largest_rows else {"pairs": 0}
    largest_summary["group"] = largest_group
    largest_summary["size"] = size_by[largest_group]
    within_shape = summaries[0].get("method_shape_jaccard_median", 0.0)
    null_shape = summaries[1].get("method_shape_jaccard_median", 0.0)
    if largest_summary.get("same_dex_hash_list_rate", 0) >= .5 or largest_summary.get("method_shape_jaccard_median", 0) >= .9:
        decision = "STRONG_CODE_LINEAGE_EVIDENCE"
    elif within_shape >= null_shape + .15:
        decision = "MODERATE_CODE_LINEAGE_EVIDENCE"
    else:
        decision = "P31_EQUIVALENCE_NOT_EXPLAINED_BY_CODE_LINEAGE"
    result = {"decision": decision, "within": summaries[0], "matched_null": summaries[1],
              "paired_descriptive_bootstrap": boot, "largest_group": largest_summary,
              "future_rows_loaded": 0, "labels_used_only_after_label_free_fingerprints": True,
              "warning": "P31-equivalent does not imply exact APK duplicate or malware family without independent family labels."}
    dump(OUTPUT / "lineage_summary.json", result)
    return result


def accuracy_row(name: str, ids: list[str], formal_by: dict[str, dict[str, Any]], group_prob: dict[str, float]) -> dict[str, Any]:
    y = np.asarray([formal_by[sid]["label"] for sid in ids], int)
    formal_p = np.asarray([formal_by[sid]["stage1_probability"] for sid in ids], float)
    group_p = np.asarray([group_prob[sid] for sid in ids], float)
    return {"stratum": name, "n": len(ids), "malware_rate": float(np.mean(y)) if len(y) else None,
            "formal_accuracy": float(np.mean((formal_p >= .5) == y)) if len(y) else None,
            "group_safe_accuracy": float(np.mean((group_p >= .5) == y)) if len(y) else None,
            "formal_errors": int(np.sum((formal_p >= .5) != y)), "group_safe_errors": int(np.sum((group_p >= .5) != y))}


def exposure_audit(assignments: list[dict[str, str]], formal: list[dict[str, Any]], group_predictions_path: Path) -> dict[str, Any]:
    formal_by = {row["sha256"]: row for row in formal}
    prediction_rows = read_csv(group_predictions_path)
    group_prob = {norm(row["sha256"]): float(row["GROUP_RF_PLATT__prob"]) for row in prediction_rows}
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in assignments:
        groups[int(row["group"])].append(row)
    largest = max(groups, key=lambda value: len(groups[value]))
    singleton, multi, crossfold, largest_ids = [], [], [], []
    detail = []
    for group, rows in groups.items():
        ids = [norm(row["sha256"]) for row in rows]
        folds = {int(row["original_fold"]) for row in rows}
        if len(rows) == 1:
            singleton.extend(ids)
        else:
            multi.extend(ids)
            if len(folds) > 1:
                crossfold.extend(ids)
        if group == largest:
            largest_ids.extend(ids)
        detail.append({"group": group, "size": len(rows), "original_fold_count": len(folds),
                       "label_purity": max(np.mean([formal_by[sid]["label"] == 0 for sid in ids]),
                                           np.mean([formal_by[sid]["label"] == 1 for sid in ids])),
                       "formal_accuracy": np.mean([(formal_by[sid]["stage1_probability"] >= .5) == formal_by[sid]["label"] for sid in ids]),
                       "group_safe_accuracy": np.mean([(group_prob[sid] >= .5) == formal_by[sid]["label"] for sid in ids])})
    write_csv(OUTPUT / "p31_group_performance_audit.csv", sorted(detail, key=lambda row: row["size"], reverse=True))
    rows = [accuracy_row("ALL", list(formal_by), formal_by, group_prob),
            accuracy_row("SINGLETON_GROUPS", singleton, formal_by, group_prob),
            accuracy_row("MULTI_SAMPLE_GROUPS", multi, formal_by, group_prob),
            accuracy_row("CROSS_FOLD_EQUIVALENT_EXPOSURE", crossfold, formal_by, group_prob),
            accuracy_row("LARGEST_P31_EQUIVALENT_GROUP", largest_ids, formal_by, group_prob)]
    write_csv(OUTPUT / "p31_exposure_performance.csv", rows)
    result = {"largest_group": largest, "largest_group_size": len(largest_ids),
              "cross_fold_exposed_samples": len(crossfold), "strata": rows,
              "interpretation_boundary": "This quantifies representation-level overlap. It does not by itself prove byte-level duplicates or family leakage."}
    dump(OUTPUT / "p31_exposure_summary.json", result)
    return result


def choose_review_samples(formal: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    errors = [row for row in formal if row["routed"] and ((row["stage1_probability"] >= .5) != bool(row["label"]))]
    correct = [row for row in formal if row["routed"] and ((row["stage1_probability"] >= .5) == bool(row["label"]))]
    if len(errors) != 45:
        raise RuntimeError(f"formal routed error count changed: {len(errors)}")
    available = set(row["sha256"] for row in correct)
    selected_controls = []
    for error in sorted(errors, key=lambda row: (row["fold"], row["sha256"])):
        candidates = [row for row in correct if row["sha256"] in available and row["fold"] == error["fold"] and
                      ((row["stage1_probability"] >= .5) == (error["stage1_probability"] >= .5))]
        if not candidates:
            candidates = [row for row in correct if row["sha256"] in available and row["fold"] == error["fold"]]
        if not candidates:
            candidates = [row for row in correct if row["sha256"] in available and
                          ((row["stage1_probability"] >= .5) == (error["stage1_probability"] >= .5))]
        if not candidates:
            candidates = [row for row in correct if row["sha256"] in available]
        candidates.sort(key=lambda row: (abs(row["stage1_probability"] - error["stage1_probability"]),
                                         abs(row["router_score"] - error["router_score"]), row["sha256"]))
        for candidate in candidates[:REVIEW_COUNT_PER_ERROR]:
            selected_controls.append(candidate)
            available.remove(candidate["sha256"])
    if len(selected_controls) != len(errors) * REVIEW_COUNT_PER_ERROR:
        raise RuntimeError(f"matched routed-correct control selection incomplete {len(selected_controls)}/{len(errors)}")
    selected = errors + selected_controls
    review_ids = {row["sha256"]: "R" + hashlib.sha256((str(SEED) + row["sha256"]).encode()).hexdigest()[:10].upper() for row in selected}
    return selected, review_ids


def graph_context(graph: dict[str, Any], node_index: int) -> tuple[list[str], list[str]]:
    methods = graph["methods"]
    callers, callees = [], []
    for source, target in graph["edges"]:
        source, target = int(source), int(target)
        if target == node_index and len(callers) < 12:
            callers.append(methods[source]["signature"])
        if source == node_index and len(callees) < 12:
            callees.append(methods[target]["signature"])
    return sorted(set(callers))[:8], sorted(set(callees))[:8]


def host_review_audit(assets: dict[str, Path], formal: list[dict[str, Any]], apk_map: dict[str, Path]) -> dict[str, Any]:
    selected, review_ids = choose_review_samples(formal)
    wanted = {row["sha256"] for row in selected}
    endpoints = read_endpoint_rows(assets["endpoint"], wanted)
    attention = load_attention(assets["attention"])
    host_rows, sample_rows, key_rows = [], [], []
    sample_complete = 0
    top_total = top_exact = roles_present = external_present = context_present = 0
    for number, row in enumerate(sorted(selected, key=lambda item: review_ids[item["sha256"]]), 1):
        sid = row["sha256"]
        review_id = review_ids[sid]
        graph_path = assets["full_graph_dir"] / f"{sid}.json.gz"
        payload = endpoints.get(sid)
        attention_info = attention.get(sid)
        graph = read_gzip_json(graph_path) if graph_path.is_file() else None
        hosts = endpoint_hosts(payload) if payload else []
        att = np.asarray(attention_info["attention"], float) if attention_info else np.asarray([])
        aligned = min(len(hosts), len(att))
        chosen = []
        if aligned:
            critical = int(attention_info["critical"])
            if 0 <= critical < aligned:
                chosen.append(critical)
            for index in np.argsort(-att[:aligned]):
                if int(index) not in chosen:
                    chosen.append(int(index))
                if len(chosen) >= 3:
                    break
        method_by_id = {int(method["node_id"]): index for index, method in enumerate(graph["methods"])} if graph else {}
        wanted_signatures = set()
        provisional = []
        for rank, host_order in enumerate(chosen, 1):
            top_total += 1
            host = hosts[host_order]
            node_index = method_by_id.get(int(host["node_id"]))
            if node_index is not None:
                top_exact += 1
                signature = graph["methods"][node_index]["signature"]
                wanted_signatures.add(signature)
                callers, callees = graph_context(graph, node_index)
            else:
                signature, callers, callees = "", [], []
            if host["roles"]:
                roles_present += 1
            if callers or callees:
                context_present += 1
            provisional.append((rank, host_order, host, node_index, signature, callers, callees))
        invokes = {}
        if wanted_signatures:
            try:
                invokes = host_direct_invokes(apk_map[sid], wanted_signatures)
            except Exception:
                invokes = {}
        for rank, host_order, host, node_index, signature, callers, callees in provisional:
            direct_invokes = invokes.get(norm(signature), [])
            if direct_invokes:
                external_present += 1
            host_rows.append({
                "review_id": review_id, "stage1_prediction": int(row["stage1_probability"] >= .5),
                "host_rank": rank, "critical": int(attention_info and host_order == int(attention_info["critical"])),
                "attention": float(att[host_order]) if host_order < len(att) else None,
                "roles": ";".join(ROLE_NAMES[index] for index in host["roles"]),
                "host_method": signature, "direct_invokes": " | ".join(direct_invokes[:30]),
                "app_callers": " | ".join(callers), "app_callees": " | ".join(callees),
                "exact_host_graph_alignment": int(node_index is not None),
            })
        complete = bool(chosen and all(item[3] is not None for item in provisional) and all(item[2]["roles"] for item in provisional))
        sample_complete += int(complete)
        sample_rows.append({
            "review_id": review_id, "stage1_prediction": int(row["stage1_probability"] >= .5),
            "stage1_probability": row["stage1_probability"], "router_score": row["router_score"],
            "top_host_count": len(chosen), "evidence_complete": int(complete),
            "reviewer_id": "", "sample_direction": "", "confidence": "", "primary_evidence": "",
            "ambiguous_reason": "", "notes": "",
        })
        stage1_correct = int((row["stage1_probability"] >= .5) == bool(row["label"]))
        final_correct = int((row["final_probability"] >= .5) == bool(row["label"]))
        key_rows.append({
            "review_id": review_id, "sha256": sid, "fold": row["fold"], "label": row["label"],
            "sample_kind": "ROUTED_STAGE1_ERROR" if not stage1_correct else "MATCHED_ROUTED_CORRECT",
            "truth_direction": "MALWARE_SUPPORT" if row["label"] == 1 else "BENIGN_SUPPORT",
            "stage1_correct": stage1_correct, "formal_final_correct": final_correct,
            "formal_fixed": int(not stage1_correct and final_correct), "formal_broken": int(stage1_correct and not final_correct),
        })
        if number % 10 == 0 or number == len(selected):
            print(f"HOST_REVIEW_PACKET {number}/{len(selected)}", flush=True)
    write_csv(OUTPUT / "host_review_items_blinded.csv", host_rows)
    write_csv(OUTPUT / "host_review_form_blinded.csv", sample_rows)
    write_csv(OUTPUT / "host_review_key_sealed.csv", key_rows)
    selected_count = len(selected)
    result = {
        "selected_samples": selected_count, "routed_stage1_errors": sum(row["sample_kind"] == "ROUTED_STAGE1_ERROR" for row in key_rows),
        "matched_routed_correct": sum(row["sample_kind"] == "MATCHED_ROUTED_CORRECT" for row in key_rows),
        "sample_evidence_complete_rate": sample_complete / max(1, selected_count),
        "top_host_exact_alignment_rate": top_exact / max(1, top_total),
        "top_host_role_coverage": roles_present / max(1, top_total),
        "top_host_direct_invoke_coverage": external_present / max(1, top_total),
        "top_host_app_context_coverage": context_present / max(1, top_total),
        "selection": "all 45 formal routed Stage-I errors plus one same-fold, same-Stage-I-side nearest-probability routed-correct control per error",
        "labels_hidden_from_review_forms": True, "future_rows_loaded": 0,
    }
    criteria = {
        "sample_evidence_complete_rate>=0.80": result["sample_evidence_complete_rate"] >= .80,
        "top_host_exact_alignment_rate>=0.90": result["top_host_exact_alignment_rate"] >= .90,
        "top_host_role_coverage>=0.90": result["top_host_role_coverage"] >= .90,
        "top_host_context_or_invoke_coverage>=0.50": max(result["top_host_direct_invoke_coverage"], result["top_host_app_context_coverage"]) >= .50,
    }
    result["criteria"] = criteria
    result["decision"] = "READY_FOR_BLINDED_HOST_DIRECTION_REVIEW" if all(criteria.values()) else "HOST_DIRECTION_REVIEW_NOT_FEASIBLE"
    dump(OUTPUT / "host_annotation_feasibility.json", result)
    return result


def integrity(before: list[dict[str, Any]]) -> str:
    after = []
    for number, record in enumerate(before, 1):
        after.append(file_record(Path(record["absolute_path"]), record["purpose"]))
        if number % 50 == 0 or number == len(before):
            print(f"P31_LINEAGE_ASSET_REHASH {number}/{len(before)}", flush=True)
    dump(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after})
    before_by = {row["absolute_path"]: row for row in before}
    after_by = {row["absolute_path"]: row for row in after}
    changed = [{"path": path, "before": before_by.get(path), "after": after_by.get(path)} for path in sorted(set(before_by) | set(after_by)) if before_by.get(path) != after_by.get(path)]
    status = "PASS" if not changed else "FAIL"
    dump(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(after), "changed": changed})
    return status


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    assets = discover_assets()
    formal = formal_rows(assets["formal"])
    assignments = read_csv(assets["group_assignments"])
    size_by = {int(row["group"]): int(row["group_size"]) for row in assignments}
    multi = {norm(row["sha256"]) for row in assignments if size_by[int(row["group"])] > 1}
    routed = {row["sha256"] for row in formal if row["routed"]}
    wanted = multi | routed
    apk_map, _ = load_apk_map(wanted)
    if not wanted <= apk_map.keys():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths missing {len(wanted - apk_map.keys())}/{len(wanted)}")

    phase("asset_manifest_before", "START")
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    if before_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))["assets"]
    else:
        before = manifest_records(assets, apk_map, wanted)
        dump(before_path, {"git_used": False, "assets": before})
    phase("asset_manifest_before", "DONE", assets=len(before))

    protocol = {
        "version": "care-p31-lineage-host-audit-v1", "source_year": 2017, "source_rows": 799,
        "future_rows_loaded": 0, "new_classifier_trained": False, "formal_pipeline_modified": False,
        "lineage_scope": "all members of P31 multi-sample equivalence groups",
        "fingerprints": ["manifest", "classes", "method_exact", "method_shape_opcode", "internal_edges", "external_api", "zip_layout", "certificate", "DEX"],
        "fingerprint_features_are_label_free": True, "pair_control": "same-label different-P31-group and method-count ratio >=0.67",
        "host_review_scope": "45 routed Stage-I errors plus 45 matched routed-correct controls",
        "host_review_is_blinded": True, "host_review_does_not_train_or_change_CARE": True,
        "evidence_boundary": "code similarity supports lineage/template evidence, not a definitive malware-family identity",
    }
    dump(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "protocol_hash.txt").write_text(sha256_file(OUTPUT / "protocol.json") + "\n", encoding="utf-8")
    annotation_source = Path(__file__).resolve().parent / "annotation_protocol.md"
    (OUTPUT / "annotation_protocol.md").write_text(annotation_source.read_text(encoding="utf-8"), encoding="utf-8")

    phase("lineage_fingerprints", "START", samples=len(multi))
    fingerprints, failures = build_fingerprints(sorted(multi), apk_map)
    write_csv(OUTPUT / "fingerprint_failures.csv", failures)
    coverage = len(fingerprints) / max(1, len(multi))
    if coverage < .90:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: fingerprint coverage {coverage:.6f} < 0.90")
    phase("lineage_fingerprints", "DONE", ok=len(fingerprints), failed=len(failures))

    phase("lineage_analysis", "START")
    lineage = lineage_analysis(assignments, formal, fingerprints)
    exposure = exposure_audit(assignments, formal, assets["group_predictions"])
    phase("lineage_analysis", "DONE", decision=lineage["decision"])

    phase("host_review_packet", "START")
    review = host_review_audit(assets, formal, apk_map)
    phase("host_review_packet", "DONE", decision=review["decision"])

    phase("integrity", "START")
    integrity_status = integrity(before)
    phase("integrity", "DONE", status_result=integrity_status)
    if integrity_status != "PASS":
        final = "BLOCKED"
        reason = "A protected formal asset changed during this read-only audit."
    elif review["decision"] == "READY_FOR_BLINDED_HOST_DIRECTION_REVIEW":
        final = "LINEAGE_AUDIT_COMPLETE_HOST_REVIEW_READY"
        reason = "Representation-equivalent lineage was quantified and the blinded Host direction packet passed its evidence-completeness gate. No new model was trained."
    else:
        final = "LINEAGE_AUDIT_COMPLETE_HOST_REVIEW_NOT_FEASIBLE"
        reason = "Lineage was quantified, but the existing static assets do not expose enough Critical-Host context for a defensible blinded direction review."
    decision = {"status": final, "reason": reason, "lineage_decision": lineage["decision"],
                "host_review_decision": review["decision"], "formal_integrity": integrity_status,
                "future_rows_loaded": 0, "new_model_trained": False, "end_to_end_correction_run": False}
    dump(OUTPUT / "final_decision.json", decision)
    report = f"""# CARE-Droid P31 lineage and Host-direction annotatability audit

## Final decision

**{final}**

{reason}

## P31 representation-equivalent lineage

- Multi-sample P31 group members fingerprinted: {len(fingerprints)}/{len(multi)}.
- Largest P31-equivalent group: {exposure['largest_group_size']} APKs.
- Samples exposed to a representation-equivalent group member across original folds: {exposure['cross_fold_exposed_samples']}.
- Lineage evidence decision: **{lineage['decision']}**.
- Within-group code similarity is compared with a same-label, method-count-matched, different-group null. P31 equivalence is not called an exact duplicate or family identity without independent evidence.

## Host direction review readiness

- Selected: {review['selected_samples']} samples ({review['routed_stage1_errors']} routed errors + {review['matched_routed_correct']} matched routed-correct controls).
- Evidence-complete sample rate: {review['sample_evidence_complete_rate']:.4f}.
- Top-Host exact full-graph alignment: {review['top_host_exact_alignment_rate']:.4f}.
- Role coverage: {review['top_host_role_coverage']:.4f}.
- Direct-invoke coverage: {review['top_host_direct_invoke_coverage']:.4f}.
- App caller/callee context coverage: {review['top_host_app_context_coverage']:.4f}.
- Decision: **{review['decision']}**.

The blinded form contains no labels or fixed/broken outcome. `host_review_key_sealed.csv` must remain unopened until reviewer decisions are frozen. This run does not claim that manual review has already established direction sufficiency.

## Safety and protocol

- No future rows loaded.
- No classifier, gate, correction head or threshold was trained or changed.
- Formal asset integrity: **{integrity_status}**.
- No Git command was used.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduce\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\n"
        "bash scripts/care-p31-lineage-host-audit-v1/launch_tmux.sh\n"
        "tail -F outputs/care-p31-lineage-host-audit-v1.console.log\n"
        "bash scripts/care-p31-lineage-host-audit-v1/collect_results.sh\n```\n", encoding="utf-8")
    print(f"CARE_P31_LINEAGE_HOST_AUDIT_FINAL={final}", flush=True)
    print(f"lineage={lineage['decision']} host_review={review['decision']} integrity={integrity_status}", flush=True)
    print(f"report={OUTPUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        dump(OUTPUT / "fatal_error.json", {"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc), "future_rows_loaded": 0})
        dump(OUTPUT / "final_decision.json", {"status": "BLOCKED", "reason": str(exc), "future_rows_loaded": 0})
        print(f"CARE_P31_LINEAGE_HOST_AUDIT_FATAL {type(exc).__name__}: {exc}", flush=True)
        raise
