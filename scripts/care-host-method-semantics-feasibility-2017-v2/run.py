#!/usr/bin/env python3
"""Run the 2017 CARE-Droid Host method-level behavior-semantics feasibility audit.

Inputs: the frozen 2017 SHA whitelist, endpoint-centered Host cache, Compact43/Relation31
cache, Manifest135 autonomous-router OOF predictions, and original APKs. Outputs: new
per-APK resumable extraction caches plus coverage, alignment, independence, and final
input-feasibility reports. No classifier is trained and no existing CARE asset is changed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from common import *
from semantic_core import analyze_method

ASSETS = {
    "lineage": Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv"),
    "endpoint": Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
    "features": PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
    "router": PROJECT / "outputs/care_manifest_rf_autonomous_error_router_v1/per_apk_predictions.csv.gz",
    "apk_index": PROJECT / "outputs/care-ccbc-source-v3/apk_paths.json",
    "mapping": PROJECT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml",
}
CONFIG = {
    "schema": SCHEMA, "year": 2017, "expected_apks": 799, "expected_benign": 400,
    "expected_malware": 399, "host_definition": "direct caller of at least one frozen mapped sensitive endpoint",
    "host_alignment": "canonical DEX method signature SHA256 first uint64; exact frozen stable ID only",
    "s1": "finite constant argument categories; no raw strings",
    "s2": "conservative same-method return use over Dalvik registers",
    "s3": "same-basic-block direct/local-copy sensitive-result-to-sensitive-argument def-use",
    "unknown_policy": "unknown is explicit and never treated as absence",
    "maximum_method_instructions": 4096, "per_apk_timeout_seconds": 900,
    "workers": 2, "labels_used_for_feature_selection": False, "future_years_loaded": False,
    "training_performed": False, "payload_to_model_role": ROLE_MAP,
    "v2_correction": "unrelated move-result destination kills any older sensitive-result register token",
    "feasibility_thresholds": {"host_alignment_rate": 0.90, "minimum_independent_apks": 100,
                                "minimum_routed_union_apks": 30, "minimum_collision_variant_apks": 50},
}


def load_lineage():
    frame = pd.read_csv(ASSETS["lineage"], dtype=str)
    needed = {"sha256", "absolute_path", "metadata_year", "metadata_label"}
    if not needed.issubset(frame.columns):
        raise RuntimeError(f"lineage columns missing: {needed-set(frame.columns)}")
    frame["sha256"] = frame.sha256.str.lower()
    if len(frame) != 799 or frame.sha256.nunique() != 799 or set(frame.metadata_year.astype(int)) != {2017}:
        raise RuntimeError("2017 whitelist is not exactly 799 unique SHA256 values")
    labels = frame.metadata_label.str.casefold()
    benign = int(labels.isin(["0", "benign", "goodware"]).sum())
    malware = int(labels.isin(["1", "malware", "malicious"]).sum())
    if (benign, malware) != (400, 399):
        raise RuntimeError(f"unexpected 2017 label counts: benign={benign}, malware={malware}")
    missing = [p for p in frame.absolute_path if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"BLOCKED: {len(missing)} original APK paths are missing")
    return frame


def load_endpoints(wanted):
    rows = {}
    with gzip.open(ASSETS["endpoint"], "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sha256", "")).lower()
            # Mandatory year boundary: discard before payload access.
            if sid not in wanted:
                continue
            if sid in rows:
                raise RuntimeError(f"duplicate endpoint cache SHA: {sid}")
            if row.get("status") != "OK" or str(row.get("metadata_year")) != "2017":
                raise RuntimeError(f"invalid 2017 endpoint row: {sid}")
            rows[sid] = row
    if set(rows) != set(wanted):
        raise RuntimeError(f"endpoint whitelist coverage {len(rows)}/{len(wanted)}")
    return rows


def frozen_view(record):
    payload = record["payload"]
    ids = list(map(int, payload["node_ids"]))
    roles = payload["node_role_ids"]
    base = np.asarray(payload["node_base_features"], dtype=np.float32)
    if base.size == 0:
        base = np.empty((0, 15), dtype=np.float32)
    edges = payload["edge_index"]
    if len(ids) != len(set(ids)) or base.shape != (len(ids), 15) or len(edges) != 2:
        raise RuntimeError("frozen endpoint schema mismatch")
    model_roles = [sorted({ROLE_MAP[int(v)] for v in (values or [])}) for values in roles]
    endpoint_nodes = set(map(int, np.flatnonzero(base[:, 4] > .5)))
    targets = defaultdict(dict)
    for left, right in zip(edges[0], edges[1]):
        left, right = int(left), int(right)
        if right in endpoint_nodes and model_roles[right]:
            targets[left][str(ids[right])] = model_roles[right]
    host_nodes = sorted(node for node, values in targets.items() if values)
    return {"sha256": record["sha256"], "host_ids": [str(ids[n]) for n in host_nodes],
            "host_nodes": host_nodes, "pairs_by_host": {str(ids[n]): targets[n] for n in host_nodes},
            "summary": record.get("summary", {})}


def router_rows(wanted):
    frame = pd.read_csv(ASSETS["router"], dtype={"sha256": str})
    frame.sha256 = frame.sha256.str.lower()
    frame = frame[(frame.year == 2017) & (frame.strategy == "AUTO_ERROR_LR") & frame.sha256.isin(wanted)].copy()
    seeds = sorted(frame.manifest_seed.unique().tolist())
    if len(seeds) != 3 or len(frame) != 3 * 799:
        raise RuntimeError(f"formal router OOF coverage mismatch: rows={len(frame)} seeds={seeds}")
    for seed in seeds:
        part = frame[frame.manifest_seed == seed]
        if len(part) != 799 or part.sha256.nunique() != 799:
            raise RuntimeError(f"router seed coverage mismatch: {seed}")
    folds = frame.groupby("sha256").fold.nunique()
    if int(folds.max()) != 1:
        raise RuntimeError("outer fold differs across Manifest seeds")
    return frame[["manifest_seed", "fold", "sha256", "routed", "auto_error_probability"]]


def preflight():
    sys.dont_write_bytecode = True
    for name, path in ASSETS.items():
        if not path.is_file():
            raise RuntimeError(f"BLOCKED: missing {name}: {path}")
    lineage = load_lineage()
    wanted = set(lineage.sha256)
    routes = router_rows(wanted)
    endpoints = load_endpoints(wanted)
    npz = np.load(ASSETS["features"], allow_pickle=False)
    if list(npz["sha256"]) != list(lineage.sha256):
        # Membership is strict; order is checked against endpoint rows below.
        if set(npz["sha256"].tolist()) != wanted:
            raise RuntimeError("Host feature cache membership differs from 2017 whitelist")
    if npz["host_values"].shape[1] != 105 or int(npz["real_host_count"].sum()) != 95314:
        raise RuntimeError("frozen Host feature shape/count mismatch")
    config_path = OUTPUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != CONFIG:
        raise RuntimeError("existing output directory belongs to a different configuration")
    atomic_json(config_path, CONFIG)
    feature_index = {str(s): i for i, s in enumerate(npz["sha256"].tolist())}
    host_total = 0
    zero_host = 0
    for pos, sid in enumerate(sorted(wanted), 1):
        view = frozen_view(endpoints[sid])
        idx = feature_index[sid]
        lo, hi = int(npz["host_offsets"][idx]), int(npz["host_offsets"][idx + 1])
        real = int(npz["real_host_count"][idx])
        if len(view["host_ids"]) != real:
            raise RuntimeError(f"frozen Host count/order alignment failed: {sid} endpoint={len(view['host_ids'])} features={real}")
        values = npz["host_values"][lo:lo + real]
        for row_no, node in enumerate(view["host_nodes"]):
            expected = np.zeros(31, dtype=np.float32)
            for role_values in view["pairs_by_host"][view["host_ids"][row_no]].values():
                expected[list(map(int, role_values))] = 1.0
            if not np.array_equal(values[row_no, :31], expected):
                raise RuntimeError(f"SelfRole31 alignment failed: {sid} host-row={row_no}")
        atomic_json(OUTPUT / "frozen" / f"{sid}.json", view)
        host_total += real
        zero_host += int(real == 0)
        if pos % 100 == 0:
            print(f"PREFLIGHT_HOST_ALIGNMENT {pos}/799", flush=True)
    # Hash the small/central frozen assets before execution. Original APKs are verified by SHA during each scan.
    manifest = [file_record(path, name) for name, path in ASSETS.items() if name != "apk_index"]
    atomic_json(OUTPUT / "asset_manifest.json", {"status": "BEFORE_RUN", "protected_assets": manifest,
                "apk_whitelist_count": 799, "apk_content_sha_verified_during_scan": True})
    write_csv(OUTPUT / "router_2017_oof_membership.csv.gz", routes.to_dict("records"), compress=True)
    atomic_json(OUTPUT / "alignment_audit.json", {"status": "PREFLIGHT_PASS", "whitelist_apks": 799,
                "benign": 400, "malware": 399, "frozen_hosts": host_total, "zero_host_apks": zero_host,
                "host_identity": CONFIG["host_alignment"], "host_order": "sorted frozen endpoint node index",
                "multidex_policy": "all DEX units; duplicate canonical stable ID is ambiguous",
                "future_rows_loaded": 0, "labels_used_for_feature_selection": False})
    atomic_json(OUTPUT / "run_status.json", {"status": "PREPARED", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "completed_apks": len(list((OUTPUT / "cache").glob("*.json.gz"))), "total_apks": 799})
    print(f"PREFLIGHT_PASS apks=799 hosts={host_total} zero_host_apks={zero_host}", flush=True)


def scan_one(sid, apk_path, frozen_path, output_path):
    if sha256(apk_path) != sid:
        raise RuntimeError(f"APK SHA256 mismatch: {apk_path}")
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    if not frozen["host_ids"]:
        atomic_json_gz(output_path, {"schema": SCHEMA, "sha256": sid, "status": "OK_NO_HOST", "dex_count": 0,
                                         "hosts": [], "apk_sha256_verified": True, "wall_seconds": 0.0})
        return
    sys.path.insert(0, str(PROJECT / "scripts/care-local-control-constraint-feasibility-v2"))
    from dex_adapter import decode_method
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    start = time.monotonic()
    apk = APK(str(apk_path))
    host_set = set(frozen["host_ids"])
    methods = defaultdict(list)
    dex_count = 0
    for dex_name in sorted(apk.get_dex_names()):
        dex_count += 1
        dex = DEX(apk.get_file(dex_name))
        for cls in dex.get_classes():
            class_name = str(cls.get_name())
            for method in cls.get_methods():
                signature = canonical(class_name + "->" + str(method.get_name()) + str(method.get_descriptor()))
                hid = str(stable_id(signature))
                if hid in host_set:
                    methods[hid].append((method, dex_name))
    host_rows = []
    for hid in frozen["host_ids"]:
        candidates = methods.get(hid, [])
        base = {"host_stable_id": hid, "parse_status": "METHOD_ID_NOT_FOUND", "instruction_count": 0,
                "sensitive_event_count": 0, "s1_counts": {}, "s2_counts": {}, "s3_dependencies": [],
                "unknown_reasons": ["HOST_ALIGNMENT_UNKNOWN"]}
        if len(candidates) > 1:
            base["parse_status"] = "AMBIGUOUS_STABLE_ID"
            base["unknown_reasons"] = ["HOST_ALIGNMENT_UNKNOWN", "MULTIDEX_OR_CANONICAL_DUPLICATE"]
            host_rows.append(base)
            continue
        if not candidates:
            host_rows.append(base)
            continue
        method, dex_name = candidates[0]
        try:
            instructions, meta = decode_method(method)
            analysis = analyze_method(instructions, frozen["pairs_by_host"][hid],
                                      tries_size=int(meta.get("tries_size", 0)),
                                      maximum_instructions=CONFIG["maximum_method_instructions"])
            s1 = Counter(category for event in analysis["events"] for category in event["s1_categories"])
            s2 = Counter(event["s2_primary"] for event in analysis["events"])
            aligned_targets = {event["endpoint_node_id"] for event in analysis["events"]}
            expected_targets = set(frozen["pairs_by_host"][hid])
            reasons = list(analysis["unknown_reasons"])
            if not expected_targets.issubset(aligned_targets):
                reasons.append("FROZEN_ENDPOINT_INVOKE_NOT_RECOVERED")
            base.update(parse_status=analysis["status"], instruction_count=analysis["instruction_count"],
                        sensitive_event_count=len(analysis["events"]), s1_counts=dict(sorted(s1.items())),
                        s2_counts=dict(sorted(s2.items())), s3_dependencies=analysis["dependencies"],
                        frozen_endpoint_targets=len(expected_targets), recovered_endpoint_targets=len(aligned_targets),
                        unknown_reasons=sorted(set(reasons)), dex_name=dex_name)
        except Exception as exc:
            base["parse_status"] = "UNPARSEABLE"
            base["unknown_reasons"] = ["DEX_METHOD_DECODE_FAILED", type(exc).__name__]
        host_rows.append(base)
    atomic_json_gz(output_path, {"schema": SCHEMA, "sha256": sid, "status": "OK", "dex_count": dex_count,
                                     "hosts": host_rows, "apk_sha256_verified": True,
                                     "wall_seconds": time.monotonic() - start})


def scan_subprocess(row, smoke=False):
    sid, apk = row["sha256"], row["absolute_path"]
    folder = "smoke_cache" if smoke else "cache"
    output = OUTPUT / folder / f"{sid}.json.gz"
    if output.is_file():
        return sid, "CACHED", 0.0
    cmd = [sys.executable, str(Path(__file__).resolve()), "--scan-one", "--sha", sid, "--apk", apk,
           "--frozen", str(OUTPUT / "frozen" / f"{sid}.json"), "--scan-output", str(output)]
    start = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=CONFIG["per_apk_timeout_seconds"], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if proc.returncode:
        atomic_json(OUTPUT / ("smoke_status" if smoke else "status") / f"{sid}.json",
                    {"sha256": sid, "status": "ERROR", "returncode": proc.returncode, "output": proc.stdout[-8000:]})
        return sid, "ERROR", time.monotonic() - start
    return sid, "OK", time.monotonic() - start


def run_scans(smoke_count=0):
    lineage = load_lineage().sort_values("sha256")
    if smoke_count:
        # Prefer the smallest non-empty frozen Host bags for a fast real-APK launch check.
        candidates = []
        for row in lineage.to_dict("records"):
            view = json.loads((OUTPUT / "frozen" / f"{row['sha256']}.json").read_text(encoding="utf-8"))
            if view["host_ids"]:
                candidates.append((len(view["host_ids"]), row))
        rows = [value for _, value in sorted(candidates, key=lambda x: x[0])[:smoke_count]]
        results = [scan_subprocess(row, smoke=True) for row in rows]
        if any(status not in ("OK", "CACHED") for _, status, _ in results):
            raise RuntimeError(f"smoke scan failed: {results}")
        atomic_json(OUTPUT / "smoke_test.json", {"status": "PASS", "real_apks": len(results), "results": results})
        print(f"SMOKE_PASS real_apks={len(results)}", flush=True)
        return
    rows = lineage.to_dict("records")
    atomic_json(OUTPUT / "run_status.json", {"status": "RUNNING", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "completed_apks": len(list((OUTPUT / "cache").glob("*.json.gz"))), "total_apks": 799})
    completed, failed = 0, []
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as pool:
        futures = {pool.submit(scan_subprocess, row): row["sha256"] for row in rows}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                _, status, wall = future.result()
            except subprocess.TimeoutExpired:
                status, wall = "RESOURCE_LIMIT", CONFIG["per_apk_timeout_seconds"]
                atomic_json(OUTPUT / "status" / f"{sid}.json", {"sha256": sid, "status": status})
            except Exception as exc:
                status, wall = "ERROR", 0
                atomic_json(OUTPUT / "status" / f"{sid}.json", {"sha256": sid, "status": status, "error": repr(exc)})
            completed += 1
            if status in ("ERROR", "RESOURCE_LIMIT"):
                failed.append(sid)
            if completed % 10 == 0 or completed == 799:
                atomic_json(OUTPUT / "run_status.json", {"status": "RUNNING", "completed_apks": completed,
                            "failed_apks": len(failed), "total_apks": 799, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                print(f"SCAN_PROGRESS {completed}/799 failed={len(failed)} last={sid} status={status} wall={wall:.1f}s", flush=True)
    summarize(failed)


def summarize(failed):
    lineage = load_lineage()
    routes = router_rows(set(lineage.sha256))
    npz = np.load(ASSETS["features"], allow_pickle=False)
    feature_index = {str(s): i for i, s in enumerate(npz["sha256"].tolist())}
    fold_map = routes.groupby("sha256").fold.first().astype(int).to_dict()
    host_rows, apk_rows, unknown_counts = [], [], Counter()
    alignment = Counter()
    per_apk_effective = {}
    for sid in sorted(lineage.sha256):
        cache = OUTPUT / "cache" / f"{sid}.json.gz"
        if not cache.is_file():
            apk_rows.append({"sha256": sid, "fold": fold_map[sid], "scan_status": "FAILED", "host_count": 0,
                             "s1_effective_hosts": 0, "s2_effective_hosts": 0, "s3_positive_hosts": 0})
            per_apk_effective[sid] = (False, False, False)
            continue
        record = load_json_gz(cache)
        view = json.loads((OUTPUT / "frozen" / f"{sid}.json").read_text(encoding="utf-8"))
        idx = feature_index[sid]
        lo = int(npz["host_offsets"][idx]); real = int(npz["real_host_count"][idx])
        values = npz["host_values"][lo:lo + real]
        if len(record["hosts"]) != len(view["host_ids"]) or len(record["hosts"]) != real:
            raise RuntimeError(f"summary Host row alignment failed: {sid}")
        counts = Counter()
        for row_no, (host, hid) in enumerate(zip(record["hosts"], view["host_ids"])):
            if host["host_stable_id"] != hid:
                raise RuntimeError(f"stable Host order mismatch: {sid}")
            alignment[host["parse_status"]] += 1
            unknown_counts.update(host.get("unknown_reasons", []))
            s1_effective = any(k != "UNKNOWN" and v for k, v in host.get("s1_counts", {}).items())
            s2_effective = any(k != "UNKNOWN" and v for k, v in host.get("s2_counts", {}).items())
            s3_positive = bool(host.get("s3_dependencies"))
            counts["s1"] += s1_effective; counts["s2"] += s2_effective; counts["s3"] += s3_positive
            selfrole = values[row_no, :31].astype(int).tolist()
            endpoint_counts = np.rint(np.expm1(values[row_no, 43:74])).clip(min=0).astype(int).tolist()
            relation = values[row_no, 74:105].round(7).tolist()
            s1_sig = sorted(k for k, v in host.get("s1_counts", {}).items() if v)
            s2_sig = sorted(k for k, v in host.get("s2_counts", {}).items() if v)
            s3_sig = sorted((d["source_role"], d["target_role"], d["dependency_type"]) for d in host.get("s3_dependencies", []))
            host_rows.append({"sha256": sid, "fold": fold_map[sid], "host_stable_id": hid,
                "SelfRole31": json.dumps(selfrole, separators=(",", ":")),
                "Tail12": json.dumps(values[row_no, 31:43].round(7).tolist(), separators=(",", ":")),
                "endpoint_counts31": json.dumps(endpoint_counts, separators=(",", ":")),
                "Relation31": json.dumps(relation, separators=(",", ":")),
                "semantic_S1": json.dumps(host.get("s1_counts", {}), sort_keys=True, separators=(",", ":")),
                "semantic_S2": json.dumps(host.get("s2_counts", {}), sort_keys=True, separators=(",", ":")),
                "semantic_S3": json.dumps(host.get("s3_dependencies", []), sort_keys=True, separators=(",", ":")),
                "parse_status": host["parse_status"], "instruction_count": host.get("instruction_count", 0),
                "sensitive_event_count": host.get("sensitive_event_count", 0),
                "active_role_count": int(sum(selfrole)), "endpoint_event_count": int(sum(endpoint_counts)),
                "s1_signature": json.dumps(s1_sig), "s2_signature": json.dumps(s2_sig), "s3_signature": json.dumps(s3_sig)})
        apk_rows.append({"sha256": sid, "fold": fold_map[sid], "scan_status": record["status"], "host_count": real,
                         "s1_effective_hosts": counts["s1"], "s2_effective_hosts": counts["s2"],
                         "s3_positive_hosts": counts["s3"], "s1_effective": int(counts["s1"] > 0),
                         "s2_effective": int(counts["s2"] > 0), "s3_positive": int(counts["s3"] > 0)})
        per_apk_effective[sid] = (counts["s1"] > 0, counts["s2"] > 0, counts["s3"] > 0)
    write_csv(OUTPUT / "host_semantic_features.csv.gz", host_rows, compress=True)
    write_csv(OUTPUT / "apk_semantic_coverage.csv", apk_rows)
    route_coverage = []
    for seed, part in routes.groupby("manifest_seed"):
        for scope, subset in (("ALL", part), ("ROUTED", part[part.routed.astype(bool)]), ("NOT_ROUTED", part[~part.routed.astype(bool)])):
            vals = [per_apk_effective.get(s, (False, False, False)) for s in subset.sha256]
            route_coverage.append({"manifest_seed": int(seed), "scope": scope, "seed_apk_count": len(subset),
                                   "independent_apk_count": subset.sha256.nunique(),
                                   "s1_effective_apks": sum(v[0] for v in vals), "s2_effective_apks": sum(v[1] for v in vals),
                                   "s3_positive_apks": sum(v[2] for v in vals)})
    union = set(routes[routes.routed.astype(bool)].sha256)
    vals = [per_apk_effective.get(s, (False, False, False)) for s in union]
    route_coverage.append({"manifest_seed": "UNION", "scope": "ROUTED_BY_ANY_SEED", "seed_apk_count": int(routes.routed.sum()),
                           "independent_apk_count": len(union), "s1_effective_apks": sum(v[0] for v in vals),
                           "s2_effective_apks": sum(v[1] for v in vals), "s3_positive_apks": sum(v[2] for v in vals)})
    write_csv(OUTPUT / "routed_semantic_coverage.csv", route_coverage)

    host_frame = pd.DataFrame(host_rows)
    collision_rows = []
    variant_apks = set()
    if not host_frame.empty:
        # Exact controls: same SelfRole31, endpoint-event count, active roles, and instruction count.
        group_cols = ["SelfRole31", "endpoint_event_count", "active_role_count", "instruction_count"]
        for key, group in host_frame.groupby(group_cols, dropna=False):
            if len(group) < 2 or group.sha256.nunique() < 2:
                continue
            signatures = (group.s1_signature + "|" + group.s2_signature + "|" + group.s3_signature).nunique()
            if signatures > 1:
                variant_apks.update(group.sha256)
            collision_rows.append({"SelfRole31": key[0], "endpoint_event_count": key[1], "active_role_count": key[2],
                                   "instruction_count": key[3], "host_count": len(group),
                                   "independent_apk_count": group.sha256.nunique(), "semantic_signature_count": signatures,
                                   "has_new_semantic_variation": int(signatures > 1)})
    write_csv(OUTPUT / "selfrole31_collision_analysis.csv", collision_rows)
    valid_status = {"SUCCESS", "PARTIAL"}
    aligned_hosts = sum(alignment[s] for s in valid_status)
    total_hosts = len(host_rows)
    s1_apks = sum(v[0] for v in per_apk_effective.values()); s2_apks = sum(v[1] for v in per_apk_effective.values()); s3_apks = sum(v[2] for v in per_apk_effective.values())
    independence = [
        {"comparison": "SelfRole31", "test": "exact-role-and-size-controlled collision groups", "result": "VARIATION_OBSERVED" if variant_apks else "NO_VARIATION", "independent_apks": len(variant_apks)},
        {"comparison": "endpoint_counts31", "test": "endpoint-event count held exact in collision analysis", "result": "CONTROLLED", "independent_apks": len(variant_apks)},
        {"comparison": "Tail12", "test": "new values originate from DEX literals/register def-use rather than topology Tail12", "result": "DIFFERENT_PROVENANCE", "independent_apks": max(s1_apks, s2_apks, s3_apks)},
        {"comparison": "Relation31", "test": "direct Host-neighbor Role relation versus intramethod use", "result": "DIFFERENT_RELATION", "independent_apks": s3_apks},
        {"comparison": "PairRaw465/InterZ961", "test": "APK Role co-occurrence/call direction versus exact local def-use", "result": "DIFFERENT_RELATION", "independent_apks": s3_apks},
        {"comparison": "Background4", "test": "prototype-distance scalars versus categorized bytecode evidence", "result": "NOT_A_DETERMINISTIC_TRANSFORM_BY_CONSTRUCTION", "independent_apks": max(s1_apks, s2_apks, s3_apks)},
        {"comparison": "parse_success", "test": "aligned method extraction", "result": f"{aligned_hosts}/{total_hosts}", "independent_apks": sum(r["scan_status"].startswith("OK") for r in apk_rows)},
    ]
    write_csv(OUTPUT / "semantic_information_independence.csv", independence)
    write_csv(OUTPUT / "semantic_unknown_breakdown.csv", [{"reason": key, "host_count": value} for key, value in unknown_counts.most_common()])
    alignment_rate = aligned_hosts / total_hosts if total_hosts else 0.0
    routed_effective = max((r["s1_effective_apks"], r["s2_effective_apks"], r["s3_positive_apks"]) for r in route_coverage if r["manifest_seed"] == "UNION")
    max_apks = max(s1_apks, s2_apks, s3_apks)
    thresholds = CONFIG["feasibility_thresholds"]
    if alignment_rate < thresholds["host_alignment_rate"] or len(failed) > 79:
        decision = "HOST_SEMANTIC_EXTRACTION_UNRELIABLE"
    elif max_apks < thresholds["minimum_independent_apks"] or max(routed_effective) < thresholds["minimum_routed_union_apks"]:
        decision = "HOST_SEMANTIC_INPUT_SPARSE"
    elif len(variant_apks) < thresholds["minimum_collision_variant_apks"]:
        decision = "HOST_SEMANTIC_REDUNDANT"
    else:
        decision = "HOST_SEMANTIC_INPUT_FEASIBLE"
    top_apk_share = 0.0
    if host_rows:
        top_apk_share = pd.Series([r["sha256"] for r in host_rows]).value_counts().iloc[0] / len(host_rows)
    final = {"status": decision, "scope": "INPUT_FEASIBILITY_ONLY", "training_performed": False,
             "future_years_loaded": 0, "labels_used_for_feature_selection": False, "apks": 799,
             "failed_apks": len(failed), "frozen_hosts": total_hosts, "aligned_host_rate": alignment_rate,
             "s1_effective_apks": s1_apks, "s2_effective_apks": s2_apks, "s3_positive_apks": s3_apks,
             "same_selfrole_size_controlled_variant_apks": len(variant_apks), "largest_apk_host_share": top_apk_share,
             "claim_boundary": "This status does not imply classification performance improvement.",
             "only_allowed_followup": "If feasible: one frozen-protocol 2017 five-fold residual Host semantic encoder with matched capacity and shuffled-binding controls; not executed here."}
    atomic_json(OUTPUT / "final_decision.json", final)
    report = "# CARE-Droid Host 方法级行为语义可行性报告\n\n"
    report += f"**判定：`{decision}`**\n\n本实验仅审计输入可提取性，不训练分类器，不表示 BA 或未来年份性能提升。\n\n"
    report += f"- 2017 APK：799；冻结 Host：{total_hosts}；方法绑定/解析有效率：{alignment_rate:.2%}。\n"
    report += f"- S1 有效 APK：{s1_apks}；S2 有效 APK：{s2_apks}；存在 S3 明确局部 def-use 的 APK：{s3_apks}。\n"
    report += f"- 在 SelfRole31、敏感调用数量、活跃 Role 数量与指令数均相同的严格分组中，出现新语义差异的独立 APK：{len(variant_apks)}。\n"
    report += f"- 单个最大 APK 占全部 Host 的比例：{top_apk_share:.2%}。UNKNOWN 被单独保留，CFG 可达、Role 共现和 Host 邻接均未当作数据依赖。\n\n"
    report += "S1 只保存有限参数类别，不保存完整字符串；S2 只跟踪当前方法内可确认的返回值使用；S3 只接受同一基本块中直接寄存器或局部 move 传播。数组、堆、字段别名、异常路径、Native、反射和跨方法传播均保守记为未知。\n\n"
    report += "自主路由覆盖按 3 个 Manifest seed 分开统计，并另报独立 APK 并集；三个 seed 的重复预测没有作为三个独立样本。2018—2021 未载入。\n"
    safe(OUTPUT / "final_report.md").write_text(report, encoding="utf-8")
    before = json.loads((OUTPUT / "asset_manifest.json").read_text(encoding="utf-8"))
    after = [file_record(Path(row["absolute_path"]), row["purpose"]) for row in before["protected_assets"]]
    integrity = all(a == b for a, b in zip(before["protected_assets"], after))
    atomic_json(OUTPUT / "asset_manifest.json", {**before, "status": "PASS" if integrity else "FAIL",
                "protected_assets_after": after, "protected_assets_unchanged": integrity})
    if not integrity:
        raise RuntimeError("protected asset integrity changed during read-only audit")
    atomic_json(OUTPUT / "run_status.json", {"status": "COMPLETED", "completed_apks": 799 - len(failed),
                "failed_apks": len(failed), "total_apks": 799, "decision": decision,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"FINAL_DECISION {decision} completed={799-len(failed)}/799", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--scan-one", action="store_true")
    parser.add_argument("--sha"); parser.add_argument("--apk"); parser.add_argument("--frozen"); parser.add_argument("--scan-output")
    args = parser.parse_args()
    if args.scan_one:
        scan_one(args.sha, Path(args.apk), Path(args.frozen), Path(args.scan_output)); return
    if args.preflight:
        preflight(); return
    if args.smoke:
        run_scans(args.smoke); return
    if args.full:
        run_scans(); return
    parser.error("choose --preflight, --smoke N, --full, or --scan-one")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            atomic_json(OUTPUT / "run_status.json", {"status": "FAILED", "error": repr(exc),
                        "traceback": traceback.format_exc(), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        except Exception:
            pass
        raise

