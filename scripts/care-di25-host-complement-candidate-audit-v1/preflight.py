#!/usr/bin/env python3
"""Hard preflight for the isolated Host-complement candidate audit."""
from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_common import ROOT, PREFLIGHT, REPORTS, THRESHOLD, YEARS, FOLDS, norm, read_csv, sha256, write_json

COMPLEMENT_MANIFEST = Path("artifacts/care-di25-hostego-complementarity-audit-v1/preflight/source_manifest.json")
COHORT_CANDIDATES = (
    Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"),
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/apk_stage2_internal.csv"),
)
ASSET_ROOT_CANDIDATES = (
    Path("artifacts/care-future-local-assets-v2"),
    Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2"),
)
REPS = {"B0_graph_token32", "B1_critical_hostego24", "B2_attended_hostego24"}


def existing_manifest_paths(value):
    found = []
    if isinstance(value, dict):
        for child in value.values(): found.extend(existing_manifest_paths(child))
    elif isinstance(value, list):
        for child in value: found.extend(existing_manifest_paths(child))
    elif isinstance(value, str):
        path = Path(value)
        if path.suffix.lower() == ".csv" and path.is_file(): found.append(path.resolve())
    return found


def discover_csvs(complement):
    paths = existing_manifest_paths(complement)
    roots = [
        Path("artifacts/care-di25-hostego-complementarity-audit-v1"),
        Path("artifacts/care-di25-hostego-feature-sufficiency-v1"),
        Path("artifacts/care-di25-hostego-failure-block-audit-v1"),
    ]
    for root in roots:
        if root.is_dir(): paths.extend(path.resolve() for path in root.rglob("*.csv"))
    return list(dict.fromkeys(paths))


def resolve_prior_inputs(complement, routed_ids):
    audit, probe_path, failure_path, historical_path = [], None, None, None
    for path in discover_csvs(complement):
        try:
            rows = read_csv(path)
        except Exception as exc:
            audit.append({"path": str(path), "status": "unreadable", "error": str(exc)}); continue
        cols = set(rows[0]) if rows else set()
        row_audit = {"path": str(path), "rows": len(rows), "columns": sorted(cols)}
        probe_id = "sha256" if "sha256" in cols else ("sample_id" if "sample_id" in cols else None)
        if probe_id and {"task", "representation"}.issubset(cols):
            score_ids = defaultdict(set)
            for row in rows:
                if row.get("task") == "need" and row.get("representation") in REPS:
                    score_ids[norm(row[probe_id])].add(row["representation"])
            if set(score_ids) == routed_ids and all(value == REPS for value in score_ids.values()):
                probe_path = path; row_audit["selected_as"] = "strict_oof_probe"
        if probe_id and "failure_type" in cols:
            counts = Counter(row["failure_type"] for row in rows)
            if counts["A_WRONG_DIRECTION"] == 27 and counts["B_RIGHT_DIRECTION_TOO_WEAK"] == 29:
                failure_path = path; row_audit["selected_as"] = "failure_type"
        id_col = "sha256" if "sha256" in cols else ("sample_id" if "sample_id" in cols else None)
        if id_col and len({norm(row[id_col]) for row in rows}) == len(routed_ids) and {norm(row[id_col]) for row in rows} == routed_ids:
            if historical_path is None or "historical" in path.name.lower():
                historical_path = path; row_audit["selected_as_historical_candidate"] = True
        audit.append(row_audit)
    if probe_path is None or failure_path is None or historical_path is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: could not schema-resolve prior probe/failure/historical inputs: " + json.dumps(audit))
    return probe_path, historical_path, failure_path, audit


def choose_file(candidates, required_columns, label):
    audits = []
    for path in candidates:
        if not path.is_file():
            audits.append({"path": str(path), "status": "missing"}); continue
        rows = read_csv(path); columns = set(rows[0]) if rows else set()
        missing = required_columns - columns
        audits.append({"path": str(path), "rows": len(rows), "columns": sorted(columns), "missing": sorted(missing)})
        if not missing:
            return path.resolve(), rows, audits
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: no {label} has required schema: {audits}")


def npz_ids(saved, label):
    for key in ("sha256", "sample_ids", "apk_ids"):
        if key in saved.files:
            return [norm(value) for value in saved[key].astype(str)], key
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: {label} has no canonical sample-ID array; keys={saved.files}")


def static_code_audit():
    here = Path(__file__).resolve().parent
    files = [
        "extract_candidate_features.py", "run_candidate_probes.py", "analyze_conditional_rescue.py",
        "summarize_candidate_audit.py",
    ]
    results = []
    for name in files:
        path = here / name
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: package implementation missing: " + str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports, checkpoint_literals = set(), []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and Path(node.value).suffix.lower() in {".pt", ".pth", ".ckpt"}:
                checkpoint_literals.append({"line": node.lineno, "value": node.value})
        forbidden = sorted(imports & {"torch", "tensorflow", "xgboost", "lightgbm", "androguard", "soot", "flowdroid"})
        results.append({"path": str(path), "sha256": sha256(path), "forbidden_imports": forbidden, "checkpoint_literals": checkpoint_literals})
    if any(row["forbidden_imports"] or row["checkpoint_literals"] for row in results):
        raise RuntimeError("BLOCKED_BY_CONTRACT: forbidden implementation dependency: " + json.dumps(results))
    return results


def main():
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    if not COMPLEMENT_MANIFEST.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: prior complementarity source manifest missing")
    complement = json.loads(COMPLEMENT_MANIFEST.read_text(encoding="utf-8"))
    for raw, digest in complement.get("input_sha256", {}).items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError("BLOCKED_BY_CONTRACT: prior complementarity lineage changed: " + raw)

    cohort_path, cohort_rows, cohort_audits = choose_file(
        COHORT_CANDIDATES, {"sha256", "year", "fold", "label", "route_DI", "p_stage1"}, "1989 cohort file")
    if len(cohort_rows) != 1989 or len({norm(row["sha256"]) for row in cohort_rows}) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: cohort is not exact unique 1989")
    year_counts = Counter(int(float(row["year"])) for row in cohort_rows)
    if year_counts != Counter({2018: 793, 2019: 399, 2020: 397, 2021: 400}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: year counts changed: " + str(year_counts))
    if set(int(float(row["fold"])) for row in cohort_rows) != set(FOLDS):
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold IDs are not exact 1..5")
    routed_ids = {norm(row["sha256"]) for row in cohort_rows if int(float(row["route_DI"])) == 1}
    if not routed_ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed cohort is empty")

    probe_path, historical_path, failure_path, prior_discovery_audit = resolve_prior_inputs(complement, routed_ids)
    for path in (probe_path, historical_path, failure_path):
        if not path.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: missing prior input: " + str(path))
    probes = read_csv(probe_path)
    probe_id = "sha256" if "sha256" in probes[0] else "sample_id"
    if not any(name in probes[0] for name in ("probability", "oof_probability", "prob", "p", "score")):
        raise RuntimeError("BLOCKED_BY_CONTRACT: prior strict OOF probe has no probability column")
    score_ids = defaultdict(set)
    for row in probes:
        if row.get("task") == "need" and row.get("representation") in REPS:
            score_ids[norm(row[probe_id])].add(row["representation"])
    if set(score_ids) != routed_ids or any(values != REPS for values in score_ids.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: Graph/Critical/Attended OOF lineage does not equal routed cohort")
    historical = read_csv(historical_path)
    historical_id = "sha256" if "sha256" in historical[0] else "sample_id"
    historical_ids = {norm(row[historical_id]) for row in historical}
    if historical_ids != routed_ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical cohort mapping differs from routed cohort")
    failures = read_csv(failure_path)
    failure_counts = Counter(row["failure_type"] for row in failures)
    if failure_counts["A_WRONG_DIRECTION"] != 27 or failure_counts["B_RIGHT_DIRECTION_TOO_WEAK"] != 29:
        raise RuntimeError("BLOCKED_BY_CONTRACT: WrongDirection/TooWeak counts changed")

    asset_root = None
    asset_names = (
        "local_asset_manifest.json", "host_regions_2018_2021.npz", "ego_1hop_regions_2018_2021.npz",
        "global_role_graph_2018_2021.npz", "endpoint_centered_tensor_cache_2018_2021.jsonl.gz",
    )
    root_audit = []
    for root in ASSET_ROOT_CANDIDATES:
        missing = [name for name in asset_names if not (root / name).is_file()]
        root_audit.append({"root": str(root), "missing": missing})
        if not missing:
            asset_root = root.resolve(); break
    if asset_root is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: no complete frozen future local-asset root: " + json.dumps(root_audit))
    manifest_path = asset_root / "local_asset_manifest.json"
    host_path = asset_root / "host_regions_2018_2021.npz"
    ego_path = asset_root / "ego_1hop_regions_2018_2021.npz"
    role_graph_path = asset_root / "global_role_graph_2018_2021.npz"
    endpoint_path = asset_root / "endpoint_centered_tensor_cache_2018_2021.jsonl.gz"
    asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = list(asset_manifest.get("model_role_order", []))
    payload_to_model = list(asset_manifest.get("payload_to_model_role", []))
    if not roles or len(payload_to_model) != len(roles) or sorted(payload_to_model) != list(range(len(roles))):
        raise RuntimeError("BLOCKED_BY_CONTRACT: actual role taxonomy cannot be recovered")

    cohort_ids = {norm(row["sha256"]) for row in cohort_rows}
    with np.load(host_path, allow_pickle=False) as saved:
        host_ids, host_id_key = npz_ids(saved, "Host cache")
        if "real_host_count" in saved.files:
            real_hosts = np.asarray(saved["real_host_count"], int)
        elif "host_offsets" in saved.files:
            real_hosts = np.diff(np.asarray(saved["host_offsets"], int))
        else:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache has neither real_host_count nor host_offsets")
    with np.load(ego_path, allow_pickle=False) as saved:
        ego_ids, ego_id_key = npz_ids(saved, "Ego cache")
        host_role_count = len(saved["host_ids"]) if "host_ids" in saved.files else int(np.asarray(saved["ego_offsets"])[-1])
    with np.load(role_graph_path, allow_pickle=False) as saved:
        graph_ids, graph_id_key = npz_ids(saved, "role graph cache")
    if set(host_ids) != cohort_ids or set(ego_ids) != cohort_ids or set(graph_ids) != cohort_ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: Host/Ego/role graph lineage differs from 1989 cohort")

    endpoint_ids, example_payload = [], None
    with gzip.open(endpoint_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); endpoint_ids.append(norm(row.get("sha256")))
            if example_payload is None and row.get("payload", {}).get("node_ids"):
                example_payload = row["payload"]
    if len(endpoint_ids) != 1989 or len(set(endpoint_ids)) != 1989 or set(endpoint_ids) != cohort_ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache is not exact unique 1989 cohort")
    if example_payload is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint cache has no inspectable graph payload")
    payload_required = {"node_ids", "node_role_ids", "node_base_features", "edge_index"}
    if not payload_required.issubset(example_payload):
        raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint payload cannot support candidate extraction")

    source_files = [
        COMPLEMENT_MANIFEST.resolve(), cohort_path, probe_path.resolve(), historical_path.resolve(), failure_path.resolve(),
        manifest_path, host_path, ego_path, role_graph_path, endpoint_path,
    ]
    hashes = {str(path): sha256(path) for path in source_files}
    with (PREFLIGHT / "original_sha256_before.txt").open("w", encoding="utf-8") as handle:
        for path, digest in hashes.items(): handle.write(f"{digest}  {path}\n")

    implementation = static_code_audit()
    source_manifest = {
        "status": "HOST_COMPLEMENT_CANDIDATE_PREFLIGHT_PASS",
        "cohort_file": str(cohort_path), "strict_oof_probe_file": str(probe_path.resolve()),
        "historical_cohort_file": str(historical_path.resolve()), "failure_type_file": str(failure_path.resolve()),
        "asset_root": str(asset_root), "local_asset_manifest": str(manifest_path), "host_cache": str(host_path),
        "ego_cache": str(ego_path), "role_graph_cache": str(role_graph_path), "endpoint_cache": str(endpoint_path),
        "cohort_n": 1989, "routed_n": len(routed_ids), "year_counts": dict(sorted(year_counts.items())),
        "wrong_direction_n": 27, "too_weak_n": 29, "fixed_threshold": THRESHOLD,
        "role_count": len(roles), "role_names": roles, "payload_to_model_role": payload_to_model,
        "original_sha256": hashes, "cohort_candidate_audit": cohort_audits,
        "prior_input_discovery_audit": prior_discovery_audit,
        "implementation_static_audit": implementation,
    }
    write_json(PREFLIGHT / "source_manifest.json", source_manifest)
    data_audit = {
        "endpoint_cache": str(endpoint_path), "endpoint_cache_sha256": hashes[str(endpoint_path)],
        "sample_count": 1989, "graph_coverage": len(endpoint_ids) / 1989,
        "host_coverage": float(np.mean(real_hosts > 0)), "real_host_total": int(np.sum(real_hosts)),
        "cache_id_keys": {"host": host_id_key, "ego": ego_id_key, "role_graph": graph_id_key},
        "ego_host_metadata_rows": host_role_count, "role_taxonomy_source": str(manifest_path),
        "role_count": len(roles), "method_identity_format": "opaque frozen node_id/method identifier",
        "method_identity_example": str(example_payload["node_ids"][0]),
        "edge_direction": "edge_index[0] caller/source -> edge_index[1] callee/target",
        "contains_ordinary_method": True,
        "ordinary_definition": "frozen graph node that is neither a Host nor a sensitive endpoint",
        "can_distinguish_host_and_ordinary": True,
        "can_recover_host_sensitive_endpoint_roles": True,
        "contains_direct_caller_callee": True,
        "graph_scope": asset_manifest.get("neighbor_scope", "bounded endpoint-centred raw retained call graph"),
        "no_apk_reanalysis": True,
    }
    write_json(PREFLIGHT / "data_source_audit.json", data_audit)
    contract = {
        "status": "PASS", "diagnostic_only": True, "strict_same_year_OOF": True,
        "not_future_unseen_zero_day_or_independent_temporal": True,
        "CARE_neural_training": False, "Global24_training": False, "existing_probe_retraining": False,
        "neural_replay": False, "APK_reanalysis": False, "diagnostic_L2_LR_allowed": True,
        "hyperparameter_search": False, "label_based_feature_selection": False, "threshold_search": False,
        "fixed_threshold": THRESHOLD, "CARE_prediction_modification": False,
        "output_root": str(ROOT),
    }
    write_json(PREFLIGHT / "contract.json", contract)
    print("HOST_COMPLEMENT_CANDIDATE_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
