#!/usr/bin/env python3
"""Resolve and hard-audit the saved strict-OOF inputs without training anything."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("artifacts/care-di25-hostego-complementarity-audit-v1")
PREFLIGHT = ROOT / "preflight"
THRESHOLD = 0.5
REQUIRED_REPS = {
    "B0_graph_token32",
    "B1_critical_hostego24",
    "B2_attended_hostego24",
    "B3_attention_anomaly_summary8",
}

PROBE_CANDIDATES = (
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit/component_probe_oof_predictions.csv"),
)
COHORT_CANDIDATES = (
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/gate_cohort_audit/historical_cohort_gate_scores.csv"),
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/failure_block_audit/failure_type_block_scores.csv"),
)
FAILURE_CANDIDATES = (
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/routed_unfixed_failure_types.csv"),
)
INTERNAL_CANDIDATES = (
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/apk_stage2_internal.csv"),
)


def norm(value):
    return str(value or "").strip().lower()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_existing(candidates, description):
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: no {description} candidate exists")
    return existing[0]


def main():
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    implementation_path = Path(__file__).with_name("run_host_complementarity_audit.py")
    implementation_source = implementation_path.read_text(encoding="utf-8")
    tree = ast.parse(implementation_source)
    forbidden_imports = {"torch", "tensorflow", "sklearn", "xgboost", "lightgbm"}
    imported_roots = set()
    forbidden_calls = []
    checkpoint_literals = []
    threshold_constants = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "fit", "fit_transform", "train", "backward", "step", "save", "save_checkpoint",
                "state_dict", "load_state_dict",
            }:
                forbidden_calls.append({"line": node.lineno, "call": node.func.attr})
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "THRESHOLD" for target in node.targets):
                if isinstance(node.value, ast.Constant):
                    threshold_constants.append(node.value.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if Path(node.value).suffix.lower() in {".pt", ".pth", ".ckpt"}:
                checkpoint_literals.append({"line": node.lineno, "literal": node.value})
    implementation_audit = {
        "implementation": str(implementation_path),
        "implementation_sha256": sha256(implementation_path),
        "forbidden_training_framework_imports": sorted(imported_roots & forbidden_imports),
        "forbidden_training_or_checkpoint_calls": forbidden_calls,
        "checkpoint_path_literals": checkpoint_literals,
        "threshold_constants": threshold_constants,
        "threshold_fixed_exactly_once_to_0p5": threshold_constants == [0.5],
        "passed": not (imported_roots & forbidden_imports) and not forbidden_calls and not checkpoint_literals and threshold_constants == [0.5],
    }
    if not implementation_audit["passed"]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: implementation-level read-only audit failed: " + json.dumps(implementation_audit))
    probe_path = choose_existing(PROBE_CANDIDATES, "strict OOF block-probe file")
    cohort_path = choose_existing(COHORT_CANDIDATES, "historical cohort mapping")
    failure_path = choose_existing(FAILURE_CANDIDATES, "failure-type mapping")
    internal_existing = [path for path in INTERNAL_CANDIDATES if path.is_file()]
    internal_path = internal_existing[0] if internal_existing else None

    probes = read_csv(probe_path)
    probe_columns = set(probes[0]) if probes else set()
    required_probe_columns = {"task", "year", "fold", "sha256", "label", "representation", "probability"}
    if not required_probe_columns.issubset(probe_columns):
        raise RuntimeError("BLOCKED_BY_CONTRACT: strict OOF probe schema missing columns: " + str(sorted(required_probe_columns - probe_columns)))
    selected = [row for row in probes if row["task"] == "need" and row["representation"] in REQUIRED_REPS]
    by_sha = defaultdict(dict)
    for row in selected:
        sample = norm(row["sha256"])
        rep = row["representation"]
        if rep in by_sha[sample]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate strict OOF block probability")
        probability = float(row["probability"])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("BLOCKED_BY_CONTRACT: probe probability outside [0,1]")
        by_sha[sample][rep] = probability
    if len(by_sha) != 497 or any(set(values) != REQUIRED_REPS for values in by_sha.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected four complete saved block probabilities for 497 routed APKs")

    cohorts = read_csv(cohort_path)
    cohort_columns = set(cohorts[0]) if cohorts else set()
    required_cohort_columns = {"sha256", "year", "fold", "label", "cohort", "failure_type"}
    if not required_cohort_columns.issubset(cohort_columns):
        raise RuntimeError("BLOCKED_BY_CONTRACT: cohort schema missing columns: " + str(sorted(required_cohort_columns - cohort_columns)))
    cohort_by_sha = {}
    for row in cohorts:
        sample = norm(row["sha256"])
        if sample in cohort_by_sha:
            raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate historical cohort APK")
        cohort_by_sha[sample] = row
    if len(cohort_by_sha) != 497 or set(cohort_by_sha) != set(by_sha):
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical cohort and strict OOF probability lineage differ")
    cohort_counts = Counter(row["cohort"] for row in cohorts)
    if cohort_counts != Counter({"E": 360, "D": 74, "C": 52, "F": 11}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked C/D/E/F cohort counts changed: " + str(cohort_counts))

    failures = read_csv(failure_path)
    failure_columns = set(failures[0]) if failures else set()
    if not {"sha256", "failure_type"}.issubset(failure_columns):
        raise RuntimeError("BLOCKED_BY_CONTRACT: failure mapping schema is incomplete")
    failure_counts = Counter(row["failure_type"] for row in failures)
    expected_failure_counts = Counter({
        "A_WRONG_DIRECTION": 27,
        "B_RIGHT_DIRECTION_TOO_WEAK": 29,
        "C_BOUND_LIMIT": 14,
        "D_NEAR_ZERO": 4,
    })
    if failure_counts != expected_failure_counts:
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked failure counts changed: " + str(failure_counts))
    failure_by_sha = {norm(row["sha256"]): row["failure_type"] for row in failures}
    if set(failure_by_sha) != {sample for sample, row in cohort_by_sha.items() if row["cohort"] == "D"}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: failure mapping is not the exact routed-unfixed cohort")
    for sample, failure_type in failure_by_sha.items():
        existing = cohort_by_sha[sample].get("failure_type", "")
        if existing and existing != failure_type:
            raise RuntimeError("BLOCKED_BY_CONTRACT: cohort/failure mapping disagreement")

    source_paths = [probe_path, cohort_path, failure_path]
    if internal_path is not None:
        source_paths.append(internal_path)
    hashes = {str(path): sha256(path) for path in source_paths}
    manifest = {
        "status": "HOST_COMPLEMENTARITY_PREFLIGHT_PASS",
        "threshold": THRESHOLD,
        "strict_oof_probe_file": str(probe_path),
        "historical_cohort_file": str(cohort_path),
        "failure_type_file": str(failure_path),
        "optional_existing_mechanism_internal_file": str(internal_path) if internal_path else None,
        "input_sha256": hashes,
        "probe_columns": sorted(probe_columns),
        "cohort_columns": sorted(cohort_columns),
        "probe_representations": sorted(REQUIRED_REPS),
        "routed_apk_n": len(cohort_by_sha),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "lineage_exact_match": True,
    }
    (PREFLIGHT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (PREFLIGHT / "sha256_before.txt").open("w", encoding="utf-8") as handle:
        for path, digest in hashes.items():
            handle.write(f"{digest}  {path}\n")
    contract = {
        "status": "PASS",
        "post_hoc_diagnostic_only": True,
        "same_year_oof_not_unseen_future_or_zero_day": True,
        "CARE_Droid_training_allowed": False,
        "StageII_neural_training_allowed": False,
        "probe_training_allowed": False,
        "checkpoint_write_allowed": False,
        "DI25_route_change_allowed": False,
        "threshold_search_allowed": False,
        "fixed_threshold": THRESHOLD,
        "CARE_Droid_prediction_change_allowed": False,
        "output_root": str(ROOT),
        "implementation_static_audit": implementation_audit,
    }
    (PREFLIGHT / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("HOST_COMPLEMENTARITY_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
