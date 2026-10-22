#!/usr/bin/env python3
"""Freeze and validate the final CARE-Droid method without training anything."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


FINAL_FROZEN_METHOD = "CARE-Droid-DI25-HostEgo-Anomaly-Bounded"
ROOT = Path("artifacts/care-di25-hostego-v1")
FINAL = ROOT / "final_audits"
ROUTER = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv")
ROUTER_SHA256 = "fc4be0871a980a1dec33dca4e915ab75f12de5c74819cd528582a01ddc5284fd"
M0 = ROOT / "source/M0_host_anomaly_bounded/source_predictions.csv"
M1 = ROOT / "source/M1_hostego_anomaly_bounded/source_predictions.csv"
EXPECTED = {
    "M0": (0.9111058897243107, 0.9410213032581454, 0.9086229086229086),
    "M1": (0.9148715538847118, 0.9433521303258148, 0.9132653061224489),
}
SEED_SETS = {
    "R0": [0, 1, 2], "R1": [10, 11, 12], "R2": [20, 21, 22],
    "R3": [30, 31, 32], "R4": [40, 41, 42],
}


def blocked(message: str) -> None:
    raise SystemExit("BLOCKED_BY_CONTRACT: " + message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def locked_metrics(path: Path, token: str) -> tuple[float, float, float]:
    rows = read_csv(path)
    if len(rows) != 799 or len({row["sha256"].strip().lower() for row in rows}) != 799:
        blocked(f"{token} source cohort is not exact 799")
    columns = [name for name in rows[0] if name.startswith("DI25__") and name.endswith("__prob")]
    columns = [name for name in columns if token in name and (token != "Host" or "HostEgo" not in name)]
    if len(columns) != 1:
        blocked(f"cannot resolve {token} bounded probability: {columns}")
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    p = np.asarray([float(row[columns[0]]) for row in rows], float)
    return (
        float(balanced_accuracy_score(y, p >= 0.5)),
        float(roc_auc_score(y, p)),
        float(f1_score(y, p >= 0.5)),
    )


def validate() -> dict:
    required = [
        ROUTER, M0, M1,
        ROOT / "audits/route_integrity_M01_summary.json",
        ROOT / "audits/parameter_count_M01.csv",
        ROOT / "audits/ego_usage_M1.json",
        ROOT / "audits/ego_fallback_summary_M1.json",
        ROOT / "audits/host_row_count_reconciliation.txt",
        ROOT / "audits/ego_semantics_audit.json",
        ROOT / "source/M0_host_anomaly_bounded/architecture.json",
        ROOT / "source/M1_hostego_anomaly_bounded/architecture.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        blocked("freeze prerequisites missing:\n- " + "\n- ".join(missing))
    actual_hash = sha256_file(ROUTER)
    if actual_hash != ROUTER_SHA256:
        blocked(f"canonical router hash mismatch; expected={ROUTER_SHA256} observed={actual_hash} affected_file={ROUTER}")
    router = read_csv(ROUTER)
    if len(router) != 799 or len({row["sha256"].strip().lower() for row in router}) != 799:
        blocked("canonical DI25 cohort mismatch")
    routed = sum(int(float(row["route_DI"])) for row in router)
    if routed != 200:
        blocked(f"DI25 route mismatch; expected=200 observed={routed} affected_APK_count={abs(routed-200)}")
    fold_counts = {fold: sum(int(float(row["fold"])) == fold for row in router) for fold in range(1, 6)}
    if fold_counts != {1: 160, 2: 160, 3: 160, 4: 160, 5: 159}:
        blocked(f"source fold mismatch expected={{1:160,2:160,3:160,4:160,5:159}} observed={fold_counts}")
    for name, path, token in (("M0", M0, "Host"), ("M1", M1, "HostEgo")):
        actual = locked_metrics(path, token)
        if not np.allclose(actual, EXPECTED[name], rtol=0.0, atol=1e-12):
            blocked(f"{name} locked metrics changed; expected={EXPECTED[name]} observed={actual} affected_file={path}")
    route_audit = json.loads((ROOT / "audits/route_integrity_M01_summary.json").read_text(encoding="utf-8"))
    for key in ("M0_vs_DI25_difference_count", "M1_vs_DI25_difference_count", "M0_vs_M1_difference_count"):
        if int(route_audit[key]) != 0:
            blocked(f"DI25 route mismatch: {key}={route_audit[key]}")
    params = read_csv(ROOT / "audits/parameter_count_M01.csv")
    counts = {row.get("arm"): int(float(row["parameter_count"])) for row in params if row.get("arm") in {"M0", "M1"}}
    if counts != {"M0": 33388, "M1": 34613}:
        blocked(f"parameter count mismatch; expected M0=33388 M1=34613 observed={counts}")
    fallback = json.loads((ROOT / "audits/ego_fallback_summary_M1.json").read_text(encoding="utf-8"))
    if float(fallback["max_abs_diff"]) != 0.0 or int(fallback["nonzero_diff_n"]) != 0:
        blocked(f"fallback nonzero; observed={fallback}")
    reconciliation = (ROOT / "audits/host_row_count_reconciliation.txt").read_text(encoding="utf-8")
    if "HOST_ROW_COUNT_RECONCILIATION_PASS" not in reconciliation:
        blocked("Host placeholder reconciliation is not PASS")
    semantics = json.loads((ROOT / "audits/ego_semantics_audit.json").read_text(encoding="utf-8"))
    semantic_checks = {
        "direction_available": semantics.get("direction_available") is True,
        "raw_method_nodes_in_ego": semantics.get("raw_method_nodes_in_ego") is False,
        "ordinary_non_host_method_nodes_in_ego": semantics.get("ordinary_non_host_method_nodes_in_ego") is False,
        "path_contraction_used": semantics.get("path_contraction_used") is False,
    }
    if not all(semantic_checks.values()):
        blocked(f"Host/Ego semantic contract mismatch: {semantic_checks}")
    for arm, folder in (("M0", "M0_host_anomaly_bounded"), ("M1", "M1_hostego_anomaly_bounded")):
        architecture = json.loads((ROOT / f"source/{folder}/architecture.json").read_text(encoding="utf-8"))
        if float(architecture.get("max_delta_logit", -1)) != 2.25 or architecture.get("router") != "DI25":
            blocked(f"{arm} architecture guard mismatch: router={architecture.get('router')} max_delta={architecture.get('max_delta_logit')}")
    return {"router_hash": actual_hash, "parameter_counts": counts,
            "locked_prediction_sha256": {"M0": sha256_file(M0), "M1": sha256_file(M1)}}


def contract() -> dict:
    checked = validate()
    return {
        "FINAL_FROZEN_METHOD": FINAL_FROZEN_METHOD,
        "method_name": FINAL_FROZEN_METHOD,
        "status": "FINAL_METHOD_FREEZE",
        "source_year": 2017,
        "source_n": 799,
        "router": "DI25",
        "router_budget": 0.25,
        "router_source_path": str(ROUTER),
        "router_source_sha256": checked["router_hash"],
        "route_column": "route_DI",
        "stage1_probability_column": "Global24__prob",
        "host_definition": "caller method retained in the sensitive-endpoint-centered bounded graph that directly calls a retained mapped sensitive endpoint",
        "ego_definition": "real directed one-hop neighborhood in the induced Host graph; adjacent Host methods only",
        "upstream_hops": 2,
        "downstream_hops": 1,
        "host_edge_definition": "retained raw call edge after Host-to-Host filtering",
        "path_contraction": False,
        "ego_directional": True,
        "ego_contains_only_hosts": True,
        "ego_projection": "concat(incoming_mean,outgoing_mean)[48] -> Linear(48,24) -> ReLU",
        "ego_gate": "sigmoid(Linear(concat(h_host,h_ego)[48],1)); h_local=h_host+gamma*h_ego; explicit identity fallback without nontrivial Ego",
        "host_only_parameter_count": 33388,
        "hostego_parameter_count": 34613,
        "extra_parameter_count": 1225,
        "relative_parameter_increase": 1225 / 33388,
        "anomaly_definition_source": "scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py",
        "attention_implementation_source": "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
        "frozen_model_implementation": "scripts/care-di25-hostego-v1/host_ego_models.py::HostEgoAnomaly",
        "bounded_formula": "sigmoid(logit(p_stage1) + 2.25*tanh(raw_delta))",
        "max_delta": 2.25,
        "fold_definition": "fixed strict 2017 five-fold outer OOF; fold column inherited from canonical source protocol",
        "seed_robustness_sets": SEED_SETS,
        "future_diagnostic_years": [2018, 2019, 2020, 2021],
        "future_diagnostic_only": True,
        "future_diagnostic_status": "previously inspected during method development",
        "final_holdout_years": [2022, 2023, 2024],
        "final_holdout_status": "readiness to be audited; not executed",
        "prohibited_selection": "No method, router, budget, bound, threshold, structure, or hyperparameter may be revised from diagnostic/final results.",
        "locked_prediction_sha256": checked["locked_prediction_sha256"],
    }


def main() -> None:
    value = contract()
    FINAL.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "frozen_method_contract.json").write_text(text, encoding="utf-8")
    (FINAL / "frozen_method_contract.json").write_text(text, encoding="utf-8")
    print("FINAL_FROZEN_METHOD_CONTRACT_PASS")
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
