#!/usr/bin/env python3
"""Preflight and freeze label-free 2018--2021 diagnostic predictions."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from freeze_final_method import ROOT, blocked, contract, read_csv, sha256_file


OUT = ROOT / "diagnostic_future"
RESOLVED = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
ROUTER = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
M0 = Path("artifacts/care-route-stage2-future-matrix-v1/arms/anomaly_bounded/future_expert_predictions.csv")
M1 = Path("artifacts/care-di25-hostego-future-v1/m1_hostego_anomaly/future_m1_predictions_label_free.csv")
M1_PROTOCOL = Path("artifacts/care-di25-hostego-future-v1/m1_hostego_anomaly/protocol.json")
EXPECTED = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTE = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
FORBIDDEN_LABEL_FIELDS = {"label", "y", "target", "metadata_label", "ground_truth"}


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def unique(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    index: dict[str, dict] = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id")))
        if not sha or sha in index:
            blocked(f"duplicate/empty SHA in {path}: {sha}")
        index[sha] = row
    return index


def label_free(path: Path) -> None:
    rows = read_csv(path)
    if not rows:
        blocked(f"empty label-free file: {path}")
    bad = sorted({name.lower() for name in rows[0]} & FORBIDDEN_LABEL_FIELDS)
    if bad:
        blocked(f"label fields present before unsealing in {path}: {bad}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def preflight() -> dict:
    frozen = contract()
    required = [RESOLVED, ROUTER, M0, M1, M1_PROTOCOL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        blocked("future diagnostic prerequisites missing:\n- " + "\n- ".join(missing))
    for path in (ROUTER, M0, M1):
        label_free(path)
    protocol = json.loads(M1_PROTOCOL.read_text(encoding="utf-8"))
    checks = {
        "source_only_training": protocol.get("source_only_training") is True,
        "future_labels_loaded_by_model_false": protocol.get("future_labels_loaded_by_model") is False,
        "max_delta_2p25": float(protocol.get("max_delta_logit", -1)) == 2.25,
        "parameter_count_34613": int(protocol.get("parameter_count", -1)) == 34613,
        "router_DI25": frozen["router"] == "DI25",
        "budget_25pct": float(frozen["router_budget"]) == .25,
    }
    if not all(checks.values()):
        blocked(f"future protocol checks failed: {checks}")
    router, m0, m1 = unique(ROUTER), unique(M0), unique(M1)
    if len(router) != 1989 or set(m0) != set(router) or set(m1) != set(router):
        blocked(f"future cohort mismatch router={len(router)} M0={len(m0)} M1={len(m1)}")
    year_counts = {year: 0 for year in EXPECTED}; route_counts = {year: 0 for year in EXPECTED}
    for row in router.values():
        year = int(row["year"]); year_counts[year] = year_counts.get(year, 0) + 1
        route_counts[year] = route_counts.get(year, 0) + int(float(row["route_DI"]))
    if year_counts != EXPECTED or route_counts != EXPECTED_ROUTE:
        blocked(f"future year/route contract mismatch years={year_counts} routes={route_counts}")
    return {
        "passed": True, "checks": checks, "n": 1989,
        "year_counts": year_counts, "route_counts": route_counts,
        "input_sha256": {str(path): sha256_file(path) for path in required},
        "probabilities_frozen_before_label_access": True,
        "future_labels_used_for_training_routing_scaling_anomaly_fitting_early_stopping_threshold_or_model_selection": False,
        "diagnostic_only": True,
    }


def stage() -> None:
    started = time.perf_counter()
    state = preflight()
    router, m0, m1 = unique(ROUTER), unique(M0), unique(M1)
    ordered = read_csv(ROUTER)
    rows, routes = [], []
    for item in ordered:
        sha = normal(item["sha256"]); zero, one = m0[sha], m1[sha]
        p0 = zero.get("p_expert", zero.get("p_expert_bounded"))
        if p0 is None:
            blocked(f"M0 bounded probability missing for {sha}")
        row = {
            "sha256": sha, "year": int(item["year"]), "p_stage1": float(item["p_stage1"]),
            "route_DI25": int(float(item["route_DI"])), "p_M0_Host_Anomaly_Bounded": float(p0),
            "p_M1_HostEgo_Anomaly_Bounded": float(one["p_expert_bounded"]),
            "M1_real_host_count": int(one["real_host_count"]),
            "M1_nontrivial_ego_count": int(one["nontrivial_ego_count"]),
            "M1_has_any_nontrivial_ego": int(one["has_any_nontrivial_ego"]),
        }
        rows.append(row)
        routes.append({"sha256": sha, "year": row["year"], "route_DI25": row["route_DI25"]})
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = OUT / "future_predictions_label_free.csv"
    write_csv(predictions, rows); write_csv(OUT / "future_route_audit.csv", routes)
    state.update({
        "staging_seconds": time.perf_counter() - started,
        "label_free_predictions": str(predictions),
        "label_free_predictions_sha256": sha256_file(predictions),
        "source_M1_training_runtime": json.loads(M1_PROTOCOL.read_text(encoding="utf-8")).get("training_seconds_including_epoch_selection", "unavailable"),
        "runtime_note": "M0/M1 predictions were already produced under the frozen source-only protocol; this step only validates and stages them, avoiding retraining after labels were previously inspected.",
    })
    (OUT / "future_protocol_preunseal.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TEMPORAL_DIAGNOSTIC_LABEL_FREE_FREEZE_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = preflight()
    (OUT / "preflight.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TEMPORAL_DIAGNOSTIC_PREFLIGHT_PASS")
    if not args.preflight_only:
        stage()


if __name__ == "__main__":
    main()
