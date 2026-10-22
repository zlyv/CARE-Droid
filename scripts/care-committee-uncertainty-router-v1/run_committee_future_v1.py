#!/usr/bin/env python3
"""Frozen 2018--2021 diagnostic of classifier-committee uncertainty routing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT / "care-malcertain-router-future-v1",
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-future-script",
    SCRIPTS_ROOT / "outputs" / "care-bounded-fixkeep-v3-script",
    SCRIPTS_ROOT / "outputs" / "care-joint-dualprototype-v2-script",
    SCRIPTS_ROOT / "outputs" / "care-local-dualpool-mil-v1-script",
    SCRIPTS_ROOT / "outputs" / "care-local-region-asset-audit-v1-script",
    SCRIPTS_ROOT / "outputs" / "global24-r-joints7-2017-2021-v1-script" / "scripts",
    SCRIPTS_ROOT / "work" / "compact76_trace" / "scripts",
    SCRIPTS_ROOT / "work" / "role_masking_code_bundle_uploaded"
    / "role_masking_code_bundle" / "repository" / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_multiview_conformal_router_source_v1 as mv
    import run_care_malcertain_router_future_v1 as future_base
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE dependencies: " + str(exc))

from committee_uncertainty_core import (  # noqa: E402
    MEMBER_NAMES,
    FutureLabelGate,
    committee_scores,
    diversity_rows,
    fit_calibrated_committee,
    groupwise_top_fraction,
    member_metric_rows,
    routing_metrics,
)


YEARS = (2018, 2019, 2020, 2021)
ROUTE_NAMES = (
    "A25_YearTop25",
    "Committee_PE_YearTop25",
    "Committee_MI_YearTop25",
)
DEFAULT_EXPERT_COLUMN = (
    "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__ensemble3__expert_prob"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_metric_row(
    method: str,
    labels: np.ndarray,
    probability: np.ndarray,
    base: np.ndarray,
    route: np.ndarray,
    *,
    year: int,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    prediction = p >= 0.5
    base_correct = (np.asarray(base, dtype=np.float64) >= 0.5) == y
    final_correct = prediction == y
    return {
        "year": int(year),
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(route)),
        "routed_ratio": float(np.mean(route)),
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "fixed": int(np.sum(~base_correct & final_correct)),
        "broken": int(np.sum(base_correct & ~final_correct)),
        "net": int(np.sum(final_correct) - np.sum(base_correct)),
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path(
        "artifacts/p1-features/features.csv"
    ))
    parser.add_argument("--schema", type=Path, default=Path(
        "artifacts/p1-features/feature_schema.json"
    ))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--future-expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/"
        "future_predictions_diagnostic.csv"
    ))
    parser.add_argument("--future-expert-column", default=DEFAULT_EXPERT_COLUMN)
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-future-v1"
    ))
    args = parser.parse_args()
    years = future_base.parse_years(args.years)
    if years != YEARS:
        raise RuntimeError("future years protocol changed")
    if args.inner_folds < 2 or args.jobs == 0 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid inner folds, jobs, or budget")
    args.output.mkdir(parents=True, exist_ok=True)

    print("Frozen 2018-2021 classifier-committee diagnostic", flush=True)
    print("Training every member and calibrator on 2017 only.", flush=True)
    manifest, source_rows, clean_rows, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    source_y = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    if len(source_y) != 799:
        raise RuntimeError("expected fixed 799-APK source cohort")
    requested = {str(year) for year in years}
    target_rows = [
        row for row in clean_rows
        if str(row.get("metadata_year", "")).strip() in requested
    ]
    target_year = np.asarray(
        [int(row["metadata_year"]) for row in target_rows], dtype=np.int64
    )
    if len(target_rows) != 1989 or tuple(sorted(set(target_year.tolist()))) != years:
        raise RuntimeError(
            "unexpected fixed future cohort: n=" + str(len(target_rows))
        )
    label_gate = FutureLabelGate(target_rows)

    # No future label is exposed to feature construction, training, calibration,
    # score computation, routing, or final-probability construction below.
    x_source, x_target, global_names = mv.global_matrices(
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    if len(global_names) != 24:
        raise RuntimeError("committee did not receive exact Global24")
    fitted = fit_calibrated_committee(
        x_source,
        source_y,
        x_target,
        inner_folds=args.inner_folds,
        split_seed=args.seed + 800000,
        model_seed=args.seed + 900000,
        jobs=args.jobs,
    )
    member_probability = fitted.target_probability
    detail = committee_scores(member_probability)
    frozen, expert_column = future_base.align_frozen_future(
        args.future_expert_predictions,
        target_rows,
        target_year,
        args.future_expert_column,
    )
    score_by_method = {
        "A25_YearTop25": frozen["a25_score"],
        "Committee_PE_YearTop25": detail["predictive_entropy"],
        "Committee_MI_YearTop25": detail["mutual_information"],
    }
    route_by_method = {
        method: groupwise_top_fraction(score, target_year, args.budget)
        for method, score in score_by_method.items()
    }
    final_by_method = {
        method: np.where(route, frozen["expert"], frozen["base"])
        for method, route in route_by_method.items()
    }
    frozen_payload = np.column_stack([
        frozen["base"],
        frozen["expert"],
        member_probability,
        detail["p_bar"],
        detail["predictive_entropy"],
        detail["mutual_information"],
        *[route_by_method[method].astype(float) for method in ROUTE_NAMES],
        *[final_by_method[method] for method in ROUTE_NAMES],
    ])
    if not np.isfinite(frozen_payload).all():
        raise RuntimeError("non-finite future probability/score/route payload")

    # Executable leakage barrier: labels become accessible only after all
    # probabilities, scores, routes, and final probabilities are immutable here.
    label_gate.freeze_predictions_and_routes()
    target_y = label_gate.labels(mv.v4.audit.future.label_of)

    annual_rows: list[dict] = []
    for year in years:
        selected = target_year == year
        annual_rows.append(final_metric_row(
            "Global24",
            target_y[selected],
            frozen["base"][selected],
            frozen["base"][selected],
            np.zeros(int(np.sum(selected)), dtype=bool),
            year=year,
        ))
        for method in ROUTE_NAMES:
            route_stats = routing_metrics(
                method,
                target_y[selected],
                frozen["base"][selected],
                frozen["expert"][selected],
                route_by_method[method][selected],
                score_by_method[method][selected],
            )
            final_stats = final_metric_row(
                method,
                target_y[selected],
                final_by_method[method][selected],
                frozen["base"][selected],
                route_by_method[method][selected],
                year=year,
            )
            annual_rows.append({
                **final_stats,
                **{key: value for key, value in route_stats.items()
                   if key not in {"method", "n", "routed_n", "routed_ratio",
                                  "fixed", "broken", "net"}},
            })

    base_correct = (frozen["base"] >= 0.5) == target_y
    expert_correct = (frozen["expert"] >= 0.5) == target_y
    repairable = ~base_correct & expert_correct
    harmful = base_correct & ~expert_correct
    unrepairable = ~base_correct & ~expert_correct
    raw_a25 = 1.0 - 2.0 * np.abs(frozen["base"] - 0.5)
    sample_rows: list[dict] = []
    for index, row in enumerate(target_rows):
        sample_rows.append({
            "sha256": mv.norm_sha(row["sha256"]),
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "p_stage1": float(frozen["base"][index]),
            "p_frozen_bounded_expert": float(frozen["expert"][index]),
            "p_lr": float(member_probability[index, 0]),
            "p_rf": float(member_probability[index, 1]),
            "p_xgb": float(member_probability[index, 2]),
            "p_svm": float(member_probability[index, 3]),
            "p_bar": float(detail["p_bar"][index]),
            "U_PE": float(detail["predictive_entropy"][index]),
            "U_MI": float(detail["mutual_information"][index]),
            "U_A25": float(frozen["a25_score"][index]),
            "U_A25_raw_ambiguity": float(raw_a25[index]),
            "route_A25": int(route_by_method["A25_YearTop25"][index]),
            "route_PE": int(route_by_method["Committee_PE_YearTop25"][index]),
            "route_MI": int(route_by_method["Committee_MI_YearTop25"][index]),
            "repairable": int(repairable[index]),
            "harmful": int(harmful[index]),
            "unrepairable": int(unrepairable[index]),
            "final_probability_A25": float(
                final_by_method["A25_YearTop25"][index]
            ),
            "final_probability_PE": float(
                final_by_method["Committee_PE_YearTop25"][index]
            ),
            "final_probability_MI": float(
                final_by_method["Committee_MI_YearTop25"][index]
            ),
            "diagnostic_only": True,
        })

    future_member_rows: list[dict] = []
    future_diversity_rows: list[dict] = []
    for year in years:
        selected = target_year == year
        future_member_rows.extend(member_metric_rows(
            target_y[selected], member_probability[selected], scope="annual_future", fold=year
        ))
        future_diversity_rows.extend(diversity_rows(
            member_probability[selected], scope="annual_future", fold=year
        ))

    calibration_audit = [{
        "training_scope": "full_2017",
        "future_labels_received": False,
        **row,
    } for row in fitted.audit]
    write_csv(args.output / "future_annual_metrics.csv", annual_rows)
    write_csv(args.output / "future_sample_scores.csv", sample_rows)
    write_csv(args.output / "future_member_metrics_diagnostic.csv", future_member_rows)
    write_csv(args.output / "future_member_diversity.csv", future_diversity_rows)
    write_csv(args.output / "calibration_audit.csv", calibration_audit)

    route_counts = {
        method: {
            str(year): int(np.sum(route_by_method[method][target_year == year]))
            for year in years
        }
        for method in ROUTE_NAMES
    }
    protocol = {
        "experiment": "CARE-Droid diverse classifier committee future diagnostic v1",
        "diagnostic_only": True,
        "source_year": 2017,
        "source_n": int(len(source_y)),
        "future_years": list(years),
        "future_n": int(len(target_rows)),
        "changed_component": "routing only",
        "official_stage1_probability": "frozen Global24 L2 probability",
        "committee_mean_used_as_stage1_probability": False,
        "committee_members": list(MEMBER_NAMES),
        "global24_columns": list(global_names),
        "calibration": "2017-only inner OOF Platt sigmoid",
        "future_labels_used_for_training_calibration_scoring_or_routing": False,
        "future_labels_first_used_after_probabilities_scores_routes_and_finals_froze": True,
        "budget": float(args.budget),
        "route_counts_by_year": route_counts,
        "future_expert_predictions": str(args.future_expert_predictions),
        "future_expert_column": expert_column,
        "stage2": "frozen CARE RGCondAttn Dedup BoundedCorrection",
        "direct_stage2_classification_included": False,
        "warning": (
            "Future years are diagnostic-only and were not used for model or router selection."
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    annual_lookup = {
        (int(row["year"]), str(row["method"])): row for row in annual_rows
    }
    lines = [
        "CARE-Droid diverse classifier committee uncertainty router v1",
        "",
        "Future diagnostic: frozen 2018-2021",
        "All four members and Platt calibrators were trained on 2017 only.",
        "Official Stage I remains frozen Global24 L2; p_bar is never used as Stage I.",
        f"Stage-II budget: {100.0 * args.budget:.2f}% per future year.",
        "",
    ]
    for year in years:
        lines.append(str(year))
        for method in ("Global24", *ROUTE_NAMES):
            row = annual_lookup[(year, method)]
            if method == "Global24":
                lines.append(
                    f"  {method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
                    f"F1={row['F1']:.6f}"
                )
            else:
                lines.append(
                    f"  {method}: routed={row['routed_n']}/{row['n']} "
                    f"error_recall={row['error_recall']:.6f} "
                    f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
                    f"RER@25={row['repairable_error_recall_at_budget']:.6f} "
                    f"HarmfulCapture={row['harmful_capture']:.6f} "
                    f"UnrepairableCapture={row['unrepairable_capture']:.6f} "
                    f"accepted_risk={row['accepted_risk']:.6f} "
                    f"BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
                    f"F1={row['F1']:.6f} fixed={row['fixed']} "
                    f"broken={row['broken']} net={row['net']}"
                )
    lines.extend([
        "",
        "IMPORTANT:",
        "Future years are diagnostic-only and were not used for model or router selection.",
    ])
    text = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(text, encoding="utf-8")
    print("Done:", args.output, flush=True)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
