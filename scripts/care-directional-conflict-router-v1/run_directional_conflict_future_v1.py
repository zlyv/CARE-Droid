#!/usr/bin/env python3
"""Frozen 2018-2021 diagnostic of directional conflict routing.

All auxiliary view models, class-conditional reliabilities, and source
thresholds are learned from 2017 only.  Future labels are first materialized
after scores, routes, and final probabilities have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-malcertain-router-future-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-future-v1",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_multiview_conformal_router_source_v1 as mv
    import run_care_malcertain_router_future_v1 as future_base
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE router dependencies: " + str(exc))

from directional_conflict_core import (  # noqa: E402
    AUX_VIEW_NAMES,
    build_route_scores,
    estimate_class_conditional_reliability,
    groupwise_top_fraction,
    routing_diagnostics,
    source_quantile_threshold,
)


SCORE_NAMES = (
    "ConflictMax",
    "SoftOR",
    "SoftOR_NoReliability",
    "SoftOR_Top2Mean",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/p1-features/features.csv")
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json")
    )
    parser.add_argument(
        "--p72-oof",
        type=Path,
        default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"),
    )
    parser.add_argument(
        "--source-expert-predictions",
        type=Path,
        default=Path(
            "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
        ),
    )
    parser.add_argument(
        "--future-expert-predictions",
        type=Path,
        default=Path(
            "artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/"
            "future_predictions_diagnostic.csv"
        ),
    )
    parser.add_argument(
        "--future-expert-column", default=future_base.DEFAULT_EXPERT_COLUMN
    )
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--reliability-smoothing", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/care-directional-conflict-router-future-v1"),
    )
    args = parser.parse_args()
    years = future_base.parse_years(args.years)
    if args.inner_folds < 2 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid inner folds or budget")
    if args.reliability_smoothing <= 0.0:
        raise SystemExit("--reliability-smoothing must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    print("Frozen 2018-2021 directional conflict router audit", flush=True)
    print("Training every reliability component on 2017 only.", flush=True)
    (
        manifest,
        source_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = mv.v4.audit.load_inputs(args)
    source_y = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    if len(source_y) != 799:
        raise RuntimeError("expected the fixed 799-APK 2017 source cohort")

    requested = {str(year) for year in years}
    target_rows = [
        row
        for row in clean_rows
        if str(row.get("metadata_year", "")).strip() in requested
    ]
    target_year = np.asarray(
        [int(row["metadata_year"]) for row in target_rows], dtype=np.int64
    )
    if len(target_rows) != 1989 or tuple(sorted(set(target_year.tolist()))) != years:
        raise RuntimeError(
            "unexpected frozen future cohort: n="
            + str(len(target_rows))
            + " years="
            + repr(sorted(set(target_year.tolist())))
        )

    # Cross-fitted source predictions estimate view reliabilities and thresholds.
    source_oof, oof_names = mv.proper_oof_views(
        source_rows,
        source_y,
        compact_by_sha,
        compact_schema,
        original_names,
        args.inner_folds,
        args.seed,
    )
    source_matrix = mv.probability_matrix(source_oof)
    calibrated_source_auxiliary = np.zeros(
        (len(source_y), len(AUX_VIEW_NAMES)), dtype=np.float64
    )
    calibrators = []
    for column, view in enumerate(AUX_VIEW_NAMES):
        raw = source_matrix[:, column + 1]
        calibrator = mv.platt_fit(raw, source_y, args.seed + 20000 + column)
        calibrators.append(calibrator)
        calibrated_source_auxiliary[:, column] = mv.platt_predict(calibrator, raw)
    positive, negative, reliability_audit = estimate_class_conditional_reliability(
        calibrated_source_auxiliary, source_y, args.reliability_smoothing
    )
    for column, row in enumerate(reliability_audit):
        row["raw_oof_nll"] = mv.nll(source_y, source_matrix[:, column + 1])
        row["calibrated_oof_nll"] = mv.nll(
            source_y, calibrated_source_auxiliary[:, column]
        )
    source_frozen = mv.load_frozen_predictions(
        args.source_expert_predictions, manifest
    )
    source_scores, source_conflict = build_route_scores(
        source_frozen["base"], calibrated_source_auxiliary, positive, negative
    )
    source_threshold = {
        name: source_quantile_threshold(source_scores[name], args.budget)
        for name in SCORE_NAMES
    }

    # Fit each auxiliary view once on all source APKs and score future APKs.
    x_source, x_target, names = mv.global_matrices(
        source_rows,
        target_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    if names != oof_names:
        raise RuntimeError("Global24 feature names changed between source and future")
    groups = mv.group_indices(names)
    target_views = mv.fit_raw_views(
        x_source, source_y, x_target, groups, args.seed + 70000
    )
    target_matrix = mv.probability_matrix(target_views)
    auxiliary_all_probability = target_matrix[:, 0].copy()
    frozen, expert_column = future_base.align_frozen_future(
        args.future_expert_predictions,
        target_rows,
        target_year,
        args.future_expert_column,
    )
    target_matrix[:, 0] = frozen["base"]
    for column, calibrator in enumerate(calibrators):
        target_matrix[:, column + 1] = mv.platt_predict(
            calibrator, target_matrix[:, column + 1]
        )
    target_scores, target_conflict = build_route_scores(
        frozen["base"], target_matrix[:, 1:], positive, negative
    )

    # Freeze every score, route, and final probability before materializing labels.
    route_by_method: dict[str, np.ndarray] = {
        "A25_Frozen": frozen["a25_route"],
        "A25_YearTop25": groupwise_top_fraction(
            frozen["a25_score"], target_year, args.budget
        ),
    }
    score_by_method: dict[str, np.ndarray] = {
        "A25_Frozen": frozen["a25_score"],
        "A25_YearTop25": frozen["a25_score"],
    }
    for score_name in SCORE_NAMES:
        annual_name = score_name + "_YearTop25"
        source_name = score_name + "_SourceQ75"
        route_by_method[annual_name] = groupwise_top_fraction(
            target_scores[score_name], target_year, args.budget
        )
        route_by_method[source_name] = (
            target_scores[score_name] >= source_threshold[score_name]
        )
        score_by_method[annual_name] = target_scores[score_name]
        score_by_method[source_name] = target_scores[score_name]
    final_by_method = {
        method: np.where(route, frozen["expert"], frozen["base"])
        for method, route in route_by_method.items()
    }

    # Future labels are first accessed here.
    target_y = np.asarray(
        [mv.v4.audit.future.label_of(row) for row in target_rows], dtype=np.int64
    )

    annual_metrics: list[dict] = []
    annual_routing: list[dict] = []
    pooled_metrics = [
        future_base.metric_row(
            "Global24",
            target_y,
            frozen["base"],
            frozen["base"],
            np.zeros(len(target_y), dtype=bool),
            "pooled_2018_2021",
        )
    ]
    pooled_routing: list[dict] = []
    for year in years:
        selected = target_year == year
        annual_metrics.append(
            future_base.metric_row(
                "Global24",
                target_y[selected],
                frozen["base"][selected],
                frozen["base"][selected],
                np.zeros(int(np.sum(selected)), dtype=bool),
                "annual",
                year,
            )
        )
        for method, route in route_by_method.items():
            annual_metrics.append(
                future_base.metric_row(
                    method,
                    target_y[selected],
                    final_by_method[method][selected],
                    frozen["base"][selected],
                    route[selected],
                    "annual",
                    year,
                )
            )
            diagnostic = routing_diagnostics(
                method,
                target_y[selected],
                frozen["base"][selected],
                route[selected],
                score_by_method[method][selected],
            )
            annual_routing.append(
                {"scope": "annual", "year": int(year), **diagnostic, "diagnostic_only": True}
            )
    for method, route in route_by_method.items():
        pooled_metrics.append(
            future_base.metric_row(
                method,
                target_y,
                final_by_method[method],
                frozen["base"],
                route,
                "pooled_2018_2021",
            )
        )
        diagnostic = routing_diagnostics(
            method,
            target_y,
            frozen["base"],
            route,
            score_by_method[method],
        )
        pooled_routing.append(
            {"scope": "pooled_2018_2021", **diagnostic, "diagnostic_only": True}
        )

    prediction_rows: list[dict] = []
    for index, row in enumerate(target_rows):
        item = {
            "sha256": future_base.norm_sha(row["sha256"]),
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenExpert__prob": float(frozen["expert"][index]),
            "AuxiliaryAllView__prob": float(auxiliary_all_probability[index]),
            "A25__score": float(frozen["a25_score"][index]),
        }
        for column, view in enumerate(mv.VIEW_NAMES):
            item["view_" + view + "__prob"] = float(target_matrix[index, column])
        for column, view in enumerate(AUX_VIEW_NAMES):
            item["conflict_" + view] = float(target_conflict[index, column])
        for score_name in SCORE_NAMES:
            item[score_name + "__score"] = float(target_scores[score_name][index])
        for method, route in route_by_method.items():
            item[method + "__routed"] = int(route[index])
            item[method + "__prob"] = float(final_by_method[method][index])
        item["diagnostic_only"] = True
        prediction_rows.append(item)

    write_csv(args.output / "future_annual_metrics_diagnostic.csv", annual_metrics)
    write_csv(args.output / "future_annual_routing_diagnostic.csv", annual_routing)
    write_csv(args.output / "future_pooled_metrics_diagnostic.csv", pooled_metrics)
    write_csv(args.output / "future_pooled_routing_diagnostic.csv", pooled_routing)
    write_csv(args.output / "future_predictions_diagnostic.csv", prediction_rows)
    write_csv(args.output / "source_class_conditional_reliability.csv", reliability_audit)

    audit = {
        "experiment": "care-directional-conflict-router-future-v1",
        "diagnostic_only": True,
        "source_year": 2017,
        "source_n": int(len(source_y)),
        "future_years": list(years),
        "future_n": int(len(target_y)),
        "budget": float(args.budget),
        "seed": int(args.seed),
        "inner_folds": int(args.inner_folds),
        "auxiliary_views": list(AUX_VIEW_NAMES),
        "positive_reliability_ppv": {
            view: float(positive[index])
            for index, view in enumerate(AUX_VIEW_NAMES)
        },
        "negative_reliability_npv": {
            view: float(negative[index])
            for index, view in enumerate(AUX_VIEW_NAMES)
        },
        "source_thresholds": {
            name: float(value) for name, value in source_threshold.items()
        },
        "auxiliary_probability_calibration": (
            "per-view Platt calibration fitted on 2017 OOF predictions only"
        ),
        "future_expert_predictions": str(args.future_expert_predictions),
        "future_expert_column": expert_column,
        "future_labels_used_for_view_training_reliability_or_routing": False,
        "future_labels_first_used_after_scores_routes_and_final_probabilities_frozen": True,
        "primary_comparison": "A25_YearTop25 versus SoftOR_YearTop25",
        "view_sufficiency_q": "not used in v1; q_iv is fixed to one",
        "future_warning": (
            "2018-2021 were already inspected in this project; these are "
            "development diagnostics, not an untouched final test"
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_lookup = {
        (int(row["year"]), str(row["method"])): row for row in annual_metrics
    }
    route_lookup = {
        (int(row["year"]), str(row["method"])): row for row in annual_routing
    }
    primary_methods = (
        "Global24",
        "A25_YearTop25",
        "ConflictMax_YearTop25",
        "SoftOR_YearTop25",
        "SoftOR_NoReliability_YearTop25",
        "SoftOR_Top2Mean_YearTop25",
    )
    lines = [
        "CARE-Droid reliability-aware directional conflict future diagnostic v1",
        "",
        "2017 only: four auxiliary views, PPV/NPV reliability, source thresholds.",
        "2018-2021 labels are first used after all routes and probabilities freeze.",
        "WARNING: future years are diagnostic-only because they were inspected before.",
        "",
        "Primary equal-budget annual comparison:",
    ]
    for year in years:
        lines.append(str(year))
        for method in primary_methods:
            metric = metric_lookup[(year, method)]
            if method == "Global24":
                lines.append(
                    f"  {method}: BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} "
                    f"F1={metric['F1']:.6f}"
                )
            else:
                route = route_lookup[(year, method)]
                lines.append(
                    f"  {method}: routed={metric['routed_n']}/{metric['n']} "
                    f"error_recall={route['error_recall']:.6f} "
                    f"highconf_error_recall={route['high_confidence_error_recall']:.6f} "
                    f"accepted_risk={route['accepted_risk']:.6f} "
                    f"BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} "
                    f"F1={metric['F1']:.6f} fixed={metric['fixed']} "
                    f"broken={metric['broken']} net={metric['net']}"
                )
    lines.extend(["", "Source-threshold transport route ratios:"])
    for year in years:
        for score_name in SCORE_NAMES:
            method = score_name + "_SourceQ75"
            metric = metric_lookup[(year, method)]
            lines.append(
                f"  {year} {method}: routed_ratio={metric['routed_ratio']:.6f} "
                f"BA={metric['BA']:.6f} F1={metric['F1']:.6f}"
            )
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
