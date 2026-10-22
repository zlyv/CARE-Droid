#!/usr/bin/env python3
"""Strict 2017 audit of reliability-aware directional conflict routing.

Only the Stage-I-to-Stage-II route changes.  The frozen Global24 probability,
local expert, outer folds, labels, and 25 percent analysis budget are reused.
Every auxiliary view and its PPV/NPV reliability is learned without the outer
validation rows.
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
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import run_multiview_conformal_router_source_v1 as mv
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE multi-view dependencies: " + str(exc))

from directional_conflict_core import (  # noqa: E402
    AUX_VIEW_NAMES,
    build_route_scores,
    estimate_class_conditional_reliability,
    exact_top_fraction,
    routing_diagnostics,
)


FROZEN_EXPERT_NAME = "CARE_RGCondAttn_Dedup_FixKeep"
ROUTE_SCORE_TO_METHOD = {
    "ConflictMax": "ConflictMax25",
    "SoftOR": "SoftOR25",
    "SoftOR_NoReliability": "SoftOR_NoReliability25",
    "SoftOR_Top2Mean": "SoftOR_Top2Mean25",
}


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


def per_fold_metric_rows(
    folds: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    final_probability: dict[str, np.ndarray],
    route_all: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        for method, probability in final_probability.items():
            route = np.zeros(int(np.sum(selected)), dtype=bool)
            if method != "Global24":
                route = route_all[method.split("__", 1)[0]][selected]
            metric = mv.v4.v2.metric_rows(
                labels[selected],
                {method: probability[selected]},
                base[selected],
                route,
            )[0]
            rows.append({"fold": int(fold), **metric})
    return rows


def overlap_rows(
    labels: np.ndarray,
    base: np.ndarray,
    routes: dict[str, np.ndarray],
) -> list[dict]:
    error = (np.asarray(base) >= 0.5) != np.asarray(labels)
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= 0.8)
    anchor = routes["A25"]
    rows: list[dict] = []
    for name, route in routes.items():
        if name == "A25":
            continue
        rows.append({
            "method_a": "A25",
            "method_b": name,
            "routed_overlap_n": int(np.sum(anchor & route)),
            "a_only_routed_n": int(np.sum(anchor & ~route)),
            "b_only_routed_n": int(np.sum(~anchor & route)),
            "captured_error_overlap_n": int(np.sum(anchor & route & error)),
            "a_only_captured_error_n": int(np.sum(anchor & ~route & error)),
            "b_only_captured_error_n": int(np.sum(~anchor & route & error)),
            "b_only_high_confidence_error_n": int(
                np.sum(~anchor & route & high_error)
            ),
        })
    return rows


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
        "--expert-predictions",
        type=Path,
        default=Path(
            "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
        ),
    )
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--reliability-smoothing", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/care-directional-conflict-router-source-v1"),
    )
    args = parser.parse_args()
    if args.inner_folds < 2 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid inner folds or budget")
    if args.reliability_smoothing <= 0.0:
        raise SystemExit("--reliability-smoothing must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    source_years = {str(row.get("metadata_year", "")).strip() for row in source_rows}
    if source_years != {"2017"}:
        raise RuntimeError("source-only guard failed: " + repr(source_years))
    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)

    route_names = ("A25", *ROUTE_SCORE_TO_METHOD.values())
    route_all = {name: np.zeros(len(labels), dtype=bool) for name in route_names}
    score_all = {name: np.full(len(labels), np.nan) for name in route_names}
    view_probability_all = {
        view: np.full(len(labels), np.nan) for view in mv.VIEW_NAMES
    }
    conflict_all = {
        view: np.full(len(labels), np.nan) for view in AUX_VIEW_NAMES
    }
    reliability_rows: list[dict] = []
    fold_audit: list[dict] = []
    per_fold_routing: list[dict] = []

    print("Strict 2017 reliability-aware directional conflict audit", flush=True)
    for fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != fold)
        outer_validation = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in outer_train]
        validation_rows = [source_rows[index] for index in outer_validation]
        train_y = labels[outer_train]
        print(
            f"  fold {int(fold)} train={len(outer_train)} "
            f"validation={len(outer_validation)}",
            flush=True,
        )

        train_oof, oof_names = mv.proper_oof_views(
            train_rows,
            train_y,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.seed + 10000 * int(fold),
        )
        train_oof_matrix = mv.probability_matrix(train_oof)
        calibrated_train_auxiliary = np.zeros(
            (len(train_y), len(AUX_VIEW_NAMES)), dtype=np.float64
        )
        calibrators = []
        for column, view in enumerate(AUX_VIEW_NAMES):
            raw = train_oof_matrix[:, column + 1]
            calibrator = mv.platt_fit(
                raw, train_y, args.seed + 20000 * int(fold) + column
            )
            calibrators.append(calibrator)
            calibrated_train_auxiliary[:, column] = mv.platt_predict(
                calibrator, raw
            )
        positive, negative, reliability_audit = estimate_class_conditional_reliability(
            calibrated_train_auxiliary, train_y, args.reliability_smoothing
        )
        for column, row in enumerate(reliability_audit):
            reliability_rows.append({
                "fold": int(fold),
                **row,
                "raw_oof_nll": mv.nll(train_y, train_oof_matrix[:, column + 1]),
                "calibrated_oof_nll": mv.nll(
                    train_y, calibrated_train_auxiliary[:, column]
                ),
            })

        x_train, x_validation, names = mv.global_matrices(
            train_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        if names != oof_names:
            raise RuntimeError("Global24 names changed between OOF and final fit")
        groups = mv.group_indices(names)
        validation_views = mv.fit_raw_views(
            x_train,
            train_y,
            x_validation,
            groups,
            args.seed + 40000 * int(fold),
        )
        validation_matrix = mv.probability_matrix(validation_views)
        auxiliary_all_probability = validation_matrix[:, 0].copy()
        validation_matrix[:, 0] = frozen["base"][outer_validation]
        for column, calibrator in enumerate(calibrators):
            validation_matrix[:, column + 1] = mv.platt_predict(
                calibrator, validation_matrix[:, column + 1]
            )

        scores, per_view_conflict = build_route_scores(
            frozen["base"][outer_validation],
            validation_matrix[:, 1:],
            positive,
            negative,
        )
        fold_routes = {"A25": frozen["a25_route"][outer_validation]}
        fold_scores = {"A25": frozen["a25_score"][outer_validation]}
        for score_name, method in ROUTE_SCORE_TO_METHOD.items():
            fold_routes[method] = exact_top_fraction(scores[score_name], args.budget)
            fold_scores[method] = scores[score_name]

        for method in route_names:
            route_all[method][outer_validation] = fold_routes[method]
            score_all[method][outer_validation] = fold_scores[method]
            row = routing_diagnostics(
                method,
                labels[outer_validation],
                frozen["base"][outer_validation],
                fold_routes[method],
                fold_scores[method],
            )
            per_fold_routing.append({"fold": int(fold), **row})
        for column, view in enumerate(mv.VIEW_NAMES):
            view_probability_all[view][outer_validation] = validation_matrix[:, column]
        for column, view in enumerate(AUX_VIEW_NAMES):
            conflict_all[view][outer_validation] = per_view_conflict[:, column]

        fold_audit.append({
            "fold": int(fold),
            "outer_train_n": int(len(outer_train)),
            "outer_validation_n": int(len(outer_validation)),
            "auxiliary_all_vs_frozen_base_label_disagreement": float(np.mean(
                (auxiliary_all_probability >= 0.5)
                != (frozen["base"][outer_validation] >= 0.5)
            )),
            "future_rows_loaded": False,
        })

    for name, collection in (
        ("route scores", score_all),
        ("view probabilities", view_probability_all),
        ("per-view conflicts", conflict_all),
    ):
        if any(np.any(~np.isfinite(value)) for value in collection.values()):
            raise RuntimeError("incomplete " + name)

    routing_summary = [
        routing_diagnostics(
            method, labels, frozen["base"], route_all[method], score_all[method]
        )
        for method in route_names
    ]
    final_probability = {"Global24": frozen["base"]}
    for method in route_names:
        final_probability[method + "__" + FROZEN_EXPERT_NAME] = mv.routed_probability(
            frozen["base"], frozen["expert"], route_all[method]
        )

    metric_summary: list[dict] = []
    for method, probability in final_probability.items():
        route = np.zeros(len(labels), dtype=bool)
        if method != "Global24":
            route = route_all[method.split("__", 1)[0]]
        metric_summary.extend(
            mv.v4.v2.metric_rows(
                labels, {method: probability}, frozen["base"], route
            )
        )
    per_fold_metrics = per_fold_metric_rows(
        folds, labels, frozen["base"], final_probability, route_all
    )
    overlap = overlap_rows(labels, frozen["base"], route_all)

    prediction_rows: list[dict] = []
    for index, row in enumerate(manifest):
        item = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenExpert__prob": float(frozen["expert"][index]),
        }
        for view in mv.VIEW_NAMES:
            item["view_" + view + "__prob"] = float(
                view_probability_all[view][index]
            )
        for view in AUX_VIEW_NAMES:
            item["conflict_" + view] = float(conflict_all[view][index])
        for method in route_names:
            item[method + "__score"] = float(score_all[method][index])
            item[method + "__routed"] = int(route_all[method][index])
            item[method + "__final_prob"] = float(
                final_probability[method + "__" + FROZEN_EXPERT_NAME][index]
            )
        prediction_rows.append(item)

    write_csv(args.output / "routing_summary.csv", routing_summary)
    write_csv(args.output / "source_metrics.csv", metric_summary)
    write_csv(args.output / "source_per_fold_routing.csv", per_fold_routing)
    write_csv(args.output / "source_per_fold_metrics.csv", per_fold_metrics)
    write_csv(args.output / "class_conditional_reliability.csv", reliability_rows)
    write_csv(args.output / "a25_overlap.csv", overlap)
    write_csv(args.output / "fold_audit.csv", fold_audit)
    write_csv(args.output / "source_predictions.csv", prediction_rows)

    architecture = {
        "experiment": "CARE-Droid reliability-aware directional conflict router v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "future_data_loaded": False,
        "changed_component": "Stage-I-to-Stage-II route only",
        "base": "frozen Global24 outer-fold OOF probability",
        "expert": "frozen deduplicated bounded RGCondAttn probability",
        "route_budget": float(args.budget),
        "auxiliary_views": list(AUX_VIEW_NAMES),
        "reliability": {
            "positive": "smoothed PPV P(y=1 | view predicts 1)",
            "negative": "smoothed NPV P(y=0 | view predicts 0)",
            "source": "outer-training OOF predictions only",
            "smoothing": float(args.reliability_smoothing),
        },
        "auxiliary_probability_calibration": (
            "per-view Platt calibration fitted on outer-training OOF predictions"
        ),
        "scores": {
            "Boundary": "1 - 2*abs(Global24_probability - 0.5)",
            "ConflictMax": "maximum reliable view evidence opposing Global24 label",
            "SoftOR": "1 - (1-Boundary)*(1-ConflictMax)",
            "SoftOR_NoReliability": "ablation with all view reliabilities equal to one",
            "SoftOR_Top2Mean": "aggregation ablation using mean of two largest conflicts",
        },
        "view_sufficiency_q": "not used in v1; q_iv is fixed to one",
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_lookup = {row["method"]: row for row in routing_summary}
    metric_lookup = {row["method"]: row for row in metric_summary}
    lines = [
        "CARE-Droid reliability-aware directional conflict router v1",
        "",
        "Only routing changes; Global24 and the local expert remain frozen.",
        "Every auxiliary view and PPV/NPV reliability excludes outer validation rows.",
        f"Stage-II budget: {100.0 * args.budget:.2f}%.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for method in route_names:
        row = route_lookup[method]
        lines.append(
            f"{method}: routed={row['routed_n']}/{row['n']} "
            f"error_recall={row['error_recall']:.6f} "
            f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
            f"precision={row['route_error_precision']:.6f} "
            f"accepted_risk={row['accepted_risk']:.6f}"
        )
    lines.extend(["", "End-to-end metrics with the same frozen local expert:"])
    for method, row in metric_lookup.items():
        lines.append(
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} "
            f"broken={row['broken']} net={row['net']}"
        )
    lines.extend([
        "",
        "Primary comparison: SoftOR25 versus A25 at the same 25% budget.",
        "ConflictMax25 tests high-confidence directional disagreement alone.",
        "SoftOR_NoReliability25 tests whether class-conditional PPV/NPV matters.",
        "SoftOR_Top2Mean25 tests max versus a less single-view-sensitive aggregator.",
        "Do not select variants using 2018-2021 labels.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
