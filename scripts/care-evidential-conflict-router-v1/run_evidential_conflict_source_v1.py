#!/usr/bin/env python3
"""Strict 2017 source audit of conflict-aware multi-view evidential routing."""

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
    raise SystemExit("Missing established CARE dependencies: " + str(exc))

from evidential_conflict_core import (  # noqa: E402
    AUX_VIEW_NAMES,
    conflict_aware_fusion,
    exact_top_fraction,
    fit_four_evidential_views,
    resolve_device,
    routing_diagnostics,
)


FROZEN_EXPERT_NAME = "CARE_RGCondAttn_Dedup_FixKeep"
METHOD_TO_SCORE = {
    "EVI_Ignorance25": "ignorance",
    "EVI_ConflictMean25": "conflict_mean",
    "EVI_ConflictMax25": "conflict_max",
    "EVI_ConflictMean_NoDiscount25": "conflict_mean_no_discount",
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


def fusion_scores(evidence: np.ndarray, gamma: float) -> tuple[dict, dict]:
    mean_result = conflict_aware_fusion(
        evidence, conflict_aggregator="mean", reliability_gamma=gamma
    )
    max_result = conflict_aware_fusion(
        evidence, conflict_aggregator="max", reliability_gamma=gamma
    )
    no_discount = conflict_aware_fusion(
        evidence, conflict_aggregator="mean", reliability_gamma=0.0
    )
    score = {
        "ignorance": mean_result["ignorance"],
        "conflict_mean": mean_result["uncertainty"],
        "conflict_max": max_result["uncertainty"],
        "conflict_mean_no_discount": no_discount["uncertainty"],
    }
    detail = {
        "mean": mean_result,
        "max": max_result,
        "no_discount": no_discount,
    }
    return score, detail


def per_fold_metric_rows(
    folds: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    final_probability: dict[str, np.ndarray],
    routes: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        for method, probability in final_probability.items():
            route = np.zeros(int(np.sum(selected)), dtype=bool)
            if method != "Global24":
                route = routes[method.split("__", 1)[0]][selected]
            metric = mv.v4.v2.metric_rows(
                labels[selected],
                {method: probability[selected]},
                base[selected],
                route,
            )[0]
            rows.append({"fold": int(fold), **metric})
    return rows


def overlap_rows(
    labels: np.ndarray, base: np.ndarray, routes: dict[str, np.ndarray]
) -> list[dict]:
    error = (base >= 0.5) != labels
    high_error = error & (np.maximum(base, 1.0 - base) >= 0.8)
    anchor = routes["A25"]
    rows: list[dict] = []
    for method, route in routes.items():
        if method == "A25":
            continue
        rows.append({
            "method_a": "A25",
            "method_b": method,
            "routed_overlap_n": int(np.sum(anchor & route)),
            "a_only_routed_n": int(np.sum(anchor & ~route)),
            "b_only_routed_n": int(np.sum(~anchor & route)),
            "captured_error_overlap_n": int(np.sum(anchor & route & error)),
            "a_only_captured_error_n": int(np.sum(anchor & ~route & error)),
            "b_only_captured_error_n": int(np.sum(~anchor & route & error)),
            "b_only_high_confidence_error_n": int(np.sum(~anchor & route & high_error)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dimension", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--annealing-epochs", type=int, default=100)
    parser.add_argument("--reliability-gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-evidential-conflict-router-source-v1"
    ))
    args = parser.parse_args()
    if not 0.0 < args.budget < 1.0 or args.epochs < 1:
        raise SystemExit("invalid budget or epoch count")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    years = {str(row.get("metadata_year", "")).strip() for row in source_rows}
    if years != {"2017"}:
        raise RuntimeError("source-only guard failed: " + repr(years))
    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)

    route_names = ("A25", *METHOD_TO_SCORE.keys())
    routes = {name: np.zeros(len(labels), dtype=bool) for name in route_names}
    scores = {name: np.full(len(labels), np.nan) for name in route_names}
    evidence_all = np.full((len(labels), len(AUX_VIEW_NAMES), 2), np.nan)
    view_probability_all = np.full((len(labels), len(AUX_VIEW_NAMES)), np.nan)
    view_uncertainty_all = np.full((len(labels), len(AUX_VIEW_NAMES)), np.nan)
    view_reliability_all = np.full((len(labels), len(AUX_VIEW_NAMES)), np.nan)
    conflict_mean_all = np.full(len(labels), np.nan)
    conflict_max_all = np.full(len(labels), np.nan)
    fused_probability_all = np.full(len(labels), np.nan)
    training_audit: list[dict] = []
    per_fold_routing: list[dict] = []

    print("Strict 2017 multi-view evidential conflict audit", flush=True)
    print("Evidential device:", device, flush=True)
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in train_index]
        validation_rows = [source_rows[index] for index in validation_index]
        x_train, x_validation, names = mv.global_matrices(
            train_rows, validation_rows, compact_by_sha, compact_schema, original_names
        )
        groups = mv.group_indices(names)
        _, validation_evidence, audit = fit_four_evidential_views(
            x_train,
            labels[train_index],
            x_validation,
            groups,
            seed=args.seed + 10000 * int(fold),
            device=device,
            hidden_dimension=args.hidden_dimension,
            dropout=args.dropout,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            kl_weight=args.kl_weight,
            annealing_epochs=args.annealing_epochs,
        )
        for row in audit:
            training_audit.append({"fold": int(fold), **row})
        fold_score, detail = fusion_scores(validation_evidence, args.reliability_gamma)
        fold_routes = {"A25": frozen["a25_route"][validation_index]}
        fold_scores = {"A25": frozen["a25_score"][validation_index]}
        for method, score_name in METHOD_TO_SCORE.items():
            fold_routes[method] = exact_top_fraction(
                fold_score[score_name], args.budget
            )
            fold_scores[method] = fold_score[score_name]
        for method in route_names:
            routes[method][validation_index] = fold_routes[method]
            scores[method][validation_index] = fold_scores[method]
            diagnostic = routing_diagnostics(
                method,
                labels[validation_index],
                frozen["base"][validation_index],
                fold_routes[method],
                fold_scores[method],
            )
            per_fold_routing.append({"fold": int(fold), **diagnostic})

        evidence_all[validation_index] = validation_evidence
        view_probability_all[validation_index] = detail["mean"]["view_probability"]
        view_uncertainty_all[validation_index] = detail["mean"]["view_uncertainty"]
        view_reliability_all[validation_index] = detail["mean"]["view_reliability"]
        conflict_mean_all[validation_index] = detail["mean"]["conflict"]
        conflict_max_all[validation_index] = detail["max"]["conflict"]
        fused_probability_all[validation_index] = detail["mean"]["fused_probability"]
        print(
            f"  fold {int(fold)} train={len(train_index)} validation={len(validation_index)}",
            flush=True,
        )

    arrays = [
        *scores.values(),
        evidence_all,
        view_probability_all,
        view_uncertainty_all,
        view_reliability_all,
        conflict_mean_all,
        conflict_max_all,
        fused_probability_all,
    ]
    if any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError("incomplete evidential source output")

    routing_summary = [
        routing_diagnostics(method, labels, frozen["base"], routes[method], scores[method])
        for method in route_names
    ]
    final_probability = {"Global24": frozen["base"]}
    for method in route_names:
        final_probability[method + "__" + FROZEN_EXPERT_NAME] = np.where(
            routes[method], frozen["expert"], frozen["base"]
        )
    metric_summary: list[dict] = []
    for method, probability in final_probability.items():
        route = np.zeros(len(labels), dtype=bool)
        if method != "Global24":
            route = routes[method.split("__", 1)[0]]
        metric_summary.extend(
            mv.v4.v2.metric_rows(labels, {method: probability}, frozen["base"], route)
        )
    per_fold_metrics = per_fold_metric_rows(
        folds, labels, frozen["base"], final_probability, routes
    )

    prediction_rows: list[dict] = []
    for index, row in enumerate(manifest):
        item = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenExpert__prob": float(frozen["expert"][index]),
            "evidential_fused_probability": float(fused_probability_all[index]),
            "evidential_conflict_mean": float(conflict_mean_all[index]),
            "evidential_conflict_max": float(conflict_max_all[index]),
        }
        for column, view in enumerate(AUX_VIEW_NAMES):
            item[f"{view}__evidence_benign"] = float(evidence_all[index, column, 0])
            item[f"{view}__evidence_malware"] = float(evidence_all[index, column, 1])
            item[f"{view}__prob_malware"] = float(view_probability_all[index, column])
            item[f"{view}__uncertainty"] = float(view_uncertainty_all[index, column])
            item[f"{view}__reliability"] = float(view_reliability_all[index, column])
        for method in route_names:
            item[method + "__score"] = float(scores[method][index])
            item[method + "__routed"] = int(routes[method][index])
            item[method + "__final_prob"] = float(
                final_probability[method + "__" + FROZEN_EXPERT_NAME][index]
            )
        prediction_rows.append(item)

    write_csv(args.output / "routing_summary.csv", routing_summary)
    write_csv(args.output / "source_metrics.csv", metric_summary)
    write_csv(args.output / "source_per_fold_routing.csv", per_fold_routing)
    write_csv(args.output / "source_per_fold_metrics.csv", per_fold_metrics)
    write_csv(args.output / "training_audit.csv", training_audit)
    write_csv(args.output / "a25_overlap.csv", overlap_rows(labels, frozen["base"], routes))
    write_csv(args.output / "source_predictions.csv", prediction_rows)

    architecture = {
        "experiment": "CARE-Droid multi-view evidential conflict router v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "future_data_loaded": False,
        "changed_component": "route only",
        "base": "frozen Global24 OOF probability",
        "expert": "frozen deduplicated bounded RGCondAttn probability",
        "auxiliary_views": list(AUX_VIEW_NAMES),
        "evidence_head": "two-output MLP with softplus nonnegative evidence",
        "opinion": "alpha=e+1, p=alpha/S, u=2/S",
        "loss": "evidential squared-error Bayes risk plus annealed KL-to-uniform",
        "reliability": "sample-specific exp(-gamma*view_uncertainty)",
        "conflict": "pairwise opposing belief mass after reliability discount",
        "evidence_contraction": "U=1-(1-U_ignorance)*(1-K), then reconstruct e_star",
        "route_budget": float(args.budget),
        "primary": "EVI_ConflictMean25 versus A25",
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
        "CARE-Droid multi-view evidential conflict router v1",
        "",
        f"Device: {device}",
        "Only routing changes; Global24 and the local expert are frozen.",
        f"Stage-II budget: {100.0 * args.budget:.2f}%.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for method in route_names:
        row = route_lookup[method]
        lines.append(
            f"{method}: error_recall={row['error_recall']:.6f} "
            f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
            f"precision={row['route_error_precision']:.6f} "
            f"accepted_risk={row['accepted_risk']:.6f}"
        )
    lines.extend(["", "End-to-end metrics with the same frozen local expert:"])
    for method, row in metric_lookup.items():
        lines.append(
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    lines.extend([
        "",
        "Primary comparison: EVI_ConflictMean25 versus A25.",
        "EVI_Ignorance25 removes conflict; EVI_ConflictMax25 changes aggregation;",
        "EVI_ConflictMean_NoDiscount25 removes sample-specific reliability discount.",
        "Do not choose variants using future-year labels.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
