#!/usr/bin/env python3
"""Frozen 2018-2021 diagnostic of conflict-aware evidential routing."""

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
    raise SystemExit("Missing established CARE dependencies: " + str(exc))

from evidential_conflict_core import (  # noqa: E402
    AUX_VIEW_NAMES,
    fit_four_evidential_views,
    groupwise_top_fraction,
    resolve_device,
    routing_diagnostics,
    source_quantile_threshold,
)
from run_evidential_conflict_source_v1 import METHOD_TO_SCORE, fusion_scores  # noqa: E402


SCORE_METHODS = tuple(METHOD_TO_SCORE)


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


def load_source_oof_evidence(path: Path, manifest: list[dict]) -> np.ndarray:
    rows = mv.read_csv(path)
    by_sha = {mv.norm_sha(row.get("sha256")): row for row in rows}
    if len(by_sha) != len(rows):
        raise RuntimeError("duplicate SHA in source evidential predictions")
    result = np.zeros((len(manifest), len(AUX_VIEW_NAMES), 2), dtype=np.float64)
    for index, item in enumerate(manifest):
        sha = mv.norm_sha(item["sha256"])
        if sha not in by_sha:
            raise RuntimeError("source evidential predictions missing SHA: " + sha)
        row = by_sha[sha]
        for column, view in enumerate(AUX_VIEW_NAMES):
            result[index, column, 0] = float(row[f"{view}__evidence_benign"])
            result[index, column, 1] = float(row[f"{view}__evidence_malware"])
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise RuntimeError("invalid source OOF evidence")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--source-evidential-predictions", type=Path, default=Path(
        "artifacts/care-evidential-conflict-router-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--future-expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-future-diagnostic-v1/"
        "future_predictions_diagnostic.csv"
    ))
    parser.add_argument("--future-expert-column", default=future_base.DEFAULT_EXPERT_COLUMN)
    parser.add_argument("--years", default="2018,2019,2020,2021")
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
        "artifacts/care-evidential-conflict-router-future-v1"
    ))
    args = parser.parse_args()
    years = future_base.parse_years(args.years)
    if not 0.0 < args.budget < 1.0 or args.epochs < 1:
        raise SystemExit("invalid budget or epoch count")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("Frozen 2018-2021 multi-view evidential conflict audit", flush=True)
    print("Evidential device:", device, flush=True)
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
    source_oof_evidence = load_source_oof_evidence(
        args.source_evidential_predictions, manifest
    )
    source_scores, _ = fusion_scores(source_oof_evidence, args.reliability_gamma)
    thresholds = {
        method: source_quantile_threshold(source_scores[score_name], args.budget)
        for method, score_name in METHOD_TO_SCORE.items()
    }

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
            "unexpected future cohort: n=" + str(len(target_rows))
            + " years=" + repr(sorted(set(target_year.tolist())))
        )

    x_source, x_target, names = mv.global_matrices(
        source_rows, target_rows, compact_by_sha, compact_schema, original_names
    )
    groups = mv.group_indices(names)
    _, target_evidence, training_audit = fit_four_evidential_views(
        x_source,
        source_y,
        x_target,
        groups,
        seed=args.seed + 70000,
        device=device,
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kl_weight=args.kl_weight,
        annealing_epochs=args.annealing_epochs,
    )
    target_scores, target_detail = fusion_scores(
        target_evidence, args.reliability_gamma
    )
    frozen, expert_column = future_base.align_frozen_future(
        args.future_expert_predictions,
        target_rows,
        target_year,
        args.future_expert_column,
    )

    # Freeze scores, routes, and probabilities before future labels are accessed.
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
    for method, score_name in METHOD_TO_SCORE.items():
        stem = method.removesuffix("25")
        annual = stem + "_YearTop25"
        source = stem + "_SourceQ75"
        score = target_scores[score_name]
        route_by_method[annual] = groupwise_top_fraction(score, target_year, args.budget)
        route_by_method[source] = score >= thresholds[method]
        score_by_method[annual] = score
        score_by_method[source] = score
    final_by_method = {
        method: np.where(route, frozen["expert"], frozen["base"])
        for method, route in route_by_method.items()
    }

    target_y = np.asarray(
        [mv.v4.audit.future.label_of(row) for row in target_rows], dtype=np.int64
    )
    annual_metrics: list[dict] = []
    annual_routing: list[dict] = []
    pooled_metrics = [future_base.metric_row(
        "Global24", target_y, frozen["base"], frozen["base"],
        np.zeros(len(target_y), dtype=bool), "pooled_2018_2021"
    )]
    pooled_routing: list[dict] = []
    for year in years:
        selected = target_year == year
        annual_metrics.append(future_base.metric_row(
            "Global24", target_y[selected], frozen["base"][selected],
            frozen["base"][selected], np.zeros(int(np.sum(selected)), dtype=bool),
            "annual", year,
        ))
        for method, route in route_by_method.items():
            annual_metrics.append(future_base.metric_row(
                method, target_y[selected], final_by_method[method][selected],
                frozen["base"][selected], route[selected], "annual", year,
            ))
            diagnostic = routing_diagnostics(
                method, target_y[selected], frozen["base"][selected], route[selected],
                score_by_method[method][selected],
            )
            annual_routing.append(
                {"scope": "annual", "year": int(year), **diagnostic, "diagnostic_only": True}
            )
    for method, route in route_by_method.items():
        pooled_metrics.append(future_base.metric_row(
            method, target_y, final_by_method[method], frozen["base"], route,
            "pooled_2018_2021",
        ))
        diagnostic = routing_diagnostics(
            method, target_y, frozen["base"], route, score_by_method[method]
        )
        pooled_routing.append(
            {"scope": "pooled_2018_2021", **diagnostic, "diagnostic_only": True}
        )

    mean_detail = target_detail["mean"]
    prediction_rows: list[dict] = []
    for index, row in enumerate(target_rows):
        item = {
            "sha256": future_base.norm_sha(row["sha256"]),
            "year": int(target_year[index]),
            "label": int(target_y[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenExpert__prob": float(frozen["expert"][index]),
            "evidential_fused_probability": float(mean_detail["fused_probability"][index]),
            "evidential_conflict_mean": float(mean_detail["conflict"][index]),
            "evidential_conflict_max": float(target_detail["max"]["conflict"][index]),
        }
        for column, view in enumerate(AUX_VIEW_NAMES):
            item[f"{view}__evidence_benign"] = float(target_evidence[index, column, 0])
            item[f"{view}__evidence_malware"] = float(target_evidence[index, column, 1])
            item[f"{view}__prob_malware"] = float(mean_detail["view_probability"][index, column])
            item[f"{view}__uncertainty"] = float(mean_detail["view_uncertainty"][index, column])
            item[f"{view}__reliability"] = float(mean_detail["view_reliability"][index, column])
        for method, score_name in METHOD_TO_SCORE.items():
            item[method + "__score"] = float(target_scores[score_name][index])
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
    write_csv(args.output / "training_audit.csv", training_audit)

    audit = {
        "experiment": "care-evidential-conflict-router-future-v1",
        "diagnostic_only": True,
        "source_year": 2017,
        "source_n": int(len(source_y)),
        "source_scores": "strict source outer-fold OOF evidence predictions",
        "future_years": list(years),
        "future_n": int(len(target_y)),
        "device": str(device),
        "budget": float(args.budget),
        "thresholds": {method: float(value) for method, value in thresholds.items()},
        "future_expert_column": expert_column,
        "future_labels_used_for_training_threshold_or_routing": False,
        "future_labels_first_used_after_scores_routes_and_probabilities_frozen": True,
        "primary": "A25_YearTop25 versus EVI_ConflictMean_YearTop25",
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "future_warning": (
            "2018-2021 were already inspected; results are development diagnostics, "
            "not an untouched final test"
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
        "EVI_Ignorance_YearTop25",
        "EVI_ConflictMean_YearTop25",
        "EVI_ConflictMax_YearTop25",
        "EVI_ConflictMean_NoDiscount_YearTop25",
    )
    lines = [
        "CARE-Droid multi-view evidential conflict future diagnostic v1",
        "",
        f"Device: {device}",
        "All evidence heads and thresholds use 2017 only.",
        "Future labels are first accessed after scores, routes, and probabilities freeze.",
        "",
        "Primary equal-budget annual comparison:",
    ]
    for year in years:
        lines.append(str(year))
        for method in primary_methods:
            metric = metric_lookup[(year, method)]
            if method == "Global24":
                lines.append(
                    f"  {method}: BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} F1={metric['F1']:.6f}"
                )
            else:
                route = route_lookup[(year, method)]
                lines.append(
                    f"  {method}: routed={metric['routed_n']}/{metric['n']} "
                    f"error_recall={route['error_recall']:.6f} "
                    f"highconf_error_recall={route['high_confidence_error_recall']:.6f} "
                    f"accepted_risk={route['accepted_risk']:.6f} "
                    f"BA={metric['BA']:.6f} AUC={metric['AUC']:.6f} F1={metric['F1']:.6f} "
                    f"fixed={metric['fixed']} broken={metric['broken']} net={metric['net']}"
                )
    lines.extend(["", "Source-threshold transport route ratios:"])
    for year in years:
        for method in SCORE_METHODS:
            stem = method.removesuffix("25")
            route_method = stem + "_SourceQ75"
            metric = metric_lookup[(year, route_method)]
            lines.append(
                f"  {year} {route_method}: routed_ratio={metric['routed_ratio']:.6f} "
                f"BA={metric['BA']:.6f} F1={metric['F1']:.6f}"
            )
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
