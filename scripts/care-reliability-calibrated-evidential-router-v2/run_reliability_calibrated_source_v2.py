#!/usr/bin/env python3
"""Strict 2017 reliability-calibrated EVI and second-layer decision audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT / "care-evidential-conflict-router-v1",
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
    resolve_device,
    routing_diagnostics,
)
from reliability_calibrated_core import (  # noqa: E402
    continuous_stage1_risk,
    fit_reliability_calibrated_router,
)
from run_evidential_conflict_source_v1 import (  # noqa: E402
    overlap_rows,
    per_fold_metric_rows,
    write_csv,
)


ROUTE_NAMES = (
    "A25",
    "EVI_ConflictMean25",
    "RC_EVI_CalOnly25",
    "RC_EVI_CalRank25",
    "RC_EVI_CalRank_Ignorance25",
)
DIRECT_COLUMN = "Global24__A25__CARE_RGCondAttn_DirectL2__expert_prob"


def load_direct_expert(path: Path, manifest: list[dict]) -> np.ndarray:
    rows = mv.read_csv(path)
    if not rows:
        raise RuntimeError(
            "Direct-L2 predictions are missing; run the direct-L2 source ablation first"
        )
    by_sha = {mv.norm_sha(row.get("sha256")): row for row in rows}
    if len(by_sha) != len(rows):
        raise RuntimeError("duplicate SHA in Direct-L2 predictions")
    columns = set(rows[0])
    column = DIRECT_COLUMN
    if column not in columns:
        candidates = sorted(
            name for name in columns
            if "DirectL2" in name and name.endswith("__expert_prob")
        )
        if len(candidates) != 1:
            raise RuntimeError("Direct-L2 expert column candidates=" + repr(candidates))
        column = candidates[0]
    value = np.asarray([
        float(by_sha[mv.norm_sha(row["sha256"])][column]) for row in manifest
    ], dtype=np.float64)
    if value.shape != (len(manifest),) or not np.isfinite(value).all():
        raise RuntimeError("invalid Direct-L2 expert probability")
    return value


def fusion(evidence: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    return conflict_aware_fusion(
        evidence, conflict_aggregator="mean", reliability_gamma=gamma
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--bounded-expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--direct-expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-direct-l2-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--inner-folds", type=int, default=4)
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
    parser.add_argument("--calibration-weight", type=float, default=0.50)
    parser.add_argument("--ranking-weight", type=float, default=0.10)
    parser.add_argument("--huber-beta", type=float, default=0.10)
    parser.add_argument("--ranking-pairs", type=int, default=8192)
    parser.add_argument("--ranking-minimum-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-reliability-calibrated-evidential-router-source-v2"
    ))
    args = parser.parse_args()
    if args.inner_folds < 2 or args.epochs < 1 or args.ranking_pairs < 1:
        raise SystemExit("invalid folds, epochs, or ranking pair count")
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("budget must be in (0,1)")
    if min(args.calibration_weight, args.ranking_weight, args.huber_beta) <= 0.0:
        raise SystemExit("loss weights and Huber beta must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    if {str(row.get("metadata_year", "")).strip() for row in source_rows} != {"2017"}:
        raise RuntimeError("source-only guard failed")
    frozen = mv.load_frozen_predictions(args.bounded_expert_predictions, manifest)
    direct_expert = load_direct_expert(args.direct_expert_predictions, manifest)

    routes = {name: np.zeros(len(labels), dtype=bool) for name in ROUTE_NAMES}
    scores = {name: np.full(len(labels), np.nan) for name in ROUTE_NAMES}
    evidence = {
        name: np.full((len(labels), len(AUX_VIEW_NAMES), 2), np.nan)
        for name in ("baseline", "cal_only", "cal_rank")
    }
    cal_rank_ignorance = np.full(len(labels), np.nan)
    cal_rank_conflict = np.full(len(labels), np.nan)
    training_audit: list[dict] = []
    per_fold_routing: list[dict] = []

    common_hyper = {
        "device": device,
        "reliability_gamma": args.reliability_gamma,
        "huber_beta": args.huber_beta,
        "ranking_pairs": args.ranking_pairs,
        "ranking_minimum_gap": args.ranking_minimum_gap,
        "hidden_dimension": args.hidden_dimension,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "annealing_epochs": args.annealing_epochs,
    }
    print("Strict 2017 reliability-calibrated evidential router v2", flush=True)
    print("Evidential device:", device, flush=True)
    print("Nested continuous Stage-I risk targets: enabled", flush=True)
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in train_index]
        validation_rows = [source_rows[index] for index in validation_index]
        inner_oof, _ = mv.proper_oof_views(
            train_rows,
            labels[train_index],
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.seed + 100000 * int(fold),
        )
        risk_target = continuous_stage1_risk(
            labels[train_index], np.asarray(inner_oof["all"], dtype=np.float64)
        )
        x_train, x_validation, names = mv.global_matrices(
            train_rows, validation_rows, compact_by_sha, compact_schema, original_names
        )
        groups = mv.group_indices(names)
        common = {
            "train_x": x_train,
            "train_y": labels[train_index],
            "train_risk": risk_target,
            "target_x": x_validation,
            "groups": groups,
            "seed": args.seed + 10000 * int(fold),
            **common_hyper,
        }
        fitted = {
            "baseline": fit_reliability_calibrated_router(
                calibration_weight=0.0, ranking_weight=0.0, **common
            ),
            "cal_only": fit_reliability_calibrated_router(
                calibration_weight=args.calibration_weight, ranking_weight=0.0,
                **common,
            ),
            "cal_rank": fit_reliability_calibrated_router(
                calibration_weight=args.calibration_weight,
                ranking_weight=args.ranking_weight,
                **common,
            ),
        }
        for name, result in fitted.items():
            training_audit.append({
                "fold": int(fold),
                "model": name,
                "nested_continuous_risk": True,
                **result.diagnostic,
            })
            evidence[name][validation_index] = result.target_evidence
        detail = {
            name: fusion(result.target_evidence, args.reliability_gamma)
            for name, result in fitted.items()
        }
        fold_scores = {
            "A25": frozen["a25_score"][validation_index],
            "EVI_ConflictMean25": detail["baseline"]["uncertainty"],
            "RC_EVI_CalOnly25": detail["cal_only"]["uncertainty"],
            "RC_EVI_CalRank25": detail["cal_rank"]["uncertainty"],
            "RC_EVI_CalRank_Ignorance25": detail["cal_rank"]["ignorance"],
        }
        fold_routes = {"A25": frozen["a25_route"][validation_index]}
        for method in ROUTE_NAMES[1:]:
            fold_routes[method] = exact_top_fraction(fold_scores[method], args.budget)
        for method in ROUTE_NAMES:
            routes[method][validation_index] = fold_routes[method]
            scores[method][validation_index] = fold_scores[method]
            per_fold_routing.append({
                "fold": int(fold),
                **routing_diagnostics(
                    method,
                    labels[validation_index],
                    frozen["base"][validation_index],
                    fold_routes[method],
                    fold_scores[method],
                ),
            })
        cal_rank_ignorance[validation_index] = detail["cal_rank"]["ignorance"]
        cal_rank_conflict[validation_index] = detail["cal_rank"]["conflict"]
        print(
            f"  fold {int(fold)} train={len(train_index)} validation={len(validation_index)} "
            f"risk_mean={float(np.mean(risk_target)):.6f}",
            flush=True,
        )

    arrays = [*scores.values(), *evidence.values(), cal_rank_ignorance, cal_rank_conflict]
    if any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError("incomplete reliability-calibrated source output")

    routing_summary = [
        routing_diagnostics(method, labels, frozen["base"], routes[method], scores[method])
        for method in ROUTE_NAMES
    ]
    final_probability = {"Global24": frozen["base"]}
    for method in ROUTE_NAMES:
        final_probability[method + "__BoundedCorrection"] = np.where(
            routes[method], frozen["expert"], frozen["base"]
        )
        final_probability[method + "__DirectL2"] = np.where(
            routes[method], direct_expert, frozen["base"]
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
            "BoundedCorrection__expert_prob": float(frozen["expert"][index]),
            "DirectL2__expert_prob": float(direct_expert[index]),
            "RC_EVI_CalRank__ignorance": float(cal_rank_ignorance[index]),
            "RC_EVI_CalRank__conflict": float(cal_rank_conflict[index]),
        }
        for model, matrix in evidence.items():
            for column, view in enumerate(AUX_VIEW_NAMES):
                item[f"{model}__{view}__evidence_benign"] = float(
                    matrix[index, column, 0]
                )
                item[f"{model}__{view}__evidence_malware"] = float(
                    matrix[index, column, 1]
                )
        for method in ROUTE_NAMES:
            item[method + "__score"] = float(scores[method][index])
            item[method + "__routed"] = int(routes[method][index])
            item[method + "__bounded_prob"] = float(
                final_probability[method + "__BoundedCorrection"][index]
            )
            item[method + "__direct_prob"] = float(
                final_probability[method + "__DirectL2"][index]
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
        "experiment": "CARE-Droid reliability-calibrated evidential router v2",
        "evaluation": "strict 2017 nested OOF",
        "continuous_target": "r_i=abs(y_i-p_i^(1)) from inner OOF Global24",
        "loss": "L_EDL + lambda_c Huber(U,r) + lambda_r weighted pair ranking",
        "calibration_weight": float(args.calibration_weight),
        "ranking_weight": float(args.ranking_weight),
        "huber_beta": float(args.huber_beta),
        "ranking_pairs": int(args.ranking_pairs),
        "route_budget": float(args.budget),
        "second_layer_comparison": {
            "bounded": "frozen bounded residual correction",
            "direct": "strict OOF independent Direct-L2 classifier",
        },
        "future_data_loaded": False,
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
        "CARE-Droid reliability-calibrated evidential router v2",
        "",
        f"Device: {device}",
        "Strict 2017 nested OOF: every risk target uses an inner unseen prediction.",
        f"Loss weights: calibration={args.calibration_weight:.3f}, "
        f"ranking={args.ranking_weight:.3f}; budget={100*args.budget:.2f}%.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for method in ROUTE_NAMES:
        row = route_lookup[method]
        lines.append(
            f"{method}: error_recall={row['error_recall']:.6f} "
            f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
            f"precision={row['route_error_precision']:.6f} "
            f"accepted_risk={row['accepted_risk']:.6f}"
        )
    lines.extend(["", "End-to-end source metrics:"])
    for method, row in metric_lookup.items():
        lines.append(
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
            f"fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    lines.extend([
        "",
        "Primary route: RC_EVI_CalRank25 versus A25, lambda=0 EVI, and CalOnly.",
        "For each route, BoundedCorrection and DirectL2 use exactly the same routed APKs.",
        "DirectL2 is a source-only ablation here; future use requires a source-side decision.",
        "Do not tune weights or select variants using future-year labels.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
