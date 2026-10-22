#!/usr/bin/env python3
"""Strict 2017 audit of Stage-I-error-aware evidential routing."""

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
from routing_aware_evidential_core import fit_joint_evidential_router  # noqa: E402
from run_evidential_conflict_source_v1 import (  # noqa: E402
    overlap_rows,
    per_fold_metric_rows,
    write_csv,
)


FROZEN_EXPERT_NAME = "CARE_RGCondAttn_Dedup_FixKeep"
ROUTE_NAMES = (
    "A25",
    "EVI_ConflictMean25",
    "EA_EVI25",
    "EA_EVI_Ignorance25",
)


def fit_kwargs(args: argparse.Namespace) -> dict:
    return {
        "device": resolve_device(args.device),
        "reliability_gamma": args.reliability_gamma,
        "hidden_dimension": args.hidden_dimension,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "annealing_epochs": args.annealing_epochs,
    }


def fused(evidence: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
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
    parser.add_argument("--expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
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
    parser.add_argument("--route-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-error-aware-evidential-router-source-v1"
    ))
    args = parser.parse_args()
    if args.inner_folds < 2 or args.epochs < 1:
        raise SystemExit("invalid fold or epoch count")
    if not 0.0 < args.budget < 1.0 or args.route_loss_weight <= 0.0:
        raise SystemExit("invalid budget or route-loss weight")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    if {str(row.get("metadata_year", "")).strip() for row in source_rows} != {"2017"}:
        raise RuntimeError("source-only guard failed")
    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)

    routes = {name: np.zeros(len(labels), dtype=bool) for name in ROUTE_NAMES}
    scores = {name: np.full(len(labels), np.nan) for name in ROUTE_NAMES}
    base_evidence_all = np.full((len(labels), len(AUX_VIEW_NAMES), 2), np.nan)
    aware_evidence_all = np.full_like(base_evidence_all, np.nan)
    aware_ignorance_all = np.full(len(labels), np.nan)
    aware_conflict_all = np.full(len(labels), np.nan)
    training_audit: list[dict] = []
    per_fold_routing: list[dict] = []

    print("Strict 2017 error-aware evidential router audit", flush=True)
    print("Evidential device:", device, flush=True)
    print("Nested Stage-I error targets: enabled", flush=True)
    kwargs = fit_kwargs(args)
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in train_index]
        validation_rows = [source_rows[index] for index in validation_index]

        # Critical leakage guard: the error target for an outer-training row is
        # made by an inner model that did not train on that row.  The fixed outer
        # OOF Global24 prediction is intentionally not reused here.
        inner_oof, _ = mv.proper_oof_views(
            train_rows,
            labels[train_index],
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.seed + 100000 * int(fold),
        )
        inner_base = np.asarray(inner_oof["all"], dtype=np.float64)
        error_target = ((inner_base >= 0.5) != labels[train_index]).astype(np.int64)
        if not np.any(error_target) or np.all(error_target):
            raise RuntimeError("degenerate nested Stage-I error target")

        x_train, x_validation, names = mv.global_matrices(
            train_rows, validation_rows, compact_by_sha, compact_schema, original_names
        )
        groups = mv.group_indices(names)
        common_seed = args.seed + 10000 * int(fold)
        baseline = fit_joint_evidential_router(
            x_train,
            labels[train_index],
            error_target,
            x_validation,
            groups,
            seed=common_seed,
            route_loss_weight=0.0,
            **kwargs,
        )
        aware = fit_joint_evidential_router(
            x_train,
            labels[train_index],
            error_target,
            x_validation,
            groups,
            seed=common_seed,
            route_loss_weight=args.route_loss_weight,
            **kwargs,
        )
        for name, result in (("baseline_lambda0", baseline), ("error_aware", aware)):
            training_audit.append({
                "fold": int(fold),
                "model": name,
                "nested_error_target": True,
                **result.diagnostic,
            })

        base_detail = fused(baseline.target_evidence, args.reliability_gamma)
        aware_detail = fused(aware.target_evidence, args.reliability_gamma)
        fold_scores = {
            "A25": frozen["a25_score"][validation_index],
            "EVI_ConflictMean25": base_detail["uncertainty"],
            "EA_EVI25": aware_detail["uncertainty"],
            "EA_EVI_Ignorance25": aware_detail["ignorance"],
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
        base_evidence_all[validation_index] = baseline.target_evidence
        aware_evidence_all[validation_index] = aware.target_evidence
        aware_ignorance_all[validation_index] = aware_detail["ignorance"]
        aware_conflict_all[validation_index] = aware_detail["conflict"]
        print(
            f"  fold {int(fold)} train={len(train_index)} validation={len(validation_index)} "
            f"nested_errors={int(np.sum(error_target))}",
            flush=True,
        )

    arrays = [*scores.values(), base_evidence_all, aware_evidence_all,
              aware_ignorance_all, aware_conflict_all]
    if any(not np.isfinite(value).all() for value in arrays):
        raise RuntimeError("incomplete source output")

    routing_summary = [
        routing_diagnostics(method, labels, frozen["base"], routes[method], scores[method])
        for method in ROUTE_NAMES
    ]
    final_probability = {"Global24": frozen["base"]}
    for method in ROUTE_NAMES:
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
            "EA_EVI__ignorance": float(aware_ignorance_all[index]),
            "EA_EVI__conflict": float(aware_conflict_all[index]),
        }
        for column, view in enumerate(AUX_VIEW_NAMES):
            item[f"baseline__{view}__evidence_benign"] = float(
                base_evidence_all[index, column, 0]
            )
            item[f"baseline__{view}__evidence_malware"] = float(
                base_evidence_all[index, column, 1]
            )
            item[f"aware__{view}__evidence_benign"] = float(
                aware_evidence_all[index, column, 0]
            )
            item[f"aware__{view}__evidence_malware"] = float(
                aware_evidence_all[index, column, 1]
            )
        for method in ROUTE_NAMES:
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
        "experiment": "CARE-Droid error-aware evidential router v1",
        "evaluation": "strict fixed 2017 outer folds; nested inner OOF error targets",
        "future_data_loaded": False,
        "changed_component": "route only",
        "uncertainty": "same conflict-aware evidential U; no separate error predictor",
        "route_target": "inner-OOF Stage-I 0/1 error indicator",
        "loss": "mean four-view EDL loss + lambda_r balanced BCE(U,z)",
        "route_loss_weight": float(args.route_loss_weight),
        "route_budget": float(args.budget),
        "caveat": "Stage-I error is a proxy for needing Stage II, not expert benefit",
        "primary": "EA_EVI25 versus A25 and EVI_ConflictMean25",
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
        "CARE-Droid error-aware evidential router v1",
        "",
        f"Device: {device}",
        "Strict source-only nested evaluation; no future row is loaded.",
        "Stage-I error targets for outer training rows come from inner OOF predictions.",
        f"Route-loss weight: {args.route_loss_weight:.6f}; Stage-II budget: {100*args.budget:.2f}%.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for method in ROUTE_NAMES:
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
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
            f"fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    lines.extend([
        "",
        "Primary: compare EA_EVI25 with A25 and EVI_ConflictMean25 at equal budget.",
        "EA_EVI_Ignorance25 tests whether the learned gain comes from evidence amount",
        "rather than the retained conflict term.",
        "Stage-I error is only a proxy for Stage-II utility; do not claim expert benefit.",
        "Do not tune lambda or variants using future-year labels.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
