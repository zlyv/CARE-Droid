#!/usr/bin/env python3
"""Strict 2017 OOF audit of diverse-classifier committee uncertainty routing."""

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
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
    # Fallbacks make the bundle auditable in the local reconstructed tree.
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE dependencies: " + str(exc))

from committee_uncertainty_core import (  # noqa: E402
    MEMBER_NAMES,
    assert_disjoint,
    committee_scores,
    diversity_rows,
    exact_top_fraction,
    fit_calibrated_committee,
    member_metric_rows,
    routing_metrics,
)


ROUTE_NAMES = ("A25", "Committee_PE25", "Committee_MI25")
FROZEN_EXPERT = "CARE_RGCondAttn_Dedup_FixKeep_BoundedCorrection"


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
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    base_prediction = np.asarray(base, dtype=np.float64) >= 0.5
    prediction = p >= 0.5
    base_correct = base_prediction == y
    final_correct = prediction == y
    return {
        "method": method,
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "fixed": int(np.sum(~base_correct & final_correct)),
        "broken": int(np.sum(base_correct & ~final_correct)),
        "net": int(np.sum(final_correct) - np.sum(base_correct)),
        "routed_n": int(np.sum(route)),
        "routed_ratio": float(np.mean(route)),
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
    parser.add_argument("--expert-predictions", type=Path, default=Path(
        "artifacts/care-rgcondattn-dedup-bounded-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-committee-uncertainty-router-source-v1"
    ))
    args = parser.parse_args()
    if args.inner_folds < 2 or args.jobs == 0 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid inner folds, jobs, or budget")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    if len(labels) != 799 or tuple(np.bincount(labels, minlength=2)) != (400, 399):
        raise RuntimeError("unexpected fixed 2017 source cohort")
    if {str(row.get("metadata_year", "")).strip() for row in source_rows} != {"2017"}:
        raise RuntimeError("source-only guard failed")
    fold_sizes = sorted(int(np.sum(folds == fold)) for fold in np.unique(folds))
    if fold_sizes != [159, 160, 160, 160, 160]:
        raise RuntimeError("fixed source folds changed: " + repr(fold_sizes))

    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)
    for name in ("base", "a25_score", "expert"):
        if frozen[name].shape != labels.shape or not np.isfinite(frozen[name]).all():
            raise RuntimeError("invalid frozen array: " + name)

    member_probability = np.full((len(labels), len(MEMBER_NAMES)), np.nan)
    scores = {name: np.full(len(labels), np.nan) for name in ROUTE_NAMES}
    routes = {name: np.zeros(len(labels), dtype=bool) for name in ROUTE_NAMES}
    routes["A25"] = frozen["a25_route"].copy()
    scores["A25"] = frozen["a25_score"].copy()
    calibration_audit: list[dict] = []
    per_fold_router: list[dict] = []
    per_fold_member: list[dict] = []
    per_fold_diversity: list[dict] = []
    global_names: list[str] | None = None

    print("Strict 2017 classifier-committee OOF audit", flush=True)
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        validation_index = np.flatnonzero(folds == fold)
        assert_disjoint(train_index, validation_index)
        train_rows = [source_rows[index] for index in train_index]
        validation_rows = [source_rows[index] for index in validation_index]
        x_train, x_validation, names = mv.global_matrices(
            train_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        if len(names) != 24:
            raise RuntimeError("committee did not receive exact Global24")
        if global_names is None:
            global_names = list(names)
        elif list(names) != global_names:
            raise RuntimeError("Global24 names/order changed across folds")

        fitted = fit_calibrated_committee(
            x_train,
            labels[train_index],
            x_validation,
            inner_folds=args.inner_folds,
            split_seed=args.seed + 100000 * int(fold),
            model_seed=args.seed + 10000 * int(fold),
            jobs=args.jobs,
        )
        member_probability[validation_index] = fitted.target_probability
        detail = committee_scores(fitted.target_probability)
        fold_scores = {
            "A25": frozen["a25_score"][validation_index],
            "Committee_PE25": detail["predictive_entropy"],
            "Committee_MI25": detail["mutual_information"],
        }
        fold_routes = {
            "A25": frozen["a25_route"][validation_index],
            "Committee_PE25": exact_top_fraction(
                fold_scores["Committee_PE25"], args.budget
            ),
            "Committee_MI25": exact_top_fraction(
                fold_scores["Committee_MI25"], args.budget
            ),
        }
        for method in ROUTE_NAMES:
            scores[method][validation_index] = fold_scores[method]
            routes[method][validation_index] = fold_routes[method]
            per_fold_router.append({
                "fold": int(fold),
                **routing_metrics(
                    method,
                    labels[validation_index],
                    frozen["base"][validation_index],
                    frozen["expert"][validation_index],
                    fold_routes[method],
                    fold_scores[method],
                ),
            })
        per_fold_member.extend(member_metric_rows(
            labels[validation_index],
            fitted.target_probability,
            scope="outer_validation",
            fold=int(fold),
        ))
        per_fold_diversity.extend(diversity_rows(
            fitted.target_probability,
            scope="outer_validation",
            fold=int(fold),
        ))
        for row in fitted.audit:
            calibration_audit.append({
                "outer_fold": int(fold),
                "outer_train_n": int(len(train_index)),
                "outer_validation_n": int(len(validation_index)),
                "outer_validation_labels_used": False,
                **row,
            })
        print(
            f"  fold {int(fold)} train={len(train_index)} "
            f"validation={len(validation_index)}",
            flush=True,
        )

    if global_names is None or any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("incomplete source score output")
    if not np.isfinite(member_probability).all():
        raise RuntimeError("incomplete source member probabilities")
    detail = committee_scores(member_probability)
    if not np.allclose(scores["Committee_PE25"], detail["predictive_entropy"]):
        raise RuntimeError("PE assembly mismatch")
    if not np.allclose(scores["Committee_MI25"], detail["mutual_information"]):
        raise RuntimeError("MI assembly mismatch")

    router_rows = [
        routing_metrics(
            method,
            labels,
            frozen["base"],
            frozen["expert"],
            routes[method],
            scores[method],
        )
        for method in ROUTE_NAMES
    ]
    final_probability = {
        method: np.where(routes[method], frozen["expert"], frozen["base"])
        for method in ROUTE_NAMES
    }
    final_rows = [final_metric_row(
        "Global24",
        labels,
        frozen["base"],
        frozen["base"],
        np.zeros(len(labels), dtype=bool),
    )]
    final_rows.extend(
        final_metric_row(
            method,
            labels,
            final_probability[method],
            frozen["base"],
            routes[method],
        )
        for method in ROUTE_NAMES
    )
    final_by_name = {str(row["method"]): row for row in final_rows}
    for row in router_rows:
        metric = final_by_name[str(row["method"])]
        row.update({key: metric[key] for key in ("BA", "AUC", "F1")})

    member_rows = member_metric_rows(
        labels, member_probability, scope="all_source_OOF"
    ) + per_fold_member
    diversity = diversity_rows(
        member_probability, scope="all_source_OOF"
    ) + per_fold_diversity

    base_prediction = frozen["base"] >= 0.5
    expert_prediction = frozen["expert"] >= 0.5
    base_correct = base_prediction == labels
    expert_correct = expert_prediction == labels
    repairable = ~base_correct & expert_correct
    harmful = base_correct & ~expert_correct
    unrepairable = ~base_correct & ~expert_correct
    raw_a25 = 1.0 - 2.0 * np.abs(frozen["base"] - 0.5)
    sample_rows: list[dict] = []
    for index, row in enumerate(manifest):
        item = {
            "sha256": row["sha256"],
            "year": 2017,
            "label": int(labels[index]),
            "fold": int(folds[index]),
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
            "route_A25": int(routes["A25"][index]),
            "route_PE": int(routes["Committee_PE25"][index]),
            "route_MI": int(routes["Committee_MI25"][index]),
            "repairable": int(repairable[index]),
            "harmful": int(harmful[index]),
            "unrepairable": int(unrepairable[index]),
            "final_probability_A25": float(final_probability["A25"][index]),
            "final_probability_PE": float(
                final_probability["Committee_PE25"][index]
            ),
            "final_probability_MI": float(
                final_probability["Committee_MI25"][index]
            ),
        }
        sample_rows.append(item)

    write_csv(args.output / "source_router_metrics.csv", router_rows)
    write_csv(args.output / "source_end_to_end_metrics.csv", final_rows)
    write_csv(args.output / "source_member_metrics.csv", member_rows)
    write_csv(args.output / "source_member_diversity.csv", diversity)
    write_csv(args.output / "source_sample_scores.csv", sample_rows)
    write_csv(args.output / "source_per_fold_router_metrics.csv", per_fold_router)
    write_csv(args.output / "calibration_audit.csv", calibration_audit)

    protocol = {
        "experiment": "CARE-Droid diverse classifier committee uncertainty router v1",
        "changed_component": "routing only",
        "official_stage1_probability": "frozen Global24 L2 probability",
        "committee_mean_used_as_stage1_probability": False,
        "committee_members": list(MEMBER_NAMES),
        "committee_input": "exact same 24 Global24 columns",
        "global24_columns": global_names,
        "calibration": "per-member Platt sigmoid on outer-train inner OOF raw scores",
        "outer_validation_labels_used_for_training_or_calibration": False,
        "source_evaluation": "fixed five-fold 2017 OOF",
        "future_rows_loaded": False,
        "budget": float(args.budget),
        "stage2": FROZEN_EXPERT,
        "direct_stage2_classification_included": False,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    router_by_name = {str(row["method"]): row for row in router_rows}
    member_aggregate = [row for row in member_rows if row["scope"] == "all_source_OOF"]
    diversity_aggregate = [row for row in diversity if row["scope"] == "all_source_OOF"]
    lines = [
        "CARE-Droid diverse classifier committee uncertainty router v1",
        "",
        "Strict 2017 OOF",
        "Committee: LR, RF, XGBoost, SVM",
        "Calibration: source-only sigmoid (outer-train inner OOF)",
        f"Stage-II budget: {100.0 * args.budget:.2f}%",
        "Official Stage I: frozen Global24 L2; p_bar is routing diagnostic only.",
        "",
        "Source routing quality and repairability:",
    ]
    for method in ROUTE_NAMES:
        row = router_by_name[method]
        lines.append(
            f"{method}: routed={row['routed_n']}/{row['n']} "
            f"error_recall={row['error_recall']:.6f} "
            f"highconf_error_recall={row['high_confidence_error_recall']:.6f} "
            f"precision={row['routing_precision']:.6f} "
            f"accepted_risk={row['accepted_risk']:.6f} "
            f"RER@25={row['repairable_error_recall_at_budget']:.6f} "
            f"HarmfulCapture={row['harmful_capture']:.6f} "
            f"UnrepairableCapture={row['unrepairable_capture']:.6f}"
        )
    lines.extend(["", "End-to-end with frozen BoundedCorrection:"])
    for row in final_rows:
        lines.append(
            f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} "
            f"broken={row['broken']} net={row['net']}"
        )
    lines.extend(["", "Committee member diagnostics:"])
    for row in member_aggregate:
        lines.append(
            f"{row['member']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} Brier={row['Brier']:.6f} "
            f"ECE10={row['ECE10']:.6f}"
        )
    lines.extend(["", "Pairwise diversity:"])
    for row in diversity_aggregate:
        lines.append(
            f"{row['member_a']} vs {row['member_b']}: "
            f"Pearson={row['probability_pearson']:.6f} "
            f"Spearman={row['probability_spearman']:.6f} "
            f"disagreement={row['prediction_disagreement_rate']:.6f}"
        )
    lines.extend([
        "",
        "Primary comparison: A25 versus Committee_MI25 at equal budget.",
        "Secondary comparison: Committee_PE25.",
        "No committee mean replaces Stage I; no direct Stage-II classifier is tested.",
    ])
    text = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(text, encoding="utf-8")
    print("Done:", args.output, flush=True)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
