#!/usr/bin/env python3
"""Strict 2017 source audit of a class-asymmetric A25 router.

Only the routing allocation changes.  Global24 probabilities and the fixed
deduplicated bounded RGCondAttn expert probabilities are reused.  Within each
fixed outer fold, Global24 is re-fitted in inner folds to obtain leakage-free
OOF probabilities.  Those inner OOF errors select how the fixed 25% routing
budget is divided between APKs predicted benign and APKs predicted malicious.
The held-out outer fold is evaluated once and never participates in quota
selection.  No future-year row is loaded.
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
    SCRIPTS_ROOT,
    SCRIPTS_ROOT / "care-multiview-conformal-router-source-v1",
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from sklearn.model_selection import StratifiedKFold
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: " + str(exc))

try:
    import run_multiview_conformal_router_source_v1 as mv
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing CARE router dependencies in scripts/: " + str(exc))


FROZEN_METHOD = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep"
ASYM_METHOD = "Global24__AsymA25__CARE_RGCondAttn_Dedup_FixKeep"


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


def ambiguity(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return 1.0 - 2.0 * np.abs(probability - 0.5)


def exact_budget(n: int, fraction: float) -> int:
    return max(1, min(int(n), int(round(float(fraction) * int(n)))))


def feasible_k0(predicted: np.ndarray, total: int) -> range:
    predicted = np.asarray(predicted, dtype=np.int64)
    n0 = int(np.sum(predicted == 0))
    n1 = int(np.sum(predicted == 1))
    lower = max(0, int(total) - n1)
    upper = min(int(total), n0)
    return range(lower, upper + 1)


def route_by_class_quota(
    score: np.ndarray,
    predicted: np.ndarray,
    total: int,
    k0: int,
) -> np.ndarray:
    """Route exactly k0 predicted-benign and total-k0 predicted-malicious rows."""
    score = np.asarray(score, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.int64)
    allowed = feasible_k0(predicted, total)
    if int(k0) not in allowed:
        raise ValueError("infeasible predicted-benign quota")
    route = np.zeros(len(score), dtype=bool)
    for label, count in ((0, int(k0)), (1, int(total) - int(k0))):
        members = np.flatnonzero(predicted == label)
        # mergesort gives deterministic tie handling by original sample order.
        order = members[np.argsort(score[members], kind="mergesort")]
        if count:
            route[order[-count:]] = True
    if int(np.sum(route)) != int(total):
        raise RuntimeError("class-quota route did not consume the exact budget")
    return route


def side_statistics(
    labels: np.ndarray,
    predicted: np.ndarray,
    route: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    route = np.asarray(route, dtype=bool)
    error = predicted != labels
    accepted = ~route
    result: dict[str, float | int] = {}
    recalls = []
    risks = []
    for side in (0, 1):
        side_mask = predicted == side
        side_error = side_mask & error
        error_n = int(np.sum(side_error))
        caught_n = int(np.sum(side_error & route))
        accepted_side = side_mask & accepted
        accepted_n = int(np.sum(accepted_side))
        residual_n = int(np.sum(accepted_side & error))
        recall = float(caught_n / error_n) if error_n else 1.0
        risk = float(residual_n / accepted_n) if accepted_n else 0.0
        result[f"pred{side}_n"] = int(np.sum(side_mask))
        result[f"pred{side}_error_n"] = error_n
        result[f"pred{side}_routed_n"] = int(np.sum(side_mask & route))
        result[f"pred{side}_caught_error_n"] = caught_n
        result[f"pred{side}_error_recall"] = recall
        result[f"pred{side}_accepted_n"] = accepted_n
        result[f"pred{side}_residual_error_n"] = residual_n
        result[f"pred{side}_accepted_risk"] = risk
        recalls.append(recall)
        risks.append(risk)
    result["captured_error_n"] = int(np.sum(route & error))
    result["max_side_accepted_risk"] = float(max(risks))
    result["min_side_error_recall"] = float(min(recalls))
    return result


def select_training_quota(
    labels: np.ndarray,
    probability: np.ndarray,
    budget: float,
) -> tuple[int, int, dict[str, float | int]]:
    """Lexicographic, weight-free quota selection on inner OOF predictions.

    1) maximize captured Global24 errors;
    2) minimize the worse accepted risk across the two predicted-label sides;
    3) maximize the worse of the two side-specific error recalls;
    4) prefer the allocation nearest equal routing rates across the sides.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability >= 0.5).astype(np.int64)
    score = ambiguity(probability)
    total = exact_budget(len(labels), budget)
    proportional_k0 = total * float(np.mean(predicted == 0))
    best = None
    for k0 in feasible_k0(predicted, total):
        route = route_by_class_quota(score, predicted, total, k0)
        stats = side_statistics(labels, predicted, route)
        key = (
            -int(stats["captured_error_n"]),
            float(stats["max_side_accepted_risk"]),
            -float(stats["min_side_error_recall"]),
            abs(float(k0) - proportional_k0),
            int(k0),
        )
        if best is None or key < best[0]:
            best = (key, int(k0), stats)
    if best is None:
        raise RuntimeError("no feasible class allocation")
    return best[1], total, best[2]


def apply_quota_fraction(
    probability: np.ndarray,
    budget: float,
    benign_budget_fraction: float,
) -> tuple[np.ndarray, int, int]:
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability >= 0.5).astype(np.int64)
    total = exact_budget(len(probability), budget)
    requested = int(round(float(benign_budget_fraction) * total))
    allowed = feasible_k0(predicted, total)
    k0 = min(max(requested, allowed.start), allowed.stop - 1)
    route = route_by_class_quota(
        ambiguity(probability), predicted, total, k0
    )
    return route, k0, total - k0


def inner_oof_global24(
    rows: list[dict],
    labels: np.ndarray,
    compact_by_sha,
    compact_schema,
    original_names,
    inner_folds: int,
    seed: int,
) -> np.ndarray:
    prediction = np.full(len(rows), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=int(inner_folds), shuffle=True, random_state=int(seed)
    )
    reference_names = None
    for inner_fold, (fit_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        fit_rows = [rows[index] for index in fit_index]
        validation_rows = [rows[index] for index in validation_index]
        x_fit, x_validation, names = mv.global_matrices(
            fit_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        mv.group_indices(names)  # fail closed if Global24 semantics drift.
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise RuntimeError("Global24 names changed across inner folds")
        prediction[validation_index] = mv.v4.g24.fit_view(
            x_fit,
            x_validation,
            labels[fit_index],
            np.arange(x_fit.shape[1], dtype=np.int64),
            int(seed + 10000 * inner_fold),
        )
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError("incomplete inner OOF Global24 predictions")
    return prediction


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
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-asymmetric-a25-router-source-v1"
    ))
    args = parser.parse_args()
    if args.inner_folds < 2 or not 0.0 < args.budget < 1.0:
        raise SystemExit("invalid --inner-folds or --budget")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        mv.v4.audit.load_inputs(args)
    )
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    years = {str(row.get("metadata_year", "")).strip() for row in source_rows}
    if years != {"2017"}:
        raise RuntimeError("source-only guard failed: " + repr(years))
    frozen = mv.load_frozen_predictions(args.expert_predictions, manifest)

    asym_route = np.zeros(len(labels), dtype=bool)
    fold_audit = []
    per_fold_routing = []
    print("Strict 2017 class-asymmetric A25 routing audit", flush=True)
    for fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != fold)
        outer_validation = np.flatnonzero(folds == fold)
        train_rows = [source_rows[index] for index in outer_train]
        train_y = labels[outer_train]
        inner_probability = inner_oof_global24(
            train_rows,
            train_y,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            int(args.seed + 100000 * int(fold)),
        )
        train_k0, train_total, train_stats = select_training_quota(
            train_y, inner_probability, args.budget
        )
        benign_budget_fraction = float(train_k0 / train_total)
        validation_probability = frozen["base"][outer_validation]
        fold_route, validation_k0, validation_k1 = apply_quota_fraction(
            validation_probability, args.budget, benign_budget_fraction
        )
        asym_route[outer_validation] = fold_route
        validation_predicted = (validation_probability >= 0.5).astype(np.int64)
        validation_stats = side_statistics(
            labels[outer_validation], validation_predicted, fold_route
        )
        fold_audit.append({
            "fold": int(fold),
            "outer_train_n": int(len(outer_train)),
            "outer_validation_n": int(len(outer_validation)),
            "inner_folds": int(args.inner_folds),
            "train_budget_n": int(train_total),
            "selected_train_pred0_budget_n": int(train_k0),
            "selected_train_pred1_budget_n": int(train_total - train_k0),
            "selected_pred0_budget_fraction": benign_budget_fraction,
            "validation_budget_n": int(validation_k0 + validation_k1),
            "validation_pred0_budget_n": int(validation_k0),
            "validation_pred1_budget_n": int(validation_k1),
            **{"train_" + key: value for key, value in train_stats.items()},
            **{"validation_" + key: value for key, value in validation_stats.items()},
        })
        for route_name, route_value, route_score in (
            (
                "A25",
                frozen["a25_route"][outer_validation],
                frozen["a25_score"][outer_validation],
            ),
            ("AsymA25", fold_route, ambiguity(validation_probability)),
        ):
            per_fold_routing.append({
                "fold": int(fold),
                **mv.routing_row(
                    route_name,
                    labels[outer_validation],
                    validation_probability,
                    route_value,
                    route_score,
                ),
            })
        print(
            f"  fold {int(fold)}: train allocation pred0/pred1="
            f"{train_k0}/{train_total-train_k0}; validation="
            f"{validation_k0}/{validation_k1}",
            flush=True,
        )

    routes = {"A25": frozen["a25_route"], "AsymA25": asym_route}
    routing_summary = [
        mv.routing_row(
            name,
            labels,
            frozen["base"],
            route,
            frozen["a25_score"] if name == "A25" else ambiguity(frozen["base"]),
        )
        for name, route in routes.items()
    ]
    final_probability = {
        "Global24": frozen["base"],
        FROZEN_METHOD: mv.routed_probability(
            frozen["base"], frozen["expert"], frozen["a25_route"]
        ),
        ASYM_METHOD: mv.routed_probability(
            frozen["base"], frozen["expert"], asym_route
        ),
    }
    route_for_method = {
        "Global24": np.zeros(len(labels), dtype=bool),
        FROZEN_METHOD: frozen["a25_route"],
        ASYM_METHOD: asym_route,
    }
    metric_summary = []
    per_fold_metric = []
    for method, probability in final_probability.items():
        metric_summary.extend(mv.v4.v2.metric_rows(
            labels,
            {method: probability},
            frozen["base"],
            route_for_method[method],
        ))
        for fold in sorted(np.unique(folds)):
            selected = folds == fold
            row = mv.v4.v2.metric_rows(
                labels[selected],
                {method: probability[selected]},
                frozen["base"][selected],
                route_for_method[method][selected],
            )[0]
            per_fold_metric.append({"fold": int(fold), **row})

    prediction_rows = []
    for index, row in enumerate(manifest):
        prediction_rows.append({
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "Global24__prob": float(frozen["base"][index]),
            "A25__score": float(frozen["a25_score"][index]),
            "A25__routed": int(frozen["a25_route"][index]),
            "AsymA25__score": float(ambiguity(frozen["base"])[index]),
            "AsymA25__routed": int(asym_route[index]),
            "FrozenDedupExpert__prob": float(frozen["expert"][index]),
            "AsymA25__final_prob": float(final_probability[ASYM_METHOD][index]),
        })

    write_csv(args.output / "routing_summary.csv", routing_summary)
    write_csv(args.output / "source_metrics.csv", metric_summary)
    write_csv(args.output / "source_per_fold_metrics.csv", per_fold_metric)
    write_csv(args.output / "source_per_fold_routing.csv", per_fold_routing)
    write_csv(args.output / "fold_quota_audit.csv", fold_audit)
    write_csv(args.output / "source_predictions.csv", prediction_rows)

    architecture = {
        "experiment": "CARE-Droid class-asymmetric A25 router source v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "future_data_loaded": False,
        "changed_component": "routing allocation only",
        "base": "frozen Global24 outer-fold OOF probabilities",
        "expert": "frozen deduplicated bounded RGCondAttn probabilities",
        "ambiguity_score": "1-2*abs(p-0.5)",
        "budget": float(args.budget),
        "quota_training": "nested Global24 inner OOF predictions in each outer training fold",
        "quota_parameter": "fraction of the 25% budget allocated to predicted-benign APKs",
        "selection": [
            "maximize captured inner-OOF Global24 errors",
            "minimize maximum accepted risk across predicted-label sides",
            "maximize minimum side-specific error recall",
            "prefer allocation nearest equal routing rates across sides",
        ],
        "outer_validation_labels_used_for_quota": False,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = {row["method"]: row for row in metric_summary}
    routing = {row["method"]: row for row in routing_summary}
    lines = [
        "CARE-Droid class-asymmetric A25 source audit v1",
        "",
        "Strict source-only evaluation: fixed 2017 outer folds.",
        "Future years loaded: no.",
        "The 25% budget split is selected only from nested inner OOF source errors.",
        "The frozen Global24 base and deduplicated bounded RGCondAttn expert are unchanged.",
        "",
        "Routing quality:",
    ]
    for name in ("A25", "AsymA25"):
        row = routing[name]
        lines.append(
            f"{name}: routed={int(row['routed_n'])}/{int(row['n'])} "
            f"({100*float(row['routed_ratio']):.2f}%) "
            f"error_recall={float(row['error_recall']):.6f} "
            f"route_precision={float(row['route_error_precision']):.6f} "
            f"accepted_risk={float(row['accepted_risk']):.6f}"
        )
    lines.extend(["", "End-to-end source metrics:"])
    for method in ("Global24", FROZEN_METHOD, ASYM_METHOD):
        row = metrics[method]
        lines.append(
            f"{method}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} "
            f"F1={float(row['F1']):.6f} fixed={int(float(row['fixed']))} "
            f"broken={int(float(row['broken']))} net={int(float(row['net']))}"
        )
    lines.extend([
        "",
        "Decision rule:",
        "Replace A25 only if AsymA25 improves aggregate and per-fold routing error",
        "recall/accepted risk and preserves or improves BA and fixed/broken/net.",
        "Do not inspect future years before making this source-side decision.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
