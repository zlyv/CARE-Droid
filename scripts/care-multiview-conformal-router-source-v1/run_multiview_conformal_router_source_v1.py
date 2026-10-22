#!/usr/bin/env python3
"""Strict 2017 source-only audit of multi-view class-conditional routing.

The frozen CARE-Droid components are not retrained here.  The script reuses:

* the established Global24 first-layer OOF probability;
* the fixed deduplicated bounded RGCondAttn expert probability for every APK.

Only the route is changed.  Five cheap auxiliary Logistic heads are learned
inside every outer training fold: all Global24 features plus four disjoint
semantic feature groups.  Their probabilities are Platt calibrated from OOF
predictions produced only in a proper-training subset.  A disjoint calibration
subset supplies class-conditional conformal reference scores and the frozen
source threshold.  No 2018--2021 row is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (
    SCRIPT_DIR,
    SCRIPTS_ROOT,
    SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script",
    SCRIPTS_ROOT / "care-rgcondattn-dedup-bounded-source-v1",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: " + str(exc))

try:
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing established CARE v4 dependencies: " + str(exc))


FROZEN_METHOD = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep"
EXPERT_COLUMN = FROZEN_METHOD + "__expert_prob"
VIEW_NAMES = ("all", "manifest", "code_call", "sensitive", "graph")

MANIFEST_FEATURES = {
    "manifest_permission_count",
    "manifest_permission_count_log",
    "manifest_activity_count",
    "manifest_service_count",
    "manifest_receiver_count",
    "manifest_provider_count",
    "manifest_component_count",
    "manifest_component_count_log",
    "min_sdk",
    "target_sdk",
}
CODE_CALL_FEATURES = {
    "method_count",
    "graph_node_count",
    "graph_edge_count",
}
SENSITIVE_FEATURES = {
    # Compact76 removes the constant manifest_parse_ok feature and retains this
    # semantic persistence-context signal in Global24.  Treat it explicitly as
    # sensitive-behaviour evidence so the four views partition Global24.
    "context_role__boot_persistence",
    "mapped_api_node_count",
    "sensitive_node_count",
    "sensitive_edge_count",
    "manifest_context_match_count",
}
GRAPH_FEATURES = {
    "internal_node_count",
    "internal_edge_count",
    "external_node_count",
    "external_edge_count",
    "unmapped_external_node_count",
    "mapped_external_ratio",
}
EXPECTED_GROUPS = {
    "manifest": MANIFEST_FEATURES,
    "code_call": CODE_CALL_FEATURES,
    "sensitive": SENSITIVE_FEATURES,
    "graph": GRAPH_FEATURES,
}


def norm_sha(value: object) -> str:
    return str(value or "").strip().casefold()


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
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


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def platt_fit(raw_probability: np.ndarray, labels: np.ndarray, seed: int):
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(clipped_logit(raw_probability).reshape(-1, 1), labels)
    return model


def platt_predict(model, raw_probability: np.ndarray) -> np.ndarray:
    result = model.predict_proba(
        clipped_logit(raw_probability).reshape(-1, 1)
    )[:, 1]
    return np.clip(np.asarray(result, dtype=np.float64), 1e-8, 1.0 - 1e-8)


def nll(labels: np.ndarray, positive_probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(positive_probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def group_indices(global_names: list[str]) -> dict[str, np.ndarray]:
    actual = set(global_names)
    expected = set().union(*EXPECTED_GROUPS.values())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if len(global_names) != 24 or missing or unexpected:
        raise RuntimeError(
            "Global24 semantic grouping mismatch: "
            + repr({
                "dimension": len(global_names),
                "missing": missing,
                "unexpected": unexpected,
                "actual": global_names,
            })
        )
    result = {"all": np.arange(len(global_names), dtype=np.int64)}
    for view, fields in EXPECTED_GROUPS.items():
        result[view] = np.asarray(
            [index for index, name in enumerate(global_names) if name in fields],
            dtype=np.int64,
        )
        if len(result[view]) != len(fields):
            raise RuntimeError("incomplete view " + view)
    return result


def global_matrices(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    original_names,
):
    x_train, x_test, names = v4.audit.compact_matrix_variant(
        v4.VARIANT,
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    )
    global_index, _ = v4.evidence.split_view_indices(names)
    global_names = [str(names[index]) for index in global_index]
    return (
        np.asarray(x_train[:, global_index], dtype=np.float64),
        np.asarray(x_test[:, global_index], dtype=np.float64),
        global_names,
    )


def fit_raw_views(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    groups: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    result = {}
    for offset, view in enumerate(VIEW_NAMES):
        result[view] = v4.g24.fit_view(
            x_train,
            x_test,
            y_train,
            groups[view],
            int(seed + 100 * offset),
        )
    return result


def proper_oof_views(
    proper_rows,
    proper_y,
    compact_by_sha,
    compact_schema,
    original_names,
    inner_folds: int,
    seed: int,
):
    result = {
        view: np.full(len(proper_rows), np.nan, dtype=np.float64)
        for view in VIEW_NAMES
    }
    reference_names = None
    splitter = StratifiedKFold(
        n_splits=int(inner_folds), shuffle=True, random_state=int(seed)
    )
    for inner_fold, (fit_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(proper_y)), proper_y), start=1
    ):
        fit_rows = [proper_rows[index] for index in fit_index]
        validation_rows = [proper_rows[index] for index in validation_index]
        x_fit, x_validation, names = global_matrices(
            fit_rows,
            validation_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise RuntimeError("Global24 names changed across inner folds")
        groups = group_indices(names)
        prediction = fit_raw_views(
            x_fit,
            proper_y[fit_index],
            x_validation,
            groups,
            seed + 10000 * inner_fold,
        )
        for view in VIEW_NAMES:
            result[view][validation_index] = prediction[view]
    if any(np.any(~np.isfinite(value)) for value in result.values()):
        raise RuntimeError("incomplete proper-training OOF view probabilities")
    return result, reference_names


def probability_matrix(probability: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([probability[view] for view in VIEW_NAMES])


def nonconformity(
    calibrated_positive: np.ndarray,
    reliability_weight: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    positive = np.clip(calibrated_positive, epsilon, 1.0 - epsilon)
    negative = 1.0 - positive
    return np.column_stack([
        -np.sum(reliability_weight[None, :] * np.log(negative), axis=1),
        -np.sum(reliability_weight[None, :] * np.log(positive), axis=1),
    ])


def conformal_pvalues(
    calibration_nonconformity: np.ndarray,
    calibration_labels: np.ndarray,
    target_nonconformity: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(target_nonconformity), 2), dtype=np.float64)
    for label in (0, 1):
        reference = calibration_nonconformity[
            np.asarray(calibration_labels, dtype=np.int64) == label,
            label,
        ]
        if len(reference) < 10:
            raise RuntimeError("too few class-conditional calibration rows")
        result[:, label] = (
            1.0
            + np.sum(
                reference[None, :] >= target_nonconformity[:, label, None],
                axis=1,
            )
        ) / (len(reference) + 1.0)
    return result


def calibration_self_pvalues(
    calibration_nonconformity: np.ndarray,
    calibration_labels: np.ndarray,
) -> np.ndarray:
    """Leave-self-out p-values used only to freeze the routing threshold."""
    labels = np.asarray(calibration_labels, dtype=np.int64)
    result = np.zeros((len(labels), 2), dtype=np.float64)
    for index in range(len(labels)):
        for candidate in (0, 1):
            members = np.flatnonzero(labels == candidate)
            if labels[index] == candidate:
                members = members[members != index]
            reference = calibration_nonconformity[members, candidate]
            result[index, candidate] = (
                1.0
                + np.sum(reference >= calibration_nonconformity[index, candidate])
            ) / (len(reference) + 1.0)
    return result


def route_score(pvalues: np.ndarray, predicted_label: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_label, dtype=np.int64)
    row = np.arange(len(predicted))
    return 1.0 - pvalues[row, predicted] + pvalues[row, 1 - predicted]


def top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    count = max(1, min(len(score), int(round(float(budget) * len(score)))))
    order = np.argsort(score, kind="mergesort")
    result = np.zeros(len(score), dtype=bool)
    result[order[-count:]] = True
    return result


def routed_probability(base, expert, route):
    return np.where(np.asarray(route, dtype=bool), expert, base)


def routing_row(method, labels, base, route, score, fold=None):
    labels = np.asarray(labels, dtype=np.int64)
    route = np.asarray(route, dtype=bool)
    error = (np.asarray(base) >= 0.5) != labels
    accepted = ~route
    captured = int(np.sum(route & error))
    row = {
        "method": method,
        "n": int(len(labels)),
        "routed_n": int(np.sum(route)),
        "routed_ratio": float(np.mean(route)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured,
        "error_recall": float(captured / np.sum(error)) if np.any(error) else 0.0,
        "route_error_precision": float(captured / np.sum(route)) if np.any(route) else 0.0,
        "accepted_n": int(np.sum(accepted)),
        "accepted_error_n": int(np.sum(accepted & error)),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "score_mean": float(np.mean(score)),
        "score_max": float(np.max(score)),
    }
    if fold is not None:
        row = {"fold": int(fold), **row}
    return row


def load_frozen_predictions(path: Path, manifest: list[dict]):
    rows = read_csv(path)
    by_sha = {norm_sha(row.get("sha256")): row for row in rows}
    if len(by_sha) != len(rows):
        raise RuntimeError("duplicate SHA in frozen expert predictions")
    missing = [row["sha256"] for row in manifest if row["sha256"] not in by_sha]
    if missing:
        raise RuntimeError("frozen predictions missing source SHA: " + repr(missing[:5]))
    aligned = [by_sha[row["sha256"]] for row in manifest]
    required = {"Global24__prob", "A25__score", "A25__routed", EXPERT_COLUMN}
    absent = sorted(required - set(aligned[0]))
    if absent:
        raise RuntimeError("frozen predictions missing columns: " + repr(absent))
    return {
        "base": np.asarray([float(row["Global24__prob"]) for row in aligned]),
        "a25_score": np.asarray([float(row["A25__score"]) for row in aligned]),
        "a25_route": np.asarray([int(float(row["A25__routed"])) for row in aligned], dtype=bool),
        "expert": np.asarray([float(row[EXPERT_COLUMN]) for row in aligned]),
    }


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
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--platt-fraction", type=float, default=0.70)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-multiview-conformal-router-source-v1"
    ))
    args = parser.parse_args()
    if args.inner_folds < 2:
        raise SystemExit("--inner-folds must be >=2")
    if not 0.1 <= args.calibration_fraction <= 0.4:
        raise SystemExit("--calibration-fraction must be in [0.1,0.4]")
    if not 0.5 <= args.platt_fraction <= 0.9:
        raise SystemExit("--platt-fraction must be in [0.5,0.9]")
    if not 0.0 < args.budget < 1.0 or args.eta < 0.0:
        raise SystemExit("invalid budget or eta")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = (
        v4.audit.load_inputs(args)
    )
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    years = {str(row.get("metadata_year", "")).strip() for row in source_rows}
    if years != {"2017"}:
        raise RuntimeError("source-only guard failed: " + repr(years))
    frozen = load_frozen_predictions(args.expert_predictions, manifest)
    if np.any(~np.isfinite(frozen["base"])) or np.any(~np.isfinite(frozen["expert"])):
        raise RuntimeError("non-finite frozen probabilities")

    route_names = (
        "A25",
        "ViewRange25",
        "MVCC_Top25",
        "MVCC_FrozenTau",
    )
    route_all = {name: np.zeros(len(labels), dtype=bool) for name in route_names}
    score_all = {name: np.full(len(labels), np.nan) for name in route_names}
    view_probability_all = {
        view: np.full(len(labels), np.nan) for view in VIEW_NAMES
    }
    pvalue_all = np.full((len(labels), 2), np.nan)
    fold_audit = []
    weight_rows = []
    per_fold_routing = []

    print("Strict 2017 multi-view conformal routing audit", flush=True)
    for fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != fold)
        outer_validation = np.flatnonzero(folds == fold)
        train_y = labels[outer_train]
        split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(args.calibration_fraction),
            random_state=int(args.seed + 1000 * int(fold)),
        )
        proper_local, calibration_local = next(
            split.split(np.zeros(len(train_y)), train_y)
        )
        proper_index = outer_train[proper_local]
        calibration_index = outer_train[calibration_local]
        proper_rows = [source_rows[index] for index in proper_index]
        calibration_rows = [source_rows[index] for index in calibration_index]
        validation_rows = [source_rows[index] for index in outer_validation]
        proper_y = labels[proper_index]
        calibration_y = labels[calibration_index]

        print(
            f"  fold {int(fold)} proper={len(proper_index)} "
            f"calibration={len(calibration_index)} validation={len(outer_validation)}",
            flush=True,
        )
        raw_oof, oof_names = proper_oof_views(
            proper_rows,
            proper_y,
            compact_by_sha,
            compact_schema,
            original_names,
            args.inner_folds,
            args.seed + 10000 * int(fold),
        )
        split_platt = StratifiedShuffleSplit(
            n_splits=1,
            train_size=float(args.platt_fraction),
            random_state=int(args.seed + 2000 * int(fold)),
        )
        platt_index, weight_index = next(
            split_platt.split(np.zeros(len(proper_y)), proper_y)
        )
        platt_models = {}
        heldout_nll = []
        for offset, view in enumerate(VIEW_NAMES):
            model = platt_fit(
                raw_oof[view][platt_index],
                proper_y[platt_index],
                args.seed + 3000 * int(fold) + offset,
            )
            platt_models[view] = model
            probability = platt_predict(model, raw_oof[view][weight_index])
            heldout_nll.append(nll(proper_y[weight_index], probability))
        heldout_nll = np.asarray(heldout_nll, dtype=np.float64)
        reliability_weight = np.exp(-float(args.eta) * heldout_nll)
        reliability_weight /= np.sum(reliability_weight)

        target_rows = calibration_rows + validation_rows
        x_proper, x_target, names = global_matrices(
            proper_rows,
            target_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        if names != oof_names:
            raise RuntimeError("Global24 names changed between OOF and final fit")
        groups = group_indices(names)
        raw_target = fit_raw_views(
            x_proper,
            proper_y,
            x_target,
            groups,
            args.seed + 40000 * int(fold),
        )
        calibrated_target = {
            view: platt_predict(platt_models[view], raw_target[view])
            for view in VIEW_NAMES
        }
        calibration_n = len(calibration_rows)
        calibration_probability = {
            view: calibrated_target[view][:calibration_n] for view in VIEW_NAMES
        }
        validation_probability = {
            view: calibrated_target[view][calibration_n:] for view in VIEW_NAMES
        }
        calibration_matrix = probability_matrix(calibration_probability)
        validation_matrix = probability_matrix(validation_probability)
        calibration_a = nonconformity(
            calibration_matrix, reliability_weight, args.epsilon
        )
        validation_a = nonconformity(
            validation_matrix, reliability_weight, args.epsilon
        )
        validation_pi = conformal_pvalues(
            calibration_a, calibration_y, validation_a
        )
        calibration_pi = calibration_self_pvalues(calibration_a, calibration_y)

        # The operational decision label is the frozen Global24 prediction.
        validation_predicted = (
            frozen["base"][outer_validation] >= 0.5
        ).astype(np.int64)
        # The disjoint calibration split has no frozen model fitted only on the
        # proper subset, so its all-view auxiliary head freezes tau.
        calibration_predicted = (
            calibration_probability["all"] >= 0.5
        ).astype(np.int64)
        validation_r = route_score(validation_pi, validation_predicted)
        calibration_r = route_score(calibration_pi, calibration_predicted)
        tau = float(np.quantile(
            calibration_r, 1.0 - float(args.budget), method="higher"
        ))
        view_range = np.max(validation_matrix, axis=1) - np.min(validation_matrix, axis=1)

        fold_routes = {
            "A25": frozen["a25_route"][outer_validation],
            "ViewRange25": top_fraction(view_range, args.budget),
            "MVCC_Top25": top_fraction(validation_r, args.budget),
            "MVCC_FrozenTau": validation_r >= tau,
        }
        fold_scores = {
            "A25": frozen["a25_score"][outer_validation],
            "ViewRange25": view_range,
            "MVCC_Top25": validation_r,
            "MVCC_FrozenTau": validation_r,
        }
        for name in route_names:
            route_all[name][outer_validation] = fold_routes[name]
            score_all[name][outer_validation] = fold_scores[name]
            per_fold_routing.append(routing_row(
                name,
                labels[outer_validation],
                frozen["base"][outer_validation],
                fold_routes[name],
                fold_scores[name],
                fold,
            ))
        for view in VIEW_NAMES:
            view_probability_all[view][outer_validation] = validation_probability[view]
        pvalue_all[outer_validation] = validation_pi

        for offset, view in enumerate(VIEW_NAMES):
            weight_rows.append({
                "fold": int(fold),
                "view": view,
                "feature_n": int(len(groups[view])),
                "features": "|".join(names[index] for index in groups[view]),
                "heldout_oof_nll": float(heldout_nll[offset]),
                "reliability_weight": float(reliability_weight[offset]),
                "eta": float(args.eta),
            })
        fold_audit.append({
            "fold": int(fold),
            "outer_train_n": int(len(outer_train)),
            "proper_train_n": int(len(proper_index)),
            "conformal_calibration_n": int(len(calibration_index)),
            "outer_validation_n": int(len(outer_validation)),
            "calibration_benign_n": int(np.sum(calibration_y == 0)),
            "calibration_malware_n": int(np.sum(calibration_y == 1)),
            "frozen_tau": tau,
            "calibration_route_ratio_at_tau": float(np.mean(calibration_r >= tau)),
            "validation_route_ratio_at_tau": float(np.mean(validation_r >= tau)),
            "auxiliary_all_vs_frozen_base_label_disagreement": float(np.mean(
                (validation_probability["all"] >= 0.5)
                != (frozen["base"][outer_validation] >= 0.5)
            )),
        })

    if any(np.any(~np.isfinite(value)) for value in score_all.values()):
        raise RuntimeError("incomplete route scores")
    if any(np.any(~np.isfinite(value)) for value in view_probability_all.values()):
        raise RuntimeError("incomplete view probabilities")
    if np.any(~np.isfinite(pvalue_all)):
        raise RuntimeError("incomplete conformal p-values")

    final_probability = {"Global24": frozen["base"]}
    routing_summary = []
    for name in route_names:
        method = name + "__CARE_RGCondAttn_Dedup_FixKeep"
        final_probability[method] = routed_probability(
            frozen["base"], frozen["expert"], route_all[name]
        )
        routing_summary.append(routing_row(
            name, labels, frozen["base"], route_all[name], score_all[name]
        ))

    metric_summary = []
    for name, probability in final_probability.items():
        route = np.zeros(len(labels), dtype=bool)
        if name != "Global24":
            route_name = name.split("__", 1)[0]
            route = route_all[route_name]
        metric_summary.extend(v4.v2.metric_rows(
            labels,
            {name: probability},
            frozen["base"],
            route,
        ))

    per_fold_metric = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        for name, probability in final_probability.items():
            route = np.zeros(int(np.sum(selected)), dtype=bool)
            if name != "Global24":
                route_name = name.split("__", 1)[0]
                route = route_all[route_name][selected]
            rows = v4.v2.metric_rows(
                labels[selected],
                {name: probability[selected]},
                frozen["base"][selected],
                route,
            )
            per_fold_metric.append({"fold": int(fold), **rows[0]})

    prediction_rows = []
    for index, row in enumerate(manifest):
        item = {
            "sha256": row["sha256"],
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "Global24__prob": float(frozen["base"][index]),
            "FrozenDedupExpert__prob": float(frozen["expert"][index]),
            "conformal_pi_0": float(pvalue_all[index, 0]),
            "conformal_pi_1": float(pvalue_all[index, 1]),
        }
        for view in VIEW_NAMES:
            item["view_" + view + "__calibrated_prob"] = float(
                view_probability_all[view][index]
            )
        for name in route_names:
            item[name + "__score"] = float(score_all[name][index])
            item[name + "__routed"] = int(route_all[name][index])
            item[name + "__final_prob"] = float(
                final_probability[name + "__CARE_RGCondAttn_Dedup_FixKeep"][index]
            )
        prediction_rows.append(item)

    write_csv(args.output / "routing_summary.csv", routing_summary)
    write_csv(args.output / "source_metrics.csv", metric_summary)
    write_csv(args.output / "source_per_fold_metrics.csv", per_fold_metric)
    write_csv(args.output / "source_per_fold_routing.csv", per_fold_routing)
    write_csv(args.output / "view_weights.csv", weight_rows)
    write_csv(args.output / "fold_audit.csv", fold_audit)
    write_csv(args.output / "source_predictions.csv", prediction_rows)

    architecture = {
        "experiment": "CARE-Droid multi-view class-conditional conformal router source v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "future_data_loaded": False,
        "base": "frozen Global24 OOF probability",
        "expert": "frozen deduplicated bounded RGCondAttn probability",
        "changed_component": "route only",
        "views": {"all": sorted(set().union(*EXPECTED_GROUPS.values())), **{
            name: sorted(fields) for name, fields in EXPECTED_GROUPS.items()
        }},
        "probability_calibration": (
            "Platt scaling fitted on a subset of proper-training OOF probabilities"
        ),
        "weight_estimation": (
            "softmax(-eta * heldout OOF NLL) on a disjoint proper-training subset"
        ),
        "conformal_calibration": "class-conditional split calibration disjoint from proper training",
        "route_score": "1-pi(predicted_label)+pi(opposite_label)",
        "top25": "exact outer-validation unlabeled ranking; fixed-budget comparator",
        "frozen_tau": "75th percentile of leave-self-out calibration routing scores",
        "formal_boundary": (
            "class-conditional p-values rely on exchangeability; the composite routing "
            "score itself is not claimed to guarantee accepted-set error under time drift"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics_by_name = {row["method"]: row for row in metric_summary}
    route_by_name = {row["method"]: row for row in routing_summary}
    lines = [
        "CARE-Droid multi-view class-conditional conformal router source v1",
        "",
        "Strict source-only evaluation: fixed 2017 outer folds; future rows loaded: no.",
        "Frozen components: Global24 base and deduplicated bounded RGCondAttn expert.",
        "Only the route changes. Platt/NLL/conformal operations are nested in outer training.",
        f"Budget: {100.0 * args.budget:.2f}%; eta={args.eta:.3f}.",
        "",
        "Routing quality against frozen Global24 errors:",
    ]
    for name in route_names:
        row = route_by_name[name]
        lines.append(
            f"{name}: routed={int(row['routed_n'])}/{int(row['n'])} "
            f"({100.0 * float(row['routed_ratio']):.2f}%) "
            f"error_recall={float(row['error_recall']):.6f} "
            f"route_precision={float(row['route_error_precision']):.6f} "
            f"accepted_risk={float(row['accepted_risk']):.6f}"
        )
    lines.extend(["", "End-to-end source metrics with the same frozen expert:"])
    for name, row in metrics_by_name.items():
        lines.append(
            f"{name}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} "
            f"F1={float(row['F1']):.6f} fixed={int(row['fixed'])} "
            f"broken={int(row['broken'])} net={int(row['net'])}"
        )
    lines.extend([
        "",
        "Decision rule (source only):",
        "At equal Top-25% budget, MVCC must improve Global24 error recall and",
        "accepted risk over A25, while final BA is non-inferior in at least 4/5 folds",
        "and fixed/broken/net do not deteriorate materially.",
        "FrozenTau is reported separately because its deployment route rate need not be 25%.",
        "Do not use 2018-2021 to tune eta, groups, calibration split, or threshold.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
