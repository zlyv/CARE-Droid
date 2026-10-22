#!/usr/bin/env python3
"""Strict 2017 ACN routing ablation for the fixed C76 two-layer mainline.

The predictive models are deliberately held fixed:
  * first layer: allratio76_ratio52 (C76) logistic model;
  * second-layer expert: C76 + Stats120 + PairRaw465 JointLogistic.

Only the label-free-at-test-time routing score changes.  Under the same budget
the script compares Random, Ambiguity, Collision, Novelty, and ACN=max(A,C,N).
All risk calibration is nested inside each 2017 outer training fold.  No future
rows or labels are loaded by this source-only experiment.

Copy this file into ``scripts/`` in the tasa-droid repository and run it from
the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_allratio70_vs_nodrop76_experiment as audit


THRESHOLD = 0.5
VARIANT = "allratio76_ratio52"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_fraction(score, fraction):
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1:
        raise ValueError("routing score must be one-dimensional")
    count = max(1, int(round(len(score) * float(fraction))))
    count = min(len(score), count)
    order = np.argsort(-score, kind="mergesort")
    route = np.zeros(len(score), dtype=bool)
    route[order[:count]] = True
    return route


def random_fraction(size, fraction, seed):
    count = max(1, int(round(int(size) * float(fraction))))
    count = min(int(size), count)
    rng = np.random.RandomState(int(seed))
    chosen = rng.choice(int(size), size=count, replace=False)
    route = np.zeros(int(size), dtype=bool)
    route[chosen] = True
    return route


def percentile_against(reference, values):
    """Empirical CDF fitted only on the current outer-training OOF risks."""
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(values)):
        raise RuntimeError("non-finite value in risk calibration")
    ordered = np.sort(reference, kind="mergesort")
    return np.searchsorted(ordered, values, side="right") / float(len(ordered))


def ambiguity(probability):
    probability = np.asarray(probability, dtype=np.float64)
    return 1.0 - 2.0 * np.abs(probability - THRESHOLD)


def neighborhood_risks(x_train, y_train, x_eval, k):
    rows = audit.e5.collision_scores(
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(x_eval, dtype=np.float64),
        int(k),
    )
    collision = np.asarray(
        [row["local_label_entropy"] for row in rows], dtype=np.float64
    )
    novelty = np.asarray(
        [row["mean_knn_distance"] for row in rows], dtype=np.float64
    )
    proximity = np.asarray(
        [row["local_proximity"] for row in rows], dtype=np.float64
    )
    malware_fraction = np.asarray(
        [row["local_malware_fraction"] for row in rows], dtype=np.float64
    )
    return collision, novelty, proximity, malware_fraction


def routed_probability(coarse, expert, route):
    result = np.asarray(coarse, dtype=np.float64).copy()
    route = np.asarray(route, dtype=bool)
    result[route] = np.asarray(expert, dtype=np.float64)[route]
    return result


def metric_row(method, y, probability, coarse, route):
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    coarse = np.asarray(coarse, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    result = audit.base.metrics(y, probability, threshold=THRESHOLD)
    coarse_correct = (coarse >= THRESHOLD) == y
    final_correct = (probability >= THRESHOLD) == y
    fixed = int(np.sum(~coarse_correct & final_correct))
    broken = int(np.sum(coarse_correct & ~final_correct))
    changed = fixed + broken
    return {
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(route)),
        "routed_ratio": float(np.mean(route)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "correction_precision": float(fixed / changed) if changed else 0.0,
        "ACC": float(result["accuracy"]),
        "BA": float(result["balanced_accuracy"]),
        "AUC": float(result["auc"]),
        "F1": float(result["f1"]),
        "TP": int(result["tp"]),
        "TN": int(result["tn"]),
        "FP": int(result["fp"]),
        "FN": int(result["fn"]),
    }


def route_diagnostic_row(method, y, coarse, route, score):
    y = np.asarray(y, dtype=np.int64)
    coarse = np.asarray(coarse, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    coarse_pred = coarse >= THRESHOLD
    errors = coarse_pred != y
    confidence = np.maximum(coarse, 1.0 - coarse)
    high_conf_errors = errors & (confidence >= 0.80)
    error_count = int(np.sum(errors))
    high_conf_count = int(np.sum(high_conf_errors))
    routed_errors = int(np.sum(route & errors))
    routed_high_conf_errors = int(np.sum(route & high_conf_errors))
    return {
        "route": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(route)),
        "routed_ratio": float(np.mean(route)),
        "coarse_errors": error_count,
        "routed_coarse_errors": routed_errors,
        "error_capture_rate": float(routed_errors / error_count) if error_count else 0.0,
        "route_error_precision": float(routed_errors / np.sum(route)) if np.sum(route) else 0.0,
        "high_confidence_error_threshold": 0.80,
        "coarse_high_confidence_errors": high_conf_count,
        "routed_high_confidence_errors": routed_high_conf_errors,
        "high_confidence_error_capture_rate": (
            float(routed_high_conf_errors / high_conf_count) if high_conf_count else 0.0
        ),
        "mean_score_routed": float(np.mean(score[route])) if np.any(route) else 0.0,
        "mean_score_not_routed": float(np.mean(score[~route])) if np.any(~route) else 0.0,
    }


def exact_mcnemar(left_correct, right_correct):
    return audit.exact_mcnemar_p(
        np.asarray(left_correct, dtype=bool),
        np.asarray(right_correct, dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path("artifacts/p1-features/feature_schema.json"),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--p72-oof", type=Path,
        default=Path(
            "artifacts/p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/acn-routing-ablation-2017-v1"),
    )
    parser.add_argument("--budget", type=float, default=0.20)
    # Keep the already used semantic-collision neighborhood size as the first
    # frozen value; sensitivity analysis, if needed, remains source-only.
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument("--random-repeats", type=int, default=100)
    args = parser.parse_args()

    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be between zero and one")
    if args.knn_k < 2:
        raise SystemExit("--knn-k must be at least two")
    if args.inner_folds < 2:
        raise SystemExit("--inner-folds must be at least two")
    if args.random_repeats < 1:
        raise SystemExit("--random-repeats must be at least one")

    args.output.mkdir(parents=True, exist_ok=True)
    (
        manifest,
        cohort_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    cache = audit.exp.read_cache(args.cache)
    print("Reconstructing source Stats120+PairRaw465...")
    structure, structure_names = audit.future.reconstruct_structure(
        cohort_rows, cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if structure.shape != (799, 585):
        raise RuntimeError(
            "Expected source structure shape (799,585), got "
            + repr(structure.shape)
        )

    y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    unique_folds = sorted(np.unique(folds))
    n = len(y)
    budget_tag = str(int(round(100.0 * args.budget)))
    route_names = [
        "Random" + budget_tag,
        "Ambiguity" + budget_tag,
        "Collision" + budget_tag,
        "Novelty" + budget_tag,
        "ACN" + budget_tag,
    ]
    methods = ["C76", "JointLogistic100"] + route_names

    probabilities = {name: np.full(n, np.nan) for name in methods}
    routes = {name: np.zeros(n, dtype=bool) for name in route_names}
    raw_risks = {
        "ambiguity": np.full(n, np.nan),
        "collision": np.full(n, np.nan),
        "novelty": np.full(n, np.nan),
        "local_proximity": np.full(n, np.nan),
        "local_malware_fraction": np.full(n, np.nan),
    }
    calibrated_risks = {
        "ambiguity": np.full(n, np.nan),
        "collision": np.full(n, np.nan),
        "novelty": np.full(n, np.nan),
        "acn": np.full(n, np.nan),
    }
    random_probabilities = [np.full(n, np.nan) for _ in range(args.random_repeats)]
    random_routes = [np.zeros(n, dtype=bool) for _ in range(args.random_repeats)]
    fold_audit = []

    for fold in unique_folds:
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[i] for i in train_idx]
        val_rows = [cohort_rows[i] for i in val_idx]
        x_train, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        coarse, _, _ = audit.p72.fit_predict(
            x_train, y[train_idx], x_val, seed=int(fold) - 1
        )
        expert = audit.fit_config(
            x_train,
            x_val,
            structure[train_idx],
            structure[val_idx],
            y[train_idx],
            seed=int(fold) - 1,
        )

        # Strict inner OOF risks define the calibration distribution for this
        # outer fold.  Each outer-training APK is scored by a model/neighborhood
        # that did not include that APK.
        inner_ambiguity = np.full(len(train_idx), np.nan)
        inner_collision = np.full(len(train_idx), np.nan)
        inner_novelty = np.full(len(train_idx), np.nan)
        splitter = StratifiedKFold(
            n_splits=int(args.inner_folds),
            shuffle=True,
            random_state=int(args.split_seed + 1000 * int(fold)),
        )
        for inner_fold, (inner_train_local, inner_val_local) in enumerate(
            splitter.split(np.zeros(len(train_idx)), y[train_idx]), start=1
        ):
            inner_train_rows = [train_rows[i] for i in inner_train_local]
            inner_val_rows = [train_rows[i] for i in inner_val_local]
            x_inner_train, x_inner_val, _ = audit.compact_matrix_variant(
                VARIANT,
                inner_train_rows,
                inner_val_rows,
                compact_by_sha,
                compact_schema,
                original_names,
            )
            inner_probability, _, _ = audit.p72.fit_predict(
                x_inner_train,
                y[train_idx][inner_train_local],
                x_inner_val,
                seed=int(args.split_seed + 10000 * int(fold) + inner_fold),
            )
            collision_part, novelty_part, _, _ = neighborhood_risks(
                x_inner_train,
                y[train_idx][inner_train_local],
                x_inner_val,
                args.knn_k,
            )
            inner_ambiguity[inner_val_local] = ambiguity(inner_probability)
            inner_collision[inner_val_local] = collision_part
            inner_novelty[inner_val_local] = novelty_part

        if any(
            np.any(~np.isfinite(value))
            for value in (inner_ambiguity, inner_collision, inner_novelty)
        ):
            raise RuntimeError(f"fold {fold}: incomplete inner OOF risks")

        outer_collision, outer_novelty, outer_proximity, outer_fraction = (
            neighborhood_risks(x_train, y[train_idx], x_val, args.knn_k)
        )
        outer_ambiguity = ambiguity(coarse)
        calibrated_ambiguity = percentile_against(
            inner_ambiguity, outer_ambiguity
        )
        calibrated_collision = percentile_against(
            inner_collision, outer_collision
        )
        calibrated_novelty = percentile_against(
            inner_novelty, outer_novelty
        )
        calibrated_acn = np.maximum.reduce(
            [calibrated_ambiguity, calibrated_collision, calibrated_novelty]
        )

        fold_routes = {
            "Ambiguity" + budget_tag: top_fraction(
                calibrated_ambiguity, args.budget
            ),
            "Collision" + budget_tag: top_fraction(
                calibrated_collision, args.budget
            ),
            "Novelty" + budget_tag: top_fraction(
                calibrated_novelty, args.budget
            ),
            "ACN" + budget_tag: top_fraction(calibrated_acn, args.budget),
        }
        for repeat in range(args.random_repeats):
            random_route = random_fraction(
                len(val_idx),
                args.budget,
                args.random_seed + 100000 * repeat + 1000 * int(fold),
            )
            random_routes[repeat][val_idx] = random_route
            random_probabilities[repeat][val_idx] = routed_probability(
                coarse, expert, random_route
            )
        fold_routes["Random" + budget_tag] = random_routes[0][val_idx]

        probabilities["C76"][val_idx] = coarse
        probabilities["JointLogistic100"][val_idx] = expert
        for name, route in fold_routes.items():
            routes[name][val_idx] = route
            probabilities[name][val_idx] = routed_probability(coarse, expert, route)

        raw_risks["ambiguity"][val_idx] = outer_ambiguity
        raw_risks["collision"][val_idx] = outer_collision
        raw_risks["novelty"][val_idx] = outer_novelty
        raw_risks["local_proximity"][val_idx] = outer_proximity
        raw_risks["local_malware_fraction"][val_idx] = outer_fraction
        calibrated_risks["ambiguity"][val_idx] = calibrated_ambiguity
        calibrated_risks["collision"][val_idx] = calibrated_collision
        calibrated_risks["novelty"][val_idx] = calibrated_novelty
        calibrated_risks["acn"][val_idx] = calibrated_acn

        fold_audit.append({
            "fold": int(fold),
            "outer_train_n": int(len(train_idx)),
            "outer_val_n": int(len(val_idx)),
            "inner_folds": int(args.inner_folds),
            "compact_dim": int(len(names)),
            "structure_dim": int(structure.shape[1]),
            "knn_k": int(args.knn_k),
            "budget": float(args.budget),
            "route_n": int(np.sum(fold_routes["ACN" + budget_tag])),
            "inner_ambiguity_min": float(np.min(inner_ambiguity)),
            "inner_ambiguity_max": float(np.max(inner_ambiguity)),
            "inner_collision_min": float(np.min(inner_collision)),
            "inner_collision_max": float(np.max(inner_collision)),
            "inner_novelty_min": float(np.min(inner_novelty)),
            "inner_novelty_max": float(np.max(inner_novelty)),
        })
        print("fold", int(fold), "done")

    if any(np.any(~np.isfinite(value)) for value in probabilities.values()):
        raise RuntimeError("incomplete OOF probabilities")
    if any(
        np.any(~np.isfinite(value))
        for value in list(raw_risks.values()) + list(calibrated_risks.values())
    ):
        raise RuntimeError("incomplete OOF risks")
    if any(
        np.any(~np.isfinite(value)) for value in random_probabilities
    ):
        raise RuntimeError("incomplete random-repeat probabilities")

    zero_route = np.zeros(n, dtype=bool)
    full_route = np.ones(n, dtype=bool)
    method_routes = {
        "C76": zero_route,
        "JointLogistic100": full_route,
        **routes,
    }
    summary_rows = [
        metric_row(
            method,
            y,
            probabilities[method],
            probabilities["C76"],
            method_routes[method],
        )
        for method in methods
    ]
    per_fold_rows = []
    for fold in unique_folds:
        idx = folds == fold
        for method in methods:
            per_fold_rows.append({
                **metric_row(
                    method,
                    y[idx],
                    probabilities[method][idx],
                    probabilities["C76"][idx],
                    method_routes[method][idx],
                ),
                "fold": int(fold),
            })

    score_by_route = {
        "Random" + budget_tag: np.zeros(n, dtype=np.float64),
        "Ambiguity" + budget_tag: calibrated_risks["ambiguity"],
        "Collision" + budget_tag: calibrated_risks["collision"],
        "Novelty" + budget_tag: calibrated_risks["novelty"],
        "ACN" + budget_tag: calibrated_risks["acn"],
    }
    route_diagnostic_rows = []
    route_fold_rows = []
    for name in route_names:
        route_diagnostic_rows.append(
            route_diagnostic_row(
                name, y, probabilities["C76"], routes[name], score_by_route[name]
            )
        )
        for fold in unique_folds:
            idx = folds == fold
            route_fold_rows.append({
                **route_diagnostic_row(
                    name,
                    y[idx],
                    probabilities["C76"][idx],
                    routes[name][idx],
                    score_by_route[name][idx],
                ),
                "fold": int(fold),
            })

    pairwise_rows = []
    for left, right in combinations(methods, 2):
        left_correct = (probabilities[left] >= THRESHOLD) == y
        right_correct = (probabilities[right] >= THRESHOLD) == y
        left_only, right_only, p_value = exact_mcnemar(left_correct, right_correct)
        pairwise_rows.append({
            "left": left,
            "right": right,
            "left_only_correct": int(left_only),
            "right_only_correct": int(right_only),
            "discordant": int(left_only + right_only),
            "mcnemar_exact_p": float(p_value),
            "BA_left": float(audit.base.metrics(
                y, probabilities[left], threshold=THRESHOLD
            )["balanced_accuracy"]),
            "BA_right": float(audit.base.metrics(
                y, probabilities[right], threshold=THRESHOLD
            )["balanced_accuracy"]),
        })

    overlap_rows = []
    for left, right in combinations(route_names[1:], 2):
        intersection = int(np.sum(routes[left] & routes[right]))
        union = int(np.sum(routes[left] | routes[right]))
        overlap_rows.append({
            "left": left,
            "right": right,
            "intersection": intersection,
            "union": union,
            "jaccard": float(intersection / union) if union else 1.0,
        })

    random_rows = []
    for repeat in range(args.random_repeats):
        random_rows.append({
            **metric_row(
                "Random" + budget_tag,
                y,
                random_probabilities[repeat],
                probabilities["C76"],
                random_routes[repeat],
            ),
            "repeat": int(repeat),
            "seed": int(args.random_seed + 100000 * repeat),
        })

    prediction_rows = []
    for i, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(y[i]),
            "fold": int(folds[i]),
            "ambiguity_raw": float(raw_risks["ambiguity"][i]),
            "collision_raw_entropy": float(raw_risks["collision"][i]),
            "novelty_raw_mean_knn_distance": float(raw_risks["novelty"][i]),
            "local_proximity": float(raw_risks["local_proximity"][i]),
            "local_malware_fraction": float(raw_risks["local_malware_fraction"][i]),
            "ambiguity_percentile": float(calibrated_risks["ambiguity"][i]),
            "collision_percentile": float(calibrated_risks["collision"][i]),
            "novelty_percentile": float(calibrated_risks["novelty"][i]),
            "acn_score": float(calibrated_risks["acn"][i]),
        }
        for name in methods:
            output[name + "__prob"] = float(probabilities[name][i])
        for name in route_names:
            output[name + "__routed"] = int(routes[name][i])
        prediction_rows.append(output)

    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "route_diagnostics.csv", route_diagnostic_rows)
    write_csv(args.output / "route_diagnostics_per_fold.csv", route_fold_rows)
    write_csv(args.output / "route_overlap.csv", overlap_rows)
    write_csv(args.output / "pairwise_tests.csv", pairwise_rows)
    write_csv(args.output / "random_repeats.csv", random_rows)
    write_csv(args.output / "fold_audit.csv", fold_audit)

    acn_name = "ACN" + budget_tag
    ambiguity_name = "Ambiguity" + budget_tag
    acn_fold_ba = {
        int(row["fold"]): float(row["BA"])
        for row in per_fold_rows if row["method"] == acn_name
    }
    ambiguity_fold_ba = {
        int(row["fold"]): float(row["BA"])
        for row in per_fold_rows if row["method"] == ambiguity_name
    }
    noninferior_folds = sum(
        acn_fold_ba[fold] + 1e-12 >= ambiguity_fold_ba[fold]
        for fold in unique_folds
    )
    summary_by_method = {row["method"]: row for row in summary_rows}
    route_by_method = {row["route"]: row for row in route_diagnostic_rows}
    improves_error_capture = (
        route_by_method[acn_name]["error_capture_rate"]
        > route_by_method[ambiguity_name]["error_capture_rate"] + 1e-12
    )
    improves_net_correction = (
        summary_by_method[acn_name]["net"]
        > summary_by_method[ambiguity_name]["net"]
    )
    passes_fold_part = bool(noninferior_folds >= 4)
    passes_full_retention = bool(
        passes_fold_part and (improves_error_capture or improves_net_correction)
    )
    metadata = {
        "experiment": "acn-routing-ablation-2017-v1",
        "source_only": True,
        "source_apks": int(n),
        "outer_folds": [int(value) for value in unique_folds],
        "coarse": "allratio76_ratio52 / C76",
        "coarse_dim": 76,
        "expert": "C76+Stats120+PairRaw465 JointLogistic",
        "structure_dim": int(structure.shape[1]),
        "structure_name_count": int(len(structure_names)),
        "budget": float(args.budget),
        "knn_k": int(args.knn_k),
        "inner_folds": int(args.inner_folds),
        "risk_definitions": {
            "ambiguity": "1 - 2*abs(p_C76-0.5)",
            "collision": "binary label entropy among k nearest outer-train C76 neighbors",
            "novelty": "mean distance to k nearest outer-train C76 neighbors",
            "calibration": "empirical percentile against strict inner OOF source risks",
            "acn": "max(calibrated ambiguity, collision, novelty)",
        },
        "test_label_used_by_router": False,
        "future_rows_loaded": False,
        "random_repeats": int(args.random_repeats),
        "predeclared_acn_retention_check": {
            "criterion": "ACN BA is not below Ambiguity BA in at least 4/5 outer folds, and improves error capture or net correction",
            "noninferior_outer_folds": int(noninferior_folds),
            "passes_fold_part": passes_fold_part,
            "improves_error_capture": bool(improves_error_capture),
            "improves_net_correction": bool(improves_net_correction),
            "passes_full_retention": passes_full_retention,
        },
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done:", args.output)


if __name__ == "__main__":
    main()
