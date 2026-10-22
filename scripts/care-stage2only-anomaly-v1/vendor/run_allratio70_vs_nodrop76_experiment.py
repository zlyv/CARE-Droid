#!/usr/bin/env python3
"""
Paired audit of the six future-selected role dimensions.

Run this file from the tasa-droid repository root after copying it into
``scripts/``.  It keeps the fixed 799-APK / five-fold 2017 protocol and
compares three representations:

1. allratio70_current
   Current 24 global + 46 semantic-ratio representation.  The six role
   dimensions are removed.
2. nodrop76_raw6
   Exact deletion reversal: keep the same 46 semantic ratios and restore
   the six excluded dimensions in their original fold-standardized form.
3. allratio76_ratio52
   Coherent no-drop representation: all 52 role_node/role_edge dimensions
   are ratios, with statistics fitted on the current training fold only.

For every representation the script evaluates coarse-only, Stats120,
PairRaw465, Stats120+PairRaw465, and a strict source-calibrated 20% semantic
collision gate.  Optional 2018-2021 output is diagnostic only because those
years have already participated in earlier representation selection.
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_compact_quantity_shared_caller_2x3_2017_eval as exp
import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72
import run_benefitdroid_e5_semantic_count_collision_2017 as e5
import run_benefitdroid_e7b_strict_source_joint_gate_2017 as e7b
import run_semantic_collision_frozen_future_eval as future


THRESHOLD = 0.5
DEFAULT_K = 15
DEFAULT_BUDGET = 0.20

VARIANTS = (
    "allratio70_current",
    "nodrop76_raw6",
    "allratio76_ratio52",
)

STRUCTURE_CONFIGS = (
    "C",
    "C+Stats",
    "C+PairRaw",
    "C+Stats+PairRaw",
)


def norm_sha(value):
    return str(value or "").strip().casefold()


def safe_float(value):
    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return 0.0


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def restore_raw_compact76(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
):
    """Restore the fixed source Compact76 schema without validation leakage."""
    fold_train, fold_test, fold_names = (
        allratio.original_compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
        )
    )

    fold_train = np.asarray(fold_train, dtype=np.float64)
    fold_test = np.asarray(fold_test, dtype=np.float64)
    fold_names = list(fold_names)
    canonical_original_names = list(canonical_original_names)

    if len(canonical_original_names) != 76:
        raise RuntimeError(
            "Expected canonical Compact76, got "
            + str(len(canonical_original_names))
        )
    if len(set(canonical_original_names)) != 76:
        raise RuntimeError("Duplicate names in canonical Compact76")
    if len(set(fold_names)) != len(fold_names):
        raise RuntimeError("Duplicate names in fold Compact")

    canonical_set = set(canonical_original_names)
    unexpected = sorted(set(fold_names) - canonical_set)
    if unexpected:
        raise RuntimeError(
            "Fold produced features outside canonical Compact76: "
            + repr(unexpected)
        )

    fold_index = {name: i for i, name in enumerate(fold_names)}
    restored_train = np.zeros(
        (len(train_rows), 76),
        dtype=np.float64,
    )
    restored_test = np.zeros(
        (len(test_rows), 76),
        dtype=np.float64,
    )

    for target_index, name in enumerate(canonical_original_names):
        source_index = fold_index.get(name)
        if source_index is None:
            continue
        restored_train[:, target_index] = fold_train[:, source_index]
        restored_test[:, target_index] = fold_test[:, source_index]

    return restored_train, restored_test, canonical_original_names


def raw_ratio_values(
    rows,
    compact_by_sha,
    numerator_name,
    denominator_name,
):
    values = []
    for row in rows:
        source = compact_by_sha[norm_sha(row["sha256"])]
        numerator = safe_float(source.get(numerator_name, 0.0))
        denominator = safe_float(source.get(denominator_name, 0.0))
        values.append(
            numerator / denominator
            if denominator > 0.0
            else 0.0
        )
    return np.asarray(values, dtype=np.float64)


def replace_with_ratios(
    train_matrix,
    test_matrix,
    names,
    train_rows,
    test_rows,
    compact_by_sha,
    ratio_feature_names,
):
    train_matrix = np.asarray(train_matrix, dtype=np.float64).copy()
    test_matrix = np.asarray(test_matrix, dtype=np.float64).copy()
    names = list(names)

    for numerator_name in ratio_feature_names:
        if numerator_name.startswith("role_node__"):
            denominator_name = "sensitive_node_count"
        elif numerator_name.startswith("role_edge__"):
            denominator_name = "sensitive_edge_count"
        else:
            raise RuntimeError(
                "Unknown semantic feature type: " + numerator_name
            )

        column_index = names.index(numerator_name)
        train_ratio = raw_ratio_values(
            train_rows,
            compact_by_sha,
            numerator_name,
            denominator_name,
        )
        test_ratio = raw_ratio_values(
            test_rows,
            compact_by_sha,
            numerator_name,
            denominator_name,
        )

        mean = float(np.mean(train_ratio))
        std = float(np.std(train_ratio))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0

        train_matrix[:, column_index] = (train_ratio - mean) / std
        test_matrix[:, column_index] = (test_ratio - mean) / std

    return train_matrix, test_matrix, names


def compact_matrix_variant(
    variant,
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
):
    if variant == "allratio70_current":
        x_train, x_test, names = (
            p72.compact_matrix_allratio70_foldsafe(
                train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )
        )
    else:
        x_train, x_test, names = restore_raw_compact76(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )

        semantic_names = [
            name
            for name in names
            if name.startswith("role_node__")
            or name.startswith("role_edge__")
        ]
        if len(semantic_names) != 52:
            raise RuntimeError(
                "Expected 52 raw semantic dimensions, got "
                + str(len(semantic_names))
            )

        drop_features = set(allratio.DROP_FEATURES)
        if len(drop_features) != 6:
            raise RuntimeError(
                "Expected six DROP_FEATURES, got "
                + repr(sorted(drop_features))
            )
        if not drop_features.issubset(set(names)):
            raise RuntimeError(
                "Compact76 is missing DROP_FEATURES: "
                + repr(sorted(drop_features - set(names)))
            )

        if variant == "nodrop76_raw6":
            ratio_names = [
                name
                for name in semantic_names
                if name not in drop_features
            ]
        elif variant == "allratio76_ratio52":
            ratio_names = semantic_names
        else:
            raise RuntimeError("Unknown variant: " + str(variant))

        x_train, x_test, names = replace_with_ratios(
            x_train,
            x_test,
            names,
            train_rows,
            test_rows,
            compact_by_sha,
            ratio_names,
        )

    expected_dim = 70 if variant == "allratio70_current" else 76
    expected_semantic = 46 if variant == "allratio70_current" else 52
    semantic_count = sum(
        name.startswith("role_node__")
        or name.startswith("role_edge__")
        for name in names
    )

    if len(names) != expected_dim:
        raise RuntimeError(
            f"{variant}: expected dim={expected_dim}, got {len(names)}"
        )
    if semantic_count != expected_semantic:
        raise RuntimeError(
            f"{variant}: expected semantic={expected_semantic}, "
            f"got {semantic_count}"
        )

    return (
        np.asarray(x_train, dtype=np.float64),
        np.asarray(x_test, dtype=np.float64),
        list(names),
    )


def semantic_indices(names):
    indices = [
        i
        for i, name in enumerate(names)
        if name.startswith("role_node__")
        or name.startswith("role_edge__")
    ]
    return np.asarray(indices, dtype=np.int64)


def structure_configurations(structure_all):
    if structure_all.shape[1] != 585:
        raise RuntimeError(
            "Expected Stats120+PairRaw465, got "
            + str(structure_all.shape)
        )
    return {
        "C": None,
        "C+Stats": structure_all[:, :120],
        "C+PairRaw": structure_all[:, 120:],
        "C+Stats+PairRaw": structure_all,
    }


def fit_config(
    compact_train,
    compact_test,
    structure_train,
    structure_test,
    y_train,
    seed,
):
    if structure_train is None:
        x_train = compact_train
        x_test = compact_test
    else:
        added_names = [
            f"structure__{i:03d}"
            for i in range(structure_train.shape[1])
        ]
        z_train, z_test, _ = p72.standardize_added(
            structure_train,
            structure_test,
            added_names,
        )
        x_train = np.concatenate([compact_train, z_train], axis=1)
        x_test = np.concatenate([compact_test, z_test], axis=1)

    probability, _, _ = p72.fit_predict(
        x_train,
        y_train,
        x_test,
        seed=int(seed),
    )
    return np.asarray(probability, dtype=np.float64)


def metric_row(
    variant,
    method,
    labels,
    probability,
    coarse_probability,
    routed_n,
):
    result = base.metrics(
        labels,
        probability,
        threshold=THRESHOLD,
    )

    coarse_pred = np.asarray(coarse_probability >= THRESHOLD, dtype=int)
    pred = np.asarray(probability >= THRESHOLD, dtype=int)
    fixed = int(np.sum((coarse_pred != labels) & (pred == labels)))
    broken = int(np.sum((coarse_pred == labels) & (pred != labels)))
    tp = int(result["tp"])
    tn = int(result["tn"])
    fp = int(result["fp"])
    fn = int(result["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "variant": variant,
        "method": method,
        "compact_dim": 70 if variant == "allratio70_current" else 76,
        "semantic_dim": 46 if variant == "allratio70_current" else 52,
        "routed_n": int(routed_n),
        "routed_ratio": float(routed_n / len(labels)),
        "ACC": float(result["accuracy"]),
        "BA": float(result["balanced_accuracy"]),
        "AUC": float(result["auc"]),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(result["f1"]),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "fixed_vs_C": fixed,
        "broken_vs_C": broken,
        "net_vs_C": fixed - broken,
    }


def exact_mcnemar_p(left_correct, right_correct):
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0

    tail = min(left_only, right_only)
    numerator = sum(math.comb(discordant, i) for i in range(tail + 1))
    p_value = min(1.0, 2.0 * numerator / (2 ** discordant))
    return left_only, right_only, float(p_value)


def paired_comparison_rows(labels, probability_by_key, methods):
    rows = []
    baseline_variant = "allratio70_current"
    for variant in VARIANTS:
        if variant == baseline_variant:
            continue
        for method in methods:
            left_prob = probability_by_key[(baseline_variant, method)]
            right_prob = probability_by_key[(variant, method)]
            left_correct = (
                np.asarray(left_prob >= THRESHOLD, dtype=int) == labels
            )
            right_correct = (
                np.asarray(right_prob >= THRESHOLD, dtype=int) == labels
            )
            left_only, right_only, p_value = exact_mcnemar_p(
                left_correct,
                right_correct,
            )
            left_metrics = base.metrics(labels, left_prob, threshold=THRESHOLD)
            right_metrics = base.metrics(labels, right_prob, threshold=THRESHOLD)
            rows.append({
                "baseline_variant": baseline_variant,
                "candidate_variant": variant,
                "method": method,
                "baseline_BA": float(left_metrics["balanced_accuracy"]),
                "candidate_BA": float(right_metrics["balanced_accuracy"]),
                "delta_BA": float(
                    right_metrics["balanced_accuracy"]
                    - left_metrics["balanced_accuracy"]
                ),
                "baseline_only_correct": left_only,
                "candidate_only_correct": right_only,
                "exact_mcnemar_p": p_value,
            })
    return rows


def strict_collision_gate(
    variant,
    cohort_rows,
    labels,
    folds,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
    k,
    budget,
):
    n = len(cohort_rows)
    raw_collision = np.full(n, np.nan, dtype=np.float64)
    collision_percentile = np.full(n, np.nan, dtype=np.float64)
    routed = np.zeros(n, dtype=bool)
    threshold_rows = []
    unique_folds = sorted(np.unique(folds))

    for outer_fold in unique_folds:
        outer_train_idx = np.flatnonzero(folds != outer_fold)
        outer_test_idx = np.flatnonzero(folds == outer_fold)
        inner_collision = np.full(n, np.nan, dtype=np.float64)

        for inner_fold in unique_folds:
            if inner_fold == outer_fold:
                continue
            inner_train_idx = np.flatnonzero(
                (folds != outer_fold) & (folds != inner_fold)
            )
            inner_val_idx = np.flatnonzero(
                (folds != outer_fold) & (folds == inner_fold)
            )
            train_rows = [cohort_rows[i] for i in inner_train_idx]
            val_rows = [cohort_rows[i] for i in inner_val_idx]
            x_train, x_val, names = compact_matrix_variant(
                variant,
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )
            sem_idx = semantic_indices(names)
            rows = e5.collision_scores(
                x_train[:, sem_idx],
                labels[inner_train_idx],
                x_val[:, sem_idx],
                k,
            )
            inner_collision[inner_val_idx] = [
                float(row["collision_score"])
                for row in rows
            ]

        calibration_raw = inner_collision[outer_train_idx]
        if np.any(~np.isfinite(calibration_raw)):
            raise RuntimeError(
                f"{variant}: incomplete inner collision OOF"
            )

        train_rows = [cohort_rows[i] for i in outer_train_idx]
        test_rows = [cohort_rows[i] for i in outer_test_idx]
        x_train, x_test, names = compact_matrix_variant(
            variant,
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )
        sem_idx = semantic_indices(names)
        rows = e5.collision_scores(
            x_train[:, sem_idx],
            labels[outer_train_idx],
            x_test[:, sem_idx],
            k,
        )
        test_raw = np.asarray(
            [float(row["collision_score"]) for row in rows],
            dtype=np.float64,
        )

        test_percentile = e7b.empirical_percentile(
            calibration_raw,
            test_raw,
        )
        calibration_percentile = e7b.empirical_percentile(
            calibration_raw,
            calibration_raw,
        )
        threshold, source_actual_ratio = e7b.threshold_for_top_fraction(
            calibration_percentile,
            budget,
        )
        test_selected = test_percentile >= threshold

        raw_collision[outer_test_idx] = test_raw
        collision_percentile[outer_test_idx] = test_percentile
        routed[outer_test_idx] = test_selected
        threshold_rows.append({
            "variant": variant,
            "outer_fold": int(outer_fold),
            "k": int(k),
            "nominal_budget": float(budget),
            "source_threshold": float(threshold),
            "source_actual_route_ratio": float(source_actual_ratio),
            "test_routed_n": int(np.sum(test_selected)),
            "test_n": int(len(outer_test_idx)),
            "test_route_ratio": float(np.mean(test_selected)),
        })

    if np.any(~np.isfinite(raw_collision)):
        raise RuntimeError(f"{variant}: incomplete outer collision OOF")

    return raw_collision, collision_percentile, routed, threshold_rows


def load_inputs(args):
    reference_rows = read_csv(args.p72_oof)
    if len(reference_rows) != 799:
        raise RuntimeError(
            f"Expected fixed source cohort=799, got {len(reference_rows)}"
        )

    manifest = []
    for row in reference_rows:
        manifest.append({
            "sha256": norm_sha(row["sha256"]),
            "label": int(row["label"]),
            "metadata_label": row.get("metadata_label", ""),
            "fold": int(row["fold"]),
        })
    if len({row["sha256"] for row in manifest}) != 799:
        raise RuntimeError("Duplicate SHA in fixed source manifest")

    all_feature_rows = read_csv(args.features)
    compact_by_sha = {}
    clean_rows = []
    for row in all_feature_rows:
        if str(row.get("status", "")).strip() != "OK":
            continue
        sha = norm_sha(row.get("sha256", ""))
        if not sha:
            continue
        copied = dict(row)
        copied["sha256"] = sha
        compact_by_sha[sha] = copied
        clean_rows.append(copied)

    missing = sorted(
        {row["sha256"] for row in manifest} - set(compact_by_sha)
    )
    if missing:
        raise RuntimeError("Missing source feature rows: " + repr(missing[:10]))

    cohort_rows = [compact_by_sha[row["sha256"]] for row in manifest]
    for manifest_row, feature_row in zip(manifest, cohort_rows):
        if str(feature_row.get("metadata_year", "")).strip() != "2017":
            raise RuntimeError(
                "Fixed source manifest contains a non-2017 feature row: "
                + manifest_row["sha256"]
            )
        if future.label_of(feature_row) != manifest_row["label"]:
            raise RuntimeError(
                "Label mismatch between P72 manifest and features.csv: "
                + manifest_row["sha256"]
            )

    with Path(args.schema).open("r", encoding="utf-8") as handle:
        compact_schema = json.load(handle)

    _, _, canonical_original_names = (
        allratio.original_compact_matrix_named(
            cohort_rows,
            cohort_rows[:1],
            compact_by_sha,
            compact_schema,
        )
    )
    canonical_original_names = list(canonical_original_names)
    if len(canonical_original_names) != 76:
        raise RuntimeError(
            "Expected canonical source Compact76, got "
            + str(len(canonical_original_names))
        )

    return (
        manifest,
        cohort_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    )


def source_experiment(
    args,
    manifest,
    cohort_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
    source_structure,
):
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    n = len(labels)
    structure_configs = structure_configurations(source_structure)

    oof = {
        (variant, config): np.zeros(n, dtype=np.float64)
        for variant in VARIANTS
        for config in STRUCTURE_CONFIGS
    }
    per_fold_rows = []

    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        train_rows = [cohort_rows[i] for i in train_idx]
        val_rows = [cohort_rows[i] for i in val_idx]

        for variant in VARIANTS:
            compact_train, compact_val, names = compact_matrix_variant(
                variant,
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )

            for config, structure_all in structure_configs.items():
                if structure_all is None:
                    structure_train = None
                    structure_val = None
                else:
                    structure_train = structure_all[train_idx]
                    structure_val = structure_all[val_idx]

                probability = fit_config(
                    compact_train,
                    compact_val,
                    structure_train,
                    structure_val,
                    labels[train_idx],
                    seed=int(fold) - 1,
                )
                oof[(variant, config)][val_idx] = probability
                metrics = base.metrics(
                    labels[val_idx],
                    probability,
                    threshold=THRESHOLD,
                )
                per_fold_rows.append({
                    "variant": variant,
                    "config": config,
                    "fold": int(fold),
                    "train_n": int(len(train_idx)),
                    "val_n": int(len(val_idx)),
                    "compact_dim": int(len(names)),
                    "semantic_dim": int(len(semantic_indices(names))),
                    "structure_dim": 0 if structure_all is None else int(
                        structure_all.shape[1]
                    ),
                    "ACC": float(metrics["accuracy"]),
                    "BA": float(metrics["balanced_accuracy"]),
                    "AUC": float(metrics["auc"]),
                    "F1": float(metrics["f1"]),
                })

    collision_data = {}
    threshold_rows = []
    for variant in VARIANTS:
        raw, percentile, routed, rows = strict_collision_gate(
            variant,
            cohort_rows,
            labels,
            folds,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
            args.k,
            args.budget,
        )
        collision_data[variant] = {
            "raw": raw,
            "percentile": percentile,
            "routed": routed,
        }
        threshold_rows.extend(rows)

    summary_rows = []
    probability_by_key = {}
    for variant in VARIANTS:
        coarse = oof[(variant, "C")]
        for config in STRUCTURE_CONFIGS:
            probability = oof[(variant, config)]
            probability_by_key[(variant, config)] = probability
            routed_n = 0 if config == "C" else n
            summary_rows.append(
                metric_row(
                    variant,
                    config,
                    labels,
                    probability,
                    coarse,
                    routed_n,
                )
            )

        routed = collision_data[variant]["routed"]
        final_probability = coarse.copy()
        full_structure = oof[(variant, "C+Stats+PairRaw")]
        final_probability[routed] = full_structure[routed]
        probability_by_key[(variant, "Collision20")] = final_probability
        summary_rows.append(
            metric_row(
                variant,
                "Collision20",
                labels,
                final_probability,
                coarse,
                int(np.sum(routed)),
            )
        )

    paired_rows = paired_comparison_rows(
        labels,
        probability_by_key,
        tuple(STRUCTURE_CONFIGS) + ("Collision20",),
    )

    prediction_rows = []
    for i, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(labels[i]),
            "fold": int(folds[i]),
        }
        for variant in VARIANTS:
            prefix = variant
            for config in STRUCTURE_CONFIGS:
                safe_config = config.replace("+", "_plus_")
                output[f"{prefix}__{safe_config}__prob"] = float(
                    oof[(variant, config)][i]
                )
            output[f"{prefix}__collision_raw"] = float(
                collision_data[variant]["raw"][i]
            )
            output[f"{prefix}__collision_percentile"] = float(
                collision_data[variant]["percentile"][i]
            )
            output[f"{prefix}__collision20_routed"] = int(
                collision_data[variant]["routed"][i]
            )
            output[f"{prefix}__collision20_prob"] = float(
                probability_by_key[(variant, "Collision20")][i]
            )
        prediction_rows.append(output)

    write_csv(args.output / "source_summary.csv", summary_rows)
    write_csv(args.output / "source_per_fold.csv", per_fold_rows)
    write_csv(args.output / "source_oof_predictions.csv", prediction_rows)
    write_csv(args.output / "source_collision_thresholds.csv", threshold_rows)
    write_csv(args.output / "source_paired_comparisons.csv", paired_rows)

    return {
        "labels": labels,
        "folds": folds,
        "oof": oof,
        "collision": collision_data,
        "summary_rows": summary_rows,
    }


def future_experiment(
    args,
    years,
    manifest,
    cohort_rows,
    clean_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
    cache,
    source_structure,
    source_result,
):
    source_labels = source_result["labels"]
    source_structure_configs = structure_configurations(source_structure)
    summary_rows = []
    prediction_rows = []
    paired_rows = []

    for year in years:
        target_rows = [
            row
            for row in clean_rows
            if str(row.get("metadata_year", "")).strip() == str(year)
        ]
        if not target_rows:
            raise RuntimeError(f"No feature rows for future year {year}")
        target_labels = np.asarray(
            [future.label_of(row) for row in target_rows],
            dtype=np.int64,
        )
        target_structure, _ = future.reconstruct_structure(
            target_rows,
            cache,
            int(exp.ROLE_VOCABULARY_SIZE),
        )
        target_structure_configs = structure_configurations(target_structure)

        probability_by_key = {}
        collision_by_variant = {}

        for variant in VARIANTS:
            compact_train, compact_test, names = compact_matrix_variant(
                variant,
                cohort_rows,
                target_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )

            for config in STRUCTURE_CONFIGS:
                probability_by_key[(variant, config)] = fit_config(
                    compact_train,
                    compact_test,
                    source_structure_configs[config],
                    target_structure_configs[config],
                    source_labels,
                    seed=0,
                )

            sem_idx = semantic_indices(names)
            rows = e5.collision_scores(
                compact_train[:, sem_idx],
                source_labels,
                compact_test[:, sem_idx],
                args.k,
            )
            target_raw = np.asarray(
                [float(row["collision_score"]) for row in rows],
                dtype=np.float64,
            )
            source_raw = source_result["collision"][variant]["raw"]
            source_percentile = e7b.empirical_percentile(
                source_raw,
                source_raw,
            )
            target_percentile = e7b.empirical_percentile(
                source_raw,
                target_raw,
            )
            threshold, source_route_ratio = e7b.threshold_for_top_fraction(
                source_percentile,
                args.budget,
            )
            routed = target_percentile >= threshold

            coarse = probability_by_key[(variant, "C")]
            full_structure = probability_by_key[
                (variant, "C+Stats+PairRaw")
            ]
            final_probability = coarse.copy()
            final_probability[routed] = full_structure[routed]
            probability_by_key[(variant, "Collision20")] = final_probability
            collision_by_variant[variant] = {
                "raw": target_raw,
                "percentile": target_percentile,
                "routed": routed,
                "threshold": threshold,
                "source_route_ratio": source_route_ratio,
            }

            for config in STRUCTURE_CONFIGS:
                probability = probability_by_key[(variant, config)]
                routed_n = 0 if config == "C" else len(target_labels)
                output = metric_row(
                    variant,
                    config,
                    target_labels,
                    probability,
                    coarse,
                    routed_n,
                )
                output["year"] = int(year)
                output["diagnostic_only"] = True
                summary_rows.append(output)

            output = metric_row(
                variant,
                "Collision20",
                target_labels,
                final_probability,
                coarse,
                int(np.sum(routed)),
            )
            output["year"] = int(year)
            output["diagnostic_only"] = True
            output["source_collision_threshold"] = float(threshold)
            summary_rows.append(output)

        year_pairs = paired_comparison_rows(
            target_labels,
            probability_by_key,
            tuple(STRUCTURE_CONFIGS) + ("Collision20",),
        )
        for row in year_pairs:
            row["year"] = int(year)
            row["diagnostic_only"] = True
        paired_rows.extend(year_pairs)

        for i, target_row in enumerate(target_rows):
            output = {
                "year": int(year),
                "sha256": target_row["sha256"],
                "label": int(target_labels[i]),
                "diagnostic_only": True,
            }
            for variant in VARIANTS:
                for config in STRUCTURE_CONFIGS:
                    safe_config = config.replace("+", "_plus_")
                    output[f"{variant}__{safe_config}__prob"] = float(
                        probability_by_key[(variant, config)][i]
                    )
                output[f"{variant}__collision_raw"] = float(
                    collision_by_variant[variant]["raw"][i]
                )
                output[f"{variant}__collision_percentile"] = float(
                    collision_by_variant[variant]["percentile"][i]
                )
                output[f"{variant}__collision20_routed"] = int(
                    collision_by_variant[variant]["routed"][i]
                )
                output[f"{variant}__collision20_prob"] = float(
                    probability_by_key[(variant, "Collision20")][i]
                )
            prediction_rows.append(output)

    mean_rows = []
    for variant in VARIANTS:
        for method in tuple(STRUCTURE_CONFIGS) + ("Collision20",):
            group = [
                row
                for row in summary_rows
                if row["variant"] == variant
                and row["method"] == method
            ]
            if not group:
                continue
            mean_rows.append({
                "variant": variant,
                "method": method,
                "years": "|".join(str(row["year"]) for row in group),
                "mean_routed_ratio": float(np.mean([
                    row["routed_ratio"] for row in group
                ])),
                "mean_ACC": float(np.mean([row["ACC"] for row in group])),
                "mean_BA": float(np.mean([row["BA"] for row in group])),
                "min_BA": float(np.min([row["BA"] for row in group])),
                "mean_AUC": float(np.mean([row["AUC"] for row in group])),
                "mean_F1": float(np.mean([row["F1"] for row in group])),
                "mean_net_vs_C": float(np.mean([
                    row["net_vs_C"] for row in group
                ])),
                "diagnostic_only": True,
            })

    write_csv(args.output / "future_summary_diagnostic.csv", summary_rows)
    write_csv(
        args.output / "future_mean_summary_diagnostic.csv",
        mean_rows,
    )
    write_csv(args.output / "future_predictions_diagnostic.csv", prediction_rows)
    write_csv(args.output / "future_paired_comparisons_diagnostic.csv", paired_rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strict paired AllRatio70 vs no-drop Compact76 audit."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("artifacts/p1-features/feature_schema.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--p72-oof",
        type=Path,
        default=Path(
            "artifacts/p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument(
        "--skip-future",
        action="store_true",
        help="Run only the strict 2017 source experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/allratio70-vs-nodrop76-audit"
        ),
    )
    args = parser.parse_args()

    if args.k < 2:
        raise RuntimeError("--k must be at least 2")
    if not 0.0 < args.budget < 1.0:
        raise RuntimeError("--budget must be between 0 and 1")

    args.output.mkdir(parents=True, exist_ok=True)
    years = [
        int(value.strip())
        for value in args.years.split(",")
        if value.strip()
    ]

    (
        manifest,
        cohort_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    ) = load_inputs(args)

    cache = exp.read_cache(args.cache)
    required_shas = {row["sha256"] for row in cohort_rows}
    missing_cache = sorted(required_shas - set(cache))
    if missing_cache:
        raise RuntimeError(
            "Structure cache is missing source APKs: "
            + repr(missing_cache[:10])
        )

    print("Reconstructing 2017 Stats120 + PairRaw465...")
    source_structure, structure_names = future.reconstruct_structure(
        cohort_rows,
        cache,
        int(exp.ROLE_VOCABULARY_SIZE),
    )
    if source_structure.shape != (799, 585):
        raise RuntimeError(
            "Expected source structure (799, 585), got "
            + str(source_structure.shape)
        )

    source_result = source_experiment(
        args,
        manifest,
        cohort_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
        source_structure,
    )

    if not args.skip_future:
        print(
            "WARNING: 2018-2021 results are diagnostic only; "
            "these years were used in earlier feature selection."
        )
        future_experiment(
            args,
            years,
            manifest,
            cohort_rows,
            clean_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
            cache,
            source_structure,
            source_result,
        )

    audit = {
        "source_apks": 799,
        "source_year": 2017,
        "outer_folds": 5,
        "k_neighbors": int(args.k),
        "collision_budget": float(args.budget),
        "classifier_threshold": THRESHOLD,
        "variants": {
            "allratio70_current": {
                "compact_dim": 70,
                "semantic_dim": 46,
                "six_dimensions_removed": True,
            },
            "nodrop76_raw6": {
                "compact_dim": 76,
                "semantic_dim": 52,
                "six_dimensions_removed": False,
                "restored_six_representation": "original fold-standardized counts",
            },
            "allratio76_ratio52": {
                "compact_dim": 76,
                "semantic_dim": 52,
                "six_dimensions_removed": False,
                "restored_six_representation": "train-fold semantic ratios",
            },
        },
        "restored_dimensions": sorted(allratio.DROP_FEATURES),
        "structure": {
            "stats_dim": 120,
            "pair_raw_dim": 465,
            "total_dim": 585,
            "feature_name_count": len(structure_names),
        },
        "source_collision_calibration": (
            "Each outer test fold uses only inner OOF collision scores "
            "from the corresponding outer-training folds."
        ),
        "future_results": (
            "skipped"
            if args.skip_future
            else "diagnostic_only_due_to_prior_future_selection"
        ),
        "future_years": [] if args.skip_future else years,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done. Output:", args.output)


if __name__ == "__main__":
    main()
