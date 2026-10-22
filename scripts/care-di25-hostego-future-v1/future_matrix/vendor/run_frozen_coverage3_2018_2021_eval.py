#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows
from run_dual_view_error_attribution import compact_matrix_named
from run_sensitive_upstream_structural_eval import label_value, standardize_named
from run_clean_semantic_relation_2017_eval import evaluate_matrix
from run_pilot_compact_eval import metrics
from run_semantic_coverage_shortcut_control_2017 import (
    build_coverage_table,
    standardize_control,
)
from run_frozen_semantic_2018_2021_eval import (
    read_gz_csv,
    write_csv,
    support150_names,
    per_seed_metrics,
    summarize_by_year,
)


ALPHA = 0.25
THRESHOLD = 0.5
YEARS = (2018, 2019, 2020, 2021)
EXPECTED_TEST_COUNTS = {
    2018: 789,
    2019: 399,
    2020: 397,
    2021: 400,
}
VARIANTS = (
    "compact_only",
    "compact_plus_coverage3",
    "compact_plus_support150",
)
COMPARISONS = (
    ("compact_only", "compact_plus_coverage3"),
    ("compact_only", "compact_plus_support150"),
    ("compact_plus_coverage3", "compact_plus_support150"),
)
COVERAGE_NAMES = (
    "coverage_any_relation",
    "coverage_nonzero_pair_count",
    "coverage_active_anchor_count",
)


def normalized_rows_by_sha(rows):
    result = {}
    for row in rows:
        sha = str(row["sha256"]).strip().casefold()
        if sha in result:
            raise SystemExit("duplicate SHA: " + sha)
        result[sha] = row
    return result


def select_fixed_splits(splits):
    selected = {}
    split_to_year = {}
    all_train = set()
    all_test = set()

    for split_name, parts in splits.items():
        year = None
        for candidate in YEARS:
            if split_name == f"train_fixed_2017__test_{candidate}":
                year = candidate
                break
        if year is None:
            continue

        train_rows = parts["train"]
        test_rows = parts["test"]
        if len(train_rows) != 799:
            raise SystemExit(
                f"{split_name}: train count {len(train_rows)} != 799"
            )
        expected = EXPECTED_TEST_COUNTS[year]
        if len(test_rows) != expected:
            raise SystemExit(
                f"{split_name}: test count {len(test_rows)} != {expected}"
            )

        selected[split_name] = parts
        split_to_year[split_name] = year
        all_train.update(row["sha256"] for row in train_rows)
        all_test.update(row["sha256"] for row in test_rows)

    if set(split_to_year.values()) != set(YEARS):
        raise SystemExit("fixed 2018-2021 splits are incomplete")
    if len(all_train) != 799:
        raise SystemExit(f"unique 2017 train SHA count is {len(all_train)}, not 799")
    if len(all_test) != 1985:
        raise SystemExit(f"unique future test SHA count is {len(all_test)}, not 1985")
    return selected, split_to_year, all_train, all_test


def average_prediction_index(prediction_rows):
    grouped = defaultdict(list)
    labels = {}
    for row in prediction_rows:
        split_name = row["split_name"]
        variant = row["variant"]
        sha = row["sha256"]
        grouped[(split_name, variant, sha)].append(float(row["probability"]))
        labels[(split_name, sha)] = int(row["label"])
    probabilities = {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }
    return probabilities, labels


def exact_mcnemar_p(corrected, degraded):
    discordant = int(corrected) + int(degraded)
    if discordant == 0:
        return 1.0
    tail = min(int(corrected), int(degraded))
    numerator = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2.0 ** discordant))


def comparison_arrays(
    probabilities,
    labels,
    split_name,
    old_variant,
    new_variant,
):
    shas = sorted(
        sha
        for current_split, sha in labels
        if current_split == split_name
    )
    y = np.asarray([labels[(split_name, sha)] for sha in shas], dtype=int)
    old = np.asarray(
        [probabilities[(split_name, old_variant, sha)] for sha in shas],
        dtype=float,
    )
    new = np.asarray(
        [probabilities[(split_name, new_variant, sha)] for sha in shas],
        dtype=float,
    )
    return shas, y, old, new


def pairwise_corrections(probabilities, labels, split_to_year):
    rows = []
    for old_variant, new_variant in COMPARISONS:
        accumulated = []
        for split_name, year in sorted(
            split_to_year.items(), key=lambda item: item[1]
        ):
            _, y, old, new = comparison_arrays(
                probabilities,
                labels,
                split_name,
                old_variant,
                new_variant,
            )
            accumulated.append((y, old, new))
            rows.append(
                correction_row(year, old_variant, new_variant, y, old, new)
            )

        y = np.concatenate([item[0] for item in accumulated])
        old = np.concatenate([item[1] for item in accumulated])
        new = np.concatenate([item[2] for item in accumulated])
        rows.append(
            correction_row("all", old_variant, new_variant, y, old, new)
        )
    return rows


def correction_row(year, old_variant, new_variant, y, old, new):
    old_pred = (old >= THRESHOLD).astype(int)
    new_pred = (new >= THRESHOLD).astype(int)
    old_ok = old_pred == y
    new_ok = new_pred == y
    corrected_mask = (~old_ok) & new_ok
    degraded_mask = old_ok & (~new_ok)
    corrected = int(np.sum(corrected_mask))
    degraded = int(np.sum(degraded_mask))
    signed_movement = (2 * y - 1) * (new - old)
    return {
        "year": year,
        "old_variant": old_variant,
        "new_variant": new_variant,
        "n": len(y),
        "corrected": corrected,
        "degraded": degraded,
        "net_corrected": corrected - degraded,
        "benign_corrected": int(np.sum(corrected_mask & (y == 0))),
        "malware_corrected": int(np.sum(corrected_mask & (y == 1))),
        "toward_truth_rate": float(np.mean(signed_movement > 1e-15)),
        "away_from_truth_rate": float(np.mean(signed_movement < -1e-15)),
        "mcnemar_exact_p": exact_mcnemar_p(corrected, degraded),
    }


def pairwise_metric_deltas(year_rows):
    index = {
        (int(row["year"]), row["variant"]): row
        for row in year_rows
    }
    rows = []
    for old_variant, new_variant in COMPARISONS:
        per_year = []
        for year in YEARS:
            old = index[(year, old_variant)]
            new = index[(year, new_variant)]
            row = {
                "year": year,
                "old_variant": old_variant,
                "new_variant": new_variant,
            }
            for field in ("BA", "AUC", "ACC", "F1", "FP", "FN"):
                row["delta_" + field] = float(new[field]) - float(old[field])
            rows.append(row)
            per_year.append(row)

        mean_row = {
            "year": "four_year_mean",
            "old_variant": old_variant,
            "new_variant": new_variant,
        }
        for field in ("BA", "AUC", "ACC", "F1", "FP", "FN"):
            mean_row["delta_" + field] = float(
                np.mean([row["delta_" + field] for row in per_year])
            )
        rows.append(mean_row)
    return rows


def zero_group_audit(
    probabilities,
    labels,
    split_to_year,
    raw_zero_map,
    usable_zero_by_year,
):
    population_rows = []
    metric_rows = []
    for zero_definition in ("raw150", "usable_train"):
        for split_name, year in sorted(
            split_to_year.items(), key=lambda item: item[1]
        ):
            shas = sorted(
                sha
                for current_split, sha in labels
                if current_split == split_name
            )
            for group_name, desired_zero in (("zero", True), ("nonzero", False)):
                selected = []
                for sha in shas:
                    is_zero = (
                        raw_zero_map[sha]
                        if zero_definition == "raw150"
                        else usable_zero_by_year[(year, sha)]
                    )
                    if is_zero == desired_zero:
                        selected.append(sha)

                y = np.asarray([labels[(split_name, sha)] for sha in selected], dtype=int)
                population_rows.append({
                    "zero_definition": zero_definition,
                    "year": year,
                    "group": group_name,
                    "n": len(selected),
                    "benign_n": int(np.sum(y == 0)),
                    "malware_n": int(np.sum(y == 1)),
                })

                for variant in VARIANTS:
                    p = np.asarray(
                        [probabilities[(split_name, variant, sha)] for sha in selected],
                        dtype=float,
                    )
                    result = metrics(y, p, threshold=THRESHOLD)
                    metric_rows.append({
                        "zero_definition": zero_definition,
                        "year": year,
                        "group": group_name,
                        "variant": variant,
                        "n": len(selected),
                        "BA": result["balanced_accuracy"],
                        "AUC": result["auc"],
                        "ACC": result["accuracy"],
                        "F1": result["f1"],
                        "FP": result["fp"],
                        "FN": result["fn"],
                    })
    return population_rows, metric_rows


def stratified_sample_indices(y, rng):
    pieces = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) == 0:
            raise SystemExit("bootstrap population is missing a class")
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)


def metric_delta(y, old, new):
    old_m = metrics(y, old, threshold=THRESHOLD)
    new_m = metrics(y, new, threshold=THRESHOLD)
    return {
        "BA": float(new_m["balanced_accuracy"] - old_m["balanced_accuracy"]),
        "AUC": float(new_m["auc"] - old_m["auc"]),
        "ACC": float(new_m["accuracy"] - old_m["accuracy"]),
    }


def bootstrap_intervals(
    probabilities,
    labels,
    split_to_year,
    repeats,
    seed,
):
    rng = np.random.default_rng(seed)
    rows = []
    ordered_splits = sorted(split_to_year, key=lambda name: split_to_year[name])

    for old_variant, new_variant in COMPARISONS:
        arrays = {}
        for split_name in ordered_splits:
            _, y, old, new = comparison_arrays(
                probabilities,
                labels,
                split_name,
                old_variant,
                new_variant,
            )
            arrays[split_name] = (y, old, new)

        for split_name in ordered_splits:
            year = split_to_year[split_name]
            y, old, new = arrays[split_name]
            point = metric_delta(y, old, new)
            draws = {field: [] for field in point}
            for _ in range(repeats):
                indices = stratified_sample_indices(y, rng)
                current = metric_delta(y[indices], old[indices], new[indices])
                for field, value in current.items():
                    draws[field].append(value)
            for field, value in point.items():
                low, high = np.quantile(draws[field], [0.025, 0.975])
                rows.append({
                    "scope": year,
                    "old_variant": old_variant,
                    "new_variant": new_variant,
                    "metric": field,
                    "point_delta": value,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_repeats": repeats,
                })

        point_by_year = [
            metric_delta(*arrays[split_name])
            for split_name in ordered_splits
        ]
        mean_point = {
            field: float(np.mean([row[field] for row in point_by_year]))
            for field in ("BA", "AUC", "ACC")
        }
        draws = {field: [] for field in mean_point}
        for _ in range(repeats):
            current_years = []
            for split_name in ordered_splits:
                y, old, new = arrays[split_name]
                indices = stratified_sample_indices(y, rng)
                current_years.append(
                    metric_delta(y[indices], old[indices], new[indices])
                )
            for field in draws:
                draws[field].append(
                    float(np.mean([row[field] for row in current_years]))
                )
        for field, value in mean_point.items():
            low, high = np.quantile(draws[field], [0.025, 0.975])
            rows.append({
                "scope": "four_year_mean",
                "old_variant": old_variant,
                "new_variant": new_variant,
                "metric": field,
                "point_delta": value,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_repeats": repeats,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path("artifacts/p1-features/feature_schema.json"),
    )
    parser.add_argument(
        "--semantic-2017",
        type=Path,
        default=Path(
            "artifacts/clean-local-context-2017/semantic_relation_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--semantic-future",
        type=Path,
        default=Path(
            "artifacts/clean-local-context-future/semantic_relation_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "artifacts/p1-splits-fixed-2017-test-2018-2021/split_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/frozen-coverage3-2018-2021"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    args = parser.parse_args()

    compact_rows = read_csv_rows(args.compact_features)
    compact_by_sha = normalized_rows_by_sha(compact_rows)
    compact_schema = json.loads(args.compact_schema.read_text(encoding="utf-8"))

    semantic_2017_rows = read_gz_csv(args.semantic_2017)
    semantic_future_rows = read_gz_csv(args.semantic_future)
    if not semantic_2017_rows or not semantic_future_rows:
        raise SystemExit("semantic feature file is empty")

    names2017 = support150_names(semantic_2017_rows[0].keys())
    names_future = support150_names(semantic_future_rows[0].keys())
    if names2017 != names_future:
        raise SystemExit("2017 and future support150 definitions differ")
    names150 = names2017

    semantic_rows = semantic_2017_rows + semantic_future_rows
    semantic_by_sha = normalized_rows_by_sha(semantic_rows)
    coverage_by_sha = build_coverage_table(semantic_rows, names150)

    raw_zero_map = {
        sha: all(abs(float(row[name])) <= 1e-15 for name in names150)
        for sha, row in semantic_by_sha.items()
    }

    splits = read_split_rows(args.splits)
    selected_splits, split_to_year, all_train, all_test = select_fixed_splits(splits)
    needed = all_train | all_test
    if needed - set(compact_by_sha):
        raise SystemExit(f"Compact is missing {len(needed - set(compact_by_sha))} samples")
    if needed - set(semantic_by_sha):
        raise SystemExit(f"semantic table is missing {len(needed - set(semantic_by_sha))} samples")
    if needed - set(coverage_by_sha):
        raise SystemExit(f"coverage table is missing {len(needed - set(coverage_by_sha))} samples")

    print("======================================")
    print("Frozen coverage3 control: 2017 -> 2018-2021")
    print("======================================")
    print("2017 unique train SHA =", len(all_train))
    print("future unique test SHA =", len(all_test))
    print("support representation dim =", len(names150))
    print("alpha =", ALPHA)
    print("iterations =", args.iterations)
    print("learning_rate =", args.learning_rate)
    print("l2 =", args.l2)
    print("seeds =", args.seeds)

    prediction_rows = []
    dimension_rows = []
    usable_zero_by_year = {}
    common_usable_names = None

    for split_name in sorted(selected_splits, key=lambda name: split_to_year[name]):
        year = split_to_year[split_name]
        train_rows = selected_splits[split_name]["train"]
        test_rows = selected_splits[split_name]["test"]
        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get("metadata_label", "")
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )
        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get("metadata_label", "")
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
        )
        semantic_train, semantic_test, semantic_usable = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )
        coverage_train, coverage_test, coverage_usable = standardize_control(
            train_rows,
            test_rows,
            coverage_by_sha,
            list(COVERAGE_NAMES),
        )

        if common_usable_names is None:
            common_usable_names = list(semantic_usable)
        elif common_usable_names != list(semantic_usable):
            raise SystemExit("support150 usable columns differ across fixed splits")

        for row in test_rows:
            sha = row["sha256"]
            usable_zero_by_year[(year, sha)] = all(
                abs(float(semantic_by_sha[sha][name])) <= 1e-15
                for name in semantic_usable
            )

        matrices = {
            "compact_only": (compact_train, compact_test),
            "compact_plus_coverage3": (
                np.concatenate([compact_train, ALPHA * coverage_train], axis=1),
                np.concatenate([compact_test, ALPHA * coverage_test], axis=1),
            ),
            "compact_plus_support150": (
                np.concatenate([compact_train, ALPHA * semantic_train], axis=1),
                np.concatenate([compact_test, ALPHA * semantic_test], axis=1),
            ),
        }

        for variant in VARIANTS:
            x_train, x_test = matrices[variant]
            evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                y_train,
                y_test,
                x_train,
                x_test,
                variant,
                args.seeds,
                args.iterations,
                args.learning_rate,
                args.l2,
                prediction_rows,
            )

        raw_zero_n = sum(raw_zero_map[row["sha256"]] for row in test_rows)
        usable_zero_n = sum(
            usable_zero_by_year[(year, row["sha256"])] for row in test_rows
        )
        dimension_rows.append({
            "year": year,
            "split_name": split_name,
            "train_n": len(train_rows),
            "test_n": len(test_rows),
            "compact_dim": len(compact_names),
            "support_original_dim": len(names150),
            "support_usable_dim": len(semantic_usable),
            "coverage_original_dim": len(COVERAGE_NAMES),
            "coverage_usable_dim": len(coverage_usable),
            "raw150_zero_n": raw_zero_n,
            "usable_support_zero_n": usable_zero_n,
        })
        print(
            f"{year} complete: train={len(train_rows)} test={len(test_rows)} "
            f"compact={len(compact_names)} support_usable={len(semantic_usable)} "
            f"coverage_usable={len(coverage_usable)} "
            f"raw_zero={raw_zero_n} usable_zero={usable_zero_n}",
            flush=True,
        )

    seed_rows = per_seed_metrics(prediction_rows, split_to_year)
    year_rows = summarize_by_year(seed_rows)
    delta_rows = pairwise_metric_deltas(year_rows)
    probabilities, labels = average_prediction_index(prediction_rows)
    correction_rows = pairwise_corrections(probabilities, labels, split_to_year)
    population_rows, zero_metric_rows = zero_group_audit(
        probabilities,
        labels,
        split_to_year,
        raw_zero_map,
        usable_zero_by_year,
    )
    bootstrap_rows = bootstrap_intervals(
        probabilities,
        labels,
        split_to_year,
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "predictions.csv", prediction_rows)
    write_csv(args.output / "per_seed_metrics.csv", seed_rows)
    write_csv(args.output / "per_year_summary.csv", year_rows)
    write_csv(args.output / "pairwise_delta.csv", delta_rows)
    write_csv(args.output / "pairwise_correction.csv", correction_rows)
    write_csv(args.output / "dimensions.csv", dimension_rows)
    write_csv(args.output / "zero_population.csv", population_rows)
    write_csv(args.output / "zero_group_metrics.csv", zero_metric_rows)
    write_csv(args.output / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(
        args.output / "usable_support_names.csv",
        [{"index": index, "feature": name} for index, name in enumerate(common_usable_names)],
    )

    print()
    print("======================================")
    print("Per-year results")
    print("======================================")
    for year in YEARS:
        print(f"---------- {year} ----------")
        for row in year_rows:
            if int(row["year"]) != year:
                continue
            print(
                f"{row['variant']:<28} "
                f"BA={float(row['BA']):.6f} "
                f"AUC={float(row['AUC']):.6f} "
                f"ACC={float(row['ACC']):.6f} "
                f"FP={float(row['FP']):.2f} "
                f"FN={float(row['FN']):.2f}"
            )

    print()
    print("======================================")
    print("Four-year mean pairwise deltas")
    print("======================================")
    for row in delta_rows:
        if row["year"] != "four_year_mean":
            continue
        print(
            f"{row['new_variant']} vs {row['old_variant']}: "
            f"dBA={float(row['delta_BA']):+.6f} "
            f"dAUC={float(row['delta_AUC']):+.6f} "
            f"dACC={float(row['delta_ACC']):+.6f}"
        )

    print()
    print("======================================")
    print("All-year paired corrections")
    print("======================================")
    for row in correction_rows:
        if row["year"] != "all":
            continue
        print(
            f"{row['new_variant']} vs {row['old_variant']}: "
            f"corrected={row['corrected']} degraded={row['degraded']} "
            f"net={int(row['net_corrected']):+d} "
            f"McNemar-p={float(row['mcnemar_exact_p']):.6g}"
        )

    print()
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
