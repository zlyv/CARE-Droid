#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
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
from run_frozen_semantic_2018_2021_eval import (
    read_gz_csv,
    write_csv,
    support150_names,
)


YEARS = (2018, 2019, 2020, 2021)
EXPECTED_TEST_COUNTS = {2018: 789, 2019: 399, 2020: 397, 2021: 400}
THRESHOLD = 0.5


def normalized_rows(rows):
    result = {}
    for row in rows:
        sha = row["sha256"].strip().casefold()
        if sha in result:
            raise SystemExit("duplicate SHA: " + sha)
        result[sha] = row
    return result


def fixed_splits(splits):
    selected = {}
    split_to_year = {}
    train_shas = set()
    test_shas = set()
    for split_name, parts in splits.items():
        year = None
        for candidate in YEARS:
            if split_name == f"train_fixed_2017__test_{candidate}":
                year = candidate
                break
        if year is None:
            continue
        if len(parts["train"]) != 799:
            raise SystemExit(f"{split_name}: train count is not 799")
        if len(parts["test"]) != EXPECTED_TEST_COUNTS[year]:
            raise SystemExit(f"{split_name}: test count is unexpected")
        selected[split_name] = parts
        split_to_year[split_name] = year
        train_shas.update(row["sha256"] for row in parts["train"])
        test_shas.update(row["sha256"] for row in parts["test"])
    if set(split_to_year.values()) != set(YEARS):
        raise SystemExit("missing fixed future split")
    if len(train_shas) != 799 or len(test_shas) != 1985:
        raise SystemExit("unexpected unique population size")
    return selected, split_to_year


def average_predictions(prediction_rows):
    grouped = defaultdict(list)
    labels = {}
    for row in prediction_rows:
        key = (row["split_name"], row["variant"], row["sha256"])
        grouped[key].append(float(row["probability"]))
        labels[(row["split_name"], row["sha256"])] = int(row["label"])
    return {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }, labels


def correction_rows(probabilities, labels, split_to_year, baseline_variant):
    rows = []
    variants = sorted({key[1] for key in probabilities if key[1] != baseline_variant})
    for variant in variants:
        all_values = []
        for split_name, year in sorted(split_to_year.items(), key=lambda x: x[1]):
            shas = sorted(
                sha
                for current_split, sha in labels
                if current_split == split_name
            )
            y = np.asarray([labels[(split_name, sha)] for sha in shas], dtype=int)
            old = np.asarray(
                [probabilities[(split_name, baseline_variant, sha)] for sha in shas]
            )
            new = np.asarray(
                [probabilities[(split_name, variant, sha)] for sha in shas]
            )
            old_ok = (old >= THRESHOLD) == y
            new_ok = (new >= THRESHOLD) == y
            corrected = int(np.sum((~old_ok) & new_ok))
            degraded = int(np.sum(old_ok & (~new_ok)))
            rows.append({
                "year": year,
                "alpha_variant": variant,
                "corrected": corrected,
                "degraded": degraded,
                "net_corrected": corrected - degraded,
            })
            all_values.append((corrected, degraded))
        rows.append({
            "year": "all",
            "alpha_variant": variant,
            "corrected": sum(item[0] for item in all_values),
            "degraded": sum(item[1] for item in all_values),
            "net_corrected": sum(item[0] - item[1] for item in all_values),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--compact-schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument(
        "--semantic-features",
        type=Path,
        default=Path("artifacts/clean-local-context-future/semantic_relation_features.csv.gz"),
    )
    parser.add_argument(
        "--semantic-2017",
        type=Path,
        default=Path("artifacts/clean-local-context-2017/semantic_relation_features.csv.gz"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("artifacts/p1-splits-fixed-2017-test-2018-2021/split_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/future-alpha-oracle-sweep"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

    compact_by_sha = normalized_rows(read_csv_rows(args.compact_features))
    compact_schema = json.loads(args.compact_schema.read_text(encoding="utf-8"))
    semantic_2017 = read_gz_csv(args.semantic_2017)
    semantic_future = read_gz_csv(args.semantic_features)
    names_2017 = support150_names(semantic_2017[0].keys())
    names_future = support150_names(semantic_future[0].keys())
    if names_2017 != names_future:
        raise SystemExit("2017/future support150 columns differ")
    semantic_by_sha = normalized_rows(semantic_2017 + semantic_future)
    splits, split_to_year = fixed_splits(read_split_rows(args.splits))

    predictions = []
    per_year = []

    print("======================================")
    print("Future alpha oracle sweep")
    print("======================================")
    print("alphas =", args.alphas)
    print("This is a test-set oracle/development diagnostic.")

    for alpha in args.alphas:
        variant = f"alpha_{alpha:g}"
        print()
        print("alpha =", alpha, flush=True)
        for split_name in sorted(splits, key=lambda name: split_to_year[name]):
            year = split_to_year[split_name]
            train_rows = splits[split_name]["train"]
            test_rows = splits[split_name]["test"]
            y_train = np.asarray([
                label_value(compact_by_sha[row["sha256"]].get("metadata_label", ""))
                for row in train_rows
            ], dtype=np.int64)
            y_test = np.asarray([
                label_value(compact_by_sha[row["sha256"]].get("metadata_label", ""))
                for row in test_rows
            ], dtype=np.int64)
            compact_train, compact_test, compact_names = compact_matrix_named(
                train_rows, test_rows, compact_by_sha, compact_schema
            )
            semantic_train, semantic_test, semantic_usable = standardize_named(
                train_rows, test_rows, semantic_by_sha, names_2017
            )
            x_train = np.concatenate([compact_train, alpha * semantic_train], axis=1)
            x_test = np.concatenate([compact_test, alpha * semantic_test], axis=1)
            result = evaluate_matrix(
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
                predictions,
            )
            per_year.append({
                "alpha": alpha,
                "year": year,
                "split_name": split_name,
                "compact_dim": len(compact_names),
                "semantic_usable_dim": len(semantic_usable),
                "BA": result["balanced_accuracy_mean"],
                "AUC": result["auc_mean"],
                "ACC": result["accuracy_mean"],
                "F1": result["f1_mean"],
                "FP": result["fp_mean"],
                "FN": result["fn_mean"],
            })
            print(
                f"{year}: BA={result['balanced_accuracy_mean']:.6f} "
                f"AUC={result['auc_mean']:.6f} "
                f"ACC={result['accuracy_mean']:.6f}",
                flush=True,
            )

    future_mean = []
    for alpha in args.alphas:
        current = [row for row in per_year if float(row["alpha"]) == alpha]
        future_mean.append({
            "alpha": alpha,
            "BA": float(np.mean([row["BA"] for row in current])),
            "AUC": float(np.mean([row["AUC"] for row in current])),
            "ACC": float(np.mean([row["ACC"] for row in current])),
            "F1": float(np.mean([row["F1"] for row in current])),
            "FP": float(np.mean([row["FP"] for row in current])),
            "FN": float(np.mean([row["FN"] for row in current])),
        })

    probabilities, labels = average_predictions(predictions)
    baseline_variant = "alpha_0"
    correction = correction_rows(probabilities, labels, split_to_year, baseline_variant)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_year.csv", per_year)
    write_csv(args.output / "future_mean.csv", future_mean)
    write_csv(args.output / "predictions.csv", predictions)
    write_csv(args.output / "corrections_vs_alpha0.csv", correction)

    ranked = sorted(future_mean, key=lambda row: row["BA"], reverse=True)
    print()
    print("======================================")
    print("Future alpha ranking by four-year mean BA")
    print("======================================")
    for row in ranked:
        print(
            f"alpha={float(row['alpha']):g} "
            f"BA={float(row['BA']):.6f} "
            f"AUC={float(row['AUC']):.6f} "
            f"ACC={float(row['ACC']):.6f}"
        )
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
