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
from run_clean_semantic_relation_2017_eval import predict_logistic
from run_pilot_compact_eval import metrics


def read_gz_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows(rows):
    result = {}
    for row in rows:
        sha = row["sha256"].strip().casefold()
        if sha in result:
            raise SystemExit("duplicate SHA: " + sha)
        result[sha] = row
    return result


def support150_names(header):
    names = sorted(
        name for name in header if name.startswith("semrel_support__")
    )
    selected = [
        name for name in names
        if name.split("__")[1] != name.split("__")[2]
    ]
    if len(names) != 155 or len(selected) != 150:
        raise SystemExit("unexpected support150 definition")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--compact-schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument(
        "--semantic-features",
        type=Path,
        default=Path("artifacts/clean-local-context-2017/semantic_relation_features.csv.gz"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("artifacts/p9-2017-rolling-splits/split_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/2017-alpha-threshold-sweep"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[round(x, 2) for x in np.arange(0.30, 0.701, 0.01)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

    compact_by_sha = normalized_rows(read_csv_rows(args.compact_features))
    compact_schema = json.loads(args.compact_schema.read_text(encoding="utf-8"))
    semantic_rows = read_gz_csv(args.semantic_features)
    names150 = support150_names(semantic_rows[0].keys())
    semantic_by_sha = normalized_rows(semantic_rows)
    splits = read_split_rows(args.splits)

    selected = {}
    for split_name, parts in splits.items():
        if not parts["train"] or not parts["test"]:
            continue
        if "rolling" not in split_name:
            continue
        selected[split_name] = parts
    if not selected:
        raise SystemExit("no rolling 2017 splits found")

    for split_name, parts in selected.items():
        for row in parts["train"] + parts["test"]:
            sha = row["sha256"]
            year = str(compact_by_sha[sha].get("metadata_year", "")).strip()
            if year and year != "2017":
                raise SystemExit(f"non-2017 sample in rolling split: {sha}")

    print("======================================")
    print("2017 rolling alpha + threshold sweep")
    print("======================================")
    print("splits =", sorted(selected))
    print("alphas =", args.alphas)
    print("thresholds =", len(args.thresholds))

    # Store metrics by (alpha, threshold, split, seed).
    metric_store = defaultdict(list)

    for alpha in args.alphas:
        print("alpha =", alpha, flush=True)
        for split_name in sorted(selected):
            train_rows = selected[split_name]["train"]
            test_rows = selected[split_name]["test"]
            y_train = np.asarray([
                label_value(compact_by_sha[row["sha256"]].get("metadata_label", ""))
                for row in train_rows
            ], dtype=np.int64)
            y_test = np.asarray([
                label_value(compact_by_sha[row["sha256"]].get("metadata_label", ""))
                for row in test_rows
            ], dtype=np.int64)
            compact_train, compact_test, _ = compact_matrix_named(
                train_rows, test_rows, compact_by_sha, compact_schema
            )
            semantic_train, semantic_test, _ = standardize_named(
                train_rows, test_rows, semantic_by_sha, names150
            )
            x_train = np.concatenate([compact_train, alpha * semantic_train], axis=1)
            x_test = np.concatenate([compact_test, alpha * semantic_test], axis=1)

            for seed in args.seeds:
                probability = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                for threshold in args.thresholds:
                    result = metrics(y_test, probability, threshold=threshold)
                    metric_store[(alpha, threshold, split_name, seed)].append(result)

    rows = []
    for alpha in args.alphas:
        for threshold in args.thresholds:
            split_ba = []
            split_auc = []
            split_acc = []
            for split_name in selected:
                current = []
                for seed in args.seeds:
                    current.extend(metric_store[(alpha, threshold, split_name, seed)])
                split_ba.append(float(np.mean([row["balanced_accuracy"] for row in current])))
                split_auc.append(float(np.mean([row["auc"] for row in current])))
                split_acc.append(float(np.mean([row["accuracy"] for row in current])))
            rows.append({
                "alpha": alpha,
                "threshold": threshold,
                "mean_BA": float(np.mean(split_ba)),
                "min_split_BA": float(np.min(split_ba)),
                "mean_AUC": float(np.mean(split_auc)),
                "mean_ACC": float(np.mean(split_acc)),
            })

    rows.sort(key=lambda row: (row["mean_BA"], row["min_split_BA"], row["mean_AUC"]), reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ranking.csv", rows)

    print()
    print("===== 2017 source-only ranking by mean BA =====")
    for row in rows[:30]:
        print(
            f"alpha={float(row['alpha']):g} "
            f"threshold={float(row['threshold']):.2f} "
            f"BA={float(row['mean_BA']):.6f} "
            f"minBA={float(row['min_split_BA']):.6f} "
            f"AUC={float(row['mean_AUC']):.6f}"
        )
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
