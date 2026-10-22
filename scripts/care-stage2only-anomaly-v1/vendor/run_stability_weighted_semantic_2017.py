#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)

from run_dual_view_error_attribution import (
    compact_matrix_named,
)

from run_sensitive_upstream_structural_eval import (
    label_value,
    standardize_named,
)

from run_pilot_compact_eval import (
    fit_logistic,
    sigmoid,
    metrics,
)

from run_compact_20_150_ablation_2017 import (
    support150_names,
)


ALPHA = 0.25
THRESHOLD = 0.5

SEEDS = [0, 1, 2, 3, 4]

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

STABILITY_PATH = Path(
    "artifacts/semantic-weight-stability-2017/"
    "per_fold_relation_stability.csv"
)

OUTPUT_DIR = Path(
    "artifacts/stability-weighted-semantic-2017"
)


def read_gz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    variant,
    split_name,
    seed,
    x_train,
    y_train,
    x_test,
    y_test,
):
    w, b = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        l2=L2,
    )

    probability = sigmoid(
        x_test @ w + b
    )

    result = metrics(
        y_test,
        probability,
        threshold=THRESHOLD,
    )

    return {
        "variant": variant,
        "split_name": split_name,
        "seed": seed,
        **result,
    }


def summarize(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["variant"],
                row["split_name"],
            )
        ].append(row)

    per_split = []

    for (
        variant,
        split_name,
    ), current in sorted(
        grouped.items()
    ):
        out = {
            "variant": variant,
            "split_name": split_name,
        }

        for field in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "fp",
            "fn",
        ):
            out[
                field + "_mean"
            ] = float(
                np.mean([
                    float(row[field])
                    for row in current
                ])
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, current in (
        by_variant.items()
    ):
        ba = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in current
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(ba.mean()),

            "min_BA":
                float(ba.min()),

            "std_BA":
                float(ba.std()),

            "mean_AUC":
                float(np.mean([
                    row["auc_mean"]
                    for row in current
                ])),

            "mean_ACC":
                float(np.mean([
                    row["accuracy_mean"]
                    for row in current
                ])),

            "mean_FP":
                float(np.mean([
                    row["fp_mean"]
                    for row in current
                ])),

            "mean_FN":
                float(np.mean([
                    row["fn_mean"]
                    for row in current
                ])),
        })

    summary.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return per_split, summary


def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        SEMANTIC_PATH,
        SPLIT_PATH,
        STABILITY_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    stability_rows = read_csv(
        STABILITY_PATH
    )

    stability = {}

    for row in stability_rows:
        if int(row["usable"]) != 1:
            continue

        key = (
            row["split_name"],
            row["feature"],
        )

        stability[key] = float(
            row[
                "dominant_sign_rate"
            ]
        )

    splits = read_split_rows(
        SPLIT_PATH
    )

    print(
        "======================================"
    )
    print(
        "2017 稳定性感知语义收缩"
    )
    print(
        "======================================"
    )

    print(
        "规则：标准化后每个关系"
        "乘训练折内部方向一致率"
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    metric_rows = []
    scale_rows = []

    for split_name in sorted(
        splits
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        test_rows = (
            splits[
                split_name
            ]["test"]
        )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        (
            compact_train,
            compact_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            schema,
        )

        (
            semantic_train,
            semantic_test,
            semantic_names,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        scales = []

        for feature in semantic_names:
            key = (
                split_name,
                feature,
            )

            if key not in stability:
                raise SystemExit(
                    "缺少稳定性："
                    + str(key)
                )

            value = float(
                stability[key]
            )

            if not (
                0.0 <= value <= 1.0
            ):
                raise SystemExit(
                    f"异常稳定性："
                    f"{feature}={value}"
                )

            scales.append(
                value
            )

            scale_rows.append({
                "split_name":
                    split_name,

                "feature":
                    feature,

                "stability_scale":
                    value,

                "effective_l2_multiplier":
                    float(
                        1.0
                        / max(
                            value ** 2,
                            1e-12,
                        )
                    ),
            })

        scales = np.asarray(
            scales,
            dtype=float,
        )

        weighted_train = (
            semantic_train
            * scales[None, :]
        )

        weighted_test = (
            semantic_test
            * scales[None, :]
        )

        raw_joint_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        raw_joint_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        weighted_joint_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * weighted_train,
                ],
                axis=1,
            )
        )

        weighted_joint_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * weighted_test,
                ],
                axis=1,
            )
        )

        print()
        print(
            split_name
        )

        print(
            "train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "语义可用=",
            len(semantic_names),
        )

        print(
            "稳定性：",
            f"min={scales.min():.3f}",
            f"median={np.median(scales):.3f}",
            f"mean={scales.mean():.3f}",
            f"max={scales.max():.3f}",
        )

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus_raw150",
                raw_joint_train,
                raw_joint_test,
            ),

            (
                "compact_plus_stability150",
                weighted_joint_train,
                weighted_joint_test,
            ),
        )

        for seed in SEEDS:
            for (
                variant,
                x_train,
                x_test,
            ) in variants:

                metric_rows.append(
                    evaluate(
                        variant,
                        split_name,
                        seed,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
                )

        print(
            "完成",
            flush=True,
        )

    per_split, summary = (
        summarize(
            metric_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_split.csv",
        per_split,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT_DIR
        / "stability_scales.csv",
        scale_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017结果"
    )
    print(
        "======================================"
    )

    for row in summary:
        print(
            f"{row['variant']:32s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"std={row['std_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    lookup = {
        row["variant"]:
            row
        for row in summary
    }

    raw = lookup[
        "compact_plus_raw150"
    ]

    weighted = lookup[
        "compact_plus_stability150"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "稳定性感知150 相对 原始150"
    )
    print(
        "======================================"
    )

    print(
        "dBA =",
        f"{weighted['mean_BA'] - raw['mean_BA']:+.6f}",
    )

    print(
        "dMin =",
        f"{weighted['min_BA'] - raw['min_BA']:+.6f}",
    )

    print(
        "dStd =",
        f"{weighted['std_BA'] - raw['std_BA']:+.6f}",
    )

    print(
        "dAUC =",
        f"{weighted['mean_AUC'] - raw['mean_AUC']:+.6f}",
    )


if __name__ == "__main__":
    main()
