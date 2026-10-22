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


ALPHA = 0.25
THRESHOLD = 0.5
SEEDS = [0, 1, 2, 3, 4]

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

OLD_METRICS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

UPSTREAM_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/compact-20-150-ablation-2017"
)


def read_gz(path):
    with gzip.open(
        path,
        "rt",
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

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def support150_names(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            f"support列不是155维：{len(support)}"
        )

    selected = []

    for name in support:
        parts = name.split("__")

        if len(parts) != 3:
            raise SystemExit(
                "异常语义特征名：" + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor != context:
            selected.append(name)

    if len(selected) != 150:
        raise SystemExit(
            f"最终语义不是150维：{len(selected)}"
        )

    return selected


def old20_names(header):
    """
    自动寻找旧Top5 × 4项结构特征。

    要求列名同时包含：
      角色名
      指标名

    如果发现多个候选，优先：
      1. 名字以 up 开头
      2. 名字较短
    """

    header = list(header)
    selected = []

    print()
    print(
        "======================================"
    )
    print(
        "旧20维列映射"
    )
    print(
        "======================================"
    )

    for role in TOP5:
        for metric in OLD_METRICS:

            candidates = [
                name
                for name in header
                if (
                    role.casefold()
                    in name.casefold()
                    and metric.casefold()
                    in name.casefold()
                )
            ]

            if not candidates:
                nearby = [
                    name
                    for name in header
                    if role.casefold()
                    in name.casefold()
                ]

                print()
                print(
                    "找不到：",
                    role,
                    metric,
                )
                print(
                    "该角色相关列："
                )

                for name in nearby[:50]:
                    print(
                        " ",
                        name,
                    )

                raise SystemExit(
                    "旧20维自动匹配失败"
                )

            candidates.sort(
                key=lambda name: (
                    0
                    if name.startswith("up")
                    else 1,
                    len(name),
                    name,
                )
            )

            chosen = candidates[0]

            if len(candidates) > 1:
                print(
                    "注意：多个候选",
                    role,
                    metric,
                    "=>",
                    candidates,
                )

            selected.append(
                chosen
            )

            print(
                f"{role:28s} "
                f"{metric:20s} "
                f"=> {chosen}"
            )

    if len(selected) != 20:
        raise SystemExit(
            f"旧结构不是20维：{len(selected)}"
        )

    if len(set(selected)) != 20:
        raise SystemExit(
            "旧20维存在重复列映射"
        )

    return selected


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
        "variant":
            variant,

        "split_name":
            split_name,

        "seed":
            seed,

        **result,
    }


def summarize(metric_rows):
    grouped = defaultdict(list)

    for row in metric_rows:
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
    ), rows in sorted(grouped.items()):

        out = {
            "variant":
                variant,

            "split_name":
                split_name,

            "seed_count":
                len(rows),
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
                    for row in rows
                ])
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, rows in (
        by_variant.items()
    ):
        bas = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in rows
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(
                    bas.mean()
                ),

            "min_BA":
                float(
                    bas.min()
                ),

            "std_BA":
                float(
                    bas.std()
                ),

            "mean_AUC":
                float(
                    np.mean([
                        row["auc_mean"]
                        for row in rows
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        row["accuracy_mean"]
                        for row in rows
                    ])
                ),

            "mean_F1":
                float(
                    np.mean([
                        row["f1_mean"]
                        for row in rows
                    ])
                ),

            "mean_FP":
                float(
                    np.mean([
                        row["fp_mean"]
                        for row in rows
                    ])
                ),

            "mean_FN":
                float(
                    np.mean([
                        row["fn_mean"]
                        for row in rows
                    ])
                ),
        })

    summary.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return (
        per_split,
        summary,
    )


def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        UPSTREAM_PATH,
        SEMANTIC_PATH,
        SPLIT_PATH,
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

    upstream_rows = read_gz(
        UPSTREAM_PATH
    )

    upstream_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in upstream_rows
    }

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

    names20 = old20_names(
        upstream_rows[0].keys()
    )

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    splits = read_split_rows(
        SPLIT_PATH
    )

    all_shas = set()

    for parts in splits.values():
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                all_shas.add(
                    row["sha256"]
                )

    print()
    print(
        "======================================"
    )
    print(
        "人口检查"
    )
    print(
        "======================================"
    )

    print(
        "2017滚动参与唯一SHA =",
        len(all_shas),
    )

    missing_compact = (
        all_shas
        - set(compact_by_sha)
    )

    missing20 = (
        all_shas
        - set(upstream_by_sha)
    )

    missing150 = (
        all_shas
        - set(semantic_by_sha)
    )

    print(
        "Compact缺失 =",
        len(missing_compact),
    )

    print(
        "旧20维缺失 =",
        len(missing20),
    )

    print(
        "新150维缺失 =",
        len(missing150),
    )

    if missing_compact:
        raise SystemExit(
            "Compact人口不完整"
        )

    if missing20:
        raise SystemExit(
            "旧20维人口不完整"
        )

    if missing150:
        raise SystemExit(
            "新150维人口不完整"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2017：Compact / 20 / 150 / 20+150"
    )
    print(
        "======================================"
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    print(
        "iterations =",
        ITERATIONS,
    )

    print(
        "L2 =",
        L2,
    )

    metric_rows = []
    dimension_rows = []

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
            old20_train,
            old20_test,
            old20_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            upstream_by_sha,
            names20,
        )

        (
            semantic_train,
            semantic_test,
            semantic_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        compact_plus20_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * old20_train,
                ],
                axis=1,
            )
        )

        compact_plus20_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * old20_test,
                ],
                axis=1,
            )
        )

        compact_plus150_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * semantic_train,
                ],
                axis=1,
            )
        )

        compact_plus150_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * semantic_test,
                ],
                axis=1,
            )
        )

        compact_plus20_150_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * old20_train,
                    ALPHA
                    * semantic_train,
                ],
                axis=1,
            )
        )

        compact_plus20_150_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * old20_test,
                    ALPHA
                    * semantic_test,
                ],
                axis=1,
            )
        )

        dimension_rows.append({
            "split_name":
                split_name,

            "train_n":
                len(train_rows),

            "test_n":
                len(test_rows),

            "compact_dim":
                len(compact_names),

            "old20_original_dim":
                20,

            "old20_usable_dim":
                len(old20_usable),

            "semantic_original_dim":
                150,

            "semantic_usable_dim":
                len(semantic_usable),

            "combined_dim":
                (
                    compact_plus20_150_train
                    .shape[1]
                ),
        })

        print()
        print(
            split_name
        )

        print(
            " train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "Compact=",
            len(compact_names),
            "20可用=",
            len(old20_usable),
            "150可用=",
            len(semantic_usable),
        )

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus20",
                compact_plus20_train,
                compact_plus20_test,
            ),

            (
                "compact_plus150",
                compact_plus150_train,
                compact_plus150_test,
            ),

            (
                "compact_plus20_plus150",
                compact_plus20_150_train,
                compact_plus20_150_test,
            ),
        )

        for seed in SEEDS:
            for (
                variant,
                x_train,
                x_test,
            ) in variants:

                row = evaluate(
                    variant,
                    split_name,
                    seed,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

                metric_rows.append(
                    row
                )

        print(
            " 完成",
            flush=True,
        )

    (
        per_split,
        summary,
    ) = summarize(
        metric_rows
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
        / "dimensions.csv",
        dimension_rows,
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
            f"{row['variant']:28s} "
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

    base = lookup[
        "compact_only"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "相对Compact"
    )
    print(
        "======================================"
    )

    for variant in (
        "compact_plus20",
        "compact_plus150",
        "compact_plus20_plus150",
    ):
        row = lookup[
            variant
        ]

        print(
            f"{variant:28s} "
            f"dBA="
            f"{row['mean_BA'] - base['mean_BA']:+.6f} "
            f"dMin="
            f"{row['min_BA'] - base['min_BA']:+.6f} "
            f"dAUC="
            f"{row['mean_AUC'] - base['mean_AUC']:+.6f}"
        )

    row20 = lookup[
        "compact_plus20"
    ]

    row150 = lookup[
        "compact_plus150"
    ]

    rowboth = lookup[
        "compact_plus20_plus150"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "20+150的额外贡献"
    )
    print(
        "======================================"
    )

    print(
        "相对20维："
        f"dBA="
        f"{rowboth['mean_BA'] - row20['mean_BA']:+.6f} "
        f"dAUC="
        f"{rowboth['mean_AUC'] - row20['mean_AUC']:+.6f}"
    )

    print(
        "相对150维："
        f"dBA="
        f"{rowboth['mean_BA'] - row150['mean_BA']:+.6f} "
        f"dAUC="
        f"{rowboth['mean_AUC'] - row150['mean_AUC']:+.6f}"
    )


if __name__ == "__main__":
    main()
