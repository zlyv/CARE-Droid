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
    old20_names,
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
    "artifacts/20block-ablation-with150-2017"
)


COUNT_SUFFIXES = (
    "__caller_log",
    "__caller_ratio",
)

DEPTH_SUFFIXES = (
    "__mean_depth_norm",
    "__deep_caller_ratio",
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
            "seed_count": len(current),
        }

        for field in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "fp",
            "fn",
        ):
            out[field + "_mean"] = float(
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

            "mean_F1":
                float(np.mean([
                    row["f1_mean"]
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


def classify_old20_columns(
    usable_names,
):
    count_idx = []
    depth_idx = []

    count_names = []
    depth_names = []

    for idx, name in enumerate(
        usable_names
    ):
        if name.endswith(
            COUNT_SUFFIXES
        ):
            count_idx.append(idx)
            count_names.append(name)

        elif name.endswith(
            DEPTH_SUFFIXES
        ):
            depth_idx.append(idx)
            depth_names.append(name)

        else:
            raise RuntimeError(
                "旧20维出现未知列："
                + name
            )

    return (
        np.asarray(
            count_idx,
            dtype=int,
        ),
        np.asarray(
            depth_idx,
            dtype=int,
        ),
        count_names,
        depth_names,
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

    # ========================================================
    # 读取数据
    # ========================================================

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

    # ========================================================
    # 人口检查
    # ========================================================

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
        "参与唯一SHA =",
        len(all_shas),
    )

    print(
        "Compact缺失 =",
        len(missing_compact),
    )

    print(
        "20维缺失 =",
        len(missing20),
    )

    print(
        "150维缺失 =",
        len(missing150),
    )

    if missing_compact:
        raise SystemExit(
            "Compact人口不完整"
        )

    if missing20:
        raise SystemExit(
            "20维人口不完整"
        )

    if missing150:
        raise SystemExit(
            "150维人口不完整"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2017：20维内部拆分 + 150"
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

    # ========================================================
    # 三个滚动折
    # ========================================================

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

        # ----------------------------------------------------
        # Compact
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 旧20维一次性标准化
        #
        # 然后在标准化后的矩阵里拆10+10，
        # 避免分别标准化导致实验口径不同。
        # ----------------------------------------------------

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
            count_idx,
            depth_idx,
            count_names,
            depth_names,
        ) = classify_old20_columns(
            old20_usable
        )

        count_train = (
            old20_train[
                :,
                count_idx,
            ]
        )

        count_test = (
            old20_test[
                :,
                count_idx,
            ]
        )

        depth_train = (
            old20_train[
                :,
                depth_idx,
            ]
        )

        depth_test = (
            old20_test[
                :,
                depth_idx,
            ]
        )

        # ----------------------------------------------------
        # 150维
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 四个主要模型
        # ----------------------------------------------------

        x150_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        x150_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        x150_count_train = (
            np.concatenate(
                [
                    compact_train,

                    ALPHA
                    * semantic_train,

                    ALPHA
                    * count_train,
                ],
                axis=1,
            )
        )

        x150_count_test = (
            np.concatenate(
                [
                    compact_test,

                    ALPHA
                    * semantic_test,

                    ALPHA
                    * count_test,
                ],
                axis=1,
            )
        )

        x150_depth_train = (
            np.concatenate(
                [
                    compact_train,

                    ALPHA
                    * semantic_train,

                    ALPHA
                    * depth_train,
                ],
                axis=1,
            )
        )

        x150_depth_test = (
            np.concatenate(
                [
                    compact_test,

                    ALPHA
                    * semantic_test,

                    ALPHA
                    * depth_test,
                ],
                axis=1,
            )
        )

        x150_all20_train = (
            np.concatenate(
                [
                    compact_train,

                    ALPHA
                    * semantic_train,

                    ALPHA
                    * old20_train,
                ],
                axis=1,
            )
        )

        x150_all20_test = (
            np.concatenate(
                [
                    compact_test,

                    ALPHA
                    * semantic_test,

                    ALPHA
                    * old20_test,
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

            "semantic_usable_dim":
                len(semantic_usable),

            "old20_usable_dim":
                len(old20_usable),

            "count_usable_dim":
                len(count_names),

            "depth_usable_dim":
                len(depth_names),
        })

        print()
        print(
            split_name
        )

        print(
            "train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "150可用=",
            len(semantic_usable),
            "20可用=",
            len(old20_usable),
            "数量可用=",
            len(count_names),
            "深度可用=",
            len(depth_names),
        )

        print(
            "数量/覆盖列："
        )

        for name in count_names:
            print(
                " ",
                name,
            )

        print(
            "深度组织列："
        )

        for name in depth_names:
            print(
                " ",
                name,
            )

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus150",
                x150_train,
                x150_test,
            ),

            (
                "compact_plus150_plus_count10",
                x150_count_train,
                x150_count_test,
            ),

            (
                "compact_plus150_plus_depth10",
                x150_depth_train,
                x150_depth_test,
            ),

            (
                "compact_plus150_plus_all20",
                x150_all20_train,
                x150_all20_test,
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

    # ========================================================
    # 汇总
    # ========================================================

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
            f"{row['variant']:36s} "
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
        "compact_plus150"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "相对 Compact+150"
    )
    print(
        "======================================"
    )

    for variant in (
        "compact_plus150_plus_count10",
        "compact_plus150_plus_depth10",
        "compact_plus150_plus_all20",
    ):
        row = lookup[
            variant
        ]

        print(
            f"{variant:36s} "
            f"dBA="
            f"{row['mean_BA'] - base['mean_BA']:+.6f} "
            f"dMin="
            f"{row['min_BA'] - base['min_BA']:+.6f} "
            f"dStd="
            f"{row['std_BA'] - base['std_BA']:+.6f} "
            f"dAUC="
            f"{row['mean_AUC'] - base['mean_AUC']:+.6f}"
        )

    count = lookup[
        "compact_plus150_plus_count10"
    ]

    depth = lookup[
        "compact_plus150_plus_depth10"
    ]

    all20 = lookup[
        "compact_plus150_plus_all20"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "谁解释了20维的额外收益"
    )
    print(
        "======================================"
    )

    print(
        "数量10相对150：",
        f"dBA={count['mean_BA'] - base['mean_BA']:+.6f}",
        f"dAUC={count['mean_AUC'] - base['mean_AUC']:+.6f}",
    )

    print(
        "深度10相对150：",
        f"dBA={depth['mean_BA'] - base['mean_BA']:+.6f}",
        f"dAUC={depth['mean_AUC'] - base['mean_AUC']:+.6f}",
    )

    print(
        "全20相对150：",
        f"dBA={all20['mean_BA'] - base['mean_BA']:+.6f}",
        f"dAUC={all20['mean_AUC'] - base['mean_AUC']:+.6f}",
    )

    print()
    print(
        "历史参考："
    )

    print(
        "Compact+150："
        "BA≈0.906712 "
        "min≈0.890250 "
        "AUC≈0.950344"
    )

    print(
        "Compact+20+150："
        "BA≈0.909085 "
        "min≈0.903948 "
        "AUC≈0.952079"
    )


if __name__ == "__main__":
    main()
