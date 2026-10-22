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
    TOP5,
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
    "artifacts/structure-interaction20-with150-2017"
)


METRICS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
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


def exact_old20_map(names20):
    """
    精确建立：
        role -> metric -> column

    避免 caller_ratio 被
    deep_caller_ratio 子串误匹配。
    """

    result = {}

    for role in TOP5:
        result[role] = {}

        for metric in METRICS:
            suffix = (
                f"__{role}__{metric}"
            )

            candidates = [
                name
                for name in names20
                if name.endswith(suffix)
            ]

            if len(candidates) != 1:
                raise SystemExit(
                    f"{role}/{metric}: "
                    f"候选={candidates}"
                )

            result[role][metric] = (
                candidates[0]
            )

    return result


def build_interaction_table(
    upstream_rows,
    old_map,
):
    """
    先在旧20维的原始存储值上构造乘积，
    后面再按每个训练折标准化。

    每个锚点4维：

      caller_log * mean_depth_norm
      caller_log * deep_caller_ratio
      caller_ratio * mean_depth_norm
      caller_ratio * deep_caller_ratio
    """

    feature_names = []

    for role in TOP5:
        feature_names.extend([
            (
                f"struct_inter__{role}"
                "__caller_log_x_mean_depth_norm"
            ),
            (
                f"struct_inter__{role}"
                "__caller_log_x_deep_caller_ratio"
            ),
            (
                f"struct_inter__{role}"
                "__caller_ratio_x_mean_depth_norm"
            ),
            (
                f"struct_inter__{role}"
                "__caller_ratio_x_deep_caller_ratio"
            ),
        ])

    output = {}

    for row in upstream_rows:
        sha = str(
            row["sha256"]
        ).strip().casefold()

        out = {
            "sha256": sha,
        }

        for role in TOP5:
            caller_log = float(
                row[
                    old_map[
                        role
                    ][
                        "caller_log"
                    ]
                ]
            )

            caller_ratio = float(
                row[
                    old_map[
                        role
                    ][
                        "caller_ratio"
                    ]
                ]
            )

            mean_depth = float(
                row[
                    old_map[
                        role
                    ][
                        "mean_depth_norm"
                    ]
                ]
            )

            deep_ratio = float(
                row[
                    old_map[
                        role
                    ][
                        "deep_caller_ratio"
                    ]
                ]
            )

            values = (
                caller_log
                * mean_depth,

                caller_log
                * deep_ratio,

                caller_ratio
                * mean_depth,

                caller_ratio
                * deep_ratio,
            )

            names = (
                (
                    f"struct_inter__{role}"
                    "__caller_log_x_mean_depth_norm"
                ),
                (
                    f"struct_inter__{role}"
                    "__caller_log_x_deep_caller_ratio"
                ),
                (
                    f"struct_inter__{role}"
                    "__caller_ratio_x_mean_depth_norm"
                ),
                (
                    f"struct_inter__{role}"
                    "__caller_ratio_x_deep_caller_ratio"
                ),
            )

            for name, value in zip(
                names,
                values,
            ):
                if not np.isfinite(value):
                    raise SystemExit(
                        f"{sha}: "
                        f"{name}不是有限值"
                    )

                out[name] = float(value)

        output[sha] = out

    if len(feature_names) != 20:
        raise RuntimeError(
            "交互特征不是20维"
        )

    return (
        output,
        feature_names,
    )


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
            "variant":
                variant,

            "split_name":
                split_name,

            "seed_count":
                len(current),
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
    # 读取
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

    old_map = exact_old20_map(
        names20
    )

    (
        interaction_by_sha,
        interaction_names,
    ) = build_interaction_table(
        upstream_rows,
        old_map,
    )

    print()
    print(
        "======================================"
    )
    print(
        "20维结构交互定义"
    )
    print(
        "======================================"
    )

    for name in interaction_names:
        print(
            name
        )

    # ========================================================
    # 划分与人口
    # ========================================================

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

    for name, table in (
        (
            "Compact",
            compact_by_sha,
        ),
        (
            "旧20维",
            upstream_by_sha,
        ),
        (
            "150维",
            semantic_by_sha,
        ),
        (
            "交互20维",
            interaction_by_sha,
        ),
    ):
        missing = (
            all_shas
            - set(table)
        )

        print(
            name,
            "缺失 =",
            len(missing),
        )

        if missing:
            raise SystemExit(
                name
                + "人口不完整"
            )

    print()
    print(
        "参与唯一SHA =",
        len(all_shas),
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017：结构规模×深度交互"
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

        (
            interaction_train,
            interaction_test,
            interaction_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            interaction_by_sha,
            interaction_names,
        )

        # ----------------------------------------------
        # Compact +150
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Compact +150 + 原20
        # ----------------------------------------------

        x150_old20_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
                ALPHA
                * old20_train,
            ],
            axis=1,
        )

        x150_old20_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
                ALPHA
                * old20_test,
            ],
            axis=1,
        )

        # ----------------------------------------------
        # Compact +150 + 新交互20
        # ----------------------------------------------

        x150_inter20_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
                ALPHA
                * interaction_train,
            ],
            axis=1,
        )

        x150_inter20_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
                ALPHA
                * interaction_test,
            ],
            axis=1,
        )

        # ----------------------------------------------
        # Compact +150 + 原20 + 新交互20
        # ----------------------------------------------

        x150_all_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
                ALPHA
                * old20_train,
                ALPHA
                * interaction_train,
            ],
            axis=1,
        )

        x150_all_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
                ALPHA
                * old20_test,
                ALPHA
                * interaction_test,
            ],
            axis=1,
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

            "interaction20_usable_dim":
                len(interaction_usable),

            "all_total_dim":
                x150_all_train.shape[1],
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
            "原20可用=",
            len(old20_usable),
            "交互20可用=",
            len(interaction_usable),
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
                "compact_plus150_plus_old20",
                x150_old20_train,
                x150_old20_test,
            ),

            (
                "compact_plus150_plus_inter20",
                x150_inter20_train,
                x150_inter20_test,
            ),

            (
                "compact_plus150_plus_old20_plus_inter20",
                x150_all_train,
                x150_all_test,
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
            f"{row['variant']:44s} "
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

    base150 = lookup[
        "compact_plus150"
    ]

    old20 = lookup[
        "compact_plus150_plus_old20"
    ]

    inter20 = lookup[
        "compact_plus150_plus_inter20"
    ]

    both = lookup[
        "compact_plus150_plus_old20_plus_inter20"
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
        "compact_plus150_plus_old20",
        "compact_plus150_plus_inter20",
        "compact_plus150_plus_old20_plus_inter20",
    ):
        row = lookup[
            variant
        ]

        print(
            f"{variant:44s} "
            f"dBA="
            f"{row['mean_BA'] - base150['mean_BA']:+.6f} "
            f"dMin="
            f"{row['min_BA'] - base150['min_BA']:+.6f} "
            f"dStd="
            f"{row['std_BA'] - base150['std_BA']:+.6f} "
            f"dAUC="
            f"{row['mean_AUC'] - base150['mean_AUC']:+.6f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "交互20是否补充原20"
    )
    print(
        "======================================"
    )

    print(
        "原20：",
        f"BA={old20['mean_BA']:.6f}",
        f"min={old20['min_BA']:.6f}",
        f"AUC={old20['mean_AUC']:.6f}",
    )

    print(
        "原20+交互20：",
        f"BA={both['mean_BA']:.6f}",
        f"min={both['min_BA']:.6f}",
        f"AUC={both['mean_AUC']:.6f}",
    )

    print(
        "交互20在原20基础上的增量：",
        f"dBA={both['mean_BA'] - old20['mean_BA']:+.6f}",
        f"dMin={both['min_BA'] - old20['min_BA']:+.6f}",
        f"dAUC={both['mean_AUC'] - old20['mean_AUC']:+.6f}",
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
        "Compact+150+原20："
        "BA≈0.909085 "
        "min≈0.903948 "
        "AUC≈0.952079"
    )


if __name__ == "__main__":
    main()
