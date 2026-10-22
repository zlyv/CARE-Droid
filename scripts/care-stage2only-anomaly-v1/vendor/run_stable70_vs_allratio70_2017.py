#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# 当前 Stable70 正式实现
# ============================================================

import run_frozen_anchor_stable70_count10_150_2017_2021 as stable


base = stable.base


# ============================================================
# 固定输入
# ============================================================

COMPACT_FEATURES = (
    ROOT
    / "artifacts/p1-features/features.csv"
)

COMPACT_SCHEMA = (
    ROOT
    / "artifacts/p1-features/feature_schema.json"
)

SEMANTIC_2017 = (
    ROOT
    / "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

ROLLING_SPLITS = (
    ROOT
    / "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/stable70-vs-allratio70-2017"
)


# ============================================================
# 当前 Stable70 仍然排除的6个维度
# ============================================================

DROP_FEATURES = set(
    stable.DROP_FEATURES
)


# ============================================================
# 必须从原始 Compact76 构造
# ============================================================

original_compact_matrix_named = (
    stable.original_compact_matrix_named
)


def norm_sha(value):
    return str(value).strip().casefold()


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return 0.0


def read_csv_rows(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def raw_ratio_values(
    rows,
    compact_by_sha,
    numerator_name,
    denominator_name,
):
    values = []

    for row in rows:
        sha = norm_sha(
            row["sha256"]
        )

        source = compact_by_sha[
            sha
        ]

        numerator = safe_float(
            source.get(
                numerator_name,
                0.0,
            )
        )

        denominator = safe_float(
            source.get(
                denominator_name,
                0.0,
            )
        )

        if denominator > 0.0:
            value = (
                numerator
                / denominator
            )
        else:
            value = 0.0

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# 新版本：
#
# 24维原全局特征保持不动
#
# 23类语义 * 2：
#
# node / sensitive_node_count
# edge / sensitive_edge_count
#
# 共46个语义维全部统一比例化
# ============================================================

def compact_matrix_allratio70(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):
    (
        compact_train,
        compact_test,
        compact_names,
    ) = original_compact_matrix_named(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    compact_train = (
        compact_train.copy()
    )

    compact_test = (
        compact_test.copy()
    )

    compact_names = list(
        compact_names
    )

    if len(compact_names) != 76:
        raise RuntimeError(
            "预期原始Compact为76维，实际="
            + str(len(compact_names))
        )

    semantic_features = [
        name
        for name in compact_names
        if (
            name.startswith("role_node__")
            or name.startswith("role_edge__")
        )
    ]

    if len(semantic_features) != 52:
        raise RuntimeError(
            "预期原始语义特征52维，实际="
            + str(
                len(semantic_features)
            )
        )

    ratio_features = [
        name
        for name in semantic_features
        if name not in DROP_FEATURES
    ]

    if len(ratio_features) != 46:
        raise RuntimeError(
            "预期需要统一比例化46维，实际="
            + str(
                len(ratio_features)
            )
        )

    # ========================================================
    # 所有保留语义统一转换为比例
    # ========================================================

    for numerator_name in ratio_features:

        if numerator_name.startswith(
            "role_node__"
        ):
            denominator_name = (
                "sensitive_node_count"
            )

        elif numerator_name.startswith(
            "role_edge__"
        ):
            denominator_name = (
                "sensitive_edge_count"
            )

        else:
            raise RuntimeError(
                "无法识别特征类型："
                + numerator_name
            )

        column_index = (
            compact_names.index(
                numerator_name
            )
        )

        train_ratio = (
            raw_ratio_values(
                train_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        test_ratio = (
            raw_ratio_values(
                test_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        # ====================================================
        # 标准化统计量严格只从当前训练折计算
        # ====================================================

        mean = float(
            np.mean(train_ratio)
        )

        std = float(
            np.std(train_ratio)
        )

        if (
            not np.isfinite(std)
            or std <= 1e-12
        ):
            std = 1.0

        compact_train[
            :,
            column_index,
        ] = (
            train_ratio
            - mean
        ) / std

        compact_test[
            :,
            column_index,
        ] = (
            test_ratio
            - mean
        ) / std

    # ========================================================
    # 保持和当前 Stable70 完全相同：
    # 排除相同6个特征
    # ========================================================

    found = {
        name
        for name in compact_names
        if name in DROP_FEATURES
    }

    if found != DROP_FEATURES:
        raise RuntimeError(
            "没有找到全部6个排除特征："
            + str(
                sorted(
                    DROP_FEATURES
                    - found
                )
            )
        )

    keep_idx = [
        i
        for i, name in enumerate(
            compact_names
        )
        if name not in DROP_FEATURES
    ]

    compact_train = (
        compact_train[
            :,
            keep_idx,
        ]
    )

    compact_test = (
        compact_test[
            :,
            keep_idx,
        ]
    )

    compact_names = [
        compact_names[i]
        for i in keep_idx
    ]

    if len(compact_names) != 70:
        raise RuntimeError(
            "最终预期70维，实际="
            + str(
                len(compact_names)
            )
        )

    return (
        compact_train,
        compact_test,
        compact_names,
    )


# ============================================================
# 只评价冻结的 Compact 基础分类器
# ============================================================

def evaluate_base_only(
    model,
    data,
    y_val,
    device,
):
    model.eval()

    with torch.no_grad():

        (
            _,
            base_logits,
            _,
            _,
        ) = model(
            base.tensor(
                data["compact_test"],
                device,
            ),
            base.tensor(
                data["semantic_test"],
                device,
            ),
            base.tensor(
                data["coverage_test"],
                device,
            ),
            base.tensor(
                data["presence_test"],
                device,
            ),
        )

        probability = (
            torch.sigmoid(
                base_logits
            )
            .cpu()
            .numpy()
            .ravel()
        )

    result = base.metrics(
        y_val.astype(int),
        probability,
        threshold=base.THRESHOLD,
    )

    return (
        result,
        probability,
    )


def summarize_rows(
    variant,
    rows,
):
    by_split = defaultdict(list)

    for row in rows:
        by_split[
            row["split_name"]
        ].append(
            row
        )

    split_rows = []

    for split_name in sorted(
        by_split
    ):
        part = by_split[
            split_name
        ]

        split_rows.append({
            "variant":
                variant,

            "split_name":
                split_name,

            "BA":
                float(
                    np.mean([
                        x["BA"]
                        for x in part
                    ])
                ),

            "ACC":
                float(
                    np.mean([
                        x["ACC"]
                        for x in part
                    ])
                ),

            "AUC":
                float(
                    np.mean([
                        x["AUC"]
                        for x in part
                    ])
                ),

            "F1":
                float(
                    np.mean([
                        x["F1"]
                        for x in part
                    ])
                ),

            "FP":
                float(
                    np.mean([
                        x["FP"]
                        for x in part
                    ])
                ),

            "FN":
                float(
                    np.mean([
                        x["FN"]
                        for x in part
                    ])
                ),
        })

    summary = {
        "variant":
            variant,

        "mean_BA":
            float(
                np.mean([
                    x["BA"]
                    for x in split_rows
                ])
            ),

        "min_split_BA":
            float(
                np.min([
                    x["BA"]
                    for x in split_rows
                ])
            ),

        "mean_ACC":
            float(
                np.mean([
                    x["ACC"]
                    for x in rows
                ])
            ),

        "mean_AUC":
            float(
                np.mean([
                    x["AUC"]
                    for x in rows
                ])
            ),

        "mean_F1":
            float(
                np.mean([
                    x["F1"]
                    for x in rows
                ])
            ),

        "mean_FP":
            float(
                np.mean([
                    x["FP"]
                    for x in rows
                ])
            ),

        "mean_FN":
            float(
                np.mean([
                    x["FN"]
                    for x in rows
                ])
            ),
    }

    return (
        summary,
        split_rows,
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 阈值必须固定0.5
    # ========================================================

    if abs(
        float(base.THRESHOLD)
        - 0.5
    ) > 1e-12:
        raise RuntimeError(
            "当前基础脚本阈值不是0.5："
            + str(
                base.THRESHOLD
            )
        )

    # ========================================================
    # 设备
    # ========================================================

    if args.device == "cpu":
        device = torch.device(
            "cpu"
        )

    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA不可用"
            )

        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        "device =",
        device,
        flush=True,
    )

    print(
        "threshold =",
        base.THRESHOLD,
        flush=True,
    )

    seeds = [
        0,
        1,
        2,
        3,
        4,
    ]

    print(
        "seeds =",
        seeds,
        flush=True,
    )

    # ========================================================
    # 读取2017数据
    # 不读取任何未来年份
    # ========================================================

    compact_rows = (
        read_csv_rows(
            COMPACT_FEATURES
        )
    )

    compact_schema = json.loads(
        COMPACT_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    compact_by_sha = (
        base.normalized_rows(
            compact_rows
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            SEMANTIC_2017
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
        )
    )

    rolling_parts = (
        base.read_split_rows(
            ROLLING_SPLITS
        )
    )

    rolling_splits = {
        name: parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling" in name
            and parts["train"]
            and parts["test"]
        )
    }

    if not rolling_splits:
        raise RuntimeError(
            "没有找到2017滚动划分"
        )

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
        flush=True,
    )

    # ========================================================
    # 当前正式配置
    # ========================================================

    if len(base.CONFIGS) != 1:
        raise RuntimeError(
            "预期当前只有一个配置，实际="
            + str(
                len(base.CONFIGS)
            )
        )

    config = copy.deepcopy(
        base.CONFIGS[0]
    )

    variants = [
        (
            "stable70_current",
            stable.compact_matrix_stable70,
        ),
        (
            "stable70_allratio46",
            compact_matrix_allratio70,
        ),
    ]

    all_seed_rows = []
    all_split_rows = []
    summaries = []

    # ========================================================
    # 两个版本逐个跑
    # ========================================================

    for (
        variant_name,
        matrix_function,
    ) in variants:

        print()
        print(
            "=" * 72
        )

        print(
            "variant =",
            variant_name,
            flush=True,
        )

        print(
            "=" * 72
        )

        # 当前版本只通过这里改变 Compact 表示
        base.compact_matrix_named = (
            matrix_function
        )

        variant_rows = []

        for split_name in sorted(
            rolling_splits
        ):

            train_rows = (
                rolling_splits[
                    split_name
                ]["train"]
            )

            val_rows = (
                rolling_splits[
                    split_name
                ]["test"]
            )

            data = (
                base.build_fold_data(
                    train_rows,
                    val_rows,
                    compact_by_sha,
                    compact_schema,
                    semantic_by_sha,
                    names150,
                )
            )

            if (
                data[
                    "compact_train"
                ].shape[1]
                != 70
            ):
                raise RuntimeError(
                    variant_name
                    + " 不是70维"
                )

            y_train = (
                base.labels_for(
                    train_rows,
                    compact_by_sha,
                )
            )

            y_val = (
                base.labels_for(
                    val_rows,
                    compact_by_sha,
                )
            )

            for seed in seeds:

                # ============================================
                # 完全沿用当前正式训练流程。
                #
                # fit_one 内部先得到基础分类器，
                # 随后基础分类器被冻结，
                # 再训练残差。
                #
                # 我们最后只读取 base_logits。
                # ============================================

                (
                    model,
                    best_epoch,
                    history,
                ) = base.fit_one(
                    data[
                        "compact_train"
                    ],
                    data[
                        "semantic_train"
                    ],
                    data[
                        "coverage_train"
                    ],
                    data[
                        "presence_train"
                    ],
                    y_train,
                    data[
                        "compact_test"
                    ],
                    data[
                        "semantic_test"
                    ],
                    data[
                        "coverage_test"
                    ],
                    data[
                        "presence_test"
                    ],
                    y_val,
                    config,
                    seed,
                    device,
                )

                (
                    result,
                    probability,
                ) = evaluate_base_only(
                    model,
                    data,
                    y_val,
                    device,
                )

                row = {
                    "variant":
                        variant_name,

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "BA":
                        float(
                            result[
                                "balanced_accuracy"
                            ]
                        ),

                    "ACC":
                        float(
                            result[
                                "accuracy"
                            ]
                        ),

                    "AUC":
                        float(
                            result[
                                "auc"
                            ]
                        ),

                    "F1":
                        float(
                            result[
                                "f1"
                            ]
                        ),

                    "FP":
                        int(
                            result[
                                "fp"
                            ]
                        ),

                    "FN":
                        int(
                            result[
                                "fn"
                            ]
                        ),
                }

                variant_rows.append(
                    row
                )

                all_seed_rows.append(
                    row
                )

                print(
                    f"{split_name} "
                    f"seed={seed} "
                    f"BA={row['BA']:.6f} "
                    f"ACC={row['ACC']:.6f} "
                    f"AUC={row['AUC']:.6f} "
                    f"FP={row['FP']} "
                    f"FN={row['FN']}",
                    flush=True,
                )

        (
            summary,
            split_rows,
        ) = summarize_rows(
            variant_name,
            variant_rows,
        )

        summaries.append(
            summary
        )

        all_split_rows.extend(
            split_rows
        )

        print()
        print(
            "SUMMARY",
            variant_name,
        )

        print(
            "mean_BA =",
            f"{summary['mean_BA']:.6f}",
        )

        print(
            "min_split_BA =",
            f"{summary['min_split_BA']:.6f}",
        )

        print(
            "mean_ACC =",
            f"{summary['mean_ACC']:.6f}",
        )

        print(
            "mean_AUC =",
            f"{summary['mean_AUC']:.6f}",
        )

        print(
            "mean_F1 =",
            f"{summary['mean_F1']:.6f}",
        )

        print(
            "mean_FP =",
            f"{summary['mean_FP']:.3f}",
        )

        print(
            "mean_FN =",
            f"{summary['mean_FN']:.3f}",
        )

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        args.output
        / "per_seed.csv",
        all_seed_rows,
    )

    write_csv(
        args.output
        / "per_split.csv",
        all_split_rows,
    )

    write_csv(
        args.output
        / "summary.csv",
        summaries,
    )

    # ========================================================
    # 直接打印差值
    # ========================================================

    by_name = {
        row["variant"]: row
        for row in summaries
    }

    old = by_name[
        "stable70_current"
    ]

    new = by_name[
        "stable70_allratio46"
    ]

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        "Current Stable70:"
    )

    print(
        f"  BA   = {old['mean_BA']:.6f}"
    )

    print(
        f"  ACC  = {old['mean_ACC']:.6f}"
    )

    print(
        f"  AUC  = {old['mean_AUC']:.6f}"
    )

    print(
        f"  minBA= {old['min_split_BA']:.6f}"
    )

    print()
    print(
        "All-ratio Stable70:"
    )

    print(
        f"  BA   = {new['mean_BA']:.6f}"
    )

    print(
        f"  ACC  = {new['mean_ACC']:.6f}"
    )

    print(
        f"  AUC  = {new['mean_AUC']:.6f}"
    )

    print(
        f"  minBA= {new['min_split_BA']:.6f}"
    )

    print()
    print(
        "Delta all-ratio - current:"
    )

    print(
        "  BA  =",
        f"{(new['mean_BA'] - old['mean_BA']) * 100:+.3f}",
        "percentage points",
    )

    print(
        "  ACC =",
        f"{(new['mean_ACC'] - old['mean_ACC']) * 100:+.3f}",
        "percentage points",
    )

    print(
        "  AUC =",
        f"{(new['mean_AUC'] - old['mean_AUC']) * 100:+.3f}",
        "percentage points",
    )

    print(
        "  minBA =",
        f"{(new['min_split_BA'] - old['min_split_BA']) * 100:+.3f}",
        "percentage points",
    )

    print()
    print(
        "saved to:",
        args.output,
    )


if __name__ == "__main__":
    main()
