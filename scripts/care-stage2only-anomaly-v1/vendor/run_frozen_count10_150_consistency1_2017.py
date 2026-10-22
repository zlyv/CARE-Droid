#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)


# ============================================================
# 先导入原始 Frozen+150
#
# 必须先保存原始build_fold_data，
# 因为后面导入20+150脚本时会修改base中的实现。
# ============================================================

import run_frozen_anchor_residual_2017_2021 as base


ORIGINAL_BUILD_FOLD_DATA = (
    base.build_fold_data
)

ORIGINAL_NET = (
    base.FrozenAnchorResidualNet
)

ORIGINAL_CONFIGS = copy.deepcopy(
    base.CONFIGS
)


# ============================================================
# 导入已经验证过的结构+150实现
# ============================================================

import run_frozen_anchor_residual_20_150_2017_2021 as aug


OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-count10-150-consistency1-2017"
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

EPS = 1e-12


# ============================================================
# 固定Top-5与数量10
# ============================================================

ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

COUNT_METRICS = (
    "caller_log",
    "caller_ratio",
)


COUNT10_NAMES = [
    f"up_role__{role}__{metric}"
    for role in ROLES
    for metric in COUNT_METRICS
]


if len(COUNT10_NAMES) != 10:
    raise RuntimeError(
        "数量结构块不是10维"
    )


CONSISTENCY_NAME = (
    "source_relation_consistency"
)


# ============================================================
# 工具
# ============================================================

def normalize_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

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

        writer.writerows(
            rows
        )


def safe_float(value):
    if value is None:
        return 0.0

    text = str(
        value
    ).strip()

    if not text:
        return 0.0

    try:
        result = float(
            text
        )
    except Exception:
        return 0.0

    if not np.isfinite(
        result
    ):
        return 0.0

    return result


# ============================================================
# 从原始150维构造“关系是否存在”
#
# 重要：
# 绝不能从standardize后的150维判断是否存在。
# ============================================================

def raw_relation_active_matrix(
    rows,
    semantic_by_sha,
    names150,
):
    matrix = np.zeros(
        (
            len(rows),
            len(names150),
        ),
        dtype=np.float64,
    )

    for i, row in enumerate(
        rows
    ):
        sha = normalize_sha(
            row[
                "sha256"
            ]
        )

        if sha not in semantic_by_sha:
            raise SystemExit(
                "150维缺少SHA: "
                + sha
            )

        source = (
            semantic_by_sha[
                sha
            ]
        )

        for j, name in enumerate(
            names150
        ):
            value = safe_float(
                source.get(
                    name,
                    0.0,
                )
            )

            matrix[
                i,
                j,
            ] = (
                1.0
                if abs(value) > EPS
                else 0.0
            )

    return matrix


# ============================================================
# 计算关系类别方向
#
# d_j =
# P(relation_j present | malware)
# -
# P(relation_j present | benign)
# ============================================================

def full_relation_direction(
    active,
    labels,
):
    active = np.asarray(
        active,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    benign = (
        labels == 0
    )

    malware = (
        labels == 1
    )

    n0 = int(
        benign.sum()
    )

    n1 = int(
        malware.sum()
    )

    if n0 <= 0 or n1 <= 0:
        raise RuntimeError(
            "训练折缺少某个类别"
        )

    benign_rate = (
        active[
            benign
        ].sum(
            axis=0
        )
        / float(n0)
    )

    malware_rate = (
        active[
            malware
        ].sum(
            axis=0
        )
        / float(n1)
    )

    return (
        malware_rate
        - benign_rate
    )


# ============================================================
# 单个APK的一致性
#
# C(x) =
# |sum_j a_j d_j|
# -----------------
# sum_j a_j |d_j|
#
# 0 -> 活跃关系类别指向冲突
# 1 -> 活跃关系基本指向同一类别
#
# 不需要APK真实标签。
# ============================================================

def consistency_from_direction(
    active_vector,
    direction,
):
    active_vector = np.asarray(
        active_vector,
        dtype=np.float64,
    )

    direction = np.asarray(
        direction,
        dtype=np.float64,
    )

    weighted = (
        active_vector
        * direction
    )

    denominator = float(
        np.sum(
            np.abs(
                weighted
            )
        )
    )

    if denominator <= EPS:
        return 0.0

    numerator = abs(
        float(
            np.sum(
                weighted
            )
        )
    )

    value = (
        numerator
        / denominator
    )

    # 数值保险。
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ============================================================
# 训练样本：
#
# 每个样本自己的C必须使用留一法。
#
# 如果当前样本是恶意：
#   计算恶意关系率时排除它自己。
#
# 如果当前样本是良性：
#   计算良性关系率时排除它自己。
#
# 因此当前样本自己的标签不会被用于构造自己的d_j。
# ============================================================

def loo_train_consistency(
    active,
    labels,
):
    active = np.asarray(
        active,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    n = len(
        labels
    )

    benign_mask = (
        labels == 0
    )

    malware_mask = (
        labels == 1
    )

    n0 = int(
        benign_mask.sum()
    )

    n1 = int(
        malware_mask.sum()
    )

    if n0 <= 1:
        raise RuntimeError(
            "良性训练样本不足以做留一法"
        )

    if n1 <= 1:
        raise RuntimeError(
            "恶意训练样本不足以做留一法"
        )

    count0 = (
        active[
            benign_mask
        ].sum(
            axis=0
        )
    )

    count1 = (
        active[
            malware_mask
        ].sum(
            axis=0
        )
    )

    result = np.zeros(
        (
            n,
            1,
        ),
        dtype=np.float64,
    )

    for i in range(
        n
    ):
        ai = active[
            i
        ]

        yi = int(
            labels[
                i
            ]
        )

        if yi == 1:
            malware_rate = (
                (
                    count1
                    - ai
                )
                / float(
                    n1 - 1
                )
            )

            benign_rate = (
                count0
                / float(
                    n0
                )
            )

        elif yi == 0:
            malware_rate = (
                count1
                / float(
                    n1
                )
            )

            benign_rate = (
                (
                    count0
                    - ai
                )
                / float(
                    n0 - 1
                )
            )

        else:
            raise RuntimeError(
                "标签必须为0/1"
            )

        direction = (
            malware_rate
            - benign_rate
        )

        result[
            i,
            0,
        ] = (
            consistency_from_direction(
                ai,
                direction,
            )
        )

    return result


# ============================================================
# 验证样本：
#
# d_j只能来自训练折。
# 验证标签完全不参与。
# ============================================================

def test_consistency(
    train_active,
    train_labels,
    test_active,
):
    direction = (
        full_relation_direction(
            train_active,
            train_labels,
        )
    )

    result = np.zeros(
        (
            len(test_active),
            1,
        ),
        dtype=np.float64,
    )

    for i in range(
        len(test_active)
    ):
        result[
            i,
            0,
        ] = (
            consistency_from_direction(
                test_active[
                    i
                ],
                direction,
            )
        )

    return result


# ============================================================
# 用训练折统计量标准化这一维。
#
# 不引入新超参数。
# ============================================================

def standardize_one_feature(
    train,
    test,
):
    train = np.asarray(
        train,
        dtype=np.float64,
    )

    test = np.asarray(
        test,
        dtype=np.float64,
    )

    mean = float(
        train.mean()
    )

    std = float(
        train.std()
    )

    if (
        not np.isfinite(mean)
        or
        not np.isfinite(std)
    ):
        raise RuntimeError(
            "一致性统计量非有限"
        )

    if std <= EPS:
        return (
            np.zeros_like(
                train,
                dtype=np.float32,
            ),
            np.zeros_like(
                test,
                dtype=np.float32,
            ),
            mean,
            std,
        )

    train_z = (
        (
            train
            - mean
        )
        / std
    )

    test_z = (
        (
            test
            - mean
        )
        / std
    )

    return (
        train_z.astype(
            np.float32
        ),
        test_z.astype(
            np.float32
        ),
        mean,
        std,
    )


# ============================================================
# 新的数据构造：
#
# [数量10 | 一致性1 | 原150标准化关系]
#
# presence仍然完全来自原150维。
# ============================================================

def build_fold_data_count10_consistency1_150(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    semantic_by_sha,
    names150,
):
    # --------------------------------------------------------
    # 先调用最原始Frozen+150数据构造。
    #
    # 这样：
    # Compact、150标准化、coverage、presence
    # 全部与现有正式模型保持完全一致。
    # --------------------------------------------------------

    data = (
        ORIGINAL_BUILD_FOLD_DATA(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
            semantic_by_sha,
            names150,
        )
    )

    # --------------------------------------------------------
    # 数量10：
    # 继续使用已经验证过的旧20维文件，
    # 只选择caller_log/caller_ratio。
    # --------------------------------------------------------

    old20_by_sha = (
        aug.load_old20_rows()
    )

    (
        count_train,
        count_test,
        count_names,
    ) = base.standardize_named(
        train_rows,
        test_rows,
        old20_by_sha,
        COUNT10_NAMES,
    )

    if len(
        count_names
    ) != 10:
        raise SystemExit(
            "当前折数量10可用维数不是10："
            + str(
                len(
                    count_names
                )
            )
        )

    # --------------------------------------------------------
    # 原始150关系是否存在。
    # --------------------------------------------------------

    train_active = (
        raw_relation_active_matrix(
            train_rows,
            semantic_by_sha,
            names150,
        )
    )

    test_active = (
        raw_relation_active_matrix(
            test_rows,
            semantic_by_sha,
            names150,
        )
    )

    y_train = (
        base.labels_for(
            train_rows,
            compact_by_sha,
        )
    )

    # --------------------------------------------------------
    # 训练：留一法
    # 验证：只使用训练折类别统计
    # --------------------------------------------------------

    consistency_train_raw = (
        loo_train_consistency(
            train_active,
            y_train,
        )
    )

    consistency_test_raw = (
        test_consistency(
            train_active,
            y_train,
            test_active,
        )
    )

    (
        consistency_train,
        consistency_test,
        consistency_mean,
        consistency_std,
    ) = standardize_one_feature(
        consistency_train_raw,
        consistency_test_raw,
    )

    semantic150_train = (
        data[
            "semantic_train"
        ]
    )

    semantic150_test = (
        data[
            "semantic_test"
        ]
    )

    # --------------------------------------------------------
    # 网络前11维作为structure：
    #
    # 0:10  数量10
    # 10    一致性1
    #
    # 后面仍是原150关系。
    # --------------------------------------------------------

    data[
        "semantic_train"
    ] = np.concatenate(
        [
            count_train,
            consistency_train,
            semantic150_train,
        ],
        axis=1,
    )

    data[
        "semantic_test"
    ] = np.concatenate(
        [
            count_test,
            consistency_test,
            semantic150_test,
        ],
        axis=1,
    )

    data[
        "structure_names"
    ] = (
        list(
            count_names
        )
        + [
            CONSISTENCY_NAME
        ]
    )

    data[
        "semantic150_names"
    ] = list(
        data[
            "semantic_names"
        ]
    )

    data[
        "semantic_names"
    ] = (
        data[
            "structure_names"
        ]
        + list(
            data[
                "semantic_names"
            ]
        )
    )

    data[
        "consistency_train_raw"
    ] = (
        consistency_train_raw
    )

    data[
        "consistency_test_raw"
    ] = (
        consistency_test_raw
    )

    data[
        "consistency_mean"
    ] = consistency_mean

    data[
        "consistency_std"
    ] = consistency_std

    return data


# ============================================================
# 两个实验配置
# ============================================================

VARIANTS = (
    "baseline_count10_150",
    "count10_consistency1_150",
)


def configure_variant(
    variant,
):
    if (
        variant
        ==
        "baseline_count10_150"
    ):
        # 已验证的数量10+150原实现。
        aug.STRUCTURE_DIM = 10

        aug.OLD20_NAMES = list(
            COUNT10_NAMES
        )

        base.build_fold_data = (
            aug.build_fold_data_20_150
        )

        base.FrozenAnchorResidualNet = (
            aug.FrozenAnchorResidual20_150Net
        )

        return

    if (
        variant
        ==
        "count10_consistency1_150"
    ):
        # 数量10 + 一致性1。
        aug.STRUCTURE_DIM = 11

        base.build_fold_data = (
            build_fold_data_count10_consistency1_150
        )

        # 原结构+150网络本身通过STRUCTURE_DIM动态切分，
        # 无需改网络。
        base.FrozenAnchorResidualNet = (
            aug.FrozenAnchorResidual20_150Net
        )

        return

    raise ValueError(
        "未知variant: "
        + variant
    )


def get_config(
    variant,
):
    if len(
        ORIGINAL_CONFIGS
    ) != 1:
        raise SystemExit(
            "预期基础Frozen脚本只有一个固定配置，实际="
            + str(
                len(
                    ORIGINAL_CONFIGS
                )
            )
        )

    config = copy.deepcopy(
        ORIGINAL_CONFIGS[
            0
        ]
    )

    config[
        "name"
    ] = variant

    return config


# ============================================================
# 汇总
# ============================================================

def summarize_variant(
    rows,
):
    by_split = defaultdict(
        list
    )

    for row in rows:
        by_split[
            row[
                "split_name"
            ]
        ].append(
            row
        )

    split_rows = []

    for split_name in sorted(
        by_split
    ):
        rs = by_split[
            split_name
        ]

        split_rows.append({
            "split_name":
                split_name,

            "mean_BA":
                float(
                    np.mean(
                        [
                            r[
                                "balanced_accuracy"
                            ]
                            for r in rs
                        ]
                    )
                ),

            "mean_AUC":
                float(
                    np.mean(
                        [
                            r[
                                "auc"
                            ]
                            for r in rs
                        ]
                    )
                ),

            "mean_ACC":
                float(
                    np.mean(
                        [
                            r[
                                "accuracy"
                            ]
                            for r in rs
                        ]
                    )
                ),

            "mean_epoch":
                float(
                    np.mean(
                        [
                            r[
                                "epoch"
                            ]
                            for r in rs
                        ]
                    )
                ),
        })

    split_ba = np.asarray(
        [
            r[
                "mean_BA"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    split_auc = np.asarray(
        [
            r[
                "mean_AUC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    split_acc = np.asarray(
        [
            r[
                "mean_ACC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    return (
        {
            "mean_BA":
                float(
                    split_ba.mean()
                ),

            "min_split_BA":
                float(
                    split_ba.min()
                ),

            "std_split_BA":
                float(
                    split_ba.std()
                ),

            "mean_AUC":
                float(
                    split_auc.mean()
                ),

            "mean_ACC":
                float(
                    split_acc.mean()
                ),
        },
        split_rows,
    )


# ============================================================
# 主程序
# ============================================================

def main():
    print(
        "======================================"
    )

    print(
        "2017-only：数量10+150 vs +一致性1"
    )

    print(
        "======================================"
    )

    print(
        "本脚本不会读取任何2018-2021文件。"
    )

    # --------------------------------------------------------
    # 固定2017输入
    # --------------------------------------------------------

    compact_path = Path(
        "artifacts/p1-features/"
        "features.csv"
    )

    schema_path = Path(
        "artifacts/p1-features/"
        "feature_schema.json"
    )

    semantic_path = Path(
        "artifacts/clean-local-context-2017/"
        "semantic_relation_features.csv.gz"
    )

    split_path = Path(
        "artifacts/p9-2017-rolling-splits/"
        "split_manifest.csv"
    )

    for path in (
        compact_path,
        schema_path,
        semantic_path,
        split_path,
    ):
        if not path.exists():
            raise SystemExit(
                "找不到输入: "
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                compact_path
            )
        )
    )

    compact_schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            semantic_path
        )
    )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    if len(
        names150
    ) != 150:
        raise SystemExit(
            "150维数量异常: "
            + str(
                len(
                    names150
                )
            )
        )

    rolling_parts = (
        base.read_split_rows(
            split_path
        )
    )

    rolling_splits = {
        name:
            parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling" in name
            and parts[
                "train"
            ]
            and parts[
                "test"
            ]
        )
    }

    if len(
        rolling_splits
    ) != 3:
        raise SystemExit(
            "2017滚动划分应为3个，实际="
            + str(
                len(
                    rolling_splits
                )
            )
        )

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
    )

    print(
        "final150 =",
        len(
            names150
        ),
    )

    # 强制加载数量结构文件，
    # 防止运行到一半才报错。
    old20_by_sha = (
        aug.load_old20_rows()
    )

    required_shas = set()

    for parts in (
        rolling_splits.values()
    ):
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                required_shas.add(
                    normalize_sha(
                        row[
                            "sha256"
                        ]
                    )
                )

    missing_old20 = (
        required_shas
        - set(
            old20_by_sha
        )
    )

    if missing_old20:
        raise SystemExit(
            "数量结构缺少2017 APK: "
            + str(
                len(
                    missing_old20
                )
            )
        )

    # --------------------------------------------------------
    # 正式实验
    # --------------------------------------------------------

    all_run_rows = []

    split_output_rows = []

    summary_rows = []

    for variant in (
        VARIANTS
    ):
        print()
        print(
            "======================================"
        )

        print(
            variant
        )

        print(
            "======================================"
        )

        configure_variant(
            variant
        )

        config = get_config(
            variant
        )

        current_rows = []

        consistency_qc_rows = []

        for split_name in sorted(
            rolling_splits
        ):
            train_rows = (
                rolling_splits[
                    split_name
                ][
                    "train"
                ]
            )

            val_rows = (
                rolling_splits[
                    split_name
                ][
                    "test"
                ]
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

            print()
            print(
                split_name,
                "train=",
                len(
                    train_rows
                ),
                "val=",
                len(
                    val_rows
                ),
            )

            print(
                " semantic_dim=",
                data[
                    "semantic_train"
                ].shape[
                    1
                ],
            )

            if (
                variant
                ==
                "count10_consistency1_150"
            ):
                tr_raw = (
                    data[
                        "consistency_train_raw"
                    ].ravel()
                )

                va_raw = (
                    data[
                        "consistency_test_raw"
                    ].ravel()
                )

                print(
                    " consistency train:"
                    " mean=",
                    f"{tr_raw.mean():.4f}",
                    " median=",
                    f"{np.median(tr_raw):.4f}",
                    " min=",
                    f"{tr_raw.min():.4f}",
                    " max=",
                    f"{tr_raw.max():.4f}",
                    sep="",
                )

                print(
                    " consistency val:"
                    " mean=",
                    f"{va_raw.mean():.4f}",
                    " median=",
                    f"{np.median(va_raw):.4f}",
                    " min=",
                    f"{va_raw.min():.4f}",
                    " max=",
                    f"{va_raw.max():.4f}",
                    sep="",
                )

                consistency_qc_rows.append({
                    "split_name":
                        split_name,

                    "train_mean":
                        float(
                            tr_raw.mean()
                        ),

                    "train_median":
                        float(
                            np.median(
                                tr_raw
                            )
                        ),

                    "train_min":
                        float(
                            tr_raw.min()
                        ),

                    "train_max":
                        float(
                            tr_raw.max()
                        ),

                    "val_mean":
                        float(
                            va_raw.mean()
                        ),

                    "val_median":
                        float(
                            np.median(
                                va_raw
                            )
                        ),

                    "val_min":
                        float(
                            va_raw.min()
                        ),

                    "val_max":
                        float(
                            va_raw.max()
                        ),

                    "standardize_mean":
                        float(
                            data[
                                "consistency_mean"
                            ]
                        ),

                    "standardize_std":
                        float(
                            data[
                                "consistency_std"
                            ]
                        ),
                })

            for seed in (
                SEEDS
            ):
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

                if not history:
                    raise RuntimeError(
                        "fit_one返回空history"
                    )

                # 与现有2017消融完全一致：
                # 当前训练轨迹取最高BA。
                best = max(
                    history,
                    key=lambda row:
                        row[
                            "balanced_accuracy"
                        ],
                )

                row = {
                    "variant":
                        variant,

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "balanced_accuracy":
                        float(
                            best[
                                "balanced_accuracy"
                            ]
                        ),

                    "auc":
                        float(
                            best[
                                "auc"
                            ]
                        ),

                    "accuracy":
                        float(
                            best[
                                "accuracy"
                            ]
                        ),

                    "epoch":
                        int(
                            best.get(
                                "epoch",
                                best_epoch,
                            )
                        ),
                }

                current_rows.append(
                    row
                )

                all_run_rows.append(
                    row
                )

                print(
                    f" seed={seed} "
                    f"BA={row['balanced_accuracy']:.6f} "
                    f"AUC={row['auc']:.6f} "
                    f"ACC={row['accuracy']:.6f} "
                    f"epoch={row['epoch']}"
                )

        (
            summary,
            split_rows,
        ) = summarize_variant(
            current_rows
        )

        summary[
            "variant"
        ] = variant

        summary_rows.append(
            summary
        )

        for row in (
            split_rows
        ):
            row[
                "variant"
            ] = variant

            split_output_rows.append(
                row
            )

        if consistency_qc_rows:
            write_csv(
                OUTPUT_DIR
                / "consistency_qc.csv",
                consistency_qc_rows,
            )

        print()
        print(
            "SUMMARY",
            variant,
        )

        print(
            f" meanBA={summary['mean_BA']:.6f}"
            f" minBA={summary['min_split_BA']:.6f}"
            f" stdBA={summary['std_split_BA']:.6f}"
            f" AUC={summary['mean_AUC']:.6f}"
            f" ACC={summary['mean_ACC']:.6f}"
        )

    # ========================================================
    # 配对比较：
    # 完全相同split + seed。
    # ========================================================

    baseline = {
        (
            row[
                "split_name"
            ],
            row[
                "seed"
            ],
        ):
            row
        for row in all_run_rows
        if (
            row[
                "variant"
            ]
            ==
            "baseline_count10_150"
        )
    }

    consistency = {
        (
            row[
                "split_name"
            ],
            row[
                "seed"
            ],
        ):
            row
        for row in all_run_rows
        if (
            row[
                "variant"
            ]
            ==
            "count10_consistency1_150"
        )
    }

    if set(
        baseline
    ) != set(
        consistency
    ):
        raise RuntimeError(
            "两个实验split×seed不一致"
        )

    pair_rows = []

    ba_deltas = []

    auc_deltas = []

    acc_deltas = []

    wins = 0
    ties = 0
    losses = 0

    for key in sorted(
        baseline
    ):
        b = baseline[
            key
        ]

        c = consistency[
            key
        ]

        d_ba = (
            c[
                "balanced_accuracy"
            ]
            -
            b[
                "balanced_accuracy"
            ]
        )

        d_auc = (
            c[
                "auc"
            ]
            -
            b[
                "auc"
            ]
        )

        d_acc = (
            c[
                "accuracy"
            ]
            -
            b[
                "accuracy"
            ]
        )

        ba_deltas.append(
            d_ba
        )

        auc_deltas.append(
            d_auc
        )

        acc_deltas.append(
            d_acc
        )

        if d_ba > 1e-12:
            result = "win"
            wins += 1

        elif d_ba < -1e-12:
            result = "loss"
            losses += 1

        else:
            result = "tie"
            ties += 1

        pair_rows.append({
            "split_name":
                key[
                    0
                ],

            "seed":
                key[
                    1
                ],

            "baseline_BA":
                b[
                    "balanced_accuracy"
                ],

            "consistency_BA":
                c[
                    "balanced_accuracy"
                ],

            "delta_BA":
                d_ba,

            "delta_AUC":
                d_auc,

            "delta_ACC":
                d_acc,

            "BA_result":
                result,
        })

    pair_summary = [{
        "pairs":
            len(
                ba_deltas
            ),

        "BA_wins":
            wins,

        "BA_ties":
            ties,

        "BA_losses":
            losses,

        "mean_delta_BA":
            float(
                np.mean(
                    ba_deltas
                )
            ),

        "median_delta_BA":
            float(
                np.median(
                    ba_deltas
                )
            ),

        "mean_delta_AUC":
            float(
                np.mean(
                    auc_deltas
                )
            ),

        "mean_delta_ACC":
            float(
                np.mean(
                    acc_deltas
                )
            ),
    }]

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        OUTPUT_DIR
        / "per_run.csv",
        all_run_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        split_output_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_comparison.csv",
        pair_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_summary.csv",
        pair_summary,
    )

    # ========================================================
    # 最终输出
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "最终2017-only比较"
    )

    print(
        "======================================"
    )

    for row in (
        summary_rows
    ):
        print(
            f"{row['variant']:32s} "
            f"BA={row['mean_BA']:.6f} "
            f"minBA={row['min_split_BA']:.6f} "
            f"stdBA={row['std_split_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f}"
        )

    p = pair_summary[
        0
    ]

    print()
    print(
        "======================================"
    )

    print(
        "15个split×seed配对"
    )

    print(
        "======================================"
    )

    print(
        "胜/平/负 =",
        p[
            "BA_wins"
        ],
        "/",
        p[
            "BA_ties"
        ],
        "/",
        p[
            "BA_losses"
        ],
    )

    print(
        "平均 ΔBA =",
        f"{p['mean_delta_BA']:+.6f}",
    )

    print(
        "中位 ΔBA =",
        f"{p['median_delta_BA']:+.6f}",
    )

    print(
        "平均 ΔAUC =",
        f"{p['mean_delta_AUC']:+.6f}",
    )

    print(
        "平均 ΔACC =",
        f"{p['mean_delta_ACC']:+.6f}",
    )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "本实验只使用2017滚动划分。"
    )


if __name__ == "__main__":
    main()
