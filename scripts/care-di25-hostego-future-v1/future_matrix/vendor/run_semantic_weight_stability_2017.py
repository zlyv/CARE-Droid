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
)

from run_compact_20_150_ablation_2017 import (
    support150_names,
)


# ============================================================
# 冻结配置
# ============================================================

ALPHA = 0.25

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03

RESAMPLES = 30

# 固定优化器种子：
# 权重变化只观察训练样本重采样带来的变化。
OPTIMIZER_SEED = 0

# 仅用于判断数值意义上的0，不是特征筛选阈值。
WEIGHT_EPS = 1e-10


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

OUTPUT_DIR = Path(
    "artifacts/semantic-weight-stability-2017"
)


# ============================================================
# IO
# ============================================================

def read_gz_csv(path):
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

    opener = (
        gzip.open
        if str(path).endswith(".gz")
        else open
    )

    kwargs = {
        "mode": "wt" if str(path).endswith(".gz") else "w",
        "encoding": "utf-8",
        "newline": "",
    }

    with opener(
        path,
        **kwargs,
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def norm_sha(value):
    return str(
        value
    ).strip().casefold()


def strict_float(
    value,
    sha,
    feature,
):
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(
            f"{sha}: "
            f"{feature}无法转成数值："
            f"{value!r}"
        ) from exc

    if not np.isfinite(result):
        raise ValueError(
            f"{sha}: "
            f"{feature}不是有限数值"
        )

    return result


# ============================================================
# 关系名
# ============================================================

def parse_relation(name):
    parts = name.split("__")

    if len(parts) != 3:
        raise ValueError(
            "异常关系名："
            + name
        )

    return (
        parts[1],
        parts[2],
    )


# ============================================================
# 分层有放回重采样
# ============================================================

def stratified_resample_indices(
    y,
    seed,
):
    """
    良性和恶意分别有放回抽样，
    每类抽回原来的样本数量。

    因此每次重采样：
      - 总训练样本数不变
      - 两类样本数量不变
      - 只有具体抽中的APK发生变化
    """

    rng = np.random.default_rng(
        seed
    )

    sampled = []

    for label in (0, 1):
        idx = np.flatnonzero(
            y == label
        )

        if len(idx) == 0:
            raise RuntimeError(
                f"类别{label}在训练折中为空"
            )

        draw = rng.choice(
            idx,
            size=len(idx),
            replace=True,
        )

        sampled.append(
            draw
        )

    sampled = np.concatenate(
        sampled
    )

    rng.shuffle(
        sampled
    )

    return sampled


# ============================================================
# 符号诊断
# ============================================================

def sign_code(value):
    if value > WEIGHT_EPS:
        return 1

    if value < -WEIGHT_EPS:
        return -1

    return 0


def sign_statistics(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    signs = np.asarray(
        [
            sign_code(v)
            for v in values
        ],
        dtype=int,
    )

    pos = int(
        np.sum(signs > 0)
    )

    neg = int(
        np.sum(signs < 0)
    )

    zero = int(
        np.sum(signs == 0)
    )

    # 这里分母保留全部重采样次数。
    # 如果权重经常接近0，也会降低一致率。
    dominant_rate = (
        max(pos, neg)
        / len(signs)
    )

    return {
        "positive_count":
            pos,

        "negative_count":
            neg,

        "near_zero_count":
            zero,

        "dominant_sign_rate":
            float(
                dominant_rate
            ),
    }


# ============================================================
# 主程序
# ============================================================

def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
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

    # --------------------------------------------------------
    # Compact
    # --------------------------------------------------------

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        norm_sha(
            row["sha256"]
        ):
            row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # 150维
    # --------------------------------------------------------

    semantic_rows = read_gz_csv(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        norm_sha(
            row["sha256"]
        ):
            row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    if len(names150) != 150:
        raise SystemExit(
            f"关系数量不是150："
            f"{len(names150)}"
        )

    splits = read_split_rows(
        SPLIT_PATH
    )

    print(
        "======================================"
    )
    print(
        "2017 语义关系权重稳定性诊断"
    )
    print(
        "======================================"
    )

    print(
        "关系总数 =",
        len(names150),
    )

    print(
        "每折重采样次数 =",
        RESAMPLES,
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "L2 =",
        L2,
    )

    print(
        "iterations =",
        ITERATIONS,
    )

    print(
        "优化器种子固定 =",
        OPTIMIZER_SEED,
    )

    print()
    print(
        "注意：每个训练折只标准化一次，"
        "30次重采样共用同一尺度。"
    )

    # --------------------------------------------------------
    # 输出容器
    # --------------------------------------------------------

    weight_rows = []
    per_fold_rows = []
    fold_overview_rows = []
    resample_rows = []

    split_names = sorted(
        splits
    )

    # ========================================================
    # 每一个2017训练折
    # ========================================================

    for split_index, split_name in enumerate(
        split_names
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        # 注意：
        # 完全不读取这个滚动划分的test标签做诊断。
        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        norm_sha(
                            row["sha256"]
                        )
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        benign_n = int(
            np.sum(
                y_train == 0
            )
        )

        malware_n = int(
            np.sum(
                y_train == 1
            )
        )

        # ----------------------------------------------------
        # 标准化只在完整训练折上做一次。
        #
        # 之后30次重采样只抽行，
        # 避免每次重新标准化导致权重尺度本身变化。
        # ----------------------------------------------------

        (
            compact_train,
            _,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            train_rows,
            compact_by_sha,
            schema,
        )

        (
            semantic_train,
            _,
            semantic_names,
        ) = standardize_named(
            train_rows,
            train_rows,
            semantic_by_sha,
            names150,
        )

        usable_set = set(
            semantic_names
        )

        # ----------------------------------------------------
        # 原始关系出现次数
        # ----------------------------------------------------

        raw_nonzero_count = {}

        for feature in names150:
            count = 0

            for row in train_rows:
                sha = norm_sha(
                    row["sha256"]
                )

                value = strict_float(
                    semantic_by_sha[
                        sha
                    ][feature],
                    sha,
                    feature,
                )

                if value > 0:
                    count += 1

            raw_nonzero_count[
                feature
            ] = count

        # ----------------------------------------------------
        # 完整训练折模型
        # ----------------------------------------------------

        joint_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        full_w, full_b = fit_logistic(
            joint_train,
            y_train,
            OPTIMIZER_SEED,
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            l2=L2,
        )

        compact_dim = (
            compact_train.shape[1]
        )

        semantic_full_w = (
            full_w[
                compact_dim:
                compact_dim
                + len(semantic_names)
            ]
        )

        if (
            len(semantic_full_w)
            != len(semantic_names)
        ):
            raise RuntimeError(
                "语义权重维数错误"
            )

        # 输入是 alpha*S，
        # 因此对标准化后的原始语义特征S而言，
        # 有效系数是 alpha*w。
        full_effective = {
            feature:
                float(
                    ALPHA
                    * weight
                )
            for feature, weight
            in zip(
                semantic_names,
                semantic_full_w,
            )
        }

        print()
        print(
            "======================================"
        )
        print(
            split_name
        )
        print(
            "======================================"
        )

        print(
            "train =",
            len(train_rows),
            "良性 =",
            benign_n,
            "恶意 =",
            malware_n,
        )

        print(
            "Compact可用维 =",
            len(compact_names),
        )

        print(
            "150中训练期可用关系 =",
            len(semantic_names),
        )

        # ----------------------------------------------------
        # 30次分层重采样
        # ----------------------------------------------------

        for resample_id in range(
            RESAMPLES
        ):
            seed = (
                17000
                + split_index * 1000
                + resample_id
            )

            idx = (
                stratified_resample_indices(
                    y_train,
                    seed,
                )
            )

            x_resampled = (
                joint_train[
                    idx
                ]
            )

            y_resampled = (
                y_train[
                    idx
                ]
            )

            unique_n = int(
                len(
                    np.unique(idx)
                )
            )

            resample_rows.append({
                "split_name":
                    split_name,

                "resample_id":
                    resample_id,

                "sampling_seed":
                    seed,

                "train_n":
                    len(idx),

                "unique_n":
                    unique_n,

                "unique_rate":
                    float(
                        unique_n
                        / len(idx)
                    ),

                "benign_n":
                    int(
                        np.sum(
                            y_resampled == 0
                        )
                    ),

                "malware_n":
                    int(
                        np.sum(
                            y_resampled == 1
                        )
                    ),
            })

            w, b = fit_logistic(
                x_resampled,
                y_resampled,
                OPTIMIZER_SEED,
                iterations=ITERATIONS,
                learning_rate=LEARNING_RATE,
                l2=L2,
            )

            semantic_w = (
                w[
                    compact_dim:
                    compact_dim
                    + len(semantic_names)
                ]
            )

            for feature, model_weight in zip(
                semantic_names,
                semantic_w,
            ):
                anchor, context = (
                    parse_relation(
                        feature
                    )
                )

                effective_weight = float(
                    ALPHA
                    * model_weight
                )

                weight_rows.append({
                    "split_name":
                        split_name,

                    "resample_id":
                        resample_id,

                    "feature":
                        feature,

                    "anchor":
                        anchor,

                    "context":
                        context,

                    "model_weight":
                        float(
                            model_weight
                        ),

                    "effective_weight":
                        effective_weight,

                    "sign":
                        sign_code(
                            effective_weight
                        ),
                })

            if (
                (resample_id + 1) % 5
                == 0
            ):
                print(
                    " 重采样",
                    resample_id + 1,
                    "/",
                    RESAMPLES,
                    "完成",
                    flush=True,
                )

        # ----------------------------------------------------
        # 本折关系稳定性汇总
        # ----------------------------------------------------

        fold_weights = defaultdict(
            list
        )

        for row in weight_rows:
            if (
                row["split_name"]
                == split_name
            ):
                fold_weights[
                    row["feature"]
                ].append(
                    float(
                        row[
                            "effective_weight"
                        ]
                    )
                )

        for feature in names150:
            anchor, context = (
                parse_relation(
                    feature
                )
            )

            nonzero_count = (
                raw_nonzero_count[
                    feature
                ]
            )

            nonzero_rate = float(
                nonzero_count
                / len(train_rows)
            )

            if feature not in usable_set:
                per_fold_rows.append({
                    "split_name":
                        split_name,

                    "feature":
                        feature,

                    "anchor":
                        anchor,

                    "context":
                        context,

                    "train_n":
                        len(train_rows),

                    "raw_nonzero_count":
                        nonzero_count,

                    "raw_nonzero_rate":
                        nonzero_rate,

                    "usable":
                        0,

                    "full_effective_weight":
                        0.0,

                    "weight_mean":
                        0.0,

                    "weight_std":
                        0.0,

                    "weight_median":
                        0.0,

                    "weight_mean_abs":
                        0.0,

                    "weight_p10":
                        0.0,

                    "weight_p90":
                        0.0,

                    "positive_count":
                        0,

                    "negative_count":
                        0,

                    "near_zero_count":
                        RESAMPLES,

                    "dominant_sign_rate":
                        0.0,

                    "agreement_with_full":
                        0.0,

                    "relative_dispersion":
                        0.0,

                    "stable_importance_score":
                        0.0,

                    "instability_score":
                        0.0,
                })

                continue

            values = np.asarray(
                fold_weights[
                    feature
                ],
                dtype=float,
            )

            if (
                len(values)
                != RESAMPLES
            ):
                raise RuntimeError(
                    f"{split_name}: "
                    f"{feature}只有"
                    f"{len(values)}个权重"
                )

            signs = sign_statistics(
                values
            )

            full_value = (
                full_effective[
                    feature
                ]
            )

            full_sign = sign_code(
                full_value
            )

            sampled_signs = np.asarray(
                [
                    sign_code(v)
                    for v in values
                ],
                dtype=int,
            )

            if full_sign == 0:
                agreement = float(
                    np.mean(
                        sampled_signs == 0
                    )
                )
            else:
                agreement = float(
                    np.mean(
                        sampled_signs
                        == full_sign
                    )
                )

            mean_abs = float(
                np.mean(
                    np.abs(values)
                )
            )

            weight_std = float(
                np.std(values)
            )

            relative_dispersion = float(
                weight_std
                / (
                    mean_abs
                    + 1e-12
                )
            )

            sign_rate = (
                signs[
                    "dominant_sign_rate"
                ]
            )

            per_fold_rows.append({
                "split_name":
                    split_name,

                "feature":
                    feature,

                "anchor":
                    anchor,

                "context":
                    context,

                "train_n":
                    len(train_rows),

                "raw_nonzero_count":
                    nonzero_count,

                "raw_nonzero_rate":
                    nonzero_rate,

                "usable":
                    1,

                "full_effective_weight":
                    full_value,

                "weight_mean":
                    float(
                        np.mean(values)
                    ),

                "weight_std":
                    weight_std,

                "weight_median":
                    float(
                        np.median(values)
                    ),

                "weight_mean_abs":
                    mean_abs,

                "weight_p10":
                    float(
                        np.percentile(
                            values,
                            10,
                        )
                    ),

                "weight_p90":
                    float(
                        np.percentile(
                            values,
                            90,
                        )
                    ),

                **signs,

                "agreement_with_full":
                    agreement,

                "relative_dispersion":
                    relative_dispersion,

                # 这里只用于排序诊断，
                # 不作为特征选择规则。
                "stable_importance_score":
                    float(
                        mean_abs
                        * sign_rate
                    ),

                "instability_score":
                    float(
                        mean_abs
                        * (
                            1.0
                            - sign_rate
                        )
                    ),
            })

        current = [
            row
            for row in per_fold_rows
            if (
                row["split_name"]
                == split_name
                and int(
                    row["usable"]
                ) == 1
            )
        ]

        sign_rates = np.asarray(
            [
                float(
                    row[
                        "dominant_sign_rate"
                    ]
                )
                for row in current
            ],
            dtype=float,
        )

        relative_dispersion = (
            np.asarray(
                [
                    float(
                        row[
                            "relative_dispersion"
                        ]
                    )
                    for row in current
                ],
                dtype=float,
            )
        )

        current_resamples = [
            row
            for row in resample_rows
            if row["split_name"]
            == split_name
        ]

        fold_overview_rows.append({
            "split_name":
                split_name,

            "train_n":
                len(train_rows),

            "benign_n":
                benign_n,

            "malware_n":
                malware_n,

            "usable_relation_n":
                len(current),

            "sign_consistency_100pct_n":
                int(
                    np.sum(
                        sign_rates >= 1.0
                    )
                ),

            "sign_consistency_ge_90pct_n":
                int(
                    np.sum(
                        sign_rates >= 0.90
                    )
                ),

            "sign_consistency_lt_70pct_n":
                int(
                    np.sum(
                        sign_rates < 0.70
                    )
                ),

            "median_sign_consistency":
                float(
                    np.median(
                        sign_rates
                    )
                ),

            "median_relative_dispersion":
                float(
                    np.median(
                        relative_dispersion
                    )
                ),

            "mean_resample_unique_rate":
                float(
                    np.mean([
                        row[
                            "unique_rate"
                        ]
                        for row
                        in current_resamples
                    ])
                ),
        })

    # ========================================================
    # 跨折汇总
    # ========================================================

    cross_rows = []

    for feature in names150:
        anchor, context = (
            parse_relation(
                feature
            )
        )

        fold_items = [
            row
            for row in per_fold_rows
            if (
                row["feature"]
                == feature
                and int(
                    row["usable"]
                ) == 1
            )
        ]

        all_boot = [
            row
            for row in weight_rows
            if row["feature"]
            == feature
        ]

        if not fold_items:
            cross_rows.append({
                "feature":
                    feature,

                "anchor":
                    anchor,

                "context":
                    context,

                "usable_fold_count":
                    0,

                "usable_all_folds":
                    0,

                "pooled_resample_n":
                    0,

                "pooled_dominant_sign_rate":
                    0.0,

                "full_fold_sign_consistency":
                    0.0,

                "mean_fold_sign_consistency":
                    0.0,

                "min_fold_sign_consistency":
                    0.0,

                "mean_abs_effective_weight":
                    0.0,

                "mean_nonzero_rate":
                    0.0,

                "stable_importance_score":
                    0.0,

                "instability_score":
                    0.0,
            })

            continue

        boot_values = np.asarray(
            [
                float(
                    row[
                        "effective_weight"
                    ]
                )
                for row in all_boot
            ],
            dtype=float,
        )

        pooled_sign = (
            sign_statistics(
                boot_values
            )
        )

        full_signs = np.asarray(
            [
                sign_code(
                    float(
                        row[
                            "full_effective_weight"
                        ]
                    )
                )
                for row in fold_items
            ],
            dtype=int,
        )

        full_pos = int(
            np.sum(
                full_signs > 0
            )
        )

        full_neg = int(
            np.sum(
                full_signs < 0
            )
        )

        full_zero = int(
            np.sum(
                full_signs == 0
            )
        )

        full_consistency = float(
            max(
                full_pos,
                full_neg,
                full_zero,
            )
            / len(full_signs)
        )

        mean_abs = float(
            np.mean(
                np.abs(
                    boot_values
                )
            )
        )

        fold_sign_rates = np.asarray(
            [
                float(
                    row[
                        "dominant_sign_rate"
                    ]
                )
                for row in fold_items
            ],
            dtype=float,
        )

        pooled_rate = (
            pooled_sign[
                "dominant_sign_rate"
            ]
        )

        cross_rows.append({
            "feature":
                feature,

            "anchor":
                anchor,

            "context":
                context,

            "usable_fold_count":
                len(fold_items),

            "usable_all_folds":
                int(
                    len(fold_items)
                    == len(split_names)
                ),

            "pooled_resample_n":
                len(boot_values),

            "pooled_positive_count":
                pooled_sign[
                    "positive_count"
                ],

            "pooled_negative_count":
                pooled_sign[
                    "negative_count"
                ],

            "pooled_near_zero_count":
                pooled_sign[
                    "near_zero_count"
                ],

            "pooled_dominant_sign_rate":
                pooled_rate,

            "full_fold_positive_count":
                full_pos,

            "full_fold_negative_count":
                full_neg,

            "full_fold_near_zero_count":
                full_zero,

            "full_fold_sign_consistency":
                full_consistency,

            "mean_fold_sign_consistency":
                float(
                    fold_sign_rates.mean()
                ),

            "min_fold_sign_consistency":
                float(
                    fold_sign_rates.min()
                ),

            "mean_abs_effective_weight":
                mean_abs,

            "mean_nonzero_rate":
                float(
                    np.mean([
                        float(
                            row[
                                "raw_nonzero_rate"
                            ]
                        )
                        for row in fold_items
                    ])
                ),

            # 仍然只是诊断排序。
            "stable_importance_score":
                float(
                    mean_abs
                    * pooled_rate
                ),

            "instability_score":
                float(
                    mean_abs
                    * (
                        1.0
                        - pooled_rate
                    )
                ),
        })

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        OUTPUT_DIR
        / "bootstrap_weights.csv.gz",
        weight_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_fold_relation_stability.csv",
        per_fold_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "cross_fold_relation_stability.csv",
        cross_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "fold_overview.csv",
        fold_overview_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "resample_diagnostics.csv",
        resample_rows,
    )

    # ========================================================
    # 打印总体诊断
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "逐折稳定性概览"
    )
    print(
        "======================================"
    )

    for row in fold_overview_rows:
        print()
        print(
            row["split_name"]
        )

        print(
            " train=",
            row["train_n"],
            "可用关系=",
            row[
                "usable_relation_n"
            ],
        )

        print(
            " 方向100%一致=",
            row[
                "sign_consistency_100pct_n"
            ],
        )

        print(
            " 方向>=90%一致=",
            row[
                "sign_consistency_ge_90pct_n"
            ],
        )

        print(
            " 方向<70%一致=",
            row[
                "sign_consistency_lt_70pct_n"
            ],
        )

        print(
            " 方向一致率中位数=",
            f"{row['median_sign_consistency']:.3f}",
        )

        print(
            " 相对波动中位数=",
            f"{row['median_relative_dispersion']:.3f}",
        )

        print(
            " 每次重采样平均保留不同APK比例=",
            f"{row['mean_resample_unique_rate']:.3f}",
        )

    # --------------------------------------------------------
    # 跨折：权重大 + 方向稳定
    # --------------------------------------------------------

    eligible = [
        row
        for row in cross_rows
        if int(
            row[
                "usable_fold_count"
            ]
        ) >= 2
    ]

    stable_rank = sorted(
        eligible,
        key=lambda row: (
            float(
                row[
                    "stable_importance_score"
                ]
            ),
            int(
                row[
                    "usable_fold_count"
                ]
            ),
        ),
        reverse=True,
    )

    unstable_rank = sorted(
        eligible,
        key=lambda row: (
            float(
                row[
                    "instability_score"
                ]
            ),
            float(
                row[
                    "mean_abs_effective_weight"
                ]
            ),
        ),
        reverse=True,
    )

    print()
    print(
        "======================================"
    )
    print(
        "跨折稳定且权重较大的关系"
    )
    print(
        "仅诊断，不代表最终特征选择"
    )
    print(
        "======================================"
    )

    for row in stable_rank[:15]:
        print(
            f"{row['anchor']:28s}"
            f" -> "
            f"{row['context']:28s} "
            f"folds={row['usable_fold_count']} "
            f"sign={row['pooled_dominant_sign_rate']:.3f} "
            f"minSign={row['min_fold_sign_consistency']:.3f} "
            f"|w|={row['mean_abs_effective_weight']:.5f} "
            f"presence={row['mean_nonzero_rate']:.3f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "跨折不稳定风险较大的关系"
    )
    print(
        "风险分数=平均绝对权重×(1-方向一致率)"
    )
    print(
        "仍然只是诊断，不删除任何关系"
    )
    print(
        "======================================"
    )

    for row in unstable_rank[:15]:
        print(
            f"{row['anchor']:28s}"
            f" -> "
            f"{row['context']:28s} "
            f"folds={row['usable_fold_count']} "
            f"sign={row['pooled_dominant_sign_rate']:.3f} "
            f"minSign={row['min_fold_sign_consistency']:.3f} "
            f"|w|={row['mean_abs_effective_weight']:.5f} "
            f"presence={row['mean_nonzero_rate']:.3f} "
            f"risk={row['instability_score']:.6f}"
        )

    # --------------------------------------------------------
    # 一点总体统计
    # --------------------------------------------------------

    all_three = [
        row
        for row in cross_rows
        if int(
            row[
                "usable_all_folds"
            ]
        ) == 1
    ]

    if all_three:
        rates = np.asarray(
            [
                float(
                    row[
                        "pooled_dominant_sign_rate"
                    ]
                )
                for row in all_three
            ],
            dtype=float,
        )

        full_rates = np.asarray(
            [
                float(
                    row[
                        "full_fold_sign_consistency"
                    ]
                )
                for row in all_three
            ],
            dtype=float,
        )

        print()
        print(
            "======================================"
        )
        print(
            "跨三折总体"
        )
        print(
            "======================================"
        )

        print(
            "三折都可用的关系数 =",
            len(all_three),
        )

        print(
            "重采样方向一致率中位数 =",
            f"{np.median(rates):.3f}",
        )

        print(
            "重采样方向一致率>=0.9的关系 =",
            int(
                np.sum(
                    rates >= 0.90
                )
            ),
            "/",
            len(all_three),
        )

        print(
            "三个完整训练折权重方向完全一致 =",
            int(
                np.sum(
                    full_rates >= 1.0
                )
            ),
            "/",
            len(all_three),
        )

    print()
    print(
        "======================================"
    )
    print(
        "保存目录"
    )
    print(
        "======================================"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
