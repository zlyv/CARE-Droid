#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72


THRESHOLD = 0.5


def norm_sha(x):
    return str(x or "").strip().casefold()


def read_csv(path):
    with Path(path).open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def train_only_standardize(
    x_train,
    x_test,
):
    mean = np.mean(
        x_train,
        axis=0,
    )

    std = np.std(
        x_train,
        axis=0,
    )

    std = np.where(
        std > 1e-12,
        std,
        1.0,
    )

    return (
        (x_train - mean) / std,
        (x_test - mean) / std,
    )


def squared_distance_matrix(
    left,
    right,
):
    left_norm = np.sum(
        left * left,
        axis=1,
        keepdims=True,
    )

    right_norm = np.sum(
        right * right,
        axis=1,
        keepdims=True,
    ).T

    d2 = (
        left_norm
        + right_norm
        - 2.0 * left @ right.T
    )

    return np.maximum(
        d2,
        0.0,
    )


def entropy_binary(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0

    return float(
        -p * np.log2(p)
        - (1.0 - p)
        * np.log2(1.0 - p)
    )


def collision_scores(
    x_train,
    y_train,
    x_test,
    k,
):
    """
    只用外层训练折：
      1. 训练折拟合标准化；
      2. 测试APK找k个最近训练APK；
      3. 计算局部类别混杂度；
      4. 再乘局部紧密程度。

    测试标签不参与。
    """

    (
        z_train,
        z_test,
    ) = train_only_standardize(
        x_train,
        x_test,
    )

    effective_k = min(
        int(k),
        len(z_train) - 1,
    )

    if effective_k < 2:
        raise RuntimeError(
            "Too few training samples for kNN"
        )

    # -----------------------------------------
    # 训练折自身的典型局部距离
    # 用于定义“这个测试样本离训练数据近不近”
    # -----------------------------------------

    train_d2 = squared_distance_matrix(
        z_train,
        z_train,
    )

    np.fill_diagonal(
        train_d2,
        np.inf,
    )

    nearest_train_d2 = np.partition(
        train_d2,
        kth=effective_k - 1,
        axis=1,
    )[
        :,
        :effective_k
    ]

    nearest_train_dist = np.sqrt(
        nearest_train_d2
    )

    train_mean_knn_distance = np.mean(
        nearest_train_dist,
        axis=1,
    )

    reference_distance = float(
        np.median(
            train_mean_knn_distance
        )
    )

    if not np.isfinite(reference_distance):
        raise RuntimeError(
            "Invalid reference distance"
        )

    if reference_distance <= 1e-12:
        reference_distance = 1.0

    # -----------------------------------------
    # 测试APK -> 外层训练折
    # -----------------------------------------

    test_d2 = squared_distance_matrix(
        z_test,
        z_train,
    )

    nearest_idx = np.argpartition(
        test_d2,
        kth=effective_k - 1,
        axis=1,
    )[
        :,
        :effective_k
    ]

    rows = []

    for i in range(
        len(z_test)
    ):
        idx = nearest_idx[i]

        distances = np.sqrt(
            test_d2[
                i,
                idx,
            ]
        )

        neighbor_labels = y_train[
            idx
        ]

        malware_fraction = float(
            np.mean(
                neighbor_labels
            )
        )

        local_entropy = entropy_binary(
            malware_fraction
        )

        mean_distance = float(
            np.mean(
                distances
            )
        )

        # 越靠近训练数据越大，范围(0,1]
        local_proximity = (
            reference_distance
            / (
                reference_distance
                + mean_distance
            )
        )

        score = (
            local_entropy
            * local_proximity
        )

        rows.append({
            "collision_score":
                float(score),

            "local_label_entropy":
                float(local_entropy),

            "local_proximity":
                float(local_proximity),

            "local_malware_fraction":
                malware_fraction,

            "mean_knn_distance":
                mean_distance,

            "reference_train_distance":
                reference_distance,
        })

    return rows


def metric_summary(
    y,
    probability,
):
    result = base.metrics(
        y,
        probability,
        threshold=THRESHOLD,
    )

    return {
        "ACC":
            result["accuracy"],

        "BA":
            result["balanced_accuracy"],

        "AUC":
            result["auc"],

        "F1":
            result["f1"],

        "TP":
            result["tp"],

        "TN":
            result["tn"],

        "FP":
            result["fp"],

        "FN":
            result["fn"],
    }


def correction_counts(
    y,
    coarse_probability,
    structure_probability,
):
    coarse_pred = (
        coarse_probability
        >= THRESHOLD
    ).astype(int)

    structure_pred = (
        structure_probability
        >= THRESHOLD
    ).astype(int)

    fixed = (
        (coarse_pred != y)
        & (structure_pred == y)
    )

    broken = (
        (coarse_pred == y)
        & (structure_pred != y)
    )

    return {
        "fixed":
            int(np.sum(fixed)),

        "broken":
            int(np.sum(broken)),

        "net":
            int(
                np.sum(fixed)
                - np.sum(broken)
            ),
    }


def equal_count_bins(
    scores,
    indices,
    n_bins=5,
):
    """
    按分数排序后切成等样本数区间。
    避免大量同分导致分位数边界异常。
    """

    ordered = sorted(
        map(int, indices),
        key=lambda i: (
            float(scores[i]),
            i,
        ),
    )

    chunks = np.array_split(
        np.asarray(
            ordered,
            dtype=int,
        ),
        n_bins,
    )

    return chunks


def top_by_fold(
    folds,
    score,
    budget,
    shas,
):
    selected = []

    for fold in sorted(
        np.unique(folds)
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        k = int(
            round(
                len(idx)
                * budget
            )
        )

        k = max(
            1,
            min(
                len(idx),
                k,
            ),
        )

        ordered = sorted(
            map(int, idx),
            key=lambda i: (
                -float(score[i]),
                shas[i],
            ),
        )

        selected.extend(
            ordered[:k]
        )

    return np.asarray(
        selected,
        dtype=int,
    )


def random_by_fold(
    folds,
    budget,
    seed,
):
    selected = []

    for fold in sorted(
        np.unique(folds)
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        k = int(
            round(
                len(idx)
                * budget
            )
        )

        k = max(
            1,
            min(
                len(idx),
                k,
            ),
        )

        rng = np.random.default_rng(
            2026
            + seed * 1009
            + int(fold) * 7919
        )

        chosen = rng.choice(
            idx,
            size=k,
            replace=False,
        )

        selected.extend(
            map(int, chosen)
        )

    return np.asarray(
        selected,
        dtype=int,
    )


def route_result(
    y,
    coarse_probability,
    structure_probability,
    selected,
):
    final_probability = np.array(
        coarse_probability,
        copy=True,
    )

    final_probability[
        selected
    ] = structure_probability[
        selected
    ]

    metrics = metric_summary(
        y,
        final_probability,
    )

    corr = correction_counts(
        y,
        coarse_probability,
        final_probability,
    )

    routed_n = len(
        selected
    )

    return {
        **metrics,

        **corr,

        "routed_n":
            routed_n,

        "routed_ratio":
            routed_n / len(y),

        "net_per_100_routed":
            (
                corr["net"]
                / routed_n
                * 100.0
                if routed_n
                else 0.0
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--e3-predictions",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e3-top5-benefit-2017/"
            "e3_oof_predictions.csv"
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e5-semantic-count-collision-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # 读取已经冻结的2017 OOF结果
    # =====================================================

    pred_rows = read_csv(
        args.e3_predictions
    )

    if len(pred_rows) != 799:
        raise RuntimeError(
            f"Expected 799 rows, got {len(pred_rows)}"
        )

    manifest = []

    for row in pred_rows:
        manifest.append({
            "sha256":
                norm_sha(
                    row["sha256"]
                ),

            "label":
                int(
                    row["label"]
                ),

            "fold":
                int(
                    row["fold"]
                ),

            "coarse_probability":
                float(
                    row[
                        "coarse_probability"
                    ]
                ),

            "structure_probability":
                float(
                    row[
                        "structure_probability"
                    ]
                ),
        })

    shas = [
        row["sha256"]
        for row in manifest
    ]

    labels = np.asarray(
        [
            row["label"]
            for row in manifest
        ],
        dtype=int,
    )

    folds = np.asarray(
        [
            row["fold"]
            for row in manifest
        ],
        dtype=int,
    )

    coarse_probability = np.asarray(
        [
            row[
                "coarse_probability"
            ]
            for row in manifest
        ],
        dtype=float,
    )

    structure_probability = np.asarray(
        [
            row[
                "structure_probability"
            ]
            for row in manifest
        ],
        dtype=float,
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E5"
    )
    print(
        "语义计数碰撞验证"
    )
    print(
        "========================================"
    )

    print(
        "APK数量:",
        len(manifest),
    )

    print(
        "近邻数量 k:",
        args.k,
    )

    # =====================================================
    # Compact源特征
    # =====================================================

    compact_rows = read_csv(
        args.compact
    )

    compact_by_sha = {}

    for row in compact_rows:
        sha = norm_sha(
            row.get(
                "sha256",
                "",
            )
        )

        if sha:
            row = dict(row)
            row["sha256"] = sha
            compact_by_sha[
                sha
            ] = row

    missing = [
        sha
        for sha in shas
        if sha not in compact_by_sha
    ]

    if missing:
        raise RuntimeError(
            "Missing compact rows: "
            + repr(
                missing[:10]
            )
        )

    cohort_rows = [
        compact_by_sha[
            sha
        ]
        for sha in shas
    ]

    with args.compact_schema.open(
        encoding="utf-8",
    ) as f:
        compact_schema = json.load(f)

    (
        _,
        _,
        canonical_original_names,
    ) = allratio.original_compact_matrix_named(
        cohort_rows,
        cohort_rows[:1],
        compact_by_sha,
        compact_schema,
    )

    canonical_original_names = list(
        canonical_original_names
    )

    if len(
        canonical_original_names
    ) != 76:
        raise RuntimeError(
            "Expected canonical Compact raw dim=76"
        )

    # =====================================================
    # 每个外层测试折：
    # 只使用另外四折定义碰撞空间和标签混杂
    # =====================================================

    collision_score = np.zeros(
        len(manifest),
        dtype=float,
    )

    entropy_score = np.zeros(
        len(manifest),
        dtype=float,
    )

    proximity_score = np.zeros(
        len(manifest),
        dtype=float,
    )

    malware_fraction = np.zeros(
        len(manifest),
        dtype=float,
    )

    mean_distance = np.zeros(
        len(manifest),
        dtype=float,
    )

    reference_distance = np.zeros(
        len(manifest),
        dtype=float,
    )

    for fold in sorted(
        np.unique(folds)
    ):
        train_idx = np.flatnonzero(
            folds != fold
        )

        test_idx = np.flatnonzero(
            folds == fold
        )

        train_rows = [
            cohort_rows[i]
            for i in train_idx
        ]

        test_rows = [
            cohort_rows[i]
            for i in test_idx
        ]

        (
            compact_train,
            compact_test,
            compact_names,
        ) = (
            p72
            .compact_matrix_allratio70_foldsafe(
                train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )
        )

        if len(compact_names) != 70:
            raise RuntimeError(
                "Expected Compact dim=70"
            )

        result_rows = collision_scores(
            compact_train,
            labels[
                train_idx
            ],
            compact_test,
            args.k,
        )

        for local_i, global_i in enumerate(
            test_idx
        ):
            r = result_rows[
                local_i
            ]

            collision_score[
                global_i
            ] = r[
                "collision_score"
            ]

            entropy_score[
                global_i
            ] = r[
                "local_label_entropy"
            ]

            proximity_score[
                global_i
            ] = r[
                "local_proximity"
            ]

            malware_fraction[
                global_i
            ] = r[
                "local_malware_fraction"
            ]

            mean_distance[
                global_i
            ] = r[
                "mean_knn_distance"
            ]

            reference_distance[
                global_i
            ] = r[
                "reference_train_distance"
            ]

        print(
            "Fold",
            int(fold),
            "完成:",
            len(test_idx),
            "APK",
        )

    if not np.all(
        np.isfinite(
            collision_score
        )
    ):
        raise RuntimeError(
            "Invalid collision scores"
        )

    # =====================================================
    # 保存每个APK碰撞分数
    # =====================================================

    coarse_pred = (
        coarse_probability
        >= THRESHOLD
    ).astype(int)

    structure_pred = (
        structure_probability
        >= THRESHOLD
    ).astype(int)

    error_mask = (
        coarse_pred
        != labels
    )

    fixed_mask = (
        (coarse_pred != labels)
        & (structure_pred == labels)
    )

    broken_mask = (
        (coarse_pred == labels)
        & (structure_pred != labels)
    )

    score_rows = []

    for i, row in enumerate(
        manifest
    ):
        score_rows.append({
            "sha256":
                row["sha256"],

            "label":
                row["label"],

            "fold":
                row["fold"],

            "coarse_probability":
                coarse_probability[i],

            "coarse_margin":
                abs(
                    coarse_probability[i]
                    - 0.5
                ),

            "coarse_error":
                int(
                    error_mask[i]
                ),

            "structure_probability":
                structure_probability[i],

            "structure_fixed":
                int(
                    fixed_mask[i]
                ),

            "structure_broken":
                int(
                    broken_mask[i]
                ),

            "collision_score":
                collision_score[i],

            "local_label_entropy":
                entropy_score[i],

            "local_proximity":
                proximity_score[i],

            "local_malware_fraction":
                malware_fraction[i],

            "mean_knn_distance":
                mean_distance[i],

            "reference_train_distance":
                reference_distance[i],
        })

    write_csv(
        args.output
        / "e5_collision_scores.csv",
        score_rows,
    )

    # =====================================================
    # 实验1：
    # 碰撞程度越高，第一层错误是否越集中？
    #
    # 同时看结构增益。
    # =====================================================

    all_indices = np.arange(
        len(manifest)
    )

    bins = equal_count_bins(
        collision_score,
        all_indices,
        n_bins=5,
    )

    bin_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验1：按碰撞程度分5组"
    )
    print(
        "========================================"
    )

    for q, idx in enumerate(
        bins,
        start=1,
    ):
        y = labels[
            idx
        ]

        c_prob = coarse_probability[
            idx
        ]

        s_prob = structure_probability[
            idx
        ]

        c_metrics = metric_summary(
            y,
            c_prob,
        )

        s_metrics = metric_summary(
            y,
            s_prob,
        )

        corr = correction_counts(
            y,
            c_prob,
            s_prob,
        )

        error_rate = float(
            np.mean(
                coarse_pred[
                    idx
                ] != y
            )
        )

        row = {
            "collision_group":
                q,

            "group_name":
                (
                    "最低碰撞"
                    if q == 1
                    else
                    "最高碰撞"
                    if q == 5
                    else
                    f"第{q}组"
                ),

            "n":
                len(idx),

            "mean_collision_score":
                float(
                    np.mean(
                        collision_score[
                            idx
                        ]
                    )
                ),

            "coarse_error_rate":
                error_rate,

            "coarse_ACC":
                c_metrics[
                    "ACC"
                ],

            "structure_ACC":
                s_metrics[
                    "ACC"
                ],

            "delta_ACC":
                s_metrics[
                    "ACC"
                ]
                - c_metrics[
                    "ACC"
                ],

            "coarse_BA":
                c_metrics[
                    "BA"
                ],

            "structure_BA":
                s_metrics[
                    "BA"
                ],

            "delta_BA":
                s_metrics[
                    "BA"
                ]
                - c_metrics[
                    "BA"
                ],

            **corr,
        }

        bin_rows.append(
            row
        )

        print(
            f"第{q}组",
            f"n={len(idx):3d}",
            f"碰撞={row['mean_collision_score']:.4f}",
            f"第一层错误率={error_rate:.3f}",
            f"第一层ACC={row['coarse_ACC']:.3f}",
            f"结构ACC={row['structure_ACC']:.3f}",
            f"提升={row['delta_ACC']:+.3f}",
            f"净修复={row['net']:+d}",
        )

    write_csv(
        args.output
        / "e5_collision_quintiles.csv",
        bin_rows,
    )

    # =====================================================
    # 实验2：
    # 第一层已经很自信的APK里，
    # 碰撞还能否找到错误？
    #
    # 高置信度：
    # p <= 0.2 或 p >= 0.8
    # =====================================================

    high_conf_mask = (
        np.abs(
            coarse_probability
            - 0.5
        )
        >= 0.30
    )

    high_conf_idx = np.flatnonzero(
        high_conf_mask
    )

    hc_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验2：高置信度APK中的碰撞"
    )
    print(
        "p <= 0.2 或 p >= 0.8"
    )
    print(
        "========================================"
    )

    print(
        "高置信度APK数量:",
        len(high_conf_idx),
    )

    if len(high_conf_idx) >= 25:

        hc_bins = equal_count_bins(
            collision_score,
            high_conf_idx,
            n_bins=5,
        )

        for q, idx in enumerate(
            hc_bins,
            start=1,
        ):
            y = labels[
                idx
            ]

            c_prob = coarse_probability[
                idx
            ]

            s_prob = structure_probability[
                idx
            ]

            c_metrics = metric_summary(
                y,
                c_prob,
            )

            s_metrics = metric_summary(
                y,
                s_prob,
            )

            corr = correction_counts(
                y,
                c_prob,
                s_prob,
            )

            error_rate = float(
                np.mean(
                    coarse_pred[
                        idx
                    ] != y
                )
            )

            row = {
                "collision_group":
                    q,

                "n":
                    len(idx),

                "mean_collision_score":
                    float(
                        np.mean(
                            collision_score[
                                idx
                            ]
                        )
                    ),

                "coarse_error_rate":
                    error_rate,

                "coarse_ACC":
                    c_metrics[
                        "ACC"
                    ],

                "structure_ACC":
                    s_metrics[
                        "ACC"
                    ],

                "delta_ACC":
                    s_metrics[
                        "ACC"
                    ]
                    - c_metrics[
                        "ACC"
                    ],

                **corr,
            }

            hc_rows.append(
                row
            )

            print(
                f"第{q}组",
                f"n={len(idx):3d}",
                f"碰撞={row['mean_collision_score']:.4f}",
                f"错误率={error_rate:.3f}",
                f"结构提升={row['delta_ACC']:+.3f}",
                f"净修复={row['net']:+d}",
            )

    write_csv(
        args.output
        / "e5_high_confidence_collision_quintiles.csv",
        hc_rows,
    )

    # =====================================================
    # 碰撞与低置信度是不是基本一回事？
    # =====================================================

    uncertainty_score = (
        -np.abs(
            coarse_probability
            - 0.5
        )
    )

    try:
        from scipy.stats import spearmanr

        rho, rho_p = spearmanr(
            collision_score,
            uncertainty_score,
        )

        rho = float(rho)
        rho_p = float(rho_p)

    except Exception:
        rho = float("nan")
        rho_p = float("nan")

    print()
    print(
        "碰撞分数 vs 低置信度 Spearman相关:"
    )
    print(
        "rho =",
        rho,
        "p =",
        rho_p,
    )

    # =====================================================
    # 实验3：
    # 固定预算下 Collision Route vs Low Confidence
    # =====================================================

    budgets = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    random_repeats = int(
        args.random_repeats
    )

    oracle_score = np.zeros(
        len(manifest),
        dtype=float,
    )

    oracle_score[
        fixed_mask
    ] = 1.0

    oracle_score[
        broken_mask
    ] = -1.0

    budget_rows = []
    random_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验3：固定预算路由"
    )
    print(
        "========================================"
    )

    for budget in budgets:

        print()
        print(
            f"预算 {budget:.0%}"
        )

        # -------------------------
        # Random
        # -------------------------

        random_results = []

        for seed in range(
            random_repeats
        ):
            selected = random_by_fold(
                folds,
                budget,
                seed,
            )

            result = route_result(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            random_results.append(
                result
            )

            random_rows.append({
                "budget":
                    budget,

                "seed":
                    seed,

                **result,
            })

        random_summary = {
            "method":
                "随机",

            "budget":
                budget,
        }

        for key in (
            "ACC",
            "BA",
            "AUC",
            "fixed",
            "broken",
            "net",
            "routed_n",
            "routed_ratio",
            "net_per_100_routed",
        ):
            values = np.asarray(
                [
                    r[key]
                    for r
                    in random_results
                ],
                dtype=float,
            )

            random_summary[
                key
            ] = float(
                np.mean(values)
            )

            random_summary[
                key + "_std"
            ] = float(
                np.std(values)
            )

        budget_rows.append(
            random_summary
        )

        print(
            "随机       ",
            f"ACC={random_summary['ACC']:.4f}",
            f"净修复={random_summary['net']:+.2f}",
        )

        # -------------------------
        # Low confidence
        # -------------------------

        low_selected = top_by_fold(
            folds,
            uncertainty_score,
            budget,
            shas,
        )

        low_result = route_result(
            labels,
            coarse_probability,
            structure_probability,
            low_selected,
        )

        budget_rows.append({
            "method":
                "低置信度",

            "budget":
                budget,

            **low_result,
        })

        print(
            "低置信度   ",
            f"ACC={low_result['ACC']:.4f}",
            f"修复={low_result['fixed']}",
            f"破坏={low_result['broken']}",
            f"净修复={low_result['net']:+d}",
        )

        # -------------------------
        # Collision
        # -------------------------

        collision_selected = (
            top_by_fold(
                folds,
                collision_score,
                budget,
                shas,
            )
        )

        collision_result = (
            route_result(
                labels,
                coarse_probability,
                structure_probability,
                collision_selected,
            )
        )

        budget_rows.append({
            "method":
                "语义碰撞",

            "budget":
                budget,

            **collision_result,
        })

        print(
            "语义碰撞   ",
            f"ACC={collision_result['ACC']:.4f}",
            f"修复={collision_result['fixed']}",
            f"破坏={collision_result['broken']}",
            f"净修复={collision_result['net']:+d}",
        )

        # -------------------------
        # Oracle
        # -------------------------

        oracle_selected = top_by_fold(
            folds,
            oracle_score,
            budget,
            shas,
        )

        oracle_result = route_result(
            labels,
            coarse_probability,
            structure_probability,
            oracle_selected,
        )

        budget_rows.append({
            "method":
                "理论上界",

            "budget":
                budget,

            **oracle_result,
        })

        print(
            "理论上界   ",
            f"ACC={oracle_result['ACC']:.4f}",
            f"净修复={oracle_result['net']:+d}",
        )

    write_csv(
        args.output
        / "e5_budget_summary.csv",
        budget_rows,
    )

    write_csv(
        args.output
        / "e5_random_runs.csv",
        random_rows,
    )

    # =====================================================
    # 总体参考
    # =====================================================

    coarse_metrics = metric_summary(
        labels,
        coarse_probability,
    )

    structure_metrics = metric_summary(
        labels,
        structure_probability,
    )

    overall_corr = correction_counts(
        labels,
        coarse_probability,
        structure_probability,
    )

    q1 = bin_rows[0]
    q5 = bin_rows[-1]

    summary = {
        "source_apks":
            len(manifest),

        "k_neighbors":
            args.k,

        "coarse_ACC":
            coarse_metrics[
                "ACC"
            ],

        "structure_ACC":
            structure_metrics[
                "ACC"
            ],

        "overall_fixed":
            overall_corr[
                "fixed"
            ],

        "overall_broken":
            overall_corr[
                "broken"
            ],

        "overall_net":
            overall_corr[
                "net"
            ],

        "lowest_collision_error_rate":
            q1[
                "coarse_error_rate"
            ],

        "highest_collision_error_rate":
            q5[
                "coarse_error_rate"
            ],

        "error_rate_difference_high_minus_low":
            (
                q5[
                    "coarse_error_rate"
                ]
                - q1[
                    "coarse_error_rate"
                ]
            ),

        "lowest_collision_structure_gain":
            q1[
                "delta_ACC"
            ],

        "highest_collision_structure_gain":
            q5[
                "delta_ACC"
            ],

        "collision_uncertainty_spearman_rho":
            rho,

        "collision_uncertainty_spearman_p":
            rho_p,
    }

    (
        args.output
        / "e5_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "collision_definition":
            (
                "local binary label entropy among k nearest "
                "outer-training Compact neighbors multiplied "
                "by training-relative local proximity"
            ),

        "no_test_label_in_collision_score":
            True,

        "compact_dimension":
            70,

        "outer_folds":
            5,

        "k_neighbors":
            args.k,

        "high_confidence_definition":
            (
                "abs(p_coarse - 0.5) >= 0.30, "
                "equivalent to p<=0.2 or p>=0.8"
            ),

        "budgets":
            budgets,

        "random_repeats":
            random_repeats,
    }

    (
        args.output
        / "e5_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )
    print(
        "E5 核心摘要"
    )
    print(
        "========================================"
    )

    print(
        "最低碰撞组第一层错误率:",
        f"{q1['coarse_error_rate']:.4f}",
    )

    print(
        "最高碰撞组第一层错误率:",
        f"{q5['coarse_error_rate']:.4f}",
    )

    print(
        "最高 - 最低:",
        f"{summary['error_rate_difference_high_minus_low']:+.4f}",
    )

    print()

    print(
        "最低碰撞组结构提升:",
        f"{q1['delta_ACC']:+.4f}",
    )

    print(
        "最高碰撞组结构提升:",
        f"{q5['delta_ACC']:+.4f}",
    )

    print()

    print(
        "碰撞与低置信度相关:",
        f"rho={rho:.4f}",
    )

    print()

    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
