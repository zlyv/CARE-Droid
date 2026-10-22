#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


THRESHOLD = 0.5


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
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


def equal_count_bins(values, indices, n_bins=5):
    """
    Score越大，bin越高。
    只根据score排序，不使用label。
    """
    indices = np.asarray(
        indices,
        dtype=int,
    )

    ordered = sorted(
        map(int, indices),
        key=lambda i: (
            float(values[i]),
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

    result = np.zeros(
        len(values),
        dtype=np.int64,
    )

    for q, chunk in enumerate(
        chunks,
        start=1,
    ):
        result[chunk] = q

    return result


def foldwise_quintiles(
    values,
    folds,
):
    result = np.zeros(
        len(values),
        dtype=np.int64,
    )

    for fold in sorted(
        np.unique(folds)
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        local = equal_count_bins(
            values,
            idx,
            n_bins=5,
        )

        result[idx] = local[idx]

    if np.any(result == 0):
        raise RuntimeError(
            "Incomplete foldwise quintiles"
        )

    return result


def group_stats(
    indices,
    labels,
    coarse_prob,
    structure_prob,
):
    indices = np.asarray(
        indices,
        dtype=int,
    )

    if len(indices) == 0:
        return {
            "n": 0,
            "coarse_error_count": 0,
            "coarse_error_rate": float("nan"),
            "coarse_ACC": float("nan"),
            "structure_ACC": float("nan"),
            "delta_ACC": float("nan"),
            "fixed": 0,
            "broken": 0,
            "net": 0,
        }

    y = labels[indices]

    c_pred = (
        coarse_prob[indices]
        >= THRESHOLD
    ).astype(int)

    s_pred = (
        structure_prob[indices]
        >= THRESHOLD
    ).astype(int)

    c_error = (
        c_pred != y
    )

    fixed = (
        c_error
        & (s_pred == y)
    )

    broken = (
        (~c_error)
        & (s_pred != y)
    )

    coarse_acc = float(
        np.mean(
            c_pred == y
        )
    )

    structure_acc = float(
        np.mean(
            s_pred == y
        )
    )

    return {
        "n":
            int(len(indices)),

        "coarse_error_count":
            int(
                np.sum(
                    c_error
                )
            ),

        "coarse_error_rate":
            float(
                np.mean(
                    c_error
                )
            ),

        "coarse_ACC":
            coarse_acc,

        "structure_ACC":
            structure_acc,

        "delta_ACC":
            structure_acc
            - coarse_acc,

        "fixed":
            int(
                np.sum(
                    fixed
                )
            ),

        "broken":
            int(
                np.sum(
                    broken
                )
            ),

        "net":
            int(
                np.sum(
                    fixed
                )
                - np.sum(
                    broken
                )
            ),
    }


def fisher_error_test(
    high_idx,
    low_idx,
    labels,
    coarse_prob,
):
    """
    比较两组第一层错误率。
    """
    high_idx = np.asarray(
        high_idx,
        dtype=int,
    )

    low_idx = np.asarray(
        low_idx,
        dtype=int,
    )

    c_pred = (
        coarse_prob
        >= THRESHOLD
    ).astype(int)

    high_err = int(
        np.sum(
            c_pred[high_idx]
            != labels[high_idx]
        )
    )

    high_ok = (
        len(high_idx)
        - high_err
    )

    low_err = int(
        np.sum(
            c_pred[low_idx]
            != labels[low_idx]
        )
    )

    low_ok = (
        len(low_idx)
        - low_err
    )

    try:
        from scipy.stats import fisher_exact

        odds_ratio, p = fisher_exact(
            [
                [
                    high_err,
                    high_ok,
                ],
                [
                    low_err,
                    low_ok,
                ],
            ],
            alternative="greater",
        )

        return (
            float(odds_ratio),
            float(p),
        )

    except Exception:
        return (
            float("nan"),
            float("nan"),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e5-scores",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e5-semantic-count-collision-2017/"
            "e5_collision_scores.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e6-collision-uncertainty-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(
        args.e5_scores
    )

    if len(rows) != 799:
        raise RuntimeError(
            f"Expected 799 APKs, got {len(rows)}"
        )

    sha = np.asarray(
        [
            row["sha256"]
            for row in rows
        ],
        dtype=object,
    )

    labels = np.asarray(
        [
            int(row["label"])
            for row in rows
        ],
        dtype=np.int64,
    )

    folds = np.asarray(
        [
            int(row["fold"])
            for row in rows
        ],
        dtype=np.int64,
    )

    coarse_prob = np.asarray(
        [
            float(
                row[
                    "coarse_probability"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    structure_prob = np.asarray(
        [
            float(
                row[
                    "structure_probability"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    collision = np.asarray(
        [
            float(
                row[
                    "collision_score"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    # 越大越不确定
    uncertainty = (
        0.5
        - np.abs(
            coarse_prob
            - 0.5
        )
    )

    margin = np.abs(
        coarse_prob
        - 0.5
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E6"
    )
    print(
        "粗粒度表示碰撞 × 模型不确定性"
    )
    print(
        "========================================"
    )

    print(
        "APK数量:",
        len(rows),
    )

    # =====================================================
    # 每折内部排名，避免不同fold分数尺度问题
    # =====================================================

    collision_q = foldwise_quintiles(
        collision,
        folds,
    )

    uncertainty_q = foldwise_quintiles(
        uncertainty,
        folds,
    )

    # =====================================================
    # 5 × 5
    # =====================================================

    grid_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验1：5×5 碰撞-不确定性网格"
    )
    print(
        "U1最自信 → U5最不自信"
    )
    print(
        "C1最低碰撞 → C5最高碰撞"
    )
    print(
        "========================================"
    )

    for uq in range(
        1,
        6,
    ):
        print()
        print(
            f"不确定性 U{uq}:"
        )

        for cq in range(
            1,
            6,
        ):
            idx = np.flatnonzero(
                (uncertainty_q == uq)
                & (collision_q == cq)
            )

            stats = group_stats(
                idx,
                labels,
                coarse_prob,
                structure_prob,
            )

            row = {
                "uncertainty_quintile":
                    uq,

                "collision_quintile":
                    cq,

                "mean_uncertainty":
                    (
                        float(
                            np.mean(
                                uncertainty[
                                    idx
                                ]
                            )
                        )
                        if len(idx)
                        else float("nan")
                    ),

                "mean_collision":
                    (
                        float(
                            np.mean(
                                collision[
                                    idx
                                ]
                            )
                        )
                        if len(idx)
                        else float("nan")
                    ),

                **stats,
            }

            grid_rows.append(
                row
            )

            print(
                f"  C{cq}",
                f"n={stats['n']:3d}",
                f"错误率={stats['coarse_error_rate']:.3f}",
                f"结构提升={stats['delta_ACC']:+.3f}",
                f"净修复={stats['net']:+d}",
            )

    write_csv(
        args.output
        / "e6_grid_5x5.csv",
        grid_rows,
    )

    # =====================================================
    # 在每一个不确定性水平内部：
    # 最高碰撞20% vs 其余80%
    #
    # 这直接回答：
    # 在“不确定性大致相同”时，
    # 碰撞还能不能解释额外错误。
    # =====================================================

    within_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验2：控制不确定性后看碰撞"
    )
    print(
        "每个U层：C5 vs C1-C4"
    )
    print(
        "========================================"
    )

    for uq in range(
        1,
        6,
    ):
        high_idx = np.flatnonzero(
            (uncertainty_q == uq)
            & (collision_q == 5)
        )

        rest_idx = np.flatnonzero(
            (uncertainty_q == uq)
            & (collision_q < 5)
        )

        high_stats = group_stats(
            high_idx,
            labels,
            coarse_prob,
            structure_prob,
        )

        rest_stats = group_stats(
            rest_idx,
            labels,
            coarse_prob,
            structure_prob,
        )

        odds_ratio, p = fisher_error_test(
            high_idx,
            rest_idx,
            labels,
            coarse_prob,
        )

        error_lift = (
            high_stats[
                "coarse_error_rate"
            ]
            - rest_stats[
                "coarse_error_rate"
            ]
        )

        structure_gain_lift = (
            high_stats[
                "delta_ACC"
            ]
            - rest_stats[
                "delta_ACC"
            ]
        )

        row = {
            "uncertainty_quintile":
                uq,

            "high_collision_n":
                high_stats[
                    "n"
                ],

            "other_collision_n":
                rest_stats[
                    "n"
                ],

            "high_collision_error_rate":
                high_stats[
                    "coarse_error_rate"
                ],

            "other_collision_error_rate":
                rest_stats[
                    "coarse_error_rate"
                ],

            "error_rate_lift":
                error_lift,

            "high_collision_structure_gain":
                high_stats[
                    "delta_ACC"
                ],

            "other_collision_structure_gain":
                rest_stats[
                    "delta_ACC"
                ],

            "structure_gain_lift":
                structure_gain_lift,

            "high_collision_fixed":
                high_stats[
                    "fixed"
                ],

            "high_collision_broken":
                high_stats[
                    "broken"
                ],

            "high_collision_net":
                high_stats[
                    "net"
                ],

            "fisher_error_odds_ratio":
                odds_ratio,

            "fisher_one_sided_p":
                p,
        }

        within_rows.append(
            row
        )

        print()
        print(
            f"U{uq}:"
        )

        print(
            "  C5错误率      =",
            f"{high_stats['coarse_error_rate']:.4f}",
        )

        print(
            "  C1-C4错误率   =",
            f"{rest_stats['coarse_error_rate']:.4f}",
        )

        print(
            "  错误率差      =",
            f"{error_lift:+.4f}",
        )

        print(
            "  C5结构提升    =",
            f"{high_stats['delta_ACC']:+.4f}",
        )

        print(
            "  C1-C4结构提升 =",
            f"{rest_stats['delta_ACC']:+.4f}",
        )

        print(
            "  C5净修复      =",
            f"{high_stats['net']:+d}",
        )

        print(
            "  Fisher OR/p   =",
            odds_ratio,
            "/",
            p,
        )

    write_csv(
        args.output
        / "e6_collision_within_uncertainty.csv",
        within_rows,
    )

    # =====================================================
    # 容易解释的2×2
    #
    # 高置信度：
    # p<=0.2 or p>=0.8
    #
    # 高碰撞：
    # fold-wise C5
    # =====================================================

    high_confidence = (
        margin >= 0.30
    )

    high_collision = (
        collision_q == 5
    )

    two_by_two_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验3：2×2直观表"
    )
    print(
        "高置信度 = p<=0.2 或 p>=0.8"
    )
    print(
        "高碰撞 = 每折最高20%"
    )
    print(
        "========================================"
    )

    for conf_value, conf_name in (
        (
            True,
            "高置信度",
        ),
        (
            False,
            "非高置信度",
        ),
    ):
        for collision_value, collision_name in (
            (
                False,
                "非最高碰撞20%",
            ),
            (
                True,
                "最高碰撞20%",
            ),
        ):
            idx = np.flatnonzero(
                (high_confidence == conf_value)
                & (
                    high_collision
                    == collision_value
                )
            )

            stats = group_stats(
                idx,
                labels,
                coarse_prob,
                structure_prob,
            )

            row = {
                "confidence_group":
                    conf_name,

                "collision_group":
                    collision_name,

                **stats,
            }

            two_by_two_rows.append(
                row
            )

            print(
                f"{conf_name:8s}",
                f"{collision_name:12s}",
                f"n={stats['n']:3d}",
                f"错误率={stats['coarse_error_rate']:.4f}",
                f"结构提升={stats['delta_ACC']:+.4f}",
                f"净修复={stats['net']:+d}",
            )

    write_csv(
        args.output
        / "e6_table_2x2.csv",
        two_by_two_rows,
    )

    # =====================================================
    # Top20碰撞与Top20不确定性的重叠关系
    # =====================================================

    high_uncertainty = (
        uncertainty_q == 5
    )

    overlap_rows = []

    overlap_groups = [
        (
            "都不是",
            (~high_collision)
            & (~high_uncertainty),
        ),
        (
            "仅高碰撞",
            high_collision
            & (~high_uncertainty),
        ),
        (
            "仅高不确定",
            (~high_collision)
            & high_uncertainty,
        ),
        (
            "同时高碰撞和高不确定",
            high_collision
            & high_uncertainty,
        ),
    ]

    print()
    print(
        "========================================"
    )
    print(
        "实验4：最高20%碰撞与最高20%不确定性的重叠"
    )
    print(
        "========================================"
    )

    for name, mask in overlap_groups:
        idx = np.flatnonzero(
            mask
        )

        stats = group_stats(
            idx,
            labels,
            coarse_prob,
            structure_prob,
        )

        row = {
            "group":
                name,

            **stats,
        }

        overlap_rows.append(
            row
        )

        print(
            f"{name:20s}",
            f"n={stats['n']:3d}",
            f"错误率={stats['coarse_error_rate']:.4f}",
            f"结构提升={stats['delta_ACC']:+.4f}",
            f"修复={stats['fixed']:2d}",
            f"破坏={stats['broken']:2d}",
            f"净修复={stats['net']:+3d}",
        )

    write_csv(
        args.output
        / "e6_top20_overlap.csv",
        overlap_rows,
    )

    # =====================================================
    # 每APK保存分组结果
    # =====================================================

    apk_rows = []

    for i in range(
        len(rows)
    ):
        apk_rows.append({
            "sha256":
                sha[i],

            "label":
                int(
                    labels[i]
                ),

            "fold":
                int(
                    folds[i]
                ),

            "coarse_probability":
                float(
                    coarse_prob[i]
                ),

            "uncertainty":
                float(
                    uncertainty[i]
                ),

            "uncertainty_quintile":
                int(
                    uncertainty_q[i]
                ),

            "collision_score":
                float(
                    collision[i]
                ),

            "collision_quintile":
                int(
                    collision_q[i]
                ),

            "high_confidence":
                int(
                    high_confidence[i]
                ),

            "top20_collision":
                int(
                    high_collision[i]
                ),

            "top20_uncertainty":
                int(
                    high_uncertainty[i]
                ),

            "coarse_error":
                int(
                    (
                        coarse_prob[i]
                        >= THRESHOLD
                    )
                    != labels[i]
                ),

            "structure_probability":
                float(
                    structure_prob[i]
                ),
        })

    write_csv(
        args.output
        / "e6_apk_assignments.csv",
        apk_rows,
    )

    # =====================================================
    # 综合摘要
    # =====================================================

    confident_high_collision = np.flatnonzero(
        high_confidence
        & high_collision
    )

    confident_other = np.flatnonzero(
        high_confidence
        & (~high_collision)
    )

    hc_high_stats = group_stats(
        confident_high_collision,
        labels,
        coarse_prob,
        structure_prob,
    )

    hc_other_stats = group_stats(
        confident_other,
        labels,
        coarse_prob,
        structure_prob,
    )

    hc_or, hc_p = fisher_error_test(
        confident_high_collision,
        confident_other,
        labels,
        coarse_prob,
    )

    summary = {
        "source_apks":
            len(rows),

        "collision_definition":
            (
                "E5 source-only OOF coarse-representation "
                "kNN collision score"
            ),

        "uncertainty_definition":
            (
                "0.5 - abs(p_coarse - 0.5)"
            ),

        "binning":
            (
                "fold-wise equal-count quintiles; "
                "no labels used for bin assignment"
            ),

        "high_confidence_definition":
            (
                "p<=0.2 or p>=0.8"
            ),

        "high_collision_definition":
            (
                "fold-wise highest collision quintile"
            ),

        "high_conf_high_collision_n":
            hc_high_stats[
                "n"
            ],

        "high_conf_other_collision_n":
            hc_other_stats[
                "n"
            ],

        "high_conf_high_collision_error_rate":
            hc_high_stats[
                "coarse_error_rate"
            ],

        "high_conf_other_collision_error_rate":
            hc_other_stats[
                "coarse_error_rate"
            ],

        "high_conf_error_rate_lift":
            (
                hc_high_stats[
                    "coarse_error_rate"
                ]
                - hc_other_stats[
                    "coarse_error_rate"
                ]
            ),

        "high_conf_fisher_odds_ratio":
            hc_or,

        "high_conf_fisher_p":
            hc_p,

        "high_conf_high_collision_structure_gain":
            hc_high_stats[
                "delta_ACC"
            ],

        "high_conf_high_collision_net":
            hc_high_stats[
                "net"
            ],
    }

    (
        args.output
        / "e6_summary.json"
    ).write_text(
        json.dumps(
            summary,
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
        "E6 核心摘要"
    )
    print(
        "========================================"
    )

    print(
        "高置信度 + 最高碰撞20%:"
    )

    print(
        "  APK数:",
        hc_high_stats[
            "n"
        ],
    )

    print(
        "  第一层错误率:",
        f"{hc_high_stats['coarse_error_rate']:.4f}",
    )

    print(
        "  结构提升:",
        f"{hc_high_stats['delta_ACC']:+.4f}",
    )

    print(
        "  净修复:",
        f"{hc_high_stats['net']:+d}",
    )

    print()

    print(
        "高置信度 + 其他碰撞:"
    )

    print(
        "  APK数:",
        hc_other_stats[
            "n"
        ],
    )

    print(
        "  第一层错误率:",
        f"{hc_other_stats['coarse_error_rate']:.4f}",
    )

    print()

    print(
        "同为高置信度时的碰撞错误率增量:",
        f"{summary['high_conf_error_rate_lift']:+.4f}",
    )

    print(
        "Fisher OR:",
        hc_or,
    )

    print(
        "Fisher p:",
        hc_p,
    )

    print()
    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
