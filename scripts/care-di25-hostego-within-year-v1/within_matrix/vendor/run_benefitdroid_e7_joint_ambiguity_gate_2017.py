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


THRESHOLD = 0.5


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


def percentile_rank(values, folds):
    """
    每个外层测试折内部只根据分数排名。
    不使用标签。

    越大的原始值 -> 越大的百分位。
    """

    result = np.zeros(
        len(values),
        dtype=np.float64,
    )

    try:
        from scipy.stats import rankdata
    except Exception as exc:
        raise RuntimeError(
            "SciPy is required for tie-aware percentile ranks"
        ) from exc

    for fold in sorted(
        np.unique(folds)
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        local = values[
            idx
        ]

        ranks = rankdata(
            local,
            method="average",
        )

        result[
            idx
        ] = ranks / len(idx)

    return result


def select_top_budget(
    score,
    folds,
    shas,
    budget,
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
        dtype=np.int64,
    )


def select_random(
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
        dtype=np.int64,
    )


def evaluate(
    labels,
    coarse_prob,
    structure_prob,
    selected,
):
    final_prob = np.array(
        coarse_prob,
        copy=True,
    )

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    final_prob[
        selected
    ] = structure_prob[
        selected
    ]

    coarse_pred = (
        coarse_prob
        >= THRESHOLD
    ).astype(int)

    final_pred = (
        final_prob
        >= THRESHOLD
    ).astype(int)

    fixed_mask = (
        (coarse_pred != labels)
        & (final_pred == labels)
    )

    broken_mask = (
        (coarse_pred == labels)
        & (final_pred != labels)
    )

    metrics = base.metrics(
        labels,
        final_prob,
        threshold=THRESHOLD,
    )

    fixed = int(
        np.sum(
            fixed_mask
        )
    )

    broken = int(
        np.sum(
            broken_mask
        )
    )

    net = (
        fixed
        - broken
    )

    routed_n = len(
        selected
    )

    return {
        "routed_n":
            routed_n,

        "routed_ratio":
            routed_n
            / len(labels),

        "ACC":
            metrics[
                "accuracy"
            ],

        "BA":
            metrics[
                "balanced_accuracy"
            ],

        "AUC":
            metrics[
                "auc"
            ],

        "F1":
            metrics[
                "f1"
            ],

        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            net,

        "net_per_100_routed":
            (
                net
                / routed_n
                * 100.0
                if routed_n
                else 0.0
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e5-semantic-count-collision-2017/"
            "e5_collision_scores.csv"
        ),
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
            "benefitdroid-e7-joint-ambiguity-gate-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_csv(
        args.scores
    )

    if len(rows) != 799:
        raise RuntimeError(
            f"Expected 799 rows, got {len(rows)}"
        )

    shas = [
        row["sha256"]
        for row in rows
    ]

    labels = np.asarray(
        [
            int(
                row[
                    "label"
                ]
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    folds = np.asarray(
        [
            int(
                row[
                    "fold"
                ]
            )
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

    uncertainty = (
        0.5
        - np.abs(
            coarse_prob
            - 0.5
        )
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E7"
    )
    print(
        "碰撞 × 不确定性 联合歧义门控"
    )
    print(
        "========================================"
    )

    # =====================================================
    # 两个可观察信号转换为同尺度百分位
    # =====================================================

    collision_rank = percentile_rank(
        collision,
        folds,
    )

    uncertainty_rank = percentile_rank(
        uncertainty,
        folds,
    )

    # -----------------------------------------------------
    # 核心联合分数：
    # 两个都高才高。
    # 无权重、无学习参数。
    # -----------------------------------------------------

    joint_score = np.minimum(
        collision_rank,
        uncertainty_rank,
    )

    # -----------------------------------------------------
    # E6直接观察到的 Top20 ∩ Top20 规则
    # 只作为机制复现实验。
    # -----------------------------------------------------

    intersection_mask = (
        (collision_rank >= 0.80)
        & (
            uncertainty_rank
            >= 0.80
        )
    )

    intersection_idx = np.flatnonzero(
        intersection_mask
    )

    intersection_result = evaluate(
        labels,
        coarse_prob,
        structure_prob,
        intersection_idx,
    )

    print()
    print(
        "===== Top20碰撞 ∩ Top20不确定 ====="
    )

    print(
        "分析APK:",
        intersection_result[
            "routed_n"
        ],
    )

    print(
        "比例:",
        f"{intersection_result['routed_ratio']:.4f}",
    )

    print(
        "ACC:",
        f"{intersection_result['ACC']:.6f}",
    )

    print(
        "修复:",
        intersection_result[
            "fixed"
        ],
    )

    print(
        "破坏:",
        intersection_result[
            "broken"
        ],
    )

    print(
        "净修复:",
        intersection_result[
            "net"
        ],
    )

    # =====================================================
    # Oracle，仅理论上界
    # =====================================================

    coarse_pred = (
        coarse_prob
        >= THRESHOLD
    ).astype(int)

    structure_pred = (
        structure_prob
        >= THRESHOLD
    ).astype(int)

    oracle_score = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    oracle_score[
        (coarse_pred != labels)
        & (structure_pred == labels)
    ] = 1.0

    oracle_score[
        (coarse_pred == labels)
        & (structure_pred != labels)
    ] = -1.0

    # =====================================================
    # 等预算比较
    # =====================================================

    budgets = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    summary_rows = []
    random_rows = []

    deterministic_scores = {
        "低置信度":
            uncertainty_rank,

        "仅碰撞":
            collision_rank,

        "联合歧义":
            joint_score,

        "Oracle理论上界":
            oracle_score,
    }

    print()
    print(
        "========================================"
    )
    print(
        "等预算比较"
    )
    print(
        "========================================"
    )

    for budget in budgets:

        print()
        print(
            f"预算 {budget:.0%}:"
        )

        # Random
        run_results = []

        for seed in range(
            args.random_repeats
        ):
            selected = select_random(
                folds,
                budget,
                seed,
            )

            result = evaluate(
                labels,
                coarse_prob,
                structure_prob,
                selected,
            )

            run_results.append(
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
            "routed_n",
            "routed_ratio",
            "ACC",
            "BA",
            "AUC",
            "F1",
            "fixed",
            "broken",
            "net",
            "net_per_100_routed",
        ):
            vals = np.asarray(
                [
                    r[key]
                    for r in run_results
                ],
                dtype=float,
            )

            random_summary[
                key
            ] = float(
                np.mean(vals)
            )

            random_summary[
                key + "_std"
            ] = float(
                np.std(vals)
            )

        summary_rows.append(
            random_summary
        )

        print(
            "随机       ",
            f"ACC={random_summary['ACC']:.6f}",
            f"净修复={random_summary['net']:+.2f}",
        )

        # Deterministic
        for method, score in (
            deterministic_scores.items()
        ):
            selected = select_top_budget(
                score,
                folds,
                shas,
                budget,
            )

            result = evaluate(
                labels,
                coarse_prob,
                structure_prob,
                selected,
            )

            summary_rows.append({
                "method":
                    method,

                "budget":
                    budget,

                **result,
            })

            print(
                f"{method:12s}",
                f"ACC={result['ACC']:.6f}",
                f"修复={result['fixed']:2d}",
                f"破坏={result['broken']:2d}",
                f"净修复={result['net']:+3d}",
                f"每100个={result['net_per_100_routed']:.2f}",
            )

    # =====================================================
    # 参考点
    # =====================================================

    base_result = evaluate(
        labels,
        coarse_prob,
        structure_prob,
        [],
    )

    full_result = evaluate(
        labels,
        coarse_prob,
        structure_prob,
        np.arange(
            len(rows)
        ),
    )

    print()
    print(
        "========================================"
    )
    print(
        "参考"
    )
    print(
        "========================================"
    )

    print(
        "仅第一层:",
        f"ACC={base_result['ACC']:.6f}",
    )

    print(
        "全部结构:",
        f"ACC={full_result['ACC']:.6f}",
        f"净修复={full_result['net']:+d}",
    )

    # =====================================================
    # 保存每个APK分数
    # =====================================================

    score_rows = []

    for i, row in enumerate(
        rows
    ):
        score_rows.append({
            "sha256":
                row[
                    "sha256"
                ],

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            "fold":
                int(
                    row[
                        "fold"
                    ]
                ),

            "coarse_probability":
                float(
                    coarse_prob[i]
                ),

            "collision_score":
                float(
                    collision[i]
                ),

            "collision_percentile":
                float(
                    collision_rank[i]
                ),

            "uncertainty":
                float(
                    uncertainty[i]
                ),

            "uncertainty_percentile":
                float(
                    uncertainty_rank[i]
                ),

            "joint_ambiguity_score":
                float(
                    joint_score[i]
                ),

            "top20_intersection":
                int(
                    intersection_mask[i]
                ),
        })

    write_csv(
        args.output
        / "e7_scores.csv",
        score_rows,
    )

    write_csv(
        args.output
        / "e7_equal_budget_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "e7_random_runs.csv",
        random_rows,
    )

    write_csv(
        args.output
        / "e7_top20_intersection.csv",
        [
            {
                "method":
                    "Top20碰撞_交_Top20不确定",

                **intersection_result,
            }
        ],
    )

    audit = {
        "source_apks":
            len(rows),

        "collision_input":
            (
                "E5 source-only OOF coarse-representation "
                "collision score"
            ),

        "uncertainty":
            (
                "0.5 - abs(p_coarse - 0.5)"
            ),

        "percentile":
            (
                "fold-wise unlabeled percentile rank"
            ),

        "joint_score":
            (
                "min(collision_percentile, "
                "uncertainty_percentile)"
            ),

        "joint_score_trainable_parameters":
            0,

        "test_labels_used_in_joint_score":
            False,

        "budgets":
            budgets,

        "random_repeats":
            args.random_repeats,
    }

    (
        args.output
        / "e7_audit.json"
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
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
