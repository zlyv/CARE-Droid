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
import run_benefitdroid_e5_semantic_count_collision_2017 as e5


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


def empirical_percentile(
    calibration,
    values,
):
    """
    仅利用 source calibration 分布，
    把新样本映射到经验百分位。

    对并列值采用 mid-rank：
      (# smaller + 0.5 * # equal) / n
    """
    calibration = np.asarray(
        calibration,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    ordered = np.sort(
        calibration
    )

    left = np.searchsorted(
        ordered,
        values,
        side="left",
    )

    right = np.searchsorted(
        ordered,
        values,
        side="right",
    )

    return (
        left
        + 0.5 * (right - left)
    ) / len(ordered)


def fold_compact_matrix(
    train_idx,
    test_idx,
    cohort_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
):
    train_rows = [
        cohort_rows[i]
        for i in train_idx
    ]

    test_rows = [
        cohort_rows[i]
        for i in test_idx
    ]

    (
        x_train,
        x_test,
        names,
    ) = p72.compact_matrix_allratio70_foldsafe(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    )

    if len(names) != 70:
        raise RuntimeError(
            f"Expected Compact70, got {len(names)}"
        )

    return (
        np.asarray(
            x_train,
            dtype=np.float64,
        ),
        np.asarray(
            x_test,
            dtype=np.float64,
        ),
    )


def threshold_for_top_fraction(
    source_scores,
    budget,
):
    source_scores = np.asarray(
        source_scores,
        dtype=np.float64,
    )

    n = len(
        source_scores
    )

    k = max(
        1,
        min(
            n,
            int(
                round(
                    n * budget
                )
            ),
        ),
    )

    ordered = np.sort(
        source_scores
    )[::-1]

    threshold = float(
        ordered[
            k - 1
        ]
    )

    source_selected_ratio = float(
        np.mean(
            source_scores
            >= threshold
        )
    )

    return (
        threshold,
        source_selected_ratio,
    )


def select_top_budget_deterministic(
    scores,
    folds,
    shas,
    budget,
):
    selected = []

    for fold in sorted(
        np.unique(
            folds
        )
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        k = max(
            1,
            min(
                len(idx),
                int(
                    round(
                        len(idx)
                        * budget
                    )
                ),
            ),
        )

        ordered = sorted(
            map(int, idx),
            key=lambda i: (
                -float(
                    scores[i]
                ),
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


def select_top_budget_random_ties(
    scores,
    folds,
    budget,
    seed,
):
    selected = []

    for fold in sorted(
        np.unique(
            folds
        )
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        k = max(
            1,
            min(
                len(idx),
                int(
                    round(
                        len(idx)
                        * budget
                    )
                ),
            ),
        )

        values = scores[
            idx
        ]

        ordered_values = np.sort(
            values
        )[::-1]

        cutoff = float(
            ordered_values[
                k - 1
            ]
        )

        higher_mask = (
            values
            > cutoff
        )

        tie_mask = np.isclose(
            values,
            cutoff,
            rtol=0.0,
            atol=1e-15,
        )

        higher = idx[
            higher_mask
        ]

        tied = idx[
            tie_mask
        ]

        need = (
            k
            - len(higher)
        )

        if need < 0:
            raise RuntimeError(
                "Tie selection logic error"
            )

        rng = np.random.default_rng(
            20260822
            + seed * 1009
            + int(fold) * 7919
        )

        if need > 0:
            if need > len(tied):
                raise RuntimeError(
                    "Not enough tied samples"
                )

            chosen_ties = rng.choice(
                tied,
                size=need,
                replace=False,
            )

            fold_selected = np.concatenate(
                [
                    higher,
                    chosen_ties,
                ]
            )
        else:
            fold_selected = higher

        if len(
            fold_selected
        ) != k:
            raise RuntimeError(
                "Wrong budget after tie handling"
            )

        selected.extend(
            map(
                int,
                fold_selected,
            )
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
        np.unique(
            folds
        )
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        k = max(
            1,
            min(
                len(idx),
                int(
                    round(
                        len(idx)
                        * budget
                    )
                ),
            ),
        )

        rng = np.random.default_rng(
            2026
            + seed * 1009
            + int(fold) * 9176
        )

        chosen = rng.choice(
            idx,
            size=k,
            replace=False,
        )

        selected.extend(
            map(
                int,
                chosen,
            )
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def evaluate(
    labels,
    coarse_probability,
    structure_probability,
    selected,
):
    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    final_probability = np.array(
        coarse_probability,
        copy=True,
    )

    final_probability[
        selected
    ] = structure_probability[
        selected
    ]

    coarse_pred = (
        coarse_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    final_pred = (
        final_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    fixed_mask = (
        (coarse_pred != labels)
        & (
            final_pred
            == labels
        )
    )

    broken_mask = (
        (coarse_pred == labels)
        & (
            final_pred
            != labels
        )
    )

    result = base.metrics(
        labels,
        final_probability,
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

    routed_n = int(
        len(
            selected
        )
    )

    return {
        "routed_n":
            routed_n,

        "routed_ratio":
            routed_n
            / len(labels),

        "ACC":
            result[
                "accuracy"
            ],

        "BA":
            result[
                "balanced_accuracy"
            ],

        "AUC":
            result[
                "auc"
            ],

        "F1":
            result[
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
                100.0
                * net
                / routed_n
                if routed_n
                else 0.0
            ),
    }


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
        "--k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--tie-repeats",
        type=int,
        default=200,
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
            "benefitdroid-e7b-strict-source-joint-gate-2017"
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

    shas = [
        norm_sha(
            row[
                "sha256"
            ]
        )
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

    coarse_probability = np.asarray(
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

    structure_probability = np.asarray(
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

    saved_collision = np.asarray(
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

    outer_uncertainty = (
        0.5
        - np.abs(
            coarse_probability
            - 0.5
        )
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E7b"
    )
    print(
        "严格 Source-only 联合歧义门控"
    )
    print(
        "========================================"
    )

    print(
        "APK数量:",
        len(rows),
    )

    print(
        "k:",
        args.k,
    )

    # =====================================================
    # Compact原始数据
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
            copied = dict(
                row
            )

            copied[
                "sha256"
            ] = sha

            compact_by_sha[
                sha
            ] = copied

    missing = [
        sha
        for sha in shas
        if sha not in compact_by_sha
    ]

    if missing:
        raise RuntimeError(
            "Missing compact APKs: "
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
        compact_schema = json.load(
            f
        )

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
            "Expected canonical raw Compact dim=76"
        )

    # =====================================================
    # 最终严格source-calibrated分数
    # =====================================================

    collision_percentile = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    uncertainty_percentile = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    joint_score = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    # 保存每个outer fold自己的source calibration score
    calibration_by_fold = {}

    fold_audit_rows = []

    max_collision_reproduction_diff = 0.0

    unique_folds = sorted(
        np.unique(
            folds
        )
    )

    for outer_fold in unique_folds:

        print()
        print(
            "========================================"
        )
        print(
            "外层折:",
            int(
                outer_fold
            ),
        )
        print(
            "========================================"
        )

        outer_train_idx = np.flatnonzero(
            folds
            != outer_fold
        )

        outer_test_idx = np.flatnonzero(
            folds
            == outer_fold
        )

        # -------------------------------------------------
        # A. 为outer train生成严格inner OOF：
        #    collision + uncertainty
        # -------------------------------------------------

        inner_collision = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        inner_uncertainty = np.full(
            len(rows),
            np.nan,
            dtype=np.float64,
        )

        inner_folds = [
            fold
            for fold in unique_folds
            if fold
            != outer_fold
        ]

        for inner_val_fold in inner_folds:

            inner_train_idx = np.flatnonzero(
                (folds != outer_fold)
                & (
                    folds
                    != inner_val_fold
                )
            )

            inner_val_idx = np.flatnonzero(
                (folds != outer_fold)
                & (
                    folds
                    == inner_val_fold
                )
            )

            print(
                "  内层验证折",
                int(
                    inner_val_fold
                ),
                "训练/验证=",
                len(
                    inner_train_idx
                ),
                "/",
                len(
                    inner_val_idx
                ),
            )

            (
                x_inner_train,
                x_inner_val,
            ) = fold_compact_matrix(
                inner_train_idx,
                inner_val_idx,
                cohort_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )

            # 内层OOF coarse probability
            (
                inner_prob,
                _,
                _,
            ) = p72.fit_predict(
                x_inner_train,
                labels[
                    inner_train_idx
                ],
                x_inner_val,
                seed=int(
                    inner_val_fold
                ) - 1,
            )

            inner_uncertainty[
                inner_val_idx
            ] = (
                0.5
                - np.abs(
                    inner_prob
                    - 0.5
                )
            )

            # 内层OOF collision
            collision_rows = e5.collision_scores(
                x_inner_train,
                labels[
                    inner_train_idx
                ],
                x_inner_val,
                args.k,
            )

            for local_i, global_i in enumerate(
                inner_val_idx
            ):
                inner_collision[
                    global_i
                ] = float(
                    collision_rows[
                        local_i
                    ][
                        "collision_score"
                    ]
                )

        if np.any(
            ~np.isfinite(
                inner_collision[
                    outer_train_idx
                ]
            )
        ):
            raise RuntimeError(
                "Incomplete inner collision OOF"
            )

        if np.any(
            ~np.isfinite(
                inner_uncertainty[
                    outer_train_idx
                ]
            )
        ):
            raise RuntimeError(
                "Incomplete inner uncertainty OOF"
            )

        calibration_collision = (
            inner_collision[
                outer_train_idx
            ]
        )

        calibration_uncertainty = (
            inner_uncertainty[
                outer_train_idx
            ]
        )

        # -------------------------------------------------
        # B. outer test collision重新计算
        # -------------------------------------------------

        (
            x_outer_train,
            x_outer_test,
        ) = fold_compact_matrix(
            outer_train_idx,
            outer_test_idx,
            cohort_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )

        outer_collision_rows = (
            e5.collision_scores(
                x_outer_train,
                labels[
                    outer_train_idx
                ],
                x_outer_test,
                args.k,
            )
        )

        outer_collision = np.asarray(
            [
                float(
                    row[
                        "collision_score"
                    ]
                )
                for row
                in outer_collision_rows
            ],
            dtype=np.float64,
        )

        reproduction_diff = float(
            np.max(
                np.abs(
                    outer_collision
                    - saved_collision[
                        outer_test_idx
                    ]
                )
            )
        )

        max_collision_reproduction_diff = max(
            max_collision_reproduction_diff,
            reproduction_diff,
        )

        print(
            "  E5 collision复现最大误差:",
            reproduction_diff,
        )

        if reproduction_diff > 1e-8:
            raise RuntimeError(
                "E5 collision reproduction mismatch"
            )

        # -------------------------------------------------
        # C. 只由outer-train的inner OOF分布标定百分位
        # -------------------------------------------------

        test_collision_percentile = (
            empirical_percentile(
                calibration_collision,
                outer_collision,
            )
        )

        test_uncertainty_percentile = (
            empirical_percentile(
                calibration_uncertainty,
                outer_uncertainty[
                    outer_test_idx
                ],
            )
        )

        test_joint = np.minimum(
            test_collision_percentile,
            test_uncertainty_percentile,
        )

        collision_percentile[
            outer_test_idx
        ] = test_collision_percentile

        uncertainty_percentile[
            outer_test_idx
        ] = test_uncertainty_percentile

        joint_score[
            outer_test_idx
        ] = test_joint

        # -------------------------------------------------
        # D. 产生source-only online threshold所需的
        #    calibration joint score
        # -------------------------------------------------

        calib_collision_percentile = (
            empirical_percentile(
                calibration_collision,
                calibration_collision,
            )
        )

        calib_uncertainty_percentile = (
            empirical_percentile(
                calibration_uncertainty,
                calibration_uncertainty,
            )
        )

        calib_joint = np.minimum(
            calib_collision_percentile,
            calib_uncertainty_percentile,
        )

        calibration_by_fold[
            int(
                outer_fold
            )
        ] = {
            "collision_percentile":
                calib_collision_percentile,

            "uncertainty_percentile":
                calib_uncertainty_percentile,

            "joint":
                calib_joint,
        }

        fold_audit_rows.append({
            "outer_fold":
                int(
                    outer_fold
                ),

            "outer_train_n":
                len(
                    outer_train_idx
                ),

            "outer_test_n":
                len(
                    outer_test_idx
                ),

            "collision_reproduction_max_diff":
                reproduction_diff,

            "source_collision_mean":
                float(
                    np.mean(
                        calibration_collision
                    )
                ),

            "source_uncertainty_mean":
                float(
                    np.mean(
                        calibration_uncertainty
                    )
                ),

            "test_collision_percentile_mean":
                float(
                    np.mean(
                        test_collision_percentile
                    )
                ),

            "test_uncertainty_percentile_mean":
                float(
                    np.mean(
                        test_uncertainty_percentile
                    )
                ),

            "test_joint_mean":
                float(
                    np.mean(
                        test_joint
                    )
                ),
        })

    print()
    print(
        "E5 collision总体复现最大误差:",
        max_collision_reproduction_diff,
    )

    # =====================================================
    # 保存严格分数
    # =====================================================

    score_rows = []

    for i, row in enumerate(
        rows
    ):
        score_rows.append({
            "sha256":
                shas[i],

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
                    coarse_probability[i]
                ),

            "raw_collision":
                float(
                    saved_collision[i]
                ),

            "source_calibrated_collision_percentile":
                float(
                    collision_percentile[i]
                ),

            "raw_uncertainty":
                float(
                    outer_uncertainty[i]
                ),

            "source_calibrated_uncertainty_percentile":
                float(
                    uncertainty_percentile[i]
                ),

            "joint_ambiguity":
                float(
                    joint_score[i]
                ),

            "structure_probability":
                float(
                    structure_probability[i]
                ),
        })

    write_csv(
        args.output
        / "e7b_scores.csv",
        score_rows,
    )

    write_csv(
        args.output
        / "e7b_fold_audit.csv",
        fold_audit_rows,
    )

    # =====================================================
    # Oracle
    # =====================================================

    coarse_pred = (
        coarse_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    structure_pred = (
        structure_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    oracle_score = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    oracle_score[
        (coarse_pred != labels)
        & (
            structure_pred
            == labels
        )
    ] = 1.0

    oracle_score[
        (coarse_pred == labels)
        & (
            structure_pred
            != labels
        )
    ] = -1.0

    budgets = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    # =====================================================
    # 实验1：严格source标定后的固定预算比较
    # =====================================================

    batch_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验1：严格Source标定后的固定预算"
    )
    print(
        "========================================"
    )

    score_map = {
        "低置信度":
            uncertainty_percentile,

        "仅碰撞":
            collision_percentile,

        "联合歧义":
            joint_score,

        "Oracle理论上界":
            oracle_score,
    }

    for budget in budgets:

        print()
        print(
            f"预算 {budget:.0%}:"
        )

        # Random baseline
        random_results = []

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
                coarse_probability,
                structure_probability,
                selected,
            )

            random_results.append(
                result
            )

        random_acc = float(
            np.mean(
                [
                    r["ACC"]
                    for r
                    in random_results
                ]
            )
        )

        random_net = float(
            np.mean(
                [
                    r["net"]
                    for r
                    in random_results
                ]
            )
        )

        batch_rows.append({
            "method":
                "随机",

            "budget":
                budget,

            "ACC":
                random_acc,

            "ACC_std":
                float(
                    np.std(
                        [
                            r["ACC"]
                            for r
                            in random_results
                        ]
                    )
                ),

            "net":
                random_net,

            "net_std":
                float(
                    np.std(
                        [
                            r["net"]
                            for r
                            in random_results
                        ]
                    )
                ),

            "routed_n":
                float(
                    np.mean(
                        [
                            r["routed_n"]
                            for r
                            in random_results
                        ]
                    )
                ),
        })

        print(
            "随机       ",
            f"ACC={random_acc:.6f}",
            f"净修复={random_net:+.2f}",
        )

        for method, score in score_map.items():

            selected = (
                select_top_budget_deterministic(
                    score,
                    folds,
                    shas,
                    budget,
                )
            )

            result = evaluate(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            batch_rows.append({
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

    write_csv(
        args.output
        / "e7b_batch_equal_budget.csv",
        batch_rows,
    )

    # =====================================================
    # 实验2：Joint并列敏感性
    # 重点看10/20/30%
    # =====================================================

    tie_raw_rows = []
    tie_summary_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验2：联合歧义边界并列稳定性"
    )
    print(
        "========================================"
    )

    for budget in (
        0.10,
        0.20,
        0.30,
    ):

        results = []

        for seed in range(
            args.tie_repeats
        ):
            selected = (
                select_top_budget_random_ties(
                    joint_score,
                    folds,
                    budget,
                    seed,
                )
            )

            result = evaluate(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            results.append(
                result
            )

            tie_raw_rows.append({
                "budget":
                    budget,

                "seed":
                    seed,

                **result,
            })

        acc_values = np.asarray(
            [
                r["ACC"]
                for r
                in results
            ],
            dtype=np.float64,
        )

        net_values = np.asarray(
            [
                r["net"]
                for r
                in results
            ],
            dtype=np.float64,
        )

        summary = {
            "budget":
                budget,

            "repeats":
                args.tie_repeats,

            "ACC_mean":
                float(
                    np.mean(
                        acc_values
                    )
                ),

            "ACC_std":
                float(
                    np.std(
                        acc_values
                    )
                ),

            "ACC_min":
                float(
                    np.min(
                        acc_values
                    )
                ),

            "ACC_max":
                float(
                    np.max(
                        acc_values
                    )
                ),

            "net_mean":
                float(
                    np.mean(
                        net_values
                    )
                ),

            "net_std":
                float(
                    np.std(
                        net_values
                    )
                ),

            "net_min":
                float(
                    np.min(
                        net_values
                    )
                ),

            "net_max":
                float(
                    np.max(
                        net_values
                    )
                ),
        }

        tie_summary_rows.append(
            summary
        )

        print()
        print(
            f"预算 {budget:.0%}:"
        )

        print(
            "  ACC mean/std =",
            f"{summary['ACC_mean']:.6f}",
            "/",
            f"{summary['ACC_std']:.6f}",
        )

        print(
            "  ACC min/max  =",
            f"{summary['ACC_min']:.6f}",
            "/",
            f"{summary['ACC_max']:.6f}",
        )

        print(
            "  net mean     =",
            f"{summary['net_mean']:+.2f}",
        )

        print(
            "  net min/max  =",
            f"{summary['net_min']:+.0f}",
            "/",
            f"{summary['net_max']:+.0f}",
        )

    write_csv(
        args.output
        / "e7b_joint_tie_runs.csv",
        tie_raw_rows,
    )

    write_csv(
        args.output
        / "e7b_joint_tie_summary.csv",
        tie_summary_rows,
    )

    # =====================================================
    # 实验3：真正online/source-threshold模式
    #
    # 每个outer fold都只根据outer-train inner OOF
    # 冻结阈值。
    #
    # 新APK单独到来即可判断，不需要目标批次排序。
    # =====================================================

    online_rows = []
    threshold_rows = []

    print()
    print(
        "========================================"
    )
    print(
        "实验3：Source冻结阈值在线门控"
    )
    print(
        "========================================"
    )

    online_score_maps = {
        "低置信度":
            uncertainty_percentile,

        "仅碰撞":
            collision_percentile,

        "联合歧义":
            joint_score,
    }

    calibration_key_map = {
        "低置信度":
            "uncertainty_percentile",

        "仅碰撞":
            "collision_percentile",

        "联合歧义":
            "joint",
    }

    for budget in budgets:

        print()
        print(
            f"Source名义预算 {budget:.0%}:"
        )

        for method, all_test_scores in (
            online_score_maps.items()
        ):

            selected_mask = np.zeros(
                len(rows),
                dtype=bool,
            )

            for outer_fold in unique_folds:

                fold = int(
                    outer_fold
                )

                test_idx = np.flatnonzero(
                    folds
                    == fold
                )

                source_scores = (
                    calibration_by_fold[
                        fold
                    ][
                        calibration_key_map[
                            method
                        ]
                    ]
                )

                (
                    source_threshold,
                    source_route_ratio,
                ) = threshold_for_top_fraction(
                    source_scores,
                    budget,
                )

                test_selected = (
                    all_test_scores[
                        test_idx
                    ]
                    >= source_threshold
                )

                selected_mask[
                    test_idx
                ] = test_selected

                threshold_rows.append({
                    "method":
                        method,

                    "nominal_budget":
                        budget,

                    "outer_fold":
                        fold,

                    "source_threshold":
                        source_threshold,

                    "source_actual_route_ratio":
                        source_route_ratio,

                    "test_routed_n":
                        int(
                            np.sum(
                                test_selected
                            )
                        ),

                    "test_n":
                        len(
                            test_idx
                        ),

                    "test_route_ratio":
                        float(
                            np.mean(
                                test_selected
                            )
                        ),
                })

            selected = np.flatnonzero(
                selected_mask
            )

            result = evaluate(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            online_rows.append({
                "method":
                    method,

                "nominal_source_budget":
                    budget,

                **result,
            })

            print(
                f"{method:12s}",
                f"实际分析={result['routed_n']:3d}",
                f"比例={result['routed_ratio']:.3f}",
                f"ACC={result['ACC']:.6f}",
                f"修复={result['fixed']:2d}",
                f"破坏={result['broken']:2d}",
                f"净修复={result['net']:+3d}",
            )

    write_csv(
        args.output
        / "e7b_online_source_threshold.csv",
        online_rows,
    )

    write_csv(
        args.output
        / "e7b_online_thresholds_by_fold.csv",
        threshold_rows,
    )

    # =====================================================
    # Reference
    # =====================================================

    base_result = evaluate(
        labels,
        coarse_probability,
        structure_probability,
        [],
    )

    full_result = evaluate(
        labels,
        coarse_probability,
        structure_probability,
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

    audit = {
        "source_apks":
            len(rows),

        "outer_folds":
            5,

        "inner_calibration":
            (
                "For each outer fold, collision and uncertainty "
                "calibration distributions are generated by "
                "inner OOF predictions using only the other "
                "four source folds."
            ),

        "test_fold_used_for_percentile_calibration":
            False,

        "collision":
            (
                "Compact70 kNN label entropy * local proximity"
            ),

        "k_neighbors":
            args.k,

        "uncertainty":
            (
                "0.5 - abs(p_coarse - 0.5)"
            ),

        "percentile":
            (
                "empirical source-only mid-rank percentile"
            ),

        "joint":
            (
                "min(source_collision_percentile, "
                "source_uncertainty_percentile)"
            ),

        "trainable_joint_parameters":
            0,

        "tie_repeats":
            args.tie_repeats,

        "random_repeats":
            args.random_repeats,

        "online_mode":
            (
                "Threshold frozen independently for each outer "
                "fold from outer-training inner-OOF scores. "
                "No target batch ranking is required."
            ),
    }

    (
        args.output
        / "e7b_audit.json"
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
        "E7b 完成"
    )
    print(
        "========================================"
    )

    print(
        "E5碰撞复现最大误差:",
        max_collision_reproduction_diff,
    )

    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
