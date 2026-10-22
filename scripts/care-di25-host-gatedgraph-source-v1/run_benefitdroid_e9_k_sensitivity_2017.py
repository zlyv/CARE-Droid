#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72
import run_benefitdroid_e5_semantic_count_collision_2017 as e5
import run_benefitdroid_e7b_strict_source_joint_gate_2017 as e7b


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


def read_roles(path):
    with Path(path).open(
        encoding="utf-8",
    ) as f:
        obj = yaml.safe_load(f)

    roles = list(
        obj["roles"]
    )

    if len(roles) != 31:
        raise RuntimeError(
            f"Expected 31 roles, got {len(roles)}"
        )

    return roles


def build_fold(
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

    names = list(names)

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
        names,
    )


def feature_groups(
    names,
    roles,
):
    """
    语义维度必须能够从最终70维特征名中
    映射到 semantic_mapping_v1 的角色名称。

    当前协议预期：
      SemanticOnly = 46
      NonSemantic  = 24

    如果不是这个结果，主动停止，不猜。
    """

    semantic = []

    role_set = {
        role.casefold()
        for role in roles
    }

    for i, name in enumerate(names):
        lower = str(
            name
        ).casefold()

        # E8中的“语义计数空间”严格限定为
        # role_node__* 与 role_edge__*。
        #
        # 例如 context_role__boot_persistence
        # 虽然名称包含语义角色，但不是这46维
        # role-count表示的一部分，因此归入补集。
        is_role_count = (
            lower.startswith("role_node__")
            or lower.startswith("role_edge__")
        )

        if is_role_count:
            if "__" not in lower:
                raise RuntimeError(
                    f"Malformed semantic feature name: {name}"
                )

            suffix = lower.split(
                "__",
                1,
            )[1]

            if suffix not in role_set:
                raise RuntimeError(
                    "Role-count feature does not map to "
                    f"semantic_mapping_v1 role: {name}"
                )

            semantic.append(i)

    semantic_set = set(
        semantic
    )

    nonsemantic = [
        i
        for i in range(
            len(names)
        )
        if i not in semantic_set
    ]

    if (
        len(semantic) != 46
        or len(nonsemantic) != 24
    ):
        print()
        print(
            "===== 最终70维特征名 ====="
        )

        for i, name in enumerate(
            names
        ):
            print(
                f"{i:02d}",
                name,
                "SEM"
                if i in semantic
                else "NONSEM",
            )

        raise RuntimeError(
            "Expected SemanticOnly=46 and "
            "NonSemantic=24. "
            f"Got {len(semantic)} / {len(nonsemantic)}. "
            "Do not guess feature grouping."
        )

    return {
        "Full70":
            np.arange(
                70,
                dtype=np.int64,
            ),

        "SemanticOnly":
            np.asarray(
                semantic,
                dtype=np.int64,
            ),

        "NonSemantic":
            np.asarray(
                nonsemantic,
                dtype=np.int64,
            ),
    }


def collision_array(
    x_train,
    y_train,
    x_test,
    columns,
    k,
):
    rows = e5.collision_scores(
        x_train[
            :,
            columns
        ],
        y_train,
        x_test[
            :,
            columns
        ],
        k,
    )

    return np.asarray(
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


def foldwise_quintiles(
    score,
    folds,
    shas,
):
    """
    每折内部等人数5组。
    score相同时只用SHA作确定性打破并列，
    不使用标签。
    """

    result = np.zeros(
        len(score),
        dtype=np.int64,
    )

    for fold in sorted(
        np.unique(folds)
    ):
        idx = np.flatnonzero(
            folds == fold
        )

        ordered = sorted(
            map(int, idx),
            key=lambda i: (
                float(
                    score[i]
                ),
                shas[i],
            ),
        )

        chunks = np.array_split(
            np.asarray(
                ordered,
                dtype=np.int64,
            ),
            5,
        )

        for q, chunk in enumerate(
            chunks,
            start=1,
        ):
            result[
                chunk
            ] = q

    if np.any(
        result == 0
    ):
        raise RuntimeError(
            "Incomplete quintile assignment"
        )

    return result


def group_stats(
    idx,
    labels,
    coarse_probability,
    structure_probability,
):
    idx = np.asarray(
        idx,
        dtype=np.int64,
    )

    if len(idx) == 0:
        return {
            "n": 0,
            "error_count": 0,
            "error_rate": float("nan"),
            "coarse_ACC": float("nan"),
            "structure_ACC": float("nan"),
            "delta_ACC": float("nan"),
            "fixed": 0,
            "broken": 0,
            "net": 0,
        }

    y = labels[
        idx
    ]

    coarse_pred = (
        coarse_probability[
            idx
        ]
        >= THRESHOLD
    ).astype(
        np.int64
    )

    structure_pred = (
        structure_probability[
            idx
        ]
        >= THRESHOLD
    ).astype(
        np.int64
    )

    error = (
        coarse_pred
        != y
    )

    fixed = (
        error
        & (
            structure_pred
            == y
        )
    )

    broken = (
        (~error)
        & (
            structure_pred
            != y
        )
    )

    coarse_acc = float(
        np.mean(
            coarse_pred
            == y
        )
    )

    structure_acc = float(
        np.mean(
            structure_pred
            == y
        )
    )

    return {
        "n":
            int(
                len(idx)
            ),

        "error_count":
            int(
                np.sum(
                    error
                )
            ),

        "error_rate":
            float(
                np.mean(
                    error
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e7b-scores",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e7b-strict-source-joint-gate-2017/"
            "e7b_scores.csv"
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
        "--semantic-mapping",
        type=Path,
        default=Path(
            "configs/"
            "protocol/"
            "semantic_mapping_v1.yaml"
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e8-collision-source-ablation-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # E7b严格Source-only结果
    # =====================================================

    rows = read_csv(
        args.e7b_scores
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

    saved_full_collision = np.asarray(
        [
            float(
                row[
                    "raw_collision"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    uncertainty_percentile = np.asarray(
        [
            float(
                row[
                    "source_calibrated_uncertainty_percentile"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E9"
    )
    print(
        "碰撞来源消融"
    )
    print(
        "Full70 vs SemanticOnly vs NonSemantic"
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

    print(
        "联合门控预算:",
        args.budget,
    )

    # =====================================================
    # Compact数据
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
            "Expected canonical raw Compact=76"
        )

    roles = read_roles(
        args.semantic_mapping
    )

    # =====================================================
    # 三种空间的source-calibrated collision
    # =====================================================

    space_names = [
        "Full70",
        "SemanticOnly",
        "NonSemantic",
    ]

    raw_collision = {
        name:
            np.zeros(
                len(rows),
                dtype=np.float64,
            )
        for name in space_names
    }

    collision_percentile = {
        name:
            np.zeros(
                len(rows),
                dtype=np.float64,
            )
        for name in space_names
    }

    reference_feature_names = None
    reference_groups = None

    max_full_reproduction_diff = 0.0

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
        # 为outer train生成三个空间的inner OOF collision
        # -------------------------------------------------

        inner_collision = {
            name:
                np.full(
                    len(rows),
                    np.nan,
                    dtype=np.float64,
                )
            for name in space_names
        }

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
                x_train,
                x_val,
                feature_names,
            ) = build_fold(
                inner_train_idx,
                inner_val_idx,
                cohort_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )

            groups = feature_groups(
                feature_names,
                roles,
            )

            if reference_feature_names is None:
                reference_feature_names = list(
                    feature_names
                )

                reference_groups = {
                    key:
                        np.asarray(
                            value,
                            dtype=np.int64,
                        )
                    for key, value
                    in groups.items()
                }
            else:
                if (
                    list(
                        feature_names
                    )
                    != reference_feature_names
                ):
                    raise RuntimeError(
                        "Final Compact70 feature names "
                        "changed across folds"
                    )

            for space in space_names:
                values = collision_array(
                    x_train,
                    labels[
                        inner_train_idx
                    ],
                    x_val,
                    groups[
                        space
                    ],
                    args.k,
                )

                inner_collision[
                    space
                ][
                    inner_val_idx
                ] = values

        # -------------------------------------------------
        # Outer test collision
        # -------------------------------------------------

        (
            x_outer_train,
            x_outer_test,
            outer_feature_names,
        ) = build_fold(
            outer_train_idx,
            outer_test_idx,
            cohort_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )

        outer_groups = feature_groups(
            outer_feature_names,
            roles,
        )

        if (
            list(
                outer_feature_names
            )
            != reference_feature_names
        ):
            raise RuntimeError(
                "Outer Compact70 names mismatch"
            )

        for space in space_names:

            calibration = inner_collision[
                space
            ][
                outer_train_idx
            ]

            if np.any(
                ~np.isfinite(
                    calibration
                )
            ):
                raise RuntimeError(
                    f"Incomplete inner collision: {space}"
                )

            test_raw = collision_array(
                x_outer_train,
                labels[
                    outer_train_idx
                ],
                x_outer_test,
                outer_groups[
                    space
                ],
                args.k,
            )

            raw_collision[
                space
            ][
                outer_test_idx
            ] = test_raw

            collision_percentile[
                space
            ][
                outer_test_idx
            ] = e7b.empirical_percentile(
                calibration,
                test_raw,
            )

        if args.k == 15:
            diff = float(
                np.max(
                    np.abs(
                        raw_collision[
                            "Full70"
                        ][
                            outer_test_idx
                        ]
                        - saved_full_collision[
                            outer_test_idx
                        ]
                    )
                )
            )

            max_full_reproduction_diff = max(
                max_full_reproduction_diff,
                diff,
            )

            print(
                "  Full70 E5/E7b复现最大误差:",
                diff,
            )

            if diff > 1e-8:
                raise RuntimeError(
                    "Full70 collision reproduction mismatch"
                )
        else:
            print(
                "  当前邻居数量 k=",
                args.k,
                "，跳过与旧 k=15 碰撞分数的复现检查",
            )

    print()
    print(
        "Full70总体复现最大误差:",
        max_full_reproduction_diff,
    )

    # =====================================================
    # 保存最终特征分组
    # =====================================================

    feature_rows = []

    semantic_set = set(
        map(
            int,
            reference_groups[
                "SemanticOnly"
            ],
        )
    )

    for i, name in enumerate(
        reference_feature_names
    ):
        feature_rows.append({
            "feature_index":
                i,

            "feature_name":
                name,

            "group":
                (
                    "SemanticOnly"
                    if i
                    in semantic_set
                    else "NonSemantic"
                ),
        })

    write_csv(
        args.output
        / "e8_feature_groups.csv",
        feature_rows,
    )

    print()
    print(
        "最终70维分组:"
    )

    print(
        "  SemanticOnly:",
        len(
            reference_groups[
                "SemanticOnly"
            ]
        ),
    )

    print(
        "  NonSemantic:",
        len(
            reference_groups[
                "NonSemantic"
            ]
        ),
    )

    # =====================================================
    # 每种碰撞空间：
    #   1. 五分位错误富集
    #   2. 五分位结构增益
    #   3. 与uncertainty相关性
    #   4. 20%碰撞路由
    #   5. 20%联合门控
    # =====================================================

    quintile_rows = []
    summary_rows = []
    score_rows = []

    total_coarse_errors = int(
        np.sum(
            (
                coarse_probability
                >= THRESHOLD
            ).astype(
                np.int64
            )
            != labels
        )
    )

    overall_structure = e7b.evaluate(
        labels,
        coarse_probability,
        structure_probability,
        np.arange(
            len(rows)
        ),
    )

    low_conf_selected = (
        e7b.select_top_budget_deterministic(
            uncertainty_percentile,
            folds,
            shas,
            args.budget,
        )
    )

    low_conf_result = e7b.evaluate(
        labels,
        coarse_probability,
        structure_probability,
        low_conf_selected,
    )

    print()
    print(
        "========================================"
    )
    print(
        "E9 邻居数量鲁棒性：碰撞来源对比"
    )
    print(
        "========================================"
    )

    print()
    print(
        "20%低置信度参考:"
    )

    print(
        f"  ACC={low_conf_result['ACC']:.6f}",
        f"修复={low_conf_result['fixed']}",
        f"破坏={low_conf_result['broken']}",
        f"净修复={low_conf_result['net']:+d}",
    )

    for space in space_names:

        c_score = collision_percentile[
            space
        ]

        joint_score = np.minimum(
            c_score,
            uncertainty_percentile,
        )

        quintiles = foldwise_quintiles(
            c_score,
            folds,
            shas,
        )

        space_q_rows = []

        for q in range(
            1,
            6,
        ):
            idx = np.flatnonzero(
                quintiles == q
            )

            stats = group_stats(
                idx,
                labels,
                coarse_probability,
                structure_probability,
            )

            row = {
                "space":
                    space,

                "collision_quintile":
                    q,

                "n":
                    stats[
                        "n"
                    ],

                "mean_collision_percentile":
                    float(
                        np.mean(
                            c_score[
                                idx
                            ]
                        )
                    ),

                **stats,
            }

            quintile_rows.append(
                row
            )

            space_q_rows.append(
                row
            )

        lowest = space_q_rows[
            0
        ]

        highest = space_q_rows[
            -1
        ]

        try:
            from scipy.stats import spearmanr

            rho, rho_p = spearmanr(
                c_score,
                uncertainty_percentile,
            )

            rho = float(rho)
            rho_p = float(
                rho_p
            )

        except Exception:
            rho = float(
                "nan"
            )

            rho_p = float(
                "nan"
            )

        collision_selected = (
            e7b.select_top_budget_deterministic(
                c_score,
                folds,
                shas,
                args.budget,
            )
        )

        collision_result = e7b.evaluate(
            labels,
            coarse_probability,
            structure_probability,
            collision_selected,
        )

        joint_selected = (
            e7b.select_top_budget_deterministic(
                joint_score,
                folds,
                shas,
                args.budget,
            )
        )

        joint_result = e7b.evaluate(
            labels,
            coarse_probability,
            structure_probability,
            joint_selected,
        )

        error_share = (
            highest[
                "error_count"
            ]
            / total_coarse_errors
        )

        net_share = (
            highest[
                "net"
            ]
            / overall_structure[
                "net"
            ]
            if overall_structure[
                "net"
            ] != 0
            else float(
                "nan"
            )
        )

        summary = {
            "space":
                space,

            "feature_dim":
                len(
                    reference_groups[
                        space
                    ]
                ),

            "lowest20_error_rate":
                lowest[
                    "error_rate"
                ],

            "highest20_error_rate":
                highest[
                    "error_rate"
                ],

            "error_rate_lift_high_minus_low":
                highest[
                    "error_rate"
                ]
                - lowest[
                    "error_rate"
                ],

            "highest20_error_count":
                highest[
                    "error_count"
                ],

            "highest20_error_share_of_all_errors":
                error_share,

            "lowest20_structure_gain":
                lowest[
                    "delta_ACC"
                ],

            "highest20_structure_gain":
                highest[
                    "delta_ACC"
                ],

            "highest20_fixed":
                highest[
                    "fixed"
                ],

            "highest20_broken":
                highest[
                    "broken"
                ],

            "highest20_net":
                highest[
                    "net"
                ],

            "highest20_net_share_of_overall_structure_net":
                net_share,

            "collision_uncertainty_spearman_rho":
                rho,

            "collision_uncertainty_spearman_p":
                rho_p,

            "collision_only_20_ACC":
                collision_result[
                    "ACC"
                ],

            "collision_only_20_fixed":
                collision_result[
                    "fixed"
                ],

            "collision_only_20_broken":
                collision_result[
                    "broken"
                ],

            "collision_only_20_net":
                collision_result[
                    "net"
                ],

            "joint_20_ACC":
                joint_result[
                    "ACC"
                ],

            "joint_20_fixed":
                joint_result[
                    "fixed"
                ],

            "joint_20_broken":
                joint_result[
                    "broken"
                ],

            "joint_20_net":
                joint_result[
                    "net"
                ],
        }

        summary_rows.append(
            summary
        )

        print()
        print(
            "----------------------------------------"
        )
        print(
            space,
            f"({len(reference_groups[space])}维)",
        )
        print(
            "----------------------------------------"
        )

        print(
            "最低20%错误率:",
            f"{lowest['error_rate']:.4f}",
        )

        print(
            "最高20%错误率:",
            f"{highest['error_rate']:.4f}",
        )

        print(
            "错误率增量:",
            f"{summary['error_rate_lift_high_minus_low']:+.4f}",
        )

        print(
            "最高20%包含第一层错误:",
            f"{highest['error_count']}/{total_coarse_errors}",
            f"({100.0 * error_share:.1f}%)",
        )

        print(
            "最高20%结构提升:",
            f"{highest['delta_ACC']:+.4f}",
        )

        print(
            "最高20%修复/破坏/净:",
            highest[
                "fixed"
            ],
            "/",
            highest[
                "broken"
            ],
            "/",
            f"{highest['net']:+d}",
        )

        print(
            "碰撞 vs 不确定性 rho:",
            f"{rho:.4f}",
        )

        print(
            "20%仅碰撞:",
            f"ACC={collision_result['ACC']:.6f}",
            f"净={collision_result['net']:+d}",
        )

        print(
            "20%联合门控:",
            f"ACC={joint_result['ACC']:.6f}",
            f"修复={joint_result['fixed']}",
            f"破坏={joint_result['broken']}",
            f"净={joint_result['net']:+d}",
        )

        for i in range(
            len(rows)
        ):
            score_rows.append({
                "sha256":
                    shas[i],

                "fold":
                    int(
                        folds[i]
                    ),

                "label":
                    int(
                        labels[i]
                    ),

                "space":
                    space,

                "raw_collision":
                    float(
                        raw_collision[
                            space
                        ][i]
                    ),

                "source_collision_percentile":
                    float(
                        c_score[i]
                    ),

                "source_uncertainty_percentile":
                    float(
                        uncertainty_percentile[i]
                    ),

                "joint_score":
                    float(
                        joint_score[i]
                    ),

                "collision_quintile":
                    int(
                        quintiles[i]
                    ),
            })

    # =====================================================
    # 保存
    # =====================================================

    write_csv(
        args.output
        / "e8_quintiles.csv",
        quintile_rows,
    )

    write_csv(
        args.output
        / "e8_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "e8_scores_long.csv",
        score_rows,
    )

    reference_rows = [
        {
            "method":
                "第一层",

            "ACC":
                e7b.evaluate(
                    labels,
                    coarse_probability,
                    structure_probability,
                    [],
                )[
                    "ACC"
                ],
        },

        {
            "method":
                "全部结构",

            "ACC":
                overall_structure[
                    "ACC"
                ],

            "fixed":
                overall_structure[
                    "fixed"
                ],

            "broken":
                overall_structure[
                    "broken"
                ],

            "net":
                overall_structure[
                    "net"
                ],
        },

        {
            "method":
                "20%低置信度",

            "ACC":
                low_conf_result[
                    "ACC"
                ],

            "fixed":
                low_conf_result[
                    "fixed"
                ],

            "broken":
                low_conf_result[
                    "broken"
                ],

            "net":
                low_conf_result[
                    "net"
                ],
        },
    ]

    write_csv(
        args.output
        / "e8_reference.csv",
        reference_rows,
    )

    audit = {
        "source_apks":
            len(rows),

        "outer_folds":
            5,

        "k_neighbors":
            args.k,

        "joint_budget":
            args.budget,

        "spaces": {
            "Full70":
                70,

            "SemanticOnly":
                len(
                    reference_groups[
                        "SemanticOnly"
                    ]
                ),

            "NonSemantic":
                len(
                    reference_groups[
                        "NonSemantic"
                    ]
                ),
        },

        "semantic_feature_definition":
            (
                "Final Compact70 feature names containing one "
                "of the 31 semantic role identifiers from "
                "semantic_mapping_v1.yaml. "
                "Expected exactly 46 dimensions."
            ),

        "nonsemantic_feature_definition":
            (
                "Complement of SemanticOnly in final Compact70. "
                "Expected exactly 24 dimensions."
            ),

        "collision_protocol":
            (
                "Same kNN local label entropy * proximity "
                "definition as E5/E7b."
            ),

        "collision_percentile_calibration":
            (
                "Strict outer-training inner-OOF source-only "
                "calibration, same principle as E7b."
            ),

        "uncertainty_percentile":
            (
                "Reused strict source-only calibrated "
                "uncertainty percentile from E7b."
            ),

        "joint_gate":
            (
                "min(source_collision_percentile, "
                "source_uncertainty_percentile)"
            ),

        "future_years_used":
            False,
    }

    (
        args.output
        / "e8_audit.json"
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
        "E9 单个k最终对比"
    )
    print(
        "========================================"
    )

    print(
        "空间                 "
        "高20错误率   高20结构提升   高20净修复   "
        "20%联合ACC   联合净修复"
    )

    for row in summary_rows:
        print(
            f"{row['space']:18s}",
            f"{row['highest20_error_rate']:.4f}",
            f"{row['highest20_structure_gain']:+.4f}",
            f"{row['highest20_net']:+4d}",
            f"{row['joint_20_ACC']:.6f}",
            f"{row['joint_20_net']:+4d}",
        )

    print()
    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
