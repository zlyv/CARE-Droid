#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_compact_quantity_shared_caller_2x3_2017_eval as exp
import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72
import run_benefitdroid_e1_exact_pair_2017 as e1
import run_benefitdroid_e5_semantic_count_collision_2017 as e5
import run_benefitdroid_e7b_strict_source_joint_gate_2017 as e7b


THRESHOLD = 0.5
K_NEIGHBORS = 15


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
    path.parent.mkdir(parents=True, exist_ok=True)

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


def label_of(row):
    value = str(
        row.get(
            "metadata_label",
            "",
        )
    ).strip().casefold()

    if value == "malicious":
        return 1

    if value == "benign":
        return 0

    raise RuntimeError(
        f"Unknown label: {value}"
    )


def load_roles(path):
    with Path(path).open(
        encoding="utf-8",
    ) as f:
        obj = yaml.safe_load(f)

    roles = list(obj["roles"])

    if len(roles) != 31:
        raise RuntimeError(
            f"Expected 31 roles, got {len(roles)}"
        )

    return roles


def semantic_indices(names):
    idx = [
        i
        for i, name in enumerate(names)
        if str(name).casefold().startswith(
            "role_node__"
        )
        or str(name).casefold().startswith(
            "role_edge__"
        )
    ]

    if len(idx) != 46:
        raise RuntimeError(
            f"Expected 46 role-count dims, got {len(idx)}"
        )

    return np.asarray(
        idx,
        dtype=np.int64,
    )


def reconstruct_structure(
    rows,
    cache,
    role_count,
):
    pair_list = list(
        combinations(
            range(role_count),
            2,
        )
    )

    pair_index = {
        pair: i
        for i, pair in enumerate(pair_list)
    }

    features = []

    for i, row in enumerate(
        rows,
        start=1,
    ):
        sha = norm_sha(
            row["sha256"]
        )

        if sha not in cache:
            raise RuntimeError(
                f"Missing cache: {sha}"
            )

        (
            stats,
            pair_raw,
            _,
            _,
        ) = e1.reconstruct_host_structure(
            cache[sha]["payload"],
            pair_list,
            pair_index,
        )

        features.append(
            np.concatenate(
                [
                    np.asarray(
                        stats,
                        dtype=float,
                    ),
                    np.asarray(
                        pair_raw,
                        dtype=float,
                    ),
                ]
            )
        )

        if i % 200 == 0 or i == len(rows):
            print(
                "  恢复结构:",
                i,
                "/",
                len(rows),
            )

    matrix = np.asarray(
        features,
        dtype=np.float64,
    )

    if matrix.shape[1] != 585:
        raise RuntimeError(
            f"Expected structure=585, got {matrix.shape}"
        )

    names = (
        [
            f"shared_stats__{i:03d}"
            for i in range(120)
        ]
        +
        [
            f"pair_raw__{i:03d}"
            for i in range(465)
        ]
    )

    return matrix, names


def evaluate(
    labels,
    coarse_prob,
    structure_prob,
    route_mask,
):
    final_prob = np.where(
        route_mask,
        structure_prob,
        coarse_prob,
    )

    result = base.metrics(
        labels,
        final_prob,
        threshold=THRESHOLD,
    )

    coarse_pred = (
        coarse_prob >= THRESHOLD
    ).astype(int)

    final_pred = (
        final_prob >= THRESHOLD
    ).astype(int)

    fixed = int(
        np.sum(
            (coarse_pred != labels)
            & (final_pred == labels)
        )
    )

    broken = int(
        np.sum(
            (coarse_pred == labels)
            & (final_pred != labels)
        )
    )

    tp = int(result["tp"])
    tn = int(result["tn"])
    fp = int(result["fp"])
    fn = int(result["fn"])

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    return {
        "ACC":
            result["accuracy"],

        "BA":
            result["balanced_accuracy"],

        "AUC":
            result["auc"],

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            result["f1"],

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "routed_n":
            int(np.sum(route_mask)),

        "routed_ratio":
            float(np.mean(route_mask)),

        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            fixed - broken,
    }


def source_threshold(
    scores,
    fraction,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    k = max(
        1,
        int(
            round(
                len(scores)
                * fraction
            )
        ),
    )

    ordered = np.sort(
        scores
    )[::-1]

    return float(
        ordered[k - 1]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        default="2018,2019,2020,2021",
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "artifacts/p1-features/features.csv"
        ),
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/feature_schema.json"
        ),
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/"
            "p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            "configs/protocol/"
            "semantic_mapping_v1.yaml"
        ),
    )

    parser.add_argument(
        "--e7b",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e7b-strict-source-joint-gate-2017/"
            "e7b_scores.csv"
        ),
    )

    parser.add_argument(
        "--e8",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e8-collision-source-ablation-2017/"
            "e8_scores_long.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "semantic-collision-frozen-future-eval"
        ),
    )

    args = parser.parse_args()

    years = [
        int(x.strip())
        for x in args.years.split(",")
        if x.strip()
    ]

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "========================================"
    )
    print(
        "冻结跨年测试"
    )
    print(
        "2017训练 -> 未来年份测试"
    )
    print(
        "========================================"
    )

    all_rows = read_csv(
        args.features
    )

    compact_by_sha = {}

    clean_rows = []

    for row in all_rows:
        if str(
            row.get("status", "")
        ).strip() != "OK":
            continue

        sha = norm_sha(
            row.get(
                "sha256",
                "",
            )
        )

        if not sha:
            continue

        copied = dict(row)
        copied["sha256"] = sha

        compact_by_sha[sha] = copied
        clean_rows.append(copied)

    source_rows = [
        row
        for row in clean_rows
        if str(
            row.get(
                "metadata_year",
                "",
            )
        ).strip() == "2017"
    ]

    if len(source_rows) != 799:
        raise RuntimeError(
            f"Expected source=799, got {len(source_rows)}"
        )

    source_labels = np.asarray(
        [
            label_of(row)
            for row in source_rows
        ],
        dtype=np.int64,
    )

    source_shas = [
        row["sha256"]
        for row in source_rows
    ]

    print(
        "2017来源APK:",
        len(source_rows),
    )

    # -------------------------------------------------
    # canonical Compact定义
    # -------------------------------------------------

    with args.schema.open(
        encoding="utf-8",
    ) as f:
        compact_schema = json.load(f)

    (
        _,
        _,
        canonical_original_names,
    ) = allratio.original_compact_matrix_named(
        source_rows,
        source_rows[:1],
        compact_by_sha,
        compact_schema,
    )

    canonical_original_names = list(
        canonical_original_names
    )

    if len(canonical_original_names) != 76:
        raise RuntimeError(
            "Expected canonical raw Compact=76"
        )

    # -------------------------------------------------
    # 读取结构缓存
    # -------------------------------------------------

    cache = exp.read_cache(
        args.cache
    )

    print(
        "结构缓存APK:",
        len(cache),
    )

    roles = load_roles(
        args.mapping
    )

    print()
    print(
        "恢复2017结构..."
    )

    (
        source_structure,
        structure_names,
    ) = reconstruct_structure(
        source_rows,
        cache,
        len(roles),
    )

    # -------------------------------------------------
    # 2017 OOF校准分布
    # -------------------------------------------------

    e7b_rows = read_csv(
        args.e7b
    )

    e7b_by_sha = {
        norm_sha(
            row["sha256"]
        ): row
        for row in e7b_rows
    }

    e8_rows = read_csv(
        args.e8
    )

    semantic_e8 = {
        norm_sha(
            row["sha256"]
        ): row
        for row in e8_rows
        if row[
            "space"
        ] == "SemanticOnly"
    }

    if len(e7b_by_sha) != 799:
        raise RuntimeError(
            "E7b source calibration must contain 799 APKs"
        )

    if len(semantic_e8) != 799:
        raise RuntimeError(
            "E8 SemanticOnly must contain 799 APKs"
        )

    source_collision_raw = np.asarray(
        [
            float(
                semantic_e8[sha][
                    "raw_collision"
                ]
            )
            for sha in source_shas
        ],
        dtype=float,
    )

    source_uncertainty_raw = np.asarray(
        [
            float(
                e7b_by_sha[sha][
                    "raw_uncertainty"
                ]
            )
            for sha in source_shas
        ],
        dtype=float,
    )

    source_collision_pct = (
        e7b.empirical_percentile(
            source_collision_raw,
            source_collision_raw,
        )
    )

    source_uncertainty_pct = (
        e7b.empirical_percentile(
            source_uncertainty_raw,
            source_uncertainty_raw,
        )
    )

    source_joint = np.minimum(
        source_collision_pct,
        source_uncertainty_pct,
    )

    joint_thresholds = {
        budget:
            source_threshold(
                source_joint,
                budget,
            )
        for budget in (
            0.10,
            0.20,
            0.30,
        )
    }

    uncertainty20_threshold = (
        source_threshold(
            source_uncertainty_pct,
            0.20,
        )
    )

    collision20_threshold = (
        source_threshold(
            source_collision_pct,
            0.20,
        )
    )

    print()
    print(
        "===== 2017冻结阈值 ====="
    )

    print(
        "低置信度20%:",
        uncertainty20_threshold,
    )

    print(
        "碰撞20%:",
        collision20_threshold,
    )

    for budget, threshold in (
        joint_thresholds.items()
    ):
        print(
            f"联合歧义{budget:.0%}:",
            threshold,
        )

    threshold_json = {
        "k_neighbors":
            K_NEIGHBORS,

        "collision_space":
            "46-dimensional role_node/role_edge semantic-count space",

        "uncertainty_definition":
            "0.5 - abs(p_coarse - 0.5)",

        "joint_definition":
            "min(collision_percentile, uncertainty_percentile)",

        "uncertainty20_threshold":
            uncertainty20_threshold,

        "collision20_threshold":
            collision20_threshold,

        "joint_thresholds": {
            str(k):
                v
            for k, v
            in joint_thresholds.items()
        },

        "threshold_source":
            "2017 source OOF only",
    }

    (
        args.output
        / "frozen_2017_thresholds.json"
    ).write_text(
        json.dumps(
            threshold_json,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 未来年份
    # -------------------------------------------------

    summary_rows = []
    prediction_rows = []

    for year in years:

        target_rows = [
            row
            for row in clean_rows
            if str(
                row.get(
                    "metadata_year",
                    "",
                )
            ).strip() == str(year)
        ]

        if not target_rows:
            raise RuntimeError(
                f"No rows for {year}"
            )

        target_labels = np.asarray(
            [
                label_of(row)
                for row in target_rows
            ],
            dtype=np.int64,
        )

        print()
        print(
            "========================================"
        )
        print(
            "2017训练 ->",
            year,
            "测试",
        )
        print(
            "测试APK:",
            len(target_rows),
        )
        print(
            "========================================"
        )

        # ---------------------------------------------
        # 第一层70维
        # ---------------------------------------------

        (
            compact_train,
            compact_test,
            compact_names,
        ) = p72.compact_matrix_allratio70_foldsafe(
            source_rows,
            target_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )

        compact_names = list(
            compact_names
        )

        if len(compact_names) != 70:
            raise RuntimeError(
                f"Expected Compact70, got {len(compact_names)}"
            )

        sem_idx = semantic_indices(
            compact_names
        )

        (
            coarse_probability,
            _,
            _,
        ) = p72.fit_predict(
            compact_train,
            source_labels,
            compact_test,
            seed=0,
        )

        # ---------------------------------------------
        # 46维角色语义计数碰撞
        # ---------------------------------------------

        collision_rows = e5.collision_scores(
            compact_train[
                :,
                sem_idx
            ],
            source_labels,
            compact_test[
                :,
                sem_idx
            ],
            K_NEIGHBORS,
        )

        target_collision_raw = np.asarray(
            [
                float(
                    row[
                        "collision_score"
                    ]
                )
                for row
                in collision_rows
            ],
            dtype=float,
        )

        collision_pct = (
            e7b.empirical_percentile(
                source_collision_raw,
                target_collision_raw,
            )
        )

        target_uncertainty_raw = (
            0.5
            - np.abs(
                coarse_probability
                - 0.5
            )
        )

        uncertainty_pct = (
            e7b.empirical_percentile(
                source_uncertainty_raw,
                target_uncertainty_raw,
            )
        )

        joint_score = np.minimum(
            collision_pct,
            uncertainty_pct,
        )

        # ---------------------------------------------
        # 第二层585维宿主结构
        # ---------------------------------------------

        print(
            "恢复",
            year,
            "结构..."
        )

        (
            target_structure,
            _,
        ) = reconstruct_structure(
            target_rows,
            cache,
            len(roles),
        )

        (
            structure_train_scaled,
            structure_test_scaled,
            _,
        ) = p72.standardize_added(
            source_structure,
            target_structure,
            structure_names,
        )

        x_structure_train = np.concatenate(
            [
                compact_train,
                structure_train_scaled,
            ],
            axis=1,
        )

        x_structure_test = np.concatenate(
            [
                compact_test,
                structure_test_scaled,
            ],
            axis=1,
        )

        (
            structure_probability,
            _,
            _,
        ) = p72.fit_predict(
            x_structure_train,
            source_labels,
            x_structure_test,
            seed=0,
        )

        # ---------------------------------------------
        # 路由
        # ---------------------------------------------

        route_masks = {
            "第一层":
                np.zeros(
                    len(target_rows),
                    dtype=bool,
                ),

            "全部结构":
                np.ones(
                    len(target_rows),
                    dtype=bool,
                ),

            "20%低置信度阈值":
                uncertainty_pct
                >= uncertainty20_threshold,

            "20%语义碰撞阈值":
                collision_pct
                >= collision20_threshold,

            "10%联合歧义":
                joint_score
                >= joint_thresholds[
                    0.10
                ],

            "20%联合歧义":
                joint_score
                >= joint_thresholds[
                    0.20
                ],

            "30%联合歧义":
                joint_score
                >= joint_thresholds[
                    0.30
                ],
        }

        print()
        print(
            "方法                 "
            "分析比例    ACC       BA        "
            "Precision  Recall    F1        AUC"
        )

        for method, mask in (
            route_masks.items()
        ):
            result = evaluate(
                target_labels,
                coarse_probability,
                structure_probability,
                mask,
            )

            summary_rows.append({
                "train_year":
                    2017,

                "test_year":
                    year,

                "method":
                    method,

                "n":
                    len(target_rows),

                **result,
            })

            print(
                f"{method:20s}",
                f"{result['routed_ratio']:7.3f}",
                f"{result['ACC']:.6f}",
                f"{result['BA']:.6f}",
                f"{result['Precision']:.6f}",
                f"{result['Recall']:.6f}",
                f"{result['F1']:.6f}",
                f"{result['AUC']:.6f}",
            )

        # 每APK保存，方便后面分析
        for i, row in enumerate(
            target_rows
        ):
            prediction_rows.append({
                "sha256":
                    row["sha256"],

                "year":
                    year,

                "label":
                    int(
                        target_labels[i]
                    ),

                "coarse_probability":
                    float(
                        coarse_probability[i]
                    ),

                "structure_probability":
                    float(
                        structure_probability[i]
                    ),

                "collision_raw":
                    float(
                        target_collision_raw[i]
                    ),

                "collision_percentile":
                    float(
                        collision_pct[i]
                    ),

                "uncertainty_raw":
                    float(
                        target_uncertainty_raw[i]
                    ),

                "uncertainty_percentile":
                    float(
                        uncertainty_pct[i]
                    ),

                "joint_ambiguity":
                    float(
                        joint_score[i]
                    ),

                "route_joint_10":
                    int(
                        route_masks[
                            "10%联合歧义"
                        ][i]
                    ),

                "route_joint_20":
                    int(
                        route_masks[
                            "20%联合歧义"
                        ][i]
                    ),

                "route_joint_30":
                    int(
                        route_masks[
                            "30%联合歧义"
                        ][i]
                    ),
            })

    write_csv(
        args.output
        / "future_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "future_predictions.csv",
        prediction_rows,
    )

    audit = {
        "source_year":
            2017,

        "target_years":
            years,

        "source_samples":
            len(source_rows),

        "future_labels_used_for":
            "final evaluation only",

        "future_labels_used_for_feature_selection":
            False,

        "future_labels_used_for_threshold_tuning":
            False,

        "future_distribution_used_for_threshold_calibration":
            False,

        "k_neighbors":
            K_NEIGHBORS,

        "collision_space":
            "46 role-count dimensions",

        "structure":
            "Stats120 + PairRaw465",

        "classifier_threshold":
            0.5,

        "main_gate":
            "20% source-frozen joint ambiguity threshold",
    }

    (
        args.output
        / "future_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
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
