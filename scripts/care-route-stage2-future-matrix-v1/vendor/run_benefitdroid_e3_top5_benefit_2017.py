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


import run_compact_quantity_shared_caller_2x3_2017_eval as exp
import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72
import run_benefitdroid_e1_exact_pair_2017 as e1


THRESHOLD = 0.5


ROLE_ZH = {
    "command_execution": "命令执行",
    "dynamic_code_loading": "动态代码加载",
    "native_library_loading": "本地库加载",
    "reflective_invocation": "反射调用",
    "package_installation": "应用安装",
    "package_uninstallation": "应用卸载",
    "device_administration": "设备管理",
    "accessibility_control": "无障碍控制",
    "sms_send_receive": "短信收发",
    "call_control_or_recording": "通话控制或录音",
    "account_or_credential_access": "账户或凭据访问",
    "contacts_or_calendar_access": "联系人或日历访问",
    "device_identifier_access": "设备标识访问",
    "location_access": "位置访问",
    "microphone_access": "麦克风访问",
    "camera_access": "相机访问",
    "clipboard_access": "剪贴板访问",
    "network_transmission_or_upload": "网络传输或上传",
    "sensitive_file_or_content_write": "敏感文件或内容写入",
    "boot_persistence": "开机持久化",
    "task_or_alarm_scheduling": "任务或闹钟调度",
    "background_service_control": "后台服务控制",
    "overlay_or_ui_deception": "悬浮窗或界面欺骗",
    "icon_or_component_hiding": "图标或组件隐藏",
    "process_or_wakelock_control": "进程或唤醒锁控制",
    "cryptography": "加密",
    "encoding_or_compression": "编码或压缩",
    "url_or_http_setup": "URL或HTTP设置",
    "file_read": "文件读取",
    "database_access": "数据库访问",
    "other": "其他",
}


def norm_sha(value):
    return str(value or "").strip().casefold()


def safe_float(value):
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return 0.0


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def get_role_names(path: Path):
    with path.open(
        encoding="utf-8",
    ) as f:
        obj = yaml.safe_load(f)

    roles = list(obj["roles"])

    if len(roles) != 31:
        raise RuntimeError(
            f"Expected 31 roles, got {len(roles)}"
        )

    if len(roles) != int(exp.ROLE_VOCABULARY_SIZE):
        raise RuntimeError(
            "semantic_mapping_v1 roles do not match cache role dimension"
        )

    return roles


def role_presence_matrix(
    manifest,
    compact_by_sha,
    roles,
):
    matrix = np.zeros(
        (
            len(manifest),
            len(roles),
        ),
        dtype=np.int64,
    )

    print()
    print("===== 第一层语义角色输入检查 =====")

    for j, role in enumerate(roles):
        node_col = f"role_node__{role}"
        edge_col = f"role_edge__{role}"

        node_exists = any(
            node_col in compact_by_sha[row["sha256"]]
            for row in manifest
        )

        edge_exists = any(
            edge_col in compact_by_sha[row["sha256"]]
            for row in manifest
        )

        if not node_exists and not edge_exists:
            raise RuntimeError(
                f"No first-layer role feature found for {role}"
            )

        for i, row in enumerate(manifest):
            feature_row = compact_by_sha[
                row["sha256"]
            ]

            node_value = safe_float(
                feature_row.get(
                    node_col,
                    0.0,
                )
            )

            edge_value = safe_float(
                feature_row.get(
                    edge_col,
                    0.0,
                )
            )

            matrix[i, j] = int(
                node_value > 0.0
                or edge_value > 0.0
            )

        print(
            f"{j:02d}",
            role,
            ROLE_ZH.get(role, role),
            "APK数=",
            int(matrix[:, j].sum()),
        )

    return matrix


def reconstruct_structure(
    manifest,
    cache,
    roles,
):
    from itertools import combinations

    pair_list = list(
        combinations(
            range(len(roles)),
            2,
        )
    )

    pair_index = {
        pair: i
        for i, pair in enumerate(pair_list)
    }

    stats_rows = []
    pair_rows = []

    for i, row in enumerate(
        manifest,
        start=1,
    ):
        sha = row["sha256"]

        (
            stats,
            pair_raw,
            _pair_norm,
            _audit,
        ) = e1.reconstruct_host_structure(
            cache[sha]["payload"],
            pair_list,
            pair_index,
        )

        stats_rows.append(stats)
        pair_rows.append(pair_raw)

        if i % 100 == 0 or i == len(manifest):
            print(
                "恢复结构:",
                i,
                "/",
                len(manifest),
            )

    stats_all = np.asarray(
        stats_rows,
        dtype=np.float64,
    )

    pair_all = np.asarray(
        pair_rows,
        dtype=np.float64,
    )

    if stats_all.shape[1] != 120:
        raise RuntimeError(
            f"Expected Stats=120, got {stats_all.shape}"
        )

    if pair_all.shape[1] != 465:
        raise RuntimeError(
            f"Expected Pair=465, got {pair_all.shape}"
        )

    structure_all = np.concatenate(
        [
            stats_all,
            pair_all,
        ],
        axis=1,
    )

    structure_names = (
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

    return (
        structure_all,
        structure_names,
    )


def build_compact_matrix(
    train_indices,
    val_indices,
    cohort_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
):
    train_rows = [
        cohort_rows[i]
        for i in train_indices
    ]

    val_rows = [
        cohort_rows[i]
        for i in val_indices
    ]

    (
        compact_train,
        compact_val,
        compact_names,
    ) = p72.compact_matrix_allratio70_foldsafe(
        train_rows,
        val_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    )

    if len(compact_names) != 70:
        raise RuntimeError(
            f"Expected Compact=70, got {len(compact_names)}"
        )

    return (
        compact_train,
        compact_val,
    )


def fit_one_split(
    train_indices,
    val_indices,
    labels,
    cohort_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
    structure_all,
    structure_names,
    seed,
):
    (
        compact_train,
        compact_val,
    ) = build_compact_matrix(
        train_indices,
        val_indices,
        cohort_rows,
        compact_by_sha,
        compact_schema,
        canonical_original_names,
    )

    # 第一层预测
    (
        coarse_probability,
        _,
        _,
    ) = p72.fit_predict(
        compact_train,
        labels[train_indices],
        compact_val,
        seed=seed,
    )

    # 第二层结构特征只用训练部分拟合标准化
    (
        structure_train,
        structure_val,
        _,
    ) = p72.standardize_added(
        structure_all[train_indices],
        structure_all[val_indices],
        structure_names,
    )

    x_train = np.concatenate(
        [
            compact_train,
            structure_train,
        ],
        axis=1,
    )

    x_val = np.concatenate(
        [
            compact_val,
            structure_val,
        ],
        axis=1,
    )

    (
        structure_probability,
        _,
        _,
    ) = p72.fit_predict(
        x_train,
        labels[train_indices],
        x_val,
        seed=seed,
    )

    return (
        coarse_probability,
        structure_probability,
    )


def calculate_role_scores(
    indices,
    labels,
    coarse_probability,
    structure_probability,
    role_presence,
    roles,
    gamma,
    alpha,
):
    coarse_pred = (
        coarse_probability >= THRESHOLD
    ).astype(np.int64)

    structure_pred = (
        structure_probability >= THRESHOLD
    ).astype(np.int64)

    rows = []

    for r, role in enumerate(roles):
        present = (
            role_presence[indices, r] > 0
        )

        y = labels[indices]

        c = coarse_pred[indices]
        s = structure_pred[indices]

        support = int(
            np.sum(present)
        )

        errors = int(
            np.sum(
                present
                & (c != y)
            )
        )

        fixed = int(
            np.sum(
                present
                & (c != y)
                & (s == y)
            )
        )

        broken = int(
            np.sum(
                present
                & (c == y)
                & (s != y)
            )
        )

        net = (
            fixed
            - gamma * broken
        )

        benefit_score = (
            net
            / (
                support
                + alpha
            )
        )

        rows.append({
            "role_index":
                r,

            "role":
                role,

            "role_zh":
                ROLE_ZH.get(
                    role,
                    role,
                ),

            "support":
                support,

            "coarse_error_count":
                errors,

            "fixed":
                fixed,

            "broken":
                broken,

            "net_benefit":
                float(net),

            "benefit_score":
                float(benefit_score),
        })

    return rows


def choose_top5_error(score_rows):
    ordered = sorted(
        score_rows,
        key=lambda row: (
            -row[
                "coarse_error_count"
            ],
            -row[
                "support"
            ],
            row[
                "role_index"
            ],
        ),
    )

    return ordered[:5]


def choose_top5_benefit(score_rows):
    ordered = sorted(
        score_rows,
        key=lambda row: (
            -row[
                "benefit_score"
            ],
            -row[
                "net_benefit"
            ],
            -row[
                "fixed"
            ],
            row[
                "role_index"
            ],
        ),
    )

    return ordered[:5]


def route_mask(
    indices,
    role_presence,
    selected_rows,
):
    selected_indices = [
        int(row["role_index"])
        for row in selected_rows
    ]

    if not selected_indices:
        return np.zeros(
            len(indices),
            dtype=bool,
        )

    return np.any(
        role_presence[
            np.ix_(
                indices,
                selected_indices,
            )
        ] > 0,
        axis=1,
    )


def correction_counts(
    labels,
    base_probability,
    final_probability,
):
    base_pred = (
        base_probability >= THRESHOLD
    ).astype(np.int64)

    final_pred = (
        final_probability >= THRESHOLD
    ).astype(np.int64)

    fixed_mask = (
        (base_pred != labels)
        & (final_pred == labels)
    )

    broken_mask = (
        (base_pred == labels)
        & (final_pred != labels)
    )

    fixed = int(
        np.sum(fixed_mask)
    )

    broken = int(
        np.sum(broken_mask)
    )

    fixed_fn = int(
        np.sum(
            fixed_mask
            & (labels == 1)
        )
    )

    fixed_fp = int(
        np.sum(
            fixed_mask
            & (labels == 0)
        )
    )

    broken_to_fn = int(
        np.sum(
            broken_mask
            & (labels == 1)
        )
    )

    broken_to_fp = int(
        np.sum(
            broken_mask
            & (labels == 0)
        )
    )

    return {
        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            fixed - broken,

        "fixed_false_negative":
            fixed_fn,

        "fixed_false_positive":
            fixed_fp,

        "introduced_false_negative":
            broken_to_fn,

        "introduced_false_positive":
            broken_to_fp,
    }


def main():
    parser = argparse.ArgumentParser()

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
        "--p72-oof",
        type=Path,
        default=Path(
            "artifacts/"
            "p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )

    parser.add_argument(
        "--e1b-oof",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e1b-exact-pair-2017/"
            "e1_oof_predictions.csv"
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e3-top5-benefit-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 固定 799 APK / 5 折
    # ========================================================

    p72_rows = read_csv(
        args.p72_oof
    )

    e1_rows = read_csv(
        args.e1b_oof
    )

    if len(p72_rows) != 799:
        raise RuntimeError(
            f"Expected P72 n=799, got {len(p72_rows)}"
        )

    if len(e1_rows) != 799:
        raise RuntimeError(
            f"Expected E1b n=799, got {len(e1_rows)}"
        )

    e1_by_sha = {
        norm_sha(
            row["sha256"]
        ): row
        for row in e1_rows
    }

    manifest = []

    for row in p72_rows:
        sha = norm_sha(
            row["sha256"]
        )

        if sha not in e1_by_sha:
            raise RuntimeError(
                f"Missing E1b row: {sha}"
            )

        e1_row = e1_by_sha[
            sha
        ]

        manifest.append({
            "sha256":
                sha,

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            "metadata_label":
                row[
                    "metadata_label"
                ],

            "fold":
                int(
                    row[
                        "fold"
                    ]
                ),

            "coarse_probability":
                float(
                    row[
                        "C_prob"
                    ]
                ),

            "structure_probability":
                float(
                    e1_row[
                        "C_plus_Stats_plus_PairRaw_prob"
                    ]
                ),

            "e1_coarse_probability":
                float(
                    e1_row[
                        "C_prob"
                    ]
                ),
        })

    labels = np.asarray(
        [
            row["label"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    fold_ids = np.asarray(
        [
            row["fold"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    coarse_oof = np.asarray(
        [
            row[
                "coarse_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    structure_oof = np.asarray(
        [
            row[
                "structure_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    e1_coarse = np.asarray(
        [
            row[
                "e1_coarse_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    max_c_diff = float(
        np.max(
            np.abs(
                coarse_oof
                - e1_coarse
            )
        )
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E3"
    )
    print(
        "错误Top5 vs 结构收益Top5"
    )
    print(
        "========================================"
    )

    print(
        "APK数量:",
        len(manifest),
    )

    print(
        "第一层重复误差最大值:",
        max_c_diff,
    )

    if max_c_diff > 1e-7:
        raise RuntimeError(
            "P72 C and E1b C do not match"
        )

    # ========================================================
    # 31类语义名称
    # ========================================================

    roles = get_role_names(
        args.semantic_mapping
    )

    print()
    print("===== 31类语义索引 =====")

    for i, role in enumerate(roles):
        print(
            f"{i:02d}",
            role,
            "=>",
            ROLE_ZH.get(
                role,
                role,
            ),
        )

    # ========================================================
    # 第一层原始特征
    # ========================================================

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

        if not sha:
            continue

        row = dict(row)
        row["sha256"] = sha

        compact_by_sha[
            sha
        ] = row

    missing = [
        row["sha256"]
        for row in manifest
        if row["sha256"]
        not in compact_by_sha
    ]

    if missing:
        raise RuntimeError(
            "Missing Compact APKs: "
            + repr(
                missing[:10]
            )
        )

    cohort_rows = [
        compact_by_sha[
            row["sha256"]
        ]
        for row in manifest
    ]

    role_presence = (
        role_presence_matrix(
            manifest,
            compact_by_sha,
            roles,
        )
    )

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

    # ========================================================
    # 从已有缓存恢复 Stats + PairRaw
    # 不重新分析 APK
    # ========================================================

    if not args.cache.is_file():
        raise RuntimeError(
            f"Cache not found: {args.cache}"
        )

    cache = exp.read_cache(
        args.cache
    )

    missing_cache = [
        row["sha256"]
        for row in manifest
        if row["sha256"]
        not in cache
    ]

    if missing_cache:
        raise RuntimeError(
            "Cache missing APKs: "
            + repr(
                missing_cache[:10]
            )
        )

    print()
    print("===== 恢复结构特征 =====")

    (
        structure_all,
        structure_names,
    ) = reconstruct_structure(
        manifest,
        cache,
        roles,
    )

    print(
        "结构特征维度:",
        structure_all.shape[1],
        "(120旧统计 + 465精确语义对)",
    )

    # ========================================================
    # 外层5折
    #
    # 对每个外层测试折：
    #   只在另外4折内部再次做交叉预测，
    #   用这些内部未见预测计算收益并选择Top5。
    #
    # 因此外层测试折标签不会影响Top5选择。
    # ========================================================

    top5_error_final_prob = np.array(
        coarse_oof,
        copy=True,
    )

    top5_benefit_final_prob = np.array(
        coarse_oof,
        copy=True,
    )

    top5_error_route = np.zeros(
        len(manifest),
        dtype=np.int64,
    )

    top5_benefit_route = np.zeros(
        len(manifest),
        dtype=np.int64,
    )

    role_score_rows = []
    selection_rows = []
    per_fold_rows = []

    unique_folds = sorted(
        np.unique(
            fold_ids
        )
    )

    for outer_fold in unique_folds:

        print()
        print(
            "========================================"
        )
        print(
            "外层测试折:",
            int(outer_fold),
        )
        print(
            "========================================"
        )

        outer_train_idx = np.flatnonzero(
            fold_ids != outer_fold
        )

        outer_test_idx = np.flatnonzero(
            fold_ids == outer_fold
        )

        # --------------------------------------------
        # 只为外层训练集生成内部OOF预测
        # --------------------------------------------

        inner_coarse = np.full(
            len(manifest),
            np.nan,
            dtype=np.float64,
        )

        inner_structure = np.full(
            len(manifest),
            np.nan,
            dtype=np.float64,
        )

        inner_folds = [
            fold
            for fold in unique_folds
            if fold != outer_fold
        ]

        for inner_val_fold in inner_folds:

            inner_val_idx = np.flatnonzero(
                (fold_ids != outer_fold)
                & (
                    fold_ids
                    == inner_val_fold
                )
            )

            inner_train_idx = np.flatnonzero(
                (fold_ids != outer_fold)
                & (
                    fold_ids
                    != inner_val_fold
                )
            )

            print(
                "  内层验证折",
                int(inner_val_fold),
                "训练/验证=",
                len(inner_train_idx),
                "/",
                len(inner_val_idx),
            )

            (
                c_prob,
                s_prob,
            ) = fit_one_split(
                inner_train_idx,
                inner_val_idx,
                labels,
                cohort_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
                structure_all,
                structure_names,
                seed=int(inner_val_fold) - 1,
            )

            inner_coarse[
                inner_val_idx
            ] = c_prob

            inner_structure[
                inner_val_idx
            ] = s_prob

        if np.any(
            ~np.isfinite(
                inner_coarse[
                    outer_train_idx
                ]
            )
        ):
            raise RuntimeError(
                "Inner coarse OOF incomplete"
            )

        if np.any(
            ~np.isfinite(
                inner_structure[
                    outer_train_idx
                ]
            )
        ):
            raise RuntimeError(
                "Inner structure OOF incomplete"
            )

        # --------------------------------------------
        # 只利用外层训练集的内部OOF结果选择Top5
        # --------------------------------------------

        scores = calculate_role_scores(
            outer_train_idx,
            labels,
            inner_coarse,
            inner_structure,
            role_presence,
            roles,
            gamma=args.gamma,
            alpha=args.alpha,
        )

        for score in scores:
            role_score_rows.append({
                "outer_fold":
                    int(outer_fold),

                **score,
            })

        selected_error = (
            choose_top5_error(
                scores
            )
        )

        selected_benefit = (
            choose_top5_benefit(
                scores
            )
        )

        print()
        print("错误最多的5类:")

        for rank, row in enumerate(
            selected_error,
            start=1,
        ):
            print(
                f"  {rank}.",
                row["role_zh"],
                f"错误数={row['coarse_error_count']}",
                f"出现={row['support']}",
            )

            selection_rows.append({
                "outer_fold":
                    int(outer_fold),

                "method":
                    "Top5_Error",

                "rank":
                    rank,

                **row,
            })

        print()
        print("结构净收益最高的5类:")

        for rank, row in enumerate(
            selected_benefit,
            start=1,
        ):
            print(
                f"  {rank}.",
                row["role_zh"],
                f"修复={row['fixed']}",
                f"破坏={row['broken']}",
                f"净收益={row['net_benefit']:.1f}",
                f"得分={row['benefit_score']:.6f}",
            )

            selection_rows.append({
                "outer_fold":
                    int(outer_fold),

                "method":
                    "Top5_Benefit",

                "rank":
                    rank,

                **row,
            })

        # --------------------------------------------
        # 在真正未见的外层测试折上路由
        # --------------------------------------------

        route_error = route_mask(
            outer_test_idx,
            role_presence,
            selected_error,
        )

        route_benefit = route_mask(
            outer_test_idx,
            role_presence,
            selected_benefit,
        )

        test_structure_prob = (
            structure_oof[
                outer_test_idx
            ]
        )

        test_coarse_prob = (
            coarse_oof[
                outer_test_idx
            ]
        )

        error_final = np.where(
            route_error,
            test_structure_prob,
            test_coarse_prob,
        )

        benefit_final = np.where(
            route_benefit,
            test_structure_prob,
            test_coarse_prob,
        )

        top5_error_final_prob[
            outer_test_idx
        ] = error_final

        top5_benefit_final_prob[
            outer_test_idx
        ] = benefit_final

        top5_error_route[
            outer_test_idx
        ] = route_error.astype(
            np.int64
        )

        top5_benefit_route[
            outer_test_idx
        ] = route_benefit.astype(
            np.int64
        )

        y_test = labels[
            outer_test_idx
        ]

        error_metrics = base.metrics(
            y_test,
            error_final,
            threshold=THRESHOLD,
        )

        benefit_metrics = base.metrics(
            y_test,
            benefit_final,
            threshold=THRESHOLD,
        )

        error_corr = correction_counts(
            y_test,
            test_coarse_prob,
            error_final,
        )

        benefit_corr = correction_counts(
            y_test,
            test_coarse_prob,
            benefit_final,
        )

        for method, route, result, corr in (
            (
                "Top5_Error",
                route_error,
                error_metrics,
                error_corr,
            ),
            (
                "Top5_Benefit",
                route_benefit,
                benefit_metrics,
                benefit_corr,
            ),
        ):
            per_fold_rows.append({
                "outer_fold":
                    int(outer_fold),

                "method":
                    method,

                "test_n":
                    len(
                        outer_test_idx
                    ),

                "routed_n":
                    int(
                        np.sum(route)
                    ),

                "routed_ratio":
                    float(
                        np.mean(route)
                    ),

                "BA":
                    result[
                        "balanced_accuracy"
                    ],

                "ACC":
                    result[
                        "accuracy"
                    ],

                "AUC":
                    result[
                        "auc"
                    ],

                "F1":
                    result[
                        "f1"
                    ],

                **corr,
            })

    # ========================================================
    # 汇总结果
    # ========================================================

    configs = {
        "仅第一层":
            (
                coarse_oof,
                np.zeros(
                    len(manifest),
                    dtype=np.int64,
                ),
            ),

        "全部APK都做结构":
            (
                structure_oof,
                np.ones(
                    len(manifest),
                    dtype=np.int64,
                ),
            ),

        "错误最多5类路由":
            (
                top5_error_final_prob,
                top5_error_route,
            ),

        "结构收益最高5类路由":
            (
                top5_benefit_final_prob,
                top5_benefit_route,
            ),
    }

    summary_rows = []

    for name, (
        probability,
        route,
    ) in configs.items():

        result = base.metrics(
            labels,
            probability,
            threshold=THRESHOLD,
        )

        corr = correction_counts(
            labels,
            coarse_oof,
            probability,
        )

        routed_n = int(
            np.sum(route)
        )

        routed_ratio = float(
            np.mean(route)
        )

        net_per_100 = (
            corr["net"]
            / routed_n
            * 100.0
            if routed_n > 0
            else 0.0
        )

        summary_rows.append({
            "method":
                name,

            "n":
                len(manifest),

            "routed_n":
                routed_n,

            "routed_ratio":
                routed_ratio,

            "BA":
                result[
                    "balanced_accuracy"
                ],

            "ACC":
                result[
                    "accuracy"
                ],

            "AUC":
                result[
                    "auc"
                ],

            "F1":
                result[
                    "f1"
                ],

            "TP":
                result[
                    "tp"
                ],

            "TN":
                result[
                    "tn"
                ],

            "FP":
                result[
                    "fp"
                ],

            "FN":
                result[
                    "fn"
                ],

            **corr,

            "net_corrections_per_100_routed":
                net_per_100,
        })

    # ========================================================
    # 保存每个APK的最终结果
    # ========================================================

    prediction_rows = []

    for i, row in enumerate(
        manifest
    ):
        prediction_rows.append({
            "sha256":
                row[
                    "sha256"
                ],

            "metadata_label":
                row[
                    "metadata_label"
                ],

            "label":
                row[
                    "label"
                ],

            "fold":
                row[
                    "fold"
                ],

            "coarse_probability":
                float(
                    coarse_oof[i]
                ),

            "structure_probability":
                float(
                    structure_oof[i]
                ),

            "top5_error_routed":
                int(
                    top5_error_route[i]
                ),

            "top5_error_probability":
                float(
                    top5_error_final_prob[i]
                ),

            "top5_benefit_routed":
                int(
                    top5_benefit_route[i]
                ),

            "top5_benefit_probability":
                float(
                    top5_benefit_final_prob[i]
                ),
        })

    # ========================================================
    # 全799个2017 OOF结果计算最终Source-only Top5
    #
    # 这个结果不用于上面的2017评估，
    # 只作为之后2018-2021冻结测试时的Source-only选择。
    # ========================================================

    all_idx = np.arange(
        len(manifest)
    )

    global_scores = (
        calculate_role_scores(
            all_idx,
            labels,
            coarse_oof,
            structure_oof,
            role_presence,
            roles,
            gamma=args.gamma,
            alpha=args.alpha,
        )
    )

    global_error = (
        choose_top5_error(
            global_scores
        )
    )

    global_benefit = (
        choose_top5_benefit(
            global_scores
        )
    )

    global_score_rows = []

    for row in global_scores:
        global_score_rows.append(
            dict(row)
        )

    future_top5 = {
        "warning":
            (
                "These Top5 sets are selected only from "
                "2017 source OOF results and are intended "
                "to be frozen before 2018-2021 evaluation."
            ),

        "gamma":
            args.gamma,

        "alpha":
            args.alpha,

        "Top5_Error":
            [
                {
                    "role_index":
                        row[
                            "role_index"
                        ],

                    "role":
                        row[
                            "role"
                        ],

                    "role_zh":
                        row[
                            "role_zh"
                        ],

                    "coarse_error_count":
                        row[
                            "coarse_error_count"
                        ],
                }
                for row in global_error
            ],

        "Top5_Benefit":
            [
                {
                    "role_index":
                        row[
                            "role_index"
                        ],

                    "role":
                        row[
                            "role"
                        ],

                    "role_zh":
                        row[
                            "role_zh"
                        ],

                    "fixed":
                        row[
                            "fixed"
                        ],

                    "broken":
                        row[
                            "broken"
                        ],

                    "net_benefit":
                        row[
                            "net_benefit"
                        ],

                    "benefit_score":
                        row[
                            "benefit_score"
                        ],
                }
                for row in global_benefit
            ],
    }

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        args.output
        / "e3_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "e3_per_outer_fold.csv",
        per_fold_rows,
    )

    write_csv(
        args.output
        / "e3_role_scores_by_outer_fold.csv",
        role_score_rows,
    )

    write_csv(
        args.output
        / "e3_top5_selection_by_outer_fold.csv",
        selection_rows,
    )

    write_csv(
        args.output
        / "e3_oof_predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "e3_global_2017_role_scores.csv",
        global_score_rows,
    )

    (
        args.output
        / "e3_frozen_top5_for_future.json"
    ).write_text(
        json.dumps(
            future_top5,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "source_apk_count":
            len(manifest),

        "outer_folds":
            5,

        "inner_folds_per_outer_fold":
            4,

        "utility":
            (
                "+1 if structure fixes coarse error; "
                "-gamma if structure breaks coarse correct; "
                "0 otherwise"
            ),

        "gamma":
            args.gamma,

        "alpha":
            args.alpha,

        "benefit_score":
            (
                "(fixed - gamma * broken) "
                "/ (role_support + alpha)"
            ),

        "role_selection":
            (
                "strict nested cross-fitting: "
                "outer test fold labels are not used "
                "for Top5 selection"
            ),

        "route_input":
            (
                "first-layer semantic-role presence only"
            ),

        "structure_model":
            (
                "Compact70 + existing Shared-Caller Stats120 "
                "+ exact Host-Role PairRaw465"
            ),
    }

    (
        args.output
        / "e3_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 控制台摘要
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "E3 最终汇总"
    )
    print(
        "========================================"
    )

    for row in summary_rows:
        print(
            f"{row['method']:20s}",
            f"分析APK={row['routed_n']:3d}",
            f"比例={row['routed_ratio']:.3f}",
            f"ACC={row['ACC']:.6f}",
            f"BA={row['BA']:.6f}",
            f"AUC={row['AUC']:.6f}",
            f"修复={row['fixed']:3d}",
            f"破坏={row['broken']:3d}",
            f"净修复={row['net']:+3d}",
            f"每100个分析净修复="
            f"{row['net_corrections_per_100_routed']:.2f}",
        )

    print()
    print(
        "===== 之后2018-2021冻结使用的2017 Top5 ====="
    )

    print()
    print("错误最多的5类:")

    for i, row in enumerate(
        global_error,
        start=1,
    ):
        print(
            f"  {i}.",
            row["role_zh"],
            f"错误数={row['coarse_error_count']}",
        )

    print()
    print("结构收益最高的5类:")

    for i, row in enumerate(
        global_benefit,
        start=1,
    ):
        print(
            f"  {i}.",
            row["role_zh"],
            f"修复={row['fixed']}",
            f"破坏={row['broken']}",
            f"净收益={row['net_benefit']:.1f}",
            f"得分={row['benefit_score']:.6f}",
        )

    print()
    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
