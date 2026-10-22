#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import sys

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_frozen_anchor_residual_2017_2021 as base


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

        fieldnames = []
        seen = set()

        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

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


def get_roles(path: Path):
    with path.open(
        encoding="utf-8",
    ) as f:
        obj = yaml.safe_load(f)

    roles = list(obj["roles"])

    if len(roles) != 31:
        raise RuntimeError(
            f"Expected 31 roles, got {len(roles)}"
        )

    return roles


def role_presence_matrix(
    prediction_rows,
    compact_by_sha,
    roles,
):
    result = np.zeros(
        (
            len(prediction_rows),
            len(roles),
        ),
        dtype=np.int64,
    )

    for i, row in enumerate(prediction_rows):
        feature_row = compact_by_sha[
            row["sha256"]
        ]

        for j, role in enumerate(roles):

            node = safe_float(
                feature_row.get(
                    f"role_node__{role}",
                    0,
                )
            )

            edge = safe_float(
                feature_row.get(
                    f"role_edge__{role}",
                    0,
                )
            )

            result[i, j] = int(
                node > 0
                or edge > 0
            )

    return result


def correction_counts(
    labels,
    base_probability,
    final_probability,
):
    base_pred = (
        base_probability
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
        (base_pred != labels)
        & (final_pred == labels)
    )

    broken_mask = (
        (base_pred == labels)
        & (final_pred != labels)
    )

    return {
        "fixed":
            int(
                np.sum(
                    fixed_mask
                )
            ),

        "broken":
            int(
                np.sum(
                    broken_mask
                )
            ),

        "net":
            int(
                np.sum(
                    fixed_mask
                )
                -
                np.sum(
                    broken_mask
                )
            ),

        "fixed_false_negative":
            int(
                np.sum(
                    fixed_mask
                    & (labels == 1)
                )
            ),

        "fixed_false_positive":
            int(
                np.sum(
                    fixed_mask
                    & (labels == 0)
                )
            ),

        "introduced_false_negative":
            int(
                np.sum(
                    broken_mask
                    & (labels == 1)
                )
            ),

        "introduced_false_positive":
            int(
                np.sum(
                    broken_mask
                    & (labels == 0)
                )
            ),
    }


def evaluate_route(
    labels,
    coarse_probability,
    structure_probability,
    routed_indices,
):
    final_probability = np.array(
        coarse_probability,
        copy=True,
    )

    routed_indices = np.asarray(
        sorted(
            set(
                int(x)
                for x in routed_indices
            )
        ),
        dtype=np.int64,
    )

    final_probability[
        routed_indices
    ] = structure_probability[
        routed_indices
    ]

    result = base.metrics(
        labels,
        final_probability,
        threshold=THRESHOLD,
    )

    corr = correction_counts(
        labels,
        coarse_probability,
        final_probability,
    )

    routed_n = len(
        routed_indices
    )

    return {
        "routed_n":
            routed_n,

        "routed_ratio":
            routed_n
            / len(labels),

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

        "net_per_100_routed":
            (
                corr["net"]
                / routed_n
                * 100.0
                if routed_n
                else 0.0
            ),
    }


def top_k_by_score(
    indices,
    score,
    sha_values,
    k,
):
    indices = list(
        map(
            int,
            indices,
        )
    )

    ordered = sorted(
        indices,
        key=lambda i: (
            -float(
                score[i]
            ),
            sha_values[i],
        ),
    )

    return ordered[:k]


def fold_budget_count(
    fold_size,
    budget,
):
    if budget <= 0:
        return 0

    return min(
        fold_size,
        max(
            1,
            int(
                round(
                    fold_size
                    * budget
                )
            ),
        ),
    )


def select_by_fold_score(
    folds,
    score,
    sha_values,
    budget,
):
    selected = []

    for fold in sorted(
        np.unique(
            folds
        )
    ):
        indices = np.flatnonzero(
            folds == fold
        )

        k = fold_budget_count(
            len(indices),
            budget,
        )

        selected.extend(
            top_k_by_score(
                indices,
                score,
                sha_values,
                k,
            )
        )

    return selected


def select_random_by_fold(
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
        indices = np.flatnonzero(
            folds == fold
        )

        k = fold_budget_count(
            len(indices),
            budget,
        )

        rng = np.random.default_rng(
            seed
            * 1009
            + int(fold)
            * 9176
            + 2026
        )

        chosen = rng.choice(
            indices,
            size=k,
            replace=False,
        )

        selected.extend(
            map(
                int,
                chosen,
            )
        )

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e3-dir",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e3-top5-benefit-2017"
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
        "--semantic-mapping",
        type=Path,
        default=Path(
            "configs/"
            "protocol/"
            "semantic_mapping_v1.yaml"
        ),
    )

    parser.add_argument(
        "--budgets",
        default="0.10,0.20,0.30,0.40,0.50",
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
            "benefitdroid-e4-equal-budget-routing-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    budgets = [
        float(x.strip())
        for x in args.budgets.split(",")
        if x.strip()
    ]

    for budget in budgets:
        if not (
            0 < budget < 1
        ):
            raise RuntimeError(
                f"Budget must be between 0 and 1: {budget}"
            )

    predictions_path = (
        args.e3_dir
        / "e3_oof_predictions.csv"
    )

    selection_path = (
        args.e3_dir
        / "e3_top5_selection_by_outer_fold.csv"
    )

    prediction_rows = read_csv(
        predictions_path
    )

    selection_rows = read_csv(
        selection_path
    )

    if len(prediction_rows) != 799:
        raise RuntimeError(
            f"Expected 799 APKs, got {len(prediction_rows)}"
        )

    if len({
        norm_sha(
            row["sha256"]
        )
        for row in prediction_rows
    }) != 799:
        raise RuntimeError(
            "Duplicate SHA256 in E3 predictions"
        )

    manifest = []

    for row in prediction_rows:
        manifest.append({
            "sha256":
                norm_sha(
                    row[
                        "sha256"
                    ]
                ),

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

    labels = np.asarray(
        [
            row["label"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    folds = np.asarray(
        [
            row["fold"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    coarse_probability = np.asarray(
        [
            row[
                "coarse_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    structure_probability = np.asarray(
        [
            row[
                "structure_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    sha_values = [
        row["sha256"]
        for row in manifest
    ]

    print(
        "========================================"
    )
    print(
        "BenefitDroid 主线 E4"
    )
    print(
        "固定分析预算公平路由"
    )
    print(
        "========================================"
    )

    print(
        "APK数量:",
        len(manifest),
    )

    print(
        "预算:",
        budgets,
    )

    print(
        "随机重复:",
        args.random_repeats,
    )

    # --------------------------------------------------------
    # 读取31类语义存在信息
    # --------------------------------------------------------

    roles = get_roles(
        args.semantic_mapping
    )

    role_to_index = {
        role: i
        for i, role in enumerate(
            roles
        )
    }

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
            compact_by_sha[
                sha
            ] = row

    missing = [
        sha
        for sha in sha_values
        if sha not in compact_by_sha
    ]

    if missing:
        raise RuntimeError(
            "Missing compact rows: "
            + repr(
                missing[:10]
            )
        )

    presence = role_presence_matrix(
        manifest,
        compact_by_sha,
        roles,
    )

    # --------------------------------------------------------
    # E3每一个外层测试折自己的Top5
    # 注意：这些Top5在E3中只由另外4折产生。
    # --------------------------------------------------------

    selected_by_fold = defaultdict(
        lambda: defaultdict(list)
    )

    for row in selection_rows:
        fold = int(
            row[
                "outer_fold"
            ]
        )

        method = row[
            "method"
        ]

        selected_by_fold[
            fold
        ][
            method
        ].append(
            row
        )

    for fold in sorted(
        np.unique(
            folds
        )
    ):
        fold = int(
            fold
        )

        for method in (
            "Top5_Error",
            "Top5_Benefit",
        ):
            rows = selected_by_fold[
                fold
            ][
                method
            ]

            rows.sort(
                key=lambda row: int(
                    row[
                        "rank"
                    ]
                )
            )

            if len(rows) != 5:
                raise RuntimeError(
                    f"Fold {fold} {method} "
                    f"expected 5 roles, got {len(rows)}"
                )

    # --------------------------------------------------------
    # 为每个APK生成两种便宜的路由分数。
    #
    # 错误Top5:
    #   当前APK包含某个Top5角色时，
    #   加上该角色在训练部分的第一层错误数量。
    #
    # 收益Top5:
    #   当前APK包含某个Top5角色时，
    #   加上该角色的结构净收益得分。
    # --------------------------------------------------------

    error_score = np.zeros(
        len(manifest),
        dtype=np.float64,
    )

    benefit_score = np.zeros(
        len(manifest),
        dtype=np.float64,
    )

    route_score_rows = []

    for fold in sorted(
        np.unique(
            folds
        )
    ):
        fold = int(
            fold
        )

        test_indices = np.flatnonzero(
            folds == fold
        )

        error_roles = selected_by_fold[
            fold
        ][
            "Top5_Error"
        ]

        benefit_roles = selected_by_fold[
            fold
        ][
            "Top5_Benefit"
        ]

        for index in test_indices:

            e_score = 0.0
            b_score = 0.0

            error_hit = []
            benefit_hit = []

            for row in error_roles:
                role = row[
                    "role"
                ]

                role_index = (
                    role_to_index[
                        role
                    ]
                )

                if presence[
                    index,
                    role_index
                ]:
                    e_score += float(
                        row[
                            "coarse_error_count"
                        ]
                    )

                    error_hit.append(
                        ROLE_ZH.get(
                            role,
                            role,
                        )
                    )

            for row in benefit_roles:
                role = row[
                    "role"
                ]

                role_index = (
                    role_to_index[
                        role
                    ]
                )

                if presence[
                    index,
                    role_index
                ]:
                    b_score += float(
                        row[
                            "benefit_score"
                        ]
                    )

                    benefit_hit.append(
                        ROLE_ZH.get(
                            role,
                            role,
                        )
                    )

            error_score[
                index
            ] = e_score

            benefit_score[
                index
            ] = b_score

            route_score_rows.append({
                "sha256":
                    sha_values[
                        index
                    ],

                "fold":
                    fold,

                "label":
                    int(
                        labels[
                            index
                        ]
                    ),

                "coarse_probability":
                    float(
                        coarse_probability[
                            index
                        ]
                    ),

                "uncertainty":
                    float(
                        0.5
                        - abs(
                            coarse_probability[
                                index
                            ]
                            - 0.5
                        )
                    ),

                "error_top5_score":
                    float(
                        e_score
                    ),

                "benefit_top5_score":
                    float(
                        b_score
                    ),

                "error_top5_hits":
                    "|".join(
                        error_hit
                    ),

                "benefit_top5_hits":
                    "|".join(
                        benefit_hit
                    ),
            })

    # --------------------------------------------------------
    # 低置信度：
    # 越靠近0.5，分数越高。
    # --------------------------------------------------------

    uncertainty_score = (
        -np.abs(
            coarse_probability
            - 0.5
        )
    )

    # --------------------------------------------------------
    # Oracle理论上界。
    #
    # +1: 第一层错，结构对
    #  0: 最终分类结果不变
    # -1: 第一层对，结构错
    #
    # 这使用真实标签，只作理论上界，
    # 不属于可部署方法。
    # --------------------------------------------------------

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
        len(manifest),
        dtype=np.float64,
    )

    fix_mask = (
        (coarse_pred != labels)
        & (structure_pred == labels)
    )

    break_mask = (
        (coarse_pred == labels)
        & (structure_pred != labels)
    )

    oracle_score[
        fix_mask
    ] = 1.0

    oracle_score[
        break_mask
    ] = -1.0

    print()
    print(
        "结构模型相对第一层:"
    )

    print(
        "可修复APK:",
        int(
            np.sum(
                fix_mask
            )
        ),
    )

    print(
        "会破坏APK:",
        int(
            np.sum(
                break_mask
            )
        ),
    )

    # --------------------------------------------------------
    # 基础模型与全量结构参考
    # --------------------------------------------------------

    base_reference = evaluate_route(
        labels,
        coarse_probability,
        structure_probability,
        [],
    )

    full_reference = evaluate_route(
        labels,
        coarse_probability,
        structure_probability,
        np.arange(
            len(manifest)
        ),
    )

    print()
    print(
        "第一层 ACC:",
        f"{base_reference['ACC']:.6f}",
    )

    print(
        "全部结构 ACC:",
        f"{full_reference['ACC']:.6f}",
    )

    # --------------------------------------------------------
    # 等预算实验
    # --------------------------------------------------------

    deterministic_rows = []
    random_rows = []
    summary_rows = []

    strategy_scores = {
        "低置信度":
            uncertainty_score,

        "错误Top5":
            error_score,

        "收益Top5":
            benefit_score,

        "Oracle理论上界":
            oracle_score,
    }

    for budget in budgets:

        print()
        print(
            "========================================"
        )
        print(
            f"分析预算: {budget:.0%}"
        )
        print(
            "========================================"
        )

        # ----------------------------------------------------
        # 随机基线
        # ----------------------------------------------------

        random_metric_rows = []

        for seed in range(
            args.random_repeats
        ):
            selected = (
                select_random_by_fold(
                    folds,
                    budget,
                    seed,
                )
            )

            result = evaluate_route(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            row = {
                "budget":
                    budget,

                "random_seed":
                    seed,

                **result,
            }

            random_rows.append(
                row
            )

            random_metric_rows.append(
                row
            )

        random_summary = {
            "method":
                "随机选择",

            "budget":
                budget,

            "routed_n":
                float(
                    np.mean([
                        row[
                            "routed_n"
                        ]
                        for row
                        in random_metric_rows
                    ])
                ),

            "routed_ratio":
                float(
                    np.mean([
                        row[
                            "routed_ratio"
                        ]
                        for row
                        in random_metric_rows
                    ])
                ),
        }

        for metric in (
            "ACC",
            "BA",
            "AUC",
            "F1",
            "fixed",
            "broken",
            "net",
            "fixed_false_negative",
            "fixed_false_positive",
            "introduced_false_negative",
            "introduced_false_positive",
            "net_per_100_routed",
        ):
            values = np.asarray(
                [
                    row[
                        metric
                    ]
                    for row
                    in random_metric_rows
                ],
                dtype=float,
            )

            random_summary[
                metric
            ] = float(
                np.mean(
                    values
                )
            )

            random_summary[
                metric
                + "_std"
            ] = float(
                np.std(
                    values
                )
            )

        summary_rows.append(
            random_summary
        )

        print(
            "随机选择      ",
            f"分析={random_summary['routed_n']:.0f}",
            f"ACC={random_summary['ACC']:.6f}",
            f"BA={random_summary['BA']:.6f}",
            f"净修复={random_summary['net']:+.2f}",
            f"每100个={random_summary['net_per_100_routed']:.2f}",
        )

        # ----------------------------------------------------
        # 四个确定性策略
        # ----------------------------------------------------

        for method, score in (
            strategy_scores.items()
        ):
            selected = (
                select_by_fold_score(
                    folds,
                    score,
                    sha_values,
                    budget,
                )
            )

            result = evaluate_route(
                labels,
                coarse_probability,
                structure_probability,
                selected,
            )

            row = {
                "method":
                    method,

                "budget":
                    budget,

                **result,
            }

            deterministic_rows.append(
                row
            )

            summary_row = dict(
                row
            )

            for metric in (
                "ACC",
                "BA",
                "AUC",
                "F1",
                "fixed",
                "broken",
                "net",
                "fixed_false_negative",
                "fixed_false_positive",
                "introduced_false_negative",
                "introduced_false_positive",
                "net_per_100_routed",
            ):
                summary_row[
                    metric
                    + "_std"
                ] = 0.0

            summary_rows.append(
                summary_row
            )

            print(
                f"{method:12s}",
                f"分析={result['routed_n']:3d}",
                f"ACC={result['ACC']:.6f}",
                f"BA={result['BA']:.6f}",
                f"修复={result['fixed']:2d}",
                f"破坏={result['broken']:2d}",
                f"净修复={result['net']:+3d}",
                f"每100个={result['net_per_100_routed']:.2f}",
            )

    # --------------------------------------------------------
    # 加入0%和100%两个参考点
    # --------------------------------------------------------

    reference_rows = [
        {
            "method":
                "仅第一层",

            "budget":
                0.0,

            **base_reference,
        },

        {
            "method":
                "全部做结构",

            "budget":
                1.0,

            **full_reference,
        },
    ]

    # --------------------------------------------------------
    # 输出文件
    # --------------------------------------------------------

    write_csv(
        args.output
        / "e4_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "e4_deterministic_results.csv",
        deterministic_rows,
    )

    write_csv(
        args.output
        / "e4_random_runs.csv",
        random_rows,
    )

    write_csv(
        args.output
        / "e4_route_scores.csv",
        route_score_rows,
    )

    write_csv(
        args.output
        / "e4_reference_points.csv",
        reference_rows,
    )

    audit = {
        "source_apk_count":
            len(manifest),

        "budgets":
            budgets,

        "random_repeats":
            args.random_repeats,

        "budget_enforcement":
            (
                "Within each fixed outer fold, exactly round("
                "budget * fold_size) APKs are routed."
            ),

        "structure_prediction":
            (
                "2017 OOF Compact70 + Stats120 + PairRaw465"
            ),

        "top5_source":
            (
                "E3 nested source-only Top5 selection; "
                "each outer test fold uses Top5 learned "
                "without its labels."
            ),

        "error_top5_score":
            (
                "sum of training-only coarse error counts "
                "for selected Top5-Error roles present in APK"
            ),

        "benefit_top5_score":
            (
                "sum of training-only benefit scores "
                "for selected Top5-Benefit roles present in APK"
            ),

        "low_confidence_score":
            "-abs(coarse_probability - 0.5)",

        "oracle":
            (
                "uses test labels only as theoretical upper bound; "
                "+1 for fix, 0 neutral, -1 for break"
            ),
    }

    (
        args.output
        / "e4_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 最后给一个紧凑汇总
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "E4 最终预算曲线"
    )
    print(
        "========================================"
    )

    methods = [
        "随机选择",
        "低置信度",
        "错误Top5",
        "收益Top5",
        "Oracle理论上界",
    ]

    for budget in budgets:
        print()
        print(
            f"预算 {budget:.0%}:"
        )

        for method in methods:
            candidates = [
                row
                for row in summary_rows
                if row[
                    "method"
                ] == method
                and abs(
                    float(
                        row[
                            "budget"
                        ]
                    )
                    - budget
                ) < 1e-12
            ]

            if not candidates:
                continue

            row = candidates[0]

            print(
                f"  {method:12s}",
                f"ACC={float(row['ACC']):.6f}",
                f"BA={float(row['BA']):.6f}",
                f"净修复={float(row['net']):+.2f}",
                f"每100个={float(row['net_per_100_routed']):.2f}",
            )

    print()
    print(
        "输出目录:",
        args.output,
    )


if __name__ == "__main__":
    main()
