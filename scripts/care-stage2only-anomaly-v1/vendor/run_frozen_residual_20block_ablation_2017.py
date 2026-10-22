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
# 必须在导入20+150封装脚本之前保存原始类和数据构造，
# 因为20+150脚本导入时会替换它们。
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
# 再导入已经跑通的20+150实现
# ============================================================

import run_frozen_anchor_residual_20_150_2017_2021 as aug


OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-residual-20block-ablation-2017"
)

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]


# ============================================================
# 固定Top5及20维定义
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

DEPTH_METRICS = (
    "mean_depth_norm",
    "deep_caller_ratio",
)

ALL_METRICS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)


def make_names(metrics):
    return [
        f"up_role__{role}__{metric}"
        for role in ROLES
        for metric in metrics
    ]


COUNT10_NAMES = make_names(
    COUNT_METRICS
)

DEPTH10_NAMES = make_names(
    DEPTH_METRICS
)

ALL20_NAMES = make_names(
    ALL_METRICS
)


if len(COUNT10_NAMES) != 10:
    raise RuntimeError(
        "数量块不是10维"
    )

if len(DEPTH10_NAMES) != 10:
    raise RuntimeError(
        "深度块不是10维"
    )

if len(ALL20_NAMES) != 20:
    raise RuntimeError(
        "完整结构块不是20维"
    )


ABLATIONS = (
    {
        "name":
            "frozen_150",

        "structure_names":
            (),
    },
    {
        "name":
            "frozen_count10_150",

        "structure_names":
            tuple(
                COUNT10_NAMES
            ),
    },
    {
        "name":
            "frozen_depth10_150",

        "structure_names":
            tuple(
                DEPTH10_NAMES
            ),
    },
    {
        "name":
            "frozen_all20_150",

        "structure_names":
            tuple(
                ALL20_NAMES
            ),
    },
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
                fields.append(key)

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


def configure_ablation(
    structure_names,
):
    """
    只切换结构块。

    150基线：
        使用原 Frozen+150 网络和数据构造。

    10/20维：
        使用已经验证过的 Frozen+20+150 实现，
        但把结构维数和列名改成当前消融块。

    presence始终由原150维决定。
    """

    if not structure_names:
        base.build_fold_data = (
            ORIGINAL_BUILD_FOLD_DATA
        )

        base.FrozenAnchorResidualNet = (
            ORIGINAL_NET
        )

        return

    structure_names = list(
        structure_names
    )

    aug.STRUCTURE_DIM = len(
        structure_names
    )

    aug.OLD20_NAMES = (
        structure_names
    )

    base.build_fold_data = (
        aug.build_fold_data_20_150
    )

    base.FrozenAnchorResidualNet = (
        aug.FrozenAnchorResidual20_150Net
    )


def get_config(
    name,
):
    """
    只使用当前已经固定的网络训练参数。

    不对四个消融分别调参数。
    """

    if len(ORIGINAL_CONFIGS) != 1:
        raise SystemExit(
            "预期 Frozen 基础脚本只有一个配置，"
            f"实际={len(ORIGINAL_CONFIGS)}"
        )

    config = copy.deepcopy(
        ORIGINAL_CONFIGS[0]
    )

    config["name"] = name

    return config


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 只读取2017
    #
    # 明确不读取 semantic_future / future_splits。
    # ========================================================

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
        aug.OLD20_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在: "
                + str(path)
            )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

        torch.backends.cudnn.deterministic = (
            True
        )

        torch.backends.cudnn.benchmark = (
            False
        )

    else:
        device = torch.device(
            "cpu"
        )

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

    if len(names150) != 150:
        raise SystemExit(
            "150维数量异常: "
            + str(
                len(names150)
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
            and parts["train"]
            and parts["test"]
        )
    }

    if len(
        rolling_splits
    ) != 3:
        raise SystemExit(
            "2017滚动划分数量异常: "
            + str(
                len(
                    rolling_splits
                )
            )
        )

    # 先强制读取正式旧20维一次，
    # 后续10/20维全部来自同一个文件。
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
                    str(
                        row[
                            "sha256"
                        ]
                    )
                    .strip()
                    .casefold()
                )

    missing_old20 = (
        required_shas
        - set(
            old20_by_sha
        )
    )

    if missing_old20:
        raise SystemExit(
            "正式旧20维缺失2017样本: "
            + str(
                len(
                    missing_old20
                )
            )
        )

    print(
        "======================================"
    )

    print(
        "Frozen residual 2017 structure ablation"
    )

    print(
        "======================================"
    )

    print(
        "device =",
        device,
    )

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
    )

    print(
        "seeds =",
        SEEDS,
    )

    print(
        "threshold =",
        base.THRESHOLD,
    )

    print(
        "150维 =",
        len(names150),
    )

    print(
        "正式结构来源 =",
        aug.OLD20_PATH,
    )

    print(
        "未来数据读取 = 否"
    )

    print()

    cv_rows = []

    summary_rows = []

    # ========================================================
    # 四个消融
    # ========================================================

    for ablation in ABLATIONS:
        name = ablation[
            "name"
        ]

        structure_names = (
            ablation[
                "structure_names"
            ]
        )

        configure_ablation(
            structure_names
        )

        config = get_config(
            name
        )

        print()
        print(
            "======================================"
        )

        print(
            "模型 =",
            name,
        )

        print(
            "结构维数 =",
            len(
                structure_names
            ),
        )

        if structure_names:
            print(
                "结构列 ="
            )

            for column in (
                structure_names
            ):
                print(
                    " ",
                    column,
                )

        print(
            "======================================"
        )

        current_rows = []

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

            for seed in SEEDS:
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

                # 与原源域选择逻辑一致：
                # 当前训练轨迹内取最高BA。
                best = max(
                    history,
                    key=lambda row:
                        row[
                            "balanced_accuracy"
                        ],
                )

                row = {
                    "model":
                        name,

                    "structure_dim":
                        len(
                            structure_names
                        ),

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "best_epoch":
                        best_epoch,

                    "BA":
                        float(
                            best[
                                "balanced_accuracy"
                            ]
                        ),

                    "AUC":
                        float(
                            best[
                                "auc"
                            ]
                        ),

                    "ACC":
                        float(
                            best[
                                "accuracy"
                            ]
                        ),

                    "F1":
                        float(
                            best[
                                "f1"
                            ]
                        ),

                    "FP":
                        int(
                            best[
                                "fp"
                            ]
                        ),

                    "FN":
                        int(
                            best[
                                "fn"
                            ]
                        ),
                }

                cv_rows.append(
                    row
                )

                current_rows.append(
                    row
                )

                print(
                    f"{split_name} "
                    f"seed={seed} "
                    f"epoch={best_epoch} "
                    f"BA={row['BA']:.6f} "
                    f"AUC={row['AUC']:.6f}",
                    flush=True,
                )

        # ====================================================
        # 与原脚本一样：
        # BA先按split内种子平均，
        # 再跨三个split取平均和最小。
        # ====================================================

        by_split = defaultdict(
            list
        )

        for row in current_rows:
            by_split[
                row[
                    "split_name"
                ]
            ].append(
                row
            )

        split_ba = []

        split_auc = []

        split_acc = []

        for split_name in sorted(
            by_split
        ):
            rows = by_split[
                split_name
            ]

            split_ba.append(
                float(
                    np.mean([
                        row[
                            "BA"
                        ]
                        for row in rows
                    ])
                )
            )

            split_auc.append(
                float(
                    np.mean([
                        row[
                            "AUC"
                        ]
                        for row in rows
                    ])
                )
            )

            split_acc.append(
                float(
                    np.mean([
                        row[
                            "ACC"
                        ]
                        for row in rows
                    ])
                )
            )

        summary = {
            "model":
                name,

            "structure_dim":
                len(
                    structure_names
                ),

            "mean_BA":
                float(
                    np.mean(
                        split_ba
                    )
                ),

            "min_split_BA":
                float(
                    np.min(
                        split_ba
                    )
                ),

            "std_split_BA":
                float(
                    np.std(
                        split_ba
                    )
                ),

            "mean_AUC":
                float(
                    np.mean([
                        row[
                            "AUC"
                        ]
                        for row
                        in current_rows
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        row[
                            "ACC"
                        ]
                        for row
                        in current_rows
                    ])
                ),

            "median_best_epoch":
                int(
                    round(
                        np.median([
                            row[
                                "best_epoch"
                            ]
                            for row
                            in current_rows
                        ])
                    )
                ),
        }

        summary_rows.append(
            summary
        )

        print()
        print(
            "汇总：",
            name,
        )

        print(
            f" BA={summary['mean_BA']:.6f}"
            f" minBA={summary['min_split_BA']:.6f}"
            f" stdBA={summary['std_split_BA']:.6f}"
            f" AUC={summary['mean_AUC']:.6f}"
            f" ACC={summary['mean_ACC']:.6f}"
            f" epoch={summary['median_best_epoch']}"
        )

    # ========================================================
    # 保存原始结果
    # ========================================================

    write_csv(
        OUTPUT_DIR
        / "cv_rows.csv",
        cv_rows,
    )

    # 保持预先定义顺序，
    # 不按照结果排序后做新的模型搜索。
    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary_rows,
    )

    # ========================================================
    # 相对150基线的增量
    # ========================================================

    baseline = next(
        row
        for row in summary_rows
        if row[
            "model"
        ] == "frozen_150"
    )

    delta_rows = []

    for row in summary_rows:
        delta_rows.append({
            "model":
                row[
                    "model"
                ],

            "dBA_vs_150":
                row[
                    "mean_BA"
                ]
                - baseline[
                    "mean_BA"
                ],

            "dMinBA_vs_150":
                row[
                    "min_split_BA"
                ]
                - baseline[
                    "min_split_BA"
                ],

            "dAUC_vs_150":
                row[
                    "mean_AUC"
                ]
                - baseline[
                    "mean_AUC"
                ],

            "dACC_vs_150":
                row[
                    "mean_ACC"
                ]
                - baseline[
                    "mean_ACC"
                ],
        })

    write_csv(
        OUTPUT_DIR
        / "delta_vs_150.csv",
        delta_rows,
    )

    print()
    print(
        "======================================"
    )

    print(
        "2017-only最终消融"
    )

    print(
        "======================================"
    )

    for row in summary_rows:
        delta = next(
            d
            for d in delta_rows
            if d[
                "model"
            ] == row[
                "model"
            ]
        )

        print(
            f"{row['model']:25s} "
            f"BA={row['mean_BA']:.6f} "
            f"minBA={row['min_split_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"epoch={row['median_best_epoch']:3d} "
            f"dBA={delta['dBA_vs_150']:+.6f} "
            f"dMin={delta['dMinBA_vs_150']:+.6f} "
            f"dAUC={delta['dAUC_vs_150']:+.6f}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )

    print(
        "本脚本未读取2018-2021语义文件或未来划分。"
    )


if __name__ == "__main__":
    main()
