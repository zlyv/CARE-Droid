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
)

from run_pilot_compact_eval import (
    fit_logistic,
)

from run_compact_20_150_ablation_2017 import (
    support150_names,
)


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

OLD20_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/source-presence-group-diagnostic-2017"
)

SEEDS = [0, 1, 2, 3, 4]

THRESHOLD = 0.5

OLD20_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

OLD20_METRICS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)

OLD20_NAMES = [
    f"up_role__{role}__{metric}"
    for role in OLD20_ROLES
    for metric in OLD20_METRICS
]


def read_gz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def normalize_sha(value):
    return str(
        value
    ).strip().casefold()


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
        writer.writerows(rows)


def sigmoid(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    out = np.empty_like(
        x,
        dtype=float,
    )

    positive = x >= 0

    out[positive] = (
        1.0
        /
        (
            1.0
            + np.exp(
                -x[positive]
            )
        )
    )

    exp_x = np.exp(
        x[~positive]
    )

    out[~positive] = (
        exp_x
        /
        (
            1.0
            + exp_x
        )
    )

    return out


def raw_present(
    row,
    names,
):
    return int(
        any(
            abs(
                float(
                    row[name]
                )
            ) > 1e-12
            for name in names
        )
    )


def group_name(
    structure_present,
    semantic_present,
):
    if (
        not structure_present
        and not semantic_present
    ):
        return "A_20zero_150zero"

    if (
        structure_present
        and not semantic_present
    ):
        return "B_20present_150zero"

    if (
        not structure_present
        and semantic_present
    ):
        return "C_20zero_150present"

    return "D_20present_150present"


def margin_bin(margin):
    if margin < 0.05:
        return "0-0.05"

    if margin < 0.10:
        return "0.05-0.10"

    if margin < 0.20:
        return "0.10-0.20"

    if margin < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def safe_mean(values):
    if not values:
        return float("nan")

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def safe_median(values):
    if not values:
        return float("nan")

    return float(
        np.median(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        SEMANTIC_PATH,
        OLD20_PATH,
        SPLIT_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在: "
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 读取
    # ========================================================

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        normalize_sha(
            row["sha256"]
        ): row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        normalize_sha(
            row["sha256"]
        ): row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    if len(names150) != 150:
        raise SystemExit(
            "150维列数异常: "
            + str(
                len(names150)
            )
        )

    old20_rows = read_gz(
        OLD20_PATH
    )

    old20_by_sha = {
        normalize_sha(
            row["sha256"]
        ): row
        for row in old20_rows
    }

    if len(OLD20_NAMES) != 20:
        raise RuntimeError(
            "20维定义异常"
        )

    missing_old20_columns = (
        set(OLD20_NAMES)
        - set(
            old20_rows[0].keys()
        )
    )

    if missing_old20_columns:
        raise SystemExit(
            "旧20维文件缺少列: "
            + ", ".join(
                sorted(
                    missing_old20_columns
                )
            )
        )

    splits_all = read_split_rows(
        SPLIT_PATH
    )

    splits = {
        name:
            parts
        for name, parts
        in splits_all.items()
        if (
            "rolling" in name
            and parts["train"]
            and parts["test"]
        )
    }

    if len(splits) != 3:
        raise SystemExit(
            "2017滚动折数量异常: "
            + str(
                len(splits)
            )
        )

    # ========================================================
    # 人口检查
    # ========================================================

    required_shas = set()

    for parts in splits.values():
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                required_shas.add(
                    normalize_sha(
                        row["sha256"]
                    )
                )

    for name, table in (
        (
            "Compact",
            compact_by_sha,
        ),
        (
            "150维",
            semantic_by_sha,
        ),
        (
            "旧20维",
            old20_by_sha,
        ),
    ):
        missing = (
            required_shas
            - set(table)
        )

        print(
            name,
            "缺失 =",
            len(missing),
        )

        if missing:
            for sha in sorted(
                missing
            )[:20]:
                print(
                    " ",
                    sha,
                )

            raise SystemExit(
                name
                + "人口不完整"
            )

    print()
    print(
        "2017参与唯一SHA =",
        len(required_shas),
    )

    print(
        "旧20维来源 =",
        OLD20_PATH,
    )

    print(
        "20维数 =",
        len(OLD20_NAMES),
    )

    print(
        "150维数 =",
        len(names150),
    )

    # ========================================================
    # Compact基础预测
    # ========================================================

    prediction_rows = []
    sample_presence = {}

    for split_name in sorted(
        splits
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        test_rows = (
            splits[
                split_name
            ]["test"]
        )

        (
            x_train,
            x_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            schema,
        )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        normalize_sha(
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

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        normalize_sha(
                            row["sha256"]
                        )
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        # ----------------------------------------------
        # 每个测试APK确定20/150原始存在状态
        # ----------------------------------------------

        for row, label in zip(
            test_rows,
            y_test,
        ):
            sha = normalize_sha(
                row["sha256"]
            )

            p20 = raw_present(
                old20_by_sha[sha],
                OLD20_NAMES,
            )

            p150 = raw_present(
                semantic_by_sha[sha],
                names150,
            )

            sample_presence[
                (
                    split_name,
                    sha,
                )
            ] = {
                "split_name":
                    split_name,

                "sha256":
                    sha,

                "label":
                    int(label),

                "presence20":
                    p20,

                "presence150":
                    p150,

                "group":
                    group_name(
                        p20,
                        p150,
                    ),
            }

        # ----------------------------------------------
        # 完全按原Compact协议
        # ----------------------------------------------

        for seed in SEEDS:
            w, b = fit_logistic(
                x_train,
                y_train,
                seed,
                iterations=6000,
                learning_rate=0.05,
                l2=0.03,
            )

            probability = sigmoid(
                x_test @ w + b
            )

            for i, row in enumerate(
                test_rows
            ):
                sha = normalize_sha(
                    row["sha256"]
                )

                info = sample_presence[
                    (
                        split_name,
                        sha,
                    )
                ]

                label = int(
                    y_test[i]
                )

                p = float(
                    probability[i]
                )

                pred = int(
                    p >= THRESHOLD
                )

                correct = int(
                    pred == label
                )

                margin = abs(
                    p
                    - THRESHOLD
                )

                uncertainty = (
                    4.0
                    * p
                    * (
                        1.0
                        - p
                    )
                )

                prediction_rows.append({
                    **info,

                    "seed":
                        seed,

                    "base_probability":
                        p,

                    "base_prediction":
                        pred,

                    "base_correct":
                        correct,

                    "base_margin":
                        margin,

                    "base_uncertainty":
                        uncertainty,

                    "margin_bin":
                        margin_bin(
                            margin
                        ),
                })

        print(
            split_name,
            "train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "完成",
        )

    # ========================================================
    # 每个样本先对5个种子平均概率
    # ========================================================

    by_sample = defaultdict(
        list
    )

    for row in prediction_rows:
        by_sample[
            (
                row["split_name"],
                row["sha256"],
            )
        ].append(row)

    sample_rows = []

    for key, current in sorted(
        by_sample.items()
    ):
        first = current[0]

        p = safe_mean([
            row[
                "base_probability"
            ]
            for row in current
        ])

        label = int(
            first["label"]
        )

        pred = int(
            p >= THRESHOLD
        )

        margin = abs(
            p
            - THRESHOLD
        )

        uncertainty = (
            4.0
            * p
            * (
                1.0
                - p
            )
        )

        sample_rows.append({
            "split_name":
                first[
                    "split_name"
                ],

            "sha256":
                first[
                    "sha256"
                ],

            "label":
                label,

            "presence20":
                first[
                    "presence20"
                ],

            "presence150":
                first[
                    "presence150"
                ],

            "group":
                first[
                    "group"
                ],

            "mean_base_probability":
                p,

            "base_prediction":
                pred,

            "base_correct":
                int(
                    pred == label
                ),

            "base_margin":
                margin,

            "base_uncertainty":
                uncertainty,

            "margin_bin":
                margin_bin(
                    margin
                ),
        })

    # ========================================================
    # 四组汇总
    # ========================================================

    group_order = (
        "A_20zero_150zero",
        "B_20present_150zero",
        "C_20zero_150present",
        "D_20present_150present",
    )

    group_summary = []

    for group in group_order:
        current = [
            row
            for row in sample_rows
            if row["group"]
            == group
        ]

        wrong = [
            row
            for row in current
            if not int(
                row[
                    "base_correct"
                ]
            )
        ]

        benign = [
            row
            for row in current
            if int(
                row["label"]
            ) == 0
        ]

        malware = [
            row
            for row in current
            if int(
                row["label"]
            ) == 1
        ]

        fp = sum(
            int(row["label"]) == 0
            and
            int(
                row[
                    "base_prediction"
                ]
            ) == 1
            for row in current
        )

        fn = sum(
            int(row["label"]) == 1
            and
            int(
                row[
                    "base_prediction"
                ]
            ) == 0
            for row in current
        )

        wrong_lt020 = sum(
            float(
                row[
                    "base_margin"
                ]
            ) < 0.20
            for row in wrong
        )

        group_summary.append({
            "group":
                group,

            "n":
                len(current),

            "benign":
                len(benign),

            "malware":
                len(malware),

            "base_wrong":
                len(wrong),

            "base_error_rate":
                (
                    len(wrong)
                    / len(current)
                    if current
                    else float("nan")
                ),

            "FP":
                fp,

            "FN":
                fn,

            "wrong_margin_mean":
                safe_mean([
                    row[
                        "base_margin"
                    ]
                    for row in wrong
                ]),

            "wrong_margin_median":
                safe_median([
                    row[
                        "base_margin"
                    ]
                    for row in wrong
                ]),

            "wrong_uncertainty_mean":
                safe_mean([
                    row[
                        "base_uncertainty"
                    ]
                    for row in wrong
                ]),

            "wrong_margin_lt_020":
                wrong_lt020,

            "wrong_margin_lt_020_rate":
                (
                    wrong_lt020
                    / len(wrong)
                    if wrong
                    else float("nan")
                ),
        })

    # ========================================================
    # B组错误的边界距离
    # ========================================================

    b_wrong = [
        row
        for row in sample_rows
        if (
            row["group"]
            == "B_20present_150zero"
            and
            not int(
                row[
                    "base_correct"
                ]
            )
        )
    ]

    margin_order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    b_margin_rows = []

    for name in margin_order:
        current = [
            row
            for row in b_wrong
            if row[
                "margin_bin"
            ] == name
        ]

        b_margin_rows.append({
            "margin_bin":
                name,

            "wrong_n":
                len(current),

            "benign_FP":
                sum(
                    int(
                        row[
                            "label"
                        ]
                    ) == 0
                    for row in current
                ),

            "malware_FN":
                sum(
                    int(
                        row[
                            "label"
                        ]
                    ) == 1
                    for row in current
                ),

            "mean_uncertainty":
                safe_mean([
                    row[
                        "base_uncertainty"
                    ]
                    for row in current
                ]),
        })

    # ========================================================
    # 每个滚动折的B组
    # ========================================================

    b_split_rows = []

    for split_name in sorted(
        splits
    ):
        current = [
            row
            for row in sample_rows
            if (
                row[
                    "split_name"
                ] == split_name
                and
                row[
                    "group"
                ] == "B_20present_150zero"
            )
        ]

        wrong = [
            row
            for row in current
            if not int(
                row[
                    "base_correct"
                ]
            )
        ]

        b_split_rows.append({
            "split_name":
                split_name,

            "n":
                len(current),

            "benign":
                sum(
                    int(
                        row["label"]
                    ) == 0
                    for row in current
                ),

            "malware":
                sum(
                    int(
                        row["label"]
                    ) == 1
                    for row in current
                ),

            "wrong":
                len(wrong),

            "FP":
                sum(
                    int(
                        row["label"]
                    ) == 0
                    for row in wrong
                ),

            "FN":
                sum(
                    int(
                        row["label"]
                    ) == 1
                    for row in wrong
                ),

            "wrong_margin_lt_020":
                sum(
                    float(
                        row[
                            "base_margin"
                        ]
                    ) < 0.20
                    for row in wrong
                ),
        })

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        OUTPUT_DIR
        / "per_seed_predictions.csv",
        prediction_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_sample_mean.csv",
        sample_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "group_summary.csv",
        group_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "group_B_margin.csv",
        b_margin_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "group_B_per_split.csv",
        b_split_rows,
    )

    # ========================================================
    # 打印
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "2017四组诊断"
    )
    print(
        "======================================"
    )

    for row in group_summary:
        print()
        print(
            row["group"]
        )

        print(
            f" N={row['n']} "
            f"良性={row['benign']} "
            f"恶意={row['malware']}"
        )

        print(
            f" Compact错误={row['base_wrong']} "
            f"错误率={row['base_error_rate']:.2%} "
            f"FP={row['FP']} "
            f"FN={row['FN']}"
        )

        if row[
            "base_wrong"
        ]:
            print(
                " 错误margin："
                f"mean={row['wrong_margin_mean']:.4f} "
                f"median={row['wrong_margin_median']:.4f}"
            )

            print(
                " 错误平均uncertainty="
                f"{row['wrong_uncertainty_mean']:.4f} "
                "|margin|<0.20="
                f"{row['wrong_margin_lt_020']}"
                "/"
                f"{row['base_wrong']} "
                f"({row['wrong_margin_lt_020_rate']:.2%})"
            )

    print()
    print(
        "======================================"
    )
    print(
        "最关键：B组错误距离0.5分布"
    )
    print(
        "20存在、150不存在"
    )
    print(
        "======================================"
    )

    for row in b_margin_rows:
        print(
            f"{row['margin_bin']:10s} "
            f"错误={row['wrong_n']:3d} "
            f"FP={row['benign_FP']:3d} "
            f"FN={row['malware_FN']:3d} "
            f"uncertainty="
            f"{row['mean_uncertainty']:.4f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "B组在三个滚动折"
    )
    print(
        "======================================"
    )

    for row in b_split_rows:
        print(
            row[
                "split_name"
            ]
        )

        print(
            f" N={row['n']} "
            f"良性={row['benign']} "
            f"恶意={row['malware']} "
            f"错误={row['wrong']} "
            f"FP={row['FP']} "
            f"FN={row['FN']} "
            f"错误中margin<0.20="
            f"{row['wrong_margin_lt_020']}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
