#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


THRESHOLD = 0.5

GATED_PATH = Path(
    "artifacts/gated-residual-neural-2017-2021/"
    "future_predictions.csv"
)

LINEAR_PATH = Path(
    "artifacts/frozen-20-150-2018-2021/"
    "predictions.csv"
)

SOURCE_PATH = Path(
    "artifacts/gated-residual-neural-2017-2021/"
    "source_cv_per_seed.csv"
)

OUTPUT_DIR = Path(
    "artifacts/gated-residual-mechanism-diagnostic"
)

GATED_CONFIG = "gated16_d10"
LINEAR_VARIANT = "compact_plus20_plus150"


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


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
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def mean(values):
    return float(
        np.mean(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def median(values):
    return float(
        np.median(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def is_correct(probability, label):
    pred = int(
        probability >= THRESHOLD
    )

    return pred == label


def truth_shift(p_new, p_old, label):
    """
    > 0:
        概率向真实标签方向移动
    < 0:
        概率背离真实标签
    """

    if label == 1:
        return p_new - p_old

    return p_old - p_new


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


def aggregate_gated(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            int(row["year"]),
            row["sha256"].strip().casefold(),
        )

        grouped[key].append(row)

    result = {}

    for key, current in grouped.items():
        labels = {
            int(row["label"])
            for row in current
        }

        if len(labels) != 1:
            raise RuntimeError(
                f"{key}: label不一致"
            )

        result[key] = {
            "label":
                labels.pop(),

            "gated_probability":
                mean([
                    float(
                        row["probability"]
                    )
                    for row in current
                ]),

            "base_probability":
                mean([
                    float(
                        row["base_probability"]
                    )
                    for row in current
                ]),

            "gate":
                mean([
                    float(
                        row["gate"]
                    )
                    for row in current
                ]),

            "semantic_delta":
                mean([
                    float(
                        row["semantic_delta"]
                    )
                    for row in current
                ]),

            "seed_count":
                len(current),
        }

    return result


def aggregate_linear(rows):
    if not rows:
        raise RuntimeError(
            "线性预测文件为空"
        )

    header = set(rows[0])

    if "variant" not in header:
        raise RuntimeError(
            "线性predictions.csv没有variant列，"
            f"实际列={sorted(header)}"
        )

    selected = [
        row
        for row in rows
        if row["variant"]
        == LINEAR_VARIANT
    ]

    if not selected:
        variants = sorted({
            row["variant"]
            for row in rows
        })

        raise RuntimeError(
            f"找不到 {LINEAR_VARIANT}，"
            f"实际variant={variants}"
        )

    grouped = defaultdict(list)

    for row in selected:
        key = (
            int(row["year"]),
            row["sha256"].strip().casefold(),
        )

        grouped[key].append(row)

    result = {}

    for key, current in grouped.items():
        labels = {
            int(row["label"])
            for row in current
        }

        if len(labels) != 1:
            raise RuntimeError(
                f"{key}: 线性label不一致"
            )

        result[key] = {
            "label":
                labels.pop(),

            "linear_probability":
                mean([
                    float(
                        row["probability"]
                    )
                    for row in current
                ]),

            "seed_count":
                len(current),
        }

    return result


def build_samples(
    gated,
    linear,
):
    gated_keys = set(gated)
    linear_keys = set(linear)

    missing_linear = (
        gated_keys
        - linear_keys
    )

    missing_gated = (
        linear_keys
        - gated_keys
    )

    print(
        "门控有、线性没有 =",
        len(missing_linear),
    )

    print(
        "线性有、门控没有 =",
        len(missing_gated),
    )

    common = sorted(
        gated_keys
        & linear_keys
    )

    rows = []

    for year, sha in common:
        g = gated[
            (
                year,
                sha,
            )
        ]

        l = linear[
            (
                year,
                sha,
            )
        ]

        if (
            g["label"]
            != l["label"]
        ):
            raise RuntimeError(
                f"{year}/{sha}: "
                "两个模型label不一致"
            )

        label = g["label"]

        p_linear = (
            l[
                "linear_probability"
            ]
        )

        p_gated = (
            g[
                "gated_probability"
            ]
        )

        p_base = (
            g[
                "base_probability"
            ]
        )

        linear_ok = is_correct(
            p_linear,
            label,
        )

        gated_ok = is_correct(
            p_gated,
            label,
        )

        base_ok = is_correct(
            p_base,
            label,
        )

        corrected = (
            (not linear_ok)
            and gated_ok
        )

        regressed = (
            linear_ok
            and (not gated_ok)
        )

        internal_corrected = (
            (not base_ok)
            and gated_ok
        )

        internal_regressed = (
            base_ok
            and (not gated_ok)
        )

        rows.append({
            "year":
                year,

            "sha256":
                sha,

            "label":
                label,

            "linear_probability":
                p_linear,

            "gated_probability":
                p_gated,

            "base_probability":
                p_base,

            "gate":
                g["gate"],

            "semantic_delta":
                g["semantic_delta"],

            "linear_margin":
                abs(
                    p_linear
                    - THRESHOLD
                ),

            "base_margin":
                abs(
                    p_base
                    - THRESHOLD
                ),

            "linear_correct":
                int(linear_ok),

            "gated_correct":
                int(gated_ok),

            "base_correct":
                int(base_ok),

            "corrected_vs_linear":
                int(corrected),

            "regressed_vs_linear":
                int(regressed),

            "internal_corrected":
                int(
                    internal_corrected
                ),

            "internal_regressed":
                int(
                    internal_regressed
                ),

            "shift_vs_linear":
                p_gated
                - p_linear,

            "truth_shift_vs_linear":
                truth_shift(
                    p_gated,
                    p_linear,
                    label,
                ),

            "shift_vs_internal_base":
                p_gated
                - p_base,

            "truth_shift_vs_internal_base":
                truth_shift(
                    p_gated,
                    p_base,
                    label,
                ),

            "abs_shift_vs_internal_base":
                abs(
                    p_gated
                    - p_base
                ),

            "margin_bin":
                margin_bin(
                    abs(
                        p_linear
                        - THRESHOLD
                    )
                ),
        })

    return rows


def summarize_scope(
    rows,
    scope,
):
    n = len(rows)

    corrected = [
        row
        for row in rows
        if int(
            row[
                "corrected_vs_linear"
            ]
        )
    ]

    regressed = [
        row
        for row in rows
        if int(
            row[
                "regressed_vs_linear"
            ]
        )
    ]

    linear_wrong = [
        row
        for row in rows
        if not int(
            row["linear_correct"]
        )
    ]

    linear_correct = [
        row
        for row in rows
        if int(
            row["linear_correct"]
        )
    ]

    internal_corrected = [
        row
        for row in rows
        if int(
            row[
                "internal_corrected"
            ]
        )
    ]

    internal_regressed = [
        row
        for row in rows
        if int(
            row[
                "internal_regressed"
            ]
        )
    ]

    shifts = np.asarray(
        [
            float(
                row[
                    "truth_shift_vs_linear"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    internal_shifts = np.asarray(
        [
            float(
                row[
                    "truth_shift_vs_internal_base"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    return {
        "scope":
            scope,

        "n":
            n,

        "linear_wrong":
            len(
                linear_wrong
            ),

        "corrected":
            len(corrected),

        "regressed":
            len(regressed),

        "net_corrected":
            len(corrected)
            - len(regressed),

        "benign_corrected":
            sum(
                int(row["label"]) == 0
                for row in corrected
            ),

        "malware_corrected":
            sum(
                int(row["label"]) == 1
                for row in corrected
            ),

        "benign_regressed":
            sum(
                int(row["label"]) == 0
                for row in regressed
            ),

        "malware_regressed":
            sum(
                int(row["label"]) == 1
                for row in regressed
            ),

        "truth_direction_rate_vs_linear":
            float(
                np.mean(
                    shifts > 0
                )
            ),

        "mean_truth_shift_vs_linear":
            float(
                shifts.mean()
            ),

        "internal_base_corrected":
            len(
                internal_corrected
            ),

        "internal_base_regressed":
            len(
                internal_regressed
            ),

        "truth_direction_rate_internal":
            float(
                np.mean(
                    internal_shifts
                    > 0
                )
            ),

        "mean_gate_all":
            mean([
                row["gate"]
                for row in rows
            ]),

        "mean_gate_linear_wrong":
            (
                mean([
                    row["gate"]
                    for row
                    in linear_wrong
                ])
                if linear_wrong
                else float("nan")
            ),

        "mean_gate_linear_correct":
            (
                mean([
                    row["gate"]
                    for row
                    in linear_correct
                ])
                if linear_correct
                else float("nan")
            ),

        "mean_gate_corrected":
            (
                mean([
                    row["gate"]
                    for row
                    in corrected
                ])
                if corrected
                else float("nan")
            ),

        "mean_gate_regressed":
            (
                mean([
                    row["gate"]
                    for row
                    in regressed
                ])
                if regressed
                else float("nan")
            ),

        "zero_gate_rate":
            float(
                np.mean([
                    float(
                        row["gate"]
                    ) <= 1e-12
                    for row in rows
                ])
            ),

        "mean_abs_internal_shift":
            mean([
                row[
                    "abs_shift_vs_internal_base"
                ]
                for row in rows
            ]),
    }


def margin_summary(
    rows,
    scope,
):
    order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    output = []

    for bin_name in order:
        current = [
            row
            for row in rows
            if row["margin_bin"]
            == bin_name
        ]

        wrong = [
            row
            for row in current
            if not int(
                row[
                    "linear_correct"
                ]
            )
        ]

        corrected = [
            row
            for row in wrong
            if int(
                row[
                    "corrected_vs_linear"
                ]
            )
        ]

        regressed = [
            row
            for row in current
            if int(
                row[
                    "regressed_vs_linear"
                ]
            )
        ]

        output.append({
            "scope":
                scope,

            "margin_bin":
                bin_name,

            "n":
                len(current),

            "linear_wrong":
                len(wrong),

            "corrected":
                len(corrected),

            "wrong_correction_rate":
                (
                    len(corrected)
                    / len(wrong)
                    if wrong
                    else float("nan")
                ),

            "regressed":
                len(regressed),

            "mean_gate":
                (
                    mean([
                        row["gate"]
                        for row
                        in current
                    ])
                    if current
                    else float("nan")
                ),

            "mean_gate_wrong":
                (
                    mean([
                        row["gate"]
                        for row
                        in wrong
                    ])
                    if wrong
                    else float("nan")
                ),

            "mean_gate_corrected":
                (
                    mean([
                        row["gate"]
                        for row
                        in corrected
                    ])
                    if corrected
                    else float("nan")
                ),
        })

    return output


def group_gate_stats(
    rows,
    scope,
):
    definitions = {
        "linear_correct":
            lambda r:
                int(
                    r[
                        "linear_correct"
                    ]
                ) == 1,

        "linear_wrong":
            lambda r:
                int(
                    r[
                        "linear_correct"
                    ]
                ) == 0,

        "corrected":
            lambda r:
                int(
                    r[
                        "corrected_vs_linear"
                    ]
                ) == 1,

        "regressed":
            lambda r:
                int(
                    r[
                        "regressed_vs_linear"
                    ]
                ) == 1,

        "both_correct":
            lambda r:
                (
                    int(
                        r[
                            "linear_correct"
                        ]
                    ) == 1
                    and
                    int(
                        r[
                            "gated_correct"
                        ]
                    ) == 1
                ),

        "both_wrong":
            lambda r:
                (
                    int(
                        r[
                            "linear_correct"
                        ]
                    ) == 0
                    and
                    int(
                        r[
                            "gated_correct"
                        ]
                    ) == 0
                ),
    }

    output = []

    for name, fn in definitions.items():
        current = [
            row
            for row in rows
            if fn(row)
        ]

        if not current:
            continue

        output.append({
            "scope":
                scope,

            "group":
                name,

            "n":
                len(current),

            "gate_mean":
                mean([
                    row["gate"]
                    for row in current
                ]),

            "gate_median":
                median([
                    row["gate"]
                    for row in current
                ]),

            "abs_internal_shift_mean":
                mean([
                    row[
                        "abs_shift_vs_internal_base"
                    ]
                    for row in current
                ]),

            "truth_shift_vs_linear_mean":
                mean([
                    row[
                        "truth_shift_vs_linear"
                    ]
                    for row in current
                ]),

            "truth_direction_rate_vs_linear":
                float(
                    np.mean([
                        float(
                            row[
                                "truth_shift_vs_linear"
                            ]
                        ) > 0
                        for row in current
                    ])
                ),
        })

    return output


def source_split_summary():
    rows = read_csv(
        SOURCE_PATH
    )

    rows = [
        row
        for row in rows
        if row["config"]
        == GATED_CONFIG
    ]

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["split_name"]
        ].append(row)

    output = []

    for split_name, current in sorted(
        grouped.items()
    ):
        output.append({
            "split_name":
                split_name,

            "seed_n":
                len(current),

            "BA":
                mean([
                    float(row["BA"])
                    for row in current
                ]),

            "AUC":
                mean([
                    float(row["AUC"])
                    for row in current
                ]),

            "ACC":
                mean([
                    float(row["ACC"])
                    for row in current
                ]),

            "FP":
                mean([
                    float(row["FP"])
                    for row in current
                ]),

            "FN":
                mean([
                    float(row["FN"])
                    for row in current
                ]),

            "best_epoch_median":
                median([
                    float(
                        row[
                            "best_epoch"
                        ]
                    )
                    for row in current
                ]),
        })

    return output


def main():
    for path in (
        GATED_PATH,
        LINEAR_PATH,
        SOURCE_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gated_rows = read_csv(
        GATED_PATH
    )

    linear_rows = read_csv(
        LINEAR_PATH
    )

    print(
        "======================================"
    )
    print(
        "门控残差机制诊断"
    )
    print(
        "======================================"
    )

    print(
        "门控逐样本行数 =",
        len(gated_rows),
    )

    print(
        "线性逐样本行数 =",
        len(linear_rows),
    )

    gated = aggregate_gated(
        gated_rows
    )

    linear = aggregate_linear(
        linear_rows
    )

    samples = build_samples(
        gated,
        linear,
    )

    print(
        "共同测试APK =",
        len(samples),
    )

    year_summary = []
    margin_rows = []
    gate_rows = []

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        current = [
            row
            for row in samples
            if int(row["year"])
            == year
        ]

        year_summary.append(
            summarize_scope(
                current,
                str(year),
            )
        )

        margin_rows.extend(
            margin_summary(
                current,
                str(year),
            )
        )

        gate_rows.extend(
            group_gate_stats(
                current,
                str(year),
            )
        )

    # 四年总体
    year_summary.append(
        summarize_scope(
            samples,
            "2018-2021",
        )
    )

    margin_rows.extend(
        margin_summary(
            samples,
            "2018-2021",
        )
    )

    gate_rows.extend(
        group_gate_stats(
            samples,
            "2018-2021",
        )
    )

    source_rows = (
        source_split_summary()
    )

    write_csv(
        OUTPUT_DIR
        / "per_sample.csv",
        samples,
    )

    write_csv(
        OUTPUT_DIR
        / "year_summary.csv",
        year_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "margin_summary.csv",
        margin_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "gate_group_summary.csv",
        gate_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "source_split_summary.csv",
        source_rows,
    )

    # ======================================================
    # 打印
    # ======================================================

    print()
    print(
        "======================================"
    )
    print(
        "2017三个源域滚动折"
    )
    print(
        "======================================"
    )

    for row in source_rows:
        print(
            row["split_name"]
        )
        print(
            f" BA={row['BA']:.6f}"
            f" AUC={row['AUC']:.6f}"
            f" ACC={row['ACC']:.6f}"
            f" FP={row['FP']:.2f}"
            f" FN={row['FN']:.2f}"
            f" epoch中位数="
            f"{row['best_epoch_median']:.0f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "相对线性20+150的逐样本变化"
    )
    print(
        "======================================"
    )

    for row in year_summary:
        print()
        print(
            row["scope"]
        )

        print(
            f" n={row['n']}"
            f" 线性错误={row['linear_wrong']}"
            f" 修正={row['corrected']}"
            f" 退化={row['regressed']}"
            f" 净修正={row['net_corrected']:+d}"
        )

        print(
            " 良性修正=",
            row["benign_corrected"],
            "恶意修正=",
            row["malware_corrected"],
            "良性退化=",
            row["benign_regressed"],
            "恶意退化=",
            row["malware_regressed"],
        )

        print(
            " 向真实方向比例=",
            f"{row['truth_direction_rate_vs_linear']:.2%}",
            "平均真实方向变化=",
            f"{row['mean_truth_shift_vs_linear']:+.6f}",
        )

        print(
            " gate："
            f"全部={row['mean_gate_all']:.4f} "
            f"线性错={row['mean_gate_linear_wrong']:.4f} "
            f"线性对={row['mean_gate_linear_correct']:.4f} "
            f"修正={row['mean_gate_corrected']:.4f} "
            f"退化={row['mean_gate_regressed']:.4f}"
        )

        print(
            " 内部基线→最终："
            f"修正={row['internal_base_corrected']} "
            f"退化={row['internal_base_regressed']} "
            f"向真实方向="
            f"{row['truth_direction_rate_internal']:.2%}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "四年总体：线性边界距离分组"
    )
    print(
        "======================================"
    )

    for row in margin_rows:
        if (
            row["scope"]
            != "2018-2021"
        ):
            continue

        rate = (
            "NA"
            if np.isnan(
                row[
                    "wrong_correction_rate"
                ]
            )
            else
            f"{row['wrong_correction_rate']:.2%}"
        )

        print(
            f"{row['margin_bin']:10s} "
            f"N={row['n']:4d} "
            f"原错误={row['linear_wrong']:3d} "
            f"修正={row['corrected']:3d} "
            f"修正率={rate:>7s} "
            f"退化={row['regressed']:3d} "
            f"gate={row['mean_gate']:.4f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2021边界距离分组"
    )
    print(
        "======================================"
    )

    for row in margin_rows:
        if row["scope"] != "2021":
            continue

        rate = (
            "NA"
            if np.isnan(
                row[
                    "wrong_correction_rate"
                ]
            )
            else
            f"{row['wrong_correction_rate']:.2%}"
        )

        print(
            f"{row['margin_bin']:10s} "
            f"N={row['n']:3d} "
            f"原错误={row['linear_wrong']:3d} "
            f"修正={row['corrected']:3d} "
            f"修正率={rate:>7s} "
            f"退化={row['regressed']:3d} "
            f"gate错样本="
            f"{row['mean_gate_wrong']:.4f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "四年总体 gate 分组"
    )
    print(
        "======================================"
    )

    for row in gate_rows:
        if (
            row["scope"]
            != "2018-2021"
        ):
            continue

        print(
            f"{row['group']:16s} "
            f"N={row['n']:4d} "
            f"gate_mean={row['gate_mean']:.4f} "
            f"gate_median={row['gate_median']:.4f} "
            f"|内部修正|={row['abs_internal_shift_mean']:.5f} "
            f"向真实方向="
            f"{row['truth_direction_rate_vs_linear']:.2%}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
