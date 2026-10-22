#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


THRESHOLD = 0.5
EPS = 1e-7

GATED_PATH = Path(
    "artifacts/gated-residual-neural-2017-2021/"
    "future_predictions.csv"
)

LINEAR_PATH = Path(
    "artifacts/frozen-20-150-2018-2021/"
    "predictions.csv"
)

OUTPUT_DIR = Path(
    "artifacts/gated-residual-mechanism-diagnostic-v2"
)

LINEAR_VARIANT = (
    "compact_plus20_plus150"
)


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


def logit(p):
    p = float(
        np.clip(
            p,
            EPS,
            1.0 - EPS,
        )
    )

    return float(
        np.log(
            p / (1.0 - p)
        )
    )


def correct(p, y):
    return int(
        (p >= THRESHOLD)
        == bool(y)
    )


def truth_direction(
    change,
    label,
):
    if label == 1:
        return change

    return -change


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


def main():
    for path in (
        GATED_PATH,
        LINEAR_PATH,
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

    linear_rows = [
        row
        for row in read_csv(
            LINEAR_PATH
        )
        if row["variant"]
        == LINEAR_VARIANT
    ]

    gated = {}

    for row in gated_rows:
        key = (
            int(row["year"]),
            row["sha256"]
                .strip()
                .casefold(),
            int(row["seed"]),
        )

        gated[key] = row

    linear = {}

    for row in linear_rows:
        key = (
            int(row["year"]),
            row["sha256"]
                .strip()
                .casefold(),
            int(row["seed"]),
        )

        linear[key] = row

    common = sorted(
        set(gated)
        & set(linear)
    )

    print(
        "======================================"
    )
    print(
        "逐种子门控残差机制诊断"
    )
    print(
        "======================================"
    )

    print(
        "门控记录 =",
        len(gated),
    )

    print(
        "线性记录 =",
        len(linear),
    )

    print(
        "匹配记录 =",
        len(common),
    )

    detail_rows = []

    logit_errors = []

    for key in common:
        year, sha, seed = key

        g = gated[key]
        l = linear[key]

        label = int(
            g["label"]
        )

        if label != int(
            l["label"]
        ):
            raise RuntimeError(
                "标签不一致："
                + str(key)
            )

        p_final = float(
            g["probability"]
        )

        p_base = float(
            g["base_probability"]
        )

        p_linear = float(
            l["probability"]
        )

        gate = float(
            g["gate"]
        )

        delta = float(
            g["semantic_delta"]
        )

        # 因为 sigmoid 输出严格大于0，
        # 所以门值为0来自 presence=0。
        presence = int(
            gate > 1e-12
        )

        residual_logit = (
            gate * delta
        )

        observed_logit_change = (
            logit(p_final)
            - logit(p_base)
        )

        logit_errors.append(
            abs(
                residual_logit
                - observed_logit_change
            )
        )

        base_ok = correct(
            p_base,
            label,
        )

        final_ok = correct(
            p_final,
            label,
        )

        linear_ok = correct(
            p_linear,
            label,
        )

        detail_rows.append({
            "year":
                year,

            "sha256":
                sha,

            "seed":
                seed,

            "label":
                label,

            "presence":
                presence,

            "gate":
                gate,

            "delta":
                delta,

            "residual_logit":
                residual_logit,

            "base_probability":
                p_base,

            "final_probability":
                p_final,

            "linear_probability":
                p_linear,

            "base_correct":
                base_ok,

            "final_correct":
                final_ok,

            "linear_correct":
                linear_ok,

            # 真正隔离残差模块：
            "residual_corrected":
                int(
                    (not base_ok)
                    and final_ok
                ),

            "residual_regressed":
                int(
                    base_ok
                    and
                    (not final_ok)
                ),

            # 与外部线性模型整体比较：
            "vs_linear_corrected":
                int(
                    (not linear_ok)
                    and final_ok
                ),

            "vs_linear_regressed":
                int(
                    linear_ok
                    and
                    (not final_ok)
                ),

            "base_margin":
                abs(
                    p_base
                    - THRESHOLD
                ),

            "base_margin_bin":
                margin_bin(
                    abs(
                        p_base
                        - THRESHOLD
                    )
                ),

            "truth_residual_logit":
                truth_direction(
                    residual_logit,
                    label,
                ),
        })

    print(
        "残差公式校验："
    )

    print(
        " logit变化误差中位数 =",
        f"{np.median(logit_errors):.8f}",
    )

    print(
        " logit变化误差P99 =",
        f"{np.percentile(logit_errors, 99):.8f}",
    )

    # ======================================================
    # 每年、每种子先统计
    # ======================================================

    per_seed_rows = []

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        for seed in sorted({
            int(row["seed"])
            for row in detail_rows
            if int(row["year"])
            == year
        }):
            current = [
                row
                for row in detail_rows
                if (
                    int(row["year"])
                    == year
                    and
                    int(row["seed"])
                    == seed
                )
            ]

            active = [
                row
                for row in current
                if int(
                    row["presence"]
                ) == 1
            ]

            inactive = [
                row
                for row in current
                if int(
                    row["presence"]
                ) == 0
            ]

            base_wrong_active = [
                row
                for row in active
                if not int(
                    row[
                        "base_correct"
                    ]
                )
            ]

            base_correct_active = [
                row
                for row in active
                if int(
                    row[
                        "base_correct"
                    ]
                )
            ]

            residual_corrected = [
                row
                for row in current
                if int(
                    row[
                        "residual_corrected"
                    ]
                )
            ]

            residual_regressed = [
                row
                for row in current
                if int(
                    row[
                        "residual_regressed"
                    ]
                )
            ]

            linear_corrected = [
                row
                for row in current
                if int(
                    row[
                        "vs_linear_corrected"
                    ]
                )
            ]

            linear_regressed = [
                row
                for row in current
                if int(
                    row[
                        "vs_linear_regressed"
                    ]
                )
            ]

            benign = [
                row
                for row in current
                if int(row["label"])
                == 0
            ]

            malware = [
                row
                for row in current
                if int(row["label"])
                == 1
            ]

            per_seed_rows.append({
                "year":
                    year,

                "seed":
                    seed,

                "n":
                    len(current),

                "presence_rate":
                    safe_mean([
                        row["presence"]
                        for row in current
                    ]),

                "presence_rate_benign":
                    safe_mean([
                        row["presence"]
                        for row in benign
                    ]),

                "presence_rate_malware":
                    safe_mean([
                        row["presence"]
                        for row in malware
                    ]),

                # 控制 presence 后再比较门值
                "active_gate_base_wrong":
                    safe_mean([
                        row["gate"]
                        for row
                        in base_wrong_active
                    ]),

                "active_gate_base_correct":
                    safe_mean([
                        row["gate"]
                        for row
                        in base_correct_active
                    ]),

                # 真正残差贡献
                "residual_corrected":
                    len(
                        residual_corrected
                    ),

                "residual_regressed":
                    len(
                        residual_regressed
                    ),

                "residual_net":
                    (
                        len(
                            residual_corrected
                        )
                        -
                        len(
                            residual_regressed
                        )
                    ),

                # 总模型相对旧线性
                "linear_corrected":
                    len(
                        linear_corrected
                    ),

                "linear_regressed":
                    len(
                        linear_regressed
                    ),

                "linear_net":
                    (
                        len(
                            linear_corrected
                        )
                        -
                        len(
                            linear_regressed
                        )
                    ),

                "mean_abs_residual_active":
                    safe_mean([
                        abs(
                            row[
                                "residual_logit"
                            ]
                        )
                        for row in active
                    ]),

                "truth_direction_active":
                    safe_mean([
                        float(
                            row[
                                "truth_residual_logit"
                            ]
                        ) > 0
                        for row in active
                    ]),

                "inactive_probability_max_diff":
                    (
                        max([
                            abs(
                                row[
                                    "final_probability"
                                ]
                                -
                                row[
                                    "base_probability"
                                ]
                            )
                            for row in inactive
                        ])
                        if inactive
                        else 0.0
                    ),
            })

    # ======================================================
    # 每年对5个种子求平均
    # ======================================================

    year_rows = []

    fields = (
        "presence_rate",
        "presence_rate_benign",
        "presence_rate_malware",
        "active_gate_base_wrong",
        "active_gate_base_correct",
        "residual_corrected",
        "residual_regressed",
        "residual_net",
        "linear_corrected",
        "linear_regressed",
        "linear_net",
        "mean_abs_residual_active",
        "truth_direction_active",
        "inactive_probability_max_diff",
    )

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        current = [
            row
            for row in per_seed_rows
            if int(row["year"])
            == year
        ]

        out = {
            "year":
                year,
        }

        for field in fields:
            out[field] = (
                safe_mean([
                    row[field]
                    for row in current
                ])
            )

        year_rows.append(
            out
        )

    # ======================================================
    # 内部基线边界距离
    # ======================================================

    margin_rows = []

    bins = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    for bin_name in bins:
        current = [
            row
            for row in detail_rows
            if row[
                "base_margin_bin"
            ] == bin_name
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

        corrected = [
            row
            for row in wrong
            if int(
                row[
                    "residual_corrected"
                ]
            )
        ]

        margin_rows.append({
            "margin_bin":
                bin_name,

            "n":
                len(current),

            "base_wrong":
                len(wrong),

            "corrected":
                len(corrected),

            "correction_rate":
                (
                    len(corrected)
                    / len(wrong)
                    if wrong
                    else float("nan")
                ),

            "mean_gate_active_wrong":
                safe_mean([
                    row["gate"]
                    for row in wrong
                    if int(
                        row["presence"]
                    ) == 1
                ]),
        })

    write_csv(
        OUTPUT_DIR
        / "per_sample_per_seed.csv",
        detail_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_summary.csv",
        per_seed_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_year_summary.csv",
        year_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "base_margin_summary.csv",
        margin_rows,
    )

    # ======================================================
    # 打印
    # ======================================================

    print()
    print(
        "======================================"
    )
    print(
        "逐年：每个种子独立统计后再平均"
    )
    print(
        "======================================"
    )

    for row in year_rows:
        print()
        print(
            row["year"]
        )

        print(
            " 语义存在率："
            f"全部={row['presence_rate']:.2%} "
            f"良性={row['presence_rate_benign']:.2%} "
            f"恶意={row['presence_rate_malware']:.2%}"
        )

        print(
            " 控制语义存在后门值："
            f"内部基线错={row['active_gate_base_wrong']:.4f} "
            f"内部基线对={row['active_gate_base_correct']:.4f}"
        )

        print(
            " 残差相对内部基线："
            f"修正={row['residual_corrected']:.2f} "
            f"退化={row['residual_regressed']:.2f} "
            f"净={row['residual_net']:+.2f}"
        )

        print(
            " 总模型相对线性20+150："
            f"修正={row['linear_corrected']:.2f} "
            f"退化={row['linear_regressed']:.2f} "
            f"净={row['linear_net']:+.2f}"
        )

        print(
            " 活跃样本："
            f"平均|残差logit|="
            f"{row['mean_abs_residual_active']:.4f} "
            f"向真实方向="
            f"{row['truth_direction_active']:.2%}"
        )

        print(
            " 无语义样本最终概率与内部基线最大差异="
            f"{row['inactive_probability_max_diff']:.10f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "按内部Compact基础分支边界距离"
    )
    print(
        "======================================"
    )

    for row in margin_rows:
        rate = (
            "NA"
            if np.isnan(
                row[
                    "correction_rate"
                ]
            )
            else
            f"{row['correction_rate']:.2%}"
        )

        print(
            f"{row['margin_bin']:10s} "
            f"N={row['n']:5d} "
            f"基础错误={row['base_wrong']:4d} "
            f"修正={row['corrected']:4d} "
            f"修正率={rate:>7s} "
            f"活跃错误门值="
            f"{row['mean_gate_active_wrong']:.4f}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
