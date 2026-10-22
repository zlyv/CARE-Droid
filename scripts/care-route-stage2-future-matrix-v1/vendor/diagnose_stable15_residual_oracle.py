#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


PRED_FILE = Path(
    "artifacts/"
    "frozen-anchor-stable70-stable15-2017-2021/"
    "future_predictions.csv"
)

OUT_DIR = Path(
    "artifacts/"
    "stable15-residual-oracle"
)

EPS = 1e-12


def safe_logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def hard_from_prob(p):
    return (
        np.asarray(p, dtype=float) >= 0.5
    ).astype(int)


def ba(y, pred):
    return float(
        balanced_accuracy_score(
            np.asarray(y, dtype=int),
            np.asarray(pred, dtype=int),
        )
    )


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        PRED_FILE
    )

    required = {
        "year",
        "seed",
        "sha256",
        "label",
        "probability",
        "base_probability",
        "gate",
        "semantic_delta",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "future_predictions.csv 缺列: "
            + str(sorted(missing))
        )

    # ========================================================
    # 基础量
    # ========================================================

    df["label"] = (
        df["label"]
        .astype(int)
    )

    df["base_pred"] = hard_from_prob(
        df["base_probability"]
    )

    df["residual_pred"] = hard_from_prob(
        df["probability"]
    )

    df["base_wrong"] = (
        df["base_pred"]
        != df["label"]
    )

    df["residual_wrong"] = (
        df["residual_pred"]
        != df["label"]
    )

    # gate > 0 表示 Stable15 局部关系至少有一个存在。
    df["active"] = (
        df["gate"] > EPS
    )

    df["base_logit"] = safe_logit(
        df["base_probability"]
    )

    df["margin_to_boundary"] = np.abs(
        df["base_logit"]
    )

    # ========================================================
    # 当前 bounded residual 的理论翻转条件
    #
    # z_final = z_base + gate * semantic_delta
    # semantic_delta ∈ [-1, +1]
    #
    # 所以最大可移动距离 = gate。
    #
    # 一个错误基础预测理论可翻转，当且仅当：
    #
    #     gate > |base_logit|
    #
    # 边界恰好相等不算，因为还需要真正跨过0。
    # ========================================================

    df["bound_flippable"] = (
        df["base_wrong"]
        & df["active"]
        & (
            df["gate"]
            > df["margin_to_boundary"] + EPS
        )
    )

    # ========================================================
    # 实际残差效果
    # ========================================================

    df["actually_corrected"] = (
        df["base_wrong"]
        & ~df["residual_wrong"]
    )

    df["actually_regressed"] = (
        ~df["base_wrong"]
        & df["residual_wrong"]
    )

    # 对基础错误样本：
    # 正确残差方向应该把恶意样本往正方向推，
    # 良性样本往负方向推。
    desired_sign = np.where(
        df["label"].to_numpy() == 1,
        1.0,
        -1.0,
    )

    actual_shift = (
        df["gate"].to_numpy(dtype=float)
        * df["semantic_delta"].to_numpy(dtype=float)
    )

    df["actual_logit_shift"] = (
        actual_shift
    )

    df["correct_direction"] = (
        actual_shift * desired_sign > 0
    )

    df["wrong_direction"] = (
        actual_shift * desired_sign < 0
    )

    # ========================================================
    # 两种 Oracle
    #
    # coverage oracle:
    #   只要 base 错且 Stable15 active，就假设全部纠正。
    #
    # bounded oracle:
    #   只有当前 gate 理论上能跨界的 base 错误才纠正。
    #
    # 所有 base 正确样本都保持正确，不允许 Oracle 改错。
    # ========================================================

    per_seed_year = []

    for (year, seed), g in df.groupby(
        ["year", "seed"],
        sort=True,
    ):
        g = g.copy()

        y = g["label"].to_numpy(
            dtype=int
        )

        base_pred = g[
            "base_pred"
        ].to_numpy(
            dtype=int
        )

        residual_pred = g[
            "residual_pred"
        ].to_numpy(
            dtype=int
        )

        coverage_pred = (
            base_pred.copy()
        )

        coverage_mask = (
            g["base_wrong"]
            & g["active"]
        ).to_numpy(
            dtype=bool
        )

        coverage_pred[
            coverage_mask
        ] = y[
            coverage_mask
        ]

        bounded_pred = (
            base_pred.copy()
        )

        bounded_mask = g[
            "bound_flippable"
        ].to_numpy(
            dtype=bool
        )

        bounded_pred[
            bounded_mask
        ] = y[
            bounded_mask
        ]

        base_wrong = int(
            g["base_wrong"].sum()
        )

        active_wrong = int(
            (
                g["base_wrong"]
                & g["active"]
            ).sum()
        )

        flippable_wrong = int(
            g["bound_flippable"].sum()
        )

        corrected = int(
            g["actually_corrected"].sum()
        )

        regressed = int(
            g["actually_regressed"].sum()
        )

        wrong_direction = int(
            (
                g["base_wrong"]
                & g["active"]
                & g["wrong_direction"]
            ).sum()
        )

        correct_direction = int(
            (
                g["base_wrong"]
                & g["active"]
                & g["correct_direction"]
            ).sum()
        )

        row = {
            "year":
                int(year),

            "seed":
                int(seed),

            "N":
                len(g),

            "base_BA":
                ba(y, base_pred),

            "actual_residual_BA":
                ba(y, residual_pred),

            "coverage_oracle_BA":
                ba(y, coverage_pred),

            "bounded_oracle_BA":
                ba(y, bounded_pred),

            "actual_gain_pp":
                100.0 * (
                    ba(y, residual_pred)
                    - ba(y, base_pred)
                ),

            "coverage_oracle_gain_pp":
                100.0 * (
                    ba(y, coverage_pred)
                    - ba(y, base_pred)
                ),

            "bounded_oracle_gain_pp":
                100.0 * (
                    ba(y, bounded_pred)
                    - ba(y, base_pred)
                ),

            "base_wrong":
                base_wrong,

            "active_wrong":
                active_wrong,

            "active_wrong_rate":
                (
                    active_wrong / base_wrong
                    if base_wrong
                    else np.nan
                ),

            "bound_flippable_wrong":
                flippable_wrong,

            "bound_flippable_rate_of_wrong":
                (
                    flippable_wrong / base_wrong
                    if base_wrong
                    else np.nan
                ),

            "bound_flippable_rate_of_active_wrong":
                (
                    flippable_wrong / active_wrong
                    if active_wrong
                    else np.nan
                ),

            "actually_corrected":
                corrected,

            "actually_regressed":
                regressed,

            "net_actual_corrections":
                corrected - regressed,

            "correct_direction_active_wrong":
                correct_direction,

            "wrong_direction_active_wrong":
                wrong_direction,

            "correct_direction_rate_active_wrong":
                (
                    correct_direction / active_wrong
                    if active_wrong
                    else np.nan
                ),
        }

        per_seed_year.append(
            row
        )

    ps = pd.DataFrame(
        per_seed_year
    )

    ps.to_csv(
        OUT_DIR
        / "oracle_per_seed_year.csv",
        index=False,
    )

    # ========================================================
    # 按年份平均5个seed
    # ========================================================

    numeric_cols = [
        c
        for c in ps.columns
        if c not in {
            "year",
            "seed",
        }
    ]

    by_year = (
        ps.groupby(
            "year",
            as_index=False,
        )[numeric_cols]
        .mean()
    )

    by_year.to_csv(
        OUT_DIR
        / "oracle_by_year.csv",
        index=False,
    )

    # ========================================================
    # 四年正式平均
    # ========================================================

    summary = {
        "base_mean_BA":
            by_year[
                "base_BA"
            ].mean(),

        "actual_residual_mean_BA":
            by_year[
                "actual_residual_BA"
            ].mean(),

        "coverage_oracle_mean_BA":
            by_year[
                "coverage_oracle_BA"
            ].mean(),

        "bounded_oracle_mean_BA":
            by_year[
                "bounded_oracle_BA"
            ].mean(),

        "actual_gain_pp":
            by_year[
                "actual_gain_pp"
            ].mean(),

        "coverage_oracle_gain_pp":
            by_year[
                "coverage_oracle_gain_pp"
            ].mean(),

        "bounded_oracle_gain_pp":
            by_year[
                "bounded_oracle_gain_pp"
            ].mean(),

        "base_wrong_mean":
            by_year[
                "base_wrong"
            ].mean(),

        "active_wrong_mean":
            by_year[
                "active_wrong"
            ].mean(),

        "active_wrong_rate_mean":
            by_year[
                "active_wrong_rate"
            ].mean(),

        "bound_flippable_wrong_mean":
            by_year[
                "bound_flippable_wrong"
            ].mean(),

        "bound_flippable_rate_of_wrong_mean":
            by_year[
                "bound_flippable_rate_of_wrong"
            ].mean(),

        "bound_flippable_rate_of_active_wrong_mean":
            by_year[
                "bound_flippable_rate_of_active_wrong"
            ].mean(),

        "actually_corrected_mean":
            by_year[
                "actually_corrected"
            ].mean(),

        "actually_regressed_mean":
            by_year[
                "actually_regressed"
            ].mean(),

        "correct_direction_rate_active_wrong_mean":
            by_year[
                "correct_direction_rate_active_wrong"
            ].mean(),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        OUT_DIR
        / "oracle_summary.csv",
        index=False,
    )

    # ========================================================
    # 错误样本按基础置信距离分桶
    # ========================================================

    wrong = df[
        df["base_wrong"]
    ].copy()

    bins = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.40,
        0.80,
        np.inf,
    ]

    labels = [
        "[0,.05)",
        "[.05,.10)",
        "[.10,.20)",
        "[.20,.40)",
        "[.40,.80)",
        "[.80,+inf)",
    ]

    # 用概率距离更直观。
    wrong[
        "prob_margin"
    ] = np.abs(
        wrong[
            "base_probability"
        ] - 0.5
    )

    wrong[
        "margin_bin"
    ] = pd.cut(
        wrong[
            "prob_margin"
        ],
        bins=bins,
        labels=labels,
        right=False,
    )

    margin_rows = []

    for (
        year,
        margin_bin
    ), g in wrong.groupby(
        [
            "year",
            "margin_bin",
        ],
        observed=True,
        sort=True,
    ):
        margin_rows.append({
            "year":
                int(year),

            "margin_bin":
                str(margin_bin),

            "N":
                len(g),

            "active_rate":
                float(
                    g["active"].mean()
                ),

            "bound_flippable_rate":
                float(
                    g[
                        "bound_flippable"
                    ].mean()
                ),

            "correct_direction_rate":
                float(
                    g[
                        "correct_direction"
                    ].mean()
                ),

            "actual_correct_rate":
                float(
                    g[
                        "actually_corrected"
                    ].mean()
                ),
        })

    pd.DataFrame(
        margin_rows
    ).to_csv(
        OUT_DIR
        / "wrong_margin_diagnostic.csv",
        index=False,
    )

    # ========================================================
    # 打印核心结果
    # ========================================================

    cols = [
        "year",
        "base_BA",
        "actual_residual_BA",
        "actual_gain_pp",
        "coverage_oracle_BA",
        "coverage_oracle_gain_pp",
        "bounded_oracle_BA",
        "bounded_oracle_gain_pp",
        "base_wrong",
        "active_wrong",
        "bound_flippable_wrong",
        "actually_corrected",
        "actually_regressed",
        "correct_direction_rate_active_wrong",
    ]

    print()
    print("=" * 150)
    print("按年份平均（5 seeds）")
    print("=" * 150)

    print(
        by_year[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 150)
    print("四年平均")
    print("=" * 150)

    for key, value in summary.items():
        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"{key:45s}",
                f"{value:.6f}",
            )
        else:
            print(
                f"{key:45s}",
                value,
            )

    print()
    print("=" * 150)
    print("关键判断")
    print("=" * 150)

    cov_gain = summary[
        "coverage_oracle_gain_pp"
    ]

    bound_gain = summary[
        "bounded_oracle_gain_pp"
    ]

    actual_gain = summary[
        "actual_gain_pp"
    ]

    print(
        "实际残差增益        = "
        f"{actual_gain:+.3f} pp"
    )

    print(
        "Stable15覆盖理论上限 = "
        f"{cov_gain:+.3f} pp"
    )

    print(
        "当前有界公式理论上限 = "
        f"{bound_gain:+.3f} pp"
    )

    print()

    if cov_gain < 2.0:
        print(
            "结论：Stable15覆盖上限本身不到 +2pp。"
        )

        print(
            "下一步优先扩大/重构局部语义表示，"
            "继续调当前残差网络没有机会达到目标。"
        )

    elif bound_gain < 2.0:
        print(
            "结论：Stable15覆盖允许 +2pp，"
            "但当前有界残差公式限制过强。"
        )

        print(
            "下一步优先修改残差幅度控制机制。"
        )

    else:
        print(
            "结论：当前Stable15 + 当前有界公式理论上已经允许 +2pp。"
        )

        print(
            "下一步重点不是扩大特征或放松边界，"
            "而是解决残差方向学习。"
        )

    print()
    print(
        "结果目录:",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
