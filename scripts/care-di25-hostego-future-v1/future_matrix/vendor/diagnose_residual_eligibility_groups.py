#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def balanced_accuracy(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    pos = y == 1
    neg = y == 0

    tpr = (
        (pred[pos] == 1).mean()
        if pos.any()
        else np.nan
    )

    tnr = (
        (pred[neg] == 0).mean()
        if neg.any()
        else np.nan
    )

    return float(
        0.5 * (tpr + tnr)
    )


def summarize_group(g):
    n = len(g)

    base_wrong = int(
        g["base_wrong"].sum()
    )

    changed = int(
        g["changed"].sum()
    )

    corrected = int(
        g["corrected"].sum()
    )

    regressed = int(
        g["regressed"].sum()
    )

    return pd.Series({
        "N":
            n,

        "base_wrong":
            base_wrong,

        "changed":
            changed,

        "corrected":
            corrected,

        "regressed":
            regressed,

        "net":
            corrected - regressed,

        # Compact错误中有多少被修正。
        "correction_recall":
            (
                corrected / base_wrong
                if base_wrong > 0
                else np.nan
            ),

        # 真正发生分类跨界的样本里，
        # 有多少次是改对的。
        "change_precision":
            (
                corrected / changed
                if changed > 0
                else np.nan
            ),
    })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        default=(
            "artifacts/"
            "frozen-anchor-stable70-stable15-2017-2021/"
            "future_predictions.csv"
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.pred
    )

    df = pd.read_csv(
        path
    )

    required = {
        "year",
        "sha256",
        "seed",
        "label",
        "probability",
        "base_probability",
        "gate",
        "semantic_delta",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "缺少列: "
            + str(
                sorted(missing)
            )
        )

    # ========================================================
    # 机制诊断：
    #
    # 同一个APK有5个seed。
    # 这里先对概率/残差量取平均，
    # 得到“一个APK一行”。
    #
    # 注意：
    # 这是机制诊断，不是论文正式性能计算协议。
    # ========================================================

    x = (
        df
        .groupby(
            [
                "year",
                "sha256",
                "label",
            ],
            as_index=False,
        )[
            [
                "probability",
                "base_probability",
                "gate",
                "semantic_delta",
            ]
        ]
        .mean()
    )

    x["label"] = (
        x["label"]
        .astype(int)
    )

    x["base_pred"] = (
        x["base_probability"]
        >= 0.5
    ).astype(int)

    x["final_pred"] = (
        x["probability"]
        >= 0.5
    ).astype(int)

    # Compact离0.5有多远。
    #
    # 越小 = Compact越犹豫。
    x["base_margin"] = np.abs(
        x["base_probability"]
        - 0.5
    )

    # 残差网络本身给出的方向强度。
    x["abs_delta"] = np.abs(
        x["semantic_delta"]
    )

    # gate = presence * uncertainty
    #
    # 所以gate>0基本就表示：
    # 当前APK存在Stable15局部语义。
    x["semantic_present"] = (
        x["gate"]
        > 1e-12
    )

    x["base_wrong"] = (
        x["base_pred"]
        != x["label"]
    )

    x["final_wrong"] = (
        x["final_pred"]
        != x["label"]
    )

    x["changed"] = (
        x["base_pred"]
        != x["final_pred"]
    )

    x["corrected"] = (
        x["base_wrong"]
        &
        (~x["final_wrong"])
    )

    x["regressed"] = (
        (~x["base_wrong"])
        &
        x["final_wrong"]
    )

    # ========================================================
    # 表1：
    # 按“有没有语义 + Compact置信程度”分组。
    # ========================================================

    def margin_group(row):
        if not row[
            "semantic_present"
        ]:
            return "0_no_semantic"

        m = row[
            "base_margin"
        ]

        if m <= 0.05:
            return "1_present_margin_0_005"

        if m <= 0.10:
            return "2_present_margin_005_010"

        if m <= 0.20:
            return "3_present_margin_010_020"

        return "4_present_margin_gt_020"

    x["margin_group"] = (
        x.apply(
            margin_group,
            axis=1,
        )
    )

    margin_summary = (
        x
        .groupby(
            "margin_group",
            sort=True,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )

    print()
    print(
        "=" * 130
    )
    print(
        "表1：什么APK上的残差真正有效？"
    )
    print(
        "按：局部语义存在 + Compact距离0.5的远近"
    )
    print(
        "=" * 130
    )

    print(
        margin_summary.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    # ========================================================
    # 表2：
    # 只看有语义的APK，
    # 再加入残差强度。
    # ========================================================

    semantic = x[
        x[
            "semantic_present"
        ]
    ].copy()

    semantic["margin_bin"] = pd.cut(
        semantic[
            "base_margin"
        ],
        bins=[
            -np.inf,
            0.05,
            0.10,
            0.20,
            np.inf,
        ],
        labels=[
            "<=0.05",
            "0.05-0.10",
            "0.10-0.20",
            ">0.20",
        ],
    )

    semantic["delta_bin"] = pd.cut(
        semantic[
            "abs_delta"
        ],
        bins=[
            -np.inf,
            0.25,
            0.50,
            np.inf,
        ],
        labels=[
            "<0.25",
            "0.25-0.50",
            ">=0.50",
        ],
    )

    cross = (
        semantic
        .groupby(
            [
                "margin_bin",
                "delta_bin",
            ],
            observed=True,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )

    print()
    print(
        "=" * 150
    )
    print(
        "表2：Compact不确定程度 × 残差信号强度"
    )
    print(
        "=" * 150
    )

    print(
        cross.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    # ========================================================
    # 表3：
    # 几个“可部署条件”的诊断。
    #
    # 注意：
    # 这里用到了未来标签评价效果，
    # 只能判断机制是否值得继续，
    # 不能拿这里最好的条件直接当正式模型。
    # ========================================================

    rules = {
        "base_only":
            np.zeros(
                len(x),
                dtype=bool,
            ),

        "actual_residual":
            np.ones(
                len(x),
                dtype=bool,
            ),

        # 有语义 + Compact距离0.5不超过0.20
        "present_margin_le_020":
            (
                x["semantic_present"]
                &
                (
                    x[
                        "base_margin"
                    ]
                    <= 0.20
                )
            ),

        # 更严格的不确定区
        "present_margin_le_010":
            (
                x["semantic_present"]
                &
                (
                    x[
                        "base_margin"
                    ]
                    <= 0.10
                )
            ),

        # 有语义 + 不确定 + 残差信号不能太弱
        "present_m020_delta025":
            (
                x["semantic_present"]
                &
                (
                    x[
                        "base_margin"
                    ]
                    <= 0.20
                )
                &
                (
                    x[
                        "abs_delta"
                    ]
                    >= 0.25
                )
            ),

        "present_m010_delta025":
            (
                x["semantic_present"]
                &
                (
                    x[
                        "base_margin"
                    ]
                    <= 0.10
                )
                &
                (
                    x[
                        "abs_delta"
                    ]
                    >= 0.25
                )
            ),
    }

    policy_rows = []

    for rule_name, eligible in rules.items():

        if rule_name == "base_only":
            pred = (
                x[
                    "base_pred"
                ]
                .to_numpy()
            )

        elif rule_name == "actual_residual":
            pred = (
                x[
                    "final_pred"
                ]
                .to_numpy()
            )

        else:
            pred = (
                x[
                    "base_pred"
                ]
                .to_numpy()
                .copy()
            )

            final = (
                x[
                    "final_pred"
                ]
                .to_numpy()
            )

            mask = np.asarray(
                eligible,
                dtype=bool,
            )

            pred[
                mask
            ] = final[
                mask
            ]

        year_bas = []

        for year in sorted(
            x[
                "year"
            ].unique()
        ):
            mask_year = (
                x[
                    "year"
                ].to_numpy()
                == year
            )

            ba = balanced_accuracy(
                x.loc[
                    mask_year,
                    "label",
                ],
                pred[
                    mask_year
                ],
            )

            year_bas.append(
                ba
            )

            policy_rows.append({
                "rule":
                    rule_name,

                "year":
                    int(
                        year
                    ),

                "BA":
                    ba,

                "eligible_N":
                    (
                        int(
                            np.asarray(
                                eligible,
                                dtype=bool,
                            )[
                                mask_year
                            ].sum()
                        )
                        if rule_name
                        not in {
                            "base_only",
                            "actual_residual",
                        }
                        else np.nan
                    ),
            })

    policy_df = pd.DataFrame(
        policy_rows
    )

    print()
    print(
        "=" * 120
    )
    print(
        "表3：限制“哪些APK允许残差改变分类”的诊断"
    )
    print(
        "注意：这里只用于看机制，不用于正式挑未来最优阈值"
    )
    print(
        "=" * 120
    )

    pivot = (
        policy_df
        .pivot(
            index="rule",
            columns="year",
            values="BA",
        )
    )

    pivot[
        "mean_BA"
    ] = pivot.mean(
        axis=1
    )

    pivot[
        "min_BA"
    ] = pivot[
        [
            c
            for c in pivot.columns
            if isinstance(
                c,
                (int, np.integer)
            )
        ]
    ].min(
        axis=1
    )

    print(
        pivot.sort_values(
            "mean_BA",
            ascending=False,
        ).to_string(
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "逐年允许残差参与的APK数量"
    )
    print(
        "=" * 120
    )

    print(
        policy_df[
            policy_df[
                "eligible_N"
            ].notna()
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    out = (
        path.parent
        / "residual_eligibility_diagnostic.csv"
    )

    x.to_csv(
        out,
        index=False,
    )

    print()
    print(
        "sample diagnostic saved:",
        out
    )


if __name__ == "__main__":
    main()
