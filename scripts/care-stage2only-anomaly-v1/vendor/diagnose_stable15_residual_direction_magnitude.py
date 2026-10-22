#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


PRED = Path(
    "artifacts/"
    "frozen-anchor-stable70-stable15-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "stable15-residual-direction-magnitude"
)

EPS = 1e-12


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def sigmoid(z):
    z = np.asarray(
        z,
        dtype=float,
    )

    z = np.clip(
        z,
        -60.0,
        60.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-z)
        )
    )


def pred_from_logit(z):
    return (
        np.asarray(z) >= 0.0
    ).astype(int)


def BA(y, pred):
    return float(
        balanced_accuracy_score(
            np.asarray(y, dtype=int),
            np.asarray(pred, dtype=int),
        )
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        PRED
    )

    y = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    base_logit = logit(
        df[
            "base_probability"
        ].to_numpy(
            dtype=float
        )
    )

    gate = df[
        "gate"
    ].to_numpy(
        dtype=float
    )

    delta = df[
        "semantic_delta"
    ].to_numpy(
        dtype=float
    )

    # 当前真正施加的logit修正。
    shift = (
        gate
        * delta
    )

    actual_logit = (
        base_logit
        + shift
    )

    # ========================================================
    # 先确认我们对公式的理解与保存的probability一致。
    # ========================================================

    reconstructed_prob = sigmoid(
        actual_logit
    )

    stored_prob = df[
        "probability"
    ].to_numpy(
        dtype=float
    )

    max_prob_error = float(
        np.max(
            np.abs(
                reconstructed_prob
                - stored_prob
            )
        )
    )

    print(
        "重建 probability 最大误差 =",
        f"{max_prob_error:.10f}",
    )

    if max_prob_error > 1e-5:
        print(
            "警告：保存概率与 base_logit + gate*delta "
            "存在明显差异，请先检查模型公式。"
        )

    # ========================================================
    # 1. 当前实际残差
    # ========================================================

    z_actual = actual_logit

    # ========================================================
    # 2. 全局翻转当前残差方向
    #
    # 用途：
    # 判断未来年份是不是出现整体方向反转。
    # 只是诊断，不是最终方法。
    # ========================================================

    z_flip = (
        base_logit
        - shift
    )

    # ========================================================
    # 3. 只允许残差“挑战”Stable70
    #
    # base_logit > 0：
    #   只有负残差保留
    #
    # base_logit < 0：
    #   只有正残差保留
    #
    # 不使用标签。
    # ========================================================

    challenge_mask = (
        base_logit
        * shift
        < 0.0
    )

    challenge_shift = np.where(
        challenge_mask,
        shift,
        0.0,
    )

    z_challenge = (
        base_logit
        + challenge_shift
    )

    # ========================================================
    # 4. 方向Oracle，保持“当前实际幅度”完全不变
    #
    # label=1 -> 正方向
    # label=0 -> 负方向
    #
    # 只纠正方向，不增加残差幅度。
    # ========================================================

    desired_sign = np.where(
        y == 1,
        1.0,
        -1.0,
    )

    same_magnitude_shift = (
        desired_sign
        * np.abs(
            shift
        )
    )

    z_oracle_same_magnitude = (
        base_logit
        + same_magnitude_shift
    )

    # ========================================================
    # 5. 当前有界结构的最大Oracle
    #
    # semantic_delta最大绝对值=1，
    # 所以最大shift=gate。
    #
    # 不改变gate公式本身。
    # ========================================================

    max_bound_shift = (
        desired_sign
        * gate
    )

    z_oracle_max_bound = (
        base_logit
        + max_bound_shift
    )

    # ========================================================
    # 保存逐样本信息
    # ========================================================

    df[
        "base_logit"
    ] = base_logit

    df[
        "actual_shift"
    ] = shift

    df[
        "challenge_used"
    ] = challenge_mask.astype(
        int
    )

    df[
        "oracle_same_magnitude_shift"
    ] = same_magnitude_shift

    df[
        "oracle_max_bound_shift"
    ] = max_bound_shift

    df.to_csv(
        OUT
        / "prediction_diagnostic.csv",
        index=False,
    )

    # ========================================================
    # 每个年份×seed正式计算
    # ========================================================

    rows = []

    for (year, seed), idx in df.groupby(
        [
            "year",
            "seed",
        ]
    ).groups.items():

        idx = np.asarray(
            list(idx),
            dtype=int,
        )

        yy = y[
            idx
        ]

        logits = {
            "base":
                base_logit[idx],

            "actual":
                z_actual[idx],

            "flip":
                z_flip[idx],

            "challenge_only":
                z_challenge[idx],

            "oracle_same_magnitude":
                z_oracle_same_magnitude[idx],

            "oracle_max_bound":
                z_oracle_max_bound[idx],
        }

        base_pred = pred_from_logit(
            logits[
                "base"
            ]
        )

        base_ba = BA(
            yy,
            base_pred,
        )

        base_wrong = (
            base_pred
            != yy
        )

        row = {
            "year":
                int(year),

            "seed":
                int(seed),

            "N":
                len(idx),

            "base_wrong":
                int(
                    base_wrong.sum()
                ),
        }

        for name, z in logits.items():
            pred = pred_from_logit(
                z
            )

            value = BA(
                yy,
                pred,
            )

            row[
                name + "_BA"
            ] = value

            row[
                name + "_gain_pp"
            ] = (
                100.0
                * (
                    value
                    - base_ba
                )
            )

            if name != "base":
                corrected = int(
                    (
                        base_wrong
                        & (
                            pred == yy
                        )
                    ).sum()
                )

                regressed = int(
                    (
                        ~base_wrong
                        & (
                            pred != yy
                        )
                    ).sum()
                )

                row[
                    name + "_corrected"
                ] = corrected

                row[
                    name + "_regressed"
                ] = regressed

                row[
                    name + "_net"
                ] = (
                    corrected
                    - regressed
                )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUT
        / "per_seed_year.csv",
        index=False,
    )

    # ========================================================
    # 年份平均
    # ========================================================

    numeric_cols = [
        c
        for c in result.columns
        if c not in {
            "year",
            "seed",
        }
    ]

    by_year = (
        result
        .groupby(
            "year",
            as_index=False,
        )[numeric_cols]
        .mean()
    )

    by_year.to_csv(
        OUT
        / "by_year.csv",
        index=False,
    )

    show = [
        "year",

        "base_BA",

        "actual_BA",
        "actual_gain_pp",

        "flip_BA",
        "flip_gain_pp",

        "challenge_only_BA",
        "challenge_only_gain_pp",

        "oracle_same_magnitude_BA",
        "oracle_same_magnitude_gain_pp",

        "oracle_max_bound_BA",
        "oracle_max_bound_gain_pp",
    ]

    print()
    print(
        "=" * 155
    )
    print(
        "按年份平均"
    )
    print(
        "=" * 155
    )

    print(
        by_year[
            show
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # 四年平均
    # ========================================================

    summary = {}

    for name in (
        "base",
        "actual",
        "flip",
        "challenge_only",
        "oracle_same_magnitude",
        "oracle_max_bound",
    ):
        summary[
            name + "_mean_BA"
        ] = float(
            by_year[
                name + "_BA"
            ].mean()
        )

        summary[
            name + "_gain_pp"
        ] = float(
            by_year[
                name + "_gain_pp"
            ].mean()
        )

    pd.DataFrame([
        summary
    ]).to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "四年平均"
    )
    print(
        "=" * 100
    )

    for name in (
        "base",
        "actual",
        "flip",
        "challenge_only",
        "oracle_same_magnitude",
        "oracle_max_bound",
    ):
        print(
            f"{name:30s}",
            "BA =",
            f"{summary[name + '_mean_BA']:.6f}",
            "gain =",
            f"{summary[name + '_gain_pp']:+.3f} pp",
        )

    same_gain = summary[
        "oracle_same_magnitude_gain_pp"
    ]

    max_gain = summary[
        "oracle_max_bound_gain_pp"
    ]

    flip_gain = summary[
        "flip_gain_pp"
    ]

    challenge_gain = summary[
        "challenge_only_gain_pp"
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "诊断结论"
    )
    print(
        "=" * 100
    )

    if same_gain >= 2.0:
        print(
            "A：保持当前残差幅度，只修正方向，"
            "已经可以达到 +2pp。"
        )

        print(
            "=> 下一步只解决方向学习，"
            "不要扩大残差幅度。"
        )

    elif max_gain >= 2.0:
        print(
            "B：只修正方向但保持当前实际幅度还不足 +2pp；"
            "而当前最大边界允许超过 +2pp。"
        )

        print(
            "=> 下一步需要同时解决："
            "方向学习 + 有效幅度利用，"
            "但不需要放宽现有gate上界。"
        )

    else:
        print(
            "C：当前有界结构不足以达到 +2pp。"
        )

    if flip_gain > summary[
        "actual_gain_pp"
    ]:
        print(
            "补充：整体翻转当前残差方向比原残差更好，"
            "存在系统性方向偏差迹象。"
        )
    else:
        print(
            "补充：简单整体翻转没有解决问题，"
            "方向错误更可能是样本级而不是全局反向。"
        )

    if challenge_gain > summary[
        "actual_gain_pp"
    ]:
        print(
            "补充：只允许残差挑战基础判断能够改善结果。"
        )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
