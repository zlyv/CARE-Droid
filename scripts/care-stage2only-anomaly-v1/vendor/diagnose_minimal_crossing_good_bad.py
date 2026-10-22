#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


INPUT = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "minimal-crossing-good-bad-diagnostic"
)

G_NEG = 16.25
EPS = 1e-12


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-7,
        1.0 - 1e-7,
    )
    return np.log(
        p / (1.0 - p)
    )


def describe(group, name):
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print("N =", len(group))

    if len(group) == 0:
        return

    for col in (
        "base_probability",
        "gate",
        "semantic_delta",
        "effective_residual",
        "required_gain",
    ):
        x = (
            group[col]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .to_numpy(dtype=float)
        )

        print(
            f"{col:20s} "
            f"mean={np.mean(x): .6f} "
            f"median={np.median(x): .6f} "
            f"p25={np.quantile(x,0.25): .6f} "
            f"p75={np.quantile(x,0.75): .6f}"
        )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    df["base_logit"] = logit(
        df["base_probability"]
    )

    df["effective_residual"] = (
        df["gate"]
        *
        df["semantic_delta"]
    )

    df["current_logit"] = (
        df["base_logit"]
        +
        df["effective_residual"]
    )

    df["required_gain"] = np.inf

    valid = (
        (df["base_logit"] > 0)
        &
        (df["effective_residual"] < -EPS)
    )

    df.loc[
        valid,
        "required_gain"
    ] = (
        df.loc[
            valid,
            "base_logit"
        ]
        /
        (
            -df.loc[
                valid,
                "effective_residual"
            ]
        )
    )

    # 这些就是相对当前模型会被“额外翻转”的逐seed记录。
    eligible = (
        valid
        &
        (df["current_logit"] >= 0)
        &
        (df["required_gain"] <= G_NEG)
    )

    e = df[
        eligible
    ].copy()

    e["good_flip"] = (
        e["label"] == 0
    )

    e["bad_flip"] = (
        e["label"] == 1
    )

    print(
        "eligible records =",
        len(e),
    )

    print(
        "good benign records =",
        int(e["good_flip"].sum()),
    )

    print(
        "bad malware records =",
        int(e["bad_flip"].sum()),
    )

    describe(
        e[e["good_flip"]],
        "正确负方向纠错：真实良性",
    )

    describe(
        e[e["bad_flip"]],
        "错误负方向纠错：真实恶意",
    )

    print()
    print("=" * 78)
    print("逐年")
    print("=" * 78)

    by_year = (
        e.groupby(
            ["year", "label"],
            as_index=False,
        )
        .agg(
            N=("sha256", "size"),
            mean_delta=("semantic_delta", "mean"),
            median_delta=("semantic_delta", "median"),
            mean_gate=("gate", "mean"),
            median_gate=("gate", "median"),
            mean_required=("required_gain", "mean"),
            median_required=("required_gain", "median"),
            mean_base_probability=("base_probability", "mean"),
            median_base_probability=("base_probability", "median"),
        )
    )

    print(
        by_year.to_string(index=False)
    )

    by_year.to_csv(
        OUT / "by_year_label.csv",
        index=False,
    )

    e.to_csv(
        OUT / "eligible_per_seed.csv",
        index=False,
    )

    # ========================================================
    # 单变量“纯度”扫描。
    #
    # 这里只做诊断：
    # 看更强的负 semantic_delta 是否能提高良性比例。
    # ========================================================

    rows = []

    for cut in np.arange(
        -0.95,
        -0.049,
        0.05,
    ):
        g = e[
            e["semantic_delta"] <= cut
        ]

        if len(g) == 0:
            continue

        benign = int(
            (g["label"] == 0).sum()
        )

        malware = int(
            (g["label"] == 1).sum()
        )

        rows.append({
            "delta_cut":
                float(cut),

            "N":
                len(g),

            "benign":
                benign,

            "malware":
                malware,

            "benign_rate":
                benign / len(g),
        })

    sweep = pd.DataFrame(rows)

    print()
    print("=" * 78)
    print("semantic_delta强度筛选")
    print("=" * 78)

    print(
        sweep.to_string(
            index=False
        )
    )

    sweep.to_csv(
        OUT / "semantic_delta_purity_sweep.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
