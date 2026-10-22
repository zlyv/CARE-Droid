#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


INPUT = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "frozen-count10-150-directional-gamma-sweep"
)

THRESHOLD = 0.5
TAU = 0.20
EPS = 1e-7

YEARS = (2018, 2019, 2020, 2021)

# Δ<0：往良性方向修
NEG_GAMMAS = np.round(
    np.arange(1.00, 4.01, 0.25),
    2,
)

# Δ>0：往恶意方向修
# 允许小于1，因为这一方向也可能制造新的FP。
POS_GAMMAS = np.round(
    np.arange(0.50, 3.01, 0.25),
    2,
)


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
    z = np.clip(
        np.asarray(z, dtype=float),
        -60.0,
        60.0,
    )
    return 1.0 / (
        1.0 + np.exp(-z)
    )


def probability_for(
    df,
    gamma_neg,
    gamma_pos,
):
    base_p = df[
        "base_probability"
    ].to_numpy(dtype=float)

    base_z = logit(
        base_p
    )

    residual = (
        df["gate"].to_numpy(dtype=float)
        *
        df["semantic_delta"].to_numpy(dtype=float)
    )

    margin = np.abs(
        base_p - 0.5
    )

    # 边界外严格保持当前模型。
    multiplier = np.ones(
        len(df),
        dtype=float,
    )

    active = (
        margin <= TAU
    )

    multiplier[
        active & (residual < 0)
    ] = float(
        gamma_neg
    )

    multiplier[
        active & (residual >= 0)
    ] = float(
        gamma_pos
    )

    final_z = (
        base_z
        + multiplier
        * residual
    )

    return sigmoid(
        final_z
    )


def metrics(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    pred = (
        np.asarray(p)
        >= THRESHOLD
    ).astype(int)

    return {
        "BA":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),
        "AUC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),
        "ACC":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),
        "FP":
            int(
                (
                    (y == 0)
                    &
                    (pred == 1)
                ).sum()
            ),
        "FN":
            int(
                (
                    (y == 1)
                    &
                    (pred == 0)
                ).sum()
            ),
    }


def evaluate(
    df,
    gamma_neg,
    gamma_pos,
):
    work = df.copy()

    work["p"] = probability_for(
        work,
        gamma_neg,
        gamma_pos,
    )

    seed_rows = []

    for (
        year,
        seed,
    ), g in work.groupby(
        ["year", "seed"],
        sort=True,
    ):
        m = metrics(
            g["label"],
            g["p"],
        )

        seed_rows.append({
            "year": int(year),
            "seed": int(seed),
            **m,
        })

    seed_df = pd.DataFrame(
        seed_rows
    )

    year_rows = []

    for year, g in seed_df.groupby(
        "year",
        sort=True,
    ):
        year_rows.append({
            "year":
                int(year),

            "BA":
                float(
                    g["BA"].mean()
                ),

            "AUC":
                float(
                    g["AUC"].mean()
                ),

            "ACC":
                float(
                    g["ACC"].mean()
                ),

            "FP":
                float(
                    g["FP"].mean()
                ),

            "FN":
                float(
                    g["FN"].mean()
                ),
        })

    year_df = pd.DataFrame(
        year_rows
    )

    result = {
        "gamma_neg":
            float(gamma_neg),

        "gamma_pos":
            float(gamma_pos),

        "mean_BA":
            float(
                year_df["BA"].mean()
            ),

        "min_BA":
            float(
                year_df["BA"].min()
            ),

        "mean_AUC":
            float(
                year_df["AUC"].mean()
            ),

        "mean_ACC":
            float(
                year_df["ACC"].mean()
            ),
    }

    for year in YEARS:
        r = year_df[
            year_df["year"] == year
        ].iloc[0]

        for name in (
            "BA",
            "AUC",
            "FP",
            "FN",
        ):
            result[
                f"{name}_{year}"
            ] = float(
                r[name]
            )

    return result


def sample_diagnostic(
    df,
    gamma_neg,
    gamma_pos,
):
    work = df.copy()

    work["candidate"] = (
        probability_for(
            work,
            gamma_neg,
            gamma_pos,
        )
    )

    work["baseline"] = (
        probability_for(
            work,
            1.0,
            1.0,
        )
    )

    avg = (
        work.groupby(
            [
                "year",
                "sha256",
                "label",
            ],
            as_index=False,
        )
        .agg(
            candidate=(
                "candidate",
                "mean",
            ),
            baseline=(
                "baseline",
                "mean",
            ),
        )
    )

    cp = (
        avg["candidate"]
        >= 0.5
    ).astype(int)

    bp = (
        avg["baseline"]
        >= 0.5
    ).astype(int)

    y = avg[
        "label"
    ].astype(int)

    avg["candidate_correct"] = (
        cp == y
    )

    avg["baseline_correct"] = (
        bp == y
    )

    avg["corrected"] = (
        (~avg["baseline_correct"])
        &
        avg["candidate_correct"]
    )

    avg["regressed"] = (
        avg["baseline_correct"]
        &
        (~avg["candidate_correct"])
    )

    rows = []

    for year in (
        2018,
        2019,
        2020,
        2021,
        "ALL",
    ):
        if year == "ALL":
            g = avg
        else:
            g = avg[
                avg["year"] == year
            ]

        corrected = int(
            g["corrected"].sum()
        )

        regressed = int(
            g["regressed"].sum()
        )

        rows.append({
            "year":
                year,

            "corrected":
                corrected,

            "regressed":
                regressed,

            "net":
                corrected - regressed,

            "corr_benign":
                int(
                    (
                        g["corrected"]
                        &
                        (g["label"] == 0)
                    ).sum()
                ),

            "corr_malware":
                int(
                    (
                        g["corrected"]
                        &
                        (g["label"] == 1)
                    ).sum()
                ),

            "reg_benign":
                int(
                    (
                        g["regressed"]
                        &
                        (g["label"] == 0)
                    ).sum()
                ),

            "reg_malware":
                int(
                    (
                        g["regressed"]
                        &
                        (g["label"] == 1)
                    ).sum()
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    print(
        "rows =",
        len(df),
        "APK =",
        df["sha256"].nunique(),
    )

    baseline = evaluate(
        df,
        1.0,
        1.0,
    )

    print()
    print(
        "基线:"
    )
    print(
        f"meanBA={baseline['mean_BA']:.6f} "
        f"minBA={baseline['min_BA']:.6f} "
        f"AUC={baseline['mean_AUC']:.6f}"
    )

    rows = []

    for gn in NEG_GAMMAS:
        for gp in POS_GAMMAS:
            rows.append(
                evaluate(
                    df,
                    gn,
                    gp,
                )
            )

    result = pd.DataFrame(
        rows
    )

    ranked = (
        result.sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # AUC最多下降0.005，
    # minBA至少不能低于当前基线。
    conservative = (
        result[
            (
                result["mean_AUC"]
                >= baseline["mean_AUC"] - 0.005
            )
            &
            (
                result["min_BA"]
                >= baseline["min_BA"]
            )
        ]
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # 单独看只放大负方向，
    # 即 gamma_pos 固定为1。
    negative_only = (
        result[
            np.isclose(
                result["gamma_pos"],
                1.0,
            )
        ]
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best = ranked.iloc[0]

    result.to_csv(
        OUT / "all_results.csv",
        index=False,
    )

    ranked.to_csv(
        OUT / "ranked.csv",
        index=False,
    )

    conservative.to_csv(
        OUT / "conservative.csv",
        index=False,
    )

    negative_only.to_csv(
        OUT / "negative_only.csv",
        index=False,
    )

    diag = sample_diagnostic(
        df,
        float(
            best["gamma_neg"]
        ),
        float(
            best["gamma_pos"]
        ),
    )

    diag.to_csv(
        OUT / "best_sample_diagnostic.csv",
        index=False,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "平均BA前15"
    )
    print(
        "=" * 70
    )

    for _, r in (
        ranked.head(15).iterrows()
    ):
        print(
            f"g-={r['gamma_neg']:.2f} "
            f"g+={r['gamma_pos']:.2f} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f} "
            f"BA18={r['BA_2018']:.6f} "
            f"BA19={r['BA_2019']:.6f} "
            f"BA20={r['BA_2020']:.6f} "
            f"BA21={r['BA_2021']:.6f}"
        )

    print()
    print(
        "=" * 70
    )
    print(
        "绝对最佳"
    )
    print(
        "=" * 70
    )

    print(
        f"g-={best['gamma_neg']:.2f}"
    )
    print(
        f"g+={best['gamma_pos']:.2f}"
    )
    print(
        f"meanBA "
        f"{baseline['mean_BA']:.6f}"
        f" -> "
        f"{best['mean_BA']:.6f} "
        f"Δ="
        f"{best['mean_BA']-baseline['mean_BA']:+.6f}"
    )
    print(
        f"minBA "
        f"{baseline['min_BA']:.6f}"
        f" -> "
        f"{best['min_BA']:.6f}"
    )
    print(
        f"AUC "
        f"{baseline['mean_AUC']:.6f}"
        f" -> "
        f"{best['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "只调整负方向时最佳"
    )
    print(
        "=" * 70
    )

    nb = negative_only.iloc[0]

    print(
        f"g-={nb['gamma_neg']:.2f}, "
        f"g+=1.00"
    )
    print(
        f"meanBA={nb['mean_BA']:.6f} "
        f"minBA={nb['min_BA']:.6f} "
        f"AUC={nb['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "保守约束下最佳"
    )
    print(
        "=" * 70
    )

    if len(conservative) == 0:
        print(
            "没有候选满足约束"
        )
    else:
        c = conservative.iloc[0]

        print(
            f"g-={c['gamma_neg']:.2f} "
            f"g+={c['gamma_pos']:.2f}"
        )
        print(
            f"meanBA={c['mean_BA']:.6f} "
            f"minBA={c['min_BA']:.6f} "
            f"AUC={c['mean_AUC']:.6f}"
        )

    print()
    print(
        "=" * 70
    )
    print(
        "绝对最佳样本变化"
    )
    print(
        "=" * 70
    )

    for _, r in diag.iterrows():
        print(
            f"{str(r['year']):>4s} "
            f"corrected={int(r['corrected']):3d} "
            f"regressed={int(r['regressed']):3d} "
            f"net={int(r['net']):+3d} "
            f"corrB={int(r['corr_benign']):3d} "
            f"corrM={int(r['corr_malware']):3d} "
            f"regB={int(r['reg_benign']):3d} "
            f"regM={int(r['reg_malware']):3d}"
        )


if __name__ == "__main__":
    main()
