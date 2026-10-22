#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PRED = Path(
    "artifacts/"
    "frozen-anchor-stable70-stable15-2017-2021/"
    "future_predictions.csv"
)

SEM = Path(
    "artifacts/"
    "clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

OUT = Path(
    "artifacts/"
    "fp-fn-multivariate-crossyear-error-risk"
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


def safe_auc(y, score):
    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def fit_predict(
    train_x,
    train_y,
    test_x,
):
    scaler = StandardScaler()

    train_x = scaler.fit_transform(
        train_x
    )

    test_x = scaler.transform(
        test_x
    )

    model = LogisticRegression(
        penalty="l2",
        C=0.10,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=0,
    )

    model.fit(
        train_x,
        train_y,
    )

    return model.predict_proba(
        test_x
    )[:, 1]


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred = pd.read_csv(
        PRED
    )

    sem = pd.read_csv(
        SEM
    )

    pred["sha256"] = (
        pred["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sem["sha256"] = (
        sem["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ========================================================
    # 正式150维 = 155 - 5条自关系
    # ========================================================

    all_rel = [
        c
        for c in sem.columns
        if c.startswith(
            "semrel_support__"
        )
    ]

    rel150 = []

    for name in all_rel:
        parts = name.split(
            "__",
            2,
        )

        if len(parts) != 3:
            continue

        if parts[1] == parts[2]:
            continue

        rel150.append(
            name
        )

    if len(rel150) != 150:
        raise RuntimeError(
            "预期150维，实际="
            + str(len(rel150))
        )

    sem = (
        sem[
            ["sha256"]
            + rel150
        ]
        .drop_duplicates(
            "sha256"
        )
    )

    df = pred.merge(
        sem,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    df[
        rel150
    ] = (
        df[
            rel150
        ]
        .fillna(0.0)
    )

    # ========================================================
    # Stable70基础状态
    # ========================================================

    base_p = df[
        "base_probability"
    ].to_numpy(
        dtype=float
    )

    base_z = logit(
        base_p
    )

    base_pred = (
        base_p >= 0.5
    ).astype(int)

    y = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    uncertainty = (
        4.0
        * base_p
        * (
            1.0
            - base_p
        )
    )

    flippable = (
        uncertainty
        > (
            np.abs(base_z)
            + 1e-6
        )
    )

    df["base_pred"] = (
        base_pred
    )

    df["base_wrong"] = (
        base_pred != y
    ).astype(int)

    df["abs_base_logit"] = (
        np.abs(base_z)
    )

    df["uncertainty"] = (
        uncertainty
    )

    df["flippable"] = (
        flippable.astype(int)
    )

    # ========================================================
    # 三种输入：
    #
    # base2:
    #   只看基础置信距离和不确定度
    #
    # semantic150:
    #   只看150维局部语义
    #
    # semantic150_base2:
    #   两者一起
    # ========================================================

    feature_sets = {
        "base2":
            [
                "abs_base_logit",
                "uncertainty",
            ],

        "semantic150":
            rel150,

        "semantic150_base2":
            rel150
            + [
                "abs_base_logit",
                "uncertainty",
            ],
    }

    rows = []

    years = [
        2018,
        2019,
        2020,
        2021,
    ]

    side_specs = [
        (
            "FP",
            1,
        ),
        (
            "FN",
            0,
        ),
    ]

    # ========================================================
    # 每个seed分别做：
    #
    # 3个年份训练
    # 1个年份测试
    #
    # 避免5个seed重复样本混在同一个模型里。
    # ========================================================

    for seed in sorted(
        df["seed"].unique()
    ):

        ds = df[
            df[
                "seed"
            ] == seed
        ].copy()

        for side, pred_value in side_specs:

            side_df = ds[
                (
                    ds[
                        "flippable"
                    ] == 1
                )
                & (
                    ds[
                        "base_pred"
                    ] == pred_value
                )
            ].copy()

            for test_year in years:

                train = side_df[
                    side_df[
                        "year"
                    ] != test_year
                ]

                test = side_df[
                    side_df[
                        "year"
                    ] == test_year
                ]

                train_y = train[
                    "base_wrong"
                ].to_numpy(
                    dtype=int
                )

                test_y = test[
                    "base_wrong"
                ].to_numpy(
                    dtype=int
                )

                if (
                    len(train) == 0
                    or len(test) == 0
                    or len(
                        np.unique(
                            train_y
                        )
                    ) < 2
                    or len(
                        np.unique(
                            test_y
                        )
                    ) < 2
                ):
                    continue

                for name, features in feature_sets.items():

                    train_x = (
                        train[
                            features
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    test_x = (
                        test[
                            features
                        ]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    score = fit_predict(
                        train_x,
                        train_y,
                        test_x,
                    )

                    rows.append({
                        "side":
                            side,

                        "seed":
                            int(seed),

                        "test_year":
                            int(test_year),

                        "feature_set":
                            name,

                        "train_n":
                            len(train),

                        "train_error_n":
                            int(
                                train_y.sum()
                            ),

                        "test_n":
                            len(test),

                        "test_error_n":
                            int(
                                test_y.sum()
                            ),

                        "error_auc":
                            safe_auc(
                                test_y,
                                score,
                            ),
                    })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUT
        / "per_seed_holdout_year.csv",
        index=False,
    )

    # ========================================================
    # 先平均5个seed
    # ========================================================

    by_year = (
        result
        .groupby(
            [
                "side",
                "feature_set",
                "test_year",
            ],
            as_index=False,
        )
        .agg(
            error_auc=(
                "error_auc",
                "mean",
            ),
            test_n=(
                "test_n",
                "mean",
            ),
            test_error_n=(
                "test_error_n",
                "mean",
            ),
        )
    )

    by_year.to_csv(
        OUT
        / "by_holdout_year.csv",
        index=False,
    )

    summary = (
        by_year
        .groupby(
            [
                "side",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            mean_auc=(
                "error_auc",
                "mean",
            ),
            min_auc=(
                "error_auc",
                "min",
            ),
            max_auc=(
                "error_auc",
                "max",
            ),
        )
        .sort_values(
            [
                "side",
                "mean_auc",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    summary.to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "留一年跨年验证：分侧错误风险预测"
    )
    print(
        "=" * 110
    )

    print(
        by_year.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "四个留出年份汇总"
    )
    print(
        "=" * 110
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
