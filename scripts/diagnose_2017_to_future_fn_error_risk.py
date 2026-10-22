#!/usr/bin/env python3

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_frozen_anchor_stable70_stable15_2017_2021 as stable15


base = stable15.base


COMPACT = Path(
    "artifacts/"
    "p1-features/"
    "features.csv"
)

COMPACT_SCHEMA = Path(
    "artifacts/"
    "p1-features/"
    "feature_schema.json"
)

SEM2017 = Path(
    "artifacts/"
    "clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SEMFUTURE = Path(
    "artifacts/"
    "clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

SPLITS = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUT = Path(
    "artifacts/"
    "2017-to-future-fn-error-risk"
)

EPS = 1e-12


def norm_sha(x):
    return (
        str(x)
        .strip()
        .casefold()
    )


def logit(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
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


def load_semantic():
    s17 = pd.read_csv(
        SEM2017
    )

    sf = pd.read_csv(
        SEMFUTURE
    )

    sem = pd.concat(
        [
            s17,
            sf,
        ],
        axis=0,
        ignore_index=True,
    )

    sem["sha256"] = (
        sem["sha256"]
        .astype(str)
        .map(norm_sha)
    )

    sem = sem.drop_duplicates(
        "sha256"
    )

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
            + str(
                len(rel150)
            )
        )

    return (
        sem,
        rel150,
    )


def get_base_probabilities(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    seed,
):
    # Stable70构造已经通过stable15模块挂载。
    (
        x_train,
        x_test,
        _,
    ) = base.compact_matrix_named(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    y_train = base.labels_for(
        train_rows,
        compact_by_sha,
    ).astype(int)

    from run_pilot_compact_eval import (
        fit_logistic,
    )

    w, b = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=6000,
        learning_rate=0.05,
        l2=0.03,
    )

    train_logit = (
        x_train @ w
        + b
    )

    test_logit = (
        x_test @ w
        + b
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    train_logit,
                    -60,
                    60,
                )
            )
        ),
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    test_logit,
                    -60,
                    60,
                )
            )
        ),
    )


def build_table(
    rows,
    labels,
    probabilities,
    sem_by_sha,
    rel150,
):
    records = []

    for row, y, p in zip(
        rows,
        labels,
        probabilities,
    ):
        sha = norm_sha(
            row["sha256"]
        )

        if sha not in sem_by_sha:
            raise RuntimeError(
                "缺语义SHA: "
                + sha
            )

        z = float(
            logit(
                [p]
            )[0]
        )

        u = float(
            4.0
            * p
            * (
                1.0 - p
            )
        )

        base_pred = int(
            p >= 0.5
        )

        rec = {
            "sha256":
                sha,

            "label":
                int(y),

            "base_probability":
                float(p),

            "base_logit":
                z,

            "abs_base_logit":
                abs(z),

            "uncertainty":
                u,

            "base_pred":
                base_pred,

            "flippable":
                int(
                    u
                    > (
                        abs(z)
                        + 1e-6
                    )
                ),
        }

        srow = sem_by_sha[
            sha
        ]

        for name in rel150:
            try:
                value = float(
                    srow.get(
                        name,
                        0.0,
                    )
                )

                if not np.isfinite(
                    value
                ):
                    value = 0.0

            except Exception:
                value = 0.0

            rec[name] = value

        records.append(
            rec
        )

    return pd.DataFrame(
        records
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                COMPACT
            )
        )
    )

    with COMPACT_SCHEMA.open(
        "r",
        encoding="utf-8",
    ) as f:
        compact_schema = json.load(
            f
        )

    semantic_df, rel150 = (
        load_semantic()
    )

    sem_by_sha = {
        norm_sha(row["sha256"]):
            row
        for row in semantic_df.to_dict(
            orient="records"
        )
    }

    future_splits = (
        base.read_split_rows(
            SPLITS
        )
    )

    split_to_year = {
        name:
            int(
                name.rsplit(
                    "_",
                    1,
                )[-1]
            )
        for name in future_splits
    }

    fixed_train_rows = (
        next(
            iter(
                future_splits.values()
            )
        )["train"]
    )

    y_train = base.labels_for(
        fixed_train_rows,
        compact_by_sha,
    ).astype(int)

    # Stable15名字。
    stable15_names = list(
        stable15.stable15.STABLE15
    )

    feature_sets = {
        "base2":
            [
                "abs_base_logit",
                "uncertainty",
            ],

        "stable15":
            stable15_names,

        "stable15_base2":
            stable15_names
            + [
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

    rows_out = []

    seeds = [
        0,
        1,
        2,
        3,
        4,
    ]

    for seed in seeds:

        # 每个未来年份Stable70标准化都必须
        # 使用同一2017训练人口。
        # 为了避免重复训练，这里先任选一个未来split
        # 得到2017训练概率。
        first_name = sorted(
            future_splits,
            key=lambda x: split_to_year[x],
        )[0]

        first_test = (
            future_splits[
                first_name
            ]["test"]
        )

        (
            train_prob,
            _,
        ) = get_base_probabilities(
            fixed_train_rows,
            first_test,
            compact_by_sha,
            compact_schema,
            seed,
        )

        train_df = build_table(
            fixed_train_rows,
            y_train,
            train_prob,
            sem_by_sha,
            rel150,
        )

        # 只训练：
        # Stable70预测良性
        # 且当前有界公式理论可翻转。
        train_fn = train_df[
            (
                train_df[
                    "base_pred"
                ] == 0
            )
            & (
                train_df[
                    "flippable"
                ] == 1
            )
        ].copy()

        # target:
        # label=1就是Stable70漏报。
        train_target = (
            train_fn[
                "label"
            ].to_numpy(
                dtype=int
            )
        )

        print()
        print(
            "seed =",
            seed,
            "2017 FN candidate N =",
            len(train_fn),
            "FN =",
            int(
                train_target.sum()
            ),
            "rate =",
            f"{train_target.mean():.4f}",
        )

        if len(
            np.unique(
                train_target
            )
        ) < 2:
            raise RuntimeError(
                "2017 FN候选只有单一类别"
            )

        for split_name in sorted(
            future_splits,
            key=lambda x: split_to_year[x],
        ):
            year = split_to_year[
                split_name
            ]

            test_rows = (
                future_splits[
                    split_name
                ]["test"]
            )

            y_test = base.labels_for(
                test_rows,
                compact_by_sha,
            ).astype(int)

            (
                _,
                test_prob,
            ) = get_base_probabilities(
                fixed_train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
                seed,
            )

            test_df = build_table(
                test_rows,
                y_test,
                test_prob,
                sem_by_sha,
                rel150,
            )

            test_fn = test_df[
                (
                    test_df[
                        "base_pred"
                    ] == 0
                )
                & (
                    test_df[
                        "flippable"
                    ] == 1
                )
            ].copy()

            test_target = (
                test_fn[
                    "label"
                ].to_numpy(
                    dtype=int
                )
            )

            for feature_set, names in feature_sets.items():

                score = fit_predict(
                    train_fn[
                        names
                    ].to_numpy(
                        dtype=float
                    ),
                    train_target,
                    test_fn[
                        names
                    ].to_numpy(
                        dtype=float
                    ),
                )

                rows_out.append({
                    "seed":
                        seed,

                    "year":
                        year,

                    "feature_set":
                        feature_set,

                    "train_n":
                        len(train_fn),

                    "train_fn":
                        int(
                            train_target.sum()
                        ),

                    "test_n":
                        len(test_fn),

                    "test_fn":
                        int(
                            test_target.sum()
                        ),

                    "error_auc":
                        safe_auc(
                            test_target,
                            score,
                        ),
                })

    result = pd.DataFrame(
        rows_out
    )

    result.to_csv(
        OUT
        / "per_seed_year.csv",
        index=False,
    )

    by_year = (
        result
        .groupby(
            [
                "feature_set",
                "year",
            ],
            as_index=False,
        )
        .agg(
            error_auc=(
                "error_auc",
                "mean",
            ),
            train_n=(
                "train_n",
                "mean",
            ),
            train_fn=(
                "train_fn",
                "mean",
            ),
            test_n=(
                "test_n",
                "mean",
            ),
            test_fn=(
                "test_fn",
                "mean",
            ),
        )
    )

    summary = (
        by_year
        .groupby(
            "feature_set",
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
            "mean_auc",
            ascending=False,
        )
    )

    by_year.to_csv(
        OUT
        / "by_year.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 115
    )
    print(
        "2017训练 -> 未来FN错误风险"
    )
    print(
        "=" * 115
    )

    print(
        by_year.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "=" * 115
    )
    print(
        "四年汇总"
    )
    print(
        "=" * 115
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
