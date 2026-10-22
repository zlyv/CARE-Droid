#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


INPUT = Path(
    "artifacts/relation-pattern-neighborhood-diagnostic/"
    "sample_pattern_neighborhood.csv"
)

OUT = Path(
    "artifacts/pattern-preference-control-diagnostic"
)

A_CATEGORY = "A_wrong_residual_direction"
D_CATEGORY = "D_successfully_corrected"

BOOTSTRAP_REPEATS = 2000
CV_REPEATS = 100
RANDOM_SEED = 20260817

EPS = 1e-12


def build_design(df, include_preference):
    """
    结果变量：
        0 = A 残差方向错误
        1 = D 成功纠正

    控制变量：
        Compact margin
        log(1 + 活跃关系数)
        真实类别
        年份虚拟变量

    完整模型额外增加：
        class_neighborhood_preference

    连续变量只根据当前拟合数据标准化。
    """

    x = pd.DataFrame(
        index=df.index
    )

    x["margin"] = pd.to_numeric(
        df["base_margin_probability"],
        errors="raise",
    )

    x["log_relation_count"] = np.log1p(
        pd.to_numeric(
            df["active_relation_count"],
            errors="raise",
        )
    )

    x["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int)

    years = pd.to_numeric(
        df["year"],
        errors="raise",
    ).astype(int)

    # 2018作为参考年。
    for year in (2019, 2020, 2021):
        x[f"year_{year}"] = (
            years == year
        ).astype(float)

    if include_preference:
        x["pattern_preference"] = pd.to_numeric(
            df["class_neighborhood_preference"],
            errors="raise",
        )

    return x


def fit_standardized_logistic(
    x,
    y,
):
    """
    连续变量：
      margin
      log_relation_count
      pattern_preference
    做标准化。

    年份和label保持0/1。

    使用很弱的L2，仅为避免小样本数值不稳定。
    """

    x = x.copy()

    continuous = [
        name
        for name in (
            "margin",
            "log_relation_count",
            "pattern_preference",
        )
        if name in x.columns
    ]

    scaler = StandardScaler()

    if continuous:
        x.loc[:, continuous] = scaler.fit_transform(
            x[continuous]
        )

    model = LogisticRegression(
        penalty="l2",
        C=1000.0,
        solver="liblinear",
        max_iter=5000,
        random_state=0,
    )

    model.fit(
        x.to_numpy(dtype=float),
        y,
    )

    coef = dict(
        zip(
            x.columns,
            model.coef_[0],
        )
    )

    return model, scaler, continuous, coef


def transform_with_scaler(
    x,
    scaler,
    continuous,
):
    x = x.copy()

    if continuous:
        x.loc[:, continuous] = scaler.transform(
            x[continuous]
        )

    return x


def bootstrap_coefficient(
    df,
    repeats,
    seed,
):
    """
    按 A/D 分层自助抽样，
    保持两类样本数不变。

    每次重新拟合：
      year + label + margin + relation_count + preference

    记录 preference 的标准化系数。
    """

    rng = np.random.default_rng(
        seed
    )

    a = df[
        df["outcome_D"] == 0
    ].copy()

    d = df[
        df["outcome_D"] == 1
    ].copy()

    rows = []

    for repeat in range(
        repeats
    ):
        a_idx = rng.integers(
            0,
            len(a),
            size=len(a),
        )

        d_idx = rng.integers(
            0,
            len(d),
            size=len(d),
        )

        sample = pd.concat(
            [
                a.iloc[a_idx],
                d.iloc[d_idx],
            ],
            ignore_index=True,
        )

        x = build_design(
            sample,
            include_preference=True,
        )

        y = sample[
            "outcome_D"
        ].to_numpy(dtype=int)

        try:
            _, _, _, coef = (
                fit_standardized_logistic(
                    x,
                    y,
                )
            )

            value = float(
                coef[
                    "pattern_preference"
                ]
            )

        except Exception:
            value = np.nan

        rows.append({
            "repeat":
                repeat,

            "preference_coef":
                value,
        })

    return pd.DataFrame(
        rows
    )


def repeated_cv(
    df,
):
    """
    比较：

    基础：
      年份 + 类别 + margin + 关系数

    完整：
      基础 + 组合偏好

    使用重复分层5折，仅作为解释能力诊断。
    """

    y = df[
        "outcome_D"
    ].to_numpy(dtype=int)

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_SEED,
    )

    rows = []

    for fold_id, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            np.zeros(
                len(df)
            ),
            y,
        )
    ):
        train = df.iloc[
            train_idx
        ].copy()

        test = df.iloc[
            test_idx
        ].copy()

        y_train = train[
            "outcome_D"
        ].to_numpy(dtype=int)

        y_test = test[
            "outcome_D"
        ].to_numpy(dtype=int)

        fold_row = {
            "fold_id":
                fold_id,
        }

        for model_name, include_preference in (
            ("base", False),
            ("plus_preference", True),
        ):
            x_train = build_design(
                train,
                include_preference,
            )

            x_test = build_design(
                test,
                include_preference,
            )

            (
                model,
                scaler,
                continuous,
                _,
            ) = fit_standardized_logistic(
                x_train,
                y_train,
            )

            x_test = (
                transform_with_scaler(
                    x_test,
                    scaler,
                    continuous,
                )
            )

            probability = (
                model.predict_proba(
                    x_test.to_numpy(
                        dtype=float
                    )
                )[:, 1]
            )

            auc = roc_auc_score(
                y_test,
                probability,
            )

            fold_row[
                f"{model_name}_AUC"
            ] = float(
                auc
            )

        fold_row[
            "delta_AUC"
        ] = (
            fold_row[
                "plus_preference_AUC"
            ]
            -
            fold_row[
                "base_AUC"
            ]
        )

        rows.append(
            fold_row
        )

    return pd.DataFrame(
        rows
    )


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到输入文件: "
            + str(INPUT)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    required = {
        "year",
        "label",
        "mechanism_category",
        "base_margin_probability",
        "active_relation_count",
        "class_neighborhood_preference",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "缺少列: "
            + ", ".join(
                sorted(missing)
            )
        )

    # ========================================================
    # 只比较 A vs D。
    # ========================================================

    ad = df[
        df[
            "mechanism_category"
        ].isin(
            (
                A_CATEGORY,
                D_CATEGORY,
            )
        )
    ].copy()

    ad[
        "outcome_D"
    ] = (
        ad[
            "mechanism_category"
        ]
        == D_CATEGORY
    ).astype(int)

    print(
        "A方向错误 =",
        int(
            (
                ad[
                    "outcome_D"
                ] == 0
            ).sum()
        ),
    )

    print(
        "D成功纠正 =",
        int(
            (
                ad[
                    "outcome_D"
                ] == 1
            ).sum()
        ),
    )

    if len(ad) != 142:
        print(
            "注意：当前A+D人口不是此前的142，实际=",
            len(ad),
        )

    # ========================================================
    # 1. 全样本控制模型
    # ========================================================

    y = ad[
        "outcome_D"
    ].to_numpy(
        dtype=int
    )

    x_base = build_design(
        ad,
        include_preference=False,
    )

    x_full = build_design(
        ad,
        include_preference=True,
    )

    (
        base_model,
        base_scaler,
        base_cont,
        base_coef,
    ) = fit_standardized_logistic(
        x_base,
        y,
    )

    (
        full_model,
        full_scaler,
        full_cont,
        full_coef,
    ) = fit_standardized_logistic(
        x_full,
        y,
    )

    x_base_z = transform_with_scaler(
        x_base,
        base_scaler,
        base_cont,
    )

    x_full_z = transform_with_scaler(
        x_full,
        full_scaler,
        full_cont,
    )

    base_prob = (
        base_model.predict_proba(
            x_base_z.to_numpy(
                dtype=float
            )
        )[:, 1]
    )

    full_prob = (
        full_model.predict_proba(
            x_full_z.to_numpy(
                dtype=float
            )
        )[:, 1]
    )

    apparent_base_auc = (
        roc_auc_score(
            y,
            base_prob,
        )
    )

    apparent_full_auc = (
        roc_auc_score(
            y,
            full_prob,
        )
    )

    coefficient_rows = []

    for name, value in (
        full_coef.items()
    ):
        coefficient_rows.append({
            "feature":
                name,

            "coefficient":
                float(
                    value
                ),
        })

    coef_preference = float(
        full_coef[
            "pattern_preference"
        ]
    )

    # ========================================================
    # 2. 自助抽样
    # ========================================================

    bootstrap = (
        bootstrap_coefficient(
            ad,
            BOOTSTRAP_REPEATS,
            RANDOM_SEED,
        )
    )

    valid_boot = (
        bootstrap[
            "preference_coef"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        valid_boot
    ) == 0:
        raise SystemExit(
            "所有自助抽样拟合均失败"
        )

    boot_median = float(
        np.median(
            valid_boot
        )
    )

    boot_low = float(
        np.quantile(
            valid_boot,
            0.025,
        )
    )

    boot_high = float(
        np.quantile(
            valid_boot,
            0.975,
        )
    )

    boot_positive = float(
        np.mean(
            valid_boot
            > 0.0
        )
    )

    # ========================================================
    # 3. 重复交叉验证
    # ========================================================

    cv = repeated_cv(
        ad
    )

    delta_auc = (
        cv[
            "delta_AUC"
        ]
        .to_numpy(
            dtype=float
        )
    )

    cv_summary = {
        "folds":
            len(cv),

        "base_mean_AUC":
            float(
                cv[
                    "base_AUC"
                ].mean()
            ),

        "preference_mean_AUC":
            float(
                cv[
                    "plus_preference_AUC"
                ].mean()
            ),

        "mean_delta_AUC":
            float(
                delta_auc.mean()
            ),

        "median_delta_AUC":
            float(
                np.median(
                    delta_auc
                )
            ),

        "positive_delta_fraction":
            float(
                np.mean(
                    delta_auc > EPS
                )
            ),

        "negative_delta_fraction":
            float(
                np.mean(
                    delta_auc < -EPS
                )
            ),
    }

    # ========================================================
    # 4. 年份内A/D简单展示
    # ========================================================

    year_rows = []

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        yg = ad[
            pd.to_numeric(
                ad[
                    "year"
                ]
            ).astype(int)
            == year
        ]

        for outcome, name in (
            (
                0,
                "A_wrong_direction",
            ),
            (
                1,
                "D_corrected",
            ),
        ):
            g = yg[
                yg[
                    "outcome_D"
                ]
                == outcome
            ]

            year_rows.append({
                "year":
                    year,

                "group":
                    name,

                "N":
                    len(g),

                "mean_preference":
                    (
                        float(
                            g[
                                "class_neighborhood_preference"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_margin":
                    (
                        float(
                            g[
                                "base_margin_probability"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_relations":
                    (
                        float(
                            g[
                                "active_relation_count"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),
            })

    # ========================================================
    # 保存
    # ========================================================

    ad.to_csv(
        OUT
        / "A_D_samples.csv",
        index=False,
    )

    pd.DataFrame(
        coefficient_rows
    ).to_csv(
        OUT
        / "controlled_coefficients.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT
        / "bootstrap_preference_coefficients.csv",
        index=False,
    )

    cv.to_csv(
        OUT
        / "repeated_cv.csv",
        index=False,
    )

    pd.DataFrame(
        [
            cv_summary
        ]
    ).to_csv(
        OUT
        / "repeated_cv_summary.csv",
        index=False,
    )

    pd.DataFrame(
        year_rows
    ).to_csv(
        OUT
        / "year_A_D_summary.csv",
        index=False,
    )

    overall = pd.DataFrame([
        {
            "N":
                len(ad),

            "A_N":
                int(
                    (
                        y == 0
                    ).sum()
                ),

            "D_N":
                int(
                    (
                        y == 1
                    ).sum()
                ),

            "controlled_preference_coef":
                coef_preference,

            "bootstrap_valid":
                len(
                    valid_boot
                ),

            "bootstrap_median_coef":
                boot_median,

            "bootstrap_2p5":
                boot_low,

            "bootstrap_97p5":
                boot_high,

            "bootstrap_positive_fraction":
                boot_positive,

            "apparent_base_AUC":
                apparent_base_auc,

            "apparent_plus_preference_AUC":
                apparent_full_auc,

            "apparent_delta_AUC":
                (
                    apparent_full_auc
                    - apparent_base_auc
                ),

            **{
                "cv_"
                + k:
                    v
                for k, v in (
                    cv_summary.items()
                )
            },
        }
    ])

    overall.to_csv(
        OUT
        / "overall_summary.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "控制年份+类别+margin+关系数"
    )

    print(
        "======================================"
    )

    print(
        "组合偏好标准化系数 =",
        f"{coef_preference:+.4f}",
    )

    print()
    print(
        "======================================"
    )

    print(
        "2000次自助抽样"
    )

    print(
        "======================================"
    )

    print(
        "有效次数 =",
        len(
            valid_boot
        ),
    )

    print(
        "系数中位数 =",
        f"{boot_median:+.4f}",
    )

    print(
        "2.5%-97.5% =",
        f"[{boot_low:+.4f}, {boot_high:+.4f}]",
    )

    print(
        "系数 > 0 比例 =",
        f"{100.0 * boot_positive:.2f}%",
    )

    print()
    print(
        "======================================"
    )

    print(
        "重复5折交叉验证"
    )

    print(
        "======================================"
    )

    print(
        "基础平均AUC =",
        f"{cv_summary['base_mean_AUC']:.4f}",
    )

    print(
        "+组合偏好平均AUC =",
        f"{cv_summary['preference_mean_AUC']:.4f}",
    )

    print(
        "平均ΔAUC =",
        f"{cv_summary['mean_delta_AUC']:+.4f}",
    )

    print(
        "中位ΔAUC =",
        f"{cv_summary['median_delta_AUC']:+.4f}",
    )

    print(
        "ΔAUC>0比例 =",
        f"{100.0 * cv_summary['positive_delta_fraction']:.2f}%",
    )

    print()
    print(
        "======================================"
    )

    print(
        "判断原则"
    )

    print(
        "======================================"
    )

    print(
        "只有当："
    )

    print(
        "1) 控制后组合偏好系数仍 > 0；"
    )

    print(
        "2) 自助抽样绝大多数系数 > 0；"
    )

    print(
        "3) 交叉验证加入组合偏好后平均AUC稳定提高；"
    )

    print(
        "才值得回2017做可部署的组合邻域分数实验。"
    )

    print()
    print(
        "结果保存到:",
        OUT,
    )

    print()
    print(
        "注意：当前 class_neighborhood_preference "
        "使用未来真实标签，只用于机制诊断，"
        "绝不能作为实际分类特征。"
    )


if __name__ == "__main__":
    main()
