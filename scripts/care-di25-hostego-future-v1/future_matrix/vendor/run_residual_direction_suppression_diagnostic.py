#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRED_PATH = Path(
    "artifacts/frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

SAMPLE_PATH = Path(
    "artifacts/frozen-count10-150-mechanism-diagnostic/"
    "sample_level.csv"
)

OUTPUT_DIR = Path(
    "artifacts/residual-direction-suppression-diagnostic"
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

EPS = 1e-12


def normalize_sha(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def clip_prob(x):
    return np.clip(
        np.asarray(
            x,
            dtype=float,
        ),
        1e-9,
        1.0 - 1e-9,
    )


def logit(p):
    p = clip_prob(p)
    return np.log(
        p / (1.0 - p)
    )


def sigmoid(z):
    z = np.asarray(
        z,
        dtype=float,
    )

    out = np.empty_like(z)

    positive = (
        z >= 0
    )

    out[positive] = (
        1.0
        /
        (
            1.0
            + np.exp(
                -z[positive]
            )
        )
    )

    ez = np.exp(
        z[
            ~positive
        ]
    )

    out[
        ~positive
    ] = (
        ez
        /
        (
            1.0
            + ez
        )
    )

    return out


def safe_rate(a, b):
    if b == 0:
        return np.nan

    return (
        float(a)
        / float(b)
    )


def pct(x):
    if pd.isna(x):
        return "nan"

    return (
        f"{100.0 * float(x):.2f}%"
    )


def margin_bin(x):
    x = float(x)

    if x < 0.05:
        return "0-0.05"

    if x < 0.10:
        return "0.05-0.10"

    if x < 0.20:
        return "0.10-0.20"

    if x < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def main():
    for path in (
        PRED_PATH,
        SAMPLE_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在: "
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取逐随机种子预测
    # ========================================================

    pred = pd.read_csv(
        PRED_PATH
    )

    required = {
        "year",
        "sha256",
        "seed",
        "label",
        "probability",
        "base_probability",
    }

    missing = (
        required
        - set(
            pred.columns
        )
    )

    if missing:
        raise SystemExit(
            "预测文件缺少列: "
            + ", ".join(
                sorted(
                    missing
                )
            )
            + "\n实际列:\n"
            + ", ".join(
                pred.columns
            )
        )

    pred["sha256"] = normalize_sha(
        pred["sha256"]
    )

    pred["year"] = (
        pred["year"]
        .astype(int)
    )

    pred["seed"] = (
        pred["seed"]
        .astype(int)
    )

    pred["label"] = (
        pred["label"]
        .astype(int)
    )

    # ========================================================
    # 2. 取得presence150
    #
    # 使用已经验证过的样本级诊断结果，
    # 不重新定义presence。
    # ========================================================

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    sample["sha256"] = normalize_sha(
        sample["sha256"]
    )

    sample["year"] = (
        sample["year"]
        .astype(int)
    )

    presence = (
        sample[
            [
                "year",
                "sha256",
                "presence150",
            ]
        ]
        .drop_duplicates()
    )

    if presence.duplicated(
        [
            "year",
            "sha256",
        ]
    ).any():
        raise SystemExit(
            "presence表存在重复APK"
        )

    pred = pred.merge(
        presence,
        on=[
            "year",
            "sha256",
        ],
        how="left",
        validate="many_to_one",
    )

    if pred[
        "presence150"
    ].isna().any():
        raise SystemExit(
            "部分预测样本缺少presence150"
        )

    pred[
        "presence150"
    ] = (
        pred[
            "presence150"
        ]
        .astype(int)
    )

    # ========================================================
    # 3. 从概率反推logit修正
    # ========================================================

    pred[
        "base_logit"
    ] = logit(
        pred[
            "base_probability"
        ]
    )

    pred[
        "final_logit"
    ] = logit(
        pred[
            "probability"
        ]
    )

    pred[
        "actual_logit_correction"
    ] = (
        pred[
            "final_logit"
        ]
        -
        pred[
            "base_logit"
        ]
    )

    base_p = (
        pred[
            "base_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    pred[
        "uncertainty"
    ] = (
        4.0
        * base_p
        * (
            1.0
            - base_p
        )
    )

    # --------------------------------------------------------
    # 理论上：
    #
    # actual_correction
    # =
    # presence * uncertainty * tanh(delta)
    #
    # presence=1时：
    #
    # tanh(delta)
    # =
    # actual_correction / uncertainty
    #
    # 这个量我们叫 raw_residual，
    # 只是机制诊断，不修改实际模型。
    # --------------------------------------------------------

    pred[
        "raw_residual"
    ] = 0.0

    active = (
        pred[
            "presence150"
        ] == 1
    )

    valid_u = (
        pred[
            "uncertainty"
        ].abs()
        > EPS
    )

    valid = (
        active
        &
        valid_u
    )

    pred.loc[
        valid,
        "raw_residual",
    ] = (
        pred.loc[
            valid,
            "actual_logit_correction",
        ]
        /
        pred.loc[
            valid,
            "uncertainty",
        ]
    )

    # tanh理论范围[-1,1]。
    max_abs_raw = float(
        pred.loc[
            active,
            "raw_residual",
        ]
        .abs()
        .max()
    )

    print(
        "======================================"
    )
    print(
        "残差反推一致性"
    )
    print(
        "======================================"
    )

    print(
        "presence150=1 最大 |反推tanh(delta)| =",
        f"{max_abs_raw:.12f}",
    )

    if max_abs_raw > 1.00001:
        print(
            "警告：反推值超过tanh理论范围，"
            "请检查保存概率精度或模型公式。"
        )

    # --------------------------------------------------------
    # 重新构造实际修正，检查是否与保存结果一致。
    # --------------------------------------------------------

    pred[
        "reconstructed_correction"
    ] = (
        pred[
            "presence150"
        ]
        *
        pred[
            "uncertainty"
        ]
        *
        pred[
            "raw_residual"
        ]
    )

    pred[
        "reconstruction_error"
    ] = np.abs(
        pred[
            "actual_logit_correction"
        ]
        -
        pred[
            "reconstructed_correction"
        ]
    )

    print(
        "修正重构最大误差 =",
        f"{pred['reconstruction_error'].max():.12g}",
    )

    # ========================================================
    # 4. 判断基础分类和最终分类
    # ========================================================

    pred[
        "base_prediction"
    ] = (
        pred[
            "base_probability"
        ]
        >= 0.5
    ).astype(int)

    pred[
        "final_prediction"
    ] = (
        pred[
            "probability"
        ]
        >= 0.5
    ).astype(int)

    pred[
        "base_correct"
    ] = (
        pred[
            "base_prediction"
        ]
        ==
        pred[
            "label"
        ]
    )

    pred[
        "final_correct"
    ] = (
        pred[
            "final_prediction"
        ]
        ==
        pred[
            "label"
        ]
    )

    pred[
        "base_margin_probability"
    ] = np.abs(
        pred[
            "base_probability"
        ]
        - 0.5
    )

    pred[
        "margin_bin"
    ] = pred[
        "base_margin_probability"
    ].map(
        margin_bin
    )

    # ========================================================
    # 5. 真值方向
    #
    # label=1：
    # logit应该向正方向移动
    #
    # label=0：
    # logit应该向负方向移动
    # ========================================================

    pred[
        "truth_direction"
    ] = (
        2.0
        * pred[
            "label"
        ]
        - 1.0
    )

    pred[
        "raw_truth_correction"
    ] = (
        pred[
            "truth_direction"
        ]
        *
        pred[
            "raw_residual"
        ]
    )

    pred[
        "actual_truth_correction"
    ] = (
        pred[
            "truth_direction"
        ]
        *
        pred[
            "actual_logit_correction"
        ]
    )

    # --------------------------------------------------------
    # 对一个Compact已经预测错误的样本：
    #
    # 距离logit决策边界0所需的修正量
    # 就是 |base_logit|
    # --------------------------------------------------------

    pred[
        "required_logit_to_boundary"
    ] = np.abs(
        pred[
            "base_logit"
        ]
    )

    # ========================================================
    # 6. 去掉不确定度系数的“反事实诊断”
    #
    # 注意：
    # 这不是新的模型结果，也不是性能报告。
    #
    # 只是问：
    # 如果保留已经学到的tanh(delta)，
    # 但暂时不乘u，
    # 它在方向和幅度上是否足以跨过边界？
    # ========================================================

    pred[
        "counterfactual_no_u_logit"
    ] = (
        pred[
            "base_logit"
        ]
        +
        pred[
            "presence150"
        ]
        *
        pred[
            "raw_residual"
        ]
    )

    pred[
        "counterfactual_no_u_probability"
    ] = sigmoid(
        pred[
            "counterfactual_no_u_logit"
        ]
    )

    pred[
        "counterfactual_no_u_prediction"
    ] = (
        pred[
            "counterfactual_no_u_probability"
        ]
        >= 0.5
    ).astype(int)

    pred[
        "counterfactual_no_u_correct"
    ] = (
        pred[
            "counterfactual_no_u_prediction"
        ]
        ==
        pred[
            "label"
        ]
    )

    # ========================================================
    # 7. 对Compact错误 + presence150=1分类
    # ========================================================

    def classify_row(row):
        if bool(
            row[
                "base_correct"
            ]
        ):
            return "base_correct"

        if int(
            row[
                "presence150"
            ]
        ) == 0:
            return "no_semantic_relation"

        if bool(
            row[
                "final_correct"
            ]
        ):
            return "D_successfully_corrected"

        raw_toward = float(
            row[
                "raw_truth_correction"
            ]
        )

        actual_toward = float(
            row[
                "actual_truth_correction"
            ]
        )

        needed = float(
            row[
                "required_logit_to_boundary"
            ]
        )

        if raw_toward <= 0.0:
            return "A_wrong_residual_direction"

        # raw残差足以越界，
        # 但乘uncertainty之后不够。
        if (
            raw_toward
            >= needed
            and
            actual_toward
            < needed
        ):
            return "B_suppressed_by_uncertainty"

        # 方向正确，
        # 但即使完全去掉uncertainty也不够越界。
        if raw_toward < needed:
            return "C_correct_direction_but_too_small"

        return "E_other_check"

    pred[
        "mechanism_category"
    ] = pred.apply(
        classify_row,
        axis=1,
    )

    # ========================================================
    # 8. 先逐随机种子分析
    # ========================================================

    category_order = (
        "A_wrong_residual_direction",
        "B_suppressed_by_uncertainty",
        "C_correct_direction_but_too_small",
        "D_successfully_corrected",
    )

    seed_rows = []

    for year in YEARS:
        errors = pred[
            (
                pred[
                    "year"
                ] == year
            )
            &
            (
                ~pred[
                    "base_correct"
                ]
            )
            &
            (
                pred[
                    "presence150"
                ] == 1
            )
        ]

        total = len(
            errors
        )

        for category in (
            category_order
        ):
            g = errors[
                errors[
                    "mechanism_category"
                ] == category
            ]

            seed_rows.append({
                "year":
                    year,

                "category":
                    category,

                "N":
                    len(g),

                "share":
                    safe_rate(
                        len(g),
                        total,
                    ),

                "mean_base_probability_margin":
                    (
                        float(
                            g[
                                "base_margin_probability"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_uncertainty":
                    (
                        float(
                            g[
                                "uncertainty"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_required_logit":
                    (
                        float(
                            g[
                                "required_logit_to_boundary"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_raw_truth_correction":
                    (
                        float(
                            g[
                                "raw_truth_correction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_actual_truth_correction":
                    (
                        float(
                            g[
                                "actual_truth_correction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "FP":
                    int(
                        (
                            (
                                g[
                                    "label"
                                ] == 0
                            )
                        ).sum()
                    ),

                "FN":
                    int(
                        (
                            (
                                g[
                                    "label"
                                ] == 1
                            )
                        ).sum()
                    ),
            })

    seed_summary = pd.DataFrame(
        seed_rows
    )

    # ========================================================
    # 9. 按概率距离分桶，看A/B/C/D如何变化
    # ========================================================

    margin_rows = []

    margin_order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    for year in YEARS:
        errors = pred[
            (
                pred[
                    "year"
                ] == year
            )
            &
            (
                ~pred[
                    "base_correct"
                ]
            )
            &
            (
                pred[
                    "presence150"
                ] == 1
            )
        ]

        for margin_name in (
            margin_order
        ):
            mg = errors[
                errors[
                    "margin_bin"
                ] == margin_name
            ]

            row = {
                "year":
                    year,

                "margin":
                    margin_name,

                "N":
                    len(mg),
            }

            for category in (
                category_order
            ):
                row[
                    category
                ] = int(
                    (
                        mg[
                            "mechanism_category"
                        ]
                        == category
                    ).sum()
                )

            margin_rows.append(
                row
            )

    margin_summary = pd.DataFrame(
        margin_rows
    )

    # ========================================================
    # 10. 对5个随机种子概率先平均，再做一次APK级分析
    #
    # 这与之前样本级机制诊断口径一致。
    #
    # 注意：
    # raw_residual不能直接对种子平均，
    # 所以先平均base/final概率，
    # 再通过平均概率重新反推。
    # ========================================================

    apk = (
        pred.groupby(
            [
                "year",
                "sha256",
            ],
            as_index=False,
        )
        .agg(
            label=(
                "label",
                "first",
            ),
            presence150=(
                "presence150",
                "first",
            ),
            base_probability=(
                "base_probability",
                "mean",
            ),
            probability=(
                "probability",
                "mean",
            ),
            seed_n=(
                "seed",
                "nunique",
            ),
        )
    )

    apk[
        "base_logit"
    ] = logit(
        apk[
            "base_probability"
        ]
    )

    apk[
        "final_logit"
    ] = logit(
        apk[
            "probability"
        ]
    )

    apk[
        "actual_logit_correction"
    ] = (
        apk[
            "final_logit"
        ]
        -
        apk[
            "base_logit"
        ]
    )

    p = apk[
        "base_probability"
    ].to_numpy(
        dtype=float
    )

    apk[
        "uncertainty"
    ] = (
        4.0
        * p
        * (
            1.0
            - p
        )
    )

    apk[
        "raw_residual"
    ] = 0.0

    valid = (
        (
            apk[
                "presence150"
            ] == 1
        )
        &
        (
            apk[
                "uncertainty"
            ].abs()
            > EPS
        )
    )

    apk.loc[
        valid,
        "raw_residual",
    ] = (
        apk.loc[
            valid,
            "actual_logit_correction",
        ]
        /
        apk.loc[
            valid,
            "uncertainty",
        ]
    )

    apk[
        "base_prediction"
    ] = (
        apk[
            "base_probability"
        ]
        >= 0.5
    ).astype(int)

    apk[
        "final_prediction"
    ] = (
        apk[
            "probability"
        ]
        >= 0.5
    ).astype(int)

    apk[
        "base_correct"
    ] = (
        apk[
            "base_prediction"
        ]
        ==
        apk[
            "label"
        ]
    )

    apk[
        "final_correct"
    ] = (
        apk[
            "final_prediction"
        ]
        ==
        apk[
            "label"
        ]
    )

    apk[
        "truth_direction"
    ] = (
        2.0
        * apk[
            "label"
        ]
        - 1.0
    )

    apk[
        "raw_truth_correction"
    ] = (
        apk[
            "truth_direction"
        ]
        *
        apk[
            "raw_residual"
        ]
    )

    apk[
        "actual_truth_correction"
    ] = (
        apk[
            "truth_direction"
        ]
        *
        apk[
            "actual_logit_correction"
        ]
    )

    apk[
        "required_logit_to_boundary"
    ] = np.abs(
        apk[
            "base_logit"
        ]
    )

    apk[
        "base_margin_probability"
    ] = np.abs(
        apk[
            "base_probability"
        ]
        - 0.5
    )

    apk[
        "margin_bin"
    ] = apk[
        "base_margin_probability"
    ].map(
        margin_bin
    )

    apk[
        "mechanism_category"
    ] = apk.apply(
        classify_row,
        axis=1,
    )

    apk_rows = []

    for year in YEARS:
        errors = apk[
            (
                apk[
                    "year"
                ] == year
            )
            &
            (
                ~apk[
                    "base_correct"
                ]
            )
            &
            (
                apk[
                    "presence150"
                ] == 1
            )
        ]

        total = len(
            errors
        )

        for category in (
            category_order
        ):
            g = errors[
                errors[
                    "mechanism_category"
                ] == category
            ]

            apk_rows.append({
                "year":
                    year,

                "category":
                    category,

                "N":
                    len(g),

                "share":
                    safe_rate(
                        len(g),
                        total,
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

                "mean_uncertainty":
                    (
                        float(
                            g[
                                "uncertainty"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_required_logit":
                    (
                        float(
                            g[
                                "required_logit_to_boundary"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_raw_truth_correction":
                    (
                        float(
                            g[
                                "raw_truth_correction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_actual_truth_correction":
                    (
                        float(
                            g[
                                "actual_truth_correction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "FP":
                    int(
                        (
                            g[
                                "label"
                            ] == 0
                        ).sum()
                    ),

                "FN":
                    int(
                        (
                            g[
                                "label"
                            ] == 1
                        ).sum()
                    ),
            })

    apk_summary = pd.DataFrame(
        apk_rows
    )

    # ========================================================
    # 11. 总体汇总
    # ========================================================

    all_rows = []

    for level_name, frame in (
        (
            "seed_level",
            pred,
        ),
        (
            "apk_mean_probability",
            apk,
        ),
    ):
        errors = frame[
            (
                ~frame[
                    "base_correct"
                ]
            )
            &
            (
                frame[
                    "presence150"
                ] == 1
            )
        ]

        for category in (
            category_order
        ):
            g = errors[
                errors[
                    "mechanism_category"
                ] == category
            ]

            all_rows.append({
                "level":
                    level_name,

                "category":
                    category,

                "N":
                    len(g),

                "share":
                    safe_rate(
                        len(g),
                        len(errors),
                    ),
            })

    overall_summary = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # 12. 保存
    # ========================================================

    pred.to_csv(
        OUTPUT_DIR
        / "seed_level_diagnostic.csv",
        index=False,
    )

    apk.to_csv(
        OUTPUT_DIR
        / "apk_mean_probability_diagnostic.csv",
        index=False,
    )

    seed_summary.to_csv(
        OUTPUT_DIR
        / "seed_level_year_summary.csv",
        index=False,
    )

    apk_summary.to_csv(
        OUTPUT_DIR
        / "apk_level_year_summary.csv",
        index=False,
    )

    margin_summary.to_csv(
        OUTPUT_DIR
        / "seed_level_margin_breakdown.csv",
        index=False,
    )

    overall_summary.to_csv(
        OUTPUT_DIR
        / "overall_summary.csv",
        index=False,
    )

    # 单独保存B类，后面可以逐样本检查。
    pred[
        pred[
            "mechanism_category"
        ]
        ==
        "B_suppressed_by_uncertainty"
    ].to_csv(
        OUTPUT_DIR
        / "uncertainty_suppressed_seed_samples.csv",
        index=False,
    )

    apk[
        apk[
            "mechanism_category"
        ]
        ==
        "B_suppressed_by_uncertainty"
    ].to_csv(
        OUTPUT_DIR
        / "uncertainty_suppressed_apks.csv",
        index=False,
    )

    # ========================================================
    # 13. 打印
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "APK级：Compact错误 + 150存在 的机制分解"
    )
    print(
        "======================================"
    )

    readable = {
        "A_wrong_residual_direction":
            "A 纠错方向错误",

        "B_suppressed_by_uncertainty":
            "B 方向正确但被不确定度压住",

        "C_correct_direction_but_too_small":
            "C 方向正确但原始幅度也不足",

        "D_successfully_corrected":
            "D 成功纠正",
    }

    for year in YEARS:
        yg = apk_summary[
            apk_summary[
                "year"
            ] == year
        ]

        total = int(
            yg[
                "N"
            ].sum()
        )

        print()
        print(
            year,
            "可介入Compact错误 =",
            total,
        )

        for category in (
            category_order
        ):
            row = yg[
                yg[
                    "category"
                ] == category
            ].iloc[0]

            print(
                f"{readable[category]:32s} "
                f"N={int(row['N']):3d} "
                f"share={pct(row['share']):>7s} "
                f"margin={row['mean_margin']:.4f} "
                f"u={row['mean_uncertainty']:.4f} "
                f"需要={row['mean_required_logit']:.4f} "
                f"原始修正={row['mean_raw_truth_correction']:.4f} "
                f"实际修正={row['mean_actual_truth_correction']:.4f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "APK级总计"
    )
    print(
        "======================================"
    )

    all_apk = overall_summary[
        overall_summary[
            "level"
        ]
        ==
        "apk_mean_probability"
    ]

    for category in (
        category_order
    ):
        row = all_apk[
            all_apk[
                "category"
            ] == category
        ].iloc[0]

        print(
            f"{readable[category]:32s} "
            f"N={int(row['N']):3d} "
            f"share={pct(row['share']):>7s}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐随机种子结果"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        yg = seed_summary[
            seed_summary[
                "year"
            ] == year
        ]

        print()
        print(
            year
        )

        for category in (
            category_order
        ):
            row = yg[
                yg[
                    "category"
                ] == category
            ].iloc[0]

            print(
                f"{readable[category]:32s} "
                f"N={int(row['N']):4d} "
                f"share={pct(row['share']):>7s}"
            )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "重要：B类仅表示一个机制反事实："
        "现有残差方向和未乘不确定度时的幅度足够，"
        "但实际乘上4p(1-p)后没有跨过决策边界。"
    )

    print(
        "不能把去掉不确定度后的反事实结果"
        "当作新的正式模型性能。"
    )


if __name__ == "__main__":
    main()
