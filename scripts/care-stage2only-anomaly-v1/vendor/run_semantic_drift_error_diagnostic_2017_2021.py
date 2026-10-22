#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_frozen_anchor_residual_2017_2021 as frozen_base


SOURCE_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

FUTURE_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

SAMPLE_PATH = Path(
    "artifacts/frozen-count10-150-mechanism-diagnostic/"
    "sample_level.csv"
)

OUTPUT_DIR = Path(
    "artifacts/semantic-drift-error-diagnostic-2017-2021"
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)


def normalize_sha(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def safe_rate(a, b):
    if b == 0:
        return np.nan

    return float(a) / float(b)


def pct(x):
    if pd.isna(x):
        return "nan"

    return f"{100.0 * float(x):.2f}%"


def main():
    for path in (
        SOURCE_PATH,
        FUTURE_PATH,
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
    # 1. 读取2017与未来150维
    # ========================================================

    source = pd.read_csv(
        SOURCE_PATH,
        compression="gzip",
    )

    future = pd.read_csv(
        FUTURE_PATH,
        compression="gzip",
    )

    source["sha256"] = normalize_sha(
        source["sha256"]
    )

    future["sha256"] = normalize_sha(
        future["sha256"]
    )

    names150_source = (
        frozen_base.support150_names(
            source.columns
        )
    )

    names150_future = (
        frozen_base.support150_names(
            future.columns
        )
    )

    if len(
        names150_source
    ) != 150:
        raise SystemExit(
            "2017的final150列数量异常: "
            + str(
                len(
                    names150_source
                )
            )
        )

    if set(
        names150_source
    ) != set(
        names150_future
    ):
        raise SystemExit(
            "2017与未来150维列集合不一致"
        )

    # 统一顺序。
    names150 = list(
        names150_source
    )

    source_values = (
        source[
            names150
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    future_values = (
        future[
            names150
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        source_values
    ).all():
        raise SystemExit(
            "2017存在非有限语义值"
        )

    if not np.isfinite(
        future_values
    ).all():
        raise SystemExit(
            "未来存在非有限语义值"
        )

    # ========================================================
    # 2. 读取最终模型逐样本诊断
    #
    # 用它获得：
    # year / label / Compact是否错误 /
    # 最终是否纠正 / 是否退化
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

    required = {
        "year",
        "sha256",
        "label",
        "presence150",
        "base_correct",
        "base_margin",
        "count10_correct",
        "count10_corrected",
        "count10_regressed",
    }

    missing = (
        required
        - set(
            sample.columns
        )
    )

    if missing:
        raise SystemExit(
            "sample_level.csv缺少列: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # future语义人口和正式测试人口严格对应。
    future = future.merge(
        sample[
            [
                "sha256",
                "year",
            ]
        ].drop_duplicates(),
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    if len(future) != len(sample):
        raise SystemExit(
            "未来语义人口与正式1985测试人口不一致: "
            f"semantic={len(future)} "
            f"sample={len(sample)}"
        )

    # merge后重新按future当前行顺序取得矩阵，
    # 避免前面矩阵顺序已经失效。
    future_values = (
        future[
            names150
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # 3. 计算2017每个关系的激活频率
    # ========================================================

    source_active = (
        np.abs(
            source_values
        )
        > 1e-12
    )

    source_rate = (
        source_active.mean(
            axis=0
        )
    )

    positive_source_rates = (
        source_rate[
            source_rate > 0
        ]
    )

    if len(
        positive_source_rates
    ) == 0:
        raise SystemExit(
            "2017没有任何非零关系"
        )

    q25_relation = float(
        np.quantile(
            positive_source_rates,
            0.25,
        )
    )

    q75_relation = float(
        np.quantile(
            positive_source_rates,
            0.75,
        )
    )

    print(
        "======================================"
    )
    print(
        "2017关系频率基准"
    )
    print(
        "======================================"
    )

    print(
        "2017 APK数 =",
        len(source),
    )

    print(
        "final150 =",
        len(names150),
    )

    print(
        "2017有出现过的关系 =",
        int(
            (
                source_rate
                > 0
            ).sum()
        ),
    )

    print(
        "2017关系频率Q25 =",
        f"{q25_relation:.6f}",
    )

    print(
        "2017关系频率Q75 =",
        f"{q75_relation:.6f}",
    )

    # ========================================================
    # 4. 2017样本自身的“源域熟悉度”
    #
    # 对每个APK：
    #   找出它活跃的关系，
    #   取这些关系在2017总体中的平均出现率。
    #
    # 一个APK使用的关系越常见，
    # 这个分数越高。
    #
    # 熟悉度分桶阈值只从2017得到，
    # 不使用任何未来标签。
    # ========================================================

    source_sample_scores = []

    for i in range(
        len(source)
    ):
        active = (
            source_active[
                i
            ]
        )

        if not active.any():
            continue

        score = float(
            np.mean(
                source_rate[
                    active
                ]
            )
        )

        source_sample_scores.append(
            score
        )

    if not source_sample_scores:
        raise SystemExit(
            "无法计算2017样本熟悉度"
        )

    source_sample_scores = np.asarray(
        source_sample_scores,
        dtype=float,
    )

    familiarity_t1 = float(
        np.quantile(
            source_sample_scores,
            1.0 / 3.0,
        )
    )

    familiarity_t2 = float(
        np.quantile(
            source_sample_scores,
            2.0 / 3.0,
        )
    )

    print(
        "2017样本熟悉度低/中阈值 =",
        f"{familiarity_t1:.6f}",
    )

    print(
        "2017样本熟悉度中/高阈值 =",
        f"{familiarity_t2:.6f}",
    )

    # ========================================================
    # 5. 每年每个关系的漂移
    # ========================================================

    relation_rows = []

    year_rate_map = {}

    for year in YEARS:
        mask = (
            future[
                "year"
            ].to_numpy()
            == year
        )

        year_values = (
            future_values[
                mask
            ]
        )

        year_active = (
            np.abs(
                year_values
            )
            > 1e-12
        )

        year_rate = (
            year_active.mean(
                axis=0
            )
        )

        year_rate_map[
            year
        ] = year_rate

        for j, name in enumerate(
            names150
        ):
            sr = float(
                source_rate[j]
            )

            fr = float(
                year_rate[j]
            )

            delta = (
                fr
                - sr
            )

            # ----------------------------------------------
            # 关系分类：
            #
            # 高频/低频界线只由2017决定。
            # 未来只用于描述它是上升还是下降。
            # ----------------------------------------------

            if (
                sr == 0.0
                and fr > 0.0
            ):
                drift_class = (
                    "source_zero_emerging"
                )

            elif (
                sr >= q75_relation
                and fr < sr
            ):
                drift_class = (
                    "source_high_declining"
                )

            elif (
                sr >= q75_relation
                and fr >= sr
            ):
                drift_class = (
                    "source_high_nondeclining"
                )

            elif (
                0.0 < sr <= q25_relation
                and fr > sr
            ):
                drift_class = (
                    "source_low_increasing"
                )

            else:
                drift_class = (
                    "other"
                )

            relation_rows.append({
                "year":
                    year,

                "relation":
                    name,

                "source_2017_rate":
                    sr,

                "future_rate":
                    fr,

                "delta_future_minus_2017":
                    delta,

                "abs_delta":
                    abs(delta),

                "drift_class":
                    drift_class,
            })

    relation_drift = pd.DataFrame(
        relation_rows
    )

    # ========================================================
    # 6. 每个未来APK计算：
    #
    # - 活跃关系数
    # - 这些关系在2017有多常见
    # - 相对2017平均漂移多少
    # - 有多少属于源域高频衰退关系
    # - 有多少属于未来新出现关系
    # ========================================================

    future_feature_rows = []

    for i in range(
        len(future)
    ):
        year = int(
            future.iloc[
                i
            ][
                "year"
            ]
        )

        sha = str(
            future.iloc[
                i
            ][
                "sha256"
            ]
        )

        active = (
            np.abs(
                future_values[
                    i
                ]
            )
            > 1e-12
        )

        active_count = int(
            active.sum()
        )

        if active_count == 0:
            future_feature_rows.append({
                "year":
                    year,

                "sha256":
                    sha,

                "active_relation_count":
                    0,

                "source_familiarity":
                    0.0,

                "mean_active_relation_drift":
                    0.0,

                "frac_high_source_declining":
                    0.0,

                "frac_high_source_nondeclining":
                    0.0,

                "frac_low_source_increasing":
                    0.0,

                "frac_source_zero_emerging":
                    0.0,

                "dominant_drift_class":
                    "no_relation",

                "familiarity_bin":
                    "no_relation",
            })

            continue

        active_indices = np.flatnonzero(
            active
        )

        sr = source_rate[
            active_indices
        ]

        yr = year_rate_map[
            year
        ][
            active_indices
        ]

        familiarity = float(
            np.mean(sr)
        )

        mean_drift = float(
            np.mean(
                yr - sr
            )
        )

        classes = []

        for j in active_indices:
            relation_row = (
                relation_drift[
                    (
                        relation_drift[
                            "year"
                        ] == year
                    )
                    &
                    (
                        relation_drift[
                            "relation"
                        ] == names150[j]
                    )
                ]
                .iloc[0]
            )

            classes.append(
                relation_row[
                    "drift_class"
                ]
            )

        class_counts = (
            pd.Series(
                classes
            )
            .value_counts()
        )

        max_count = int(
            class_counts.max()
        )

        winners = (
            class_counts[
                class_counts
                == max_count
            ]
            .index
            .tolist()
        )

        if len(winners) == 1:
            dominant = (
                winners[0]
            )
        else:
            dominant = (
                "mixed"
            )

        if (
            familiarity
            <= familiarity_t1
        ):
            familiarity_bin = (
                "low_source_familiarity"
            )

        elif (
            familiarity
            <= familiarity_t2
        ):
            familiarity_bin = (
                "medium_source_familiarity"
            )

        else:
            familiarity_bin = (
                "high_source_familiarity"
            )

        def fraction(name):
            return (
                classes.count(
                    name
                )
                / active_count
            )

        future_feature_rows.append({
            "year":
                year,

            "sha256":
                sha,

            "active_relation_count":
                active_count,

            "source_familiarity":
                familiarity,

            "mean_active_relation_drift":
                mean_drift,

            "frac_high_source_declining":
                fraction(
                    "source_high_declining"
                ),

            "frac_high_source_nondeclining":
                fraction(
                    "source_high_nondeclining"
                ),

            "frac_low_source_increasing":
                fraction(
                    "source_low_increasing"
                ),

            "frac_source_zero_emerging":
                fraction(
                    "source_zero_emerging"
                ),

            "dominant_drift_class":
                dominant,

            "familiarity_bin":
                familiarity_bin,
        })

    future_features = pd.DataFrame(
        future_feature_rows
    )

    # ========================================================
    # 7. 与模型错误/纠错状态合并
    # ========================================================

    diagnostic = sample.merge(
        future_features,
        on=[
            "year",
            "sha256",
        ],
        how="left",
        validate="one_to_one",
    )

    if diagnostic[
        "active_relation_count"
    ].isna().any():
        raise SystemExit(
            "未来样本漂移特征合并失败"
        )

    presence_check = (
        (
            diagnostic[
                "active_relation_count"
            ] > 0
        ).astype(int)
        !=
        diagnostic[
            "presence150"
        ].astype(int)
    )

    if presence_check.any():
        raise SystemExit(
            "重新计算的关系存在状态与presence150不一致: "
            + str(
                int(
                    presence_check.sum()
                )
            )
        )

    diagnostic[
        "base_wrong"
    ] = (
        ~diagnostic[
            "base_correct"
        ].astype(bool)
    )

    diagnostic[
        "base_FP"
    ] = (
        diagnostic[
            "base_wrong"
        ]
        &
        (
            diagnostic[
                "label"
            ] == 0
        )
    )

    diagnostic[
        "base_FN"
    ] = (
        diagnostic[
            "base_wrong"
        ]
        &
        (
            diagnostic[
                "label"
            ] == 1
        )
    )

    # ========================================================
    # 8. 每年：纠正与未纠正错误的“2017熟悉度”
    # ========================================================

    error_familiarity_rows = []

    for year in YEARS:
        errors = diagnostic[
            (
                diagnostic[
                    "year"
                ] == year
            )
            &
            (
                diagnostic[
                    "base_wrong"
                ]
            )
            &
            (
                diagnostic[
                    "presence150"
                ] == 1
            )
        ]

        for status_name, mask in (
            (
                "corrected",
                errors[
                    "count10_corrected"
                ].astype(bool),
            ),
            (
                "uncorrected",
                ~errors[
                    "count10_corrected"
                ].astype(bool),
            ),
        ):
            g = errors[
                mask
            ]

            error_familiarity_rows.append({
                "year":
                    year,

                "status":
                    status_name,

                "N":
                    len(g),

                "mean_source_familiarity":
                    (
                        float(
                            g[
                                "source_familiarity"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "median_source_familiarity":
                    (
                        float(
                            g[
                                "source_familiarity"
                            ].median()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_active_relations":
                    (
                        float(
                            g[
                                "active_relation_count"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_relation_drift":
                    (
                        float(
                            g[
                                "mean_active_relation_drift"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_frac_high_source_declining":
                    (
                        float(
                            g[
                                "frac_high_source_declining"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_frac_source_zero_emerging":
                    (
                        float(
                            g[
                                "frac_source_zero_emerging"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_base_margin":
                    (
                        float(
                            g[
                                "base_margin"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),
            })

    error_familiarity = pd.DataFrame(
        error_familiarity_rows
    )

    # ========================================================
    # 9. 按源域熟悉度低/中/高看纠正率
    # ========================================================

    familiarity_rows = []

    familiarity_order = (
        "low_source_familiarity",
        "medium_source_familiarity",
        "high_source_familiarity",
    )

    for year in YEARS:
        errors = diagnostic[
            (
                diagnostic[
                    "year"
                ] == year
            )
            &
            (
                diagnostic[
                    "base_wrong"
                ]
            )
            &
            (
                diagnostic[
                    "presence150"
                ] == 1
            )
        ]

        for bin_name in (
            familiarity_order
        ):
            g = errors[
                errors[
                    "familiarity_bin"
                ] == bin_name
            ]

            corrected = int(
                g[
                    "count10_corrected"
                ].sum()
            )

            familiarity_rows.append({
                "year":
                    year,

                "familiarity_bin":
                    bin_name,

                "base_errors":
                    len(g),

                "corrected":
                    corrected,

                "correction_rate":
                    safe_rate(
                        corrected,
                        len(g),
                    ),

                "FP":
                    int(
                        g[
                            "base_FP"
                        ].sum()
                    ),

                "FN":
                    int(
                        g[
                            "base_FN"
                        ].sum()
                    ),

                "mean_margin":
                    (
                        float(
                            g[
                                "base_margin"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_active_relations":
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

    familiarity_summary = pd.DataFrame(
        familiarity_rows
    )

    # ========================================================
    # 10. 按主导漂移类型看Compact错误纠正率
    # ========================================================

    drift_class_rows = []

    for year in YEARS:
        errors = diagnostic[
            (
                diagnostic[
                    "year"
                ] == year
            )
            &
            (
                diagnostic[
                    "base_wrong"
                ]
            )
            &
            (
                diagnostic[
                    "presence150"
                ] == 1
            )
        ]

        for class_name, g in (
            errors.groupby(
                "dominant_drift_class"
            )
        ):
            corrected = int(
                g[
                    "count10_corrected"
                ].sum()
            )

            drift_class_rows.append({
                "year":
                    year,

                "dominant_drift_class":
                    class_name,

                "base_errors":
                    len(g),

                "corrected":
                    corrected,

                "correction_rate":
                    safe_rate(
                        corrected,
                        len(g),
                    ),

                "mean_source_familiarity":
                    float(
                        g[
                            "source_familiarity"
                        ].mean()
                    ),

                "mean_margin":
                    float(
                        g[
                            "base_margin"
                        ].mean()
                    ),
            })

    drift_class_summary = (
        pd.DataFrame(
            drift_class_rows
        )
    )

    # ========================================================
    # 11. 每个关系：
    # 在Compact错误中出现多少次，纠正多少次
    #
    # 注意：
    # 一个APK可以同时包含多个关系。
    # 因此这是共现诊断，不是因果贡献。
    # ========================================================

    relation_error_rows = []

    relation_index = {
        name:
            i
        for i, name in enumerate(
            names150
        )
    }

    future_index = {
        (
            int(
                future.iloc[i][
                    "year"
                ]
            ),
            str(
                future.iloc[i][
                    "sha256"
                ]
            ),
        ):
            i
        for i in range(
            len(future)
        )
    }

    for year in YEARS:
        errors = diagnostic[
            (
                diagnostic[
                    "year"
                ] == year
            )
            &
            (
                diagnostic[
                    "base_wrong"
                ]
            )
            &
            (
                diagnostic[
                    "presence150"
                ] == 1
            )
        ]

        for name in names150:
            j = relation_index[
                name
            ]

            active_error_rows = []

            for _, row in (
                errors.iterrows()
            ):
                idx = future_index[
                    (
                        year,
                        row[
                            "sha256"
                        ],
                    )
                ]

                if (
                    abs(
                        future_values[
                            idx,
                            j,
                        ]
                    )
                    > 1e-12
                ):
                    active_error_rows.append(
                        row
                    )

            n = len(
                active_error_rows
            )

            if n == 0:
                continue

            corrected = sum(
                bool(
                    row[
                        "count10_corrected"
                    ]
                )
                for row
                in active_error_rows
            )

            relation_row = (
                relation_drift[
                    (
                        relation_drift[
                            "year"
                        ] == year
                    )
                    &
                    (
                        relation_drift[
                            "relation"
                        ] == name
                    )
                ]
                .iloc[0]
            )

            relation_error_rows.append({
                "year":
                    year,

                "relation":
                    name,

                "source_2017_rate":
                    float(
                        relation_row[
                            "source_2017_rate"
                        ]
                    ),

                "future_rate":
                    float(
                        relation_row[
                            "future_rate"
                        ]
                    ),

                "delta":
                    float(
                        relation_row[
                            "delta_future_minus_2017"
                        ]
                    ),

                "drift_class":
                    relation_row[
                        "drift_class"
                    ],

                "base_errors_with_relation":
                    n,

                "corrected_with_relation":
                    int(
                        corrected
                    ),

                "correction_rate":
                    safe_rate(
                        corrected,
                        n,
                    ),
            })

    relation_error = pd.DataFrame(
        relation_error_rows
    )

    # ========================================================
    # 12. 保存
    # ========================================================

    relation_drift.to_csv(
        OUTPUT_DIR
        / "relation_frequency_2017_2021.csv",
        index=False,
    )

    diagnostic.to_csv(
        OUTPUT_DIR
        / "sample_semantic_drift_diagnostic.csv",
        index=False,
    )

    error_familiarity.to_csv(
        OUTPUT_DIR
        / "corrected_vs_uncorrected_familiarity.csv",
        index=False,
    )

    familiarity_summary.to_csv(
        OUTPUT_DIR
        / "error_correction_by_source_familiarity.csv",
        index=False,
    )

    drift_class_summary.to_csv(
        OUTPUT_DIR
        / "error_correction_by_drift_class.csv",
        index=False,
    )

    relation_error.to_csv(
        OUTPUT_DIR
        / "relation_error_correction.csv",
        index=False,
    )

    # 最大变化关系
    (
        relation_drift
        .sort_values(
            [
                "year",
                "abs_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "year",
            as_index=False,
        )
        .head(20)
        .to_csv(
            OUTPUT_DIR
            / "top20_relation_drift_each_year.csv",
            index=False,
        )
    )

    # ========================================================
    # 13. 控制台核心结果
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "纠正错误 vs 未纠正错误：源域熟悉度"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            year
        )

        yg = error_familiarity[
            error_familiarity[
                "year"
            ] == year
        ]

        for _, row in (
            yg.iterrows()
        ):
            print(
                f"{row['status']:12s} "
                f"N={int(row['N']):3d} "
                f"熟悉度mean="
                f"{row['mean_source_familiarity']:.5f} "
                f"median="
                f"{row['median_source_familiarity']:.5f} "
                f"关系数="
                f"{row['mean_active_relations']:.2f} "
                f"漂移="
                f"{row['mean_relation_drift']:+.5f} "
                f"margin="
                f"{row['mean_base_margin']:.4f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "按2017源域熟悉度分组的Compact错误纠正率"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            year
        )

        yg = familiarity_summary[
            familiarity_summary[
                "year"
            ] == year
        ]

        for bin_name in (
            familiarity_order
        ):
            row = yg[
                yg[
                    "familiarity_bin"
                ] == bin_name
            ].iloc[0]

            print(
                f"{bin_name:28s} "
                f"errors="
                f"{int(row['base_errors']):3d} "
                f"corrected="
                f"{int(row['corrected']):3d} "
                f"rate="
                f"{pct(row['correction_rate']):>7s} "
                f"margin="
                f"{row['mean_margin']:.4f} "
                f"关系数="
                f"{row['mean_active_relations']:.2f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "每年关系整体漂移概况"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        g = relation_drift[
            relation_drift[
                "year"
            ] == year
        ]

        print()
        print(
            year
        )

        print(
            " 平均|频率变化| =",
            f"{g['abs_delta'].mean():.5f}",
        )

        print(
            " 源域高频但下降 =",
            int(
                (
                    g[
                        "drift_class"
                    ]
                    ==
                    "source_high_declining"
                ).sum()
            ),
        )

        print(
            " 源域高频且不下降 =",
            int(
                (
                    g[
                        "drift_class"
                    ]
                    ==
                    "source_high_nondeclining"
                ).sum()
            ),
        )

        print(
            " 源域低频但上升 =",
            int(
                (
                    g[
                        "drift_class"
                    ]
                    ==
                    "source_low_increasing"
                ).sum()
            ),
        )

        print(
            " 2017为0、未来出现 =",
            int(
                (
                    g[
                        "drift_class"
                    ]
                    ==
                    "source_zero_emerging"
                ).sum()
            ),
        )

    print()
    print(
        "======================================"
    )
    print(
        "主导关系漂移类型与纠正率"
    )
    print(
        "======================================"
    )

    if len(
        drift_class_summary
    ):
        for year in YEARS:
            print()
            print(
                year
            )

            yg = (
                drift_class_summary[
                    drift_class_summary[
                        "year"
                    ] == year
                ]
                .sort_values(
                    "base_errors",
                    ascending=False,
                )
            )

            for _, row in (
                yg.iterrows()
            ):
                print(
                    f"{row['dominant_drift_class']:30s} "
                    f"errors="
                    f"{int(row['base_errors']):3d} "
                    f"corrected="
                    f"{int(row['corrected']):3d} "
                    f"rate="
                    f"{pct(row['correction_rate']):>7s} "
                    f"熟悉度="
                    f"{row['mean_source_familiarity']:.5f}"
                )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "注意：关系漂移分类仅用于诊断，"
        "没有参与训练、模型选择或未来参数调整。"
    )


if __name__ == "__main__":
    main()
