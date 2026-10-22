#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_frozen_anchor_residual_2017_2021 as frozen_base


# ============================================================
# 输入
# ============================================================

SAMPLE_PATH = Path(
    "artifacts/frozen-count10-150-mechanism-diagnostic/"
    "sample_level.csv"
)

COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

COUNT10_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

OUTPUT_DIR = Path(
    "artifacts/year-internal-diagnostic-2018-2021"
)


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

COUNT_METRICS = (
    "caller_log",
    "caller_ratio",
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


def margin_group(value):
    value = float(value)

    if value < 0.05:
        return "0-0.05"
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.20:
        return "0.10-0.20"
    if value < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def pct_text(x):
    if pd.isna(x):
        return "nan"
    return f"{100.0 * float(x):.2f}%"


def main():
    for path in (
        SAMPLE_PATH,
        COMPACT_PATH,
        SEMANTIC_PATH,
        COUNT10_PATH,
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
    # 1. 已经完成的最终逐样本诊断
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

    required_sample = {
        "year",
        "sha256",
        "label",
        "presence150",
        "base_probability",
        "base_prediction",
        "base_correct",
        "base_margin",
        "count10_probability",
        "count10_correct",
        "count10_corrected",
        "count10_regressed",
        "count10_better_than_150",
        "count10_worse_than_150",
    }

    missing = (
        required_sample
        - set(sample.columns)
    )

    if missing:
        raise SystemExit(
            "sample_level.csv 缺少列: "
            + ", ".join(
                sorted(missing)
            )
        )

    # ========================================================
    # 2. Compact：判断APK本身是否含Top-5语义
    # ========================================================

    compact = pd.read_csv(
        COMPACT_PATH
    )

    compact["sha256"] = normalize_sha(
        compact["sha256"]
    )

    compact_columns = [
        f"role_node__{role}"
        for role in TOP5
    ]

    missing = (
        set(compact_columns)
        - set(compact.columns)
    )

    if missing:
        raise SystemExit(
            "Compact缺少Top-5原始语义列: "
            + ", ".join(
                sorted(missing)
            )
        )

    compact_small = compact[
        [
            "sha256",
            *compact_columns,
        ]
    ].copy()

    for role in TOP5:
        raw_col = (
            f"role_node__{role}"
        )

        compact_small[
            f"top5_present__{role}"
        ] = (
            pd.to_numeric(
                compact_small[
                    raw_col
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .abs()
            > 1e-12
        ).astype(int)

    top5_presence_columns = [
        f"top5_present__{role}"
        for role in TOP5
    ]

    compact_small[
        "top5_present_count"
    ] = compact_small[
        top5_presence_columns
    ].sum(
        axis=1
    )

    compact_small[
        "top5_present"
    ] = (
        compact_small[
            "top5_present_count"
        ] > 0
    ).astype(int)

    compact_small = compact_small.drop(
        columns=compact_columns
    )

    # ========================================================
    # 3. 150维具体语义关系
    # ========================================================

    semantic = pd.read_csv(
        SEMANTIC_PATH,
        compression="gzip",
    )

    semantic["sha256"] = normalize_sha(
        semantic["sha256"]
    )

    names150 = (
        frozen_base.support150_names(
            semantic.columns
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "150维数量异常: "
            + str(len(names150))
        )

    semantic_values = (
        semantic[names150]
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
        semantic_values
    ).all():
        raise SystemExit(
            "150维中存在非有限值"
        )

    semantic_small = semantic[
        ["sha256"]
    ].copy()

    semantic_small[
        "relation_nonzero_count"
    ] = np.count_nonzero(
        np.abs(
            semantic_values
        ) > 1e-12,
        axis=1,
    )

    semantic_small[
        "relation_support_sum"
    ] = np.sum(
        semantic_values,
        axis=1,
    )

    semantic_small[
        "relation_support_mean_nonzero"
    ] = np.divide(
        np.sum(
            semantic_values,
            axis=1,
        ),
        np.maximum(
            semantic_small[
                "relation_nonzero_count"
            ].to_numpy(),
            1,
        ),
    )

    semantic_small[
        "presence150_raw"
    ] = (
        semantic_small[
            "relation_nonzero_count"
        ] > 0
    ).astype(int)

    # --------------------------------------------------------
    # final150定义：
    #
    #   5个固定锚点
    #   ×
    #   每个锚点对应另外30个语义角色
    #   =
    #   150维
    #
    # 重要：
    # 不假设names150按照TOP5顺序排列。
    #
    # 直接从列名：
    #
    # semrel_support__<anchor>__<context>
    #
    # 中解析anchor。
    # --------------------------------------------------------

    RELATIONS_PER_ANCHOR = 30

    anchor_active_columns = []
    anchor_relation_count_columns = []

    anchor_to_relation_names = {}

    for role in TOP5:
        prefix = (
            f"semrel_support__{role}__"
        )

        block_names = [
            name
            for name in names150
            if name.startswith(prefix)
        ]

        anchor_to_relation_names[
            role
        ] = block_names

        if (
            len(block_names)
            != RELATIONS_PER_ANCHOR
        ):
            print()
            print(
                "150维锚点列数量异常"
            )
            print(
                "锚点：",
                role,
            )
            print(
                "找到列数：",
                len(block_names),
            )

            for name in block_names[:10]:
                print(
                    " ",
                    name,
                )

            raise SystemExit(
                "每个Top-5锚点应恰好对应30个关系"
            )

    # --------------------------------------------------------
    # 再检查：
    # 5组30列加起来必须刚好覆盖全部150列，
    # 不能有遗漏，也不能重复。
    # --------------------------------------------------------

    grouped_names = []

    for role in TOP5:
        grouped_names.extend(
            anchor_to_relation_names[
                role
            ]
        )

    if len(grouped_names) != 150:
        raise SystemExit(
            "按锚点分组后的关系总数不是150: "
            + str(
                len(grouped_names)
            )
        )

    if len(set(grouped_names)) != 150:
        raise SystemExit(
            "按锚点分组后存在重复关系列"
        )

    ungrouped = (
        set(names150)
        - set(grouped_names)
    )

    if ungrouped:
        print(
            "未能归入Top-5锚点的关系列："
        )

        for name in sorted(
            ungrouped
        )[:20]:
            print(
                " ",
                name,
            )

        raise SystemExit(
            "150维关系分组不完整"
        )

    print()
    print(
        "150维按锚点解析成功："
    )

    for role in TOP5:
        print(
            " ",
            role,
            "=",
            len(
                anchor_to_relation_names[
                    role
                ]
            ),
            "维",
        )

    # --------------------------------------------------------
    # 按真实列名计算每个锚点：
    #
    # 1. 是否至少有一个非零跨语义关系
    # 2. 有多少个非零跨语义关系
    #
    # 不再依赖列的位置。
    # --------------------------------------------------------

    for role in TOP5:
        block_names = (
            anchor_to_relation_names[
                role
            ]
        )

        block = (
            semantic[
                block_names
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
            block
        ).all():
            raise SystemExit(
                "锚点关系块存在非有限值: "
                + role
            )

        active_col = (
            f"semantic_anchor_active__{role}"
        )

        count_col = (
            f"semantic_relation_count__{role}"
        )

        semantic_small[
            active_col
        ] = (
            np.any(
                np.abs(block)
                > 1e-12,
                axis=1,
            )
        ).astype(int)

        semantic_small[
            count_col
        ] = np.count_nonzero(
            np.abs(block)
            > 1e-12,
            axis=1,
        )

        anchor_active_columns.append(
            active_col
        )

        anchor_relation_count_columns.append(
            count_col
        )

    semantic_small[
        "active_semantic_anchor_count"
    ] = semantic_small[
        anchor_active_columns
    ].sum(
        axis=1
    )

    # ========================================================
    # 4. 数量10原始组织信息
    # ========================================================

    count10 = pd.read_csv(
        COUNT10_PATH,
        compression="gzip",
    )

    count10["sha256"] = normalize_sha(
        count10["sha256"]
    )

    count10_names = [
        f"up_role__{role}__{metric}"
        for role in TOP5
        for metric in COUNT_METRICS
    ]

    missing = (
        set(count10_names)
        - set(count10.columns)
    )

    if missing:
        raise SystemExit(
            "数量10文件缺少列: "
            + ", ".join(
                sorted(missing)
            )
        )

    count10_values = (
        count10[
            count10_names
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

    count10_small = count10[
        ["sha256"]
    ].copy()

    count10_small[
        "count10_nonzero_count"
    ] = np.count_nonzero(
        np.abs(
            count10_values
        ) > 1e-12,
        axis=1,
    )

    structure_anchor_cols = []

    for i, role in enumerate(
        TOP5
    ):
        block = count10_values[
            :,
            i * 2:
            i * 2 + 2,
        ]

        col = (
            f"structure_anchor_active__{role}"
        )

        count10_small[
            col
        ] = (
            np.any(
                np.abs(block)
                > 1e-12,
                axis=1,
            )
        ).astype(int)

        structure_anchor_cols.append(
            col
        )

    count10_small[
        "active_structure_anchor_count"
    ] = count10_small[
        structure_anchor_cols
    ].sum(
        axis=1
    )

    # ========================================================
    # 5. 合并
    # ========================================================

    data = (
        sample
        .merge(
            compact_small,
            on="sha256",
            how="left",
            validate="many_to_one",
        )
        .merge(
            semantic_small,
            on="sha256",
            how="left",
            validate="many_to_one",
        )
        .merge(
            count10_small,
            on="sha256",
            how="left",
            validate="many_to_one",
        )
    )

    critical_columns = [
        "top5_present",
        "relation_nonzero_count",
        "active_semantic_anchor_count",
        "active_structure_anchor_count",
    ]

    for col in critical_columns:
        if data[col].isna().any():
            raise SystemExit(
                f"合并后 {col} 存在缺失"
            )

    # 150存在状态必须与前一个诊断完全一致。
    mismatch = (
        data[
            "presence150"
        ].astype(int)
        !=
        data[
            "presence150_raw"
        ].astype(int)
    )

    if mismatch.any():
        raise SystemExit(
            "presence150与原始150维重新计算结果不一致，"
            f"数量={int(mismatch.sum())}"
        )

    # ========================================================
    # 6. 预测错误类型
    # ========================================================

    data[
        "base_wrong"
    ] = (
        ~data[
            "base_correct"
        ].astype(bool)
    )

    data[
        "base_FP"
    ] = (
        data[
            "base_wrong"
        ]
        &
        (
            data[
                "label"
            ] == 0
        )
    )

    data[
        "base_FN"
    ] = (
        data[
            "base_wrong"
        ]
        &
        (
            data[
                "label"
            ] == 1
        )
    )

    data[
        "margin_group"
    ] = data[
        "base_margin"
    ].map(
        margin_group
    )

    # ========================================================
    # 7. 将Compact错误拆成四类
    #
    # A：没有150语义证据
    # B：有语义证据，但Compact高置信度错误
    # C：有语义证据、margin<0.2，但仍未纠正
    # D：成功纠正
    #
    # 这是诊断分类，不是新的模型规则。
    # ========================================================

    def failure_type(row):
        if not bool(
            row["base_wrong"]
        ):
            return "base_correct"

        if int(
            row["presence150"]
        ) == 0:
            return (
                "A_no_semantic_relation"
            )

        if bool(
            row["count10_corrected"]
        ):
            return (
                "D_corrected"
            )

        if float(
            row["base_margin"]
        ) >= 0.20:
            return (
                "B_semantic_but_confident_wrong"
            )

        return (
            "C_semantic_uncertain_but_uncorrected"
        )

    data[
        "failure_type"
    ] = data.apply(
        failure_type,
        axis=1,
    )

    # ========================================================
    # 8. 每年总表
    # ========================================================

    year_rows = []

    for year in YEARS:
        g = data[
            data[
                "year"
            ] == year
        ].copy()

        if len(g) == 0:
            raise SystemExit(
                f"{year}没有样本"
            )

        errors = g[
            g[
                "base_wrong"
            ]
        ]

        p150_zero_errors = errors[
            errors[
                "presence150"
            ] == 0
        ]

        p150_one_errors = errors[
            errors[
                "presence150"
            ] == 1
        ]

        relation_positive = g[
            g[
                "presence150"
            ] == 1
        ]

        row = {
            "year":
                year,

            "N":
                len(g),

            "benign":
                int(
                    (
                        g["label"]
                        == 0
                    ).sum()
                ),

            "malware":
                int(
                    (
                        g["label"]
                        == 1
                    ).sum()
                ),

            "base_wrong":
                len(errors),

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

            # Top-5
            "top5_present_N":
                int(
                    g[
                        "top5_present"
                    ].sum()
                ),

            "top5_present_rate":
                float(
                    g[
                        "top5_present"
                    ].mean()
                ),

            "mean_top5_present_count":
                float(
                    g[
                        "top5_present_count"
                    ].mean()
                ),

            # 150
            "presence150_N":
                int(
                    g[
                        "presence150"
                    ].sum()
                ),

            "presence150_rate":
                float(
                    g[
                        "presence150"
                    ].mean()
                ),

            "no_top5_N":
                int(
                    (
                        g[
                            "top5_present"
                        ] == 0
                    ).sum()
                ),

            "top5_but_150zero_N":
                int(
                    (
                        (
                            g[
                                "top5_present"
                            ] == 1
                        )
                        &
                        (
                            g[
                                "presence150"
                            ] == 0
                        )
                    ).sum()
                ),

            # 错误落点
            "base_error_presence150_zero":
                len(
                    p150_zero_errors
                ),

            "base_error_presence150_one":
                len(
                    p150_one_errors
                ),

            "base_error_presence150_zero_rate":
                safe_rate(
                    len(
                        p150_zero_errors
                    ),
                    len(errors),
                ),

            # 关系丰富程度
            "mean_nonzero_relations_all":
                float(
                    g[
                        "relation_nonzero_count"
                    ].mean()
                ),

            "median_nonzero_relations_all":
                float(
                    g[
                        "relation_nonzero_count"
                    ].median()
                ),

            "mean_nonzero_relations_when_present":
                (
                    float(
                        relation_positive[
                            "relation_nonzero_count"
                        ].mean()
                    )
                    if len(
                        relation_positive
                    )
                    else np.nan
                ),

            "median_nonzero_relations_when_present":
                (
                    float(
                        relation_positive[
                            "relation_nonzero_count"
                        ].median()
                    )
                    if len(
                        relation_positive
                    )
                    else np.nan
                ),

            "mean_active_semantic_anchors":
                float(
                    g[
                        "active_semantic_anchor_count"
                    ].mean()
                ),

            "mean_active_structure_anchors":
                float(
                    g[
                        "active_structure_anchor_count"
                    ].mean()
                ),

            # Compact错误置信度
            "error_margin_lt_010":
                int(
                    (
                        errors[
                            "base_margin"
                        ] < 0.10
                    ).sum()
                ),

            "error_margin_010_020":
                int(
                    (
                        (
                            errors[
                                "base_margin"
                            ] >= 0.10
                        )
                        &
                        (
                            errors[
                                "base_margin"
                            ] < 0.20
                        )
                    ).sum()
                ),

            "error_margin_ge_020":
                int(
                    (
                        errors[
                            "base_margin"
                        ] >= 0.20
                    ).sum()
                ),

            "error_margin_ge_020_rate":
                safe_rate(
                    int(
                        (
                            errors[
                                "base_margin"
                            ] >= 0.20
                        ).sum()
                    ),
                    len(errors),
                ),

            "mean_error_margin":
                (
                    float(
                        errors[
                            "base_margin"
                        ].mean()
                    )
                    if len(errors)
                    else np.nan
                ),

            # 最终纠错
            "corrected":
                int(
                    g[
                        "count10_corrected"
                    ].sum()
                ),

            "regressed":
                int(
                    g[
                        "count10_regressed"
                    ].sum()
                ),

            "net_correction":
                int(
                    g[
                        "count10_corrected"
                    ].sum()
                    -
                    g[
                        "count10_regressed"
                    ].sum()
                ),

            "count10_extra_better_vs_150":
                int(
                    g[
                        "count10_better_than_150"
                    ].sum()
                ),

            "count10_extra_worse_vs_150":
                int(
                    g[
                        "count10_worse_than_150"
                    ].sum()
                ),
        }

        year_rows.append(
            row
        )

    year_summary = pd.DataFrame(
        year_rows
    )

    # ========================================================
    # 9. 错误margin × 年份
    # ========================================================

    margin_order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    margin_rows = []

    for year in YEARS:
        errors = data[
            (
                data[
                    "year"
                ] == year
            )
            &
            (
                data[
                    "base_wrong"
                ]
            )
        ]

        for bucket in margin_order:
            g = errors[
                errors[
                    "margin_group"
                ] == bucket
            ]

            corrected = int(
                g[
                    "count10_corrected"
                ].sum()
            )

            margin_rows.append({
                "year":
                    year,

                "margin":
                    bucket,

                "errors":
                    len(g),

                "corrected":
                    corrected,

                "correction_rate":
                    safe_rate(
                        corrected,
                        len(g),
                    ),

                "presence150_errors":
                    int(
                        g[
                            "presence150"
                        ].sum()
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
            })

    margin_summary = pd.DataFrame(
        margin_rows
    )

    # ========================================================
    # 10. 四类失效原因 × 年份
    # ========================================================

    failure_order = (
        "A_no_semantic_relation",
        "B_semantic_but_confident_wrong",
        "C_semantic_uncertain_but_uncorrected",
        "D_corrected",
    )

    failure_rows = []

    for year in YEARS:
        errors = data[
            (
                data[
                    "year"
                ] == year
            )
            &
            (
                data[
                    "base_wrong"
                ]
            )
        ]

        for category in (
            failure_order
        ):
            g = errors[
                errors[
                    "failure_type"
                ] == category
            ]

            failure_rows.append({
                "year":
                    year,

                "category":
                    category,

                "N":
                    len(g),

                "share_of_base_errors":
                    safe_rate(
                        len(g),
                        len(errors),
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

                "mean_relation_count":
                    (
                        float(
                            g[
                                "relation_nonzero_count"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),
            })

    failure_summary = pd.DataFrame(
        failure_rows
    )

    # ========================================================
    # 11. 每个Top-5锚点 × 年份
    # ========================================================

    anchor_rows = []

    for year in YEARS:
        yg = data[
            data[
                "year"
            ] == year
        ]

        for role in TOP5:
            top_col = (
                f"top5_present__{role}"
            )

            semantic_active_col = (
                f"semantic_anchor_active__{role}"
            )

            semantic_count_col = (
                f"semantic_relation_count__{role}"
            )

            structure_active_col = (
                f"structure_anchor_active__{role}"
            )

            top_present = yg[
                yg[
                    top_col
                ] == 1
            ]

            semantic_active = yg[
                yg[
                    semantic_active_col
                ] == 1
            ]

            base_errors = yg[
                yg[
                    "base_wrong"
                ]
            ]

            error_and_top = base_errors[
                base_errors[
                    top_col
                ] == 1
            ]

            error_and_semantic = (
                base_errors[
                    base_errors[
                        semantic_active_col
                    ] == 1
                ]
            )

            anchor_rows.append({
                "year":
                    year,

                "anchor":
                    role,

                "top5_present_N":
                    len(
                        top_present
                    ),

                "top5_present_rate":
                    safe_rate(
                        len(
                            top_present
                        ),
                        len(yg),
                    ),

                "structure_active_N":
                    int(
                        yg[
                            structure_active_col
                        ].sum()
                    ),

                "semantic_anchor_active_N":
                    len(
                        semantic_active
                    ),

                "semantic_anchor_active_rate":
                    safe_rate(
                        len(
                            semantic_active
                        ),
                        len(yg),
                    ),

                "mean_relation_count_for_anchor":
                    float(
                        yg[
                            semantic_count_col
                        ].mean()
                    ),

                "base_errors_with_top5_anchor":
                    len(
                        error_and_top
                    ),

                "base_errors_with_semantic_anchor":
                    len(
                        error_and_semantic
                    ),

                "corrected_errors_with_semantic_anchor":
                    int(
                        error_and_semantic[
                            "count10_corrected"
                        ].sum()
                    ),
            })

    anchor_summary = pd.DataFrame(
        anchor_rows
    )

    # ========================================================
    # 12. 每个150关系在每年的频率
    # ========================================================

    relation_rows = []

    # semantic中的SHA与年份通过sample映射
    sha_year = data[
        [
            "sha256",
            "year",
        ]
    ].drop_duplicates()

    semantic_for_year = (
        semantic[
            [
                "sha256",
                *names150,
            ]
        ]
        .merge(
            sha_year,
            on="sha256",
            how="inner",
            validate="one_to_one",
        )
    )

    for year in YEARS:
        yg = semantic_for_year[
            semantic_for_year[
                "year"
            ] == year
        ]

        for name in names150:
            values = (
                pd.to_numeric(
                    yg[name],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(
                    dtype=float
                )
            )

            active = (
                np.abs(values)
                > 1e-12
            )

            relation_rows.append({
                "year":
                    year,

                "relation":
                    name,

                "active_N":
                    int(
                        active.sum()
                    ),

                "active_rate":
                    float(
                        active.mean()
                    ),

                "mean_support_all":
                    float(
                        values.mean()
                    ),

                "mean_support_when_active":
                    (
                        float(
                            values[
                                active
                            ].mean()
                        )
                        if active.any()
                        else 0.0
                    ),
            })

    relation_frequency = (
        pd.DataFrame(
            relation_rows
        )
    )

    top_relations = (
        relation_frequency
        .sort_values(
            [
                "year",
                "active_rate",
                "mean_support_all",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .groupby(
            "year",
            as_index=False,
        )
        .head(15)
    )

    # 2018 -> 2021频率变化
    relation_pivot = (
        relation_frequency
        .pivot(
            index="relation",
            columns="year",
            values="active_rate",
        )
        .reset_index()
    )

    if (
        2018 in relation_pivot.columns
        and
        2021 in relation_pivot.columns
    ):
        relation_pivot[
            "delta_2021_minus_2018"
        ] = (
            relation_pivot[
                2021
            ]
            -
            relation_pivot[
                2018
            ]
        )

        relation_pivot[
            "abs_delta"
        ] = np.abs(
            relation_pivot[
                "delta_2021_minus_2018"
            ]
        )

        relation_pivot = (
            relation_pivot
            .sort_values(
                "abs_delta",
                ascending=False,
            )
        )

    # ========================================================
    # 13. 保存
    # ========================================================

    year_summary.to_csv(
        OUTPUT_DIR
        / "year_summary.csv",
        index=False,
    )

    margin_summary.to_csv(
        OUTPUT_DIR
        / "year_error_margin.csv",
        index=False,
    )

    failure_summary.to_csv(
        OUTPUT_DIR
        / "year_failure_breakdown.csv",
        index=False,
    )

    anchor_summary.to_csv(
        OUTPUT_DIR
        / "anchor_year_summary.csv",
        index=False,
    )

    relation_frequency.to_csv(
        OUTPUT_DIR
        / "relation_frequency_by_year.csv",
        index=False,
    )

    top_relations.to_csv(
        OUTPUT_DIR
        / "top_relations_by_year.csv",
        index=False,
    )

    relation_pivot.to_csv(
        OUTPUT_DIR
        / "relation_change_2018_2021.csv",
        index=False,
    )

    data.to_csv(
        OUTPUT_DIR
        / "sample_internal_diagnostic.csv",
        index=False,
    )

    # ========================================================
    # 14. 打印核心结果
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "每年总体内部情况"
    )
    print(
        "======================================"
    )

    for _, row in (
        year_summary.iterrows()
    ):
        print()
        print(
            int(
                row["year"]
            )
        )

        print(
            " N=",
            int(
                row["N"]
            ),
            " 良性=",
            int(
                row["benign"]
            ),
            " 恶意=",
            int(
                row["malware"]
            ),
            sep="",
        )

        print(
            " Compact错误=",
            int(
                row[
                    "base_wrong"
                ]
            ),
            " FP=",
            int(
                row["FP"]
            ),
            " FN=",
            int(
                row["FN"]
            ),
            sep="",
        )

        print(
            " Top5存在=",
            int(
                row[
                    "top5_present_N"
                ]
            ),
            "/",
            int(
                row["N"]
            ),
            " (",
            pct_text(
                row[
                    "top5_present_rate"
                ]
            ),
            ")",
            sep="",
        )

        print(
            " 150存在=",
            int(
                row[
                    "presence150_N"
                ]
            ),
            "/",
            int(
                row["N"]
            ),
            " (",
            pct_text(
                row[
                    "presence150_rate"
                ]
            ),
            ")",
            sep="",
        )

        print(
            " 无Top5=",
            int(
                row[
                    "no_top5_N"
                ]
            ),
            " | 有Top5但150=0=",
            int(
                row[
                    "top5_but_150zero_N"
                ]
            ),
            sep="",
        )

        print(
            " Compact错误中150=0: ",
            int(
                row[
                    "base_error_presence150_zero"
                ]
            ),
            " | 150=1: ",
            int(
                row[
                    "base_error_presence150_one"
                ]
            ),
            sep="",
        )

        print(
            " 非零关系数 mean=",
            f"{row['mean_nonzero_relations_all']:.3f}",
            " median=",
            f"{row['median_nonzero_relations_all']:.1f}",
            " | 在150存在样本中 mean=",
            f"{row['mean_nonzero_relations_when_present']:.3f}",
            sep="",
        )

        print(
            " 错误margin: <.10=",
            int(
                row[
                    "error_margin_lt_010"
                ]
            ),
            " .10-.20=",
            int(
                row[
                    "error_margin_010_020"
                ]
            ),
            " >=.20=",
            int(
                row[
                    "error_margin_ge_020"
                ]
            ),
            " (",
            pct_text(
                row[
                    "error_margin_ge_020_rate"
                ]
            ),
            ")",
            sep="",
        )

        print(
            " corrected=",
            int(
                row[
                    "corrected"
                ]
            ),
            " regressed=",
            int(
                row[
                    "regressed"
                ]
            ),
            " net=",
            f"{int(row['net_correction']):+d}",
            sep="",
        )

    print()
    print(
        "======================================"
    )
    print(
        "每年Compact错误的四类去向"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            year
        )

        yg = failure_summary[
            failure_summary[
                "year"
            ] == year
        ]

        for category in (
            failure_order
        ):
            row = yg[
                yg[
                    "category"
                ] == category
            ].iloc[0]

            print(
                f"{category:38s} "
                f"N={int(row['N']):3d} "
                f"share={pct_text(row['share_of_base_errors']):>7s} "
                f"FP={int(row['FP']):3d} "
                f"FN={int(row['FN']):3d} "
                f"margin={row['mean_margin']:.4f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "错误margin按年份"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            year
        )

        yg = margin_summary[
            margin_summary[
                "year"
            ] == year
        ]

        for bucket in (
            margin_order
        ):
            row = yg[
                yg[
                    "margin"
                ] == bucket
            ].iloc[0]

            print(
                f"{bucket:10s} "
                f"errors={int(row['errors']):3d} "
                f"corrected={int(row['corrected']):3d} "
                f"rate={pct_text(row['correction_rate']):>7s} "
                f"presence150={int(row['presence150_errors']):3d}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "Top-5锚点按年份"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            year
        )

        yg = anchor_summary[
            anchor_summary[
                "year"
            ] == year
        ]

        for _, row in (
            yg.iterrows()
        ):
            print(
                f"{row['anchor']:32s} "
                f"Top5={int(row['top5_present_N']):3d} "
                f"语义关系锚点={int(row['semantic_anchor_active_N']):3d} "
                f"错误命中={int(row['base_errors_with_semantic_anchor']):3d} "
                f"纠正={int(row['corrected_errors_with_semantic_anchor']):3d}"
            )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
