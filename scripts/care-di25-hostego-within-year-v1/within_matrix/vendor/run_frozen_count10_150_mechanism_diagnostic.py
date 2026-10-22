#!/usr/bin/env python3
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_frozen_anchor_residual_2017_2021 as frozen_base


COUNT10_DIR = Path(
    "artifacts/frozen-anchor-residual-count10-150-2017-2021"
)

ONLY150_DIR = Path(
    "artifacts/frozen-anchor-residual-2017-2021"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

OUTPUT_DIR = Path(
    "artifacts/frozen-count10-150-mechanism-diagnostic"
)

COUNT10_PRED = (
    COUNT10_DIR
    / "future_predictions.csv"
)

ONLY150_PRED = (
    ONLY150_DIR
    / "future_predictions.csv"
)

THRESHOLD = 0.5


def normalize_sha(value):
    return str(value).strip().casefold()


def load_predictions(path, model_name):
    if not path.exists():
        raise SystemExit(
            f"找不到预测文件: {path}\n"
            "请先执行:\n"
            f"ls -lh {path.parent}"
        )

    df = pd.read_csv(path)

    required = {
        "sha256",
        "seed",
        "label",
        "probability",
        "base_probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"{path} 缺少列: "
            + ", ".join(sorted(missing))
            + "\n实际列:\n"
            + ", ".join(df.columns)
        )

    df = df.copy()

    df["sha256"] = (
        df["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # 如果没有year列，就从split_name提取年份。
    # --------------------------------------------------------

    if "year" not in df.columns:
        if "split_name" not in df.columns:
            raise SystemExit(
                f"{path} 同时没有 year 和 split_name"
            )

        years = []

        for value in df["split_name"]:
            match = re.search(
                r"(2018|2019|2020|2021)",
                str(value),
            )

            if not match:
                raise SystemExit(
                    "无法从split_name提取年份: "
                    + str(value)
                )

            years.append(
                int(match.group(1))
            )

        df["year"] = years

    df["year"] = (
        df["year"]
        .astype(int)
    )

    df["seed"] = (
        df["seed"]
        .astype(int)
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    df["probability"] = (
        df["probability"]
        .astype(float)
    )

    df["base_probability"] = (
        df["base_probability"]
        .astype(float)
    )

    df["model"] = model_name

    key = [
        "year",
        "sha256",
        "seed",
    ]

    duplicated = df.duplicated(
        key,
        keep=False,
    )

    if duplicated.any():
        raise SystemExit(
            f"{path} 存在重复 year+sha+seed，数量="
            f"{int(duplicated.sum())}"
        )

    return df


def load_semantic_presence():
    if not SEMANTIC_PATH.exists():
        raise SystemExit(
            f"找不到150维语义文件: {SEMANTIC_PATH}"
        )

    semantic = pd.read_csv(
        SEMANTIC_PATH,
        compression="gzip",
    )

    if "sha256" not in semantic.columns:
        raise SystemExit(
            "未来语义文件缺少sha256"
        )

    semantic["sha256"] = (
        semantic["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    names150 = (
        frozen_base.support150_names(
            semantic.columns
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "150维列数量异常: "
            + str(len(names150))
        )

    values = (
        semantic[names150]
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(values).all():
        raise SystemExit(
            "150维存在非有限值"
        )

    semantic = semantic[
        ["sha256"]
    ].copy()

    semantic["presence150"] = (
        np.any(
            np.abs(values) > 1e-12,
            axis=1,
        )
        .astype(int)
    )

    if semantic["sha256"].duplicated().any():
        raise SystemExit(
            "未来150维文件存在重复SHA"
        )

    return semantic, names150


def average_seeds(df, prefix):
    """
    逐APK对5个随机种子的概率取平均，
    然后固定0.5阈值做最终诊断。
    """

    grouped = (
        df.groupby(
            [
                "year",
                "sha256",
            ],
            as_index=False,
        )
        .agg(
            label=("label", "first"),
            seed_n=("seed", "nunique"),
            probability=("probability", "mean"),
            base_probability=(
                "base_probability",
                "mean",
            ),
        )
    )

    bad_seed_n = (
        grouped["seed_n"] != 5
    )

    if bad_seed_n.any():
        print(
            "警告：部分样本随机种子数量不是5。"
        )

        print(
            grouped.loc[
                bad_seed_n,
                [
                    "year",
                    "sha256",
                    "seed_n",
                ],
            ].head(20)
        )

    grouped = grouped.rename(
        columns={
            "probability":
                f"{prefix}_probability",

            "base_probability":
                f"{prefix}_base_probability",
        }
    )

    return grouped


def margin_bin(value):
    if value < 0.05:
        return "0-0.05"

    if value < 0.10:
        return "0.05-0.10"

    if value < 0.20:
        return "0.10-0.20"

    if value < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def safe_rate(a, b):
    if b == 0:
        return np.nan

    return a / b


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "======================================"
    )
    print(
        "Frozen + 数量10 + 150 最终机制诊断"
    )
    print(
        "======================================"
    )
    print(
        "本脚本不重新训练模型。"
    )
    print(
        "阈值 =",
        THRESHOLD,
    )

    count_seed = load_predictions(
        COUNT10_PRED,
        "count10_150",
    )

    only150_seed = load_predictions(
        ONLY150_PRED,
        "only150",
    )

    semantic_presence, names150 = (
        load_semantic_presence()
    )

    print(
        "数量10+150逐种子预测 =",
        len(count_seed),
    )

    print(
        "仅150逐种子预测 =",
        len(only150_seed),
    )

    print(
        "150维列数 =",
        len(names150),
    )

    # ========================================================
    # 一、两个残差模型使用的固定Compact必须完全相同
    # ========================================================

    key_seed = [
        "year",
        "sha256",
        "seed",
    ]

    base_compare = count_seed.merge(
        only150_seed[
            key_seed
            + [
                "label",
                "base_probability",
            ]
        ],
        on=key_seed,
        how="inner",
        suffixes=(
            "_count10",
            "_150",
        ),
        validate="one_to_one",
    )

    if len(base_compare) != len(count_seed):
        raise SystemExit(
            "数量10+150与仅150逐种子人口不一致"
        )

    label_mismatch = (
        base_compare[
            "label_count10"
        ]
        !=
        base_compare[
            "label_150"
        ]
    )

    if label_mismatch.any():
        raise SystemExit(
            "两个实验标签不一致"
        )

    base_diff = np.abs(
        base_compare[
            "base_probability_count10"
        ].to_numpy()
        -
        base_compare[
            "base_probability_150"
        ].to_numpy()
    )

    print()
    print(
        "===== 固定Compact一致性 ====="
    )

    print(
        "两个实验base_probability最大差 =",
        f"{base_diff.max():.12g}",
    )

    print(
        "平均差 =",
        f"{base_diff.mean():.12g}",
    )

    if base_diff.max() > 1e-8:
        raise SystemExit(
            "两个实验的固定Compact不一致，"
            "停止后续归因。"
        )

    # ========================================================
    # 二、检查150全零时是否严格退回Compact
    # ========================================================

    fallback = count_seed.merge(
        semantic_presence,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    if fallback["presence150"].isna().any():
        missing = fallback.loc[
            fallback["presence150"].isna(),
            "sha256",
        ].unique()

        raise SystemExit(
            "部分预测样本没有150维记录，数量="
            + str(len(missing))
        )

    zero_seed = fallback[
        fallback[
            "presence150"
        ] == 0
    ].copy()

    zero_seed[
        "abs_probability_diff"
    ] = np.abs(
        zero_seed[
            "probability"
        ]
        -
        zero_seed[
            "base_probability"
        ]
    )

    print()
    print(
        "===== 150维全零时的精确退回检查 ====="
    )

    print(
        "逐种子记录数 =",
        len(zero_seed),
    )

    print(
        "唯一APK数 =",
        zero_seed[
            [
                "year",
                "sha256",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "最终概率与Compact概率最大差 =",
        f"{zero_seed['abs_probability_diff'].max():.12g}",
    )

    print(
        "平均差 =",
        f"{zero_seed['abs_probability_diff'].mean():.12g}",
    )

    # ========================================================
    # 三、对5个种子平均后做主诊断
    # ========================================================

    count = average_seeds(
        count_seed,
        "count10",
    )

    only150 = average_seeds(
        only150_seed,
        "only150",
    )

    sample = count.merge(
        only150[
            [
                "year",
                "sha256",
                "label",
                "only150_probability",
                "only150_base_probability",
            ]
        ],
        on=[
            "year",
            "sha256",
        ],
        how="inner",
        suffixes=(
            "",
            "_150",
        ),
        validate="one_to_one",
    )

    if len(sample) != len(count):
        raise SystemExit(
            "两个模型平均种子后人口不一致"
        )

    if (
        sample["label"]
        != sample[
            "label_150"
        ]
    ).any():
        raise SystemExit(
            "平均种子后标签不一致"
        )

    sample = sample.drop(
        columns=[
            "label_150",
        ]
    )

    sample = sample.merge(
        semantic_presence,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    if sample[
        "presence150"
    ].isna().any():
        raise SystemExit(
            "平均种子样本存在缺失的150维状态"
        )

    # 两个实验平均后的base也必须一致。
    sample[
        "base_model_diff"
    ] = np.abs(
        sample[
            "count10_base_probability"
        ]
        -
        sample[
            "only150_base_probability"
        ]
    )

    if (
        sample[
            "base_model_diff"
        ].max()
        > 1e-8
    ):
        raise SystemExit(
            "平均后的固定Compact不一致"
        )

    sample[
        "base_probability"
    ] = sample[
        "count10_base_probability"
    ]

    sample[
        "base_prediction"
    ] = (
        sample[
            "base_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    sample[
        "count10_prediction"
    ] = (
        sample[
            "count10_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    sample[
        "only150_prediction"
    ] = (
        sample[
            "only150_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    sample[
        "base_correct"
    ] = (
        sample[
            "base_prediction"
        ]
        == sample[
            "label"
        ]
    )

    sample[
        "count10_correct"
    ] = (
        sample[
            "count10_prediction"
        ]
        == sample[
            "label"
        ]
    )

    sample[
        "only150_correct"
    ] = (
        sample[
            "only150_prediction"
        ]
        == sample[
            "label"
        ]
    )

    sample[
        "base_margin"
    ] = np.abs(
        sample[
            "base_probability"
        ]
        - THRESHOLD
    )

    sample[
        "margin_bin"
    ] = sample[
        "base_margin"
    ].map(
        margin_bin
    )

    sample[
        "base_uncertainty"
    ] = (
        4.0
        * sample[
            "base_probability"
        ]
        * (
            1.0
            - sample[
                "base_probability"
            ]
        )
    )

    # 正值表示概率向真实类别方向移动。
    direction = (
        2.0
        * sample[
            "label"
        ]
        - 1.0
    )

    sample[
        "count10_truth_shift"
    ] = (
        direction
        * (
            sample[
                "count10_probability"
            ]
            -
            sample[
                "base_probability"
            ]
        )
    )

    sample[
        "only150_truth_shift"
    ] = (
        direction
        * (
            sample[
                "only150_probability"
            ]
            -
            sample[
                "base_probability"
            ]
        )
    )

    sample[
        "count10_corrected"
    ] = (
        (~sample[
            "base_correct"
        ])
        &
        sample[
            "count10_correct"
        ]
    )

    sample[
        "count10_regressed"
    ] = (
        sample[
            "base_correct"
        ]
        &
        (~sample[
            "count10_correct"
        ])
    )

    sample[
        "only150_corrected"
    ] = (
        (~sample[
            "base_correct"
        ])
        &
        sample[
            "only150_correct"
        ]
    )

    sample[
        "only150_regressed"
    ] = (
        sample[
            "base_correct"
        ]
        &
        (~sample[
            "only150_correct"
        ])
    )

    # ========================================================
    # 四、每年纠正/退化
    # ========================================================

    year_rows = []

    for year in sorted(
        sample[
            "year"
        ].unique()
    ):
        g = sample[
            sample[
                "year"
            ] == year
        ]

        base_wrong = (
            ~g[
                "base_correct"
            ]
        )

        corrected = g[
            "count10_corrected"
        ]

        regressed = g[
            "count10_regressed"
        ]

        corrected_fp = (
            corrected
            &
            (
                g[
                    "label"
                ] == 0
            )
        ).sum()

        corrected_fn = (
            corrected
            &
            (
                g[
                    "label"
                ] == 1
            )
        ).sum()

        regressed_to_fp = (
            regressed
            &
            (
                g[
                    "label"
                ] == 0
            )
        ).sum()

        regressed_to_fn = (
            regressed
            &
            (
                g[
                    "label"
                ] == 1
            )
        ).sum()

        row = {
            "year":
                int(year),

            "N":
                len(g),

            "base_wrong":
                int(
                    base_wrong.sum()
                ),

            "corrected":
                int(
                    corrected.sum()
                ),

            "regressed":
                int(
                    regressed.sum()
                ),

            "net":
                int(
                    corrected.sum()
                    - regressed.sum()
                ),

            "corrected_FP":
                int(
                    corrected_fp
                ),

            "corrected_FN":
                int(
                    corrected_fn
                ),

            "regressed_to_FP":
                int(
                    regressed_to_fp
                ),

            "regressed_to_FN":
                int(
                    regressed_to_fn
                ),

            "correction_rate_among_base_wrong":
                safe_rate(
                    int(
                        corrected.sum()
                    ),
                    int(
                        base_wrong.sum()
                    ),
                ),

            "mean_truth_shift":
                float(
                    g[
                        "count10_truth_shift"
                    ].mean()
                ),
        }

        year_rows.append(
            row
        )

    # 再加总计。
    base_wrong = (
        ~sample[
            "base_correct"
        ]
    )

    corrected = sample[
        "count10_corrected"
    ]

    regressed = sample[
        "count10_regressed"
    ]

    year_rows.append({
        "year":
            "ALL",

        "N":
            len(sample),

        "base_wrong":
            int(
                base_wrong.sum()
            ),

        "corrected":
            int(
                corrected.sum()
            ),

        "regressed":
            int(
                regressed.sum()
            ),

        "net":
            int(
                corrected.sum()
                - regressed.sum()
            ),

        "corrected_FP":
            int(
                (
                    corrected
                    &
                    (
                        sample[
                            "label"
                        ] == 0
                    )
                ).sum()
            ),

        "corrected_FN":
            int(
                (
                    corrected
                    &
                    (
                        sample[
                            "label"
                        ] == 1
                    )
                ).sum()
            ),

        "regressed_to_FP":
            int(
                (
                    regressed
                    &
                    (
                        sample[
                            "label"
                        ] == 0
                    )
                ).sum()
            ),

        "regressed_to_FN":
            int(
                (
                    regressed
                    &
                    (
                        sample[
                            "label"
                        ] == 1
                    )
                ).sum()
            ),

        "correction_rate_among_base_wrong":
            safe_rate(
                int(
                    corrected.sum()
                ),
                int(
                    base_wrong.sum()
                ),
            ),

        "mean_truth_shift":
            float(
                sample[
                    "count10_truth_shift"
                ].mean()
            ),
    })

    # ========================================================
    # 五、Compact错误的margin纠正率
    # ========================================================

    margin_order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    margin_rows = []

    base_errors = sample[
        ~sample[
            "base_correct"
        ]
    ].copy()

    for bin_name in margin_order:
        g = base_errors[
            base_errors[
                "margin_bin"
            ] == bin_name
        ]

        corrected = g[
            "count10_corrected"
        ]

        margin_rows.append({
            "margin_bin":
                bin_name,

            "base_errors":
                len(g),

            "corrected":
                int(
                    corrected.sum()
                ),

            "correction_rate":
                safe_rate(
                    int(
                        corrected.sum()
                    ),
                    len(g),
                ),

            "corrected_FP":
                int(
                    (
                        corrected
                        &
                        (
                            g[
                                "label"
                            ] == 0
                        )
                    ).sum()
                ),

            "corrected_FN":
                int(
                    (
                        corrected
                        &
                        (
                            g[
                                "label"
                            ] == 1
                        )
                    ).sum()
                ),

            "mean_truth_shift":
                (
                    float(
                        g[
                            "count10_truth_shift"
                        ].mean()
                    )
                    if len(g)
                    else np.nan
                ),

            "mean_uncertainty":
                (
                    float(
                        g[
                            "base_uncertainty"
                        ].mean()
                    )
                    if len(g)
                    else np.nan
                ),
        })

    # ========================================================
    # 六、150存在/不存在分别发生什么
    # ========================================================

    presence_rows = []

    for presence in (
        0,
        1,
    ):
        g = sample[
            sample[
                "presence150"
            ] == presence
        ]

        corrected = g[
            "count10_corrected"
        ]

        regressed = g[
            "count10_regressed"
        ]

        presence_rows.append({
            "presence150":
                presence,

            "N":
                len(g),

            "base_wrong":
                int(
                    (
                        ~g[
                            "base_correct"
                        ]
                    ).sum()
                ),

            "corrected":
                int(
                    corrected.sum()
                ),

            "regressed":
                int(
                    regressed.sum()
                ),

            "net":
                int(
                    corrected.sum()
                    - regressed.sum()
                ),

            "max_abs_final_base_probability_diff":
                float(
                    np.abs(
                        g[
                            "count10_probability"
                        ]
                        -
                        g[
                            "base_probability"
                        ]
                    ).max()
                )
                if len(g)
                else np.nan,
        })

    # ========================================================
    # 七、数量10相对仅150究竟改变哪些样本
    # ========================================================

    sample[
        "count10_better_than_150"
    ] = (
        sample[
            "count10_correct"
        ]
        &
        (~sample[
            "only150_correct"
        ])
    )

    sample[
        "count10_worse_than_150"
    ] = (
        (~sample[
            "count10_correct"
        ])
        &
        sample[
            "only150_correct"
        ]
    )

    compare_rows = []

    for year_value in list(
        sorted(
            sample[
                "year"
            ].unique()
        )
    ) + ["ALL"]:
        if year_value == "ALL":
            g = sample
        else:
            g = sample[
                sample[
                    "year"
                ] == year_value
            ]

        base_wrong = (
            ~g[
                "base_correct"
            ]
        )

        both_correct_base_error = (
            base_wrong
            &
            g[
                "count10_correct"
            ]
            &
            g[
                "only150_correct"
            ]
        )

        count_only_correct_base_error = (
            base_wrong
            &
            g[
                "count10_correct"
            ]
            &
            (~g[
                "only150_correct"
            ])
        )

        only150_only_correct_base_error = (
            base_wrong
            &
            (~g[
                "count10_correct"
            ])
            &
            g[
                "only150_correct"
            ]
        )

        neither_correct_base_error = (
            base_wrong
            &
            (~g[
                "count10_correct"
            ])
            &
            (~g[
                "only150_correct"
            ])
        )

        compare_rows.append({
            "year":
                year_value,

            "N":
                len(g),

            "count10_better_than_150":
                int(
                    g[
                        "count10_better_than_150"
                    ].sum()
                ),

            "count10_worse_than_150":
                int(
                    g[
                        "count10_worse_than_150"
                    ].sum()
                ),

            "direct_net_vs_150":
                int(
                    g[
                        "count10_better_than_150"
                    ].sum()
                    -
                    g[
                        "count10_worse_than_150"
                    ].sum()
                ),

            "base_error_both_correct":
                int(
                    both_correct_base_error.sum()
                ),

            "base_error_count10_only_correct":
                int(
                    count_only_correct_base_error.sum()
                ),

            "base_error_150_only_correct":
                int(
                    only150_only_correct_base_error.sum()
                ),

            "base_error_neither_correct":
                int(
                    neither_correct_base_error.sum()
                ),

            "count10_regressions_from_base":
                int(
                    g[
                        "count10_regressed"
                    ].sum()
                ),

            "only150_regressions_from_base":
                int(
                    g[
                        "only150_regressed"
                    ].sum()
                ),
        })

    # ========================================================
    # 八、保存“数量10独有纠正”和“数量10相对150变差”
    # ========================================================

    detail_columns = [
        "year",
        "sha256",
        "label",
        "presence150",
        "base_probability",
        "base_margin",
        "base_uncertainty",
        "only150_probability",
        "count10_probability",
        "only150_correct",
        "count10_correct",
        "count10_truth_shift",
        "only150_truth_shift",
    ]

    count10_only = sample[
        (
            ~sample[
                "base_correct"
            ]
        )
        &
        sample[
            "count10_correct"
        ]
        &
        (
            ~sample[
                "only150_correct"
            ]
        )
    ][detail_columns].copy()

    only150_only = sample[
        (
            ~sample[
                "base_correct"
            ]
        )
        &
        (
            ~sample[
                "count10_correct"
            ]
        )
        &
        sample[
            "only150_correct"
        ]
    ][detail_columns].copy()

    count10_better_direct = sample[
        sample[
            "count10_better_than_150"
        ]
    ][detail_columns].copy()

    count10_worse_direct = sample[
        sample[
            "count10_worse_than_150"
        ]
    ][detail_columns].copy()

    # ========================================================
    # 九、写结果
    # ========================================================

    pd.DataFrame(
        year_rows
    ).to_csv(
        OUTPUT_DIR
        / "year_correction_summary.csv",
        index=False,
    )

    pd.DataFrame(
        margin_rows
    ).to_csv(
        OUTPUT_DIR
        / "base_error_margin_summary.csv",
        index=False,
    )

    pd.DataFrame(
        presence_rows
    ).to_csv(
        OUTPUT_DIR
        / "presence150_summary.csv",
        index=False,
    )

    pd.DataFrame(
        compare_rows
    ).to_csv(
        OUTPUT_DIR
        / "count10_vs_only150_summary.csv",
        index=False,
    )

    sample.to_csv(
        OUTPUT_DIR
        / "sample_level.csv",
        index=False,
    )

    count10_only.to_csv(
        OUTPUT_DIR
        / "count10_only_base_error_corrections.csv",
        index=False,
    )

    only150_only.to_csv(
        OUTPUT_DIR
        / "only150_only_base_error_corrections.csv",
        index=False,
    )

    count10_better_direct.to_csv(
        OUTPUT_DIR
        / "count10_better_than_only150.csv",
        index=False,
    )

    count10_worse_direct.to_csv(
        OUTPUT_DIR
        / "count10_worse_than_only150.csv",
        index=False,
    )

    # ========================================================
    # 十、打印核心结果
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "每年：相对固定Compact的纠错"
    )
    print(
        "======================================"
    )

    for row in year_rows:
        print(
            f"{str(row['year']):4s} "
            f"N={row['N']:4d} "
            f"base_wrong={row['base_wrong']:3d} "
            f"corrected={row['corrected']:3d} "
            f"regressed={row['regressed']:3d} "
            f"net={row['net']:+3d} "
            f"corrFP={row['corrected_FP']:3d} "
            f"corrFN={row['corrected_FN']:3d}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "Compact错误按距离0.5分桶"
    )
    print(
        "======================================"
    )

    for row in margin_rows:
        rate = row[
            "correction_rate"
        ]

        rate_text = (
            "nan"
            if np.isnan(rate)
            else f"{rate:.2%}"
        )

        print(
            f"{row['margin_bin']:10s} "
            f"errors={row['base_errors']:3d} "
            f"corrected={row['corrected']:3d} "
            f"rate={rate_text:>7s} "
            f"FP={row['corrected_FP']:3d} "
            f"FN={row['corrected_FN']:3d}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "150维存在性与精确退回"
    )
    print(
        "======================================"
    )

    for row in presence_rows:
        print(
            f"presence150={row['presence150']} "
            f"N={row['N']:4d} "
            f"base_wrong={row['base_wrong']:3d} "
            f"corrected={row['corrected']:3d} "
            f"regressed={row['regressed']:3d} "
            f"net={row['net']:+3d} "
            f"max|final-base|="
            f"{row['max_abs_final_base_probability_diff']:.12g}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "数量10 + 150 对比 仅150"
    )
    print(
        "======================================"
    )

    for row in compare_rows:
        print(
            f"{str(row['year']):4s} "
            f"count10更好={row['count10_better_than_150']:3d} "
            f"count10更差={row['count10_worse_than_150']:3d} "
            f"净值={row['direct_net_vs_150']:+3d} "
            f"base错误中仅count10纠正="
            f"{row['base_error_count10_only_correct']:3d} "
            f"仅150纠正="
            f"{row['base_error_150_only_correct']:3d}"
        )

    print()
    print(
        "数量10独有的Compact错误纠正数 =",
        len(count10_only),
    )

    print(
        "仅150独有的Compact错误纠正数 =",
        len(only150_only),
    )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
