#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_frozen_anchor_residual_2017_2021 as frozen_base


SOURCE_SEM_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

FUTURE_SEM_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SPLIT_PATH = Path(
    "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

MECHANISM_PATH = Path(
    "artifacts/residual-direction-suppression-diagnostic/"
    "apk_mean_probability_diagnostic.csv"
)

OUTPUT_DIR = Path(
    "artifacts/relation-label-direction-drift-diagnostic"
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

EPS = 1e-12


def norm_sha(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def find_label_column(df):
    candidates = (
        "label",
        "metadata_label",
        "target",
        "class",
        "y",
    )

    for name in candidates:
        if name in df.columns:
            return name

    raise SystemExit(
        "无法找到标签列。\\n实际列:\\n"
        + ", ".join(df.columns)
    )


def normalize_binary_labels(series):
    """
    统一成：
      0 = 良性
      1 = 恶意

    同时支持数值0/1以及常见字符串标签。
    """

    if series.isna().any():
        raise SystemExit(
            "标签列存在缺失值"
        )

    # 先尝试纯数值0/1。
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        values = numeric.astype(int)

        unique = set(
            values.unique()
        )

        if unique == {0, 1}:
            return values

    # 再处理字符串。
    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "benign": 0,
        "goodware": 0,
        "normal": 0,

        "1": 1,
        "malware": 1,
        "malicious": 1,
    }

    unknown = sorted(
        set(normalized.unique())
        - set(mapping)
    )

    if unknown:
        raise SystemExit(
            "发现无法识别的标签值: "
            + ", ".join(unknown)
        )

    return normalized.map(
        mapping
    ).astype(int)


def safe_rate(a, b):
    if b == 0:
        return np.nan

    return float(a) / float(b)


def pct(x):
    if pd.isna(x):
        return "nan"

    return f"{100.0 * float(x):.2f}%"


def association_stats(
    active,
    labels,
):
    """
    返回：
    benign_rate
    malware_rate
    association = malware_rate - benign_rate
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    active = np.asarray(
        active,
        dtype=bool,
    )

    benign = (
        labels == 0
    )

    malware = (
        labels == 1
    )

    if benign.sum() == 0:
        raise ValueError(
            "没有良性样本"
        )

    if malware.sum() == 0:
        raise ValueError(
            "没有恶意样本"
        )

    benign_rate = float(
        active[
            benign
        ].mean()
    )

    malware_rate = float(
        active[
            malware
        ].mean()
    )

    return (
        benign_rate,
        malware_rate,
        malware_rate
        - benign_rate,
    )


def main():
    for path in (
        SOURCE_SEM_PATH,
        FUTURE_SEM_PATH,
        COMPACT_PATH,
        SPLIT_PATH,
        MECHANISM_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "找不到文件: "
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取2017语义关系
    # ========================================================

    source_sem = pd.read_csv(
        SOURCE_SEM_PATH,
        compression="gzip",
    )

    source_sem["sha256"] = norm_sha(
        source_sem[
            "sha256"
        ]
    )

    names150 = (
        frozen_base.support150_names(
            source_sem.columns
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "2017 final150数量不是150: "
            + str(
                len(names150)
            )
        )

    # ========================================================
    # 2. 严格筛正式训练使用的799个2017 SHA
    #
    # split_manifest中2017源域样本会在多个未来划分重复，
    # 所以只取与2017语义文件交集后的唯一SHA。
    # ========================================================

    split = pd.read_csv(
        SPLIT_PATH
    )

    if "sha256" not in split.columns:
        raise SystemExit(
            "split_manifest没有sha256"
        )

    split["sha256"] = norm_sha(
        split[
            "sha256"
        ]
    )

    source_sha_set = set(
        source_sem[
            "sha256"
        ]
    )

    train_sha = sorted(
        set(
            split[
                "sha256"
            ]
        )
        &
        source_sha_set
    )

    print(
        "2017语义文件APK数 =",
        len(source_sem),
    )

    print(
        "与正式固定split交集的唯一2017 APK =",
        len(train_sha),
    )

    if len(train_sha) != 799:
        raise SystemExit(
            "正式2017训练人口应为799，"
            "实际得到 "
            + str(
                len(train_sha)
            )
        )

    source_sem = (
        source_sem[
            source_sem[
                "sha256"
            ].isin(
                train_sha
            )
        ]
        .copy()
    )

    if len(source_sem) != 799:
        raise SystemExit(
            "过滤后source_sem不是799行"
        )

    # ========================================================
    # 3. 从Compact文件获得2017标签
    # ========================================================

    compact = pd.read_csv(
        COMPACT_PATH
    )

    compact["sha256"] = norm_sha(
        compact[
            "sha256"
        ]
    )

    label_col = find_label_column(
        compact
    )

    labels2017 = (
        compact[
            [
                "sha256",
                label_col,
            ]
        ]
        .drop_duplicates(
            "sha256"
        )
        .rename(
            columns={
                label_col:
                    "label"
            }
        )
    )

    source_sem = source_sem.merge(
        labels2017,
        on="sha256",
        how="left",
        validate="one_to_one",
    )

    if source_sem[
        "label"
    ].isna().any():
        raise SystemExit(
            "部分2017 SHA找不到标签"
        )

    print(
        "检测到标签列 =",
        label_col,
    )

    print(
        "原始标签取值 =",
        source_sem[
            "label"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    source_sem[
        "label"
    ] = normalize_binary_labels(
        source_sem[
            "label"
        ]
    )

    unique_labels = set(
        source_sem[
            "label"
        ].unique()
    )

    if unique_labels != {
        0,
        1,
    }:
        raise SystemExit(
            "2017标签不是0/1: "
            + str(
                unique_labels
            )
        )

    print(
        "2017良性 =",
        int(
            (
                source_sem[
                    "label"
                ] == 0
            ).sum()
        ),
    )

    print(
        "2017恶意 =",
        int(
            (
                source_sem[
                    "label"
                ] == 1
            ).sum()
        ),
    )

    # ========================================================
    # 4. 2017每个关系的类别关联方向
    # ========================================================

    source_values = (
        source_sem[
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

    source_active = (
        np.abs(
            source_values
        )
        > EPS
    )

    source_labels = (
        source_sem[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    source_relation = {}

    for j, name in enumerate(
        names150
    ):
        benign_rate, malware_rate, assoc = (
            association_stats(
                source_active[
                    :,
                    j,
                ],
                source_labels,
            )
        )

        source_relation[
            name
        ] = {
            "source_benign_rate":
                benign_rate,

            "source_malware_rate":
                malware_rate,

            "source_association":
                assoc,

            "source_active_N":
                int(
                    source_active[
                        :,
                        j,
                    ].sum()
                ),
        }

    # ========================================================
    # 5. 未来语义 + 标签 + A/B/C/D
    # ========================================================

    future_sem = pd.read_csv(
        FUTURE_SEM_PATH,
        compression="gzip",
    )

    future_sem["sha256"] = norm_sha(
        future_sem[
            "sha256"
        ]
    )

    future_names = (
        frozen_base.support150_names(
            future_sem.columns
        )
    )

    if set(
        future_names
    ) != set(
        names150
    ):
        raise SystemExit(
            "未来与2017的150维集合不一致"
        )

    mechanism = pd.read_csv(
        MECHANISM_PATH
    )

    mechanism["sha256"] = norm_sha(
        mechanism[
            "sha256"
        ]
    )

    mechanism[
        "year"
    ] = mechanism[
        "year"
    ].astype(int)

    required_mechanism = {
        "year",
        "sha256",
        "label",
        "presence150",
        "base_correct",
        "mechanism_category",
        "base_margin_probability",
    }

    missing = (
        required_mechanism
        - set(
            mechanism.columns
        )
    )

    if missing:
        raise SystemExit(
            "机制诊断文件缺少列: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    future = future_sem.merge(
        mechanism[
            [
                "year",
                "sha256",
                "label",
                "presence150",
                "base_correct",
                "mechanism_category",
                "base_margin_probability",
            ]
        ],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    if len(future) != 1985:
        raise SystemExit(
            "未来正式测试人口不是1985: "
            + str(
                len(future)
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

    future_active = (
        np.abs(
            future_values
        )
        > EPS
    )

    # ========================================================
    # 6. 每年每个关系重新计算类别关联
    #
    # 注意：
    # 未来标签只用于事后机制诊断，
    # 不参与训练或未来模型选择。
    # ========================================================

    relation_rows = []

    relation_year_lookup = {}

    for year in YEARS:
        year_mask = (
            future[
                "year"
            ].to_numpy(
                dtype=int
            )
            == year
        )

        labels = (
            future.loc[
                year_mask,
                "label",
            ]
            .to_numpy(
                dtype=int
            )
        )

        for j, name in enumerate(
            names150
        ):
            active = (
                future_active[
                    year_mask,
                    j,
                ]
            )

            benign_rate, malware_rate, assoc = (
                association_stats(
                    active,
                    labels,
                )
            )

            src = (
                source_relation[
                    name
                ]
            )

            source_assoc = float(
                src[
                    "source_association"
                ]
            )

            # ----------------------------------------------
            # 是否方向反转
            #
            # association > 0：
            # 该关系相对更常见于恶意
            #
            # association < 0：
            # 相对更常见于良性
            # ----------------------------------------------

            sign_flip = (
                source_assoc
                * assoc
                < 0.0
            )

            same_direction = (
                source_assoc
                * assoc
                > 0.0
            )

            source_neutral = (
                abs(
                    source_assoc
                )
                <= EPS
            )

            future_neutral = (
                abs(
                    assoc
                )
                <= EPS
            )

            # 有足够样本支持时单独标出。
            # 5只是诊断展示阈值，
            # 不进入任何分类决策。
            supported_flip = (
                sign_flip
                and
                int(
                    src[
                        "source_active_N"
                    ]
                ) >= 5
                and
                int(
                    active.sum()
                ) >= 5
            )

            row = {
                "year":
                    year,

                "relation":
                    name,

                "source_active_N":
                    int(
                        src[
                            "source_active_N"
                        ]
                    ),

                "source_benign_rate":
                    float(
                        src[
                            "source_benign_rate"
                        ]
                    ),

                "source_malware_rate":
                    float(
                        src[
                            "source_malware_rate"
                        ]
                    ),

                "source_association":
                    source_assoc,

                "future_active_N":
                    int(
                        active.sum()
                    ),

                "future_benign_rate":
                    benign_rate,

                "future_malware_rate":
                    malware_rate,

                "future_association":
                    assoc,

                "association_change":
                    assoc
                    - source_assoc,

                "sign_flip":
                    int(
                        sign_flip
                    ),

                "supported_flip":
                    int(
                        supported_flip
                    ),

                "same_direction":
                    int(
                        same_direction
                    ),

                "source_neutral":
                    int(
                        source_neutral
                    ),

                "future_neutral":
                    int(
                        future_neutral
                    ),
            }

            relation_rows.append(
                row
            )

            relation_year_lookup[
                (
                    year,
                    name,
                )
            ] = row

    relation_direction = pd.DataFrame(
        relation_rows
    )

    # ========================================================
    # 7. 每个未来APK：
    #
    # 看它活跃的关系中：
    #
    # - 多少关系按2017统计是支持真实标签的
    # - 多少关系按2017统计反而指向错误类别
    # - 多少关系到了该未来年份发生关联方向反转
    #
    # 这一步是我们比较 A vs D 的核心。
    # ========================================================

    sample_rows = []

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

        label = int(
            future.iloc[
                i
            ][
                "label"
            ]
        )

        truth_direction = (
            1.0
            if label == 1
            else -1.0
        )

        active_indices = np.flatnonzero(
            future_active[
                i
            ]
        )

        if len(
            active_indices
        ) == 0:
            sample_rows.append({
                "year":
                    year,

                "sha256":
                    sha,

                "active_relation_count":
                    0,

                "mean_source_truth_alignment":
                    0.0,

                "source_truth_support_fraction":
                    0.0,

                "source_truth_opposition_fraction":
                    0.0,

                "future_sign_flip_fraction":
                    0.0,

                "supported_flip_fraction":
                    0.0,

                "mean_abs_source_association":
                    0.0,
            })

            continue

        source_alignments = []
        source_support = 0
        source_oppose = 0
        sign_flips = 0
        supported_flips = 0
        abs_source_assoc = []

        for j in active_indices:
            name = names150[
                j
            ]

            src_assoc = float(
                source_relation[
                    name
                ][
                    "source_association"
                ]
            )

            alignment = (
                truth_direction
                * src_assoc
            )

            source_alignments.append(
                alignment
            )

            abs_source_assoc.append(
                abs(
                    src_assoc
                )
            )

            if alignment > 0:
                source_support += 1

            elif alignment < 0:
                source_oppose += 1

            drift = (
                relation_year_lookup[
                    (
                        year,
                        name,
                    )
                ]
            )

            if int(
                drift[
                    "sign_flip"
                ]
            ):
                sign_flips += 1

            if int(
                drift[
                    "supported_flip"
                ]
            ):
                supported_flips += 1

        n = len(
            active_indices
        )

        sample_rows.append({
            "year":
                year,

            "sha256":
                sha,

            "active_relation_count":
                n,

            "mean_source_truth_alignment":
                float(
                    np.mean(
                        source_alignments
                    )
                ),

            "source_truth_support_fraction":
                source_support
                / n,

            "source_truth_opposition_fraction":
                source_oppose
                / n,

            "future_sign_flip_fraction":
                sign_flips
                / n,

            "supported_flip_fraction":
                supported_flips
                / n,

            "mean_abs_source_association":
                float(
                    np.mean(
                        abs_source_assoc
                    )
                ),
        })

    sample_relation = pd.DataFrame(
        sample_rows
    )

    diagnostic = mechanism.merge(
        sample_relation,
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
            "逐APK关系诊断合并失败"
        )

    # ========================================================
    # 8. A vs D
    # ========================================================

    target_categories = (
        "A_wrong_residual_direction",
        "D_successfully_corrected",
    )

    ad_rows = []

    for year in YEARS:
        yg = diagnostic[
            (
                diagnostic[
                    "year"
                ] == year
            )
            &
            (
                diagnostic[
                    "mechanism_category"
                ].isin(
                    target_categories
                )
            )
        ]

        for category in (
            target_categories
        ):
            g = yg[
                yg[
                    "mechanism_category"
                ] == category
            ]

            ad_rows.append({
                "year":
                    year,

                "category":
                    category,

                "N":
                    len(g),

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

                "mean_source_truth_alignment":
                    (
                        float(
                            g[
                                "mean_source_truth_alignment"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_source_truth_support_fraction":
                    (
                        float(
                            g[
                                "source_truth_support_fraction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_source_truth_opposition_fraction":
                    (
                        float(
                            g[
                                "source_truth_opposition_fraction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_future_sign_flip_fraction":
                    (
                        float(
                            g[
                                "future_sign_flip_fraction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_supported_flip_fraction":
                    (
                        float(
                            g[
                                "supported_flip_fraction"
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

                "benign":
                    int(
                        (
                            g[
                                "label"
                            ] == 0
                        ).sum()
                    ),

                "malware":
                    int(
                        (
                            g[
                                "label"
                            ] == 1
                        ).sum()
                    ),
            })

    ad_summary = pd.DataFrame(
        ad_rows
    )

    # ========================================================
    # 9. 每年整体方向反转数量
    # ========================================================

    year_rows = []

    for year in YEARS:
        g = relation_direction[
            relation_direction[
                "year"
            ] == year
        ]

        active_future = g[
            g[
                "future_active_N"
            ] > 0
        ]

        year_rows.append({
            "year":
                year,

            "future_active_relations":
                len(
                    active_future
                ),

            "sign_flips":
                int(
                    active_future[
                        "sign_flip"
                    ].sum()
                ),

            "supported_flips":
                int(
                    active_future[
                        "supported_flip"
                    ].sum()
                ),

            "same_direction":
                int(
                    active_future[
                        "same_direction"
                    ].sum()
                ),

            "mean_abs_association_change":
                float(
                    active_future[
                        "association_change"
                    ]
                    .abs()
                    .mean()
                ),
        })

    year_summary = pd.DataFrame(
        year_rows
    )

    # ========================================================
    # 10. 找方向变化最大的关系
    # ========================================================

    strongest_changes = (
        relation_direction[
            (
                relation_direction[
                    "source_active_N"
                ] >= 5
            )
            &
            (
                relation_direction[
                    "future_active_N"
                ] >= 5
            )
        ]
        .assign(
            abs_change=lambda x:
                x[
                    "association_change"
                ].abs()
        )
        .sort_values(
            [
                "year",
                "abs_change",
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
    )

    supported_flips = (
        relation_direction[
            relation_direction[
                "supported_flip"
            ] == 1
        ]
        .sort_values(
            [
                "year",
                "source_active_N",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    # ========================================================
    # 11. 保存
    # ========================================================

    relation_direction.to_csv(
        OUTPUT_DIR
        / "relation_label_direction_by_year.csv",
        index=False,
    )

    diagnostic.to_csv(
        OUTPUT_DIR
        / "sample_relation_direction_diagnostic.csv",
        index=False,
    )

    ad_summary.to_csv(
        OUTPUT_DIR
        / "A_vs_D_summary.csv",
        index=False,
    )

    year_summary.to_csv(
        OUTPUT_DIR
        / "year_direction_drift_summary.csv",
        index=False,
    )

    strongest_changes.to_csv(
        OUTPUT_DIR
        / "top20_association_changes_each_year.csv",
        index=False,
    )

    supported_flips.to_csv(
        OUTPUT_DIR
        / "supported_sign_flips.csv",
        index=False,
    )

    # ========================================================
    # 12. 打印
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "每年语义关系类别方向变化"
    )
    print(
        "======================================"
    )

    for _, row in (
        year_summary.iterrows()
    ):
        print(
            f"{int(row['year'])} "
            f"未来出现关系={int(row['future_active_relations']):3d} "
            f"方向反转={int(row['sign_flips']):3d} "
            f"有足够样本的反转={int(row['supported_flips']):3d} "
            f"方向一致={int(row['same_direction']):3d} "
            f"平均|关联变化|="
            f"{row['mean_abs_association_change']:.5f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误 vs D成功纠正"
    )
    print(
        "======================================"
    )

    readable = {
        "A_wrong_residual_direction":
            "A方向错误",

        "D_successfully_corrected":
            "D成功纠正",
    }

    for year in YEARS:
        print()
        print(
            year
        )

        yg = ad_summary[
            ad_summary[
                "year"
            ] == year
        ]

        for category in (
            target_categories
        ):
            row = yg[
                yg[
                    "category"
                ] == category
            ].iloc[0]

            print(
                f"{readable[category]:10s} "
                f"N={int(row['N']):3d} "
                f"关系数={row['mean_active_relations']:.2f} "
                f"2017真值对齐="
                f"{row['mean_source_truth_alignment']:+.5f} "
                f"支持真值比例="
                f"{pct(row['mean_source_truth_support_fraction']):>7s} "
                f"反对真值比例="
                f"{pct(row['mean_source_truth_opposition_fraction']):>7s} "
                f"方向反转比例="
                f"{pct(row['mean_future_sign_flip_fraction']):>7s} "
                f"有效反转比例="
                f"{pct(row['mean_supported_flip_fraction']):>7s} "
                f"margin={row['mean_margin']:.4f} "
                f"B/M={int(row['benign'])}/{int(row['malware'])}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "有足够样本支持的类别方向反转"
    )
    print(
        "======================================"
    )

    if len(
        supported_flips
    ) == 0:
        print(
            "没有找到同时满足源域>=5、未来>=5的方向反转关系。"
        )

    else:
        for year in YEARS:
            yg = supported_flips[
                supported_flips[
                    "year"
                ] == year
            ]

            print()
            print(
                year,
                "N=",
                len(yg),
            )

            for _, row in (
                yg.head(15)
                .iterrows()
            ):
                print(
                    row[
                        "relation"
                    ],
                    " | 2017:",
                    f"{row['source_association']:+.4f}",
                    " → future:",
                    f"{row['future_association']:+.4f}",
                    " | N:",
                    int(
                        row[
                            "source_active_N"
                        ]
                    ),
                    "→",
                    int(
                        row[
                            "future_active_N"
                        ]
                    ),
                )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "注意：未来标签只用于事后机制解释。"
        "本诊断不能用于重新筛选关系、调整Top-5或调未来参数。"
    )


if __name__ == "__main__":
    main()
