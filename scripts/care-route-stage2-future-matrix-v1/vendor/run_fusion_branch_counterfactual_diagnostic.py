#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# 复用已经跑通的分层诊断脚本中的正式配置和路径。
import run_frozen_count10_150_layerwise_representation_diagnostic as diag


base = diag.base
aug = diag.aug

OUT = Path(
    "artifacts/"
    "fusion-branch-counterfactual-diagnostic"
)

A = "A_wrong_residual_direction"
D = "D_successfully_corrected"

SEEDS = (0, 1, 2, 3, 4)

FIXED_EPOCHS = 86


# ============================================================
# 当前正式数量10模型
# ============================================================

aug.STRUCTURE_DIM = 10
aug.OLD20_NAMES = list(
    diag.COUNT10_NAMES
)

base.build_fold_data = (
    aug.build_fold_data_20_150
)

base.FrozenAnchorResidualNet = (
    aug.FrozenAnchorResidual20_150Net
)


def norm_sha_series(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# 提取semantic_hidden
# ============================================================

def semantic_hidden_numpy(
    model,
    semantic,
    device,
):
    ts = base.tensor(
        semantic,
        device,
    )

    with torch.no_grad():
        relation = ts[
            :,
            10:,
        ]

        hidden = (
            model.semantic_encoder(
                relation
            )
        )

    return (
        hidden
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# 指定两个分支的替代值，重新计算raw residual
#
# structure_ref=None:
#   使用当前APK自己的数量10
#
# semantic_ref=None:
#   使用当前APK自己的语义16维
#
# 否则替换为2017 active population平均表示。
#
# coverage / base_logit / uncertainty全部保持原样。
# ============================================================

def counterfactual_raw_delta(
    model,
    compact,
    semantic,
    coverage,
    device,
    structure_ref=None,
    semantic_ref=None,
):
    tx = base.tensor(
        compact,
        device,
    )

    ts = base.tensor(
        semantic,
        device,
    )

    tc = base.tensor(
        coverage,
        device,
    )

    with torch.no_grad():

        base_logit = (
            tx
            @ model.base_weight
            + model.base_bias
        )

        base_probability = torch.sigmoid(
            base_logit
        )

        uncertainty = (
            4.0
            * base_probability
            * (
                1.0
                - base_probability
            )
        )

        structure = ts[
            :,
            :10,
        ]

        relation = ts[
            :,
            10:,
        ]

        semantic_hidden = (
            model.semantic_encoder(
                relation
            )
        )

        if structure_ref is not None:
            ref = torch.as_tensor(
                structure_ref,
                dtype=structure.dtype,
                device=device,
            ).reshape(
                1,
                -1,
            )

            structure = ref.expand(
                structure.shape[0],
                -1,
            )

        if semantic_ref is not None:
            ref = torch.as_tensor(
                semantic_ref,
                dtype=semantic_hidden.dtype,
                device=device,
            ).reshape(
                1,
                -1,
            )

            semantic_hidden = ref.expand(
                semantic_hidden.shape[0],
                -1,
            )

        delta_input = torch.cat(
            [
                structure,
                semantic_hidden,
                tc,
                base_logit,
                uncertainty,
            ],
            dim=1,
        )

        raw_delta = (
            model.delta_head(
                delta_input
            )
        )

    return (
        raw_delta
        .detach()
        .cpu()
        .numpy()
        .ravel()
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ========================================================
    # 数据
    # ========================================================

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                diag.COMPACT_PATH
            )
        )
    )

    import json

    compact_schema = json.loads(
        diag.SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            diag.SOURCE_SEM_PATH
        )
    )

    semantic_future = (
        base.read_gz_csv(
            diag.FUTURE_SEM_PATH
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "final150不是150维"
        )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
            + semantic_future
        )
    )

    # ========================================================
    # 正式2017 = 799
    # ========================================================

    source_df = pd.DataFrame(
        semantic_2017
    )

    source_df[
        "sha256"
    ] = norm_sha_series(
        source_df[
            "sha256"
        ]
    )

    split_df = pd.read_csv(
        diag.FIXED_SPLIT_PATH
    )

    split_df[
        "sha256"
    ] = norm_sha_series(
        split_df[
            "sha256"
        ]
    )

    source_sha = sorted(
        set(
            source_df[
                "sha256"
            ]
        )
        &
        set(
            split_df[
                "sha256"
            ]
        )
    )

    if len(source_sha) != 799:
        raise SystemExit(
            "正式2017人口异常: "
            + str(
                len(source_sha)
            )
        )

    source_rows = [
        {
            "sha256":
                sha
        }
        for sha in source_sha
    ]

    y_source = (
        base.labels_for(
            source_rows,
            compact_by_sha,
        )
    )

    # ========================================================
    # 未来机制人口
    # ========================================================

    mechanism = pd.read_csv(
        diag.MECHANISM_PATH
    )

    mechanism[
        "sha256"
    ] = norm_sha_series(
        mechanism[
            "sha256"
        ]
    )

    mechanism[
        "year"
    ] = pd.to_numeric(
        mechanism[
            "year"
        ],
        errors="raise",
    ).astype(int)

    mechanism[
        "label"
    ] = pd.to_numeric(
        mechanism[
            "label"
        ],
        errors="raise",
    ).astype(int)

    mechanism = (
        mechanism
        .drop_duplicates(
            "sha256"
        )
        .sort_values(
            [
                "year",
                "sha256",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(mechanism) != 1985:
        raise SystemExit(
            "未来人口不是1985"
        )

    future_rows = [
        {
            "sha256":
                sha
        }
        for sha in mechanism[
            "sha256"
        ]
    ]

    future_labels = (
        mechanism[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # ========================================================
    # 只用2017统计标准化
    # ========================================================

    data = (
        aug.build_fold_data_20_150(
            source_rows,
            future_rows,
            compact_by_sha,
            compact_schema,
            semantic_by_sha,
            names150,
        )
    )

    print(
        "2017正式人口 =",
        len(source_rows),
    )

    print(
        "辅助维数 =",
        data[
            "semantic_train"
        ].shape[
            1
        ],
    )

    source_presence = (
        np.asarray(
            data[
                "presence_train"
            ]
        ).ravel()
        > 0.5
    )

    print(
        "2017 presence150=1 =",
        int(
            source_presence.sum()
        ),
    )

    # A/D索引
    ad_mask = (
        mechanism[
            "mechanism_category"
        ].isin(
            (
                A,
                D,
            )
        )
        .to_numpy()
    )

    ad_indices = np.flatnonzero(
        ad_mask
    )

    print(
        "A+D =",
        len(
            ad_indices
        ),
    )

    # ========================================================
    # 固定86轮训练
    # ========================================================

    if len(
        base.CONFIGS
    ) != 1:
        raise SystemExit(
            "Frozen配置数量异常"
        )

    config = copy.deepcopy(
        base.CONFIGS[
            0
        ]
    )

    config[
        "fixed_epochs"
    ] = True

    config[
        "max_epochs"
    ] = FIXED_EPOCHS

    config[
        "name"
    ] = (
        "branch_counterfactual"
    )

    rows = []

    for seed in SEEDS:

        print()
        print(
            "seed =",
            seed,
        )

        (
            model,
            best_epoch,
            history,
        ) = base.fit_one(
            data[
                "compact_train"
            ],
            data[
                "semantic_train"
            ],
            data[
                "coverage_train"
            ],
            data[
                "presence_train"
            ],
            y_source,

            # fixed_epochs模式下不会用这些数据选epoch。
            data[
                "compact_train"
            ],
            data[
                "semantic_train"
            ],
            data[
                "coverage_train"
            ],
            data[
                "presence_train"
            ],
            y_source,

            config,
            seed,
            device,
        )

        if best_epoch != FIXED_EPOCHS:
            raise RuntimeError(
                "没有训练到固定86轮"
            )

        # ====================================================
        # 2017 active population作为中性参考
        # ====================================================

        source_sem_hidden = (
            semantic_hidden_numpy(
                model,
                data[
                    "semantic_train"
                ],
                device,
            )
        )

        structure_ref = (
            data[
                "semantic_train"
            ][
                source_presence,
                :10,
            ]
            .mean(
                axis=0
            )
        )

        semantic_ref = (
            source_sem_hidden[
                source_presence
            ]
            .mean(
                axis=0
            )
        )

        # ====================================================
        # 四种输入
        # ====================================================

        full = (
            counterfactual_raw_delta(
                model,
                data[
                    "compact_test"
                ],
                data[
                    "semantic_test"
                ],
                data[
                    "coverage_test"
                ],
                device,
            )
        )

        neutral_structure = (
            counterfactual_raw_delta(
                model,
                data[
                    "compact_test"
                ],
                data[
                    "semantic_test"
                ],
                data[
                    "coverage_test"
                ],
                device,
                structure_ref=
                    structure_ref,
            )
        )

        neutral_semantic = (
            counterfactual_raw_delta(
                model,
                data[
                    "compact_test"
                ],
                data[
                    "semantic_test"
                ],
                data[
                    "coverage_test"
                ],
                device,
                semantic_ref=
                    semantic_ref,
            )
        )

        neutral_both = (
            counterfactual_raw_delta(
                model,
                data[
                    "compact_test"
                ],
                data[
                    "semantic_test"
                ],
                data[
                    "coverage_test"
                ],
                device,
                structure_ref=
                    structure_ref,
                semantic_ref=
                    semantic_ref,
            )
        )

        truth_sign = np.where(
            future_labels == 1,
            1.0,
            -1.0,
        )

        full_align = (
            full
            * truth_sign
        )

        structure_align = (
            neutral_structure
            * truth_sign
        )

        semantic_align = (
            neutral_semantic
            * truth_sign
        )

        both_align = (
            neutral_both
            * truth_sign
        )

        for i in ad_indices:

            full_correct = (
                full_align[
                    i
                ]
                > 0
            )

            structure_correct = (
                structure_align[
                    i
                ]
                > 0
            )

            semantic_correct = (
                semantic_align[
                    i
                ]
                > 0
            )

            both_correct = (
                both_align[
                    i
                ]
                > 0
            )

            rows.append({
                "seed":
                    seed,

                "year":
                    int(
                        mechanism.iloc[
                            i
                        ][
                            "year"
                        ]
                    ),

                "sha256":
                    mechanism.iloc[
                        i
                    ][
                        "sha256"
                    ],

                "label":
                    int(
                        future_labels[
                            i
                        ]
                    ),

                "category":
                    mechanism.iloc[
                        i
                    ][
                        "mechanism_category"
                    ],

                "full_alignment":
                    float(
                        full_align[
                            i
                        ]
                    ),

                "neutral_structure_alignment":
                    float(
                        structure_align[
                            i
                        ]
                    ),

                "neutral_semantic_alignment":
                    float(
                        semantic_align[
                            i
                        ]
                    ),

                "neutral_both_alignment":
                    float(
                        both_align[
                            i
                        ]
                    ),

                "full_correct":
                    int(
                        full_correct
                    ),

                "neutral_structure_correct":
                    int(
                        structure_correct
                    ),

                "neutral_semantic_correct":
                    int(
                        semantic_correct
                    ),

                "neutral_both_correct":
                    int(
                        both_correct
                    ),

                # 完整模型错误时，
                # 去掉该分支后是否翻成正确。
                "structure_flip_to_correct":
                    int(
                        (
                            not full_correct
                        )
                        and
                        structure_correct
                    ),

                "semantic_flip_to_correct":
                    int(
                        (
                            not full_correct
                        )
                        and
                        semantic_correct
                    ),

                "both_flip_to_correct":
                    int(
                        (
                            not full_correct
                        )
                        and
                        both_correct
                    ),

                # 不要求跨过0，
                # 只看向真实类别方向移动多少。
                "structure_alignment_change":
                    float(
                        structure_align[
                            i
                        ]
                        -
                        full_align[
                            i
                        ]
                    ),

                "semantic_alignment_change":
                    float(
                        semantic_align[
                            i
                        ]
                        -
                        full_align[
                            i
                        ]
                    ),

                "both_alignment_change":
                    float(
                        both_align[
                            i
                        ]
                        -
                        full_align[
                            i
                        ]
                    ),
            })

    seed_df = pd.DataFrame(
        rows
    )

    seed_df.to_csv(
        OUT
        / "per_seed_A_D_counterfactual.csv",
        index=False,
    )

    # ========================================================
    # 五seed逐APK平均
    # ========================================================

    numeric = [
        c
        for c in seed_df.columns
        if c not in (
            "seed",
            "year",
            "sha256",
            "label",
            "category",
        )
    ]

    avg = (
        seed_df
        .groupby(
            [
                "year",
                "sha256",
                "label",
                "category",
            ],
            as_index=False,
        )[
            numeric
        ]
        .mean()
    )

    avg.to_csv(
        OUT
        / "A_D_counterfactual_average.csv",
        index=False,
    )

    # ========================================================
    # A/D总体和逐年
    # ========================================================

    summary_rows = []

    for year_value in (
        2018,
        2019,
        2020,
        2021,
        "ALL",
    ):

        if year_value == "ALL":
            yg = avg
        else:
            yg = avg[
                avg[
                    "year"
                ]
                == year_value
            ]

        for category in (
            A,
            D,
        ):
            g = yg[
                yg[
                    "category"
                ]
                == category
            ]

            summary_rows.append({
                "year":
                    year_value,

                "category":
                    category,

                "N":
                    len(g),

                "full_correct":
                    g[
                        "full_correct"
                    ].mean(),

                "neutral_structure_correct":
                    g[
                        "neutral_structure_correct"
                    ].mean(),

                "neutral_semantic_correct":
                    g[
                        "neutral_semantic_correct"
                    ].mean(),

                "neutral_both_correct":
                    g[
                        "neutral_both_correct"
                    ].mean(),

                "structure_flip_to_correct":
                    g[
                        "structure_flip_to_correct"
                    ].mean(),

                "semantic_flip_to_correct":
                    g[
                        "semantic_flip_to_correct"
                    ].mean(),

                "both_flip_to_correct":
                    g[
                        "both_flip_to_correct"
                    ].mean(),

                "structure_alignment_change":
                    g[
                        "structure_alignment_change"
                    ].mean(),

                "semantic_alignment_change":
                    g[
                        "semantic_alignment_change"
                    ].mean(),

                "both_alignment_change":
                    g[
                        "both_alignment_change"
                    ].mean(),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "A_vs_D_counterfactual_summary.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    readable = {
        A:
            "A方向错误",

        D:
            "D成功纠正",
    }

    print()
    print(
        "======================================"
    )
    print(
        "总体分支反事实结果"
    )
    print(
        "======================================"
    )

    all_rows = summary[
        summary[
            "year"
        ].astype(str)
        == "ALL"
    ]

    for category in (
        A,
        D,
    ):
        row = all_rows[
            all_rows[
                "category"
            ]
            == category
        ].iloc[0]

        print()
        print(
            readable[
                category
            ],
            "N=",
            int(
                row[
                    "N"
                ]
            ),
        )

        print(
            " 完整方向正确 =",
            f"{100*row['full_correct']:.2f}%"
        )

        print(
            " 中和数量10后 =",
            f"{100*row['neutral_structure_correct']:.2f}%"
        )

        print(
            " 中和语义16维后 =",
            f"{100*row['neutral_semantic_correct']:.2f}%"
        )

        print(
            " 两个都中和后 =",
            f"{100*row['neutral_both_correct']:.2f}%"
        )

        print(
            " 数量10翻正比例 =",
            f"{100*row['structure_flip_to_correct']:.2f}%"
        )

        print(
            " 语义16翻正比例 =",
            f"{100*row['semantic_flip_to_correct']:.2f}%"
        )

        print(
            " 数量10中和 Δ对齐 =",
            f"{row['structure_alignment_change']:+.4f}"
        )

        print(
            " 语义16中和 Δ对齐 =",
            f"{row['semantic_alignment_change']:+.4f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐年A方向错误"
    )
    print(
        "======================================"
    )

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        row = summary[
            (
                summary[
                    "year"
                ].astype(str)
                == str(year)
            )
            &
            (
                summary[
                    "category"
                ]
                == A
            )
        ].iloc[0]

        print(
            f"{year} "
            f"N={int(row['N']):3d} "
            f"full={100*row['full_correct']:.1f}% "
            f"去数量={100*row['neutral_structure_correct']:.1f}% "
            f"去语义={100*row['neutral_semantic_correct']:.1f}% "
            f"数量翻正={100*row['structure_flip_to_correct']:.1f}% "
            f"语义翻正={100*row['semantic_flip_to_correct']:.1f}%"
        )

    print()
    print(
        "结果保存到:",
        OUT,
    )

    print()
    print(
        "注意：这是固定模型上的反事实敏感性诊断，"
        "不能直接解释为因果，也不能用未来标签调模型。"
    )


if __name__ == "__main__":
    main()
