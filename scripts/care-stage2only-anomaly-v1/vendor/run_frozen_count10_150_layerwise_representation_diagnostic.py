#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)


# ============================================================
# 当前正式Frozen实现
# ============================================================

import run_frozen_anchor_residual_2017_2021 as base

import run_frozen_anchor_residual_20_150_2017_2021 as aug


# ============================================================
# 固定实验定义
# ============================================================

OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-count10-150-layerwise-representation-diagnostic"
)

COMPACT_PATH = Path(
    "artifacts/p1-features/"
    "features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/"
    "feature_schema.json"
)

SOURCE_SEM_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

FUTURE_SEM_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

FIXED_SPLIT_PATH = Path(
    "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

MECHANISM_PATH = Path(
    "artifacts/residual-direction-suppression-diagnostic/"
    "apk_mean_probability_diagnostic.csv"
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

# 2017-only数量10源域选择出的固定epoch。
FIXED_EPOCHS = 86

EPS = 1e-12


# ============================================================
# 数量10
# ============================================================

ROLES = (
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

COUNT10_NAMES = [
    f"up_role__{role}__{metric}"
    for role in ROLES
    for metric in COUNT_METRICS
]

if len(COUNT10_NAMES) != 10:
    raise RuntimeError(
        "数量块不是10维"
    )


# 强制使用数量10。
aug.STRUCTURE_DIM = 10

aug.OLD20_NAMES = list(
    COUNT10_NAMES
)

base.build_fold_data = (
    aug.build_fold_data_20_150
)

base.FrozenAnchorResidualNet = (
    aug.FrozenAnchorResidual20_150Net
)


# ============================================================
# 工具
# ============================================================

def norm_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def norm_sha_series(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def l2_normalize(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norm = np.maximum(
        norm,
        EPS,
    )

    return (
        x
        / norm
    )


# ============================================================
# 提取两层16维表示
#
# 第一层：
#   semantic_hidden
#   = 150维关系编码后的16维
#
# 第二层：
#   delta_hidden
#   = 数量10 + semantic_hidden + coverage
#     + base_logit + uncertainty
#     经过残差头最后输出层之前的16维
#
# 同时拿 raw_delta。
# ============================================================

def extract_layerwise(
    model,
    compact,
    semantic,
    coverage,
    device,
):
    model.eval()

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

        # ====================================================
        # Frozen Compact
        # ====================================================

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

        # ====================================================
        # 数量10 / 关系
        # ====================================================

        structure = ts[
            :,
            :10,
        ]

        relation = ts[
            :,
            10:,
        ]

        if structure.shape[1] != 10:
            raise RuntimeError(
                "数量块切分不是10维"
            )

        if relation.shape[1] == 0:
            raise RuntimeError(
                "没有可用关系维"
            )

        # 第一层表示。
        semantic_hidden = (
            model.semantic_encoder(
                relation
            )
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

        # ====================================================
        # 残差头：
        #
        # Linear
        # LayerNorm
        # GELU
        # Dropout
        # Linear -> raw_delta
        #
        # eval模式下Dropout不随机。
        # ====================================================

        layers = list(
            model.delta_head.children()
        )

        if len(layers) < 2:
            raise RuntimeError(
                "delta_head层数异常"
            )

        delta_hidden = (
            delta_input
        )

        for layer in layers[
            :-1
        ]:
            delta_hidden = layer(
                delta_hidden
            )

        raw_delta = layers[
            -1
        ](
            delta_hidden
        )

    return (
        semantic_hidden
        .cpu()
        .numpy(),

        delta_hidden
        .cpu()
        .numpy(),

        raw_delta
        .cpu()
        .numpy()
        .ravel(),

        base_probability
        .cpu()
        .numpy()
        .ravel(),
    )


# ============================================================
# 计算一个表示空间中的类别几何
#
# source只使用：
#   2017 presence150=1 的样本
#
# 因为presence=0时残差门为0，
# 这些样本实际上不会训练残差分支。
#
# 返回两种诊断：
#
# 1. 最近类别：
#    max(sim to malware)
#    -
#    max(sim to benign)
#
# 2. 类别中心：
#    sim to malware prototype
#    -
#    sim to benign prototype
#
# 这两个原始score都不依赖未来标签。
#
# 再乘 truth_sign 后：
#   >0 = 更靠近真实类别
#   <0 = 更靠近相反类别
# ============================================================

def class_geometry(
    source_hidden,
    source_labels,
    future_hidden,
    future_labels,
):
    source_hidden = l2_normalize(
        source_hidden
    )

    future_hidden = l2_normalize(
        future_hidden
    )

    source_labels = np.asarray(
        source_labels,
        dtype=int,
    )

    future_labels = np.asarray(
        future_labels,
        dtype=int,
    )

    benign_mask = (
        source_labels
        == 0
    )

    malware_mask = (
        source_labels
        == 1
    )

    if (
        benign_mask.sum() == 0
        or
        malware_mask.sum() == 0
    ):
        raise RuntimeError(
            "源域active population缺少类别"
        )

    similarity = (
        future_hidden
        @ source_hidden.T
    )

    max_benign = (
        similarity[
            :,
            benign_mask,
        ]
        .max(
            axis=1
        )
    )

    max_malware = (
        similarity[
            :,
            malware_mask,
        ]
        .max(
            axis=1
        )
    )

    nearest_class_score = (
        max_malware
        - max_benign
    )

    # --------------------------------------------------------
    # 类别中心
    # --------------------------------------------------------

    benign_proto = (
        source_hidden[
            benign_mask
        ]
        .mean(
            axis=0,
            keepdims=True,
        )
    )

    malware_proto = (
        source_hidden[
            malware_mask
        ]
        .mean(
            axis=0,
            keepdims=True,
        )
    )

    benign_proto = l2_normalize(
        benign_proto
    )[0]

    malware_proto = l2_normalize(
        malware_proto
    )[0]

    benign_proto_sim = (
        future_hidden
        @ benign_proto
    )

    malware_proto_sim = (
        future_hidden
        @ malware_proto
    )

    prototype_class_score = (
        malware_proto_sim
        - benign_proto_sim
    )

    # --------------------------------------------------------
    # 对真实类别对齐，仅用于事后解释。
    #
    # malware -> +1
    # benign  -> -1
    # --------------------------------------------------------

    truth_sign = np.where(
        future_labels == 1,
        1.0,
        -1.0,
    )

    nearest_truth = (
        nearest_class_score
        * truth_sign
    )

    prototype_truth = (
        prototype_class_score
        * truth_sign
    )

    return {
        "nearest_class_score":
            nearest_class_score,

        "nearest_truth_alignment":
            nearest_truth,

        "nearest_truth_correct":
            (
                nearest_truth
                > 0.0
            ).astype(int),

        "prototype_class_score":
            prototype_class_score,

        "prototype_truth_alignment":
            prototype_truth,

        "prototype_truth_correct":
            (
                prototype_truth
                > 0.0
            ).astype(int),
    }


# ============================================================
# 汇总
# ============================================================

def group_summary(
    g,
    year,
    category,
):
    return {
        "year":
            year,

        "category":
            category,

        "N":
            len(g),

        "semantic_nearest_alignment":
            (
                float(
                    g[
                        "semantic_nearest_truth_alignment"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "semantic_nearest_correct_rate":
            (
                float(
                    g[
                        "semantic_nearest_correct_fraction"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "semantic_prototype_alignment":
            (
                float(
                    g[
                        "semantic_prototype_truth_alignment"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "semantic_prototype_correct_rate":
            (
                float(
                    g[
                        "semantic_prototype_correct_fraction"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "delta_nearest_alignment":
            (
                float(
                    g[
                        "delta_nearest_truth_alignment"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "delta_nearest_correct_rate":
            (
                float(
                    g[
                        "delta_nearest_correct_fraction"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "delta_prototype_alignment":
            (
                float(
                    g[
                        "delta_prototype_truth_alignment"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "delta_prototype_correct_rate":
            (
                float(
                    g[
                        "delta_prototype_correct_fraction"
                    ].mean()
                )
                if len(g)
                else np.nan
            ),

        "raw_direction_correct_fraction":
            (
                float(
                    g[
                        "raw_direction_correct_fraction"
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
                    ]
                    == 0
                ).sum()
            ),

        "malware":
            int(
                (
                    g[
                        "label"
                    ]
                    == 1
                ).sum()
            ),
    }


def main():

    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        SOURCE_SEM_PATH,
        FUTURE_SEM_PATH,
        FIXED_SPLIT_PATH,
        MECHANISM_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "找不到输入: "
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 设备
    # ========================================================

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
        torch.backends.cudnn.deterministic = (
            True
        )

        torch.backends.cudnn.benchmark = (
            False
        )

    # ========================================================
    # Compact
    # ========================================================

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                COMPACT_PATH
            )
        )
    )

    compact_schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    # ========================================================
    # Semantic
    # ========================================================

    semantic_2017 = (
        base.read_gz_csv(
            SOURCE_SEM_PATH
        )
    )

    semantic_future = (
        base.read_gz_csv(
            FUTURE_SEM_PATH
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    future_names150 = (
        base.support150_names(
            semantic_future[
                0
            ].keys()
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "2017 final150不是150维"
        )

    if list(names150) != list(
        future_names150
    ):
        raise SystemExit(
            "2017/未来 final150列不一致"
        )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
            + semantic_future
        )
    )

    # ========================================================
    # 正式2017人口 = 799
    # ========================================================

    source_sem_df = pd.DataFrame(
        semantic_2017
    )

    source_sem_df[
        "sha256"
    ] = norm_sha_series(
        source_sem_df[
            "sha256"
        ]
    )

    split_df = pd.read_csv(
        FIXED_SPLIT_PATH
    )

    split_df[
        "sha256"
    ] = norm_sha_series(
        split_df[
            "sha256"
        ]
    )

    source_sha_set = (
        set(
            source_sem_df[
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

    if len(
        source_sha_set
    ) != 799:
        raise SystemExit(
            "正式2017人口不是799: "
            + str(
                len(
                    source_sha_set
                )
            )
        )

    source_shas = sorted(
        source_sha_set
    )

    source_rows = [
        {
            "sha256":
                sha
        }
        for sha in source_shas
    ]

    source_labels = (
        base.labels_for(
            source_rows,
            compact_by_sha,
        )
        .astype(int)
        .ravel()
    )

    print(
        "2017正式人口 =",
        len(
            source_rows
        ),
    )

    print(
        "2017良性/恶意 =",
        int(
            (
                source_labels
                == 0
            ).sum()
        ),
        "/",
        int(
            (
                source_labels
                == 1
            ).sum()
        ),
    )

    # ========================================================
    # 未来1985 + A/B/C/D信息
    # ========================================================

    mechanism = pd.read_csv(
        MECHANISM_PATH
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

    if len(
        mechanism
    ) != 1985:
        raise SystemExit(
            "未来机制人口不是1985: "
            + str(
                len(
                    mechanism
                )
            )
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
    # 一次构造：
    #
    # train = 2017正式799
    # test  = 未来1985
    #
    # 所有标准化统计只来自2017。
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
        "辅助输入维数 =",
        data[
            "semantic_train"
        ].shape[
            1
        ],
    )

    if (
        data[
            "semantic_train"
        ].shape[
            1
        ]
        <= 10
    ):
        raise RuntimeError(
            "数量10+关系维度异常"
        )

    source_presence = (
        np.asarray(
            data[
                "presence_train"
            ]
        )
        .ravel()
        > 0.5
    )

    print(
        "2017 presence150=1 =",
        int(
            source_presence.sum()
        ),
    )

    print(
        "其中良性/恶意 =",
        int(
            np.sum(
                source_presence
                &
                (
                    source_labels
                    == 0
                )
            )
        ),
        "/",
        int(
            np.sum(
                source_presence
                &
                (
                    source_labels
                    == 1
                )
            )
        ),
    )

    # 残差表示空间的参考人口只用source presence=1。
    source_active_labels = (
        source_labels[
            source_presence
        ]
    )

    # ========================================================
    # 固定训练配置
    # ========================================================

    if len(
        base.CONFIGS
    ) != 1:
        raise SystemExit(
            "基础Frozen配置不是1个"
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
        "frozen_count10_150_"
        "layerwise_diagnostic"
    )

    print(
        "固定训练epoch =",
        FIXED_EPOCHS,
    )

    # ========================================================
    # 五个随机种子
    #
    # 重要：
    # fit_one为了接口需要val。
    #
    # 这里把2017训练人口本身传给val，
    # 因为 fixed_epochs=True，
    # 不会使用val结果选择epoch。
    #
    # 未来标签完全不会进入训练过程。
    # ========================================================

    seed_rows = []

    probability_rows = []

    for seed in SEEDS:

        print()
        print(
            "======================================"
        )

        print(
            "seed =",
            seed,
        )

        print(
            "======================================"
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
            source_labels,

            # val故意仍然使用2017。
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
            source_labels,

            config,
            seed,
            device,
        )

        if best_epoch != FIXED_EPOCHS:
            raise RuntimeError(
                "固定epoch没有训练到"
                + str(
                    FIXED_EPOCHS
                )
                + "，实际="
                + str(
                    best_epoch
                )
            )

        # ====================================================
        # 2017 source representation
        # ====================================================

        (
            source_sem_hidden,
            source_delta_hidden,
            _,
            _,
        ) = extract_layerwise(
            model,

            data[
                "compact_train"
            ],

            data[
                "semantic_train"
            ],

            data[
                "coverage_train"
            ],

            device,
        )

        source_sem_hidden = (
            source_sem_hidden[
                source_presence
            ]
        )

        source_delta_hidden = (
            source_delta_hidden[
                source_presence
            ]
        )

        # ====================================================
        # Future representation
        # ====================================================

        (
            future_sem_hidden,
            future_delta_hidden,
            future_raw_delta,
            future_base_probability,
        ) = extract_layerwise(
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

        # ====================================================
        # 第一层：150关系编码空间
        # ====================================================

        sem_geo = class_geometry(
            source_sem_hidden,
            source_active_labels,
            future_sem_hidden,
            future_labels,
        )

        # ====================================================
        # 第二层：联合残差隐藏空间
        # ====================================================

        delta_geo = class_geometry(
            source_delta_hidden,
            source_active_labels,
            future_delta_hidden,
            future_labels,
        )

        truth_sign = np.where(
            future_labels == 1,
            1.0,
            -1.0,
        )

        raw_truth_alignment = (
            future_raw_delta
            * truth_sign
        )

        raw_direction_correct = (
            raw_truth_alignment
            > 0.0
        ).astype(int)

        # ====================================================
        # 每seed逐APK
        # ====================================================

        for i in range(
            len(
                mechanism
            )
        ):
            seed_rows.append({
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

                "mechanism_category":
                    mechanism.iloc[
                        i
                    ][
                        "mechanism_category"
                    ],

                "semantic_nearest_class_score":
                    float(
                        sem_geo[
                            "nearest_class_score"
                        ][
                            i
                        ]
                    ),

                "semantic_nearest_truth_alignment":
                    float(
                        sem_geo[
                            "nearest_truth_alignment"
                        ][
                            i
                        ]
                    ),

                "semantic_nearest_truth_correct":
                    int(
                        sem_geo[
                            "nearest_truth_correct"
                        ][
                            i
                        ]
                    ),

                "semantic_prototype_class_score":
                    float(
                        sem_geo[
                            "prototype_class_score"
                        ][
                            i
                        ]
                    ),

                "semantic_prototype_truth_alignment":
                    float(
                        sem_geo[
                            "prototype_truth_alignment"
                        ][
                            i
                        ]
                    ),

                "semantic_prototype_truth_correct":
                    int(
                        sem_geo[
                            "prototype_truth_correct"
                        ][
                            i
                        ]
                    ),

                "delta_nearest_class_score":
                    float(
                        delta_geo[
                            "nearest_class_score"
                        ][
                            i
                        ]
                    ),

                "delta_nearest_truth_alignment":
                    float(
                        delta_geo[
                            "nearest_truth_alignment"
                        ][
                            i
                        ]
                    ),

                "delta_nearest_truth_correct":
                    int(
                        delta_geo[
                            "nearest_truth_correct"
                        ][
                            i
                        ]
                    ),

                "delta_prototype_class_score":
                    float(
                        delta_geo[
                            "prototype_class_score"
                        ][
                            i
                        ]
                    ),

                "delta_prototype_truth_alignment":
                    float(
                        delta_geo[
                            "prototype_truth_alignment"
                        ][
                            i
                        ]
                    ),

                "delta_prototype_truth_correct":
                    int(
                        delta_geo[
                            "prototype_truth_correct"
                        ][
                            i
                        ]
                    ),

                "raw_delta":
                    float(
                        future_raw_delta[
                            i
                        ]
                    ),

                "raw_truth_alignment":
                    float(
                        raw_truth_alignment[
                            i
                        ]
                    ),

                "raw_direction_correct":
                    int(
                        raw_direction_correct[
                            i
                        ]
                    ),

                "base_probability":
                    float(
                        future_base_probability[
                            i
                        ]
                    ),
            })

        print(
            " future raw方向正确率 =",
            f"{raw_direction_correct.mean():.4f}",
        )

    # ========================================================
    # 保存seed级
    # ========================================================

    seed_df = pd.DataFrame(
        seed_rows
    )

    seed_df.to_csv(
        OUTPUT_DIR
        / "per_seed_sample_layerwise.csv",
        index=False,
    )

    # ========================================================
    # 五seed逐APK平均
    # ========================================================

    numeric_cols = [
        "semantic_nearest_class_score",
        "semantic_nearest_truth_alignment",
        "semantic_nearest_truth_correct",

        "semantic_prototype_class_score",
        "semantic_prototype_truth_alignment",
        "semantic_prototype_truth_correct",

        "delta_nearest_class_score",
        "delta_nearest_truth_alignment",
        "delta_nearest_truth_correct",

        "delta_prototype_class_score",
        "delta_prototype_truth_alignment",
        "delta_prototype_truth_correct",

        "raw_delta",
        "raw_truth_alignment",
        "raw_direction_correct",
        "base_probability",
    ]

    avg = (
        seed_df
        .groupby(
            [
                "year",
                "sha256",
                "label",
                "mechanism_category",
            ],
            as_index=False,
        )[
            numeric_cols
        ]
        .mean()
    )

    # --------------------------------------------------------
    # 这些 *_truth_correct 平均后就是：
    # 五seed中有多少比例落在真实类别一侧。
    # --------------------------------------------------------

    avg = avg.rename(
        columns={
            "semantic_nearest_truth_correct":
                "semantic_nearest_correct_fraction",

            "semantic_prototype_truth_correct":
                "semantic_prototype_correct_fraction",

            "delta_nearest_truth_correct":
                "delta_nearest_correct_fraction",

            "delta_prototype_truth_correct":
                "delta_prototype_correct_fraction",

            "raw_direction_correct":
                "raw_direction_correct_fraction",
        }
    )

    # ========================================================
    # 合并原机制诊断信息
    # ========================================================

    extra_cols = [
        "year",
        "sha256",
        "base_margin_probability",
        "presence150",
        "base_correct",
    ]

    available_extra = [
        c
        for c in extra_cols
        if c in mechanism.columns
    ]

    avg = avg.merge(
        mechanism[
            available_extra
        ],
        on=[
            "year",
            "sha256",
        ],
        how="left",
        validate="one_to_one",
    )

    avg.to_csv(
        OUTPUT_DIR
        / "sample_layerwise_average.csv",
        index=False,
    )

    # ========================================================
    # 只看A与D
    # ========================================================

    A = (
        "A_wrong_residual_direction"
    )

    D = (
        "D_successfully_corrected"
    )

    ad = avg[
        avg[
            "mechanism_category"
        ].isin(
            (
                A,
                D,
            )
        )
    ].copy()

    if len(ad) != 142:
        print(
            "注意：A+D不是142，实际=",
            len(ad),
        )

    # ========================================================
    # A/D逐年汇总
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
            yg = ad
        else:
            yg = ad[
                ad[
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
                    "mechanism_category"
                ]
                == category
            ]

            summary_rows.append(
                group_summary(
                    g,
                    year_value,
                    category,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "A_vs_D_layerwise_summary.csv",
        index=False,
    )

    # ========================================================
    # 内部表示 -> 最终方向 的四象限
    #
    # 用五seed平均：
    #
    # delta内部表示正确：
    #   nearest truth alignment > 0
    #
    # 最终残差方向多数正确：
    #   五seed中 >50% 方向正确
    #
    # A中重点看：
    #
    # 1) 表示错 + 输出错
    #    -> 表示阶段问题
    #
    # 2) 表示对 + 输出错
    #    -> 最后一层读取问题
    # ========================================================

    ad[
        "delta_representation_correct"
    ] = (
        ad[
            "delta_nearest_truth_alignment"
        ]
        > 0.0
    )

    ad[
        "semantic_representation_correct"
    ] = (
        ad[
            "semantic_nearest_truth_alignment"
        ]
        > 0.0
    )

    ad[
        "raw_majority_correct"
    ] = (
        ad[
            "raw_direction_correct_fraction"
        ]
        > 0.5
    )

    def quadrant(
        rep_correct,
        raw_correct,
    ):
        if (
            not rep_correct
            and not raw_correct
        ):
            return (
                "representation_wrong__output_wrong"
            )

        if (
            rep_correct
            and not raw_correct
        ):
            return (
                "representation_correct__output_wrong"
            )

        if (
            not rep_correct
            and raw_correct
        ):
            return (
                "representation_wrong__output_correct"
            )

        return (
            "representation_correct__output_correct"
        )

    ad[
        "delta_output_quadrant"
    ] = [
        quadrant(
            r,
            o,
        )
        for r, o in zip(
            ad[
                "delta_representation_correct"
            ],
            ad[
                "raw_majority_correct"
            ],
        )
    ]

    # ========================================================
    # 关系编码层 -> 联合融合层 的转变
    # ========================================================

    def transition(
        semantic_correct,
        delta_correct,
    ):
        if (
            semantic_correct
            and delta_correct
        ):
            return "semantic_correct__delta_correct"

        if (
            semantic_correct
            and not delta_correct
        ):
            return "semantic_correct__delta_wrong"

        if (
            not semantic_correct
            and delta_correct
        ):
            return "semantic_wrong__delta_correct"

        return "semantic_wrong__delta_wrong"

    ad[
        "semantic_to_delta_transition"
    ] = [
        transition(
            s,
            d,
        )
        for s, d in zip(
            ad[
                "semantic_representation_correct"
            ],
            ad[
                "delta_representation_correct"
            ],
        )
    ]

    # ========================================================
    # 四象限汇总
    # ========================================================

    quadrant_rows = []

    for category in (
        A,
        D,
    ):
        g = ad[
            ad[
                "mechanism_category"
            ]
            == category
        ]

        counts = (
            g[
                "delta_output_quadrant"
            ]
            .value_counts()
        )

        for name in (
            "representation_wrong__output_wrong",
            "representation_correct__output_wrong",
            "representation_wrong__output_correct",
            "representation_correct__output_correct",
        ):
            n = int(
                counts.get(
                    name,
                    0,
                )
            )

            quadrant_rows.append({
                "category":
                    category,

                "quadrant":
                    name,

                "N":
                    n,

                "share":
                    (
                        n
                        / len(g)
                        if len(g)
                        else np.nan
                    ),
            })

    quadrant_df = pd.DataFrame(
        quadrant_rows
    )

    quadrant_df.to_csv(
        OUTPUT_DIR
        / "delta_representation_output_quadrants.csv",
        index=False,
    )

    # ========================================================
    # semantic -> delta 转变汇总
    # ========================================================

    transition_rows = []

    for category in (
        A,
        D,
    ):
        g = ad[
            ad[
                "mechanism_category"
            ]
            == category
        ]

        counts = (
            g[
                "semantic_to_delta_transition"
            ]
            .value_counts()
        )

        for name in (
            "semantic_correct__delta_correct",
            "semantic_correct__delta_wrong",
            "semantic_wrong__delta_correct",
            "semantic_wrong__delta_wrong",
        ):
            n = int(
                counts.get(
                    name,
                    0,
                )
            )

            transition_rows.append({
                "category":
                    category,

                "transition":
                    name,

                "N":
                    n,

                "share":
                    (
                        n
                        / len(g)
                        if len(g)
                        else np.nan
                    ),
            })

    transition_df = pd.DataFrame(
        transition_rows
    )

    transition_df.to_csv(
        OUTPUT_DIR
        / "semantic_to_delta_transitions.csv",
        index=False,
    )

    ad.to_csv(
        OUTPUT_DIR
        / "A_D_sample_layerwise.csv",
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
        "A vs D：分层内部表示"
    )
    print(
        "======================================"
    )

    all_summary = summary[
        summary[
            "year"
        ].astype(str)
        == "ALL"
    ]

    for category in (
        A,
        D,
    ):
        row = all_summary[
            all_summary[
                "category"
            ]
            == category
        ].iloc[
            0
        ]

        print()
        print(
            readable[
                category
            ]
        )

        print(
            " 关系编码16维："
            f"最近类正确率="
            f"{100*row['semantic_nearest_correct_rate']:.2f}% "
            f"中心类正确率="
            f"{100*row['semantic_prototype_correct_rate']:.2f}% "
            f"最近类对齐="
            f"{row['semantic_nearest_alignment']:+.4f}"
        )

        print(
            " 联合残差16维："
            f"最近类正确率="
            f"{100*row['delta_nearest_correct_rate']:.2f}% "
            f"中心类正确率="
            f"{100*row['delta_prototype_correct_rate']:.2f}% "
            f"最近类对齐="
            f"{row['delta_nearest_alignment']:+.4f}"
        )

        print(
            " 最终残差方向："
            f"五seed平均正确比例="
            f"{100*row['raw_direction_correct_fraction']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误：联合表示 vs 最终输出"
    )
    print(
        "======================================"
    )

    aq = quadrant_df[
        quadrant_df[
            "category"
        ]
        == A
    ]

    for _, row in (
        aq.iterrows()
    ):
        print(
            f"{row['quadrant']:42s} "
            f"N={int(row['N']):3d} "
            f"share={100*row['share']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误：关系编码层 -> 联合残差层"
    )
    print(
        "======================================"
    )

    at = transition_df[
        transition_df[
            "category"
        ]
        == A
    ]

    for _, row in (
        at.iterrows()
    ):
        print(
            f"{row['transition']:38s} "
            f"N={int(row['N']):3d} "
            f"share={100*row['share']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐年A/D联合残差表示正确率"
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
        print()
        print(
            year
        )

        yg = summary[
            summary[
                "year"
            ].astype(str)
            == str(year)
        ]

        for category in (
            A,
            D,
        ):
            row = yg[
                yg[
                    "category"
                ]
                == category
            ].iloc[
                0
            ]

            print(
                f" {readable[category]:10s} "
                f"N={int(row['N']):3d} "
                f"关系层正确="
                f"{100*row['semantic_nearest_correct_rate']:.1f}% "
                f"联合层正确="
                f"{100*row['delta_nearest_correct_rate']:.1f}% "
                f"残差方向正确="
                f"{100*row['raw_direction_correct_fraction']:.1f}%"
            )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "注意：未来标签只用于将表示空间方向"
        "对齐到真实类别，整个训练过程不使用未来标签。"
    )


if __name__ == "__main__":
    main()
