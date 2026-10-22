#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)


# ============================================================
# 原 Frozen 模型
# ============================================================

import run_frozen_anchor_residual_2017_2021 as base

ORIGINAL_CONFIGS = copy.deepcopy(
    base.CONFIGS
)


# ============================================================
# 已验证的结构 + 150 实现
# ============================================================

import run_frozen_anchor_residual_20_150_2017_2021 as aug

from run_pilot_compact_eval import (
    fit_logistic,
)


OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-count10-150-direction-loss-2017"
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DIRECTION_WEIGHTS = (
    0.0,
    0.10,
    0.25,
    0.50,
)


# ============================================================
# 固定数量10定义
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


# ============================================================
# 强制使用数量10
# ============================================================

aug.STRUCTURE_DIM = 10

aug.OLD20_NAMES = list(
    COUNT10_NAMES
)


# ============================================================
# 保存CSV
# ============================================================

def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def normalize_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


# ============================================================
# 网络
#
# 参数、层结构与原数量10+150完全相同。
#
# 唯一区别：
# forward额外返回 raw_delta，
# 用于训练阶段的方向辅助损失。
#
# 正式残差仍然：
#
#   presence
#   * uncertainty
#   * tanh(raw_delta)
#
# ============================================================

class DirectionResidualNet(
    aug.FrozenAnchorResidual20_150Net
):

    def forward(
        self,
        compact,
        semantic,
        coverage,
        presence,
    ):
        # ====================================================
        # Frozen Compact
        # ====================================================

        base_logit = (
            compact
            @ self.base_weight
            + self.base_bias
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

        correction_gate = (
            presence
            * uncertainty
        )

        # ====================================================
        # 数量10 | 关系150
        # ====================================================

        structure = semantic[
            :,
            :self.structure_dim,
        ]

        relation = semantic[
            :,
            self.structure_dim:,
        ]

        if (
            structure.shape[1]
            != self.structure_dim
        ):
            raise RuntimeError(
                "structure维数异常"
            )

        if relation.shape[1] == 0:
            raise RuntimeError(
                "当前折没有可用语义关系"
            )

        semantic_hidden = (
            self.semantic_encoder(
                relation
            )
        )

        delta_input = torch.cat(
            [
                structure,
                semantic_hidden,
                coverage,
                base_logit,
                uncertainty,
            ],
            dim=1,
        )

        # ====================================================
        # 训练方向损失使用 raw_delta
        #
        # 正式推理仍使用 bounded_delta
        # ====================================================

        raw_delta = (
            self.delta_head(
                delta_input
            )
        )

        bounded_delta = torch.tanh(
            raw_delta
        )

        final_logit = (
            base_logit
            + correction_gate
            * bounded_delta
        )

        return (
            final_logit,
            base_logit,
            correction_gate,
            bounded_delta,
            raw_delta,
        )


# ============================================================
# 类别平衡的残差方向损失
#
# raw_delta > 0 -> 恶意方向
# raw_delta < 0 -> 良性方向
#
# 只在 presence150=1 样本上计算。
#
# benign与malware分别求平均，
# 再对两个类别平均。
# ============================================================

def balanced_direction_loss(
    raw_delta,
    labels,
    presence,
):
    per_sample = (
        F.binary_cross_entropy_with_logits(
            raw_delta,
            labels,
            reduction="none",
        )
    )

    active = (
        presence
        > 0.5
    )

    benign = (
        active
        &
        (
            labels
            < 0.5
        )
    )

    malware = (
        active
        &
        (
            labels
            >= 0.5
        )
    )

    terms = []

    if bool(
        benign.any().item()
    ):
        terms.append(
            per_sample[
                benign
            ].mean()
        )

    if bool(
        malware.any().item()
    ):
        terms.append(
            per_sample[
                malware
            ].mean()
        )

    if not terms:
        # 保持计算图，但损失严格为0。
        return (
            raw_delta.sum()
            * 0.0
        )

    if len(terms) == 1:
        return terms[0]

    return torch.stack(
        terms
    ).mean()


# ============================================================
# 新 fit_one
#
# 除方向辅助损失外，
# 与正式 Frozen fit_one 保持一致。
# ============================================================

def fit_one_direction(
    x_train,
    s_train,
    c_train,
    p_train,
    y_train,

    x_val,
    s_val,
    c_val,
    p_val,
    y_val,

    config,
    seed,
    device,
    direction_weight,
):
    base.set_seed(
        seed
    )

    # ========================================================
    # 第一阶段：
    # 原Compact逻辑回归，完全固定
    # ========================================================

    base_weight, base_bias = (
        fit_logistic(
            x_train,
            y_train.astype(int),
            seed,
            iterations=6000,
            learning_rate=0.05,
            l2=0.03,
        )
    )

    # ========================================================
    # 第二阶段：
    # 冻结Compact，只训练残差
    # ========================================================

    model = DirectionResidualNet(
        compact_dim=x_train.shape[1],
        semantic_dim=s_train.shape[1],
        coverage_dim=c_train.shape[1],
        hidden=config[
            "hidden"
        ],
        dropout=config[
            "dropout"
        ],
        base_weight=base_weight,
        base_bias=base_bias,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config[
            "learning_rate"
        ],
        weight_decay=config[
            "weight_decay"
        ],
    )

    final_loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    tx = base.tensor(
        x_train,
        device,
    )

    ts = base.tensor(
        s_train,
        device,
    )

    tc = base.tensor(
        c_train,
        device,
    )

    tp = base.tensor(
        p_train,
        device,
    )

    ty = base.tensor(
        y_train.reshape(
            -1,
            1,
        ),
        device,
    )

    vx = base.tensor(
        x_val,
        device,
    )

    vs = base.tensor(
        s_val,
        device,
    )

    vc = base.tensor(
        c_val,
        device,
    )

    vp = base.tensor(
        p_val,
        device,
    )

    best_score = (
        -float("inf")
    )

    best_epoch = 1

    best_state = None

    patience_left = config[
        "patience"
    ]

    fixed_epochs = bool(
        config.get(
            "fixed_epochs",
            False,
        )
    )

    history = []

    for epoch in range(
        1,
        config[
            "max_epochs"
        ] + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            logits,
            base_logits,
            correction_gate,
            bounded_delta,
            raw_delta,
        ) = model(
            tx,
            ts,
            tc,
            tp,
        )

        # ====================================================
        # 原始最终分类损失
        # ====================================================

        final_loss = (
            final_loss_fn(
                logits,
                ty,
            )
        )

        # ====================================================
        # 新增：
        # 不经过 uncertainty 的残差方向损失
        #
        # λ=0时严格只使用原始final_loss，
        # 保证基线复现。
        # ====================================================

        if (
            direction_weight
            > 0.0
        ):
            direction_loss = (
                balanced_direction_loss(
                    raw_delta,
                    ty,
                    tp,
                )
            )

            loss = (
                final_loss
                + direction_weight
                * direction_loss
            )

        else:
            direction_loss = (
                raw_delta.sum()
                * 0.0
            )

            loss = final_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        # ====================================================
        # 验证
        # ====================================================

        model.eval()

        with torch.no_grad():
            (
                val_logits,
                val_base_logits,
                _,
                val_bounded_delta,
                val_raw_delta,
            ) = model(
                vx,
                vs,
                vc,
                vp,
            )

            val_probability = (
                torch.sigmoid(
                    val_logits
                )
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

            val_base_probability = (
                torch.sigmoid(
                    val_base_logits
                )
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

            val_raw = (
                val_raw_delta
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

        val_metrics = base.metrics(
            y_val.astype(int),
            val_probability,
            threshold=base.THRESHOLD,
        )

        base_metrics = base.metrics(
            y_val.astype(int),
            val_base_probability,
            threshold=base.THRESHOLD,
        )

        score = float(
            val_metrics[
                "balanced_accuracy"
            ]
        )

        # ====================================================
        # 方向诊断
        #
        # 不参与提前停止。
        # ====================================================

        yv = y_val.astype(
            int
        ).ravel()

        presence_np = (
            np.asarray(
                p_val
            ).ravel()
            > 0.5
        )

        direction_prediction = (
            val_raw
            >= 0.0
        ).astype(
            int
        )

        if presence_np.any():
            direction_accuracy_presence = float(
                np.mean(
                    direction_prediction[
                        presence_np
                    ]
                    ==
                    yv[
                        presence_np
                    ]
                )
            )

        else:
            direction_accuracy_presence = (
                float("nan")
            )

        base_prediction = (
            val_base_probability
            >= base.THRESHOLD
        ).astype(
            int
        )

        active_base_wrong = (
            presence_np
            &
            (
                base_prediction
                != yv
            )
        )

        if active_base_wrong.any():
            direction_accuracy_base_wrong = float(
                np.mean(
                    direction_prediction[
                        active_base_wrong
                    ]
                    ==
                    yv[
                        active_base_wrong
                    ]
                )
            )

        else:
            direction_accuracy_base_wrong = (
                float("nan")
            )

        history.append({
            "epoch":
                epoch,

            **val_metrics,

            "base_BA":
                float(
                    base_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "base_AUC":
                float(
                    base_metrics[
                        "auc"
                    ]
                ),

            "base_ACC":
                float(
                    base_metrics[
                        "accuracy"
                    ]
                ),

            "train_final_loss":
                float(
                    final_loss
                    .detach()
                    .cpu()
                ),

            "train_direction_loss":
                float(
                    direction_loss
                    .detach()
                    .cpu()
                ),

            "direction_accuracy_presence":
                direction_accuracy_presence,

            "direction_accuracy_base_wrong":
                direction_accuracy_base_wrong,
        })

        # ====================================================
        # 与正式源域协议完全一致
        # ====================================================

        if fixed_epochs:
            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            continue

        if (
            score
            > best_score
            + 1e-8
        ):
            best_score = score

            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_left = config[
                "patience"
            ]

        else:
            patience_left -= 1

            if (
                patience_left
                <= 0
            ):
                break

    if best_state is None:
        best_state = (
            copy.deepcopy(
                model.state_dict()
            )
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        history,
    )


# ============================================================
# 汇总
# ============================================================

def summarize_variant(
    rows,
):
    by_split = defaultdict(
        list
    )

    for row in rows:
        by_split[
            row[
                "split_name"
            ]
        ].append(
            row
        )

    split_rows = []

    for split_name in sorted(
        by_split
    ):
        rs = by_split[
            split_name
        ]

        split_rows.append({
            "split_name":
                split_name,

            "mean_BA":
                float(
                    np.mean([
                        r[
                            "balanced_accuracy"
                        ]
                        for r in rs
                    ])
                ),

            "mean_AUC":
                float(
                    np.mean([
                        r[
                            "auc"
                        ]
                        for r in rs
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        r[
                            "accuracy"
                        ]
                        for r in rs
                    ])
                ),

            "mean_direction_presence":
                float(
                    np.nanmean([
                        r[
                            "direction_accuracy_presence"
                        ]
                        for r in rs
                    ])
                ),

            "mean_direction_base_wrong":
                float(
                    np.nanmean([
                        r[
                            "direction_accuracy_base_wrong"
                        ]
                        for r in rs
                    ])
                ),
        })

    ba = np.asarray(
        [
            r[
                "mean_BA"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    auc = np.asarray(
        [
            r[
                "mean_AUC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    acc = np.asarray(
        [
            r[
                "mean_ACC"
            ]
            for r in split_rows
        ],
        dtype=float,
    )

    return (
        {
            "mean_BA":
                float(
                    ba.mean()
                ),

            "min_split_BA":
                float(
                    ba.min()
                ),

            "std_split_BA":
                float(
                    ba.std()
                ),

            "mean_AUC":
                float(
                    auc.mean()
                ),

            "mean_ACC":
                float(
                    acc.mean()
                ),

            "mean_direction_presence":
                float(
                    np.mean([
                        r[
                            "mean_direction_presence"
                        ]
                        for r in split_rows
                    ])
                ),

            "mean_direction_base_wrong":
                float(
                    np.mean([
                        r[
                            "mean_direction_base_wrong"
                        ]
                        for r in split_rows
                    ])
                ),
        },
        split_rows,
    )


def lambda_name(
    value,
):
    if value == 0:
        return "direction_w0"

    return (
        "direction_w"
        + str(value)
        .replace(
            ".",
            "p",
        )
    )


# ============================================================
# 主程序
# ============================================================

def main():
    print(
        "======================================"
    )

    print(
        "2017-only 残差方向辅助损失"
    )

    print(
        "======================================"
    )

    print(
        "不会读取2018-2021数据。"
    )

    # ========================================================
    # 输入
    # ========================================================

    compact_path = Path(
        "artifacts/p1-features/"
        "features.csv"
    )

    schema_path = Path(
        "artifacts/p1-features/"
        "feature_schema.json"
    )

    semantic_path = Path(
        "artifacts/clean-local-context-2017/"
        "semantic_relation_features.csv.gz"
    )

    split_path = Path(
        "artifacts/p9-2017-rolling-splits/"
        "split_manifest.csv"
    )

    for path in (
        compact_path,
        schema_path,
        semantic_path,
        split_path,
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
    # 数据
    # ========================================================

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                compact_path
            )
        )
    )

    compact_schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            semantic_path
        )
    )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
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
            "final150数量异常: "
            + str(
                len(
                    names150
                )
            )
        )

    rolling_parts = (
        base.read_split_rows(
            split_path
        )
    )

    rolling_splits = {
        name:
            parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling"
            in name
            and parts[
                "train"
            ]
            and parts[
                "test"
            ]
        )
    }

    if len(
        rolling_splits
    ) != 3:
        raise SystemExit(
            "2017滚动划分数量异常: "
            + str(
                len(
                    rolling_splits
                )
            )
        )

    print(
        "rolling splits =",
        sorted(
            rolling_splits
        ),
    )

    # ========================================================
    # 数量10人口检查
    # ========================================================

    old20_by_sha = (
        aug.load_old20_rows()
    )

    required_shas = set()

    for parts in (
        rolling_splits.values()
    ):
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                required_shas.add(
                    normalize_sha(
                        row[
                            "sha256"
                        ]
                    )
                )

    missing = (
        required_shas
        - set(
            old20_by_sha
        )
    )

    if missing:
        raise SystemExit(
            "数量结构缺失2017样本: "
            + str(
                len(
                    missing
                )
            )
        )

    # ========================================================
    # 固定配置
    # ========================================================

    if len(
        ORIGINAL_CONFIGS
    ) != 1:
        raise SystemExit(
            "预期只有一个Frozen固定配置"
        )

    base_config = copy.deepcopy(
        ORIGINAL_CONFIGS[
            0
        ]
    )

    # ========================================================
    # 预构造每折数据
    #
    # 所有lambda完全使用相同输入。
    # ========================================================

    fold_data = {}

    for split_name in sorted(
        rolling_splits
    ):
        train_rows = (
            rolling_splits[
                split_name
            ][
                "train"
            ]
        )

        val_rows = (
            rolling_splits[
                split_name
            ][
                "test"
            ]
        )

        data = (
            aug.build_fold_data_20_150(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                semantic_by_sha,
                names150,
            )
        )

        y_train = (
            base.labels_for(
                train_rows,
                compact_by_sha,
            )
        )

        y_val = (
            base.labels_for(
                val_rows,
                compact_by_sha,
            )
        )

        p_train_np = (
            np.asarray(
                data[
                    "presence_train"
                ]
            ).ravel()
            > 0.5
        )

        y_train_int = (
            y_train.astype(
                int
            ).ravel()
        )

        train_active_benign = int(
            np.sum(
                p_train_np
                &
                (
                    y_train_int
                    == 0
                )
            )
        )

        train_active_malware = int(
            np.sum(
                p_train_np
                &
                (
                    y_train_int
                    == 1
                )
            )
        )

        print()
        print(
            split_name,
            "train=",
            len(
                train_rows
            ),
            "val=",
            len(
                val_rows
            ),
        )

        print(
            " semantic_dim=",
            data[
                "semantic_train"
            ].shape[
                1
            ],
        )

        print(
            " train presence150 benign/malware =",
            train_active_benign,
            "/",
            train_active_malware,
        )

        fold_data[
            split_name
        ] = (
            data,
            y_train,
            y_val,
        )

    # ========================================================
    # λ实验
    # ========================================================

    all_runs = []

    all_split_rows = []

    summary_rows = []

    for direction_weight in (
        DIRECTION_WEIGHTS
    ):
        variant = lambda_name(
            direction_weight
        )

        print()
        print(
            "======================================"
        )

        print(
            variant,
            "lambda=",
            direction_weight,
        )

        print(
            "======================================"
        )

        config = copy.deepcopy(
            base_config
        )

        config[
            "name"
        ] = variant

        current_rows = []

        for split_name in sorted(
            fold_data
        ):
            (
                data,
                y_train,
                y_val,
            ) = fold_data[
                split_name
            ]

            print()
            print(
                split_name
            )

            for seed in SEEDS:
                (
                    model,
                    best_epoch,
                    history,
                ) = fit_one_direction(
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
                    y_train,

                    data[
                        "compact_test"
                    ],
                    data[
                        "semantic_test"
                    ],
                    data[
                        "coverage_test"
                    ],
                    data[
                        "presence_test"
                    ],
                    y_val,

                    config,
                    seed,
                    device,
                    direction_weight,
                )

                if not history:
                    raise RuntimeError(
                        "history为空"
                    )

                best = max(
                    history,
                    key=lambda row:
                        row[
                            "balanced_accuracy"
                        ],
                )

                row = {
                    "variant":
                        variant,

                    "direction_weight":
                        direction_weight,

                    "split_name":
                        split_name,

                    "seed":
                        seed,

                    "balanced_accuracy":
                        float(
                            best[
                                "balanced_accuracy"
                            ]
                        ),

                    "auc":
                        float(
                            best[
                                "auc"
                            ]
                        ),

                    "accuracy":
                        float(
                            best[
                                "accuracy"
                            ]
                        ),

                    "epoch":
                        int(
                            best[
                                "epoch"
                            ]
                        ),

                    "direction_accuracy_presence":
                        float(
                            best[
                                "direction_accuracy_presence"
                            ]
                        ),

                    "direction_accuracy_base_wrong":
                        float(
                            best[
                                "direction_accuracy_base_wrong"
                            ]
                        ),

                    "train_final_loss":
                        float(
                            best[
                                "train_final_loss"
                            ]
                        ),

                    "train_direction_loss":
                        float(
                            best[
                                "train_direction_loss"
                            ]
                        ),
                }

                current_rows.append(
                    row
                )

                all_runs.append(
                    row
                )

                print(
                    f" seed={seed} "
                    f"BA={row['balanced_accuracy']:.6f} "
                    f"AUC={row['auc']:.6f} "
                    f"ACC={row['accuracy']:.6f} "
                    f"dirPresence="
                    f"{row['direction_accuracy_presence']:.3f} "
                    f"dirBaseWrong="
                    f"{row['direction_accuracy_base_wrong']:.3f} "
                    f"epoch={row['epoch']}"
                )

        (
            summary,
            split_rows,
        ) = summarize_variant(
            current_rows
        )

        summary[
            "variant"
        ] = variant

        summary[
            "direction_weight"
        ] = direction_weight

        summary_rows.append(
            summary
        )

        for row in split_rows:
            row[
                "variant"
            ] = variant

            row[
                "direction_weight"
            ] = direction_weight

            all_split_rows.append(
                row
            )

        print()
        print(
            "SUMMARY",
            variant,
        )

        print(
            f" BA={summary['mean_BA']:.6f}"
            f" minBA={summary['min_split_BA']:.6f}"
            f" stdBA={summary['std_split_BA']:.6f}"
            f" AUC={summary['mean_AUC']:.6f}"
            f" ACC={summary['mean_ACC']:.6f}"
            f" dirPresence="
            f"{summary['mean_direction_presence']:.3f}"
            f" dirBaseWrong="
            f"{summary['mean_direction_base_wrong']:.3f}"
        )

    # ========================================================
    # 与 lambda=0 做15个 split×seed 配对
    # ========================================================

    baseline = {
        (
            row[
                "split_name"
            ],
            row[
                "seed"
            ],
        ):
            row
        for row in all_runs
        if row[
            "direction_weight"
        ] == 0.0
    }

    paired_rows = []

    paired_summary_rows = []

    for direction_weight in (
        DIRECTION_WEIGHTS
    ):
        if direction_weight == 0.0:
            continue

        variant = lambda_name(
            direction_weight
        )

        candidate = {
            (
                row[
                    "split_name"
                ],
                row[
                    "seed"
                ],
            ):
                row
            for row in all_runs
            if row[
                "direction_weight"
            ] == direction_weight
        }

        if set(
            candidate
        ) != set(
            baseline
        ):
            raise RuntimeError(
                "配对人口不一致"
            )

        ba_delta = []

        auc_delta = []

        acc_delta = []

        wins = 0
        ties = 0
        losses = 0

        for key in sorted(
            baseline
        ):
            b = baseline[
                key
            ]

            c = candidate[
                key
            ]

            dba = (
                c[
                    "balanced_accuracy"
                ]
                -
                b[
                    "balanced_accuracy"
                ]
            )

            dauc = (
                c[
                    "auc"
                ]
                -
                b[
                    "auc"
                ]
            )

            dacc = (
                c[
                    "accuracy"
                ]
                -
                b[
                    "accuracy"
                ]
            )

            ba_delta.append(
                dba
            )

            auc_delta.append(
                dauc
            )

            acc_delta.append(
                dacc
            )

            if dba > 1e-12:
                result = "win"
                wins += 1

            elif dba < -1e-12:
                result = "loss"
                losses += 1

            else:
                result = "tie"
                ties += 1

            paired_rows.append({
                "variant":
                    variant,

                "direction_weight":
                    direction_weight,

                "split_name":
                    key[
                        0
                    ],

                "seed":
                    key[
                        1
                    ],

                "baseline_BA":
                    b[
                        "balanced_accuracy"
                    ],

                "candidate_BA":
                    c[
                        "balanced_accuracy"
                    ],

                "delta_BA":
                    dba,

                "delta_AUC":
                    dauc,

                "delta_ACC":
                    dacc,

                "result":
                    result,
            })

        paired_summary_rows.append({
            "variant":
                variant,

            "direction_weight":
                direction_weight,

            "pairs":
                len(
                    ba_delta
                ),

            "wins":
                wins,

            "ties":
                ties,

            "losses":
                losses,

            "mean_delta_BA":
                float(
                    np.mean(
                        ba_delta
                    )
                ),

            "median_delta_BA":
                float(
                    np.median(
                        ba_delta
                    )
                ),

            "mean_delta_AUC":
                float(
                    np.mean(
                        auc_delta
                    )
                ),

            "mean_delta_ACC":
                float(
                    np.mean(
                        acc_delta
                    )
                ),
        })

    # ========================================================
    # 保存
    # ========================================================

    write_csv(
        OUTPUT_DIR
        / "per_run.csv",
        all_runs,
    )

    write_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        all_split_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_comparison.csv",
        paired_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_summary.csv",
        paired_summary_rows,
    )

    # ========================================================
    # 最终打印
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "2017-only 最终汇总"
    )

    print(
        "======================================"
    )

    for row in sorted(
        summary_rows,
        key=lambda x:
            x[
                "direction_weight"
            ],
    ):
        print(
            f"lambda={row['direction_weight']:<4} "
            f"BA={row['mean_BA']:.6f} "
            f"minBA={row['min_split_BA']:.6f} "
            f"stdBA={row['std_split_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"dirWrong="
            f"{row['mean_direction_base_wrong']:.3f}"
        )

    print()
    print(
        "======================================"
    )

    print(
        "相对lambda=0的15个配对"
    )

    print(
        "======================================"
    )

    for row in (
        paired_summary_rows
    ):
        print(
            f"lambda={row['direction_weight']:<4} "
            f"胜/平/负="
            f"{row['wins']}/"
            f"{row['ties']}/"
            f"{row['losses']} "
            f"meanΔBA="
            f"{row['mean_delta_BA']:+.6f} "
            f"medianΔBA="
            f"{row['median_delta_BA']:+.6f} "
            f"meanΔAUC="
            f"{row['mean_delta_AUC']:+.6f} "
            f"meanΔACC="
            f"{row['mean_delta_ACC']:+.6f}"
        )

    print()
    print(
        "结果保存到:",
        OUTPUT_DIR,
    )

    print()
    print(
        "本实验严格只使用2017滚动划分。"
    )


if __name__ == "__main__":
    main()
