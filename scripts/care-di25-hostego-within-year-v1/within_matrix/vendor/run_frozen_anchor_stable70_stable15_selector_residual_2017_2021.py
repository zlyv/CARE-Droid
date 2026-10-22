#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


# ============================================================
# 直接复用已经跑通的：
#
# Stable70
# + Stable15
# + 原冻结锚点框架
#
# 这个模块导入后：
#   base.build_fold_data 已经只返回 Stable15，
#   count10 已经删除。
# ============================================================

import run_frozen_anchor_stable70_stable15_2017_2021 as stable15


base = stable15.base


# ============================================================
# 纠错选择型残差
#
# 不再让网络同时学习：
#   1. 修正方向
#   2. 修正幅度
#
# 方向固定为“反对当前基础判断”。
#
# 网络只学习：
#   q ∈ [0, 1]
#   表示应该多大程度挑战当前基础判断。
#
# 同时只有：
#
#   gate > |base_logit|
#
# 时，当前有界残差理论上才可能翻转基础判断，
# 所以只在这些样本上允许纠错。
# ============================================================


class SelectorBoundedResidualNet(nn.Module):

    def __init__(
        self,
        compact_dim,
        semantic_dim,
        coverage_dim,
        hidden,
        dropout,
        base_weight,
        base_bias,
    ):
        super().__init__()

        base_weight = torch.as_tensor(
            base_weight,
            dtype=torch.float32,
        ).reshape(
            compact_dim,
            1,
        )

        base_bias = torch.as_tensor(
            base_bias,
            dtype=torch.float32,
        ).reshape(1)

        # 冻结Stable70逻辑回归。
        self.register_buffer(
            "base_weight",
            base_weight,
        )

        self.register_buffer(
            "base_bias",
            base_bias,
        )

        self.semantic_encoder = nn.Sequential(
            nn.Linear(
                max(
                    semantic_dim,
                    1,
                ),
                hidden,
            ),
            nn.LayerNorm(
                hidden
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        selector_input_dim = (
            hidden
            + coverage_dim
            + 2
        )

        self.selector_head = nn.Sequential(
            nn.Linear(
                selector_input_dim,
                hidden,
            ),
            nn.LayerNorm(
                hidden
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                1,
            ),
        )

    def forward_detail(
        self,
        compact,
        semantic,
        coverage,
        presence,
    ):
        # ====================================================
        # 冻结Stable70基础分数
        # ====================================================

        base_logit = (
            compact
            @ self.base_weight
            + self.base_bias
        )

        base_probability = torch.sigmoid(
            base_logit
        )

        # ====================================================
        # 原来的不确定度上界
        #
        # 不放宽。
        # ====================================================

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
        # Stable15语义编码
        # ====================================================

        if semantic.shape[1] == 0:
            semantic = torch.zeros(
                (
                    semantic.shape[0],
                    1,
                ),
                dtype=compact.dtype,
                device=compact.device,
            )

        semantic_hidden = (
            self.semantic_encoder(
                semantic
            )
        )

        selector_input = torch.cat(
            [
                semantic_hidden,
                coverage,
                base_logit,
                uncertainty,
            ],
            dim=1,
        )

        selector_logit = (
            self.selector_head(
                selector_input
            )
        )

        # q:
        # 0 = 完全相信Stable70
        # 1 = 使用当前gate允许的最大纠错幅度
        selector_probability = torch.sigmoid(
            selector_logit
        )

        # ====================================================
        # 当前基础预测的反方向。
        #
        # base_logit >= 0:
        #   Stable70预测恶意
        #   挑战方向 = 负
        #
        # base_logit < 0:
        #   Stable70预测良性
        #   挑战方向 = 正
        #
        # 方向不再由网络学习。
        # ====================================================

        challenge_direction = torch.where(
            base_logit >= 0.0,
            -torch.ones_like(
                base_logit
            ),
            torch.ones_like(
                base_logit
            ),
        )

        # ====================================================
        # 可翻转门控
        #
        # 当前最大修正绝对值 = correction_gate。
        #
        # 只有：
        #
        #     gate > |base_logit|
        #
        # 才有可能真正越过0决策边界。
        #
        # 这个判断不使用标签。
        # ====================================================

        feasible = (
            (
                presence > 0.5
            )
            & (
                correction_gate
                > (
                    torch.abs(
                        base_logit
                    )
                    + 1e-6
                )
            )
        )

        feasible_float = (
            feasible.float()
        )

        # ====================================================
        # signed delta仍保持[-1,1]，
        # 方便完全兼容原evaluate_model和输出文件。
        # ====================================================

        bounded_delta = (
            feasible_float
            * challenge_direction
            * selector_probability
        )

        correction = (
            correction_gate
            * bounded_delta
        )

        final_logit = (
            base_logit
            + correction
        )

        return (
            final_logit,
            base_logit,
            correction_gate,
            bounded_delta,
            selector_logit,
            selector_probability,
            feasible_float,
        )

    def forward(
        self,
        compact,
        semantic,
        coverage,
        presence,
    ):
        (
            final_logit,
            base_logit,
            correction_gate,
            bounded_delta,
            _,
            _,
            _,
        ) = self.forward_detail(
            compact,
            semantic,
            coverage,
            presence,
        )

        # 保持原接口，
        # 原evaluate_model无需修改。
        return (
            final_logit,
            base_logit,
            correction_gate,
            bounded_delta,
        )


# ============================================================
# 平衡的“是否需要纠错”监督。
#
# 只在feasible样本上监督：
#
#   target=1:
#       Stable70当前判断错误
#
#   target=0:
#       Stable70当前判断正确
#
# 错误样本少，所以正负各占一半损失，
# 避免网络简单学成“永远不纠错”。
# ============================================================


def balanced_selector_loss(
    selector_logits,
    targets,
    feasible,
):
    logits = (
        selector_logits.reshape(-1)
    )

    targets = (
        targets.reshape(-1)
    )

    feasible = (
        feasible.reshape(-1)
        > 0.5
    )

    positive = (
        feasible
        & (
            targets > 0.5
        )
    )

    negative = (
        feasible
        & (
            targets <= 0.5
        )
    )

    losses = []

    if positive.any():
        positive_loss = (
            F.binary_cross_entropy_with_logits(
                logits[
                    positive
                ],
                torch.ones_like(
                    logits[
                        positive
                    ]
                ),
            )
        )

        losses.append(
            positive_loss
        )

    if negative.any():
        negative_loss = (
            F.binary_cross_entropy_with_logits(
                logits[
                    negative
                ],
                torch.zeros_like(
                    logits[
                        negative
                    ]
                ),
            )
        )

        losses.append(
            negative_loss
        )

    if not losses:
        return (
            selector_logits.sum()
            * 0.0
        )

    return (
        torch.stack(
            losses
        ).mean()
    )


# ============================================================
# 新fit_one
#
# 第一阶段Stable70完全不变。
#
# 第二阶段：
#
#   final分类损失
#   +
#   “基础判断是否错误”选择损失
#
# 不进行未来年份调参。
# ============================================================


def fit_one_selector(
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
):
    base.set_seed(
        seed
    )

    from run_pilot_compact_eval import (
        fit_logistic,
    )

    # ========================================================
    # 第一阶段：
    # Stable70逻辑回归。
    # ========================================================

    base_weight, base_bias = (
        fit_logistic(
            x_train,
            y_train.astype(
                int
            ),
            seed,
            iterations=6000,
            learning_rate=0.05,
            l2=0.03,
        )
    )

    model = (
        SelectorBoundedResidualNet(
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
        )
        .to(
            device
        )
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

    best_score = -float(
        "inf"
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
            selector_logits,
            selector_probability,
            feasible,
        ) = model.forward_detail(
            tx,
            ts,
            tc,
            tp,
        )

        # ====================================================
        # 监督目标：
        #
        # Stable70当前预测是否错误。
        #
        # base已经冻结，
        # target不会反向修改基础模型。
        # ====================================================

        with torch.no_grad():
            base_prediction = (
                base_logits
                >= 0.0
            ).float()

            selector_target = (
                base_prediction
                != (
                    ty >= 0.5
                )
            ).float()

        classification_loss = (
            final_loss_fn(
                logits,
                ty,
            )
        )

        selector_loss = (
            balanced_selector_loss(
                selector_logits,
                selector_target,
                feasible,
            )
        )

        # 暂时不做权重扫描。
        # 两个目标等权。
        loss = (
            classification_loss
            + selector_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        # ====================================================
        # 2017滚动验证
        # ====================================================

        model.eval()

        with torch.no_grad():
            (
                val_logits,
                val_base_logits,
                _,
                _,
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

        val_metrics = base.metrics(
            y_val.astype(
                int
            ),
            val_probability,
            threshold=base.THRESHOLD,
        )

        base_metrics = base.metrics(
            y_val.astype(
                int
            ),
            val_base_probability,
            threshold=base.THRESHOLD,
        )

        score = float(
            val_metrics[
                "balanced_accuracy"
            ]
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

            "train_loss":
                float(
                    loss.detach()
                    .cpu()
                    .item()
                ),

            "classification_loss":
                float(
                    classification_loss
                    .detach()
                    .cpu()
                    .item()
                ),

            "selector_loss":
                float(
                    selector_loss
                    .detach()
                    .cpu()
                    .item()
                ),
        })

        # ====================================================
        # 最终全2017训练：
        # 固定source-selected epoch。
        # ====================================================

        if fixed_epochs:
            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            continue

        # ====================================================
        # 仍只使用2017滚动验证BA选择epoch。
        # ====================================================

        if score > best_score + 1e-8:
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

            if patience_left <= 0:
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
# 替换原网络与训练函数。
# ============================================================

base.FrozenAnchorResidualNet = (
    SelectorBoundedResidualNet
)

base.fit_one = (
    fit_one_selector
)


# ============================================================
# 配置：
# 隐藏层、dropout、学习率等完全不重新搜索。
# ============================================================

if len(
    base.CONFIGS
) != 1:
    raise RuntimeError(
        "预期只有1个配置"
    )


base.CONFIGS[0]["name"] = (
    "stable70_stable15_selector_bounded_residual"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "stable70-stable15-selector-bounded-residual-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Stable70 + Stable15 correction-selector bounded residual"
    )

    print(
        "方向固定为Stable70当前判断的反方向；"
        "网络只学习纠错强度；"
        "只在当前gate理论可翻转的样本上允许纠错；"
        "gate上界完全不变；"
        "没有最小跨界后处理"
    )

    print(
        "============================================================"
    )

    base.main()
