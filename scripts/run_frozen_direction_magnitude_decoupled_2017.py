#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

import run_frozen_residual_base_state_ablation_2017 as exp


exp.OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-direction-magnitude-decoupled-2017"
)

# 复用原实验框架，只比较：
# 1. 当前正式模型
# 2. 方向-强度解耦模型
exp.VARIANTS = (
    {
        "name":
            "current_fusion",

        "use_base_logit":
            "current",

        "use_uncertainty":
            True,
    },
    {
        "name":
            "direction_magnitude_decoupled",

        "use_base_logit":
            "decoupled",

        "use_uncertainty":
            True,
    },
)


class DecoupledResidualNet(
    nn.Module
):
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

        self.structure_dim = 10

        relation_dim = (
            semantic_dim
            - self.structure_dim
        )

        if relation_dim <= 0:
            raise RuntimeError(
                "关系维数异常"
            )

        bw = torch.as_tensor(
            base_weight,
            dtype=torch.float32,
        ).reshape(
            compact_dim,
            1,
        )

        bb = torch.as_tensor(
            base_bias,
            dtype=torch.float32,
        ).reshape(1)

        self.register_buffer(
            "base_weight",
            bw,
        )

        self.register_buffer(
            "base_bias",
            bb,
        )

        # ====================================================
        # 150维关系 -> semantic hidden
        # 与当前正式模型保持一致
        # ====================================================

        self.semantic_encoder = nn.Sequential(
            nn.Linear(
                relation_dim,
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

        # ====================================================
        # 方向分支
        #
        # 只允许语义关系决定正负方向。
        # ====================================================

        self.direction_head = nn.Sequential(
            nn.Linear(
                hidden,
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

        # ====================================================
        # 强度分支
        #
        # 数量10
        # + coverage
        # + |base_logit|
        # + uncertainty
        #
        # 只能输出0~1，
        # 因而不能决定纠错方向。
        # ====================================================

        amplitude_dim = (
            self.structure_dim
            + coverage_dim
            + 2
        )

        self.amplitude_head = nn.Sequential(
            nn.Linear(
                amplitude_dim,
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

        # 原来的外部纠错门保持不变。
        correction_gate = (
            presence
            * uncertainty
        )

        # ====================================================
        # 两个辅助分支
        # ====================================================

        structure = semantic[
            :,
            :self.structure_dim,
        ]

        relation = semantic[
            :,
            self.structure_dim:,
        ]

        semantic_hidden = (
            self.semantic_encoder(
                relation
            )
        )

        # -------------------------
        # 方向：只由语义关系决定
        # -------------------------

        direction_raw = (
            self.direction_head(
                semantic_hidden
            )
        )

        direction = torch.tanh(
            direction_raw
        )

        # -------------------------
        # 强度：
        # 数量 + 覆盖 + 置信强度
        #
        # 不给它base_logit的正负号。
        # -------------------------

        amplitude_input = torch.cat(
            [
                structure,
                coverage,
                torch.abs(
                    base_logit
                ),
                uncertainty,
            ],
            dim=1,
        )

        amplitude = torch.sigmoid(
            self.amplitude_head(
                amplitude_input
            )
        )

        # ====================================================
        # 有界残差
        #
        # amplitude >= 0
        # 所以它不能翻转direction。
        # ====================================================

        delta = (
            amplitude
            * direction
        )

        final_logit = (
            base_logit
            + correction_gate
            * delta
        )

        return (
            final_logit,
            base_logit,
            correction_gate,
            delta,
        )


def make_variant_class(
    mode,
    use_uncertainty,
):
    if not use_uncertainty:
        raise RuntimeError(
            "本实验外部uncertainty门必须保留"
        )

    if mode == "current":

        class CurrentNet(
            exp.aug.FrozenAnchorResidual20_150Net
        ):
            pass

        return CurrentNet

    if mode == "decoupled":

        class CurrentNet(
            DecoupledResidualNet
        ):
            pass

        return CurrentNet

    raise RuntimeError(
        "未知模式: "
        + str(mode)
    )


exp.make_variant_class = (
    make_variant_class
)


if __name__ == "__main__":
    exp.main()
