#!/usr/bin/env python3
from __future__ import annotations

import torch

import run_frozen_residual_base_state_ablation_2017 as exp


# ============================================================
# 只比较两个版本：
#
# 1. 当前：signed base_logit
# 2. 新版：abs(base_logit)
#
# 两者：
# - 参数量完全一样
# - uncertainty输入完全一样
# - 外部 presence*uncertainty 门完全一样
# ============================================================

exp.OUTPUT_DIR = exp.Path(
    "artifacts/"
    "frozen-residual-abs-base-logit-2017"
)

exp.VARIANTS = (
    {
        "name":
            "current_base_uncertainty",

        # 这里字符串由下面的新make_variant_class解释。
        "use_base_logit":
            "signed",

        "use_uncertainty":
            True,
    },
    {
        "name":
            "abs_base_logit_uncertainty",

        "use_base_logit":
            "absolute",

        "use_uncertainty":
            True,
    },
)


class BaseModeNet(
    exp.BaseStateAblationNet
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
        base_mode,
    ):
        # 参数结构仍与当前模型完全一致：
        # 一个base scalar + uncertainty scalar。
        super().__init__(
            compact_dim=compact_dim,
            semantic_dim=semantic_dim,
            coverage_dim=coverage_dim,
            hidden=hidden,
            dropout=dropout,
            base_weight=base_weight,
            base_bias=base_bias,
            use_base_logit=True,
            use_uncertainty=True,
        )

        if base_mode not in (
            "signed",
            "absolute",
        ):
            raise ValueError(
                "未知base_mode: "
                + str(base_mode)
            )

        self.base_mode = (
            base_mode
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

        base_probability = (
            torch.sigmoid(
                base_logit
            )
        )

        uncertainty = (
            4.0
            * base_probability
            * (
                1.0
                - base_probability
            )
        )

        # 外部门完全不改。
        correction_gate = (
            presence
            * uncertainty
        )

        # ====================================================
        # 数量10 + 语义关系
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

        # ====================================================
        # 唯一实验变量
        # ====================================================

        if (
            self.base_mode
            == "signed"
        ):
            base_state = (
                base_logit
            )

        elif (
            self.base_mode
            == "absolute"
        ):
            base_state = (
                torch.abs(
                    base_logit
                )
            )

        else:
            raise RuntimeError(
                "非法base_mode"
            )

        delta_input = (
            torch.cat(
                [
                    structure,
                    semantic_hidden,
                    coverage,
                    base_state,
                    uncertainty,
                ],
                dim=1,
            )
        )

        delta = torch.tanh(
            self.delta_head(
                delta_input
            )
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
    base_mode,
    use_uncertainty,
):
    if not use_uncertainty:
        raise RuntimeError(
            "本实验必须保留uncertainty输入"
        )

    class CurrentNet(
        BaseModeNet
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
            super().__init__(
                compact_dim=compact_dim,
                semantic_dim=semantic_dim,
                coverage_dim=coverage_dim,
                hidden=hidden,
                dropout=dropout,
                base_weight=base_weight,
                base_bias=base_bias,
                base_mode=base_mode,
            )

    return CurrentNet


# 替换原实验脚本的网络构造函数。
exp.make_variant_class = (
    make_variant_class
)


if __name__ == "__main__":
    exp.main()
