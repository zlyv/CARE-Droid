#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_frozen_anchor_residual_2017_2021 as base


# ============================================================
# 固定实验定义
# ============================================================

STRUCTURE_DIM = 20

OLD20_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

OLD20_METRICS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)

OLD20_NAMES = [
    f"up_role__{role}__{metric}"
    for role in OLD20_ROLES
    for metric in OLD20_METRICS
]

if len(OLD20_NAMES) != STRUCTURE_DIM:
    raise RuntimeError(
        f"旧结构特征维数异常：{len(OLD20_NAMES)}"
    )


# ============================================================
# 安全检查：
# 必须基于已经修正为 source-selected epoch 的冻结脚本
# ============================================================

base_source = Path(base.__file__).read_text(
    encoding="utf-8"
)

if "final_epochs = max(50, final_epochs)" in base_source:
    raise SystemExit(
        "基础冻结脚本仍然强制最少50轮。\n"
        "请先确认其中已经改成：\n"
        "final_epochs = max(1, final_epochs)"
    )


# ============================================================
# 自动寻找已经存在的旧20维文件
#
# 只检查表头；只有真正包含全部20列的文件才读取。
# 不重新提取APK。
# ============================================================

_old20_by_sha = None



OLD20_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)


def load_old20_rows():
    global _old20_by_sha

    if _old20_by_sha is not None:
        return _old20_by_sha

    if not OLD20_PATH.exists():
        raise SystemExit(
            "正式旧20维文件不存在: "
            + str(OLD20_PATH)
        )

    required = {
        "sha256",
        *OLD20_NAMES,
    }

    with gzip.open(
        OLD20_PATH,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required - columns
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise SystemExit(
                "正式旧20维文件缺少列: "
                + missing_text
            )

        merged = {}

        for row in reader:
            sha = str(
                row.get(
                    "sha256",
                    "",
                )
            ).strip().casefold()

            if not sha:
                continue

            if sha in merged:
                raise SystemExit(
                    "正式旧20维文件存在重复SHA: "
                    + sha
                )

            values = np.asarray(
                [
                    float(row[name])
                    for name in OLD20_NAMES
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(values)
            ):
                raise SystemExit(
                    "旧20维存在非有限值: "
                    + sha
                )

            merged[sha] = row

    print()
    print(
        "======================================"
    )
    print(
        "正式旧20维来源"
    )
    print(
        "======================================"
    )
    print(
        OLD20_PATH
    )
    print(
        "唯一SHA =",
        len(merged),
    )
    print(
        "特征维数 =",
        len(OLD20_NAMES),
    )

    _old20_by_sha = merged

    return _old20_by_sha

# ============================================================
# 保存原来的150维数据构造函数
# ============================================================

original_build_fold_data = (
    base.build_fold_data
)


def build_fold_data_20_150(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    semantic_by_sha,
    names150,
):
    """
    先完全调用已经验证过的 Frozen+150 数据构造。

    然后只增加旧20维。

    特别注意：
        presence仍然来自原函数，
        因此仍然只由150维决定。
    """

    data = original_build_fold_data(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        semantic_by_sha,
        names150,
    )

    old20_by_sha = load_old20_rows()

    needed = {
        str(row["sha256"])
        .strip()
        .casefold()
        for row in (
            list(train_rows)
            + list(test_rows)
        )
    }

    missing = (
        needed
        - set(old20_by_sha)
    )

    if missing:
        print()
        print(
            "旧20维缺失 =",
            len(missing),
        )

        for sha in sorted(
            missing
        )[:20]:
            print(
                " ",
                sha,
            )

        raise SystemExit(
            "当前折旧20维人口不完整，"
            "停止实验。"
        )

    (
        structure_train,
        structure_test,
        structure_names,
    ) = base.standardize_named(
        train_rows,
        test_rows,
        old20_by_sha,
        OLD20_NAMES,
    )

    # 历史实验三个2017折均为20维可用。
    # 这里继续强制检查，防止悄悄改变模型定义。
    if len(structure_names) != STRUCTURE_DIM:
        raise SystemExit(
            "当前折旧20维可用维数不是20："
            f"{len(structure_names)}\n"
            f"实际列={structure_names}"
        )

    # --------------------------------------------------------
    # 为了完全复用原fit_one/evaluate_model接口，
    # 把：
    #
    #   [20维结构 | 原语义矩阵]
    #
    # 暂存在 semantic_train / semantic_test 中。
    #
    # 新网络forward中会重新拆开。
    # --------------------------------------------------------

    semantic150_train = (
        data["semantic_train"]
    )

    semantic150_test = (
        data["semantic_test"]
    )

    data["semantic_train"] = np.concatenate(
        [
            structure_train,
            semantic150_train,
        ],
        axis=1,
    )

    data["semantic_test"] = np.concatenate(
        [
            structure_test,
            semantic150_test,
        ],
        axis=1,
    )

    data["structure_names"] = list(
        structure_names
    )

    data["semantic150_names"] = list(
        data["semantic_names"]
    )

    data["semantic_names"] = (
        list(structure_names)
        + list(data["semantic_names"])
    )

    return data


# ============================================================
# 新残差网络
#
# 构造函数签名故意与 Frozen+150 保持一致，
# 因此原来的 fit_one 完全不需要改。
# ============================================================

class FrozenAnchorResidual20_150Net(
    nn.Module
):
    """
    固定 Compact
    + 20维组织结构
    + 150维语义关系
    + 覆盖信息
    + 固定不确定度残差。

    presence仍然只由150维决定。

    final_logit =
        base_logit
        + presence
        * uncertainty
        * tanh(delta)
    """

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

        if semantic_dim <= STRUCTURE_DIM:
            raise ValueError(
                "拼接后的辅助维度异常："
                f"{semantic_dim}"
            )

        self.structure_dim = (
            STRUCTURE_DIM
        )

        relation_dim = (
            semantic_dim
            - STRUCTURE_DIM
        )

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

        # 真正冻结：
        # 不进入optimizer参数。
        self.register_buffer(
            "base_weight",
            base_weight,
        )

        self.register_buffer(
            "base_bias",
            base_bias,
        )

        # 150维关系仍沿用原来的编码方式。
        self.semantic_encoder = nn.Sequential(
            nn.Linear(
                relation_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 20维不再单独加复杂编码器，
        # 直接作为组织结构证据进入残差头。
        delta_input_dim = (
            STRUCTURE_DIM
            + hidden
            + coverage_dim
            + 2
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                delta_input_dim,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
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
        # 冻结 Compact 基础分数
        # ====================================================

        base_logit = (
            compact @ self.base_weight
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

        # presence来自原150维逻辑，
        # 与20维无关。
        correction_gate = (
            presence
            * uncertainty
        )

        # ====================================================
        # 拆开：
        # 前20维 = 组织结构
        # 后面   = 当前折可用的语义关系
        # ====================================================

        structure = semantic[
            :,
            :STRUCTURE_DIM,
        ]

        relation = semantic[
            :,
            STRUCTURE_DIM:,
        ]

        if structure.shape[1] != STRUCTURE_DIM:
            raise RuntimeError(
                "structure切分维数异常"
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

        # 残差仍不能访问完整Compact。
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

        raw_delta = self.delta_head(
            delta_input
        )

        bounded_delta = torch.tanh(
            raw_delta
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
        )


# ============================================================
# 将两个新部件挂到已经跑通的冻结程序上
# ============================================================

base.build_fold_data = (
    build_fold_data_20_150
)

base.FrozenAnchorResidualNet = (
    FrozenAnchorResidual20_150Net
)


# 只改配置名称，不改任何训练参数。
if len(base.CONFIGS) != 1:
    raise SystemExit(
        "预期冻结基础脚本只有1个配置，"
        f"实际={len(base.CONFIGS)}"
    )

base.CONFIGS[0]["name"] = (
    "frozen_anchor20_150_d10"
)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 不覆盖 Frozen+150 已有结果。
    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-residual-20-150-2017-2021",
        ])

    print(
        "======================================"
    )
    print(
        "Frozen Compact + 20 + 150 residual"
    )
    print(
        "20维只增加残差内容；"
        "presence仍只由150维决定"
    )
    print(
        "======================================"
    )

    base.main()
