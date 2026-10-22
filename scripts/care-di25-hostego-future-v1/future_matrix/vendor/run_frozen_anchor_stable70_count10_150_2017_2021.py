#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 当前正式的“冻结Compact + count10 + 150”残差模型
import run_frozen_anchor_residual_count10_150_2017_2021 as current


base = current.base


# ============================================================
# 保存原始Compact构造
# ============================================================

original_compact_matrix_named = (
    base.compact_matrix_named
)


# ============================================================
# 3个比例替换
# ============================================================

RATIO_SPECS = (
    (
        "stable_ratio__dynamic_code_loading__node",
        "role_node__dynamic_code_loading",
        "sensitive_node_count",
    ),
    (
        "stable_ratio__native_library_loading__node",
        "role_node__native_library_loading",
        "sensitive_node_count",
    ),
    (
        "stable_ratio__native_library_loading__edge",
        "role_edge__native_library_loading",
        "sensitive_edge_count",
    ),
)


# ============================================================
# 已确认需要删除的3个时间反转角色
# node + edge，共6维
# ============================================================

DROP_FEATURES = {
    "role_node__command_execution",
    "role_edge__command_execution",

    "role_node__task_or_alarm_scheduling",
    "role_edge__task_or_alarm_scheduling",

    "role_node__icon_or_component_hiding",
    "role_edge__icon_or_component_hiding",
}


def norm_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return 0.0


def raw_ratio_values(
    rows,
    compact_by_sha,
    numerator_name,
    denominator_name,
):
    values = []

    for row in rows:
        sha = norm_sha(
            row["sha256"]
        )

        source = compact_by_sha[
            sha
        ]

        numerator = safe_float(
            source.get(
                numerator_name,
                0.0,
            )
        )

        denominator = safe_float(
            source.get(
                denominator_name,
                0.0,
            )
        )

        if denominator > 0.0:
            value = (
                numerator
                / denominator
            )
        else:
            value = 0.0

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=float,
    )


def compact_matrix_stable70(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):
    # ========================================================
    # 先完全复用原Compact76。
    # ========================================================

    (
        compact_train,
        compact_test,
        compact_names,
    ) = original_compact_matrix_named(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    compact_train = (
        compact_train.copy()
    )

    compact_test = (
        compact_test.copy()
    )

    compact_names = list(
        compact_names
    )

    if len(compact_names) != 76:
        raise RuntimeError(
            "预期原Compact=76维，实际="
            + str(len(compact_names))
        )

    # ========================================================
    # 第一步：
    # 对3个绝对计数做比例替换。
    #
    # 每个比例仍然只使用训练折统计量标准化。
    # ========================================================

    for (
        new_name,
        numerator_name,
        denominator_name,
    ) in RATIO_SPECS:

        if numerator_name not in compact_names:
            raise RuntimeError(
                "找不到比例替换特征："
                + numerator_name
            )

        column_index = compact_names.index(
            numerator_name
        )

        train_ratio = raw_ratio_values(
            train_rows,
            compact_by_sha,
            numerator_name,
            denominator_name,
        )

        test_ratio = raw_ratio_values(
            test_rows,
            compact_by_sha,
            numerator_name,
            denominator_name,
        )

        mean = float(
            np.mean(train_ratio)
        )

        std = float(
            np.std(train_ratio)
        )

        if (
            not np.isfinite(std)
            or std < 1e-12
        ):
            raise RuntimeError(
                "比例特征在当前训练折为常数："
                + new_name
            )

        compact_train[
            :,
            column_index,
        ] = (
            train_ratio
            - mean
        ) / std

        compact_test[
            :,
            column_index,
        ] = (
            test_ratio
            - mean
        ) / std

        compact_names[
            column_index
        ] = new_name

    # ========================================================
    # 第二步：
    # 删除3个时间反转角色的node/edge，共6维。
    # ========================================================

    found = {
        name
        for name in compact_names
        if name in DROP_FEATURES
    }

    if found != DROP_FEATURES:
        missing = (
            DROP_FEATURES
            - found
        )

        raise RuntimeError(
            "没有找到全部6个删除特征："
            + str(
                sorted(missing)
            )
        )

    keep_idx = [
        i
        for i, name in enumerate(
            compact_names
        )
        if name not in DROP_FEATURES
    ]

    compact_train = compact_train[
        :,
        keep_idx,
    ]

    compact_test = compact_test[
        :,
        keep_idx,
    ]

    compact_names = [
        compact_names[i]
        for i in keep_idx
    ]

    if len(compact_names) != 70:
        raise RuntimeError(
            "稳定Compact预期70维，实际="
            + str(
                len(compact_names)
            )
        )

    return (
        compact_train,
        compact_test,
        compact_names,
    )


# ============================================================
# 替换全局锚点
# ============================================================

base.compact_matrix_named = (
    compact_matrix_stable70
)


# ============================================================
# 残差网络、数量10、语义150、presence逻辑
# 全部保持原样。
# ============================================================

if len(base.CONFIGS) != 1:
    raise SystemExit(
        "预期只有1个配置，实际="
        + str(
            len(base.CONFIGS)
        )
    )


base.CONFIGS[0]["name"] = (
    "frozen_anchor_stable70_count10_150_d10"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-stable70-count10-150-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Stable70 Compact + count10 + 150 bounded residual"
    )

    print(
        "3个比例替换 + 删除3个反转角色；"
        "残差结构保持原样；没有最小跨界后处理"
    )

    print(
        "============================================================"
    )

    base.main()
