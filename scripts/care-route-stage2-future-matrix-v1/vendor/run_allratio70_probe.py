#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 直接复用当前已经确认的 Stable70 实现。
import run_frozen_anchor_stable70_count10_150_2017_2021 as stable


base = stable.base


# ============================================================
# 必须从原始 Compact76 开始，而不是从旧 Stable70 开始。
# ============================================================

original_compact_matrix_named = (
    stable.original_compact_matrix_named
)


# ============================================================
# 仍然保持当前 Stable70 的70维空间：
# 这3种语义不进入最终全局表示。
# ============================================================

DROP_FEATURES = set(
    stable.DROP_FEATURES
)


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


def compact_matrix_allratio70(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):
    # ========================================================
    # 1. 完全复用原始 Compact76。
    #    这样24维非语义全局特征的处理与原实验完全一致。
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
            "预期原始 Compact=76维，实际="
            + str(len(compact_names))
        )

    # ========================================================
    # 2. 找到所有语义节点/边特征。
    #
    # 原始应该：
    # 26类语义 * 2 = 52维。
    #
    # 去掉6维后：
    # 23类语义 * 2 = 46维。
    # ========================================================

    semantic_features = [
        name
        for name in compact_names
        if (
            name.startswith(
                "role_node__"
            )
            or name.startswith(
                "role_edge__"
            )
        )
    ]

    if len(semantic_features) != 52:
        raise RuntimeError(
            "预期原始语义特征=52维，实际="
            + str(len(semantic_features))
        )

    ratio_features = [
        name
        for name in semantic_features
        if name not in DROP_FEATURES
    ]

    if len(ratio_features) != 46:
        raise RuntimeError(
            "预期需要比例化46维，实际="
            + str(len(ratio_features))
        )

    # ========================================================
    # 3. 对剩余所有语义特征执行统一比例化。
    #
    # node:
    #
    # role_node__r
    # ----------------------
    # sensitive_node_count
    #
    # edge:
    #
    # role_edge__r
    # ----------------------
    # sensitive_edge_count
    #
    # 标准化统计量严格只从当前训练折计算。
    # ========================================================

    for numerator_name in ratio_features:

        if numerator_name.startswith(
            "role_node__"
        ):
            denominator_name = (
                "sensitive_node_count"
            )

        elif numerator_name.startswith(
            "role_edge__"
        ):
            denominator_name = (
                "sensitive_edge_count"
            )

        else:
            raise RuntimeError(
                "无法识别语义特征："
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

        # 常数列直接令标准差为1，
        # 避免除零，同时不会产生人为信息。
        if (
            not np.isfinite(std)
            or std <= 1e-12
        ):
            std = 1.0

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

    # ========================================================
    # 4. 仍然删除原 Stable70 中没有进入表示的6维。
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
            "没有找到全部6个排除特征："
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

    # ========================================================
    # 5. 最终检查。
    # ========================================================

    if len(compact_names) != 70:
        raise RuntimeError(
            "全比例 Compact 预期70维，实际="
            + str(
                len(compact_names)
            )
        )

    remaining_semantic = [
        name
        for name in compact_names
        if (
            name.startswith(
                "role_node__"
            )
            or name.startswith(
                "role_edge__"
            )
        )
    ]

    if len(remaining_semantic) != 46:
        raise RuntimeError(
            "最终语义维预期46，实际="
            + str(
                len(remaining_semantic)
            )
        )

    return (
        compact_train,
        compact_test,
        compact_names,
    )


# ============================================================
# 临时替换 Compact 构造器。
# ============================================================

base.compact_matrix_named = (
    compact_matrix_allratio70
)


if __name__ == "__main__":

    print(
        "============================================================"
    )

    print(
        "All-Ratio Stable Compact 70"
    )

    print(
        "24维全局特征 + "
        "46维统一语义组成比例"
    )

    print(
        "node / sensitive_node_count"
    )

    print(
        "edge / sensitive_edge_count"
    )

    print(
        "============================================================"
    )

    # 暂时只显示原评估程序支持的参数。
    # 在确认2017-only入口以前，不直接执行 base.main()。
    print(
        "base evaluator:"
    )

    print(
        base.__file__
    )
