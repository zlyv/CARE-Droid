#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


# ============================================================
# 先导入当前已经验证过的正式模型。
#
# 导入后它已经完成：
#   1. count10 + 150 数据构造替换
#   2. 当前冻结Compact残差网络替换
#
# 我们这里只改 Compact 的3个输入表示。
# ============================================================

import run_frozen_anchor_residual_count10_150_2017_2021 as current


base = current.base


# ============================================================
# 保存原Compact数据构造
# ============================================================

original_compact_matrix_named = (
    base.compact_matrix_named
)


RATIO_SPECS = (
    (
        "ratio3__dynamic_code_loading__node",
        "role_node__dynamic_code_loading",
        "sensitive_node_count",
    ),
    (
        "ratio3__native_library_loading__node",
        "role_node__native_library_loading",
        "sensitive_node_count",
    ),
    (
        "ratio3__native_library_loading__edge",
        "role_edge__native_library_loading",
        "sensitive_edge_count",
    ),
)


def norm_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def safe_float(value):
    try:
        value = float(
            value
        )

        if np.isfinite(
            value
        ):
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
            row[
                "sha256"
            ]
        )

        if sha not in compact_by_sha:
            raise RuntimeError(
                "Compact缺失SHA: "
                + sha
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


def compact_matrix_ratio3_replace(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):
    """
    先完全复用原Compact构造。

    然后只把三个对应的绝对计数列替换成：

      dynamic_code_loading node / sensitive_node_count

      native_library_loading node / sensitive_node_count

      native_library_loading edge / sensitive_edge_count

    每个比例仍然只使用当前训练折均值/标准差标准化。

    总维数保持不变。
    """

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

    original_dim = len(
        compact_names
    )

    for (
        ratio_name,
        numerator_name,
        denominator_name,
    ) in RATIO_SPECS:

        if numerator_name not in compact_names:
            raise RuntimeError(
                "当前Compact折中找不到待替换特征："
                + numerator_name
            )

        column_index = (
            compact_names.index(
                numerator_name
            )
        )

        train_ratio = (
            raw_ratio_values(
                train_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        test_ratio = (
            raw_ratio_values(
                test_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        train_mean = float(
            np.mean(
                train_ratio
            )
        )

        train_std = float(
            np.std(
                train_ratio
            )
        )

        if (
            not np.isfinite(
                train_std
            )
            or train_std < 1e-12
        ):
            raise RuntimeError(
                "Ratio3在当前训练折成为常数："
                + ratio_name
            )

        compact_train[
            :,
            column_index,
        ] = (
            train_ratio
            - train_mean
        ) / train_std

        compact_test[
            :,
            column_index,
        ] = (
            test_ratio
            - train_mean
        ) / train_std

        compact_names[
            column_index
        ] = ratio_name

    if (
        compact_train.shape[
            1
        ]
        != original_dim
    ):
        raise RuntimeError(
            "替换后Compact维数发生变化"
        )

    if (
        compact_test.shape[
            1
        ]
        != original_dim
    ):
        raise RuntimeError(
            "替换后Compact测试维数发生变化"
        )

    if len(
        compact_names
    ) != original_dim:
        raise RuntimeError(
            "替换后Compact名称维数发生变化"
        )

    return (
        compact_train,
        compact_test,
        compact_names,
    )


# ============================================================
# 挂载新的Compact构造。
#
# 当前count10+150的build_fold_data内部仍会调用
# base模块中的compact_matrix_named，
# 因此这里只需要替换这一处。
# ============================================================

base.compact_matrix_named = (
    compact_matrix_ratio3_replace
)


# ============================================================
# 改实验名称。
# 所有网络超参数仍保持当前正式配置。
# ============================================================

if len(
    base.CONFIGS
) != 1:
    raise SystemExit(
        "预期只有1个正式配置，实际="
        + str(
            len(
                base.CONFIGS
            )
        )
    )


base.CONFIGS[
    0
][
    "name"
] = (
    "frozen_anchor_ratio3replace_count10_150_d10"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-ratio3replace-count10-150-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Ratio3-replaced Compact"
        " + count10 + 150"
    )

    print(
        "只替换3个Compact绝对计数；"
        "其他训练流程完全沿用当前正式模型"
    )

    print(
        "============================================================"
    )

    base.main()
