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
# 直接复用已经验证过的：
#
# Stable70
# + count10
# + 150
# + frozen bounded residual
#
# 这个导入已经完成：
#   1. Stable70 Compact 替换
#   2. count10 + semantic 数据拼接
#   3. 原残差网络替换
# ============================================================

import run_frozen_anchor_stable70_count10_150_2017_2021 as stable70


base = stable70.base


# ============================================================
# 保存当前已经验证过的 count10 + semantic 构造函数
# ============================================================

original_build_fold_data = (
    base.build_fold_data
)


# ============================================================
# 开发年份诊断得到的8条保守稳定关系
# ============================================================

STABLE8 = (
    "semrel_support__background_service_control__reflective_invocation",

    "semrel_support__background_service_control__encoding_or_compression",

    "semrel_support__background_service_control__cryptography",

    "semrel_support__background_service_control__process_or_wakelock_control",

    "semrel_support__location_access__reflective_invocation",

    "semrel_support__location_access__file_read",

    "semrel_support__location_access__url_or_http_setup",

    "semrel_support__location_access__network_transmission_or_upload",
)


STABLE8_SET = set(
    STABLE8
)


def build_fold_data_stable8(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    semantic_by_sha,
    names150,
):
    """
    核心原则：

        原程序：
            count10 + semantic150

        现在：
            count10 + stable8

    最关键的是：
        presence
        coverage

    也必须只根据stable8重新计算。

    因此不能先构造150维后再简单切片，
    而是直接把stable8_names传入原构造函数。
    """

    available = set(
        names150
    )

    missing = (
        STABLE8_SET
        - available
    )

    if missing:
        raise RuntimeError(
            "150维中缺少稳定关系："
            + str(
                sorted(
                    missing
                )
            )
        )

    # 保持原150维中的自然顺序，
    # 不人为重新排序。
    stable_names = [
        name
        for name in names150
        if name in STABLE8_SET
    ]

    if len(
        stable_names
    ) != 8:
        raise RuntimeError(
            "稳定关系数量异常："
            + str(
                len(
                    stable_names
                )
            )
        )

    # --------------------------------------------------------
    # original_build_fold_data 当前已经是：
    #
    # count10 + semantic
    #
    # 所以传入stable_names以后：
    #
    #   semantic关系部分 = stable8
    #   count10         = 不变
    #   coverage        = 按stable8重新算
    #   presence        = 按stable8重新算
    # --------------------------------------------------------

    data = original_build_fold_data(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        semantic_by_sha,
        stable_names,
    )

    semantic_names = list(
        data[
            "semantic_names"
        ]
    )

    # 前10维是count10，
    # 后面是当前折可用的stable关系。
    relation_names = (
        semantic_names[
            10:
        ]
    )

    if len(
        relation_names
    ) == 0:
        raise RuntimeError(
            "当前折stable8全部成为常数，"
            "无法训练残差"
        )

    train_presence = np.asarray(
        data[
            "presence_train"
        ],
        dtype=float,
    ).reshape(-1)

    test_presence = np.asarray(
        data[
            "presence_test"
        ],
        dtype=float,
    ).reshape(-1)

    print()
    print(
        "Stable8 fold:"
    )

    print(
        "  stable relations requested =",
        len(
            stable_names
        ),
    )

    print(
        "  stable relations usable =",
        len(
            relation_names
        ),
    )

    print(
        "  semantic total dim =",
        data[
            "semantic_train"
        ].shape[
            1
        ],
    )

    print(
        "  train presence rate =",
        f"{train_presence.mean():.4f}",
    )

    print(
        "  test presence rate =",
        f"{test_presence.mean():.4f}",
    )

    return data


# ============================================================
# 挂载稳定8维数据构造
# ============================================================

base.build_fold_data = (
    build_fold_data_stable8
)


# ============================================================
# 网络结构和超参数完全不动
# ============================================================

if len(
    base.CONFIGS
) != 1:
    raise SystemExit(
        "预期只有1个配置，实际="
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
    "frozen_anchor_stable70_count10_stable8_d10"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-stable70-count10-stable8-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Stable70 + count10 + stable8 bounded residual"
    )

    print(
        "只把semantic150缩减为开发年份诊断得到的stable8；"
        "presence和coverage同步按stable8重算；"
        "残差结构不变；没有最小跨界"
    )

    print(
        "============================================================"
    )

    base.main()
