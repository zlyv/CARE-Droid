#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 直接复用已经跑通的：
# Stable70 + count10 + Stable15 + bounded residual
import run_frozen_anchor_stable70_count10_stable15_2017_2021 as stable15


base = stable15.base


# stable15模块导入后，
# base.build_fold_data 已经是：
#
# Stable70 + count10 + Stable15
#
# 并且：
# presence / coverage 已经只按 Stable15 计算。
build_fold_data_with_count10 = (
    base.build_fold_data
)


def build_fold_data_no_count10(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    semantic_by_sha,
    names150,
):
    data = build_fold_data_with_count10(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
        semantic_by_sha,
        names150,
    )

    x_train = np.asarray(
        data["semantic_train"],
        dtype=float,
    )

    x_test = np.asarray(
        data["semantic_test"],
        dtype=float,
    )

    names = list(
        data["semantic_names"]
    )

    # 已跑通的Stable15版本应为：
    # 10维数量 + 15维语义 = 25维
    if x_train.shape[1] != 25:
        raise RuntimeError(
            "预期辅助输入为25维"
            "（count10 + Stable15），实际="
            + str(x_train.shape[1])
        )

    if len(names) != 25:
        raise RuntimeError(
            "semantic_names预期25个，实际="
            + str(len(names))
        )

    # --------------------------------------------------------
    # 唯一改变：
    # 删除前10维数量特征。
    #
    # presence / coverage 不改，
    # 因为它们原本已经只依据Stable15计算。
    # --------------------------------------------------------

    data["semantic_train"] = (
        x_train[:, 10:].copy()
    )

    data["semantic_test"] = (
        x_test[:, 10:].copy()
    )

    data["semantic_names"] = (
        names[10:]
    )

    if data["semantic_train"].shape[1] != 15:
        raise RuntimeError(
            "删除count10后预期15维，实际="
            + str(
                data["semantic_train"].shape[1]
            )
        )

    train_presence = np.asarray(
        data["presence_train"],
        dtype=float,
    ).reshape(-1)

    test_presence = np.asarray(
        data["presence_test"],
        dtype=float,
    ).reshape(-1)

    print()
    print(
        "Stable15-only fold:"
    )

    print(
        "  auxiliary dim =",
        data["semantic_train"].shape[1],
    )

    print(
        "  count10 = removed"
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


base.build_fold_data = (
    build_fold_data_no_count10
)


if len(base.CONFIGS) != 1:
    raise RuntimeError(
        "预期只有1个配置"
    )


base.CONFIGS[0]["name"] = (
    "frozen_anchor_stable70_stable15_d10"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-stable70-stable15-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Stable70 + Stable15 bounded residual"
    )

    print(
        "删除count10；"
        "Stable15、presence、coverage、残差结构全部保持不变；"
        "没有最小跨界后处理"
    )

    print(
        "============================================================"
    )

    base.main()
