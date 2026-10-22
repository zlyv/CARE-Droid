#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# 直接复用已跑通的：
#
# Stable70 + Stable15 bounded residual
#
# 唯一改变：
#   Stable70 再删除一个角色的 node + edge 两维。
#
# Stable15 / presence / coverage / residual
# 全部保持不变。
# ============================================================

import run_frozen_anchor_stable70_stable15_2017_2021 as parent


base = parent.base


ROLE = os.environ.get(
    "REDUCE_ROLE",
    "",
).strip()


if not ROLE:
    raise RuntimeError(
        "必须设置环境变量 REDUCE_ROLE"
    )


stable70_compact_matrix_named = (
    base.compact_matrix_named
)


def compact_matrix_stable68(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
):
    (
        x_train,
        x_test,
        names,
    ) = stable70_compact_matrix_named(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    x_train = np.asarray(
        x_train,
        dtype=float,
    )

    x_test = np.asarray(
        x_test,
        dtype=float,
    )

    names = list(
        names
    )

    if x_train.shape[1] != 70:
        raise RuntimeError(
            "输入必须是Stable70，实际维数="
            + str(
                x_train.shape[1]
            )
        )

    drop_names = [
        f"role_node__{ROLE}",
        f"role_edge__{ROLE}",
    ]

    missing = [
        name
        for name in drop_names
        if name not in names
    ]

    if missing:
        raise RuntimeError(
            "Stable70中找不到待删除特征: "
            + str(
                missing
            )
        )

    drop_set = set(
        drop_names
    )

    keep_indices = [
        i
        for i, name in enumerate(
            names
        )
        if name not in drop_set
    ]

    new_names = [
        names[i]
        for i in keep_indices
    ]

    new_train = (
        x_train[
            :,
            keep_indices,
        ].copy()
    )

    new_test = (
        x_test[
            :,
            keep_indices,
        ].copy()
    )

    if new_train.shape[1] != 68:
        raise RuntimeError(
            "删除后预期68维，实际="
            + str(
                new_train.shape[1]
            )
        )

    print()
    print(
        "======================================"
    )

    print(
        "Stable68 reduction"
    )

    print(
        "======================================"
    )

    print(
        "removed role =",
        ROLE,
    )

    for name in drop_names:
        print(
            "  removed:",
            name,
        )

    print(
        "compact dim =",
        len(
            new_names
        ),
    )

    return (
        new_train,
        new_test,
        new_names,
    )


# 替换全局骨干构造。
base.compact_matrix_named = (
    compact_matrix_stable68
)


base.CONFIGS[0]["name"] = (
    "frozen_anchor_stable68_"
    + ROLE
    + "_stable15_d10"
)


if __name__ == "__main__":
    print(
        "============================================================"
    )

    print(
        "Stable68 + Stable15 bounded residual"
    )

    print(
        "removed global role:",
        ROLE,
    )

    print(
        "Stable15 / presence / coverage / residual unchanged"
    )

    print(
        "============================================================"
    )

    base.main()
