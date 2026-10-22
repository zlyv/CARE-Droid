#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


import run_frozen_anchor_stable70_count10_150_2017_2021 as stable70


base = stable70.base


# 当前已经验证过的：
# Stable70 + count10 + semantic
original_build_fold_data = (
    base.build_fold_data
)


STABILITY_TABLE = Path(
    "artifacts/"
    "semantic150-temporal-label-stability/"
    "relation_summary.csv"
)


def load_stable15():
    df = pd.read_csv(
        STABILITY_TABLE
    )

    required = {
        "feature",
        "raw_flip_count",
        "source_disc_auc",
        "source_active_n",
        "future_active_min",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "稳定性统计缺列: "
            + str(sorted(missing))
        )

    selected = df[
        (df["raw_flip_count"] == 0)
        & (df["source_disc_auc"] >= 0.52)
        & (df["source_active_n"] >= 20)
        & (df["future_active_min"] >= 5)
    ].copy()

    names = selected[
        "feature"
    ].tolist()

    if len(names) != 15:
        raise RuntimeError(
            "预期Stable15为15维，实际="
            + str(len(names))
        )

    return names


STABLE15 = load_stable15()
STABLE15_SET = set(
    STABLE15
)


print()
print(
    "======================================"
)
print(
    "Stable15 relations"
)
print(
    "======================================"
)

for i, name in enumerate(
    STABLE15,
    start=1,
):
    print(
        f"{i:2d}",
        name,
    )


def build_fold_data_stable15(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    semantic_by_sha,
    names150,
):
    available = set(
        names150
    )

    missing = (
        STABLE15_SET
        - available
    )

    if missing:
        raise RuntimeError(
            "原150维中缺少Stable15关系: "
            + str(
                sorted(missing)
            )
        )

    # 保持原150维中的自然顺序。
    stable_names = [
        name
        for name in names150
        if name in STABLE15_SET
    ]

    if len(
        stable_names
    ) != 15:
        raise RuntimeError(
            "Stable15数量异常: "
            + str(
                len(stable_names)
            )
        )

    # 这里原函数已经包含：
    # count10 + semantic关系。
    #
    # 传入stable_names以后得到：
    # count10 + stable15。
    #
    # presence / coverage也会只根据
    # 这15条关系重新构造。
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

    relation_names = (
        semantic_names[
            10:
        ]
    )

    if len(
        relation_names
    ) == 0:
        raise RuntimeError(
            "当前折Stable15全部成为常数"
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
        "Stable15 fold:"
    )

    print(
        "  requested =",
        len(stable_names),
    )

    print(
        "  usable =",
        len(relation_names),
    )

    print(
        "  semantic total dim =",
        data[
            "semantic_train"
        ].shape[1],
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
    build_fold_data_stable15
)


if len(
    base.CONFIGS
) != 1:
    raise SystemExit(
        "预期只有1个配置，实际="
        + str(
            len(base.CONFIGS)
        )
    )


base.CONFIGS[0]["name"] = (
    "frozen_anchor_stable70_count10_stable15_d10"
)


if __name__ == "__main__":

    if "--output" not in sys.argv:
        sys.argv.extend([
            "--output",
            "artifacts/"
            "frozen-anchor-stable70-count10-stable15-2017-2021",
        ])

    print(
        "============================================================"
    )

    print(
        "Stable70 + count10 + Stable15 bounded residual"
    )

    print(
        "Stable15 = 零方向反转"
        " + 2017区分能力>=0.52"
        " + 源域激活>=20"
        " + 每个未来年份激活>=5"
    )

    print(
        "残差结构不变；没有最小跨界后处理"
    )

    print(
        "============================================================"
    )

    base.main()
