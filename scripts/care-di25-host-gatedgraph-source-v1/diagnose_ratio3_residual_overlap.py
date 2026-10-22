#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


OLD_PATH = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

NEW_PATH = Path(
    "artifacts/"
    "frozen-anchor-ratio3replace-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "ratio3-residual-overlap-diagnostic"
)

THRESHOLD = 0.5


def pred(x):
    return (
        np.asarray(x, dtype=float)
        >= THRESHOLD
    ).astype(int)


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    old = pd.read_csv(
        OLD_PATH
    )

    new = pd.read_csv(
        NEW_PATH
    )

    keys = [
        "year",
        "sha256",
        "seed",
        "label",
    ]

    needed = {
        *keys,
        "probability",
        "base_probability",
        "gate",
        "semantic_delta",
    }

    for name, df in (
        ("old", old),
        ("new", new),
    ):
        missing = (
            needed
            - set(df.columns)
        )

        if missing:
            raise SystemExit(
                f"{name}缺列: "
                + str(
                    sorted(missing)
                )
            )

    merged = old.merge(
        new,
        on=keys,
        suffixes=(
            "_old",
            "_ratio3",
        ),
        validate="one_to_one",
    )

    if len(merged) != len(old):
        raise RuntimeError(
            "两份预测没有完全对应"
        )

    y = merged[
        "label"
    ].to_numpy(
        dtype=int
    )

    old_base = pred(
        merged[
            "base_probability_old"
        ]
    )

    old_final = pred(
        merged[
            "probability_old"
        ]
    )

    new_base = pred(
        merged[
            "base_probability_ratio3"
        ]
    )

    new_final = pred(
        merged[
            "probability_ratio3"
        ]
    )

    old_base_wrong = (
        old_base != y
    )

    old_base_correct = (
        old_base == y
    )

    old_final_correct = (
        old_final == y
    )

    new_base_correct = (
        new_base == y
    )

    new_base_wrong = (
        new_base != y
    )

    new_final_correct = (
        new_final == y
    )

    # ========================================================
    # Ratio3自身对原Compact的作用
    # ========================================================

    ratio3_base_fix = (
        old_base_wrong
        &
        new_base_correct
    )

    ratio3_base_hurt = (
        old_base_correct
        &
        new_base_wrong
    )

    # ========================================================
    # 原当前模型残差对原Compact的作用
    # ========================================================

    old_residual_fix = (
        old_base_wrong
        &
        old_final_correct
    )

    # ========================================================
    # 两者是不是修同一批错误
    # ========================================================

    overlap_fix = (
        ratio3_base_fix
        &
        old_residual_fix
    )

    ratio3_only_fix = (
        ratio3_base_fix
        &
        ~old_residual_fix
    )

    residual_only_fix = (
        old_residual_fix
        &
        ~ratio3_base_fix
    )

    # ========================================================
    # Ratio3已经把基础分类修好的样本，
    # 新残差有没有又改坏？
    # ========================================================

    ratio3_fix_preserved = (
        ratio3_base_fix
        &
        new_final_correct
    )

    ratio3_fix_undone = (
        ratio3_base_fix
        &
        ~new_final_correct
    )

    # Ratio3基础分类改坏的样本，
    # 新残差有没有救回来？
    ratio3_hurt_rescued = (
        ratio3_base_hurt
        &
        new_final_correct
    )

    # ========================================================
    # 保存逐记录状态
    # ========================================================

    merged[
        "ratio3_base_fix"
    ] = ratio3_base_fix.astype(int)

    merged[
        "ratio3_base_hurt"
    ] = ratio3_base_hurt.astype(int)

    merged[
        "old_residual_fix"
    ] = old_residual_fix.astype(int)

    merged[
        "overlap_fix"
    ] = overlap_fix.astype(int)

    merged[
        "ratio3_only_fix"
    ] = ratio3_only_fix.astype(int)

    merged[
        "residual_only_fix"
    ] = residual_only_fix.astype(int)

    merged[
        "ratio3_fix_preserved"
    ] = ratio3_fix_preserved.astype(int)

    merged[
        "ratio3_fix_undone"
    ] = ratio3_fix_undone.astype(int)

    merged[
        "ratio3_hurt_rescued"
    ] = ratio3_hurt_rescued.astype(int)

    merged.to_csv(
        OUT
        / "per_seed_records.csv",
        index=False,
    )

    # ========================================================
    # 正式协议对应的 year × seed 统计
    # ========================================================

    summary = []

    for (
        year,
        seed,
    ), g in merged.groupby(
        [
            "year",
            "seed",
        ],
        sort=True,
    ):

        def n(col):
            return int(
                g[
                    col
                ].sum()
            )

        ratio_fix_n = n(
            "ratio3_base_fix"
        )

        old_res_fix_n = n(
            "old_residual_fix"
        )

        overlap_n = n(
            "overlap_fix"
        )

        summary.append({
            "year":
                int(year),

            "seed":
                int(seed),

            "N":
                len(g),

            "ratio3_base_fix":
                ratio_fix_n,

            "ratio3_base_hurt":
                n(
                    "ratio3_base_hurt"
                ),

            "old_residual_fix":
                old_res_fix_n,

            "overlap_fix":
                overlap_n,

            "ratio3_only_fix":
                n(
                    "ratio3_only_fix"
                ),

            "residual_only_fix":
                n(
                    "residual_only_fix"
                ),

            "ratio3_fix_preserved":
                n(
                    "ratio3_fix_preserved"
                ),

            "ratio3_fix_undone":
                n(
                    "ratio3_fix_undone"
                ),

            "ratio3_hurt_rescued":
                n(
                    "ratio3_hurt_rescued"
                ),

            "overlap_fraction_of_ratio3":
                (
                    overlap_n
                    / ratio_fix_n
                    if ratio_fix_n
                    else np.nan
                ),

            "overlap_fraction_of_residual":
                (
                    overlap_n
                    / old_res_fix_n
                    if old_res_fix_n
                    else np.nan
                ),
        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        OUT
        / "year_seed_summary.csv",
        index=False,
    )

    # ========================================================
    # 按年份汇总
    # ========================================================

    year_rows = []

    for year, g in merged.groupby(
        "year",
        sort=True,
    ):

        def n(col):
            return int(
                g[
                    col
                ].sum()
            )

        ratio_fix_n = n(
            "ratio3_base_fix"
        )

        residual_fix_n = n(
            "old_residual_fix"
        )

        overlap_n = n(
            "overlap_fix"
        )

        year_rows.append({
            "year":
                int(year),

            "records":
                len(g),

            "ratio3_base_fix":
                ratio_fix_n,

            "ratio3_base_hurt":
                n(
                    "ratio3_base_hurt"
                ),

            "old_residual_fix":
                residual_fix_n,

            "overlap_fix":
                overlap_n,

            "ratio3_only_fix":
                n(
                    "ratio3_only_fix"
                ),

            "residual_only_fix":
                n(
                    "residual_only_fix"
                ),

            "ratio3_fix_preserved":
                n(
                    "ratio3_fix_preserved"
                ),

            "ratio3_fix_undone":
                n(
                    "ratio3_fix_undone"
                ),

            "ratio3_hurt_rescued":
                n(
                    "ratio3_hurt_rescued"
                ),

            "overlap_fraction_of_ratio3":
                (
                    overlap_n
                    / ratio_fix_n
                    if ratio_fix_n
                    else np.nan
                ),

            "overlap_fraction_of_residual":
                (
                    overlap_n
                    / residual_fix_n
                    if residual_fix_n
                    else np.nan
                ),
        })

    year_df = pd.DataFrame(
        year_rows
    )

    year_df.to_csv(
        OUT
        / "year_summary.csv",
        index=False,
    )

    # ========================================================
    # 全部未来年份
    # ========================================================

    def all_n(col):
        return int(
            merged[
                col
            ].sum()
        )

    ratio_n = all_n(
        "ratio3_base_fix"
    )

    residual_n = all_n(
        "old_residual_fix"
    )

    overlap_n = all_n(
        "overlap_fix"
    )

    print()
    print(
        "=" * 100
    )
    print(
        "按年份"
    )
    print(
        "=" * 100
    )

    print(
        year_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "全部年份"
    )
    print(
        "=" * 100
    )

    print(
        "Ratio3修复原Compact错误 =",
        ratio_n,
    )

    print(
        "Ratio3导致原本正确变错 =",
        all_n(
            "ratio3_base_hurt"
        ),
    )

    print(
        "原残差修复Compact错误 =",
        residual_n,
    )

    print(
        "两者共同修复 =",
        overlap_n,
    )

    if ratio_n:
        print(
            "Ratio3修复中与残差重合比例 =",
            f"{overlap_n / ratio_n:.4%}",
        )

    if residual_n:
        print(
            "残差修复中与Ratio3重合比例 =",
            f"{overlap_n / residual_n:.4%}",
        )

    print(
        "Ratio3独有修复 =",
        all_n(
            "ratio3_only_fix"
        ),
    )

    print(
        "原残差独有修复 =",
        all_n(
            "residual_only_fix"
        ),
    )

    print(
        "Ratio3已经修好，但新残差又改错 =",
        all_n(
            "ratio3_fix_undone"
        ),
    )

    print(
        "Ratio3改坏，但新残差救回 =",
        all_n(
            "ratio3_hurt_rescued"
        ),
    )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
