#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pilot_compact_eval import metrics


THRESHOLD = 0.5

GATED_PATH = Path(
    "artifacts/gated-residual-neural-2017-2021/"
    "future_predictions.csv"
)

LINEAR_PATH = Path(
    "artifacts/frozen-20-150-2018-2021/"
    "predictions.csv"
)

OUTPUT_DIR = Path(
    "artifacts/gated-basehead-attribution"
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def correct(p, y):
    return int(
        int(
            p >= THRESHOLD
        ) == int(y)
    )


def build_prediction_table():
    gated_rows = read_csv(
        GATED_PATH
    )

    linear_rows = read_csv(
        LINEAR_PATH
    )

    # ------------------------------------
    # 门控网络：
    # final / internal base
    # ------------------------------------

    gated = {}

    for row in gated_rows:
        key = (
            int(row["year"]),
            row["sha256"]
                .strip()
                .casefold(),
            int(row["seed"]),
        )

        gated[key] = {
            "label":
                int(row["label"]),

            "gated_final":
                float(
                    row["probability"]
                ),

            "gated_base":
                float(
                    row["base_probability"]
                ),

            "gate":
                float(
                    row["gate"]
                ),

            "delta":
                float(
                    row["semantic_delta"]
                ),
        }

    # ------------------------------------
    # 旧线性模型
    # ------------------------------------

    wanted = {
        "compact_only",
        "compact_plus20_plus150",
    }

    linear = defaultdict(dict)

    for row in linear_rows:
        variant = row[
            "variant"
        ]

        if variant not in wanted:
            continue

        key = (
            int(row["year"]),
            row["sha256"]
                .strip()
                .casefold(),
            int(row["seed"]),
        )

        linear[key][variant] = {
            "label":
                int(row["label"]),

            "probability":
                float(
                    row["probability"]
                ),
        }

    common = sorted(
        set(gated)
        & set(linear)
    )

    output = []

    for key in common:
        year, sha, seed = key

        if not wanted.issubset(
            linear[key]
        ):
            continue

        label = gated[
            key
        ]["label"]

        for variant in wanted:
            if (
                linear[key][variant][
                    "label"
                ]
                != label
            ):
                raise RuntimeError(
                    "标签不一致："
                    + str(key)
                )

        output.append({
            "year":
                year,

            "sha256":
                sha,

            "seed":
                seed,

            "label":
                label,

            "compact_linear":
                linear[
                    key
                ][
                    "compact_only"
                ][
                    "probability"
                ],

            "linear20_150":
                linear[
                    key
                ][
                    "compact_plus20_plus150"
                ][
                    "probability"
                ],

            "gated_base":
                gated[
                    key
                ][
                    "gated_base"
                ],

            "gated_final":
                gated[
                    key
                ][
                    "gated_final"
                ],

            "gate":
                gated[
                    key
                ][
                    "gate"
                ],

            "delta":
                gated[
                    key
                ][
                    "delta"
                ],
        })

    return output


def metric_rows(
    prediction_rows,
):
    variants = (
        "compact_linear",
        "linear20_150",
        "gated_base",
        "gated_final",
    )

    output = []

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        seeds = sorted({
            int(row["seed"])
            for row in prediction_rows
            if int(row["year"])
            == year
        })

        for seed in seeds:
            current = [
                row
                for row in prediction_rows
                if (
                    int(row["year"])
                    == year
                    and
                    int(row["seed"])
                    == seed
                )
            ]

            y = np.asarray(
                [
                    int(row["label"])
                    for row in current
                ],
                dtype=np.int64,
            )

            for variant in variants:
                p = np.asarray(
                    [
                        float(
                            row[variant]
                        )
                        for row in current
                    ],
                    dtype=float,
                )

                result = metrics(
                    y,
                    p,
                    threshold=THRESHOLD,
                )

                output.append({
                    "year":
                        year,

                    "seed":
                        seed,

                    "variant":
                        variant,

                    "n":
                        len(current),

                    **result,
                })

    return output


def summarize_per_year(
    rows,
):
    groups = defaultdict(list)

    for row in rows:
        groups[
            (
                int(row["year"]),
                row["variant"],
            )
        ].append(row)

    output = []

    for (
        year,
        variant,
    ), current in sorted(
        groups.items()
    ):
        output.append({
            "year":
                year,

            "variant":
                variant,

            "seed_n":
                len(current),

            "BA":
                float(
                    np.mean([
                        float(
                            row[
                                "balanced_accuracy"
                            ]
                        )
                        for row in current
                    ])
                ),

            "AUC":
                float(
                    np.mean([
                        float(row["auc"])
                        for row in current
                    ])
                ),

            "ACC":
                float(
                    np.mean([
                        float(
                            row["accuracy"]
                        )
                        for row in current
                    ])
                ),

            "F1":
                float(
                    np.mean([
                        float(row["f1"])
                        for row in current
                    ])
                ),

            "FP":
                float(
                    np.mean([
                        float(row["fp"])
                        for row in current
                    ])
                ),

            "FN":
                float(
                    np.mean([
                        float(row["fn"])
                        for row in current
                    ])
                ),
        })

    return output


def summarize_future(
    per_year,
):
    variants = sorted({
        row["variant"]
        for row in per_year
    })

    output = []

    for variant in variants:
        current = [
            row
            for row in per_year
            if row["variant"]
            == variant
        ]

        output.append({
            "variant":
                variant,

            "mean_BA":
                float(
                    np.mean([
                        row["BA"]
                        for row in current
                    ])
                ),

            "min_BA":
                float(
                    np.min([
                        row["BA"]
                        for row in current
                    ])
                ),

            "mean_AUC":
                float(
                    np.mean([
                        row["AUC"]
                        for row in current
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        row["ACC"]
                        for row in current
                    ])
                ),

            "mean_FP":
                float(
                    np.mean([
                        row["FP"]
                        for row in current
                    ])
                ),

            "mean_FN":
                float(
                    np.mean([
                        row["FN"]
                        for row in current
                    ])
                ),
        })

    output.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
        ),
        reverse=True,
    )

    return output


def correction_rows(
    prediction_rows,
):
    """
    所有统计都保持逐种子口径，
    最后再对5个种子求平均。
    """

    pairs = (
        (
            "compact_linear",
            "gated_base",
            "joint_base_vs_compact",
        ),
        (
            "gated_base",
            "gated_final",
            "residual_vs_joint_base",
        ),
        (
            "linear20_150",
            "gated_final",
            "final_vs_linear20_150",
        ),
    )

    per_seed = []

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        seeds = sorted({
            int(row["seed"])
            for row in prediction_rows
            if int(row["year"])
            == year
        })

        for seed in seeds:
            current = [
                row
                for row in prediction_rows
                if (
                    int(row["year"])
                    == year
                    and
                    int(row["seed"])
                    == seed
                )
            ]

            for (
                old_name,
                new_name,
                comparison,
            ) in pairs:

                corrected = 0
                regressed = 0

                benign_corrected = 0
                malware_corrected = 0

                benign_regressed = 0
                malware_regressed = 0

                for row in current:
                    y = int(
                        row["label"]
                    )

                    old_ok = correct(
                        float(
                            row[old_name]
                        ),
                        y,
                    )

                    new_ok = correct(
                        float(
                            row[new_name]
                        ),
                        y,
                    )

                    if (
                        not old_ok
                        and new_ok
                    ):
                        corrected += 1

                        if y == 0:
                            benign_corrected += 1
                        else:
                            malware_corrected += 1

                    elif (
                        old_ok
                        and not new_ok
                    ):
                        regressed += 1

                        if y == 0:
                            benign_regressed += 1
                        else:
                            malware_regressed += 1

                per_seed.append({
                    "year":
                        year,

                    "seed":
                        seed,

                    "comparison":
                        comparison,

                    "corrected":
                        corrected,

                    "regressed":
                        regressed,

                    "net":
                        corrected
                        - regressed,

                    "benign_corrected":
                        benign_corrected,

                    "malware_corrected":
                        malware_corrected,

                    "benign_regressed":
                        benign_regressed,

                    "malware_regressed":
                        malware_regressed,
                })

    groups = defaultdict(list)

    for row in per_seed:
        groups[
            (
                int(row["year"]),
                row["comparison"],
            )
        ].append(row)

    summary = []

    for (
        year,
        comparison,
    ), current in sorted(
        groups.items()
    ):

        out = {
            "year":
                year,

            "comparison":
                comparison,
        }

        for field in (
            "corrected",
            "regressed",
            "net",
            "benign_corrected",
            "malware_corrected",
            "benign_regressed",
            "malware_regressed",
        ):
            out[field] = float(
                np.mean([
                    float(row[field])
                    for row in current
                ])
            )

        summary.append(out)

    return (
        per_seed,
        summary,
    )


def main():
    for path in (
        GATED_PATH,
        LINEAR_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = (
        build_prediction_table()
    )

    print(
        "匹配的逐种子预测记录 =",
        len(predictions),
    )

    if len(predictions) != 9925:
        print(
            "注意：预期约9925条，"
            "请检查人口是否一致"
        )

    seed_metrics = metric_rows(
        predictions
    )

    per_year = summarize_per_year(
        seed_metrics
    )

    future = summarize_future(
        per_year
    )

    (
        correction_seed,
        correction_summary,
    ) = correction_rows(
        predictions
    )

    write_csv(
        OUTPUT_DIR
        / "prediction_table.csv",
        predictions,
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        seed_metrics,
    )

    write_csv(
        OUTPUT_DIR
        / "per_year_metrics.csv",
        per_year,
    )

    write_csv(
        OUTPUT_DIR
        / "future_summary.csv",
        future,
    )

    write_csv(
        OUTPUT_DIR
        / "correction_per_seed.csv",
        correction_seed,
    )

    write_csv(
        OUTPUT_DIR
        / "correction_summary.csv",
        correction_summary,
    )

    lookup = {
        (
            int(row["year"]),
            row["variant"],
        ):
            row
        for row in per_year
    }

    order = (
        "compact_linear",
        "gated_base",
        "linear20_150",
        "gated_final",
    )

    print()
    print(
        "======================================"
    )
    print(
        "逐年分类性能"
    )
    print(
        "======================================"
    )

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        print()
        print(
            "----------",
            year,
            "----------",
        )

        for variant in order:
            row = lookup[
                (
                    year,
                    variant,
                )
            ]

            print(
                f"{variant:20s} "
                f"BA={row['BA']:.6f} "
                f"AUC={row['AUC']:.6f} "
                f"ACC={row['ACC']:.6f} "
                f"FP={row['FP']:.2f} "
                f"FN={row['FN']:.2f}"
            )

        base = lookup[
            (
                year,
                "gated_base",
            )
        ]

        final = lookup[
            (
                year,
                "gated_final",
            )
        ]

        compact = lookup[
            (
                year,
                "compact_linear",
            )
        ]

        print(
            "联合基础分支相对原Compact："
            f"dBA="
            f"{base['BA'] - compact['BA']:+.6f} "
            f"dAUC="
            f"{base['AUC'] - compact['AUC']:+.6f}"
        )

        print(
            "残差相对联合基础分支："
            f"dBA="
            f"{final['BA'] - base['BA']:+.6f} "
            f"dAUC="
            f"{final['AUC'] - base['AUC']:+.6f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "四年平均"
    )
    print(
        "======================================"
    )

    for row in future:
        print(
            f"{row['variant']:20s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐样本净修正：每种子统计后平均"
    )
    print(
        "======================================"
    )

    for row in correction_summary:
        print(
            f"{row['year']} "
            f"{row['comparison']:25s} "
            f"修正={row['corrected']:.2f} "
            f"退化={row['regressed']:.2f} "
            f"净={row['net']:+.2f} "
            f"良性修正={row['benign_corrected']:.2f} "
            f"恶意修正={row['malware_corrected']:.2f}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
