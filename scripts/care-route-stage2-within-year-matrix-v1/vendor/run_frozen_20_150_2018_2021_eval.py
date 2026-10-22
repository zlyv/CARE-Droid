#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)

from run_dual_view_error_attribution import (
    compact_matrix_named,
)

from run_sensitive_upstream_structural_eval import (
    label_value,
    standardize_named,
)

from run_clean_semantic_relation_2017_eval import (
    evaluate_matrix,
)

from run_pilot_compact_eval import (
    metrics,
)

from run_compact_20_150_ablation_2017 import (
    old20_names,
    support150_names,
)


ALPHA = 0.25
THRESHOLD = 0.5

SEEDS = [0, 1, 2, 3, 4]

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

EXPECTED = {
    2018: 789,
    2019: 399,
    2020: 397,
    2021: 400,
}


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

UPSTREAM_PATH = Path(
    "artifacts/"
    "sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

SEMANTIC_2017_PATH = Path(
    "artifacts/"
    "clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SEMANTIC_FUTURE_PATH = Path(
    "artifacts/"
    "clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/"
    "frozen-20-150-2018-2021"
)


def read_gz(path):
    with gzip.open(
        path,
        "rt",
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


def build_per_seed_metrics(
    prediction_rows,
):
    groups = defaultdict(list)

    for row in prediction_rows:
        groups[
            (
                int(row["year"]),
                row["variant"],
                int(row["seed"]),
            )
        ].append(row)

    output = []

    for (
        year,
        variant,
        seed,
    ), rows in sorted(groups.items()):

        y = np.asarray(
            [
                int(row["label"])
                for row in rows
            ],
            dtype=np.int64,
        )

        p = np.asarray(
            [
                float(row["probability"])
                for row in rows
            ],
            dtype=np.float64,
        )

        result = metrics(
            y,
            p,
            threshold=THRESHOLD,
        )

        output.append({
            "year":
                year,

            "variant":
                variant,

            "seed":
                seed,

            "n":
                len(rows),

            **result,
        })

    return output


def summarize_per_year(
    seed_rows,
):
    groups = defaultdict(list)

    for row in seed_rows:
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
    ), rows in sorted(groups.items()):

        ba = np.asarray(
            [
                float(
                    row[
                        "balanced_accuracy"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        output.append({
            "year":
                year,

            "variant":
                variant,

            "n":
                int(rows[0]["n"]),

            "BA":
                float(
                    np.mean([
                        float(
                            row[
                                "balanced_accuracy"
                            ]
                        )
                        for row in rows
                    ])
                ),

            "BA_std":
                float(ba.std()),

            "AUC":
                float(
                    np.mean([
                        float(row["auc"])
                        for row in rows
                    ])
                ),

            "ACC":
                float(
                    np.mean([
                        float(
                            row["accuracy"]
                        )
                        for row in rows
                    ])
                ),

            "F1":
                float(
                    np.mean([
                        float(row["f1"])
                        for row in rows
                    ])
                ),

            "FP":
                float(
                    np.mean([
                        float(row["fp"])
                        for row in rows
                    ])
                ),

            "FN":
                float(
                    np.mean([
                        float(row["fn"])
                        for row in rows
                    ])
                ),
        })

    return output


def summarize_future(
    per_year_rows,
):
    variants = sorted({
        row["variant"]
        for row in per_year_rows
    })

    output = []

    for variant in variants:
        rows = [
            row
            for row in per_year_rows
            if row["variant"] == variant
        ]

        output.append({
            "variant":
                variant,

            "mean_BA":
                float(
                    np.mean([
                        row["BA"]
                        for row in rows
                    ])
                ),

            "min_BA":
                float(
                    np.min([
                        row["BA"]
                        for row in rows
                    ])
                ),

            "mean_AUC":
                float(
                    np.mean([
                        row["AUC"]
                        for row in rows
                    ])
                ),

            "mean_ACC":
                float(
                    np.mean([
                        row["ACC"]
                        for row in rows
                    ])
                ),

            "mean_F1":
                float(
                    np.mean([
                        row["F1"]
                        for row in rows
                    ])
                ),

            "mean_FP":
                float(
                    np.mean([
                        row["FP"]
                        for row in rows
                    ])
                ),

            "mean_FN":
                float(
                    np.mean([
                        row["FN"]
                        for row in rows
                    ])
                ),
        })

    output.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return output


def correction_analysis(
    prediction_rows,
    base_variant,
    new_variant,
):
    prob = defaultdict(list)
    labels = {}

    for row in prediction_rows:
        key = (
            int(row["year"]),
            row["sha256"],
            row["variant"],
        )

        prob[key].append(
            float(row["probability"])
        )

        labels[
            (
                int(row["year"]),
                row["sha256"],
            )
        ] = int(row["label"])

    output = []

    for year in YEARS:
        corrected = 0
        regressed = 0

        benign_corrected = 0
        malware_corrected = 0

        benign_regressed = 0
        malware_regressed = 0

        shifts = []

        samples = sorted(
            sha
            for (
                current_year,
                sha,
            ) in labels
            if current_year == year
        )

        for sha in samples:
            label = labels[
                (
                    year,
                    sha,
                )
            ]

            p_base = float(
                np.mean(
                    prob[
                        (
                            year,
                            sha,
                            base_variant,
                        )
                    ]
                )
            )

            p_new = float(
                np.mean(
                    prob[
                        (
                            year,
                            sha,
                            new_variant,
                        )
                    ]
                )
            )

            base_ok = (
                int(
                    p_base
                    >= THRESHOLD
                )
                == label
            )

            new_ok = (
                int(
                    p_new
                    >= THRESHOLD
                )
                == label
            )

            if (
                not base_ok
                and new_ok
            ):
                corrected += 1

                if label == 0:
                    benign_corrected += 1
                else:
                    malware_corrected += 1

            elif (
                base_ok
                and not new_ok
            ):
                regressed += 1

                if label == 0:
                    benign_regressed += 1
                else:
                    malware_regressed += 1

            if label == 1:
                shift = (
                    p_new
                    - p_base
                )
            else:
                shift = (
                    p_base
                    - p_new
                )

            shifts.append(
                shift
            )

        shifts = np.asarray(
            shifts,
            dtype=np.float64,
        )

        output.append({
            "year":
                year,

            "base_variant":
                base_variant,

            "new_variant":
                new_variant,

            "corrected":
                corrected,

            "regressed":
                regressed,

            "net_corrected":
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

            "truth_direction_rate":
                float(
                    np.mean(
                        shifts > 0
                    )
                ),

            "mean_truth_shift":
                float(
                    shifts.mean()
                ),
        })

    return output


def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        UPSTREAM_PATH,
        SEMANTIC_2017_PATH,
        SEMANTIC_FUTURE_PATH,
        SPLIT_PATH,
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

    # ==================================
    # Compact
    # ==================================

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    # ==================================
    # 旧20维
    # ==================================

    upstream_rows = read_gz(
        UPSTREAM_PATH
    )

    upstream_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in upstream_rows
    }

    names20 = old20_names(
        upstream_rows[0].keys()
    )

    # ==================================
    # 新150维
    # ==================================

    semantic_2017_rows = read_gz(
        SEMANTIC_2017_PATH
    )

    semantic_future_rows = read_gz(
        SEMANTIC_FUTURE_PATH
    )

    names150_2017 = (
        support150_names(
            semantic_2017_rows[
                0
            ].keys()
        )
    )

    names150_future = (
        support150_names(
            semantic_future_rows[
                0
            ].keys()
        )
    )

    if (
        names150_2017
        != names150_future
    ):
        raise SystemExit(
            "2017和未来150维定义不一致"
        )

    names150 = names150_2017

    semantic_by_sha = {}

    for row in (
        semantic_2017_rows
        + semantic_future_rows
    ):
        sha = str(
            row["sha256"]
        ).strip().casefold()

        if sha in semantic_by_sha:
            raise SystemExit(
                "语义SHA重复："
                + sha
            )

        semantic_by_sha[
            sha
        ] = row

    # ==================================
    # 固定时间划分
    # ==================================

    splits = read_split_rows(
        SPLIT_PATH
    )

    split_map = {
        "train_fixed_2017__test_2018":
            2018,

        "train_fixed_2017__test_2019":
            2019,

        "train_fixed_2017__test_2020":
            2020,

        "train_fixed_2017__test_2021":
            2021,
    }

    selected = {}

    all_train = set()
    all_test = set()

    for (
        split_name,
        year,
    ) in split_map.items():

        if split_name not in splits:
            raise SystemExit(
                "缺少划分："
                + split_name
            )

        parts = splits[
            split_name
        ]

        train_rows = parts[
            "train"
        ]

        test_rows = parts[
            "test"
        ]

        if len(train_rows) != 799:
            raise SystemExit(
                f"{year}: "
                f"train不是799，"
                f"实际={len(train_rows)}"
            )

        if (
            len(test_rows)
            != EXPECTED[year]
        ):
            raise SystemExit(
                f"{year}: "
                f"测试数量错误，"
                f"实际={len(test_rows)}"
            )

        selected[
            split_name
        ] = parts

        all_train.update(
            row["sha256"]
            for row in train_rows
        )

        all_test.update(
            row["sha256"]
            for row in test_rows
        )

    if len(all_train) != 799:
        raise SystemExit(
            "2017唯一训练SHA不是799"
        )

    if len(all_test) != 1985:
        raise SystemExit(
            "未来唯一测试SHA不是1985"
        )

    needed = (
        all_train
        | all_test
    )

    missing_compact = (
        needed
        - set(compact_by_sha)
    )

    missing20 = (
        needed
        - set(upstream_by_sha)
    )

    missing150 = (
        needed
        - set(semantic_by_sha)
    )

    print()
    print(
        "======================================"
    )
    print(
        "人口检查"
    )
    print(
        "======================================"
    )

    print(
        "2017唯一训练SHA =",
        len(all_train),
    )

    print(
        "未来唯一测试SHA =",
        len(all_test),
    )

    print(
        "Compact缺失 =",
        len(missing_compact),
    )

    print(
        "旧20维缺失 =",
        len(missing20),
    )

    print(
        "新150维缺失 =",
        len(missing150),
    )

    if missing_compact:
        raise SystemExit(
            "Compact人口不完整"
        )

    if missing20:
        raise SystemExit(
            "旧20维人口不完整"
        )

    if missing150:
        raise SystemExit(
            "新150维人口不完整"
        )

    print()
    print(
        "======================================"
    )
    print(
        "冻结20+150时间迁移评价"
    )
    print(
        "======================================"
    )

    print(
        "20维 =",
        len(names20),
    )

    print(
        "150维 =",
        len(names150),
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    print(
        "iterations =",
        ITERATIONS,
    )

    print(
        "L2 =",
        L2,
    )

    prediction_rows = []
    dimension_rows = []

    # ==================================
    # 四个年份
    # ==================================

    for (
        split_name,
        year,
    ) in sorted(
        split_map.items(),
        key=lambda item:
            item[1],
    ):

        train_rows = (
            selected[
                split_name
            ]["train"]
        )

        test_rows = (
            selected[
                split_name
            ]["test"]
        )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        (
            compact_train,
            compact_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            schema,
        )

        (
            old20_train,
            old20_test,
            old20_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            upstream_by_sha,
            names20,
        )

        (
            semantic_train,
            semantic_test,
            semantic_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        x20_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * old20_train,
            ],
            axis=1,
        )

        x20_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * old20_test,
            ],
            axis=1,
        )

        x150_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        x150_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        xboth_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * old20_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        xboth_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * old20_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        dimension_rows.append({
            "year":
                year,

            "train_n":
                len(train_rows),

            "test_n":
                len(test_rows),

            "compact_dim":
                len(compact_names),

            "old20_usable_dim":
                len(old20_usable),

            "semantic_usable_dim":
                len(semantic_usable),

            "combined_dim":
                xboth_train.shape[1],
        })

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus20",
                x20_train,
                x20_test,
            ),

            (
                "compact_plus150",
                x150_train,
                x150_test,
            ),

            (
                "compact_plus20_plus150",
                xboth_train,
                xboth_test,
            ),
        )

        print()
        print(
            f"{year}: "
            f"train={len(train_rows)} "
            f"test={len(test_rows)} "
            f"Compact={len(compact_names)} "
            f"20可用={len(old20_usable)} "
            f"150可用={len(semantic_usable)}"
        )

        # evaluate_matrix会把每个seed的
        # 逐样本概率写入prediction_rows。
        for (
            variant,
            x_train,
            x_test,
        ) in variants:

            before = len(
                prediction_rows
            )

            evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                y_train,
                y_test,
                x_train,
                x_test,
                variant,
                SEEDS,
                ITERATIONS,
                LEARNING_RATE,
                L2,
                prediction_rows,
            )

            # 给刚写入的预测附上年份。
            for row in prediction_rows[
                before:
            ]:
                row["year"] = year

        print(
            f"{year} 完成",
            flush=True,
        )

    # ==================================
    # 汇总
    # ==================================

    seed_rows = (
        build_per_seed_metrics(
            prediction_rows
        )
    )

    per_year_rows = (
        summarize_per_year(
            seed_rows
        )
    )

    future_rows = (
        summarize_future(
            per_year_rows
        )
    )

    correction_vs150 = (
        correction_analysis(
            prediction_rows,
            "compact_plus150",
            "compact_plus20_plus150",
        )
    )

    correction_vs_compact = (
        correction_analysis(
            prediction_rows,
            "compact_only",
            "compact_plus20_plus150",
        )
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        seed_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_year_summary.csv",
        per_year_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "future_mean_summary.csv",
        future_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "correction_vs150.csv",
        correction_vs150,
    )

    write_csv(
        OUTPUT_DIR
        / "correction_vs_compact.csv",
        correction_vs_compact,
    )

    write_csv(
        OUTPUT_DIR
        / "dimensions.csv",
        dimension_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "predictions.csv",
        prediction_rows,
    )

    # ==================================
    # 打印
    # ==================================

    lookup = {
        (
            row["year"],
            row["variant"],
        ):
            row
        for row in per_year_rows
    }

    order = (
        "compact_only",
        "compact_plus20",
        "compact_plus150",
        "compact_plus20_plus150",
    )

    print()
    print(
        "======================================"
    )
    print(
        "逐年结果"
    )
    print(
        "======================================"
    )

    for year in YEARS:
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
                f"{variant:28s} "
                f"BA={row['BA']:.6f} "
                f"AUC={row['AUC']:.6f} "
                f"ACC={row['ACC']:.6f} "
                f"FP={row['FP']:.2f} "
                f"FN={row['FN']:.2f}"
            )

        r150 = lookup[
            (
                year,
                "compact_plus150",
            )
        ]

        rboth = lookup[
            (
                year,
                "compact_plus20_plus150",
            )
        ]

        print(
            "20+150相对150："
            f"dBA="
            f"{rboth['BA'] - r150['BA']:+.6f} "
            f"dAUC="
            f"{rboth['AUC'] - r150['AUC']:+.6f} "
            f"dACC="
            f"{rboth['ACC'] - r150['ACC']:+.6f} "
            f"dFP="
            f"{rboth['FP'] - r150['FP']:+.2f} "
            f"dFN="
            f"{rboth['FN'] - r150['FN']:+.2f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2018-2021四年平均"
    )
    print(
        "======================================"
    )

    for row in future_rows:
        print(
            f"{row['variant']:28s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    future_lookup = {
        row["variant"]:
            row
        for row in future_rows
    }

    r150 = future_lookup[
        "compact_plus150"
    ]

    rboth = future_lookup[
        "compact_plus20_plus150"
    ]

    print()
    print(
        "四年平均20+150相对150："
        f"dBA="
        f"{rboth['mean_BA'] - r150['mean_BA']:+.6f} "
        f"dMin="
        f"{rboth['min_BA'] - r150['min_BA']:+.6f} "
        f"dAUC="
        f"{rboth['mean_AUC'] - r150['mean_AUC']:+.6f} "
        f"dACC="
        f"{rboth['mean_ACC'] - r150['mean_ACC']:+.6f}"
    )

    print()
    print(
        "======================================"
    )
    print(
        "20+150相对150逐样本"
    )
    print(
        "======================================"
    )

    for row in correction_vs150:
        print(
            f"{row['year']} "
            f"修正={row['corrected']} "
            f"退化={row['regressed']} "
            f"净修正="
            f"{row['net_corrected']:+d} "
            f"良性修正="
            f"{row['benign_corrected']} "
            f"恶意修正="
            f"{row['malware_corrected']} "
            f"向真实方向="
            f"{row['truth_direction_rate']:.2%}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "20+150相对Compact逐样本"
    )
    print(
        "======================================"
    )

    for row in correction_vs_compact:
        print(
            f"{row['year']} "
            f"修正={row['corrected']} "
            f"退化={row['regressed']} "
            f"净修正="
            f"{row['net_corrected']:+d} "
            f"良性修正="
            f"{row['benign_corrected']} "
            f"恶意修正="
            f"{row['malware_corrected']}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
