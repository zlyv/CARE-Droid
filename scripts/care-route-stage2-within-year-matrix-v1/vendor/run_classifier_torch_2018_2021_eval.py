#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


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

from run_pilot_compact_eval import (
    fit_logistic,
    sigmoid,
    metrics,
)

from run_classifier_torch_2017_eval import (
    ALPHA,
    THRESHOLD,
    SEEDS,
    LR_ITERATIONS,
    LR_RATE,
    LR_L2,
    fit_compact_net,
    fit_two_branch_net,
    predict_compact_net,
    predict_two_branch_net,
    support150_names,
    DEVICE,
)


YEARS = (2018, 2019, 2020, 2021)

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

SEMANTIC_2017_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SEMANTIC_FUTURE_PATH = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/classifier-torch-2018-2021"
)


def read_gz_csv(path):
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


def mean_metrics(rows):
    result = {}

    for field in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "fp",
        "fn",
    ):
        result[field] = float(
            np.mean([
                float(row[field])
                for row in rows
            ])
        )

    return result


def summarize(metric_rows):
    grouped = defaultdict(list)

    for row in metric_rows:
        grouped[
            (
                row["year"],
                row["variant"],
            )
        ].append(row)

    per_year = []

    for (
        year,
        variant,
    ), rows in sorted(grouped.items()):

        avg = mean_metrics(rows)

        bas = np.asarray(
            [
                float(
                    row["balanced_accuracy"]
                )
                for row in rows
            ],
            dtype=float,
        )

        per_year.append({
            "year":
                year,

            "variant":
                variant,

            "n":
                rows[0]["n"],

            "seed_count":
                len(rows),

            "BA":
                avg[
                    "balanced_accuracy"
                ],

            "BA_std":
                float(
                    bas.std()
                ),

            "AUC":
                avg["auc"],

            "ACC":
                avg["accuracy"],

            "F1":
                avg["f1"],

            "FP":
                avg["fp"],

            "FN":
                avg["fn"],
        })

    variants = sorted({
        row["variant"]
        for row in per_year
    })

    future_mean = []

    for variant in variants:
        rows = [
            row
            for row in per_year
            if row["variant"]
            == variant
        ]

        future_mean.append({
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

    future_mean.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return per_year, future_mean


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    semantic_2017_rows = read_gz_csv(
        SEMANTIC_2017_PATH
    )

    semantic_future_rows = read_gz_csv(
        SEMANTIC_FUTURE_PATH
    )

    names_2017 = support150_names(
        semantic_2017_rows[0].keys()
    )

    names_future = support150_names(
        semantic_future_rows[0].keys()
    )

    if names_2017 != names_future:
        raise SystemExit(
            "2017和未来150维定义不一致"
        )

    names150 = names_2017

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
                "语义SHA重复：" + sha
            )

        semantic_by_sha[sha] = row

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

    for split_name, year in (
        split_map.items()
    ):
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
                f"test数量错误，"
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

    missing_semantic = (
        needed
        - set(semantic_by_sha)
    )

    if missing_compact:
        raise SystemExit(
            "Compact缺少样本："
            + str(
                len(missing_compact)
            )
        )

    if missing_semantic:
        raise SystemExit(
            "语义缺少样本："
            + str(
                len(missing_semantic)
            )
        )

    print(
        "======================================"
    )
    print(
        "固定2017训练 -> 2018-2021"
    )
    print(
        "逻辑回归 vs 小网络"
    )
    print(
        "======================================"
    )

    print(
        "2017唯一训练SHA =",
        len(all_train),
    )

    print(
        "未来测试SHA =",
        len(all_test),
    )

    print(
        "设备 =",
        DEVICE,
    )

    if torch.cuda.is_available():
        print(
            "GPU =",
            torch.cuda.get_device_name(
                DEVICE
            ),
        )

    print(
        "冻结150维 =",
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
        "seeds =",
        SEEDS,
    )

    metric_rows = []
    training_rows = []

    for split_name, year in sorted(
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
            semantic_train,
            semantic_test,
            semantic_names,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        semantic_train = (
            ALPHA
            * semantic_train
        )

        semantic_test = (
            ALPHA
            * semantic_test
        )

        joint_train = np.concatenate(
            [
                compact_train,
                semantic_train,
            ],
            axis=1,
        )

        joint_test = np.concatenate(
            [
                compact_test,
                semantic_test,
            ],
            axis=1,
        )

        print()
        print(
            f"{year}: "
            f"train={len(train_rows)} "
            f"test={len(test_rows)} "
            f"Compact={len(compact_names)} "
            f"语义可用={len(semantic_names)}"
        )

        for seed in SEEDS:

            # -----------------------------
            # 逻辑回归 Compact
            # -----------------------------
            w, b = fit_logistic(
                compact_train,
                y_train,
                seed,
                iterations=
                    LR_ITERATIONS,
                learning_rate=
                    LR_RATE,
                l2=LR_L2,
            )

            p = sigmoid(
                compact_test @ w + b
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "year":
                    year,

                "variant":
                    "lr_compact",

                "seed":
                    seed,

                "n":
                    len(test_rows),

                **result,
            })

            # -----------------------------
            # 逻辑回归 Compact +150
            # -----------------------------
            w, b = fit_logistic(
                joint_train,
                y_train,
                seed,
                iterations=
                    LR_ITERATIONS,
                learning_rate=
                    LR_RATE,
                l2=LR_L2,
            )

            p = sigmoid(
                joint_test @ w + b
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "year":
                    year,

                "variant":
                    "lr_compact_plus150",

                "seed":
                    seed,

                "n":
                    len(test_rows),

                **result,
            })

            # -----------------------------
            # 小网络 Compact
            # -----------------------------
            (
                model,
                best_epoch,
                best_loss,
            ) = fit_compact_net(
                compact_train,
                y_train,
                seed,
            )

            p = predict_compact_net(
                model,
                compact_test,
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "year":
                    year,

                "variant":
                    "nn_compact",

                "seed":
                    seed,

                "n":
                    len(test_rows),

                **result,
            })

            training_rows.append({
                "year":
                    year,

                "variant":
                    "nn_compact",

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_validation_loss":
                    best_loss,
            })

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # -----------------------------
            # 两分支 Compact +150
            # -----------------------------
            (
                model,
                best_epoch,
                best_loss,
            ) = fit_two_branch_net(
                compact_train,
                semantic_train,
                y_train,
                seed,
            )

            p = (
                predict_two_branch_net(
                    model,
                    compact_test,
                    semantic_test,
                )
            )

            result = metrics(
                y_test,
                p,
                threshold=THRESHOLD,
            )

            metric_rows.append({
                "year":
                    year,

                "variant":
                    "nn_two_branch_plus150",

                "seed":
                    seed,

                "n":
                    len(test_rows),

                **result,
            })

            training_rows.append({
                "year":
                    year,

                "variant":
                    "nn_two_branch_plus150",

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_validation_loss":
                    best_loss,
            })

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                " seed",
                seed,
                "完成",
                flush=True,
            )

    per_year, future_mean = (
        summarize(
            metric_rows
        )
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_year_summary.csv",
        per_year,
    )

    write_csv(
        OUTPUT_DIR
        / "future_mean_summary.csv",
        future_mean,
    )

    write_csv(
        OUTPUT_DIR
        / "training_diagnostics.csv",
        training_rows,
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

    by_year = defaultdict(list)

    for row in per_year:
        by_year[
            row["year"]
        ].append(row)

    order = (
        "lr_compact",
        "lr_compact_plus150",
        "nn_compact",
        "nn_two_branch_plus150",
    )

    for year in YEARS:
        print()
        print(
            "----------",
            year,
            "----------",
        )

        lookup = {
            row["variant"]:
                row
            for row in by_year[
                year
            ]
        }

        for variant in order:
            row = lookup[
                variant
            ]

            print(
                f"{variant:28s} "
                f"BA={row['BA']:.6f} "
                f"AUC={row['AUC']:.6f} "
                f"ACC={row['ACC']:.6f} "
                f"FP={row['FP']:.2f} "
                f"FN={row['FN']:.2f}"
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

    for row in future_mean:
        print(
            f"{row['variant']:28s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print()
    print(
        "保存目录：",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
