#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows
from run_dual_view_error_attribution import compact_matrix_named
from run_pilot_compact_eval import fit_logistic, sigmoid, metrics
from run_sensitive_upstream_structural_eval import label_value, standardize_named

ALPHA = 0.25
THRESHOLD = 0.5

ANCHORS = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


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

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def support150(header):
    names = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(names) != 155:
        raise SystemExit(
            f"support列不是155维：{len(names)}"
        )

    selected = []

    for name in names:
        _, anchor, context = name.split("__")

        if anchor not in ANCHORS:
            raise SystemExit(
                f"异常锚点：{anchor}"
            )

        if context != anchor:
            selected.append(name)

    if len(selected) != 150:
        raise SystemExit(
            f"去掉自身关系后不是150维："
            f"{len(selected)}"
        )

    return selected


def zero_scale(
    train_rows,
    test_rows,
    by_sha,
    names,
):
    def matrix(rows):
        return np.asarray(
            [
                [
                    float(
                        by_sha[
                            row["sha256"]
                        ][name]
                    )
                    for name in names
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

    train = matrix(train_rows)
    test = matrix(test_rows)

    if (
        not np.all(np.isfinite(train))
        or not np.all(np.isfinite(test))
    ):
        raise SystemExit(
            "语义特征存在非有限值"
        )

    scales = train.std(axis=0)

    usable = scales > 1e-12

    if not np.any(usable):
        raise SystemExit(
            "语义特征全部为常数"
        )

    kept = [
        name
        for name, keep
        in zip(names, usable)
        if keep
    ]

    return (
        train[:, usable]
        / scales[usable],

        test[:, usable]
        / scales[usable],

        kept,
    )


def fit_semantic_correction(
    base_logits,
    y,
    x,
    iterations,
    learning_rate,
    l2,
):
    # 没有额外偏置。
    # 因此 x=0 时修正严格为0。
    weights = np.zeros(
        x.shape[1],
        dtype=np.float64,
    )

    n = float(len(y))

    for _ in range(iterations):
        probability = sigmoid(
            base_logits
            + x @ weights
        )

        error = probability - y

        grad = (
            x.T @ error
        ) / n + l2 * weights

        weights -= (
            learning_rate * grad
        )

    return weights


def summarize(metric_rows):
    grouped = defaultdict(list)

    for row in metric_rows:
        grouped[
            (
                row["variant"],
                row["split_name"],
            )
        ].append(row)

    per_split = []

    for (
        variant,
        split_name,
    ), rows in sorted(
        grouped.items()
    ):
        out = {
            "variant": variant,
            "split_name": split_name,
            "seed_count": len(rows),
        }

        for field in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "fp",
            "fn",
        ):
            out[
                field + "_mean"
            ] = float(
                np.mean(
                    [
                        float(row[field])
                        for row in rows
                    ]
                )
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, rows in (
        by_variant.items()
    ):
        ba = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in rows
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(ba.mean()),

            "min_BA":
                float(ba.min()),

            "std_BA":
                float(ba.std()),

            "mean_AUC":
                float(
                    np.mean(
                        [
                            row["auc_mean"]
                            for row in rows
                        ]
                    )
                ),

            "mean_ACC":
                float(
                    np.mean(
                        [
                            row["accuracy_mean"]
                            for row in rows
                        ]
                    )
                ),

            "mean_F1":
                float(
                    np.mean(
                        [
                            row["f1_mean"]
                            for row in rows
                        ]
                    )
                ),

            "mean_FP":
                float(
                    np.mean(
                        [
                            row["fp_mean"]
                            for row in rows
                        ]
                    )
                ),

            "mean_FN":
                float(
                    np.mean(
                        [
                            row["fn_mean"]
                            for row in rows
                        ]
                    )
                ),
        })

    summary.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return per_split, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--semantic-features",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-2017/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "artifacts/"
            "p9-2017-rolling-splits/"
            "split_manifest.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "semantic-exact-fallback-"
            "ablation-2017"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=0.03,
    )

    args = parser.parse_args()

    compact_rows = read_csv_rows(
        args.compact_features
    )

    compact_by_sha = {
        row["sha256"].strip().casefold():
            row
        for row in compact_rows
    }

    schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz(
        args.semantic_features
    )

    semantic_by_sha = {
        row["sha256"].strip().casefold():
            row
        for row in semantic_rows
    }

    names150 = support150(
        semantic_rows[0].keys()
    )

    splits = read_split_rows(
        args.splits
    )

    seen = set()

    for split_name, parts in (
        splits.items()
    ):
        for partition in (
            "train",
            "test",
        ):
            for row in parts[partition]:
                sha = row["sha256"]

                if sha not in compact_by_sha:
                    raise SystemExit(
                        "Compact缺少："
                        + sha
                    )

                if sha not in semantic_by_sha:
                    raise SystemExit(
                        "语义特征缺少："
                        + sha
                    )

                year = str(
                    compact_by_sha[
                        sha
                    ].get(
                        "metadata_year",
                        "",
                    )
                ).strip()

                if year != "2017":
                    raise SystemExit(
                        "发现非2017样本："
                        + sha
                    )

                seen.add(sha)

    print(
        "======================================"
    )
    print(
        "2017 精确回退消融"
    )
    print(
        "======================================"
    )
    print(
        "参与SHA =",
        len(seen),
    )
    print(
        "语义维数 =",
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
        args.iterations,
    )

    metric_rows = []
    prediction_rows = []
    audit_rows = []

    for split_name in sorted(
        splits
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        test_rows = (
            splits[
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

        # 当前联合模型所用：
        # 减均值 + 除标准差。
        (
            semantic_center_train,
            semantic_center_test,
            center_names,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        # 精确回退模型所用：
        # 只除标准差，不减均值。
        (
            semantic_zero_train,
            semantic_zero_test,
            zero_names,
        ) = zero_scale(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        if center_names != zero_names:
            raise SystemExit(
                f"{split_name}: "
                "可用语义列不一致"
            )

        raw_test = np.asarray(
            [
                [
                    float(
                        semantic_by_sha[
                            row["sha256"]
                        ][name]
                    )
                    for name in zero_names
                ]
                for row in test_rows
            ],
            dtype=np.float64,
        )

        raw_zero = np.all(
            np.abs(raw_test) <= 1e-15,
            axis=1,
        )

        zero_max_diffs = []

        for seed in args.seeds:

            # ==============================
            # Compact
            # ==============================
            compact_w, compact_b = (
                fit_logistic(
                    compact_train,
                    y_train,
                    seed,
                    iterations=
                        args.iterations,
                    learning_rate=
                        args.learning_rate,
                    l2=args.l2,
                )
            )

            base_train_logit = (
                compact_train
                @ compact_w
                + compact_b
            )

            base_test_logit = (
                compact_test
                @ compact_w
                + compact_b
            )

            p_compact = sigmoid(
                base_test_logit
            )

            # ==============================
            # 当前联合150维模型
            # ==============================
            joint_train = np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * semantic_center_train,
                ],
                axis=1,
            )

            joint_test = np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * semantic_center_test,
                ],
                axis=1,
            )

            joint_w, joint_b = fit_logistic(
                joint_train,
                y_train,
                seed,
                iterations=args.iterations,
                learning_rate=
                    args.learning_rate,
                l2=args.l2,
            )

            p_joint = sigmoid(
                joint_test @ joint_w
                + joint_b
            )

            # ==============================
            # 精确回退150维
            # ==============================
            semantic_train_scaled = (
                ALPHA
                * semantic_zero_train
            )

            semantic_test_scaled = (
                ALPHA
                * semantic_zero_test
            )

            semantic_w = (
                fit_semantic_correction(
                    base_train_logit,
                    y_train,
                    semantic_train_scaled,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
            )

            p_fallback = sigmoid(
                base_test_logit
                + semantic_test_scaled
                @ semantic_w
            )

            for variant, probability in (
                (
                    "compact_only",
                    p_compact,
                ),
                (
                    "joint_centered150",
                    p_joint,
                ),
                (
                    "exact_fallback150",
                    p_fallback,
                ),
            ):
                current = metrics(
                    y_test,
                    probability,
                    threshold=THRESHOLD,
                )

                metric_rows.append({
                    "variant":
                        variant,
                    "split_name":
                        split_name,
                    "seed":
                        seed,
                    **current,
                })

                for (
                    row,
                    label,
                    prob,
                    is_zero,
                ) in zip(
                    test_rows,
                    y_test,
                    probability,
                    raw_zero,
                ):
                    prediction_rows.append({
                        "variant":
                            variant,

                        "split_name":
                            split_name,

                        "seed":
                            seed,

                        "sha256":
                            row["sha256"],

                        "label":
                            int(label),

                        "probability":
                            float(prob),

                        "predicted_label":
                            int(
                                prob
                                >= THRESHOLD
                            ),

                        "raw_semantic150_all_zero":
                            int(is_zero),
                    })

            if np.any(raw_zero):
                zero_max_diffs.append(
                    float(
                        np.max(
                            np.abs(
                                p_fallback[
                                    raw_zero
                                ]
                                - p_compact[
                                    raw_zero
                                ]
                            )
                        )
                    )
                )

        audit_rows.append({
            "split_name":
                split_name,

            "test_count":
                len(test_rows),

            "raw_semantic150_all_zero_count":
                int(raw_zero.sum()),

            "raw_semantic150_all_zero_rate":
                float(raw_zero.mean()),

            "max_abs_probability_diff_"
            "fallback_vs_compact_on_zero":
                max(
                    zero_max_diffs
                    or [0.0]
                ),
        })

        print(
            split_name,
            "Compact维数=",
            len(compact_names),
            "语义可用维数=",
            len(zero_names),
            "测试150维全零=",
            int(raw_zero.sum()),
            "/",
            len(test_rows),
        )

    per_split, summary = summarize(
        metric_rows
    )

    write_csv(
        args.output
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        args.output
        / "per_split.csv",
        per_split,
    )

    write_csv(
        args.output
        / "summary.csv",
        summary,
    )

    write_csv(
        args.output
        / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "zero_semantic_audit.csv",
        audit_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "结果"
    )
    print(
        "======================================"
    )

    for row in summary:
        print(
            f"{row['variant']:24s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print()
    print(
        "150维原始全零样本的"
        "精确回退检查："
    )

    for row in audit_rows:
        print(
            row["split_name"],
            "全零=",
            row[
                "raw_semantic150_"
                "all_zero_count"
            ],
            "/",
            row["test_count"],
            "最大概率差=",
            "{:.12g}".format(
                row[
                    "max_abs_probability_"
                    "diff_fallback_vs_"
                    "compact_on_zero"
                ]
            ),
        )


if __name__ == "__main__":
    main()
