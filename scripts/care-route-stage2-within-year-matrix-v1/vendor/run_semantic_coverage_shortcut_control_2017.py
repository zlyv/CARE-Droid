#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
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
    aggregate,
    assert_2017_only,
    evaluate_matrix,
)


ALPHA = 0.25
THRESHOLD = 0.5

ANCHORS = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

ANCHOR_SET = set(ANCHORS)


def read_gz_csv(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(path: Path, rows):
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
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def support150_names(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            "支持关系列不是155维："
            + str(len(support))
        )

    selected = []

    for name in support:
        parts = name.split("__")

        if len(parts) != 3:
            raise SystemExit(
                "异常特征名："
                + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor not in ANCHOR_SET:
            raise SystemExit(
                "异常锚点："
                + anchor
            )

        # 删除五个“自己 -> 自己”
        if context != anchor:
            selected.append(name)

    if len(selected) != 150:
        raise SystemExit(
            "去掉5个自身关系后不是150维："
            + str(len(selected))
        )

    return selected


def build_coverage_table(
    semantic_rows,
    names150,
):
    """
    这里只构造“覆盖程度”，
    完全不保留具体是什么语义。

    1. 是否存在任意关系
    2. 非零关系数量
    3. 有多少个Top-5锚点产生了关系
    """

    table = {}

    for row in semantic_rows:
        sha = str(
            row["sha256"]
        ).strip().casefold()

        active_anchors = set()
        nonzero_pair_count = 0

        for name in names150:
            value = float(
                row[name]
            )

            if not math.isfinite(value):
                raise SystemExit(
                    f"{sha}: "
                    f"非有限语义值 "
                    f"{name}={value}"
                )

            if value > 0.0:
                nonzero_pair_count += 1

                anchor = (
                    name.split("__")[1]
                )

                active_anchors.add(
                    anchor
                )

        table[sha] = {
            "coverage_any_relation":
                float(
                    nonzero_pair_count > 0
                ),

            "coverage_nonzero_pair_count":
                float(
                    nonzero_pair_count
                ),

            "coverage_active_anchor_count":
                float(
                    len(active_anchors)
                ),
        }

    return table


def standardize_control(
    train_rows,
    test_rows,
    table,
    names,
):
    def matrix(rows):
        return np.asarray(
            [
                [
                    float(
                        table[
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
        not np.all(
            np.isfinite(train)
        )
        or not np.all(
            np.isfinite(test)
        )
    ):
        raise SystemExit(
            "覆盖控制特征存在非有限值"
        )

    means = train.mean(axis=0)
    scales = train.std(axis=0)

    usable = (
        scales > 1e-12
    )

    if not np.any(usable):
        return (
            np.zeros(
                (len(train_rows), 0),
                dtype=np.float64,
            ),
            np.zeros(
                (len(test_rows), 0),
                dtype=np.float64,
            ),
            [],
        )

    return (
        (
            train[:, usable]
            - means[usable]
        )
        / scales[usable],

        (
            test[:, usable]
            - means[usable]
        )
        / scales[usable],

        [
            name
            for name, keep
            in zip(names, usable)
            if keep
        ],
    )


def paired_corrections(
    prediction_rows,
    variants,
):
    probability = defaultdict(
        list
    )

    labels = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )

        probability[key].append(
            float(
                row["probability"]
            )
        )

        labels[
            (
                row["split_name"],
                row["sha256"],
            )
        ] = int(
            row["label"]
        )

    rows = []

    base_variant = (
        "compact_only"
    )

    for variant in variants:
        if variant == base_variant:
            continue

        counts = defaultdict(int)

        by_label = defaultdict(
            lambda: defaultdict(int)
        )

        for (
            split_name,
            sha,
        ), label in labels.items():

            base_key = (
                split_name,
                sha,
                base_variant,
            )

            new_key = (
                split_name,
                sha,
                variant,
            )

            if (
                base_key
                not in probability
                or new_key
                not in probability
            ):
                continue

            p_base = float(
                np.mean(
                    probability[
                        base_key
                    ]
                )
            )

            p_new = float(
                np.mean(
                    probability[
                        new_key
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
                state = "CORRECTED"

            elif (
                base_ok
                and not new_ok
            ):
                state = "REGRESSED"

            elif base_ok:
                state = (
                    "STILL_CORRECT"
                )

            else:
                state = (
                    "STILL_WRONG"
                )

            counts[state] += 1

            by_label[
                label
            ][state] += 1

        rows.append({
            "variant":
                variant,

            "corrected":
                counts[
                    "CORRECTED"
                ],

            "regressed":
                counts[
                    "REGRESSED"
                ],

            "net_corrected":
                (
                    counts[
                        "CORRECTED"
                    ]
                    - counts[
                        "REGRESSED"
                    ]
                ),

            "benign_corrected":
                by_label[
                    0
                ][
                    "CORRECTED"
                ],

            "malware_corrected":
                by_label[
                    1
                ][
                    "CORRECTED"
                ],

            "benign_regressed":
                by_label[
                    0
                ][
                    "REGRESSED"
                ],

            "malware_regressed":
                by_label[
                    1
                ][
                    "REGRESSED"
                ],

            "still_wrong":
                counts[
                    "STILL_WRONG"
                ],
        })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
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
            "semantic-coverage-"
            "shortcut-control-2017"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
            3,
            4,
        ],
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

    for path in (
        args.compact_features,
        args.compact_schema,
        args.semantic_features,
        args.splits,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_rows = read_csv_rows(
        args.compact_features
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    compact_schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz_csv(
        args.semantic_features
    )

    if not semantic_rows:
        raise SystemExit(
            "语义特征文件为空"
        )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    coverage_by_sha = (
        build_coverage_table(
            semantic_rows,
            names150,
        )
    )

    splits = read_split_rows(
        args.splits
    )

    selected_shas = (
        assert_2017_only(
            splits,
            compact_by_sha,
        )
    )

    missing_semantic = (
        selected_shas
        - set(semantic_by_sha)
    )

    missing_coverage = (
        selected_shas
        - set(coverage_by_sha)
    )

    if missing_semantic:
        raise SystemExit(
            "语义特征缺少划分样本："
            + str(
                len(
                    missing_semantic
                )
            )
        )

    if missing_coverage:
        raise SystemExit(
            "覆盖控制缺少划分样本："
            + str(
                len(
                    missing_coverage
                )
            )
        )

    coverage_names = {
        "any1": [
            "coverage_any_relation",
        ],

        "pair_count1": [
            "coverage_nonzero_pair_count",
        ],

        "anchor_count1": [
            "coverage_active_anchor_count",
        ],

        "coverage3": [
            "coverage_any_relation",
            "coverage_nonzero_pair_count",
            "coverage_active_anchor_count",
        ],
    }

    print(
        "======================================"
    )
    print(
        "2017 语义覆盖捷径控制实验"
    )
    print(
        "======================================"
    )

    print(
        "参与 SHA =",
        len(selected_shas),
    )

    print(
        "最终真实语义 = 150维支持关系"
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

    print()
    print("控制特征：")
    print(
        "  any1 = "
        "是否存在任意非自身语义关系"
    )
    print(
        "  pair_count1 = "
        "非零语义关系数量"
    )
    print(
        "  anchor_count1 = "
        "产生非自身关系的锚点数量"
    )
    print(
        "  coverage3 = "
        "上述三个覆盖量合并"
    )

    per_split_rows = []
    prediction_rows = []
    dimension_rows = []
    profile_rows = []

    all_variants = [
        "compact_only",
        "compact_plus_any1",
        "compact_plus_pair_count1",
        "compact_plus_anchor_count1",
        "compact_plus_coverage3",
        "coverage3_only",
        "compact_plus_support150",
    ]

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
            compact_schema,
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

        control_matrices = {}
        control_dims = {}

        for (
            control_name,
            names,
        ) in coverage_names.items():

            (
                x_train,
                x_test,
                usable,
            ) = standardize_control(
                train_rows,
                test_rows,
                coverage_by_sha,
                names,
            )

            control_matrices[
                control_name
            ] = (
                x_train,
                x_test,
            )

            control_dims[
                control_name
            ] = len(usable)

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),

            "compact_plus_support150": (
                np.concatenate(
                    [
                        compact_train,
                        ALPHA
                        * semantic_train,
                    ],
                    axis=1,
                ),

                np.concatenate(
                    [
                        compact_test,
                        ALPHA
                        * semantic_test,
                    ],
                    axis=1,
                ),
            ),
        }

        for control_name in (
            "any1",
            "pair_count1",
            "anchor_count1",
            "coverage3",
        ):
            (
                x_train,
                x_test,
            ) = control_matrices[
                control_name
            ]

            matrices[
                "compact_plus_"
                + control_name
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        ALPHA * x_train,
                    ],
                    axis=1,
                ),

                np.concatenate(
                    [
                        compact_test,
                        ALPHA * x_test,
                    ],
                    axis=1,
                ),
            )

        (
            x_cov_train,
            x_cov_test,
        ) = control_matrices[
            "coverage3"
        ]

        matrices[
            "coverage3_only"
        ] = (
            ALPHA * x_cov_train,
            ALPHA * x_cov_test,
        )

        dimension_rows.append({
            "split_name":
                split_name,

            "compact_dim":
                len(compact_names),

            "support150_usable_dim":
                len(
                    semantic_usable
                ),

            "any1_usable_dim":
                control_dims[
                    "any1"
                ],

            "pair_count1_usable_dim":
                control_dims[
                    "pair_count1"
                ],

            "anchor_count1_usable_dim":
                control_dims[
                    "anchor_count1"
                ],

            "coverage3_usable_dim":
                control_dims[
                    "coverage3"
                ],
        })

        # ---------------------------------
        # 单独记录“有关系/无关系”的
        # 标签组成，检查覆盖捷径强度。
        # ---------------------------------
        for (
            partition_name,
            rows,
            labels,
        ) in (
            (
                "train",
                train_rows,
                y_train,
            ),
            (
                "test",
                test_rows,
                y_test,
            ),
        ):
            any_values = np.asarray(
                [
                    int(
                        coverage_by_sha[
                            row["sha256"]
                        ][
                            "coverage_any_relation"
                        ]
                    )
                    for row in rows
                ],
                dtype=int,
            )

            for any_value in (
                0,
                1,
            ):
                mask = (
                    any_values
                    == any_value
                )

                n = int(
                    mask.sum()
                )

                if n == 0:
                    benign = 0
                    malware = 0
                else:
                    benign = int(
                        np.sum(
                            labels[
                                mask
                            ] == 0
                        )
                    )

                    malware = int(
                        np.sum(
                            labels[
                                mask
                            ] == 1
                        )
                    )

                profile_rows.append({
                    "split_name":
                        split_name,

                    "partition":
                        partition_name,

                    "has_relation":
                        any_value,

                    "count":
                        n,

                    "benign":
                        benign,

                    "malware":
                        malware,

                    "malware_rate":
                        (
                            malware
                            / max(
                                1,
                                n,
                            )
                        ),
                })

        for variant in all_variants:
            (
                x_train,
                x_test,
            ) = matrices[
                variant
            ]

            per_split_rows.append(
                evaluate_matrix(
                    split_name,
                    train_rows,
                    test_rows,
                    y_train,
                    y_test,
                    x_train,
                    x_test,
                    variant,
                    args.seeds,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                    prediction_rows,
                )
            )

        print(
            "完成",
            split_name,
            "support150可用维数=",
            len(
                semantic_usable
            ),
            "coverage3可用维数=",
            control_dims[
                "coverage3"
            ],
            flush=True,
        )

    summary_rows = aggregate(
        per_split_rows
    )

    correction_rows = (
        paired_corrections(
            prediction_rows,
            all_variants,
        )
    )

    base = next(
        row
        for row in summary_rows
        if row["variant"]
        == "compact_only"
    )

    delta_rows = []

    for row in summary_rows:
        if (
            row["variant"]
            == "compact_only"
        ):
            continue

        delta_rows.append({
            "variant":
                row["variant"],

            "delta_BA":
                float(
                    row["mean_BA"]
                )
                - float(
                    base["mean_BA"]
                ),

            "delta_min_BA":
                float(
                    row["min_BA"]
                )
                - float(
                    base["min_BA"]
                ),

            "delta_AUC":
                float(
                    row["mean_AUC"]
                )
                - float(
                    base["mean_AUC"]
                ),

            "delta_ACC":
                float(
                    row["mean_ACC"]
                )
                - float(
                    base["mean_ACC"]
                ),
        })

    write_csv(
        args.output / "per_split.csv",
        per_split_rows,
    )

    write_csv(
        args.output / "summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "delta_vs_compact.csv",
        delta_rows,
    )

    write_csv(
        args.output
        / "correction_summary.csv",
        correction_rows,
    )

    write_csv(
        args.output
        / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "dimensions.csv",
        dimension_rows,
    )

    write_csv(
        args.output
        / "coverage_label_profile.csv",
        profile_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "按平均平衡准确率排序"
    )
    print(
        "======================================"
    )

    for row in summary_rows:
        print(
            f"{row['variant']:30s} "
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
        "相对 Compact 的变化"
    )
    print(
        "======================================"
    )

    for row in delta_rows:
        print(
            f"{row['variant']:30s} "
            f"dBA="
            f"{row['delta_BA']:+.6f} "
            f"dMin="
            f"{row['delta_min_BA']:+.6f} "
            f"dAUC="
            f"{row['delta_AUC']:+.6f} "
            f"dACC="
            f"{row['delta_ACC']:+.6f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐样本修正"
    )
    print(
        "======================================"
    )

    for row in correction_rows:
        print(
            f"{row['variant']:30s} "
            f"修正="
            f"{row['corrected']:2d} "
            f"退化="
            f"{row['regressed']:2d} "
            f"净修正="
            f"{row['net_corrected']:+d} "
            f"良性修正="
            f"{row['benign_corrected']:2d} "
            f"恶意修正="
            f"{row['malware_corrected']:2d}"
        )

    print()
    print(
        "保存目录：",
        args.output,
    )


if __name__ == "__main__":
    main()
