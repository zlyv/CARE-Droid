#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
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
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(rows[0])

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts):
    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    )


def support150_layout(header):
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

    layout = {
        anchor: []
        for anchor in ANCHORS
    }

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

        if context == anchor:
            continue

        layout[anchor].append(name)
        selected.append(name)

    for anchor in ANCHORS:
        layout[anchor] = sorted(
            layout[anchor]
        )

        if len(layout[anchor]) != 30:
            raise SystemExit(
                f"{anchor}: "
                "去掉自身关系后不是30维："
                f"{len(layout[anchor])}"
            )

    if len(selected) != 150:
        raise SystemExit(
            "最终不是150维："
            + str(len(selected))
        )

    return (
        sorted(selected),
        layout,
    )


def randomize_presence_degree_preserving(
    presence,
    rng,
    swap_factor,
):
    """
    双边交换：

    保持：
      每个APK的非零关系数
      每个语义列的非零关系数

    改变：
      哪个APK对应哪个具体上下文语义
    """

    n_rows, n_cols = presence.shape

    neighbors = [
        set(
            np.flatnonzero(
                presence[row]
            )
        )
        for row in range(n_rows)
    ]

    active_rows = [
        row
        for row, cols
        in enumerate(neighbors)
        if cols
    ]

    edge_count = int(
        presence.sum()
    )

    target = (
        int(swap_factor)
        * edge_count
    )

    if (
        target == 0
        or len(active_rows) < 2
        or n_cols < 2
    ):
        return (
            presence.copy(),
            0,
            target,
            0,
        )

    accepted = 0
    attempts = 0

    max_attempts = max(
        1000,
        target * 100,
    )

    while (
        accepted < target
        and attempts < max_attempts
    ):
        attempts += 1

        r1, r2 = rng.choice(
            active_rows,
            size=2,
            replace=False,
        )

        r1 = int(r1)
        r2 = int(r2)

        cols1 = tuple(
            sorted(
                neighbors[r1]
            )
        )

        cols2 = tuple(
            sorted(
                neighbors[r2]
            )
        )

        if not cols1 or not cols2:
            continue

        c1 = int(
            cols1[
                int(
                    rng.integers(
                        len(cols1)
                    )
                )
            ]
        )

        c2 = int(
            cols2[
                int(
                    rng.integers(
                        len(cols2)
                    )
                )
            ]
        )

        if c1 == c2:
            continue

        if c2 in neighbors[r1]:
            continue

        if c1 in neighbors[r2]:
            continue

        neighbors[r1].remove(c1)
        neighbors[r1].add(c2)

        neighbors[r2].remove(c2)
        neighbors[r2].add(c1)

        accepted += 1

    out = np.zeros_like(
        presence,
        dtype=bool,
    )

    for row, cols in enumerate(
        neighbors
    ):
        if cols:
            out[
                row,
                list(cols),
            ] = True

    if not np.array_equal(
        out.sum(axis=1),
        presence.sum(axis=1),
    ):
        raise RuntimeError(
            "随机后APK关系数量改变"
        )

    if not np.array_equal(
        out.sum(axis=0),
        presence.sum(axis=0),
    ):
        raise RuntimeError(
            "随机后语义列出现次数改变"
        )

    return (
        out,
        accepted,
        target,
        attempts,
    )


def permute_partition(
    rows,
    semantic_by_sha,
    layout,
    repeat,
    split_name,
    partition_name,
    swap_factor,
):
    """
    强随机对照。

    严格保持：
      1. 每APK每锚点非零关系数量
      2. 每个语义列非零数量
      3. 每APK每锚点支持值集合
      4. 全零状态
      5. 总关系数量
      6. 有效锚点数量

    只破坏：
      具体上下文语义身份与APK的对应。
    """

    shas = [
        row["sha256"]
        for row in rows
    ]

    result = {
        sha: {
            "sha256": sha
        }
        for sha in shas
    }

    audit = []

    for anchor in ANCHORS:
        names = layout[anchor]

        values = np.asarray(
            [
                [
                    float(
                        semantic_by_sha[
                            sha
                        ][name]
                    )
                    for name in names
                ]
                for sha in shas
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise SystemExit(
                f"{split_name}/"
                f"{partition_name}/"
                f"{anchor}: "
                "存在非有限值"
            )

        if np.any(
            values < -1e-15
        ):
            raise SystemExit(
                "支持比例出现负值"
            )

        presence = (
            values > 1e-15
        )

        row_degree = (
            presence.sum(axis=1)
        )

        col_degree = (
            presence.sum(axis=0)
        )

        rng = np.random.default_rng(
            stable_seed(
                "presence",
                repeat,
                split_name,
                partition_name,
                anchor,
            )
        )

        (
            new_presence,
            accepted,
            target,
            attempts,
        ) = (
            randomize_presence_degree_preserving(
                presence,
                rng,
                swap_factor,
            )
        )

        if not np.array_equal(
            new_presence.sum(axis=1),
            row_degree,
        ):
            raise RuntimeError(
                "行覆盖没有保持"
            )

        if not np.array_equal(
            new_presence.sum(axis=0),
            col_degree,
        ):
            raise RuntimeError(
                "列覆盖没有保持"
            )

        randomized_values = (
            np.zeros_like(values)
        )

        for row_index, sha in enumerate(
            shas
        ):
            old_values = (
                values[
                    row_index,
                    presence[
                        row_index
                    ],
                ].copy()
            )

            new_columns = (
                np.flatnonzero(
                    new_presence[
                        row_index
                    ]
                )
            )

            if (
                len(old_values)
                != len(new_columns)
            ):
                raise RuntimeError(
                    "行内非零数没有保持"
                )

            row_rng = (
                np.random.default_rng(
                    stable_seed(
                        "values",
                        repeat,
                        split_name,
                        partition_name,
                        anchor,
                        sha,
                    )
                )
            )

            row_rng.shuffle(
                old_values
            )

            randomized_values[
                row_index,
                new_columns,
            ] = old_values

            old_sorted = np.sort(
                values[
                    row_index,
                    presence[
                        row_index
                    ],
                ]
            )

            new_sorted = np.sort(
                randomized_values[
                    row_index,
                    randomized_values[
                        row_index
                    ] > 1e-15,
                ]
            )

            if not np.array_equal(
                old_sorted,
                new_sorted,
            ):
                raise RuntimeError(
                    "APK内支持值集合改变"
                )

        for row_index, sha in enumerate(
            shas
        ):
            for col_index, name in enumerate(
                names
            ):
                result[sha][name] = float(
                    randomized_values[
                        row_index,
                        col_index,
                    ]
                )

        audit.append({
            "repeat":
                repeat,

            "split_name":
                split_name,

            "partition":
                partition_name,

            "anchor":
                anchor,

            "row_count":
                len(rows),

            "edge_count":
                int(
                    presence.sum()
                ),

            "target_swaps":
                target,

            "accepted_swaps":
                accepted,

            "attempts":
                attempts,

            "swap_completion_rate":
                (
                    accepted / target
                    if target > 0
                    else 1.0
                ),

            "nonzero_column_count":
                int(
                    np.sum(
                        col_degree > 0
                    )
                ),
        })

    return result, audit


def labels_of(
    rows,
    compact_by_sha,
):
    return np.asarray(
        [
            label_value(
                compact_by_sha[
                    row["sha256"]
                ].get(
                    "metadata_label",
                    "",
                )
            )
            for row in rows
        ],
        dtype=np.int64,
    )


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
            "semantic-identity-"
            "permutation-2017"
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--swap-factor",
        type=int,
        default=20,
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

    if args.repeats < 1:
        raise SystemExit(
            "repeats必须>=1"
        )

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

    (
        names150,
        layout,
    ) = support150_layout(
        semantic_rows[0].keys()
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

    missing = (
        selected_shas
        - set(semantic_by_sha)
    )

    if missing:
        raise SystemExit(
            "语义特征缺少划分样本："
            + str(len(missing))
        )

    print(
        "======================================"
    )
    print(
        "2017 语义身份保持覆盖随机对照"
    )
    print(
        "======================================"
    )

    print(
        "参与 SHA =",
        len(selected_shas),
    )

    print(
        "真实语义维数 =",
        len(names150),
    )

    print(
        "随机次数 =",
        args.repeats,
    )

    print(
        "交换倍数 =",
        args.swap_factor,
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
    print(
        "保持覆盖与数量，只破坏具体语义身份"
    )

    real_per_split = []
    compact_per_split = []

    split_cache = {}

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

        y_train = labels_of(
            train_rows,
            compact_by_sha,
        )

        y_test = labels_of(
            test_rows,
            compact_by_sha,
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
            real_train,
            real_test,
            real_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        compact_per_split.append(
            evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                y_train,
                y_test,
                compact_train,
                compact_test,
                "compact_only",
                args.seeds,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
        )

        real_per_split.append(
            evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                y_train,
                y_test,
                np.concatenate(
                    [
                        compact_train,
                        ALPHA * real_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        ALPHA * real_test,
                    ],
                    axis=1,
                ),
                "real_support150",
                args.seeds,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
        )

        split_cache[
            split_name
        ] = {
            "train_rows":
                train_rows,

            "test_rows":
                test_rows,

            "y_train":
                y_train,

            "y_test":
                y_test,

            "compact_train":
                compact_train,

            "compact_test":
                compact_test,

            "compact_dim":
                len(compact_names),

            "real_semantic_dim":
                len(real_usable),
        }

    compact_summary = aggregate(
        compact_per_split
    )[0]

    real_summary = aggregate(
        real_per_split
    )[0]

    print()
    print(
        "Compact BA =",
        f"{compact_summary['mean_BA']:.6f}",
    )

    print(
        "真实150 BA =",
        f"{real_summary['mean_BA']:.6f}",
    )

    print(
        "真实150 AUC =",
        f"{real_summary['mean_AUC']:.6f}",
    )

    null_repeat_rows = []
    null_per_split_rows = []
    swap_audit_rows = []

    for repeat in range(
        args.repeats
    ):
        current = []

        for split_name in sorted(
            splits
        ):
            cached = (
                split_cache[
                    split_name
                ]
            )

            train_rows = (
                cached[
                    "train_rows"
                ]
            )

            test_rows = (
                cached[
                    "test_rows"
                ]
            )

            (
                shuffled_train_by_sha,
                audit_train,
            ) = permute_partition(
                train_rows,
                semantic_by_sha,
                layout,
                repeat,
                split_name,
                "train",
                args.swap_factor,
            )

            (
                shuffled_test_by_sha,
                audit_test,
            ) = permute_partition(
                test_rows,
                semantic_by_sha,
                layout,
                repeat,
                split_name,
                "test",
                args.swap_factor,
            )

            swap_audit_rows.extend(
                audit_train
            )

            swap_audit_rows.extend(
                audit_test
            )

            shuffled_by_sha = dict(
                shuffled_train_by_sha
            )

            shuffled_by_sha.update(
                shuffled_test_by_sha
            )

            (
                shuffled_train,
                shuffled_test,
                shuffled_usable,
            ) = standardize_named(
                train_rows,
                test_rows,
                shuffled_by_sha,
                names150,
            )

            if (
                len(shuffled_usable)
                != cached[
                    "real_semantic_dim"
                ]
            ):
                raise RuntimeError(
                    f"{split_name}/"
                    f"repeat={repeat}: "
                    "随机后可用维数改变 "
                    f"{len(shuffled_usable)} "
                    "vs "
                    f"{cached['real_semantic_dim']}"
                )

            result = evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                cached["y_train"],
                cached["y_test"],
                np.concatenate(
                    [
                        cached[
                            "compact_train"
                        ],
                        ALPHA
                        * shuffled_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        cached[
                            "compact_test"
                        ],
                        ALPHA
                        * shuffled_test,
                    ],
                    axis=1,
                ),
                (
                    "shuffled_support150_"
                    f"r{repeat:02d}"
                ),
                args.seeds,
                args.iterations,
                args.learning_rate,
                args.l2,
            )

            current.append(result)

            null_per_split_rows.append({
                "repeat":
                    repeat,
                **result,
            })

        summary = aggregate(
            current
        )[0]

        null_repeat_rows.append({
            "repeat":
                repeat,

            "mean_BA":
                summary[
                    "mean_BA"
                ],

            "min_BA":
                summary[
                    "min_BA"
                ],

            "mean_AUC":
                summary[
                    "mean_AUC"
                ],

            "mean_ACC":
                summary[
                    "mean_ACC"
                ],

            "mean_F1":
                summary[
                    "mean_F1"
                ],

            "mean_FP":
                summary[
                    "mean_FP"
                ],

            "mean_FN":
                summary[
                    "mean_FN"
                ],
        })

        print(
            f"repeat "
            f"{repeat + 1:02d}/"
            f"{args.repeats}: "
            f"BA="
            f"{summary['mean_BA']:.6f} "
            f"AUC="
            f"{summary['mean_AUC']:.6f} "
            f"ACC="
            f"{summary['mean_ACC']:.6f}",
            flush=True,
        )

    ba_null = np.asarray(
        [
            row["mean_BA"]
            for row in null_repeat_rows
        ],
        dtype=float,
    )

    auc_null = np.asarray(
        [
            row["mean_AUC"]
            for row in null_repeat_rows
        ],
        dtype=float,
    )

    acc_null = np.asarray(
        [
            row["mean_ACC"]
            for row in null_repeat_rows
        ],
        dtype=float,
    )

    real_ba = float(
        real_summary[
            "mean_BA"
        ]
    )

    real_auc = float(
        real_summary[
            "mean_AUC"
        ]
    )

    real_acc = float(
        real_summary[
            "mean_ACC"
        ]
    )

    ba_ge = int(
        np.sum(
            ba_null
            >= real_ba - 1e-15
        )
    )

    auc_ge = int(
        np.sum(
            auc_null
            >= real_auc - 1e-15
        )
    )

    acc_ge = int(
        np.sum(
            acc_null
            >= real_acc - 1e-15
        )
    )

    comparison_rows = []

    for (
        metric_name,
        real_value,
        null_values,
        ge_count,
    ) in (
        (
            "BA",
            real_ba,
            ba_null,
            ba_ge,
        ),
        (
            "AUC",
            real_auc,
            auc_null,
            auc_ge,
        ),
        (
            "ACC",
            real_acc,
            acc_null,
            acc_ge,
        ),
    ):
        comparison_rows.append({
            "metric":
                metric_name,

            "real":
                real_value,

            "null_mean":
                float(
                    null_values.mean()
                ),

            "null_std":
                float(
                    null_values.std()
                ),

            "null_median":
                float(
                    np.median(
                        null_values
                    )
                ),

            "null_min":
                float(
                    null_values.min()
                ),

            "null_max":
                float(
                    null_values.max()
                ),

            "real_minus_null_mean":
                float(
                    real_value
                    - null_values.mean()
                ),

            "null_ge_real_count":
                ge_count,

            "repeat_count":
                args.repeats,

            "plus_one_p":
                float(
                    (
                        ge_count + 1
                    )
                    / (
                        args.repeats + 1
                    )
                ),
        })

    completion_rates = np.asarray(
        [
            float(
                row[
                    "swap_completion_rate"
                ]
            )
            for row in swap_audit_rows
        ],
        dtype=float,
    )

    write_csv(
        args.output
        / "null_repeat_summary.csv",
        null_repeat_rows,
    )

    write_csv(
        args.output
        / "null_per_split.csv",
        null_per_split_rows,
    )

    write_csv(
        args.output
        / "swap_audit.csv",
        swap_audit_rows,
    )

    write_csv(
        args.output
        / "real_vs_null.csv",
        comparison_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "真实语义 vs 保持覆盖的随机语义"
    )
    print(
        "======================================"
    )

    for row in comparison_rows:
        print(
            f"{row['metric']:4s} "
            f"real="
            f"{row['real']:.6f} "
            f"null_mean="
            f"{row['null_mean']:.6f} "
            f"null_max="
            f"{row['null_max']:.6f} "
            f"real-null="
            f"{row['real_minus_null_mean']:+.6f} "
            f"null>=real="
            f"{row['null_ge_real_count']}/"
            f"{row['repeat_count']} "
            f"p+1="
            f"{row['plus_one_p']:.4f}"
        )

    print()
    print(
        "随机交换完成度："
    )

    print(
        "  平均 =",
        f"{completion_rates.mean():.2%}",
    )

    print(
        "  最低 =",
        f"{completion_rates.min():.2%}",
    )

    print(
        "  <50% 的块 =",
        int(
            np.sum(
                completion_rates
                < 0.5
            )
        ),
    )

    print()
    print(
        "保存目录：",
        args.output,
    )


if __name__ == "__main__":
    main()
