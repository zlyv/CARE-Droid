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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


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
    aggregate,
    assert_2017_only,
)


ALPHA = 0.25
THRESHOLD = 0.5

ANCHORS = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


def read_gz(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


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
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def support_sets(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            "support维数不是155："
            + str(len(support))
        )

    nonanchor = []
    cross = []
    self_rel = []

    for name in support:
        parts = name.split("__")

        if len(parts) != 3:
            raise SystemExit(
                "异常特征名："
                + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor not in ANCHORS:
            raise SystemExit(
                "异常锚点："
                + anchor
            )

        if context not in ANCHORS:
            nonanchor.append(name)

        elif context == anchor:
            self_rel.append(name)

        else:
            cross.append(name)

    if len(nonanchor) != 130:
        raise SystemExit(
            "五类外维数错误："
            + str(len(nonanchor))
        )

    if len(cross) != 20:
        raise SystemExit(
            "锚点交叉维数错误："
            + str(len(cross))
        )

    if len(self_rel) != 5:
        raise SystemExit(
            "自身关系维数错误："
            + str(len(self_rel))
        )

    return {
        "nonanchor130":
            nonanchor,

        "cross20":
            cross,

        "self5":
            self_rel,

        "nonanchor_plus_cross150":
            nonanchor + cross,

        "nonanchor_plus_self135":
            nonanchor + self_rel,

        "all155":
            nonanchor + cross + self_rel,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/p1-features/features.csv"
        ),
    )

    ap.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/feature_schema.json"
        ),
    )

    ap.add_argument(
        "--semantic-features",
        type=Path,
        default=Path(
            "artifacts/clean-local-context-2017/"
            "semantic_relation_features.csv.gz"
        ),
    )

    ap.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "artifacts/p9-2017-rolling-splits/"
            "split_manifest.csv"
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "support-partition-ablation-2017"
        ),
    )

    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    ap.add_argument(
        "--iterations",
        type=int,
        default=1500,
    )

    ap.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--l2",
        type=float,
        default=0.03,
    )

    args = ap.parse_args()

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

    semantic_rows = read_gz(
        args.semantic_features
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    feature_sets = support_sets(
        semantic_rows[0].keys()
    )

    splits = read_split_rows(
        args.splits
    )

    selected_shas = assert_2017_only(
        splits,
        compact_by_sha,
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
        "2017 支持关系 130+20+5 消融"
    )
    print(
        "======================================"
    )
    print("alpha =", ALPHA)
    print("threshold =", THRESHOLD)
    print("参与 SHA =", len(selected_shas))
    print()

    for name, columns in feature_sets.items():
        print(
            name,
            "=",
            len(columns),
        )

    per_split_rows = []
    prediction_rows = []
    dimension_rows = []

    for split_name in sorted(splits):

        train_rows = (
            splits[split_name]["train"]
        )

        test_rows = (
            splits[split_name]["test"]
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

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            )
        }

        dimension_row = {
            "split_name": split_name,
            "compact_dim":
                len(compact_names),
        }

        for set_name, names in (
            feature_sets.items()
        ):

            (
                sem_train,
                sem_test,
                usable_names,
            ) = standardize_named(
                train_rows,
                test_rows,
                semantic_by_sha,
                names,
            )

            matrices[
                "compact_plus_" + set_name
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        ALPHA * sem_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        ALPHA * sem_test,
                    ],
                    axis=1,
                ),
            )

            dimension_row[
                set_name + "_usable_dim"
            ] = len(usable_names)

        dimension_rows.append(
            dimension_row
        )

        for variant, (
            x_train,
            x_test,
        ) in matrices.items():

            result = evaluate_matrix(
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

            per_split_rows.append(
                result
            )

        print(
            "完成",
            split_name,
            flush=True,
        )

    summary_rows = aggregate(
        per_split_rows
    )

    write_csv(
        args.output / "per_split.csv",
        per_split_rows,
    )

    write_csv(
        args.output / "summary.csv",
        summary_rows,
    )

    write_csv(
        args.output / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output / "dimensions.csv",
        dimension_rows,
    )

    # ------------------------------------------
    # 相对 Compact 的均值变化
    # ------------------------------------------
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
        args.output / "delta_vs_compact.csv",
        delta_rows,
    )

    # ------------------------------------------
    # 每种表示相对 Compact：
    # 五个随机种子概率先平均，再看翻转。
    # ------------------------------------------
    probability = defaultdict(list)
    labels = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )

        probability[key].append(
            float(row["probability"])
        )

        labels[
            (
                row["split_name"],
                row["sha256"],
            )
        ] = int(row["label"])

    variants = [
        row["variant"]
        for row in summary_rows
        if row["variant"]
        != "compact_only"
    ]

    correction_rows = []

    for variant in variants:
        counts = defaultdict(int)

        for (
            split_name,
            sha,
        ), y in labels.items():

            base_key = (
                split_name,
                sha,
                "compact_only",
            )

            new_key = (
                split_name,
                sha,
                variant,
            )

            if (
                base_key not in probability
                or new_key not in probability
            ):
                continue

            pb = float(
                np.mean(
                    probability[base_key]
                )
            )

            pn = float(
                np.mean(
                    probability[new_key]
                )
            )

            base_ok = (
                int(pb >= THRESHOLD)
                == y
            )

            new_ok = (
                int(pn >= THRESHOLD)
                == y
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
                state = "STILL_CORRECT"

            else:
                state = "STILL_WRONG"

            counts[state] += 1

        correction_rows.append({
            "variant":
                variant,

            "corrected":
                counts["CORRECTED"],

            "regressed":
                counts["REGRESSED"],

            "net_corrected":
                (
                    counts["CORRECTED"]
                    - counts["REGRESSED"]
                ),

            "still_correct":
                counts["STILL_CORRECT"],

            "still_wrong":
                counts["STILL_WRONG"],
        })

    write_csv(
        args.output
        / "correction_summary.csv",
        correction_rows,
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
            f"{row['variant']:42s} "
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
            f"{row['variant']:42s} "
            f"dBA={row['delta_BA']:+.6f} "
            f"dMin={row['delta_min_BA']:+.6f} "
            f"dAUC={row['delta_AUC']:+.6f} "
            f"dACC={row['delta_ACC']:+.6f}"
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
            f"{row['variant']:42s} "
            f"修正={row['corrected']:2d} "
            f"退化={row['regressed']:2d} "
            f"净修正={row['net_corrected']:+d}"
        )

    print()
    print(
        "保存目录：",
        args.output,
    )


if __name__ == "__main__":
    main()
