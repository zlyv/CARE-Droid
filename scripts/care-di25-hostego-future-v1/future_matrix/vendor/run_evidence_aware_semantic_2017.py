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

from run_pilot_compact_eval import (
    fit_logistic,
    sigmoid,
    metrics,
)

from run_compact_20_150_ablation_2017 import (
    old20_names,
    support150_names,
    TOP5,
)


ALPHA = 0.25
THRESHOLD = 0.5

SEEDS = [0, 1, 2, 3, 4]

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

UPSTREAM_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/evidence-aware-semantic-2017"
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


def caller_log_columns(
    names20,
):
    result = {}

    for anchor in TOP5:
        candidates = [
            name
            for name in names20
            if (
                anchor.casefold()
                in name.casefold()
                and
                "caller_log"
                in name.casefold()
            )
        ]

        if len(candidates) != 1:
            raise SystemExit(
                f"{anchor}: "
                f"caller_log候选={candidates}"
            )

        result[anchor] = (
            candidates[0]
        )

    return result


def build_evidence_aware_table(
    semantic_rows,
    upstream_by_sha,
    names150,
    caller_logs,
):
    """
    不改变150维语义身份。

    对属于锚点 a 的每个关系：
        r' = r * n/(n+1)

    其中：
        n = exp(caller_log_a) - 1

    n来自旧20维里的锚点上游调用者数量，
    因此这里称为“结构证据量门控”，
    而不是声称它就是新语义关系的精确分母。
    """

    output = {}
    audit = []

    for row in semantic_rows:
        sha = str(
            row["sha256"]
        ).strip().casefold()

        if sha not in upstream_by_sha:
            continue

        upstream = (
            upstream_by_sha[sha]
        )

        out = {
            "sha256": sha,
        }

        anchor_n = {}

        for anchor in TOP5:
            log_value = float(
                upstream[
                    caller_logs[
                        anchor
                    ]
                ]
            )

            raw_n = float(
                np.expm1(
                    log_value
                )
            )

            # caller_log原本来自整数计数。
            n = max(
                0,
                int(round(raw_n)),
            )

            confidence = (
                n / (n + 1.0)
                if n > 0
                else 0.0
            )

            anchor_n[anchor] = (
                n,
                confidence,
            )

        for name in names150:
            parts = name.split("__")

            if len(parts) != 3:
                raise SystemExit(
                    "异常语义列："
                    + name
                )

            anchor = parts[1]

            if anchor not in anchor_n:
                raise SystemExit(
                    "未知锚点："
                    + anchor
                )

            value = float(
                row[name]
            )

            if not np.isfinite(value):
                raise SystemExit(
                    f"{sha}: "
                    f"{name}非有限值"
                )

            _, confidence = (
                anchor_n[anchor]
            )

            out[name] = (
                value * confidence
            )

        output[sha] = out

        for anchor in TOP5:
            n, confidence = (
                anchor_n[anchor]
            )

            audit.append({
                "sha256":
                    sha,

                "anchor":
                    anchor,

                "evidence_n":
                    n,

                "confidence":
                    confidence,
            })

    return output, audit


def evaluate(
    variant,
    split_name,
    seed,
    x_train,
    y_train,
    x_test,
    y_test,
):
    w, b = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        l2=L2,
    )

    probability = sigmoid(
        x_test @ w + b
    )

    result = metrics(
        y_test,
        probability,
        threshold=THRESHOLD,
    )

    return {
        "variant":
            variant,

        "split_name":
            split_name,

        "seed":
            seed,

        **result,
    }


def summarize(rows):
    grouped = defaultdict(list)

    for row in rows:
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
    ), current in sorted(
        grouped.items()
    ):
        out = {
            "variant":
                variant,

            "split_name":
                split_name,

            "seed_count":
                len(current),
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
                np.mean([
                    float(row[field])
                    for row in current
                ])
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, current in (
        by_variant.items()
    ):
        bas = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in current
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(bas.mean()),

            "min_BA":
                float(bas.min()),

            "std_BA":
                float(bas.std()),

            "mean_AUC":
                float(np.mean([
                    row["auc_mean"]
                    for row in current
                ])),

            "mean_ACC":
                float(np.mean([
                    row["accuracy_mean"]
                    for row in current
                ])),

            "mean_F1":
                float(np.mean([
                    row["f1_mean"]
                    for row in current
                ])),

            "mean_FP":
                float(np.mean([
                    row["fp_mean"]
                    for row in current
                ])),

            "mean_FN":
                float(np.mean([
                    row["fn_mean"]
                    for row in current
                ])),
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
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        UPSTREAM_PATH,
        SEMANTIC_PATH,
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

    semantic_rows = read_gz(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    names20 = old20_names(
        upstream_rows[0].keys()
    )

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    caller_logs = (
        caller_log_columns(
            names20
        )
    )

    print()
    print(
        "锚点证据量列："
    )

    for anchor in TOP5:
        print(
            anchor,
            "=>",
            caller_logs[anchor],
        )

    (
        evidence_by_sha,
        evidence_audit,
    ) = build_evidence_aware_table(
        semantic_rows,
        upstream_by_sha,
        names150,
        caller_logs,
    )

    splits = read_split_rows(
        SPLIT_PATH
    )

    all_shas = set()

    for parts in splits.values():
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                all_shas.add(
                    row["sha256"]
                )

    for name, table in (
        (
            "Compact",
            compact_by_sha,
        ),
        (
            "旧20维",
            upstream_by_sha,
        ),
        (
            "原始150维",
            semantic_by_sha,
        ),
        (
            "证据量150维",
            evidence_by_sha,
        ),
    ):
        missing = (
            all_shas
            - set(table)
        )

        print(
            name,
            "缺失 =",
            len(missing),
        )

        if missing:
            raise SystemExit(
                name
                + "人口不完整"
            )

    print()
    print(
        "======================================"
    )
    print(
        "2017 证据量感知语义关系"
    )
    print(
        "======================================"
    )

    print(
        "参与SHA =",
        len(all_shas),
    )

    print(
        "150维 =",
        len(names150),
    )

    print(
        "confidence = n/(n+1)"
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    metric_rows = []
    dimension_rows = []

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
            raw150_train,
            raw150_test,
            raw150_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        (
            evidence150_train,
            evidence150_test,
            evidence150_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            evidence_by_sha,
            names150,
        )

        # 两种150的训练期非恒定列原则上应一致，
        # 若不一致就明确报告，不静默处理。
        if (
            raw150_usable
            != evidence150_usable
        ):
            print(
                "注意：",
                split_name,
                "原始150和证据量150的"
                "可用列集合不同：",
                len(raw150_usable),
                "vs",
                len(evidence150_usable),
            )

        x_raw150_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * raw150_train,
                ],
                axis=1,
            )
        )

        x_raw150_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * raw150_test,
                ],
                axis=1,
            )
        )

        x_evidence_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * evidence150_train,
                ],
                axis=1,
            )
        )

        x_evidence_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * evidence150_test,
                ],
                axis=1,
            )
        )

        x_20_150_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * old20_train,
                    ALPHA
                    * raw150_train,
                ],
                axis=1,
            )
        )

        x_20_150_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * old20_test,
                    ALPHA
                    * raw150_test,
                ],
                axis=1,
            )
        )

        dimension_rows.append({
            "split_name":
                split_name,

            "train_n":
                len(train_rows),

            "test_n":
                len(test_rows),

            "compact_dim":
                len(compact_names),

            "old20_usable_dim":
                len(old20_usable),

            "raw150_usable_dim":
                len(raw150_usable),

            "evidence150_usable_dim":
                len(evidence150_usable),
        })

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus_raw150",
                x_raw150_train,
                x_raw150_test,
            ),

            (
                "compact_plus_evidence150",
                x_evidence_train,
                x_evidence_test,
            ),

            (
                "compact_plus20_plus_raw150",
                x_20_150_train,
                x_20_150_test,
            ),
        )

        print()
        print(
            split_name,
            "train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "raw150=",
            len(raw150_usable),
            "evidence150=",
            len(evidence150_usable),
        )

        for seed in SEEDS:
            for (
                variant,
                x_train,
                x_test,
            ) in variants:

                metric_rows.append(
                    evaluate(
                        variant,
                        split_name,
                        seed,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
                )

        print(
            "完成",
            flush=True,
        )

    per_split, summary = (
        summarize(
            metric_rows
        )
    )

    # 证据量总体分布
    audit_filtered = [
        row
        for row in evidence_audit
        if row["sha256"]
        in all_shas
    ]

    audit_summary = []

    for anchor in TOP5:
        values = np.asarray(
            [
                row["evidence_n"]
                for row in audit_filtered
                if row["anchor"]
                == anchor
            ],
            dtype=float,
        )

        conf = np.asarray(
            [
                row["confidence"]
                for row in audit_filtered
                if row["anchor"]
                == anchor
            ],
            dtype=float,
        )

        audit_summary.append({
            "anchor":
                anchor,

            "mean_n":
                float(values.mean()),

            "median_n":
                float(np.median(values)),

            "p90_n":
                float(
                    np.percentile(
                        values,
                        90,
                    )
                ),

            "zero_n_rate":
                float(
                    np.mean(
                        values == 0
                    )
                ),

            "mean_confidence":
                float(conf.mean()),
        })

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_split.csv",
        per_split,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT_DIR
        / "dimensions.csv",
        dimension_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "evidence_audit.csv",
        audit_filtered,
    )

    write_csv(
        OUTPUT_DIR
        / "evidence_summary.csv",
        audit_summary,
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017结果"
    )
    print(
        "======================================"
    )

    for row in summary:
        print(
            f"{row['variant']:30s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"std={row['std_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    lookup = {
        row["variant"]:
            row
        for row in summary
    }

    raw = lookup[
        "compact_plus_raw150"
    ]

    evidence = lookup[
        "compact_plus_evidence150"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "证据量感知150 相对 原始150"
    )
    print(
        "======================================"
    )

    print(
        "dBA =",
        f"{evidence['mean_BA'] - raw['mean_BA']:+.6f}",
    )

    print(
        "dMin =",
        f"{evidence['min_BA'] - raw['min_BA']:+.6f}",
    )

    print(
        "dStd =",
        f"{evidence['std_BA'] - raw['std_BA']:+.6f}",
    )

    print(
        "dAUC =",
        f"{evidence['mean_AUC'] - raw['mean_AUC']:+.6f}",
    )

    print()
    print(
        "======================================"
    )
    print(
        "锚点结构证据量"
    )
    print(
        "======================================"
    )

    for row in audit_summary:
        print(
            f"{row['anchor']:28s} "
            f"mean_n={row['mean_n']:.2f} "
            f"median_n={row['median_n']:.1f} "
            f"p90={row['p90_n']:.1f} "
            f"zero={row['zero_n_rate']:.2%} "
            f"mean_conf="
            f"{row['mean_confidence']:.3f}"
        )


if __name__ == "__main__":
    main()
