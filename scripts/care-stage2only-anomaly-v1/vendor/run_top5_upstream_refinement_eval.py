#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import (  # noqa: E402
    read_csv_rows,
    read_split_rows,
)
from run_dual_view_error_attribution import (  # noqa: E402
    compact_matrix_named,
)
from run_locked_dual_view_module_ablation import (  # noqa: E402
    predict_logistic,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from run_sensitive_upstream_structural_eval import (  # noqa: E402
    label_value,
    read_gz_csv,
    standardize_named,
)


THRESHOLD = 0.5

# Frozen from 2017 Compact error analysis.
# NOT selected using upstream performance or future-year labels.
TOP5_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

CONTEXT_SUFFIXES = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_valid(values):
    xs = [float(x) for x in values if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--compact-features", required=True, type=Path)
    ap.add_argument("--compact-schema", required=True, type=Path)
    ap.add_argument("--upstream-features", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.03)

    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    compact_rows = read_csv_rows(args.compact_features)
    compact_by_sha = {
        str(r["sha256"]).strip().casefold(): r
        for r in compact_rows
    }

    compact_schema = json.loads(
        args.compact_schema.read_text(encoding="utf-8")
    )

    upstream_rows = read_gz_csv(args.upstream_features)
    upstream_by_sha = {
        str(r["sha256"]).strip().casefold(): r
        for r in upstream_rows
    }

    if not upstream_rows:
        raise SystemExit("empty upstream CSV")

    upstream_header = set(upstream_rows[0])

    top5_names = [
        f"up_role__{role}__{suffix}"
        for role in TOP5_ROLES
        for suffix in CONTEXT_SUFFIXES
    ]

    if len(top5_names) != 20:
        raise SystemExit(
            f"expected 20 Top5 context features, got {len(top5_names)}"
        )

    missing_columns = [
        name for name in top5_names
        if name not in upstream_header
    ]

    if missing_columns:
        raise SystemExit(
            f"missing Top5 upstream columns: {missing_columns}"
        )

    print("=== FROZEN TOP5 ROLES ===")
    for role in TOP5_ROLES:
        print(role)

    print()
    print("Top5 context features =", len(top5_names))
    for name in top5_names:
        print(" ", name)

    print()
    print("alpha =", args.alpha)
    print("threshold =", THRESHOLD)
    print("l2 =", args.l2)

    splits = read_split_rows(args.splits)

    selected_shas = {
        row["sha256"]
        for split in splits.values()
        for partition in ("train", "test")
        for row in split[partition]
    }

    missing_compact = selected_shas - set(compact_by_sha)
    missing_upstream = selected_shas - set(upstream_by_sha)

    print()
    print("selected APKs =", len(selected_shas))
    print("missing compact =", len(missing_compact))
    print("missing upstream =", len(missing_upstream))

    if missing_compact or missing_upstream:
        raise SystemExit(
            "selected rows missing; do not silently change population"
        )

    variants = (
        "compact_only",
        "top5_upstream_only",
        f"compact_plus_top5_upstream_w{args.alpha:g}",
    )

    per_split_rows = []
    prediction_rows = []
    dimension_rows = []

    for split_name in sorted(splits):
        train_rows = splits[split_name]["train"]
        test_rows = splits[split_name]["test"]

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get(
                        "metadata_label", ""
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get(
                        "metadata_label", ""
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        compact_train, compact_test, compact_names = (
            compact_matrix_named(
                train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
            )
        )

        top5_train, top5_test, usable_top5 = (
            standardize_named(
                train_rows,
                test_rows,
                upstream_by_sha,
                top5_names,
            )
        )

        print(
            split_name,
            "compact_dim=",
            len(compact_names),
            "top5_context_dim=",
            len(usable_top5),
            flush=True,
        )

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),

            # Diagnostic only.
            "top5_upstream_only": (
                top5_train,
                top5_test,
            ),

            f"compact_plus_top5_upstream_w{args.alpha:g}": (
                np.concatenate(
                    [
                        compact_train,
                        args.alpha * top5_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        args.alpha * top5_test,
                    ],
                    axis=1,
                ),
            ),
        }

        dimension_rows.append({
            "split_name": split_name,
            "compact_dim": len(compact_names),
            "top5_context_dim": len(usable_top5),
            "fused_dim": (
                len(compact_names) + len(usable_top5)
            ),
        })

        for variant in variants:
            x_train, x_test = matrices[variant]

            seed_metrics = []

            for seed in args.seeds:
                prob = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )

                m = metrics(
                    y_test,
                    prob,
                    threshold=THRESHOLD,
                )

                seed_metrics.append(m)

                for row, label, probability in zip(
                    test_rows,
                    y_test,
                    prob,
                ):
                    sha = row["sha256"]

                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": sha,
                        "year": compact_by_sha[sha].get(
                            "metadata_year", ""
                        ),
                        "seed": seed,
                        "label": int(label),
                        "probability": float(probability),
                        "predicted_label": int(
                            probability >= THRESHOLD
                        ),
                    })

            result = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(x_train.shape[1]),
            }

            for metric_name in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "fp",
                "fn",
            ):
                result[f"{metric_name}_mean"] = mean_valid(
                    [m[metric_name] for m in seed_metrics]
                )

            per_split_rows.append(result)

        print("completed", split_name, flush=True)

    # --------------------------------------------------
    # Annual-split aggregation.
    # --------------------------------------------------
    grouped = defaultdict(list)

    for row in per_split_rows:
        grouped[row["variant"]].append(row)

    summary_rows = []

    for variant, rows in grouped.items():
        ba = np.asarray(
            [
                float(r["balanced_accuracy_mean"])
                for r in rows
            ],
            dtype=float,
        )

        summary_rows.append({
            "variant": variant,
            "split_count": len(rows),
            "mean_BA": float(ba.mean()),
            "min_BA": float(ba.min()),
            "std_BA": float(ba.std()),
            "mean_AUC": mean_valid(
                [r["auc_mean"] for r in rows]
            ),
            "mean_ACC": mean_valid(
                [r["accuracy_mean"] for r in rows]
            ),
            "mean_F1": mean_valid(
                [r["f1_mean"] for r in rows]
            ),
            "mean_FP": mean_valid(
                [r["fp_mean"] for r in rows]
            ),
            "mean_FN": mean_valid(
                [r["fn_mean"] for r in rows]
            ),
        })

    summary_rows.sort(
        key=lambda r: r["mean_BA"],
        reverse=True,
    )

    # --------------------------------------------------
    # Seed-averaged paired correction analysis:
    # Compact vs Top5-refined.
    # --------------------------------------------------
    refined_variant = (
        f"compact_plus_top5_upstream_w{args.alpha:g}"
    )

    probability_groups = defaultdict(list)
    metadata = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )

        probability_groups[key].append(
            float(row["probability"])
        )

        metadata[
            (row["split_name"], row["sha256"])
        ] = (
            row["year"],
            int(row["label"]),
        )

    paired_rows = []

    for split_name, sha in sorted(metadata):
        year, label = metadata[(split_name, sha)]

        compact_prob = float(np.mean(
            probability_groups[
                (
                    split_name,
                    sha,
                    "compact_only",
                )
            ]
        ))

        refined_prob = float(np.mean(
            probability_groups[
                (
                    split_name,
                    sha,
                    refined_variant,
                )
            ]
        ))

        compact_pred = int(
            compact_prob >= THRESHOLD
        )

        refined_pred = int(
            refined_prob >= THRESHOLD
        )

        compact_ok = compact_pred == label
        refined_ok = refined_pred == label

        if not compact_ok and refined_ok:
            transition = "CORRECTED"
        elif compact_ok and not refined_ok:
            transition = "REGRESSED"
        elif compact_ok:
            transition = "STILL_CORRECT"
        else:
            transition = "STILL_WRONG"

        paired_rows.append({
            "split_name": split_name,
            "year": year,
            "sha256": sha,
            "label": label,
            "compact_probability": compact_prob,
            "refined_probability": refined_prob,
            "probability_delta": (
                refined_prob - compact_prob
            ),
            "compact_predicted_label": compact_pred,
            "refined_predicted_label": refined_pred,
            "transition": transition,
        })

    correction_rows = []

    groups = defaultdict(list)

    for row in paired_rows:
        groups[("ALL", "ALL")].append(row)
        groups[(str(row["year"]), "ALL")].append(row)
        groups[("ALL", str(row["label"]))].append(row)

    for (year, label), rows in sorted(groups.items()):
        counts = defaultdict(int)

        for row in rows:
            counts[row["transition"]] += 1

        corrected = counts["CORRECTED"]
        regressed = counts["REGRESSED"]

        correction_rows.append({
            "year": year,
            "label": label,
            "n": len(rows),
            "corrected": corrected,
            "regressed": regressed,
            "net": corrected - regressed,
            "still_correct": counts["STILL_CORRECT"],
            "still_wrong": counts["STILL_WRONG"],
        })

    write_csv(
        args.output / "top5_upstream_per_split.csv",
        per_split_rows,
    )

    write_csv(
        args.output / "top5_upstream_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output / "top5_upstream_predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output / "top5_upstream_paired_apk.csv",
        paired_rows,
    )

    write_csv(
        args.output / "top5_upstream_correction_summary.csv",
        correction_rows,
    )

    write_csv(
        args.output / "top5_upstream_dimensions.csv",
        dimension_rows,
    )

    print()
    print("=== TOP5 NO-ENTRY UPSTREAM RESULTS ===")

    for row in summary_rows:
        print(
            f"{row['variant']:42s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"F1={row['mean_F1']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    overall = next(
        row
        for row in correction_rows
        if row["year"] == "ALL"
        and row["label"] == "ALL"
    )

    print()
    print("=== COMPACT -> TOP5 REFINEMENT ===")
    print("corrected =", overall["corrected"])
    print("regressed =", overall["regressed"])
    print("net =", overall["net"])

    print()
    print("=== LABEL-CONDITIONED ===")
    for row in correction_rows:
        if row["year"] == "ALL" and row["label"] != "ALL":
            name = (
                "BENIGN"
                if row["label"] == "0"
                else "MALWARE"
            )
            print(
                f"{name:8s} "
                f"N={row['n']} "
                f"corrected={row['corrected']} "
                f"regressed={row['regressed']} "
                f"net={row['net']}"
            )

    print()
    print("=== PER YEAR ===")
    for row in correction_rows:
        if row["year"] != "ALL" and row["label"] == "ALL":
            print(
                f"{row['year']:4s} "
                f"N={row['n']} "
                f"corrected={row['corrected']} "
                f"regressed={row['regressed']} "
                f"net={row['net']}"
            )

    print()
    print(
        "Protocol note: Top5 roles are frozen from "
        "Compact source-year error analysis. "
        "No future-year role selection is performed."
    )


if __name__ == "__main__":
    main()
