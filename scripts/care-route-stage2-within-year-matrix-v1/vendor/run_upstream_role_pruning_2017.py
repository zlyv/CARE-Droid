#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows  # noqa: E402
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
from run_locked_dual_view_module_ablation import predict_logistic  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402
from run_sensitive_upstream_structural_eval import (  # noqa: E402
    label_value,
    read_gz_csv,
    standardize_named,
)

THRESHOLD = 0.5

ROLE_STRUCT_SUFFIXES = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)

# These are controls, NOT automatically deleted.
LOW_COVERAGE_CONTROL = {
    "accessibility_control",
    "boot_persistence",
    "contacts_or_calendar_access",
    "device_administration",
    "other",
    "package_installation",
    "package_uninstallation",
}

COMMON_ROLE_CONTROL = {
    "dynamic_code_loading",
    "file_read",
    "overlay_or_ui_deception",
    "reflective_invocation",
    "sensitive_file_or_content_write",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(values: list[dict]) -> dict:
    out = {}
    for name in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "fp",
        "fn",
    ):
        xs = [float(v[name]) for v in values]
        out[name] = float(np.mean(xs))
    return out


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

    header = list(upstream_rows[0])
    full_names = [x for x in header if x.startswith("up_")]

    if len(full_names) != 176:
        raise SystemExit(
            f"expected 176 formal up_* features, got {len(full_names)}"
        )

    inventory_names = [
        name
        for name in full_names
        if (
            name == "up_global_endpoint_log"
            or name == "up_global_role_coverage"
            or name.endswith("__endpoint_log")
        )
    ]

    global_structure_names = [
        name
        for name in full_names
        if name.startswith("up_global_")
        and name not in inventory_names
    ]

    role_to_features: dict[str, list[str]] = defaultdict(list)

    for name in full_names:
        if not name.startswith("up_role__"):
            continue
        if name.endswith("__endpoint_log"):
            continue

        body = name[len("up_role__"):]
        role, suffix = body.rsplit("__", 1)

        if suffix in ROLE_STRUCT_SUFFIXES:
            role_to_features[role].append(name)

    role_to_features = {
        role: sorted(names)
        for role, names in sorted(role_to_features.items())
    }

    print("roles =", len(role_to_features))
    print("global structure =", len(global_structure_names))

    for role, names in role_to_features.items():
        if len(names) != 4:
            raise SystemExit(
                f"unexpected structural width for {role}: {len(names)}"
            )

    all_role_names = [
        name
        for role in sorted(role_to_features)
        for name in role_to_features[role]
    ]

    baseline_names = global_structure_names + all_role_names

    print("all-role structural =", len(all_role_names))
    print("baseline structural =", len(baseline_names))
    print("alpha =", args.alpha)
    print("threshold =", THRESHOLD)

    variants: dict[str, list[str] | None] = {
        "compact_only": None,
        "all_roles": baseline_names,
    }

    # Leave-one-role-out.
    for role, dropped in role_to_features.items():
        dropped_set = set(dropped)
        variants[f"drop__{role}"] = [
            name
            for name in baseline_names
            if name not in dropped_set
        ]

    # Predefined group controls. These do NOT define the selected model.
    for group_name, roles in (
        ("drop_low_coverage_control", LOW_COVERAGE_CONTROL),
        ("drop_common_role_control", COMMON_ROLE_CONTROL),
        (
            "drop_low_and_common_control",
            LOW_COVERAGE_CONTROL | COMMON_ROLE_CONTROL,
        ),
    ):
        drop_features = {
            feature
            for role in roles
            for feature in role_to_features.get(role, [])
        }
        variants[group_name] = [
            name
            for name in baseline_names
            if name not in drop_features
        ]

    splits = read_split_rows(args.splits)

    selected = {
        row["sha256"]
        for split in splits.values()
        for partition in ("train", "test")
        for row in split[partition]
    }

    missing_compact = selected - set(compact_by_sha)
    missing_upstream = selected - set(upstream_by_sha)

    print("selected APKs =", len(selected))
    print("missing compact =", len(missing_compact))
    print("missing upstream =", len(missing_upstream))

    if missing_compact or missing_upstream:
        raise SystemExit("missing selected rows; aborting")

    per_split_rows = []

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

        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
        )

        for variant, upstream_names in variants.items():
            if upstream_names is None:
                x_train = compact_train
                x_test = compact_test
                usable_count = 0
            else:
                u_train, u_test, usable = standardize_named(
                    train_rows,
                    test_rows,
                    upstream_by_sha,
                    upstream_names,
                )

                x_train = np.concatenate(
                    [compact_train, args.alpha * u_train],
                    axis=1,
                )
                x_test = np.concatenate(
                    [compact_test, args.alpha * u_test],
                    axis=1,
                )
                usable_count = len(usable)

            seed_results = []

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

                seed_results.append(
                    metrics(
                        y_test,
                        prob,
                        threshold=THRESHOLD,
                    )
                )

            sm = summarize(seed_results)

            per_split_rows.append({
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "compact_dim": len(compact_names),
                "usable_upstream_dim": usable_count,
                "balanced_accuracy": sm["balanced_accuracy"],
                "accuracy": sm["accuracy"],
                "auc": sm["auc"],
                "f1": sm["f1"],
                "fp": sm["fp"],
                "fn": sm["fn"],
            })

        print("completed", split_name, flush=True)

    write_csv(
        args.output / "role_pruning_per_split.csv",
        per_split_rows,
    )

    grouped = defaultdict(list)
    for row in per_split_rows:
        grouped[row["variant"]].append(row)

    summary_rows = []

    for variant, rows in grouped.items():
        bas = np.asarray(
            [float(r["balanced_accuracy"]) for r in rows],
            dtype=float,
        )

        summary_rows.append({
            "variant": variant,
            "split_count": len(rows),
            "mean_BA": float(bas.mean()),
            "min_BA": float(bas.min()),
            "std_BA": float(bas.std()),
            "mean_AUC": float(
                np.mean([float(r["auc"]) for r in rows])
            ),
            "mean_ACC": float(
                np.mean([float(r["accuracy"]) for r in rows])
            ),
            "mean_F1": float(
                np.mean([float(r["f1"]) for r in rows])
            ),
            "mean_FP": float(
                np.mean([float(r["fp"]) for r in rows])
            ),
            "mean_FN": float(
                np.mean([float(r["fn"]) for r in rows])
            ),
        })

    summary_rows.sort(
        key=lambda r: r["mean_BA"],
        reverse=True,
    )

    write_csv(
        args.output / "role_pruning_summary.csv",
        summary_rows,
    )

    summary_by_variant = {
        r["variant"]: r
        for r in summary_rows
    }

    split_by_variant = defaultdict(dict)
    for row in per_split_rows:
        split_by_variant[row["variant"]][
            row["split_name"]
        ] = row

    baseline = summary_by_variant["all_roles"]
    baseline_splits = split_by_variant["all_roles"]

    n_splits = len(baseline_splits)
    majority = n_splits // 2 + 1

    role_rows = []

    for role in sorted(role_to_features):
        variant = f"drop__{role}"
        current = summary_by_variant[variant]

        delta_mean = (
            float(current["mean_BA"])
            - float(baseline["mean_BA"])
        )

        delta_min = (
            float(current["min_BA"])
            - float(baseline["min_BA"])
        )

        improved = 0
        not_worse = 0
        deltas = []

        for split_name, base_row in baseline_splits.items():
            cur_row = split_by_variant[variant][split_name]

            delta = (
                float(cur_row["balanced_accuracy"])
                - float(base_row["balanced_accuracy"])
            )
            deltas.append(delta)

            if delta > 1e-12:
                improved += 1
            if delta >= -1e-12:
                not_worse += 1

        # Conservative source-only candidate:
        # mean BA must improve, and deletion must be non-worse
        # in a majority of rolling splits.
        candidate = (
            delta_mean > 0.0
            and not_worse >= majority
        )

        role_rows.append({
            "role": role,
            "drop_mean_BA": float(current["mean_BA"]),
            "delta_mean_BA": delta_mean,
            "drop_min_BA": float(current["min_BA"]),
            "delta_min_BA": delta_min,
            "drop_mean_AUC": float(current["mean_AUC"]),
            "improved_splits": improved,
            "not_worse_splits": not_worse,
            "total_splits": n_splits,
            "candidate_noise_role": int(candidate),
            "split_deltas": ";".join(
                f"{x:+.6f}" for x in deltas
            ),
        })

    role_rows.sort(
        key=lambda r: r["delta_mean_BA"],
        reverse=True,
    )

    write_csv(
        args.output / "role_pruning_candidates.csv",
        role_rows,
    )

    print()
    print("=== 2017 ROLLING ROLE PRUNING ===")
    print(
        f"baseline all_roles: "
        f"BA={float(baseline['mean_BA']):.6f} "
        f"min={float(baseline['min_BA']):.6f} "
        f"AUC={float(baseline['mean_AUC']):.6f}"
    )

    print()
    print(
        "role                                      "
        "delta_BA    delta_min   nonworse candidate"
    )

    for row in role_rows:
        print(
            f"{row['role']:40s} "
            f"{row['delta_mean_BA']:+.6f} "
            f"{row['delta_min_BA']:+.6f} "
            f"{row['not_worse_splits']:2d}/{row['total_splits']:<2d} "
            f"{row['candidate_noise_role']}"
        )

    print()
    print("=== PREDEFINED GROUP CONTROLS ===")

    for name in (
        "drop_low_coverage_control",
        "drop_common_role_control",
        "drop_low_and_common_control",
    ):
        row = summary_by_variant[name]
        delta = (
            float(row["mean_BA"])
            - float(baseline["mean_BA"])
        )
        print(
            f"{name:32s} "
            f"BA={float(row['mean_BA']):.6f} "
            f"delta={delta:+.6f} "
            f"AUC={float(row['mean_AUC']):.6f}"
        )

    candidates = [
        row
        for row in role_rows
        if row["candidate_noise_role"] == 1
    ]

    print()
    print("candidate noise roles =", len(candidates))
    for row in candidates:
        print(
            f"  {row['role']:40s} "
            f"delta={row['delta_mean_BA']:+.6f}"
        )

    print()
    print(
        "NOTE: candidates are source-year diagnostics only. "
        "Do not use 2018-2021 to choose roles."
    )


if __name__ == "__main__":
    main()
