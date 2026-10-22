#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)
from run_dual_view_error_attribution import compact_matrix_named
from run_locked_dual_view_module_ablation import predict_logistic
from run_pilot_compact_eval import metrics
from run_sensitive_upstream_structural_eval import (
    label_value,
    read_gz_csv,
    standardize_named,
)


THRESHOLD = 0.5

TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

SUFFIXES = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)

TOP5_NAMES = [
    f"up_role__{role}__{suffix}"
    for role in TOP5
    for suffix in SUFFIXES
]


def k(x):
    return str(x).strip().casefold()


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def raw_matrix(rows, by_sha, names):
    x = np.zeros(
        (len(rows), len(names)),
        dtype=np.float64,
    )

    for i, row in enumerate(rows):
        src = by_sha[k(row["sha256"])]

        for j, name in enumerate(names):
            try:
                value = float(src.get(name, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0

            if np.isfinite(value):
                x[i, j] = value

    return x


def zero_preserving_scale(train, test):
    """
    Scale by training-fold standard deviation WITHOUT centering.

    Therefore raw all-zero context remains exactly all-zero:
        U_raw = 0 -> U_scaled = 0
    """
    std = train.std(axis=0)

    usable = std > 1e-12

    if not np.any(usable):
        return (
            np.zeros((len(train), 0)),
            np.zeros((len(test), 0)),
            usable,
        )

    scale = std[usable]

    return (
        train[:, usable] / scale,
        test[:, usable] / scale,
        usable,
    )


def fit_residual_ridge(
    u_train,
    residual,
    l2,
):
    """
    Linear fit to Bernoulli negative-gradient residual:
        r = y - p_compact

    No intercept:
        no caller context -> delta = 0.
    """
    d = u_train.shape[1]

    if d == 0:
        return np.zeros(0, dtype=np.float64)

    n = max(1, len(u_train))

    a = (
        (u_train.T @ u_train) / n
        + l2 * np.eye(d)
    )

    b = (u_train.T @ residual) / n

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def summarize(rows):
    out = {}

    for name in (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "fp",
        "fn",
    ):
        out[name] = float(
            np.mean([float(r[name]) for r in rows])
        )

    return out


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--compact-features",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--compact-schema",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--upstream-features",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--splits",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )

    ap.add_argument(
        "--residual-l2",
        type=float,
        default=0.03,
    )

    ap.add_argument(
        "--joint-alpha",
        type=float,
        default=0.25,
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
        "--compact-l2",
        type=float,
        default=0.03,
    )

    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    compact_rows = read_csv_rows(
        args.compact_features
    )

    compact_by_sha = {
        k(r["sha256"]): r
        for r in compact_rows
    }

    import json

    compact_schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    upstream_rows = read_gz_csv(
        args.upstream_features
    )

    upstream_by_sha = {
        k(r["sha256"]): r
        for r in upstream_rows
    }

    splits = read_split_rows(args.splits)

    print("Top5 context dim =", len(TOP5_NAMES))
    print("threshold =", THRESHOLD)
    print("residual L2 =", args.residual_l2)
    print("gammas =", args.gammas)
    print()

    per_split = []

    for split_name in sorted(splits):
        train_rows = splits[split_name]["train"]
        test_rows = splits[split_name]["test"]

        y_train = np.asarray([
            label_value(
                compact_by_sha[
                    k(r["sha256"])
                ].get("metadata_label", "")
            )
            for r in train_rows
        ], dtype=np.int64)

        y_test = np.asarray([
            label_value(
                compact_by_sha[
                    k(r["sha256"])
                ].get("metadata_label", "")
            )
            for r in test_rows
        ], dtype=np.int64)

        c_train, c_test, compact_names = (
            compact_matrix_named(
                train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
            )
        )

        # Raw Top5 caller evidence.
        u_raw_train = raw_matrix(
            train_rows,
            upstream_by_sha,
            TOP5_NAMES,
        )

        u_raw_test = raw_matrix(
            test_rows,
            upstream_by_sha,
            TOP5_NAMES,
        )

        # New correction transform:
        # variance scaling, NO mean centering.
        u_train, u_test, usable = (
            zero_preserving_scale(
                u_raw_train,
                u_raw_test,
            )
        )

        # Existing joint representation for exact control.
        joint_train_u, joint_test_u, _ = (
            standardize_named(
                train_rows,
                test_rows,
                upstream_by_sha,
                TOP5_NAMES,
            )
        )

        seed_results = defaultdict(list)

        for seed in args.seeds:

            # ------------------------------------
            # Train Compact exactly once and obtain
            # train + validation probabilities from
            # the SAME fitted model.
            # ------------------------------------
            all_eval = np.concatenate(
                [c_train, c_test],
                axis=0,
            )

            p_all = predict_logistic(
                c_train,
                y_train,
                all_eval,
                seed,
                args.iterations,
                args.learning_rate,
                args.compact_l2,
            )

            p_c_train = np.asarray(
                p_all[:len(c_train)],
                dtype=np.float64,
            )

            p_c_test = np.asarray(
                p_all[len(c_train):],
                dtype=np.float64,
            )

            seed_results["compact_only"].append(
                metrics(
                    y_test,
                    p_c_test,
                    threshold=THRESHOLD,
                )
            )

            # ------------------------------------
            # Existing concat/joint LR control.
            # ------------------------------------
            xj_train = np.concatenate(
                [
                    c_train,
                    args.joint_alpha * joint_train_u,
                ],
                axis=1,
            )

            xj_test = np.concatenate(
                [
                    c_test,
                    args.joint_alpha * joint_test_u,
                ],
                axis=1,
            )

            p_joint = predict_logistic(
                xj_train,
                y_train,
                xj_test,
                seed,
                args.iterations,
                args.learning_rate,
                args.compact_l2,
            )

            seed_results[
                f"joint_top5_a{args.joint_alpha:g}"
            ].append(
                metrics(
                    y_test,
                    p_joint,
                    threshold=THRESHOLD,
                )
            )

            # ------------------------------------
            # Error residual:
            #
            # correct + confident -> near 0
            # FP -> negative target
            # FN -> positive target
            # ------------------------------------
            residual = (
                y_train.astype(np.float64)
                - p_c_train
            )

            w = fit_residual_ridge(
                u_train,
                residual,
                args.residual_l2,
            )

            delta_test = u_test @ w
            z_c_test = logit(p_c_test)

            for gamma in args.gammas:
                p_final = sigmoid(
                    z_c_test
                    + gamma * delta_test
                )

                name = (
                    f"residual_gamma{gamma:g}"
                )

                m = metrics(
                    y_test,
                    p_final,
                    threshold=THRESHOLD,
                )

                m["mean_abs_delta"] = float(
                    np.mean(
                        np.abs(
                            gamma * delta_test
                        )
                    )
                )

                seed_results[name].append(m)

        for variant, rr in seed_results.items():
            s = summarize(rr)

            per_split.append({
                "split_name": split_name,
                "variant": variant,
                "balanced_accuracy": s[
                    "balanced_accuracy"
                ],
                "accuracy": s["accuracy"],
                "auc": s["auc"],
                "f1": s["f1"],
                "fp": s["fp"],
                "fn": s["fn"],
                "usable_context_dim": int(
                    usable.sum()
                ),
                "mean_abs_delta": float(
                    np.mean([
                        r.get(
                            "mean_abs_delta",
                            0.0,
                        )
                        for r in rr
                    ])
                ),
            })

        print(
            "completed",
            split_name,
            "usable_context_dim=",
            int(usable.sum()),
            flush=True,
        )

    write_csv(
        args.output / "residual_per_split.csv",
        per_split,
    )

    grouped = defaultdict(list)

    for r in per_split:
        grouped[r["variant"]].append(r)

    summary = []

    for variant, rr in grouped.items():
        ba = np.asarray([
            float(r["balanced_accuracy"])
            for r in rr
        ])

        summary.append({
            "variant": variant,
            "mean_BA": float(ba.mean()),
            "min_BA": float(ba.min()),
            "std_BA": float(ba.std()),
            "mean_AUC": float(np.mean([
                float(r["auc"]) for r in rr
            ])),
            "mean_ACC": float(np.mean([
                float(r["accuracy"]) for r in rr
            ])),
            "mean_F1": float(np.mean([
                float(r["f1"]) for r in rr
            ])),
            "mean_FP": float(np.mean([
                float(r["fp"]) for r in rr
            ])),
            "mean_FN": float(np.mean([
                float(r["fn"]) for r in rr
            ])),
            "mean_abs_delta": float(np.mean([
                float(r["mean_abs_delta"])
                for r in rr
            ])),
        })

    summary.sort(
        key=lambda r: r["mean_BA"],
        reverse=True,
    )

    write_csv(
        args.output / "residual_summary.csv",
        summary,
    )

    print()
    print("=== 2017 RESIDUAL CORRECTION ===")
    print(
        "variant                      "
        "mean_BA   min_BA    AUC       "
        "FP      FN      |delta|"
    )

    for r in summary:
        print(
            f"{r['variant']:28s} "
            f"{r['mean_BA']:.6f} "
            f"{r['min_BA']:.6f} "
            f"{r['mean_AUC']:.6f} "
            f"{r['mean_FP']:7.2f} "
            f"{r['mean_FN']:7.2f} "
            f"{r['mean_abs_delta']:.5f}"
        )

    residual_rows = [
        r
        for r in summary
        if r["variant"].startswith(
            "residual_gamma"
        )
    ]

    best = max(
        residual_rows,
        key=lambda r: r["mean_BA"],
    )

    print()
    print(
        "BEST RESIDUAL BY 2017 MEAN BA:",
        best["variant"],
        f"BA={best['mean_BA']:.6f}",
        f"minBA={best['min_BA']:.6f}",
        f"AUC={best['mean_AUC']:.6f}",
    )


if __name__ == "__main__":
    main()
