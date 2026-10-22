#!/usr/bin/env python3
"""
Post-hoc diagnostic only.

Question:
Does the frozen M0 representation already contain cross-year information
that separates:
  A) Fixed vs WrongDirection
  B) KeptCorrect vs Broken

IMPORTANT:
- Future labels are used ONLY for this diagnostic probe.
- Probe weights NEVER enter CARE-Droid.
- No hyperparameter search.
- No threshold tuning.
- Each neural seed is probed separately because latent coordinates from
  independently trained seeds must NOT be averaged.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

CAPTURE = ROOT / "instrumented_m0/routed_internal_capture.npz"
ROWS = ROOT / "analysis/routed_mechanism_rows.csv"
EQUIV = ROOT / "instrumented_m0/instrumentation_equivalence.json"
OUT = ROOT / "analysis/loyo_representation_probe"

YEARS = (2018, 2019, 2020, 2021)
SEEDS = (900000, 900001, 900002)

FEATURE_SETS = {
    # Global24 + frozen Stage-I logit
    "G25": (0, 25),

    # graph32 + critical24 + attended24 + summary8
    "L88": (25, 113),

    # Complete correction-head input
    "FULL113": (0, 113),
}


def check_contract():

    if not CAPTURE.is_file():
        raise RuntimeError("missing routed_internal_capture.npz")

    if not ROWS.is_file():
        raise RuntimeError("missing routed_mechanism_rows.csv")

    if not EQUIV.is_file():
        raise RuntimeError("missing instrumentation equivalence audit")

    eq = json.loads(EQUIV.read_text(encoding="utf-8"))

    if not eq.get("passed", False):
        raise RuntimeError("instrumented M0 did not pass equivalence")

    if float(eq["max_abs_probability_diff_vs_locked"]) > 1e-10:
        raise RuntimeError("instrumented M0 differs from locked M0")

    if int(eq["routed_n"]) != 497:
        raise RuntimeError("routed cohort changed")

    if int(eq["fused_dim"]) != 113:
        raise RuntimeError("fused dimension changed")


def load_data():

    d = pd.read_csv(ROWS)

    d["sha256"] = (
        d["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    with np.load(CAPTURE, allow_pickle=False) as z:

        ids = (
            z["sha256"]
            .astype(str)
        )

        years = np.asarray(
            z["year"],
            dtype=int,
        )

        fused = np.asarray(
            z["fused"],
            dtype=float,
        )

    if fused.shape != (3, 497, 113):
        raise RuntimeError(
            f"unexpected fused shape: {fused.shape}"
        )

    if len(set(ids)) != 497:
        raise RuntimeError("capture SHA uniqueness failed")

    aligned = (
        d.set_index("sha256")
        .loc[ids]
        .reset_index()
    )

    if not np.array_equal(
        aligned["year"].to_numpy(int),
        years,
    ):
        raise RuntimeError("year alignment failed")

    return aligned, fused, years


def build_task(d, task):

    if task == "Fixed_vs_WrongDirection":

        mask = (
            (d["cohort"] == "Fixed")
            |
            (d["failure_type"] == "WRONG_DIRECTION")
        ).to_numpy(bool)

        target = (
            d.loc[mask, "failure_type"]
            .eq("WRONG_DIRECTION")
            .to_numpy(int)
        )

        positive_name = "WrongDirection"

    elif task == "KeptCorrect_vs_Broken":

        mask = (
            d["cohort"]
            .isin(["KeptCorrect", "Broken"])
        ).to_numpy(bool)

        target = (
            d.loc[mask, "cohort"]
            .eq("Broken")
            .to_numpy(int)
        )

        positive_name = "Broken"

    else:
        raise ValueError(task)

    return mask, target, positive_name


def run_probe(
    x,
    y,
    years,
    task,
    feature_set,
    seed_index,
    positive_name,
):

    rows = []

    for heldout_year in YEARS:

        train = years != heldout_year
        test = years == heldout_year

        y_train = y[train]
        y_test = y[test]

        row = {
            "task": task,
            "feature_set": feature_set,
            "seed_index": seed_index,
            "seed": SEEDS[seed_index],
            "heldout_year": heldout_year,

            "train_n": int(train.sum()),
            "train_positive": int(y_train.sum()),
            "train_negative": int(len(y_train) - y_train.sum()),

            "test_n": int(test.sum()),
            "test_positive": int(y_test.sum()),
            "test_negative": int(len(y_test) - y_test.sum()),

            "positive_class": positive_name,
        }

        if (
            len(np.unique(y_train)) < 2
            or len(np.unique(y_test)) < 2
        ):
            row.update({
                "status": "SKIP_SINGLE_CLASS",
                "auc": np.nan,
                "pr_auc": np.nan,
                "balanced_accuracy": np.nan,
                "low_support_test": True,
            })

            rows.append(row)
            continue

        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x[train]
        )

        x_test = scaler.transform(
            x[test]
        )

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260922,
        )

        clf.fit(
            x_train,
            y_train,
        )

        probability = clf.predict_proba(
            x_test
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y_test,
            probability,
        )

        pr_auc = average_precision_score(
            y_test,
            probability,
        )

        ba = balanced_accuracy_score(
            y_test,
            prediction,
        )

        row.update({
            "status": "OK",
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "balanced_accuracy": float(ba),

            # <5 positives or negatives is considered
            # low-support diagnostic evidence.
            "low_support_test":
                bool(
                    min(
                        int(y_test.sum()),
                        int(len(y_test) - y_test.sum()),
                    )
                    < 5
                ),

            "coef_l2_norm":
                float(
                    np.linalg.norm(
                        clf.coef_
                    )
                ),
        })

        rows.append(row)

    return rows


def main():

    check_contract()

    d, fused, all_years = load_data()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    tasks = (
        "Fixed_vs_WrongDirection",
        "KeptCorrect_vs_Broken",
    )

    for task in tasks:

        mask, target, positive_name = build_task(
            d,
            task,
        )

        task_years = all_years[mask]

        print()
        print("=" * 100)
        print(task)
        print("=" * 100)

        for year in YEARS:

            m = task_years == year

            if not np.any(m):
                continue

            positive = int(target[m].sum())
            negative = int(m.sum() - positive)

            print(
                year,
                "n =",
                int(m.sum()),
                "positive =",
                positive,
                "negative =",
                negative,
            )

        for seed_index in range(3):

            for feature_set, (left, right) in FEATURE_SETS.items():

                # IMPORTANT:
                # each seed keeps its own latent coordinate system.
                x = fused[
                    seed_index,
                    mask,
                    left:right,
                ]

                rows = run_probe(
                    x=x,
                    y=target,
                    years=task_years,
                    task=task,
                    feature_set=feature_set,
                    seed_index=seed_index,
                    positive_name=positive_name,
                )

                all_rows.extend(rows)

    result = pd.DataFrame(all_rows)

    result.to_csv(
        OUT / "loyo_per_seed_per_year.csv",
        index=False,
    )

    # -------------------------------------------------
    # Aggregate the three neural seeds WITHIN each
    # held-out year. Latent vectors themselves were
    # never averaged.
    # -------------------------------------------------

    ok = result[
        result["status"] == "OK"
    ].copy()

    yearly = (
        ok.groupby(
            [
                "task",
                "feature_set",
                "heldout_year",
            ],
            as_index=False,
        )
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),

            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),

            ba_mean=("balanced_accuracy", "mean"),
            ba_sd=("balanced_accuracy", "std"),

            test_n=("test_n", "first"),
            test_positive=("test_positive", "first"),
            test_negative=("test_negative", "first"),
            low_support_test=("low_support_test", "first"),
        )
    )

    yearly.to_csv(
        OUT / "loyo_seed_aggregate_by_year.csv",
        index=False,
    )

    # -------------------------------------------------
    # Per-seed macro across valid held-out years.
    # Then aggregate across three seeds.
    # -------------------------------------------------

    seed_macro = (
        ok.groupby(
            [
                "task",
                "feature_set",
                "seed_index",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            macro_auc=("auc", "mean"),
            macro_pr_auc=("pr_auc", "mean"),
            macro_ba=("balanced_accuracy", "mean"),
            valid_years=("heldout_year", "nunique"),
        )
    )

    seed_macro.to_csv(
        OUT / "loyo_seed_macro.csv",
        index=False,
    )

    macro = (
        seed_macro.groupby(
            [
                "task",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_sd=("macro_auc", "std"),

            macro_pr_auc_mean=("macro_pr_auc", "mean"),
            macro_pr_auc_sd=("macro_pr_auc", "std"),

            macro_ba_mean=("macro_ba", "mean"),
            macro_ba_sd=("macro_ba", "std"),

            valid_years=("valid_years", "first"),
        )
    )

    macro.to_csv(
        OUT / "loyo_macro_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Also calculate macro excluding low-support test
    # years, e.g. safety probe 2021 has only 1 Broken.
    # -------------------------------------------------

    robust = ok[
        ~ok["low_support_test"]
    ].copy()

    if len(robust):

        robust_seed = (
            robust.groupby(
                [
                    "task",
                    "feature_set",
                    "seed_index",
                    "seed",
                ],
                as_index=False,
            )
            .agg(
                robust_macro_auc=("auc", "mean"),
                robust_macro_pr_auc=("pr_auc", "mean"),
                robust_macro_ba=("balanced_accuracy", "mean"),
                robust_valid_years=("heldout_year", "nunique"),
            )
        )

        robust_macro = (
            robust_seed.groupby(
                [
                    "task",
                    "feature_set",
                ],
                as_index=False,
            )
            .agg(
                robust_macro_auc_mean=("robust_macro_auc", "mean"),
                robust_macro_auc_sd=("robust_macro_auc", "std"),

                robust_macro_pr_auc_mean=("robust_macro_pr_auc", "mean"),
                robust_macro_pr_auc_sd=("robust_macro_pr_auc", "std"),

                robust_macro_ba_mean=("robust_macro_ba", "mean"),
                robust_macro_ba_sd=("robust_macro_ba", "std"),

                robust_valid_years=("robust_valid_years", "first"),
            )
        )

        robust_macro.to_csv(
            OUT / "loyo_robust_macro_summary.csv",
            index=False,
        )

    print()
    print("=" * 120)
    print("LOYO AUC BY HELD-OUT YEAR: mean +/- sd across 3 neural seeds")
    print("=" * 120)

    for task in tasks:

        print()
        print(task)

        q = yearly[
            yearly["task"] == task
        ].copy()

        for _, r in q.iterrows():

            support = (
                " LOW_SUPPORT"
                if bool(r["low_support_test"])
                else ""
            )

            print(
                f"{r['feature_set']:7s} "
                f"test={int(r['heldout_year'])} "
                f"AUC={r['auc_mean']:.4f}+/-{r['auc_sd']:.4f} "
                f"PR={r['pr_auc_mean']:.4f} "
                f"BA={r['ba_mean']:.4f} "
                f"n={int(r['test_n'])} "
                f"pos={int(r['test_positive'])}"
                f"{support}"
            )

    print()
    print("=" * 120)
    print("MACRO LOYO SUMMARY")
    print("=" * 120)

    print(
        macro.sort_values(
            ["task", "feature_set"]
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    robust_path = (
        OUT / "loyo_robust_macro_summary.csv"
    )

    if robust_path.is_file():

        robust_macro = pd.read_csv(
            robust_path
        )

        print()
        print("=" * 120)
        print("ROBUST MACRO: low-support held-out years excluded")
        print("=" * 120)

        print(
            robust_macro.sort_values(
                ["task", "feature_set"]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print()
    print("DIAGNOSTIC ONLY.")
    print("No probe parameter is part of CARE-Droid.")
    print("No hyperparameter or threshold selection was performed.")

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
