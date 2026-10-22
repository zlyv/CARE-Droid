#!/usr/bin/env python3
"""Read-only APK-paired audit of the four completed source arms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_core import exact_mcnemar, metrics, percentile_interval, stratified_bootstrap_indices

ARMS = ("N0_NO_GRAPHGLOBAL", "D_DELTA_ONLY", "Q_QUERY_ONLY", "QD_QUERY_AND_DELTA")
Q = "Q_QUERY_ONLY"


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def probability_column(arm):
    return f"DI25__Host__Anomaly__GraphGlobalPath__{arm}__Bounded__prob"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("artifacts/care-di25-graphglobal-path-source-v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/care-di25-graphglobal-path-paired-audit-v1"))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise RuntimeError("at least 1000 bootstrap iterations are required")

    reference_rows = read(args.input / "preflight/di25_reference/source_predictions.csv")
    reference_rows.sort(key=lambda r: r["sha256"].lower())
    ids = [r["sha256"].lower() for r in reference_rows]
    y = np.asarray([int(float(r["label"])) for r in reference_rows], dtype=np.int64)
    folds = np.asarray([int(float(r["fold"])) for r in reference_rows], dtype=np.int64)
    route = np.asarray([int(float(r["A25__routed"])) for r in reference_rows], dtype=np.int64)
    p_stage1 = np.asarray([float(r["Global24__prob"]) for r in reference_rows])

    probabilities = {}
    for arm in ARMS:
        rows = read(args.input / "arms" / arm / "source_predictions.csv")
        by_sha = {r["sha256"].lower(): r for r in rows}
        column = probability_column(arm)
        if any(column not in by_sha[sha] for sha in ids):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {column}")
        probabilities[arm] = np.asarray([float(by_sha[sha][column]) for sha in ids])

    aggregate_rows, fold_rows = [], []
    for arm in ARMS:
        aggregate_rows.append({"arm": arm, **metrics(y, probabilities[arm], p_stage1)})
        for fold in sorted(set(folds)):
            selected = folds == fold
            fold_rows.append({
                "arm": arm, "fold": int(fold),
                **metrics(y[selected], probabilities[arm][selected], p_stage1[selected]),
            })

    paired_rows, mcnemar_rows, fold_delta_rows = [], [], []
    q_pred = probabilities[Q] >= 0.5
    q_correct = q_pred == y
    for other in (arm for arm in ARMS if arm != Q):
        other_pred = probabilities[other] >= 0.5
        other_correct = other_pred == y
        n01 = int(np.sum((~other_correct) & q_correct))  # other wrong, Q correct
        n10 = int(np.sum(other_correct & (~q_correct)))  # other correct, Q wrong
        mcnemar_rows.append({
            "comparison": f"{Q}_minus_{other}",
            "other_wrong_Q_correct_n01": n01,
            "other_correct_Q_wrong_n10": n10,
            "paired_net_accuracy": n01 - n10,
            "exact_mcnemar_p": exact_mcnemar(n01, n10),
        })
        for index, sha in enumerate(ids):
            if q_correct[index] and other_correct[index]: status = "same_correct"
            elif (not q_correct[index]) and (not other_correct[index]): status = "same_wrong"
            elif q_correct[index]: status = "other_wrong_Q_correct"
            else: status = "other_correct_Q_wrong"
            paired_rows.append({
                "comparison": f"{Q}_minus_{other}", "sha256": sha,
                "label": int(y[index]), "fold": int(folds[index]),
                "route_DI": int(route[index]), "p_stage1": p_stage1[index],
                "p_other": probabilities[other][index], "p_Q": probabilities[Q][index],
                "other_correct": int(other_correct[index]), "Q_correct": int(q_correct[index]),
                "status": status,
            })
        for fold in sorted(set(folds)):
            selected = folds == fold
            mq = metrics(y[selected], probabilities[Q][selected], p_stage1[selected])
            mo = metrics(y[selected], probabilities[other][selected], p_stage1[selected])
            fold_delta_rows.append({
                "comparison": f"{Q}_minus_{other}", "fold": int(fold),
                "delta_BA": mq["BA"] - mo["BA"],
                "delta_AUC": mq["AUC"] - mo["AUC"],
                "delta_F1": mq["F1"] - mo["F1"],
                "delta_fixed": mq["fixed"] - mo["fixed"],
                "delta_broken": mq["broken"] - mo["broken"],
                "delta_net": mq["net"] - mo["net"],
            })

    # Reuse the same stratified resamples for every paired comparison.
    boot_indices = list(stratified_bootstrap_indices(
        y, args.bootstrap_iterations, args.bootstrap_seed
    ))
    bootstrap_rows = []
    for other in (arm for arm in ARMS if arm != Q):
        observed_q = metrics(y, probabilities[Q], p_stage1)
        observed_o = metrics(y, probabilities[other], p_stage1)
        samples = {name: [] for name in ("BA", "AUC", "F1", "fixed", "broken", "net")}
        for index in boot_indices:
            mq = metrics(y[index], probabilities[Q][index], p_stage1[index])
            mo = metrics(y[index], probabilities[other][index], p_stage1[index])
            for name in samples:
                samples[name].append(mq[name] - mo[name])
        for name, values in samples.items():
            ci = percentile_interval(values)
            bootstrap_rows.append({
                "comparison": f"{Q}_minus_{other}", "metric": name,
                "observed_delta": observed_q[name] - observed_o[name],
                "bootstrap_iterations": args.bootstrap_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                **ci,
            })

    write(args.output / "aggregate_metrics.csv", aggregate_rows)
    write(args.output / "per_fold_metrics.csv", fold_rows)
    write(args.output / "Q_paired_sample_changes.csv", paired_rows)
    write(args.output / "Q_exact_mcnemar.csv", mcnemar_rows)
    write(args.output / "Q_per_fold_deltas.csv", fold_delta_rows)
    write(args.output / "Q_stratified_bootstrap.csv", bootstrap_rows)

    by_arm = {r["arm"]: r for r in aggregate_rows}
    q = by_arm[Q]
    others = [by_arm[a] for a in ARMS if a != Q]
    aggregate_dominance = (
        all(q["BA"] > row["BA"] for row in others)
        and all(q["net"] > row["net"] for row in others)
        and all(q["broken"] <= row["broken"] for row in others)
    )
    n0_fold = [r for r in fold_delta_rows if r["comparison"].endswith("N0_NO_GRAPHGLOBAL")]
    # Treat floating-point roundoff around exact equality as zero.  Without
    # this tolerance, a mathematically tied fold (e.g. 1.11e-16) is
    # incorrectly counted as a positive fold and can flip the freeze rule.
    fold_tolerance = 1e-12
    q_vs_n0_positive_folds = sum(
        float(r["delta_BA"]) > fold_tolerance for r in n0_fold
    )
    q_vs_n0_nonnegative_net_folds = sum(
        float(r["delta_net"]) >= -fold_tolerance for r in n0_fold
    )
    # This is a source-side candidate rule, not an independent confirmation.
    freeze_candidate = bool(
        aggregate_dominance
        and q_vs_n0_positive_folds >= 3
        and q_vs_n0_nonnegative_net_folds >= 3
    )
    decision = {
        "source_only": True,
        "future_rows_loaded": 0,
        "training_invoked": False,
        "aggregate_Q_dominates_BA_net_without_more_broken": aggregate_dominance,
        "Q_vs_N0_positive_BA_folds": q_vs_n0_positive_folds,
        "Q_vs_N0_nonnegative_net_folds": q_vs_n0_nonnegative_net_folds,
        "fold_delta_zero_tolerance": fold_tolerance,
        "source_freeze_candidate": freeze_candidate,
        "selected_candidate_if_human_approved": Q if freeze_candidate else None,
        "evidence_boundary": (
            "The four source arms and the selection rule were inspected during method development. "
            "This is source-side development evidence, not independent temporal confirmation."
        ),
    }
    (args.output / "source_freeze_candidate.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "CARE-Droid GraphGlobal path paired source audit v1", "",
        "Read-only strict-2017 APK-paired analysis; frozen DI25=200/799.",
        "No training, future row, route change, prediction change, or Ego input.", "",
        "Aggregate metrics:",
    ]
    for row in aggregate_rows:
        lines.append(
            f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    lines += ["", "Exact paired Q comparisons:"]
    for row in mcnemar_rows:
        lines.append(
            f"{row['comparison']}: n01={row['other_wrong_Q_correct_n01']} "
            f"n10={row['other_correct_Q_wrong_n10']} paired_net={row['paired_net_accuracy']} "
            f"p={row['exact_mcnemar_p']:.6f}"
        )
    lines += ["", "Fold consistency:",
              f"Q vs N0 positive-BA folds={q_vs_n0_positive_folds}/5",
              f"Q vs N0 nonnegative-net folds={q_vs_n0_nonnegative_net_folds}/5", "",
              "SOURCE_FREEZE_CANDIDATE=" + (Q if freeze_candidate else "NONE"),
              "This candidate still requires human approval before a frozen future diagnostic.",
              "Bootstrap intervals quantify sampling uncertainty; they do not make this post-hoc development audit independent."]
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GRAPHGLOBAL_PATH_PAIRED_AUDIT_DONE")
    print((args.output / "REPORT.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

