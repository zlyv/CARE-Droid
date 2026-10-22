#!/usr/bin/env python3
"""Summarize fixed OOF need-gate predictions; no training occurs here."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_common import (
    ARMS, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, YEARS, blocked,
    classification_metrics, paired_bootstrap, read_csv, write_csv,
)


def aligned(rows: list[dict], arm: str, year=None):
    selected = [row for row in rows if year is None or int(row["year"]) == year]
    selected.sort(key=lambda row: (int(row["year"]), int(row["fold"]), row["sha256"]))
    ids = [(int(row["year"]), int(row["fold"]), row["sha256"]) for row in selected]
    y = np.asarray([int(row["need_correct"]) for row in selected], dtype=np.int64)
    short = arm.split("_", 1)[0]
    score_name = f"score_{short}"
    if any(score_name not in row for row in selected):
        raise blocked(f"OOF column missing: {score_name}")
    p = np.asarray([float(row[score_name]) for row in selected], dtype=np.float64)
    return ids, y, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/care-di25-noego-needgate-probe-v1"))
    args = parser.parse_args()
    rows = read_csv(args.output / "oof_predictions.csv")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise blocked("OOF output contains duplicate SHA values")
    reference_ids, _, _ = aligned(rows, ARMS[0])
    for arm in ARMS[1:]:
        ids, _, _ = aligned(rows, arm)
        if ids != reference_ids:
            raise blocked(f"OOF cohort/order differs for {arm}")

    annual_rows = []
    pooled_rows = []
    for arm in ARMS:
        for year in YEARS:
            _, y, p = aligned(rows, arm, year)
            annual_rows.append({"arm": arm, "year": year, **classification_metrics(y, p)})
        _, y, p = aligned(rows, arm)
        pooled_rows.append({"arm": arm, "scope": "pooled_2018_2021", **classification_metrics(y, p)})
    write_csv(args.output / "metrics_by_year.csv", annual_rows)
    write_csv(args.output / "metrics_pooled.csv", pooled_rows)

    comparison_rows = []
    for scope in list(YEARS) + ["pooled_2018_2021"]:
        year = scope if isinstance(scope, int) else None
        base_ids, y, base_probability = aligned(rows, "S0_STAGE1_LOGIT", year)
        for candidate in ("S2_NOEGO_HOST_ANOMALY", "S3_NOEGO_HOST_ANOMALY_LOGIT"):
            candidate_ids, candidate_y, candidate_probability = aligned(rows, candidate, year)
            if candidate_ids != base_ids or not np.array_equal(candidate_y, y):
                raise blocked(f"paired comparison alignment failed: {scope} {candidate}")
            for metric in ("ROC_AUC", "PR_AUC"):
                result = paired_bootstrap(
                    y, base_probability, candidate_probability, metric,
                    iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED,
                )
                comparison_rows.append({
                    "scope": scope, "baseline": "S0_STAGE1_LOGIT", "candidate": candidate,
                    **result,
                })
    write_csv(args.output / "bootstrap_comparisons.csv", comparison_rows)

    pooled_by_arm = {row["arm"]: row for row in pooled_rows}
    lines = [
        "# CARE-Droid DI25 no-Ego Need-Gate Probe v1", "",
        "Same-year five-fold OOF development/diagnostic experiment. This is not temporal generalization, unseen-future, or a new CARE-Droid prediction result.", "",
        "## Pooled routed-cohort metrics", "",
        "| Arm | ROC-AUC | PR-AUC | BA | F1 | Precision | Recall | N |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = pooled_by_arm[arm]
        lines.append(
            f"| {arm} | {row['ROC_AUC']:.6f} | {row['PR_AUC']:.6f} | {row['BA']:.6f} | "
            f"{row['F1']:.6f} | {row['Precision']:.6f} | {row['Recall']:.6f} | {row['N']} |"
        )
    lines += ["", "## Interpretation boundary", "",
              "S2 is the true Host-only 88-D structural representation; it excludes Global24, Stage-I logit and every Ego operation. S3 adds only Stage-I logit.",
              "Bootstrap intervals are paired over the already fixed routed OOF cohort and do not authorize model, router, threshold or hyperparameter revision."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("NOEGO_NEEDGATE_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    main()
