#!/usr/bin/env python3
"""Evaluate old and new OOF experts under the same frozen A25 and DI routes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SELECTION_DIR = SCRIPT_DIR.parent / "care-di-budget-bound-selection-v1"
sys.path.insert(0, str(SELECTION_DIR))
from selection_core import array, evaluate, find_column, norm_sha, read_csv, write_csv  # noqa: E402


OLD_PREFERRED = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__expert_prob"
NEW_PREFERRED = "Global24__A25__CARE_MalBenContrast_FixKeep__expert_prob"


def align(rows: list[dict], sha: list[str], context: str) -> list[dict]:
    by_sha = {norm_sha(row["sha256"]): row for row in rows}
    if len(by_sha) != len(rows) or set(by_sha) != set(sha):
        raise RuntimeError(context + " SHA set does not match the fixed source cohort")
    return [by_sha[item] for item in sha]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-scores", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv"
    ))
    parser.add_argument("--old-predictions", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv"
    ))
    parser.add_argument("--new-predictions", type=Path, default=Path(
        "artifacts/care-malben-contrast-v1/source/source_predictions.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-malben-contrast-v1/comparison"
    ))
    args = parser.parse_args()

    router = read_csv(args.router_scores)
    if len(router) != 799:
        raise RuntimeError("expected the fixed 799-row 2017 source cohort")
    sha = [norm_sha(row["sha256"]) for row in router]
    labels = array(router, "label", int).astype(np.int64)
    folds = array(router, "fold", int).astype(np.int64)
    base = array(router, "p_stage1")
    routes = {
        "A25": array(router, "route_A25", int).astype(bool),
        "DI25": array(router, "route_DI", int).astype(bool),
    }
    old_rows = align(read_csv(args.old_predictions), sha, "old expert")
    new_rows = align(read_csv(args.new_predictions), sha, "new expert")
    for name, rows in (("old", old_rows), ("new", new_rows)):
        if not np.array_equal(array(rows, "label", int), labels):
            raise RuntimeError(name + " expert labels changed")
        if not np.array_equal(array(rows, "fold", int), folds):
            raise RuntimeError(name + " expert folds changed")
        if not np.allclose(array(rows, "Global24__prob"), base, atol=2e-6, rtol=0.0):
            raise RuntimeError(name + " expert Global24 probabilities changed")
    old_col = find_column(old_rows, OLD_PREFERRED, ("Dedup", "expert_prob"))
    new_col = find_column(new_rows, NEW_PREFERRED, ("MalBenContrast", "expert_prob"))
    experts = {"AnomalyBaseline": array(old_rows, old_col), "MalBenContrast": array(new_rows, new_col)}

    metric_rows = []
    fold_rows = []
    delta_rows = []
    for router_name, route in routes.items():
        by_expert = {}
        for expert_name, expert in experts.items():
            row = evaluate(labels, base, expert, route)
            row.update({"router": router_name, "expert": expert_name})
            metric_rows.append(row)
            by_expert[expert_name] = row
            for fold in sorted(np.unique(folds)):
                selected = folds == fold
                fold_row = evaluate(
                    labels[selected], base[selected], expert[selected], route[selected]
                )
                fold_row.update({
                    "fold": int(fold), "router": router_name, "expert": expert_name
                })
                fold_rows.append(fold_row)
        old = by_expert["AnomalyBaseline"]
        new = by_expert["MalBenContrast"]
        delta_rows.append({
            "router": router_name,
            "delta_BA_new_minus_old": float(new["BA"] - old["BA"]),
            "delta_AUC_new_minus_old": float(new["AUC"] - old["AUC"]),
            "delta_F1_new_minus_old": float(new["F1"] - old["F1"]),
            "delta_fixed": int(new["fixed"] - old["fixed"]),
            "delta_broken": int(new["broken"] - old["broken"]),
            "delta_net": int(new["net"] - old["net"]),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "source_2x2_metrics.csv", metric_rows)
    write_csv(args.output / "source_2x2_per_fold.csv", fold_rows)
    write_csv(args.output / "source_old_to_new_delta.csv", delta_rows)

    sample_rows = []
    for index, sha256 in enumerate(sha):
        sample_rows.append({
            "sha256": sha256,
            "label": int(labels[index]),
            "fold": int(folds[index]),
            "p_stage1": float(base[index]),
            "route_A25": int(routes["A25"][index]),
            "route_DI25": int(routes["DI25"][index]),
            "p_anomaly_baseline_expert": float(experts["AnomalyBaseline"][index]),
            "p_malben_contrast_expert": float(experts["MalBenContrast"][index]),
        })
    write_csv(args.output / "source_aligned_predictions.csv", sample_rows)

    lines = [
        "CARE-Droid malicious-vs-benign local evidence source comparison v1",
        "",
        "Strict 2017 OOF only. Same frozen source cohort and same routes.",
        "Only the Stage-II expert differs within each router row.",
        "",
    ]
    for row in metric_rows:
        lines.append(
            f"{row['router']}__{row['expert']}: "
            f"BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
            f"fixed={row['fixed']} broken={row['broken']} net={row['net']} "
            f"RER@25={row['repairable_error_recall']:.6f}"
        )
    lines.append("")
    lines.append("Old-to-new deltas:")
    for row in delta_rows:
        lines.append(
            f"{row['router']}: dBA={row['delta_BA_new_minus_old']:+.6f} "
            f"dAUC={row['delta_AUC_new_minus_old']:+.6f} "
            f"dF1={row['delta_F1_new_minus_old']:+.6f} "
            f"dFixed={row['delta_fixed']:+d} dBroken={row['delta_broken']:+d} "
            f"dNet={row['delta_net']:+d}"
        )
    lines.extend([
        "",
        "Source-side decision rule:",
        "Advance MalBenContrast only if it improves net/BA without a material increase",
        "in broken samples under at least one predeclared router, and inspect all five",
        "outer folds before freezing a future diagnostic.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output / "REPORT.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
