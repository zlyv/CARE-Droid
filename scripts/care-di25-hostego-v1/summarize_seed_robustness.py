#!/usr/bin/env python3
"""Summarize the five predeclared M0/M1 source-seed replicates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from freeze_final_method import ROOT, SEED_SETS, blocked, read_csv, sha256_file


OUT = ROOT / "seed_robustness"
METRICS = ("BA", "AUC", "F1", "fixed", "broken", "net", "correction_precision")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def validate(rep: str) -> None:
    marker = OUT / rep / "COMPLETE.json"
    if not marker.is_file():
        blocked(f"seed replicate {rep} has no COMPLETE.json")
    state = json.loads(marker.read_text(encoding="utf-8"))
    for name, digest in state.get("outputs", {}).items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != digest:
            blocked(f"seed replicate {rep} output checksum mismatch: {name}")


def main() -> None:
    all_metrics, all_mc = [], []
    for rep in SEED_SETS:
        validate(rep)
        rows = read_csv(OUT / rep / "metrics.csv")
        if len(rows) != 2 or {row["arm"] for row in rows} != {"M0", "M1"}:
            blocked(f"{rep} metrics do not contain exact M0/M1")
        all_metrics.extend(rows)
        all_mc.extend(read_csv(OUT / rep / "mcnemar.csv"))
    write_csv(OUT / "seed_robustness_summary.csv", all_metrics)
    write_csv(OUT / "seed_replicate_mcnemar.csv", all_mc)

    deltas = []
    for rep in SEED_SETS:
        index = {row["arm"]: row for row in all_metrics if row["replicate"] == rep}
        item = {"replicate": rep, "seed_triplet": index["M0"]["seed_triplet"]}
        for key in METRICS:
            item[f"delta_{key}_M1_minus_M0"] = float(index["M1"][key]) - float(index["M0"][key])
        mc = next(row for row in all_mc if row["replicate"] == rep)
        item.update({"n01_M0_wrong_M1_correct": mc["n01"], "n10_M0_correct_M1_wrong": mc["n10"], "exact_mcnemar_p": mc["exact_mcnemar_two_sided_p"]})
        item.update({
            "delta_AUC": item["delta_AUC_M1_minus_M0"], "delta_F1": item["delta_F1_M1_minus_M0"],
            "delta_fixed": item["delta_fixed_M1_minus_M0"], "delta_broken": item["delta_broken_M1_minus_M0"],
            "delta_net": item["delta_net_M1_minus_M0"], "n01": mc["n01"], "n10": mc["n10"],
            "mcnemar_p": mc["exact_mcnemar_two_sided_p"],
        })
        deltas.append(item)
    write_csv(OUT / "seed_robustness_delta.csv", deltas)

    summary = []
    for arm in ("M0", "M1"):
        rows = [row for row in all_metrics if row["arm"] == arm]
        item = {"arm": arm, "replicate_n": len(rows)}
        for key in METRICS:
            values = np.asarray([float(row[key]) for row in rows], float)
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std_sample"] = float(np.std(values, ddof=1))
            item[f"{key}_min"] = float(np.min(values)); item[f"{key}_max"] = float(np.max(values))
        summary.append(item)
    write_csv(OUT / "seed_robustness_aggregate.csv", summary)

    positive_ba = sum(float(row["delta_BA_M1_minus_M0"]) > 0 for row in deltas)
    positive_net = sum(float(row["delta_net_M1_minus_M0"]) > 0 for row in deltas)
    zero_or_positive_ba = sum(float(row["delta_BA_M1_minus_M0"]) >= 0 for row in deltas)
    wording = (
        "The Host+Ego gain is directionally consistent across all five predeclared seed triplets."
        if positive_ba == 5 and positive_net == 5 else
        "The Host+Ego result is mixed across the five predeclared seed triplets; it must not be described as uniformly robust."
    )
    lines = [
        "CARE-Droid source-seed robustness report", "",
        "Method definition, DI25 route, 25% budget and max_delta=2.25 were frozen before these replicates.",
        "All values are strict 2017 five-fold OOF. Sample standard deviation (ddof=1) is reported.", "",
    ]
    for row in summary:
        lines.append(
            f"{row['arm']}: BA={row['BA_mean']:.6f} +/- {row['BA_std_sample']:.6f}; "
            f"AUC={row['AUC_mean']:.6f} +/- {row['AUC_std_sample']:.6f}; "
            f"F1={row['F1_mean']:.6f} +/- {row['F1_std_sample']:.6f}; "
            f"net={row['net_mean']:.2f} +/- {row['net_std_sample']:.2f}"
        )
    delta_ba = np.asarray([float(row["delta_BA_M1_minus_M0"]) for row in deltas])
    delta_auc = np.asarray([float(row["delta_AUC_M1_minus_M0"]) for row in deltas])
    delta_f1 = np.asarray([float(row["delta_F1_M1_minus_M0"]) for row in deltas])
    delta_net = np.asarray([float(row["delta_net_M1_minus_M0"]) for row in deltas])
    broken_le = sum(float(row["delta_broken_M1_minus_M0"]) <= 0 for row in deltas)
    lines.extend(["", f"Mean delta BA={np.mean(delta_ba):+.6f} +/- {np.std(delta_ba,ddof=1):.6f}; mean delta AUC={np.mean(delta_auc):+.6f}; mean delta F1={np.mean(delta_f1):+.6f}; mean delta net={np.mean(delta_net):+.2f}.",
                  f"M1 BA improved in {positive_ba}/5 replicates ({zero_or_positive_ba}/5 non-negative).",
                  f"M1 net correction improved in {positive_net}/5 replicates.", wording,
                  f"M1 broken <= M0 in {broken_le}/5 replicates.",
                  "No statistical significance claim is made from direction counts alone."])
    (OUT / "SEED_ROBUSTNESS_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
