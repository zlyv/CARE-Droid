#!/usr/bin/env python3
"""Compare same-year OOF with the already frozen 2017-to-future diagnostic."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-year", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    same = {(row["method"], int(row["year"])): row for row in read_csv(args.same_year)}
    temporal = {(row["method"], int(row["year"])): row for row in read_csv(args.temporal)}
    if set(same) != set(temporal):
        missing_left = sorted(set(temporal) - set(same))
        missing_right = sorted(set(same) - set(temporal))
        raise RuntimeError(f"method/year mismatch: same_missing={missing_left} temporal_missing={missing_right}")
    rows = []
    for method, year in sorted(same, key=lambda item: (item[1], item[0])):
        left, right = same[(method, year)], temporal[(method, year)]
        rows.append({
            "method": method, "year": year,
            "same_year_BA": float(left["BA"]), "temporal_BA": float(right["BA"]),
            "BA_same_minus_temporal": float(left["BA"]) - float(right["BA"]),
            "same_year_AUC": float(left["AUC"]), "temporal_AUC": float(right["AUC"]),
            "AUC_same_minus_temporal": float(left["AUC"]) - float(right["AUC"]),
            "same_year_F1": float(left["F1"]), "temporal_F1": float(right["F1"]),
            "F1_same_minus_temporal": float(left["F1"]) - float(right["F1"]),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "same_year_vs_temporal_gap.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "CARE-Droid same-year versus temporal gap", "",
        "Positive values mean performance recovers when training sees the evaluated year's distribution.",
        "This comparison diagnoses distribution shift; it is not an additional model-selection result.", "",
    ]
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        lines.append(
            f"{method}: mean_dBA={np.mean([row['BA_same_minus_temporal'] for row in subset]):+.6f} "
            f"mean_dAUC={np.mean([row['AUC_same_minus_temporal'] for row in subset]):+.6f} "
            f"mean_dF1={np.mean([row['F1_same_minus_temporal'] for row in subset]):+.6f}")
    (args.output / "TEMPORAL_GAP_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WITHIN_YEAR_TEMPORAL_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
