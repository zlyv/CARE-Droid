#!/usr/bin/env python3
"""Create a deterministic 50-APK source smoke cohort without using labels."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < args.count:
        raise RuntimeError("source lineage is smaller than smoke count")
    # SHA order makes the subset reproducible and independent of labels/folds.
    selected = sorted(rows, key=lambda row: row["sha256"].strip().lower())[:args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print("Fixed smoke lineage:", args.output, "n=", len(selected))


if __name__ == "__main__":
    main()
