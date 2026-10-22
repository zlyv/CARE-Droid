#!/usr/bin/env python3
"""Fail-closed input audit; reads strict 2017 outputs only."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ARMS = ("N0_NO_GRAPHGLOBAL", "D_DELTA_ONLY", "Q_QUERY_ONLY", "QD_QUERY_AND_DELTA")
ROOT = Path("artifacts/care-di25-graphglobal-path-source-v1")


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    required = [ROOT / "preflight/di25_reference/source_predictions.csv"]
    required += [ROOT / "arms" / arm / "source_predictions.csv" for arm in ARMS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing:\n- " + "\n- ".join(missing))

    reference = read(required[0])
    if len(reference) != 799 or len({r["sha256"].lower() for r in reference}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: bad canonical cohort")
    ref = {r["sha256"].lower(): r for r in reference}
    route_by_fold = Counter()
    for row in reference:
        if int(float(row["A25__routed"])):
            route_by_fold[int(float(row["fold"]))] += 1
    if dict(sorted(route_by_fold.items())) != {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: canonical route-by-fold changed")

    for arm, path in zip(ARMS, required[1:]):
        rows = read(path)
        if len(rows) != 799 or len({r["sha256"].lower() for r in rows}) != 799:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} cohort changed")
        for row in rows:
            sha = row["sha256"].lower()
            if sha not in ref:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} outside SHA {sha}")
            expected = ref[sha]
            checks = (
                int(float(row["label"])) == int(float(expected["label"])),
                int(float(row["fold"])) == int(float(expected["fold"])),
                int(float(row["A25__routed"])) == int(float(expected["A25__routed"])),
                abs(float(row["Global24__prob"]) - float(expected["Global24__prob"])) <= 1e-12,
            )
            if not all(checks):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: {arm} mismatch at {sha}")
    print("GRAPHGLOBAL_PATH_PAIRED_AUDIT_PREFLIGHT_PASS")
    print("source_rows=799")
    print("year=2017")
    print("route_DI=200")
    print("route_difference_count=0")
    print("future_rows_loaded=0")
    print("training_invoked=false")


if __name__ == "__main__":
    main()

