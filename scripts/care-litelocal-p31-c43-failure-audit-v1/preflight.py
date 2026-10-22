#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
DIRECT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
REQUIRED = (
    P31 / "preflight/p31_reference/source_predictions.csv",
    P31 / "arms/P31_C43_RQ/source_predictions.csv",
    P31 / "results/hard_audit.json",
    DIRECT / "arms/P31_C43_DIRECT/source_predictions.csv",
    DIRECT / "results/hard_audit.json",
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    missing = [str(path) for path in REQUIRED
               if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty locked input:\n- " +
                           "\n- ".join(missing))
    tables = [sorted(read(path), key=lambda row: row["sha256"].casefold())
              for path in (REQUIRED[0], REQUIRED[1], REQUIRED[3])]
    ids = [[row["sha256"].casefold() for row in table] for table in tables]
    if any(len(table) != 799 for table in tables) or ids[0] != ids[1] or ids[0] != ids[2]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked cohorts differ")
    y = np.asarray([int(float(row["label"])) for row in tables[0]])
    base = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in tables[0]])
    route = np.asarray([int(float(row["P31_DI25__routed"])) == 1 for row in tables[0]])
    bounded = np.asarray([float(row["P31Mean__DI25__C43RelationQuery__Bounded__prob"])
                          for row in tables[1]])
    direct = np.asarray([float(row["P31Mean__DI25__C43RelationQuery__Direct__prob"])
                         for row in tables[2]])
    base_correct = (base >= .5) == y
    bounded_correct = (bounded >= .5) == y
    direct_correct = (direct >= .5) == y
    checks = {
        "source_rows": len(y), "route": int(route.sum()),
        "routed_errors": int(np.sum(route & ~base_correct)),
        "routed_correct": int(np.sum(route & base_correct)),
        "bounded_fixed": int(np.sum(~base_correct & bounded_correct)),
        "bounded_broken": int(np.sum(base_correct & ~bounded_correct)),
        "direct_fixed": int(np.sum(~base_correct & direct_correct)),
        "direct_broken": int(np.sum(base_correct & ~direct_correct)),
    }
    expected = {"source_rows": 799, "route": 200, "routed_errors": 46,
                "routed_correct": 154, "bounded_fixed": 8,
                "bounded_broken": 2, "direct_fixed": 13, "direct_broken": 16}
    if checks != expected:
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked result changed: " + repr(checks))
    print("P31_C43_FAILURE_AUDIT_PREFLIGHT_PASS")
    print("source_rows=799 route_P31_DI25=200 routed_errors=46 routed_correct=154")
    print("bounded_fixed=8 bounded_broken=2 direct_fixed=13 direct_broken=16")
    print("training_invoked=false inference_invoked=false future_rows_loaded=0")


if __name__ == "__main__":
    main()
