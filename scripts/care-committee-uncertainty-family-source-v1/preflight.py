#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


DEFAULT = Path("artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv")
FALLBACK = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
REQUIRED = {"sha256", "label", "fold", "p_stage1", "p_lr", "p_rf", "p_xgb", "p_svm"}


def resolve() -> Path:
    for path in (DEFAULT, FALLBACK):
        if path.is_file() and path.stat().st_size > 100:
            return path
    raise RuntimeError("BLOCKED_BY_CONTRACT: no frozen strict-2017 committee score file")


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(rows: list[dict]) -> dict:
    if not rows or not REQUIRED.issubset(rows[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing required frozen columns: " +
                           repr(sorted(REQUIRED - (set(rows[0]) if rows else set()))))
    ids = [r["sha256"].strip().lower() for r in rows]
    folds = Counter(int(r["fold"]) for r in rows)
    years = {int(r.get("year", 2017)) for r in rows}
    if len(rows) != 799 or len(set(ids)) != 799 or years != {2017}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not exact unique 2017/799")
    expected = {1:160, 2:160, 3:160, 4:160, 5:159}
    if dict(folds) != expected:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold sizes changed: " + repr(dict(folds)))
    for r in rows:
        if int(float(r["label"])) not in (0, 1):
            raise RuntimeError("BLOCKED_BY_CONTRACT: non-binary label")
        for key in ("p_stage1", "p_lr", "p_rf", "p_xgb", "p_svm"):
            value = float(r[key])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError("BLOCKED_BY_CONTRACT: invalid probability " + key)
    return {"folds": expected, "years": years}


def main() -> None:
    path = resolve()
    rows = load(path)
    validate(rows)
    print("COMMITTEE_UNCERTAINTY_FAMILY_PREFLIGHT_PASS")
    print("source_file=" + str(path.resolve()))
    print("source_rows=799")
    print("unique_sha=799")
    print("year=2017")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("future_rows_loaded=0")
    print("stage2_training_invoked=false")


if __name__ == "__main__":
    main()

