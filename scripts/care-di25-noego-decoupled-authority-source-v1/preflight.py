#!/usr/bin/env python3
"""Hard-gate the two strict-2017 inputs used by this audit."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
GATES = Path("artifacts/care-di25-noego-needgate-source-v1/oof_predictions.csv")
EXPECTED_FOLDS = {1: 160, 2: 160, 3: 160, 4: 160, 5: 159}
EXPECTED_ROUTES = {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
ARMS = ("S0", "S1", "S2", "S3")


def blocked(message):
    raise RuntimeError("BLOCKED_BY_CONTRACT: " + message)


def rows(path):
    if not path.is_file():
        blocked(f"missing required strict-source input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value):
    return str(value).strip().lower()


def main():
    router = rows(ROUTER); gates = rows(GATES)
    required_router = {"sha256", "year", "label", "fold", "p_stage1", "route_DI"}
    required_gates = {"sha256", "year", "fold", "route_DI", *(f"score_{a}" for a in ARMS)}
    if len(router) != 799 or not required_router.issubset(router[0]):
        blocked("router must be the canonical 799-row source contract")
    if len(gates) != 200 or not required_gates.issubset(gates[0]):
        blocked("Need-Gate output must contain 200 routed rows and S0-S3 scores")
    router_ids = [norm(r["sha256"]) for r in router]
    gate_ids = [norm(r["sha256"]) for r in gates]
    if len(set(router_ids)) != 799 or len(set(gate_ids)) != 200:
        blocked("duplicate or empty SHA values")
    if {int(float(r["year"])) for r in router + gates} != {2017}:
        blocked("a non-2017 row was loaded")
    fold_sizes = Counter(int(float(r["fold"])) for r in router)
    route_by_fold = Counter(int(float(r["fold"])) for r in router if int(float(r["route_DI"])) == 1)
    routed_ids = {norm(r["sha256"]) for r in router if int(float(r["route_DI"])) == 1}
    if dict(fold_sizes) != EXPECTED_FOLDS or dict(route_by_fold) != EXPECTED_ROUTES:
        blocked(f"fold/route contract changed: {dict(fold_sizes)} {dict(route_by_fold)}")
    if set(gate_ids) != routed_ids:
        blocked("Need-Gate SHA set differs from canonical DI25 route")
    for row in gates:
        for arm in ARMS:
            value = float(row[f"score_{arm}"])
            if not 0.0 <= value <= 1.0:
                blocked(f"invalid {arm} OOF gate score")
    print("DECOUPLED_AUTHORITY_PREFLIGHT_PASS")
    print("source_rows = 799")
    print("year = 2017")
    print("route_DI = 200")
    print("needgate_oof_rows = 200")
    print("future_rows_loaded = 0")
    print("ego_consumed = false")
    print("threshold_search = false")
    print("training_invoked = false")


if __name__ == "__main__":
    main()

