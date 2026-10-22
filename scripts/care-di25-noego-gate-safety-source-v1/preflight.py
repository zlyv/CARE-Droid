#!/usr/bin/env python3
"""Validate the strict-source router and S3 OOF score contract."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
GATES = Path("artifacts/care-di25-noego-needgate-source-v1/oof_predictions.csv")
EXPECTED_FOLDS = {1: 160, 2: 160, 3: 160, 4: 160, 5: 159}
EXPECTED_ROUTES = {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}


def blocked(message): raise RuntimeError("BLOCKED_BY_CONTRACT: " + message)


def read(path):
    if not path.is_file(): blocked(f"missing strict-source input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as h: return list(csv.DictReader(h))


def norm(x): return str(x).strip().lower()


def load_and_validate():
    router, gates = read(ROUTER), read(GATES)
    rr = {"sha256", "year", "label", "fold", "p_stage1", "route_DI"}
    gr = {"sha256", "year", "fold", "route_DI", "score_S3"}
    if len(router) != 799 or not rr.issubset(router[0]): blocked("invalid 799-row router")
    if len(gates) != 200 or not gr.issubset(gates[0]): blocked("invalid 200-row S3 OOF scores")
    if {int(float(r["year"])) for r in router + gates} != {2017}: blocked("non-2017 row loaded")
    ids = [norm(r["sha256"]) for r in router]; gids = [norm(r["sha256"]) for r in gates]
    if len(set(ids)) != 799 or len(set(gids)) != 200: blocked("SHA uniqueness failed")
    fs = dict(Counter(int(float(r["fold"])) for r in router))
    rb = dict(Counter(int(float(r["fold"])) for r in router if int(float(r["route_DI"])) == 1))
    rid = {norm(r["sha256"]) for r in router if int(float(r["route_DI"])) == 1}
    if fs != EXPECTED_FOLDS or rb != EXPECTED_ROUTES or set(gids) != rid:
        blocked("fold or DI25 route identity failed")
    if any(not 0 <= float(r["score_S3"]) <= 1 for r in gates): blocked("invalid S3 score")
    return router, gates


def main():
    load_and_validate()
    print("GATE_SAFETY_PREFLIGHT_PASS")
    print("source_rows = 799")
    print("year = 2017")
    print("route_DI = 200")
    print("future_rows_loaded = 0")
    print("ego_consumed = false")
    print("model_training_invoked = false")
    print("selection = leave-one-outer-fold-out development diagnostic")


if __name__ == "__main__": main()

