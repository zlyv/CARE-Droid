#!/usr/bin/env python3
"""Immutable source-only input and cohort contract."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_ROWS = 799
EXPECTED_YEAR = 2017
EXPECTED_FOLD_SIZES = {1: 160, 2: 160, 3: 160, 4: 160, 5: 159}
EXPECTED_ROUTE_TOTAL = 200
EXPECTED_ROUTE_BY_FOLD = {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}

ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
FEATURES = Path("artifacts/p1-features/features.csv")
SCHEMA = Path("artifacts/p1-features/feature_schema.json")
P72 = Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv")
HOST_NPZ = Path("artifacts/care-source-local-assets-v2/host_regions_2017.npz")
REGIONS_NPZ = Path("artifacts/care-source-local-assets-v2/regions_2017.npz")
ENDPOINT_CACHE = Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")
OUTPUT = Path("artifacts/care-di25-noego-needgate-source-v1")

ALLOWED_READ_PATHS = (ROUTER, FEATURES, SCHEMA, P72, HOST_NPZ, REGIONS_NPZ, ENDPOINT_CACHE)
ACTUALLY_CONSUMED_FOR_PREFLIGHT = (ROUTER, FEATURES, SCHEMA, P72, HOST_NPZ, ENDPOINT_CACHE)
HOST_KEYS = ("sha256", "host_values", "host_offsets", "real_host_count")
FORBIDDEN_BASENAMES = {
    "future_interz.npz", "future_regions_2018_2021.npz",
    "endpoint_centered_tensor_cache_2018_2021.jsonl.gz",
    "within_year_router_inputs.csv", "ego_1hop_regions_2017.npz",
}


def blocked(message: str) -> RuntimeError:
    return RuntimeError("BLOCKED_BY_CONTRACT: " + message)


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    if path not in ALLOWED_READ_PATHS:
        raise blocked(f"attempted non-whitelisted read: {path}")
    if not path.is_file():
        raise blocked(f"missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    try:
        path.relative_to(OUTPUT)
    except ValueError as error:
        raise blocked(f"output path escapes dedicated directory: {path}") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_router(rows: list[dict]) -> dict:
    required = {"sha256", "year", "label", "fold", "p_stage1", "route_DI"}
    if not rows or not required.issubset(rows[0]):
        raise blocked("source router missing columns: " + ",".join(sorted(required - set(rows[0] if rows else {}))))
    if len(rows) != EXPECTED_ROWS:
        raise blocked(f"source router rows={len(rows)} expected=799")
    ids = [norm(row["sha256"]) for row in rows]
    if "" in ids or len(set(ids)) != EXPECTED_ROWS:
        raise blocked(f"source router unique_sha={len(set(ids))} expected=799")
    years = np.asarray([int(float(row["year"])) for row in rows], dtype=np.int64)
    labels = np.asarray([int(float(row["label"])) for row in rows], dtype=np.int64)
    folds = np.asarray([int(float(row["fold"])) for row in rows], dtype=np.int64)
    probabilities = np.asarray([float(row["p_stage1"]) for row in rows], dtype=np.float64)
    route = np.asarray([int(float(row["route_DI"])) for row in rows], dtype=np.int64)
    if set(years.tolist()) != {EXPECTED_YEAR}:
        raise blocked(f"source router year set={sorted(set(years.tolist()))} expected={{2017}}")
    if not set(labels.tolist()).issubset({0, 1}) or not set(route.tolist()).issubset({0, 1}):
        raise blocked("label and route_DI must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise blocked("p_stage1 must contain finite probabilities in [0,1]")
    fold_sizes = {fold: int(np.sum(folds == fold)) for fold in sorted(set(folds.tolist()))}
    route_by_fold = {fold: int(np.sum(route[folds == fold])) for fold in sorted(set(folds.tolist()))}
    if fold_sizes != EXPECTED_FOLD_SIZES:
        raise blocked(f"fold sizes={fold_sizes} expected={EXPECTED_FOLD_SIZES}")
    if int(np.sum(route)) != EXPECTED_ROUTE_TOTAL or route_by_fold != EXPECTED_ROUTE_BY_FOLD:
        raise blocked(f"route_DI total/by-fold mismatch: total={int(np.sum(route))}, by_fold={route_by_fold}")
    return {
        "rows": EXPECTED_ROWS, "unique_sha": EXPECTED_ROWS, "year": [EXPECTED_YEAR],
        "fold_sizes": fold_sizes, "route_DI": EXPECTED_ROUTE_TOTAL,
        "route_DI_per_fold": route_by_fold,
    }
