#!/usr/bin/env python3
"""Strict 2017-only loader; it has no future or Ego fallback path."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from source_contract import (
    ENDPOINT_CACHE, FEATURES, HOST_KEYS, HOST_NPZ, P72, ROUTER, SCHEMA,
    blocked, norm, read_csv, validate_router,
)


def _id_key(rows: list[dict], name: str) -> str:
    fields = set(rows[0] if rows else {})
    for candidate in ("sha256", "sample_id"):
        if candidate in fields:
            return candidate
    raise blocked(f"{name} has no sha256/sample_id column")


def _exact_map(rows: list[dict], name: str) -> tuple[list[str], dict[str, dict]]:
    key = _id_key(rows, name)
    ids = [norm(row[key]) for row in rows]
    if "" in ids or len(set(ids)) != len(ids):
        raise blocked(f"{name} contains empty or duplicate SHA values")
    return ids, dict(zip(ids, rows))


def load_host_only(path: Path, router_ids: list[str]):
    if path != HOST_NPZ:
        raise blocked(f"Host loader accepts only {HOST_NPZ}; received {path}")
    if not path.is_file():
        raise blocked(f"missing file: {path}")
    with np.load(path, allow_pickle=False) as saved:
        missing = set(HOST_KEYS) - set(saved.files)
        if missing:
            raise blocked("Host NPZ missing keys: " + ",".join(sorted(missing)))
        ids = [norm(value) for value in saved["sha256"].astype(str)]
        values = np.asarray(saved["host_values"], dtype=np.float32)
        offsets = np.asarray(saved["host_offsets"], dtype=np.int64)
        real_count = np.asarray(saved["real_host_count"], dtype=np.int64)
        available_keys = sorted(saved.files)
    if ids != router_ids:
        raise blocked("host_regions_2017 SHA order differs from source router")
    if values.ndim != 2 or values.shape[1] != 74 or not np.isfinite(values).all():
        raise blocked(f"Host values must be finite (*,74), received {values.shape}")
    if offsets.shape != (800,) or offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 1):
        raise blocked("Host offsets must encode 799 non-empty deterministic model bags")
    if real_count.shape != (799,) or np.any(real_count < 0):
        raise blocked("real_host_count contract failed")
    bags = [values[offsets[i]:offsets[i + 1]] for i in range(799)]
    return bags, real_count, available_keys


def load_source_contract() -> SimpleNamespace:
    router_rows = read_csv(ROUTER)
    router_audit = validate_router(router_rows)
    router_ids, router_by_id = _exact_map(router_rows, "source router")

    feature_rows = read_csv(FEATURES)
    feature_ids, feature_by_id = _exact_map(feature_rows, "features.csv")
    if set(feature_ids) != set(router_ids):
        missing = sorted(set(router_ids) - set(feature_ids))
        outside = sorted(set(feature_ids) - set(router_ids))
        raise blocked(f"features/router cohort mismatch missing={len(missing)} outside={len(outside)}")
    if not SCHEMA.is_file():
        raise blocked(f"missing file: {SCHEMA}")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    p72_rows = read_csv(P72)
    p72_ids, p72_by_id = _exact_map(p72_rows, "p72 OOF")
    if set(p72_ids) != set(router_ids):
        raise blocked("p72/source-router SHA cohort mismatch")
    for sha in router_ids:
        row, reference = router_by_id[sha], p72_by_id[sha]
        if int(float(reference["label"])) != int(float(row["label"])) or int(float(reference["fold"])) != int(float(row["fold"])):
            raise blocked(f"p72 label/fold mismatch for {sha}")

    bags, real_host_count, host_available_keys = load_host_only(HOST_NPZ, router_ids)

    if not ENDPOINT_CACHE.is_file():
        raise blocked(f"missing file: {ENDPOINT_CACHE}")
    endpoint_ids = []
    endpoint_years = set()
    with gzip.open(ENDPOINT_CACHE, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sha = norm(payload.get("sha256"))
            if not sha:
                raise blocked(f"endpoint cache row {line_number} lacks sha256")
            endpoint_ids.append(sha)
            for key in ("year", "metadata_year"):
                if str(payload.get(key, "")).strip():
                    endpoint_years.add(int(float(payload[key])))
    if endpoint_ids != router_ids or len(set(endpoint_ids)) != 799:
        raise blocked("2017 endpoint cache order/cohort differs from source router")
    if endpoint_years and endpoint_years != {2017}:
        raise blocked(f"endpoint cache contains non-2017 years: {sorted(endpoint_years)}")

    ordered_features = [feature_by_id[sha] for sha in router_ids]
    labels = np.asarray([int(float(router_by_id[sha]["label"])) for sha in router_ids], dtype=np.int64)
    folds = np.asarray([int(float(router_by_id[sha]["fold"])) for sha in router_ids], dtype=np.int64)
    p_stage1 = np.asarray([float(router_by_id[sha]["p_stage1"]) for sha in router_ids], dtype=np.float64)
    route = np.asarray([int(float(router_by_id[sha]["route_DI"])) == 1 for sha in router_ids], dtype=bool)
    return SimpleNamespace(
        ids=router_ids, router_rows=router_rows, feature_rows=ordered_features,
        feature_schema=schema, labels=labels, folds=folds, p_stage1=p_stage1,
        route=route, host_bags=bags, real_host_count=real_host_count,
        host_available_keys=host_available_keys, host_consumed_keys=list(HOST_KEYS),
        router_audit=router_audit, future_rows_loaded=0, ego_consumed=False,
    )
