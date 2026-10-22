#!/usr/bin/env python3
"""Read-only hard preflight for the no-Ego need-gate experiment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from probe_common import EXPECTED_YEAR_COUNTS, FOLDS, YEARS, blocked, noego_contract, norm, read_csv, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router-inputs", type=Path, default=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"))
    p.add_argument("--source-cache", type=Path, default=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
    p.add_argument("--future-cache", type=Path, default=Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
    p.add_argument("--future-regions", type=Path, default=Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"))
    p.add_argument("--future-inter-cache", type=Path, default=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"))
    p.add_argument("--resolved", type=Path, default=Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/care-di25-noego-needgate-probe-v1"))
    return p


def validate(args) -> dict:
    history = Path("scripts/care-route-stage2-within-year-matrix-v1")
    required_modules = [
        (history / "run_within_year_arm.py",),
        (history / "run_future_arm.py", history / "vendor/run_future_arm.py"),
        (history / "run_care_rolegraph_conditioned_dualstream_v4.py", history / "vendor/run_care_rolegraph_conditioned_dualstream_v4.py"),
        (history / "run_care_rgcondattn_dedup_bounded_source_v1.py", history / "vendor/run_care_rgcondattn_dedup_bounded_source_v1.py"),
    ]
    missing_modules = [" OR ".join(str(path) for path in alternatives) for alternatives in required_modules if not any(path.is_file() for path in alternatives)]
    if missing_modules:
        raise blocked("missing historical implementation files: " + ", ".join(missing_modules))
    rows = read_csv(args.router_inputs)
    expected_columns = {"sha256", "year", "label", "fold", "p_stage1", "route_DI"}
    missing_columns = expected_columns - set(rows[0] if rows else {})
    if missing_columns:
        raise blocked("router input columns missing: " + ", ".join(sorted(missing_columns)))
    if len(rows) != 1989:
        raise blocked(f"router input must contain 1989 rows, received {len(rows)}")
    ids = [norm(row["sha256"]) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != 1989:
        raise blocked("router SHA values are empty or duplicated")
    years = np.asarray([int(row["year"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    route = np.asarray([int(row["route_DI"]) for row in rows], dtype=np.int64)
    probability = np.asarray([float(row["p_stage1"]) for row in rows], dtype=np.float64)
    if set(np.unique(years).tolist()) != set(YEARS):
        raise blocked("future years differ from 2018-2021")
    counts = Counter(years.tolist())
    if dict(counts) != EXPECTED_YEAR_COUNTS:
        raise blocked(f"year counts differ: {dict(counts)}")
    if not set(np.unique(labels).tolist()).issubset({0, 1}) or not set(np.unique(route).tolist()).issubset({0, 1}):
        raise blocked("label/route_DI must be binary")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise blocked("p_stage1 is not a finite probability")
    for year in YEARS:
        if set(np.unique(folds[years == year]).tolist()) != set(FOLDS):
            raise blocked(f"year {year} does not contain fixed folds 1-5")
    if not args.resolved.is_file():
        raise blocked(f"missing resolved input contract: {args.resolved}")
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    for key in ("features", "schema", "source_manifest"):
        if key not in resolved or not Path(resolved[key]).is_file():
            raise blocked(f"resolved input missing/unavailable: {key}")
    if not args.future_regions.is_file():
        raise blocked(f"missing Host region asset: {args.future_regions}")
    for name in ("source_cache", "future_cache", "future_inter_cache"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise blocked(f"missing frozen upstream asset {name}: {path}")
    with np.load(args.future_regions, allow_pickle=False) as archive:
        required_keys = {"sha256", "host_values", "host_offsets", "real_host_count"}
        if not required_keys.issubset(set(archive.files)):
            raise blocked("Host region asset lacks keys: " + ", ".join(sorted(required_keys - set(archive.files))))
        host_values = np.asarray(archive["host_values"])
        offsets = np.asarray(archive["host_offsets"])
        region_ids = [norm(value) for value in archive["sha256"].astype(str)]
        if len(region_ids) != 1989 or len(set(region_ids)) != 1989 or set(region_ids) != set(ids):
            raise blocked("Host region SHA cohort does not exactly match router cohort")
        if host_values.ndim != 2 or host_values.shape[1] != 74:
            raise blocked(f"Host feature width must be 74, received {host_values.shape}")
        if offsets.shape != (1990,) or int(offsets[0]) != 0 or int(offsets[-1]) != len(host_values) or np.any(np.diff(offsets) < 0):
            raise blocked("Host offsets do not encode exactly 1989 ordered APK bags")
        available_keys = sorted(archive.files)
    route_counts = {str(year): int(np.sum(route[years == year])) for year in YEARS}
    if any(value <= 0 for value in route_counts.values()):
        raise blocked("at least one year has no DI-routed APK")
    audit = {
        "passed": True, "total_n": 1989, "unique_sha256": 1989,
        "year_counts": {str(k): int(v) for k, v in counts.items()},
        "DI_routed_counts": route_counts,
        "folds_by_year": {str(year): sorted(np.unique(folds[years == year]).astype(int).tolist()) for year in YEARS},
        "router_inputs": str(args.router_inputs), "future_regions": str(args.future_regions),
        "source_cache": str(args.source_cache), "future_cache": str(args.future_cache),
        "future_inter_cache": str(args.future_inter_cache),
        "future_region_available_keys": available_keys,
        "future_region_consumed_keys": ["host_values", "host_offsets", "real_host_count"],
        "no_rows_dropped": True, "upstream_artifacts_modified": False,
    }
    contract = noego_contract()
    contract["input_assets"] = [
        str(args.router_inputs), str(args.resolved), str(args.source_cache),
        str(args.future_cache), str(args.future_regions), str(args.future_inter_cache),
    ]
    contract["actual_tensor_consumption_audit"] = {
        "region_archive_keys_available": available_keys,
        "region_archive_keys_selected": ["host_values", "host_offsets", "real_host_count"],
        "ego_keys_selected": [],
    }
    write_json(args.output / "cohort_audit.json", audit)
    write_json(args.output / "noego_contract.json", contract)
    return audit


def main() -> None:
    args = parser().parse_args()
    validate(args)
    print("NOEGO_NEEDGATE_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
