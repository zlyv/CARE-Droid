#!/usr/bin/env python3
"""Read-only preflight for CARE-Droid no-Ego within-year replay."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT = Path('/home/boyuanjie/projects/tasa-droid')
ROUTER = PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv'
RESOLVED = PROJECT / 'artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json'
FUTURE_REGIONS = PROJECT / 'artifacts/care-future-local-assets-v2/regions_2018_2021.npz'
SOURCE_CACHE = PROJECT / 'artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz'
FUTURE_CACHE = PROJECT / 'artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz'
RUNNER = PROJECT / 'scripts/care-route-stage2-within-year-matrix-v1/run_within_year_arm.py'
FUTURE_ARM = PROJECT / 'scripts/care-route-stage2-within-year-matrix-v1/run_future_arm.py'
OUT = PROJECT / 'artifacts/care-di25-noego-within-year-replay-v1'

EXPECTED_YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTE_COUNTS = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
EXPECTED_FOLDS = {1, 2, 3, 4, 5}


def blocked(msg: str) -> RuntimeError:
    return RuntimeError('BLOCKED_BY_CONTRACT: ' + msg)


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def norm(x) -> str:
    return str(x or '').strip().lower()


def main() -> None:
    required = [ROUTER, RESOLVED, FUTURE_REGIONS, SOURCE_CACHE, FUTURE_CACHE, RUNNER, FUTURE_ARM]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise blocked('missing required files: ' + ', '.join(missing))

    rows = read_csv(ROUTER)
    if len(rows) != 1989:
        raise blocked(f'router rows must be 1989, got {len(rows)}')
    cols = set(rows[0]) if rows else set()
    need = {'sha256','year','label','fold','p_stage1','route_DI'}
    if not need.issubset(cols):
        raise blocked('router missing columns: ' + ', '.join(sorted(need-cols)))

    ids = [norm(r['sha256']) for r in rows]
    if any(not s for s in ids) or len(set(ids)) != 1989:
        raise blocked('router SHA values are empty or duplicated')

    years = np.asarray([int(r['year']) for r in rows], dtype=np.int64)
    labels = np.asarray([int(r['label']) for r in rows], dtype=np.int64)
    folds = np.asarray([int(r['fold']) for r in rows], dtype=np.int64)
    route = np.asarray([int(float(r['route_DI'])) for r in rows], dtype=np.int64)
    p = np.asarray([float(r['p_stage1']) for r in rows], dtype=np.float64)

    if set(np.unique(years).tolist()) != set(EXPECTED_YEAR_COUNTS):
        raise blocked('years must be exactly 2018-2021')
    if Counter(years.tolist()) != Counter(EXPECTED_YEAR_COUNTS):
        raise blocked('year counts changed: ' + repr(dict(Counter(years.tolist()))))
    if not set(np.unique(labels).tolist()).issubset({0,1}):
        raise blocked('labels must be binary')
    if not set(np.unique(route).tolist()).issubset({0,1}):
        raise blocked('route_DI must be binary')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise blocked('p_stage1 must be finite probabilities')

    route_counts = {y: int(np.sum(route[years == y])) for y in EXPECTED_YEAR_COUNTS}
    if route_counts != EXPECTED_ROUTE_COUNTS:
        raise blocked('DI25 route counts changed: ' + repr(route_counts))
    for y in EXPECTED_YEAR_COUNTS:
        if set(np.unique(folds[years == y]).tolist()) != EXPECTED_FOLDS:
            raise blocked(f'year {y} folds are not exactly 1..5')

    resolved = json.loads(RESOLVED.read_text(encoding='utf-8'))
    for key in ('features','schema','source_manifest'):
        if key not in resolved or not Path(resolved[key]).is_file():
            raise blocked(f'resolved input missing/unavailable: {key}')

    with np.load(FUTURE_REGIONS, allow_pickle=False) as z:
        required_keys = {'sha256','host_values','host_offsets','real_host_count'}
        if not required_keys.issubset(set(z.files)):
            raise blocked('future region archive missing Host keys')
        region_ids = [norm(v) for v in z['sha256'].astype(str)]
        values = np.asarray(z['host_values'])
        offsets = np.asarray(z['host_offsets'])
        real = np.asarray(z['real_host_count'])
        if len(region_ids) != 1989 or len(set(region_ids)) != 1989 or set(region_ids) != set(ids):
            raise blocked('future region SHA cohort differs from router cohort')
        if values.ndim != 2 or values.shape[1] != 74:
            raise blocked(f'Host width must be 74, got {values.shape}')
        if offsets.shape != (1990,) or int(offsets[0]) != 0 or int(offsets[-1]) != len(values):
            raise blocked('Host offsets contract failed')
        if real.shape != (1989,):
            raise blocked('real_host_count shape changed')
        available_keys = sorted(z.files)

    # Static no-Ego contract: recovered historical loader reads only Host keys.
    future_text = FUTURE_ARM.read_text(encoding='utf-8')
    start = future_text.find('def load_bags(')
    end = future_text.find('\ndef ', start + 1)
    block = future_text[start:end if end >= 0 else None]
    for token in ('host_values','host_offsets','real_host_count'):
        if token not in block:
            raise blocked('historical load_bags no longer selects ' + token)
    if 'ego_' in block.casefold() or 'ego_values' in block.casefold():
        raise blocked('historical load_bags unexpectedly references Ego')

    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        'passed': True,
        'rows': 1989,
        'unique_sha256': 1989,
        'years': EXPECTED_YEAR_COUNTS,
        'route_DI_counts': route_counts,
        'folds_per_year': {str(y): sorted(np.unique(folds[years == y]).astype(int).tolist()) for y in EXPECTED_YEAR_COUNTS},
        'future_region_available_keys': available_keys,
        'future_region_consumed_keys': ['host_values','host_offsets','real_host_count'],
        'ego_consumed': False,
        'future_inter_policy': 'rebuild into replay-local cache; do not fabricate/replace lost historical cache',
        'old_artifacts_modified': False,
    }
    (OUT / 'preflight_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print('NOEGO_WITHIN_YEAR_PREFLIGHT_PASS')
    print('rows = 1989')
    print('route_DI =', int(route.sum()))
    print('route_by_year =', route_counts)
    print('ego_consumed = false')


if __name__ == '__main__':
    main()
