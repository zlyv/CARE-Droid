#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from common import (FUTURE_INTER, OUT, PROJECT, SCRIPTS, discover_c43_future_region_cache,
                    discover_future_runner, file_sha256, read_csv, save_json)


def require(path: Path, label: str):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: missing/empty {label}: {path}')


def main():
    ref_dir = OUT / 'preflight/p31_reference'
    require(ref_dir / 'source_predictions.csv', 'P31 source reference')
    require(ref_dir / 'future_predictions_diagnostic.csv', 'P31 future reference')
    require(OUT / 'preflight/compact_features_2017_2021_eval_only.csv', 'evaluation feature table')
    require(FUTURE_INTER, 'future InterZ cache')
    compact_dir = SCRIPTS / 'care-di25-compact43-relationquery-source-v1'
    require(compact_dir / 'compact_models.py', 'C43 compact model definition')
    require(compact_dir / 'compact_background.py', 'C43 background definition')
    require(SCRIPTS / 'care-di25-graphglobal-path-source-v1/source_label_contract.py', 'source label contract')
    require(SCRIPTS / 'care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py', 'v4 source runner')

    runner = discover_future_runner()
    cache = discover_c43_future_region_cache(os.environ.get('C43_FUTURE_REGION_CACHE'))
    text = runner.read_text(encoding='utf-8', errors='replace')
    must_flags = ('--features', '--previous-result', '--future-inter-cache', '--region-cache', '--max-delta', '--output')
    missing_flags = [x for x in must_flags if x not in text]
    if missing_flags:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: generic future-runner CLI changed; missing flags={missing_flags} in {runner}')

    source_rows = read_csv(ref_dir / 'source_predictions.csv')
    future_rows = read_csv(ref_dir / 'future_predictions_diagnostic.csv')
    if len(source_rows) != 799 or len(future_rows) != 1989:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: reference row counts changed source={len(source_rows)} future={len(future_rows)}')
    required_cols = {'sha256', 'Global24__prob', 'A25__score', 'A25__routed', 'Global24__A25__JointS7_ordinary__prob'}
    if not required_cols.issubset(source_rows[0]) or not required_cols.issubset(future_rows[0]):
        raise RuntimeError('BLOCKED_BY_CONTRACT: P31 reference does not satisfy frozen CARE reference_arrays schema')
    route = sum(int(float(r['A25__routed'])) for r in future_rows)
    if route != 497:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: expected 497 future P31-DI25 routes, got {route}')

    resolved = {
        'future_runner': str(runner),
        'c43_future_region_cache': str(cache),
        'future_inter_cache': str(FUTURE_INTER),
        'eval_features': str(OUT / 'preflight/compact_features_2017_2021_eval_only.csv'),
        'previous_result': str(ref_dir),
        'p31_future_reference_sha256': file_sha256(ref_dir / 'future_predictions_diagnostic.csv'),
        'future_rows': 1989,
        'future_route_total': 497,
        'future_selection_or_tuning_allowed': False,
    }
    save_json(OUT / 'preflight/resolved_inputs.json', resolved)
    print('P31_C43_FUTURE_PREFLIGHT_PASS')
    for k, v in resolved.items():
        print(f'{k}={v}')


if __name__ == '__main__':
    main()
