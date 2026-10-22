#!/usr/bin/env python3
"""Run formal Compact and Endpoint extraction for the frozen missing subset."""
from __future__ import annotations

import json
import os
import subprocess

from common import (COMPACT, EDGE, ENDPOINT, OUT, PRIORITY, RESOLVED, ROOT,
                    compact_ids, endpoint_ids, read_csv, write_json)


def invoke(command):
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT / 'scripts/care-di25-host-gatedgraph-source-v1')
    env['TMPDIR'] = str(OUT / 'tmp'); env['XDG_CACHE_HOME'] = str(OUT / 'cache_home')
    os.makedirs(env['TMPDIR'], exist_ok=True); os.makedirs(env['XDG_CACHE_HOME'], exist_ok=True)
    print('AZ_EXTRACT_COMMAND', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), env=env, check=False).returncode


def main():
    lineage = OUT / 'pending_future_lineage_label_free.csv'
    rows = read_csv(lineage)
    if not rows: raise RuntimeError('empty pending lineage')
    mapping = json.loads(RESOLVED.read_text(encoding='utf-8'))['mapping']
    compact_out = OUT / 'stage2_missing_future/compact'
    endpoint_out = OUT / 'stage2_missing_future/endpoint'
    rc1 = invoke([os.sys.executable, COMPACT, '--lineage', lineage, '--mapping', mapping,
        '--priority', PRIORITY, '--output', compact_out, '--progress-every', 10, '--min-free-gb', 5])
    rc2 = invoke([os.sys.executable, ENDPOINT, '--lineage', lineage, '--mapping', mapping,
        '--priority', PRIORITY, '--edge-semantics', EDGE, '--output', endpoint_out,
        '--limit', 0, '--resume', '--upstream-hops', 2, '--downstream-hops', 1,
        '--K-in', 8, '--K-out', 4, '--endpoint-cap', 256, '--node-cap', 2048,
        '--edge-cap', 4096, '--reachability-max-states', 100000,
        '--min-free-gb', 5, '--progress-every', 10])
    c = compact_ids(compact_out / 'features.csv')
    e = endpoint_ids(endpoint_out / 'endpoint_centered_tensor_cache.jsonl.gz')
    expected = {r['sha256'].lower() for r in rows}; complete = c & e
    write_json(OUT / 'new_stage2_asset_audit.json', {'requested': len(expected),
        'compact_ok': len(c), 'endpoint_ok': len(e), 'complete_intersection': len(complete),
        'compact_rc': rc1, 'endpoint_rc': rc2,
        'success_rate': len(complete)/max(len(expected), 1), 'labels_loaded': False})
    if len(complete) < .90 * len(expected):
        raise RuntimeError(f'Stage-II coverage below 90%: {len(complete)}/{len(expected)}')
    print(f'AZ_AVAILABLE_STAGE2_DONE complete={len(complete)}/{len(expected)}', flush=True)


if __name__ == '__main__': main()

