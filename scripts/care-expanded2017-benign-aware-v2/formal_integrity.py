#!/usr/bin/env python3
"""Hash the named formal assets reused by V2; never modify those assets."""
from __future__ import annotations

import argparse
import json

from audit_new_apks import OUT, PROJECT, file_record, write_json


def protected_paths():
    relative = [
        'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz',
        'artifacts/care-droid-formal-diagnostics-v1/resolved_inputs.json',
        'outputs/care_multiyear_development_v1/data.npz',
        'scripts/care-multiyear-development-v1/train_stage1.py',
        'scripts/care-multiyear-development-v1/train_stage2.py',
        'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
        'scripts/care-svm-train-window-expansion-v1/stage1_core.py',
        'scripts/care-di25-host-gatedgraph-source-v1/run_compact_feature_extraction_ag.py',
        'scripts/care-di25-host-gatedgraph-source-v1/run_endpoint_centered_tensor_cache_ag.py',
        'configs/protocol/sensitive_endpoint_priority_v1.yaml',
        'configs/protocol/semantic_edge_mapping_v1.yaml',
    ]
    relative += ['outputs/care_multiyear_development_v1/models/stage1/S0R_2017/'+m+'.pkl'
                 for m in ('LR','RF','XGB','SVM')]
    relative += ['outputs/care_multiyear_development_v1/models/stage2/C0R_2017/'+str(s)+'.pkl'
                 for s in (900000,900001,900002)]
    return [PROJECT / p for p in relative]


def verify():
    before = json.loads((OUT / 'formal_asset_manifest_before.json').read_text())['assets']
    changed = []
    for record in before:
        path = record['path']
        try:
            current = file_record(path)
        except OSError as exc:
            changed.append({'path': path, 'error': str(exc)})
            continue
        if any(current[k] != record[k] for k in ('sha256','size','mtime_ns')):
            changed.append({'path': path, 'before': record, 'after': current})
    return {'status': 'PASS' if not changed else 'FAIL', 'asset_count': len(before),
            'changed': changed, 'scope': 'named reused formal code, checkpoints, P31 and data files'}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--snapshot', action='store_true')
    g.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.snapshot:
        write_json(OUT / 'formal_asset_manifest_before.json',
                   {'assets': [file_record(path) for path in protected_paths()]})
        print('EXPANDED_FORMAL_ASSET_SNAPSHOT_PASS', len(protected_paths()), flush=True)
    else:
        result = verify()
        write_json(OUT / 'formal_asset_integrity_check.json', result)
        print('EXPANDED_FORMAL_ASSET_INTEGRITY', result['status'], flush=True)
        if result['changed']: raise RuntimeError('formal asset integrity changed')


if __name__ == '__main__': main()
