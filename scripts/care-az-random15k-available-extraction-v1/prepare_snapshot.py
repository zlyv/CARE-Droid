#!/usr/bin/env python3
"""Freeze the available audited APK snapshot and compute only missing work.

Reads prior manifests/caches and APK paths without modifying them. Writes only
the new experiment output directory. Labels are deliberately omitted from the
extraction lineage.
"""
from __future__ import annotations

import json
from collections import Counter

from common import (COMPACT, EDGE, ENDPOINT, OUT, PRIOR, PRIORITY, RESOLVED,
                    digest, endpoint_ids, compact_ids, read_csv, write_csv,
                    write_json)


def main():
    if OUT.exists():
        raise RuntimeError('output already exists; use resume.sh, never overwrite')
    OUT.mkdir(parents=True, exist_ok=False); (OUT / 'logs').mkdir()
    manifest = read_csv(PRIOR / 'dataset_manifest.csv')
    eligible = [r for r in manifest if r.get('source_dataset') != 'OLD_2017'
                and r.get('duplicate_status') == 'UNIQUE_NEW'
                and (PRIOR / 'p31_shards' / (r['sha256'].lower() + '.npz')).is_file()]
    old_2017 = {r['sha256'].lower() for r in read_csv(PRIOR / 'stage2_union_new_2017/lineage.csv')}
    old_future = {r['sha256'].lower() for r in read_csv(PRIOR / 'stage2_routed_new_future/lineage.csv')}
    old_2017_ok = compact_ids(PRIOR / 'stage2_union_new_2017/compact/features.csv') & endpoint_ids(
        PRIOR / 'stage2_union_new_2017/endpoint/endpoint_centered_tensor_cache.jsonl.gz')
    old_future_ok = compact_ids(PRIOR / 'stage2_routed_new_future/compact/features.csv') & endpoint_ids(
        PRIOR / 'stage2_routed_new_future/endpoint/endpoint_centered_tensor_cache.jsonl.gz')
    if old_2017 != old_2017_ok or old_future != old_future_ok:
        raise RuntimeError('reused Stage-II cache is not complete for its frozen lineage')

    snapshot, pending = [], []
    for row in eligible:
        sid = row['sha256'].lower(); year = int(row['year'])
        if sid in old_2017_ok: status = 'REUSE_COMPLETE_2017'
        elif sid in old_future_ok: status = 'REUSE_COMPLETE_FUTURE_ROUTED'
        else: status = 'PENDING_STAGE2'
        item = {'sha256': sid, 'year': year, 'absolute_path': row['apk_path'],
                'p31_shard': str(PRIOR / 'p31_shards' / (sid + '.npz')),
                'stage2_status_before': status}
        snapshot.append(item)
        if year >= 2018 and status == 'PENDING_STAGE2':
            pending.append({'sha256': sid, 'absolute_path': row['apk_path'],
                'source_id': 'AZ_2017_2022_Random15K_AVAILABLE_SNAPSHOT',
                'lineage_id': 'AZ_2017_2022_Random15K_AVAILABLE_SNAPSHOT',
                'metadata_year': year, 'metadata_label': ''})
    unexplained = [r for r in snapshot if r['year'] == 2017 and r['stage2_status_before'] == 'PENDING_STAGE2']
    if unexplained:
        raise RuntimeError(f'2017 completed-cache mismatch count={len(unexplained)}')
    write_csv(OUT / 'snapshot_manifest_label_free.csv', snapshot)
    write_csv(OUT / 'pending_future_lineage_label_free.csv', pending)
    counts = Counter(r['stage2_status_before'] for r in snapshot)
    years = Counter(str(r['year']) for r in snapshot)
    write_json(OUT / 'coverage_before.json', {'status': 'PASS', 'labels_loaded': False,
        'p31_complete': len(snapshot), 'stage2_status': dict(counts),
        'by_year': dict(years), 'pending_future': len(pending),
        'policy': 'reuse complete read-only caches; extract only missing future Stage-II assets'})
    assets = [PRIOR / 'dataset_manifest.csv', PRIOR / 'formal_p31_replay.json',
        PRIOR / 'stage2_union_new_2017/lineage.csv',
        PRIOR / 'stage2_union_new_2017/compact/features.csv',
        PRIOR / 'stage2_union_new_2017/endpoint/endpoint_centered_tensor_cache.jsonl.gz',
        PRIOR / 'stage2_routed_new_future/lineage.csv',
        PRIOR / 'stage2_routed_new_future/compact/features.csv',
        PRIOR / 'stage2_routed_new_future/endpoint/endpoint_centered_tensor_cache.jsonl.gz',
        COMPACT, ENDPOINT, RESOLVED, PRIORITY, EDGE]
    write_json(OUT / 'asset_manifest_before.json', [digest(p) for p in assets])
    print('AZ_SNAPSHOT_READY', json.dumps({'p31': len(snapshot), 'pending': len(pending),
        'reused_2017': len(old_2017_ok), 'reused_future': len(old_future_ok)}), flush=True)


if __name__ == '__main__': main()

