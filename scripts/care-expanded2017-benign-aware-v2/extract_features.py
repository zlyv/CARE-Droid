#!/usr/bin/env python3
"""Run resumable formal P31 and Stage-II asset extraction; writes only EXPANDED_OUTPUT.

No APK is installed, executed, changed, or copied. The formal parsers and mapping
are imported from existing read-only experiment code and checked against old P31.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from audit_new_apks import OUT, PROJECT, read_csv, write_csv

MALEVAL = PROJECT / 'scripts/care-maleval2026-external-v1'
SOURCE_P31 = PROJECT / 'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
SOURCE_RESOLVED = PROJECT / 'artifacts/care-litelocal-semantic-source-v1/resolved_inputs.json'
PRIORITY = PROJECT / 'configs/protocol/sensitive_endpoint_priority_v1.yaml'
EDGE_SEMANTICS = PROJECT / 'configs/protocol/semantic_edge_mapping_v1.yaml'
COMPACT = PROJECT / 'scripts/care-di25-host-gatedgraph-source-v1/run_compact_feature_extraction_ag.py'
ENDPOINT = PROJECT / 'scripts/care-di25-host-gatedgraph-source-v1/run_endpoint_centered_tensor_cache_ag.py'


def load_extractor():
    sys.path.insert(0, str(MALEVAL))
    from extract_p31 import extract, suppress
    suppress()
    mapping_path = Path(json.loads(SOURCE_RESOLVED.read_text())['mapping'])
    mapping = yaml.safe_load(mapping_path.read_text())
    roles = list(mapping['roles'])
    assert len(roles) == 31
    rules = [r for r in mapping['rules'] if r.get('rule_kind', 'api') == 'api']
    return extract, rules, {r: i for i, r in enumerate(roles)}, mapping_path, roles


def verify_formal_p31(extract, rules, index):
    lineage = read_csv(json.loads(SOURCE_RESOLVED.read_text())['lineage'])
    with np.load(SOURCE_P31, allow_pickle=False) as a:
        ids = [str(x).lower() for x in a['sha256']]
        values = a['role_prevalence']
        lookup = {s: i for i, s in enumerate(ids)}
        for row in lineage:
            sid = row['sha256'].lower()
            if sid not in lookup or not Path(row['absolute_path']).is_file(): continue
            result, _ = extract(row, rules, index)
            diff = float(np.max(np.abs(result.astype(float) - values[lookup[sid]].astype(float))))
            if diff > 1e-6:
                raise RuntimeError(f'formal P31 replay mismatch sha={sid} max_abs={diff}; extraction forbidden')
            print('P31_FORMAL_REPLAY_PASS', sid, diff, flush=True)
            return sid
    raise RuntimeError('no old source APK available for P31 replay')


def save_shard(path, p31, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, p31=p31, audit=audit)


def extract_p31(rows, limit, resume):
    extract, rules, index, mapping_path, roles = load_extractor()
    validation = OUT / 'formal_p31_replay.json'
    if not validation.exists():
        sid = verify_formal_p31(extract, rules, index)
        validation.write_text(json.dumps({'status': 'PASS', 'sha256': sid,
                 'mapping_path': str(mapping_path), 'role_names': roles}, indent=2) + '\n')
    shards = OUT / 'p31_shards'
    shards.mkdir(exist_ok=True)
    failures = OUT / 'p31_failures.csv'
    previous = {r['sha256'] for r in read_csv(failures)} if failures.exists() else set()
    selected = rows[:limit] if limit else rows
    fields = ['sha256', 'year', 'status', 'seconds', 'error_type', 'error']
    for n, row in enumerate(selected, 1):
        sid = row['sha256'].lower()
        dest = shards / (sid + '.npz')
        if dest.exists():
            with np.load(dest, allow_pickle=False) as a:
                if a['p31'].shape != (31,) or not np.isfinite(a['p31']).all():
                    raise RuntimeError('invalid existing shard: ' + str(dest))
            continue
        if sid in previous and resume: continue
        t = time.monotonic()
        try:
            p31, audit = extract({'absolute_path': row['apk_path']}, rules, index)
            if p31.shape != (31,) or not np.isfinite(p31).all():
                raise ValueError('invalid P31 dimension/values')
            save_shard(dest, p31, audit)
            status, error_type, error = 'OK', '', ''
        except Exception as exc:
            status, error_type, error = 'FAILED', type(exc).__name__, str(exc)[:400]
        if status == 'FAILED':
            existed = failures.exists()
            with failures.open('a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if not existed: w.writeheader()
                w.writerow({'sha256': sid, 'year': row['year'], 'status': status,
                            'seconds': round(time.monotonic()-t, 3),
                            'error_type': error_type, 'error': error})
        if n % 25 == 0 or n == len(selected):
            print(f'P31_PROGRESS {n}/{len(selected)} failed={len(previous)} latest={status}', flush=True)
    ok = sum((shards / (r['sha256'].lower() + '.npz')).exists() for r in selected)
    print(f'P31_BATCH_DONE ok={ok}/{len(selected)}', flush=True)
    return ok


def lineage_rows(rows, path):
    prepared = []
    for r in rows:
        prepared.append({'sha256': r['sha256'], 'absolute_path': r['apk_path'],
                         'source_id': r['source_dataset'], 'lineage_id': r['source_dataset'],
                         'metadata_year': r['year'],
                         'metadata_label': ('benign' if r['normalized_label'] == '0' else
                                            'malicious' if r['normalized_label'] == '1' else '')})
    write_csv(path, prepared)


def invoke_builder(command):
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(PROJECT / 'src') + ':' + str(PROJECT / 'scripts/care-di25-host-gatedgraph-source-v1')
    env['TMPDIR'] = str(OUT / 'tmp')
    env['XDG_CACHE_HOME'] = str(OUT / 'cache_home')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    Path(env['XDG_CACHE_HOME']).mkdir(exist_ok=True)
    print('EXTRACT_COMMAND', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), env=env, check=False).returncode


def stage2_assets(rows, name, resume):
    selected = list(rows)
    if not selected: return
    _, _, _, mapping, _ = load_extractor()
    base = OUT / ('stage2_' + name)
    base.mkdir(exist_ok=True)
    lineage = base / 'lineage.csv'
    if not lineage.exists(): lineage_rows(selected, lineage)
    else:
        old_ids = [r['sha256'] for r in read_csv(lineage)]
        if old_ids != [r['sha256'] for r in selected]:
            raise RuntimeError('resume lineage changed: ' + str(lineage))
    compact = base / 'compact'
    endpoint = base / 'endpoint'
    cmd = [sys.executable, COMPACT, '--lineage', lineage, '--mapping', mapping,
           '--priority', PRIORITY, '--output', compact, '--progress-every', 10,
           '--min-free-gb', 2]
    rc_compact = invoke_builder(cmd)
    cmd = [sys.executable, ENDPOINT, '--lineage', lineage, '--mapping', mapping,
           '--priority', PRIORITY, '--edge-semantics', EDGE_SEMANTICS,
           '--output', endpoint, '--limit', 0, '--upstream-hops', 2,
           '--downstream-hops', 1, '--K-in', 8, '--K-out', 4,
           '--endpoint-cap', 256, '--node-cap', 2048, '--edge-cap', 4096,
           '--reachability-max-states', 100000, '--min-free-gb', 2,
           '--progress-every', 10]
    if resume: cmd.append('--resume')
    rc_endpoint = invoke_builder(cmd)
    (base / 'builder_status.json').write_text(json.dumps({'compact_rc': rc_compact,
        'endpoint_rc': rc_endpoint, 'requested': len(selected)}, indent=2) + '\n')
    if rc_compact or rc_endpoint:
        print(f'STAGE2_ASSETS_PARTIAL name={name} compact_rc={rc_compact} endpoint_rc={rc_endpoint}', flush=True)
        if name == 'smoke':
            raise RuntimeError('Stage-II smoke extraction failed; do not launch full extraction')


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true')
    group.add_argument('--extract-all', action='store_true')
    group.add_argument('--extract-stage2-train', action='store_true')
    group.add_argument('--extract-stage2-routed', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if not OUT.is_dir() or args.workers < 1 or args.workers > 2:
        raise RuntimeError('audit output missing or workers outside safe range')
    rows = [r for r in read_csv(OUT / 'dataset_manifest.csv') if r['source_dataset'] != 'OLD_2017'
            and r['duplicate_status'] == 'UNIQUE_NEW']
    if args.smoke_test:
        benign = [r for r in rows if r['year'] == '2017' and r['normalized_label'] == '0'][:10]
        if len(benign) != 10: raise RuntimeError('smoke benign cohort incomplete')
        ok = extract_p31(benign, 0, args.resume)
        if ok < 8: raise RuntimeError('P31 smoke coverage <8/10')
        stage2_assets(benign[:3], 'smoke', args.resume)
    elif args.extract_all:
        extract_p31(rows, 0, args.resume)
    elif args.extract_stage2_train:
        train = [r for r in read_csv(OUT / 'union_2017_manifest.csv')
                 if r['source_dataset'] != 'OLD_2017'
                 and (OUT / 'p31_shards' / (r['sha256'].lower() + '.npz')).is_file()]
        if len(train) < 0.9 * (len(read_csv(OUT / 'union_2017_manifest.csv')) - 799):
            raise RuntimeError('eligible 2017 P31 extraction below pre-registered 90% floor')
        stage2_assets(train, 'union_new_2017', args.resume)
    else:
        routed = read_csv(OUT / 'routed_new_future_label_free.csv')
        prepared = [{'sha256': r['sha256'], 'apk_path': r['absolute_path'],
                     'source_dataset': r['source_id'], 'year': r['metadata_year'],
                     'normalized_label': ''} for r in routed]
        stage2_assets(prepared, 'routed_new_future', args.resume)


if __name__ == '__main__': main()
