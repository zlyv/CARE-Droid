#!/usr/bin/env python3
"""Read-only Host identity and S7 exact-rebuild probe for coherent ablation feasibility.

Role: independent G1/G2 feasibility test. Writes only fresh v3 output; does not
modify APKs, original endpoint/feature caches, checkpoints or formal scripts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
OUT = ROOT / 'outputs/care_host_region_attribution_v3'
ENDPOINT = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
MANIFEST = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json')


def write(name, obj):
    path = OUT / name
    if path.exists():
        raise RuntimeError('refusing overwrite ' + str(path))
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def asset(path):
    path = path.resolve(strict=True)
    st = path.stat()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest(path))


def payload_of(sid):
    found = None
    with gzip.open(ENDPOINT, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if str(row.get('sha256', '')).casefold() == sid and row.get('status') == 'OK':
                if found is not None and found != row['payload']:
                    raise RuntimeError('conflicting endpoint payload')
                found = row['payload']
    if found is None:
        raise RuntimeError('endpoint payload not found')
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-only', action='store_true', required=True)
    p.parse_args()
    if OUT.exists():
        raise RuntimeError('v3 output exists')
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / 'tmp').mkdir(mode=0o700)
    os.environ['TMPDIR'] = str(OUT / 'tmp')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    paths = [ENDPOINT, MANIFEST,
             ROOT / 'outputs/care_multiyear_development_v1/data.npz',
             ROOT / 'outputs/care_multiyear_development_v1/predictions/T3/router.npz',
             ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/future_interz.npz',
             ROOT / 'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py',
             ROOT / 'scripts/run_clipped_intra_inter_structure_ablation_2017.py',
             ROOT / 'scripts/care-di25-host-gatedgraph-source-v1/run_benefitdroid_e1_exact_pair_2017.py',
             ROOT / 'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py']
    before = [asset(q) for q in paths]
    write('asset_manifest_before.json', before)
    sys.path.insert(0, str(ROOT / 'scripts/care-crossyear-layerwise-diagnosis-v2'))
    import diagnostic_common as dc
    _, v4, args, data = dc.original()
    sys.path.insert(0, str(ROOT / 'scripts/care-di25-compact43-relationquery-source-v1'))
    from hostrel_features import reconstruct_host_relations
    start = time.monotonic()
    with np.load(ROOT / 'outputs/care_multiyear_development_v1/predictions/T3/router.npz', allow_pickle=False) as z:
        ids = np.asarray(z['ids']).astype(str)
        route = np.asarray(z['new_route'], bool)
        s1 = np.asarray(z['s1'], float)
    pos = {str(s): i for i, s in enumerate(data['ids'])}
    selected = [s for s, r, q in zip(ids, route, s1) if r and q < .5]
    if len(selected) != 77:
        raise RuntimeError('2020 77 cohort changed')
    sid = min(selected, key=lambda s: int(data['real_host_count'][pos[s]]))
    i = pos[sid]
    payload = payload_of(sid)
    mapping = np.asarray(json.loads(MANIFEST.read_text(encoding='utf-8'))['payload_to_model_role'], int)
    records = reconstruct_host_relations(payload, mapping)
    begin = int(data['host_offsets'][i]); end = int(data['host_offsets'][i+1])
    bag = np.asarray(data['host_values'][begin:end], np.float32)
    count = int(data['real_host_count'][i])
    if count != len(records) or len(bag) != count:
        raise RuntimeError('Host count mismatch in min-Host pilot')
    if len(set(r['host_id'] for r in records)) != len(records):
        raise RuntimeError('endpoint Host stable IDs not unique')
    presence = np.stack([r['presence'] for r in records]).astype(np.float32)
    counts = np.stack([r['endpoint_counts'] for r in records]).astype(np.float32)
    relation = np.stack([r['directed_neighbor_role'] for r in records]).astype(np.float32)
    differences = dict(presence=float(np.max(np.abs(bag[:, :31]-presence))),
                       endpoint_counts=float(np.max(np.abs(bag[:,43:74]-counts))),
                       relation31=float(np.max(np.abs(bag[:,74:105]-relation))))
    print('HRA_V3_HOST', count, differences, flush=True)
    # The original frozen builder must reconstruct all three graph blocks,
    # including the randomization-based Inter-Z; no approximation is accepted.
    cache = {sid: {'payload': payload}}
    row = {'sha256': sid}
    s3, _ = v4.audit.future.reconstruct_structure([row], cache, 31)
    if s3.shape != (1,585):
        raise RuntimeError('S3 width differs')
    inter_ref = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/future_interz.npz'
    with np.load(inter_ref, allow_pickle=False) as z:
        settings = {k: z[k].item() for k in ('null_samples','swaps_per_edge','null_seed','z_clip')}
    for key,value in settings.items():
        setattr(args,key,value)
    args.progress_every = 1
    _, _, inter_z, _ = v4.v3.future_s7.clipped.build_multiscale_features(
        [row], cache, s3[:,120:], args)
    s7 = np.concatenate([s3, np.clip(inter_z, -float(args.z_clip),float(args.z_clip))],axis=1)
    saved = np.asarray(data['s7'][i:i+1],np.float64)
    graph_diff = dict(stats120=float(np.max(np.abs(s7[:,:120]-saved[:,:120]))),
                      pair465=float(np.max(np.abs(s7[:,120:585]-saved[:,120:585]))),
                      inter961=float(np.max(np.abs(s7[:,585:]-saved[:,585:]))),
                      max_abs=float(np.max(np.abs(s7-saved))))
    status = 'PASS_EXACT_REBUILD' if all(x <= 1e-6 for x in differences.values()) and graph_diff['max_abs'] <= 1e-6 else 'FAIL_REBUILD'
    write('graph_rebuild_probe.json', dict(status=status, apk_anonymous_id='2020_smallest_host_in_77',
        real_hosts=count, host_identity_unique=True,
        host_column_max_abs_difference=differences, s7_block_max_abs_difference=graph_diff,
        frozen_multiscale_settings=settings, seconds=time.monotonic()-start,
        next_step='Only if exact: validate more samples, then implement coherent modified graph with every dependent Host/Relation/S7 field recomputed; no model intervention yet.'))
    after = [asset(q) for q in paths]
    changed = [dict(before=b, after=a) for b,a in zip(before,after) if b != a]
    write('asset_manifest_after.json',after)
    write('baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',changed=changed,
        asset_count=len(before),scope='listed reused files only; no APK opened'))
    if changed:
        raise RuntimeError('asset integrity failure')
    print('HRA_V3_STATUS='+status,flush=True)


if __name__ == '__main__':
    main()
