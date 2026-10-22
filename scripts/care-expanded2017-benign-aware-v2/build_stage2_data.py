#!/usr/bin/env python3
"""Build frozen-width Direct-C43 input bags from isolated new APK caches.

Only EXPANDED_OUTPUT receives files. Original Host extraction, Role mapping,
Relation31 construction, S7 and Compact76 preprocessing code is reused without
editing formal assets. Future labels are deliberately not loaded here.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from audit_new_apks import OUT, PROJECT, read_csv, write_csv

MYD = PROJECT / 'scripts/care-multiyear-development-v1'
DIAG = PROJECT / 'artifacts/care-droid-formal-diagnostics-v1'


def norm(x): return str(x).strip().casefold()


def load_endpoint_rows(path):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'OK': result[norm(r['sha256'])] = r
    return result


def payload_mapping():
    p = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json')
    mapping = json.loads(p.read_text())['payload_to_model_role']
    value = np.asarray(mapping, np.int64)
    if value.shape != (31,): raise RuntimeError('payload Role mapping width changed')
    return value


def build_new_bags(v4, args, rows, endpoint_cache, endpoint_rows, scope):
    import hostrel_features
    args.region_cache = OUT / f'cache/{scope}_raw_regions.npz'
    args.region_cache.parent.mkdir(parents=True, exist_ok=True)
    args.rebuild_region_cache = False
    raw_bags, _, audits, _ = v4.care_v1.get_region_bags(rows, endpoint_cache, args)
    mapping = payload_mapping(); bags = []; audit_rows = []
    for row, raw, audit in zip(rows, raw_bags, audits):
        sid = norm(row['sha256']); raw = np.asarray(raw, np.float32)
        payload = endpoint_rows[sid]['payload']
        records_payload = hostrel_features.reconstruct_host_relations(payload, np.arange(31, dtype=np.int64))
        records_model = hostrel_features.reconstruct_host_relations(payload, mapping)
        real_count = int(audit.get('real_host_region_n', len(records_model)))
        if len(records_payload) != real_count or len(records_model) != real_count:
            raise RuntimeError('Host reconstruction count mismatch ' + sid)
        if [int(r['host_node']) for r in records_payload] != [int(r['host_node']) for r in records_model]:
            raise RuntimeError('Host identity/order mismatch ' + sid)
        if real_count:
            base = raw[:real_count]
            if base.shape[1] != 74: raise RuntimeError('raw Host width changed ' + sid)
            original_presence = np.stack([r['presence'] for r in records_payload]).astype(np.float32)
            if not np.allclose(base[:, :31], original_presence, atol=1e-6, rtol=0):
                raise RuntimeError('Host payload-space Role alignment failed ' + sid)
            presence = np.stack([r['presence'] for r in records_model]).astype(np.float32)
            compact43 = np.concatenate([presence, base[:, 62:74]], axis=1)
            count31 = np.stack([r['endpoint_counts'] for r in records_model]).astype(np.float32)
            relation31 = np.stack([r['directed_neighbor_role'] for r in records_model]).astype(np.float32)
            bag105 = np.concatenate([compact43, count31, relation31], axis=1)
        else:
            if raw.shape != (1, 74): raise RuntimeError('empty Host placeholder changed ' + sid)
            compact43 = np.concatenate([raw[:, :31], raw[:, 62:74]], axis=1)
            bag105 = np.concatenate([compact43, np.zeros((1, 62), np.float32)], axis=1)
        bag169 = np.concatenate([bag105, np.zeros((len(bag105), 64), np.float32)], axis=1)
        if bag169.shape[1] != 169: raise RuntimeError('A0 Host width changed')
        bags.append(bag169)
        audit_rows.append({'sha256': sid, 'real_hosts': real_count,
                           'stored_hosts': len(bag169), 'role_alignment': 'EXACT'})
    write_csv(OUT / f'{scope}_host_alignment.csv', audit_rows)
    return bags


def new_s7(v4, args, rows, endpoint, scope):
    s3, _ = v4.audit.future.reconstruct_structure(rows, endpoint, int(v4.audit.exp.ROLE_VOCABULARY_SIZE))
    args.future_inter_cache = OUT / f'cache/{scope}_interz.npz'
    args.future_inter_cache.parent.mkdir(parents=True, exist_ok=True)
    args.rebuild_future_inter_cache = False
    inter, _, _ = v4.v3.future_s7.load_or_build_future_inter(rows, s3, endpoint, args)
    value = np.concatenate([s3, inter], axis=1).astype(np.float32)
    if value.shape != (len(rows), 1546) or not np.isfinite(value).all():
        raise RuntimeError(f'{scope} S7 mismatch {value.shape}')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-stage2-data', action='store_true', required=True)
    parser.add_argument('--resume', action='store_true')
    args_cli = parser.parse_args()
    if str(MYD) not in sys.path: sys.path.insert(0, str(MYD))
    from myd_common import read_json, runtime
    v4, _, args = runtime()
    relation_dir = PROJECT / 'scripts/care-di25-compact43-relationquery-source-v1'
    if str(relation_dir) not in sys.path: sys.path.insert(0, str(relation_dir))
    source_manifest, source_rows, _, compact_by_sha, schema, canonical = v4.audit.load_inputs(args)
    if len(source_rows) != 799: raise RuntimeError('old source rows changed')
    with np.load(PROJECT / 'outputs/care_multiyear_development_v1/data.npz', allow_pickle=False) as a:
        prior = {k: a[k] for k in ('ids','years','y','s7','host_values','host_offsets')}
    old_position = {norm(s): i for i, s in enumerate(prior['ids']) if prior['years'][i] == 2017}
    if len(old_position) != 799: raise RuntimeError('old Host cache source mismatch')
    old_ids = [norm(row['sha256']) for row in source_rows]
    if set(old_ids) != set(old_position): raise RuntimeError('source Host SHA mismatch')
    train_labels = {r['sha256']: int(r['normalized_label']) for r in read_csv(OUT/'union_2017_manifest.csv')}
    new_train_ids = {sid for sid in train_labels if sid not in old_position}
    test_ids = {r['sha256'] for r in read_csv(OUT/'routed_new_future_label_free.csv')}
    train_compact = {norm(r['sha256']): r for r in read_csv(OUT/'stage2_union_new_2017/compact/features.csv') if r.get('status') == 'OK'}
    test_compact = {norm(r['sha256']): r for r in read_csv(OUT/'stage2_routed_new_future/compact/features.csv') if r.get('status') == 'OK'}
    train_endpoint_path = OUT/'stage2_union_new_2017/endpoint/endpoint_centered_tensor_cache.jsonl.gz'
    test_endpoint_path = OUT/'stage2_routed_new_future/endpoint/endpoint_centered_tensor_cache.jsonl.gz'
    train_endpoint_rows = load_endpoint_rows(train_endpoint_path)
    test_endpoint_rows = load_endpoint_rows(test_endpoint_path)
    complete_train = sorted(new_train_ids & set(train_compact) & set(train_endpoint_rows))
    complete_test = sorted(test_ids & set(test_compact) & set(test_endpoint_rows))
    if len(complete_train) < 0.9*len(new_train_ids):
        raise RuntimeError(f'new 2017 Stage-II assets insufficient {len(complete_train)}/{len(new_train_ids)}')
    if len(complete_test) < 0.9*len(test_ids):
        raise RuntimeError(f'routed future Stage-II assets insufficient {len(complete_test)}/{len(test_ids)}')
    new_train_rows = [train_compact[s] for s in complete_train]
    test_rows = [test_compact[s] for s in complete_test]
    for sid in complete_train: compact_by_sha[sid] = train_compact[sid]
    for sid in complete_test: compact_by_sha[sid] = test_compact[sid]
    formal_endpoint = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, formal_endpoint, args)
    train_endpoint = v4.audit.exp.read_cache(train_endpoint_path)
    test_endpoint = v4.audit.exp.read_cache(test_endpoint_path)
    train_new_s7 = new_s7(v4, args, new_train_rows, train_endpoint, 'new2017')
    test_s7 = new_s7(v4, args, test_rows, test_endpoint, 'newfuture')
    prior_values = prior['host_values']; prior_offsets = prior['host_offsets']
    old_bags = [np.asarray(prior_values[prior_offsets[old_position[s]]:prior_offsets[old_position[s]+1]], np.float32)
                for s in old_ids]
    train_new_bags = build_new_bags(v4, args, new_train_rows, train_endpoint, train_endpoint_rows, 'new2017')
    test_bags = build_new_bags(v4, args, test_rows, test_endpoint, test_endpoint_rows, 'newfuture')
    rows = list(source_rows) + new_train_rows + test_rows
    bags = old_bags + train_new_bags + test_bags
    s7 = np.concatenate([source_s7, train_new_s7, test_s7], axis=0)
    ids = np.asarray([norm(r['sha256']) for r in rows])
    years = np.asarray([2017]*len(source_rows)+[2017]*len(new_train_rows)+[int(r['metadata_year']) for r in test_rows], np.int16)
    labels = np.asarray([train_labels[s] for s in old_ids+complete_train]+[-1]*len(test_rows), np.int8)
    if len(set(ids)) != len(ids): raise RuntimeError('Stage-II data SHA duplicate')
    offsets = np.zeros(len(bags)+1, np.int64)
    offsets[1:] = np.cumsum([len(b) for b in bags])
    values = np.concatenate(bags, axis=0)
    if values.shape[1] != 169 or s7.shape != (len(ids),1546): raise RuntimeError('Stage-II width mismatch')
    _, global_matrix, names = v4.audit.compact_matrix_variant(v4.VARIANT, source_rows, rows,
                                                               compact_by_sha, schema, canonical)
    gi, _ = v4.evidence.split_view_indices(names)
    global24 = np.asarray(global_matrix[:,gi], np.float32)
    if global24.shape != (len(ids),24): raise RuntimeError('global24 width mismatch')
    target = OUT/'stage2_data.npz'
    if target.exists(): raise RuntimeError('Stage-II data already exists; no overwrite')
    with target.open('xb') as f:
        np.savez_compressed(f, ids=ids, years=years, y=labels, s7=s7,
                            host_values=values, host_offsets=offsets,
                            global24=global24)
    with (OUT/'stage2_feature_rows.pkl').open('xb') as f:
        pickle.dump({'rows': rows, 'compact_by_sha': compact_by_sha,
                     'schema': schema, 'canonical_names': canonical}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    (OUT/'stage2_data_audit.json').write_text(json.dumps({
        'status': 'PASS', 'old_train': 799, 'new_train_complete': len(complete_train),
        'new_train_total': len(new_train_ids), 'routed_test_complete': len(complete_test),
        'routed_test_total': len(test_ids), 'Host_width': 169, 'S7_width': 1546,
        'global_width': 24, 'test_labels_loaded': False}, indent=2) + '\n')
    print('EXPANDED_STAGE2_DATA_DONE', len(ids), len(values), flush=True)


if __name__ == '__main__': main()
