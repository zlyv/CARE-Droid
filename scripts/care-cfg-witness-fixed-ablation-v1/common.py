#!/usr/bin/env python3
"""Shared contracts for CFG Witness fixed-capacity training; reads caches only and writes only the new output directory."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-cfg-witness-fixed-ablation-v1'
OUT = ROOT / 'outputs/care_cfg_witness_fixed_ablation_v1'
WITNESS = ROOT / 'artifacts/care_cfg_witness_all_years_v2'
BFA = ROOT / 'scripts/care-background-functional-attribution-v1'
SCBE = ROOT / 'outputs/care_scbe_v1'
MYD = ROOT / 'scripts/care-multiyear-development-v1'
sys.path[:0] = [str(HERE), str(BFA), str(ROOT/'scripts/care-scbe-v1'), str(MYD)]

SEEDS = (900000, 900001, 900002)
SPLITS = ('S1','S2','S3','S4','S5','T1','T2','T3','T4')
ARMS = ('W0_FROZEN_BASELINE','W1_WITNESS3','W2_SIZE3_CONTROL','W3_SIZE_CONDITIONAL_SHUFFLE')
WITNESS_COLS = ('bidirectional_pairs','witness_pairs_interz_zero','witness_unique_fraction_vs_interz')
SIZE_COLS = ('frozen_hosts','aligned_sensitive_calls','hosts_with_two_calls')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def encode(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=encode)+'\n', encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, values):
    values = list(values); fields = list(dict.fromkeys(k for r in values for k in r)) or ['status']
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(values)
    temp.replace(path)


def write_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def asset_paths():
    old = ROOT/'outputs/care_multiyear_development_v1'
    paths = [HERE/'common.py', HERE/'cfgwa_common.py', HERE/'train.py', HERE/'evaluate.py',
             BFA/'bfa_core.py', ROOT/'scripts/care-scbe-v1/train.py',
             ROOT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
             MYD/'metrics.py', old/'data.npz', old/'feature_rows.pkl', old/'sample_manifest.csv',
             WITNESS/'cfg_witness_apk.csv', WITNESS/'final_decision.json',
             WITNESS/'baseline_integrity_check.json']
    paths += [old/'predictions'/split/'router.npz' for split in SPLITS[5:]]
    paths += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl' for split in SPLITS for seed in SEEDS]
    return paths


def verify_assets():
    manifest = OUT/'asset_manifest_before.json'
    if not manifest.is_file(): return {'status':'MISSING'}
    changed = []
    for item in json.loads(manifest.read_text(encoding='utf-8')):
        try: now = digest(item['path'])
        except OSError as exc: changed.append({'path':item['path'],'error':repr(exc)}); continue
        if now != item: changed.append({'before':item,'after':now})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(json.loads(manifest.read_text())),
            'changed':changed,'scope':'named code, frozen features/routes/checkpoints and Witness table; no APK bytes'}


def preflight(resume=False):
    if OUT.exists() and not resume: raise RuntimeError('output exists; use --resume, never overwrite')
    if not OUT.exists():
        OUT.mkdir(); (OUT/'models').mkdir(); (OUT/'tmp').mkdir(); (OUT/'logs').mkdir()
        decision = json.loads((WITNESS/'final_decision.json').read_text())
        integrity = json.loads((WITNESS/'baseline_integrity_check.json').read_text())
        if decision.get('status') != 'SUPPORTED_FOR_FIXED_ABLATION' or integrity.get('status') != 'PASS':
            raise RuntimeError('Witness audit did not authorize the fixed ablation')
        table = rows(WITNESS/'cfg_witness_apk.csv')
        if len(table) != 2788 or len({r['sha256'] for r in table}) != 2788:
            raise RuntimeError('Witness table is not 2,788 unique APKs')
        atomic_json(OUT/'asset_manifest_before.json', [digest(p) for p in asset_paths()])
        atomic_json(OUT/'protocol.json', {
            'status':'FROZEN','development_years':[2017,2018,2019,2020,2021],
            'future_years_are_development':True,'Stage1_changed':False,'Router_changed':False,
            'threshold':0.5,'global_width_baseline':24,'global_width_controls':27,
            'arms':ARMS,'witness_columns':WITNESS_COLS,'size_control_columns':SIZE_COLS,
            'selection':'three candidates frozen by the completed all-years audit; no further feature or threshold search'})
    result = verify_assets()
    if result['status'] != 'PASS': raise RuntimeError('asset integrity failed')
    atomic_json(OUT/'preflight.json', {**result,'APKs':2788,'routed_APKs':697,'APK_files_opened':0,'model_training_authorized':True})
    return result


def witness_map():
    result = {r['sha256'].lower(): r for r in rows(WITNESS/'cfg_witness_apk.csv')}
    if len(result) != 2788: raise RuntimeError('Witness membership changed')
    return result


def raw_matrix(ids, columns, mapping):
    out = []
    for sid in map(str, ids):
        row = mapping[sid.lower()]
        values = [float(row[c]) for c in columns]
        if columns != WITNESS_COLS:
            values = [np.log1p(max(v, 0.0)) for v in values]
        else:
            values[:2] = [np.log1p(max(v, 0.0)) for v in values[:2]]
        out.append(values)
    result = np.asarray(out, np.float32)
    if result.shape != (len(ids), 3) or not np.isfinite(result).all(): raise RuntimeError('invalid auxiliary matrix')
    return result


def standardize(train, val):
    mean = train.mean(0); std = train.std(0); std[std < 1e-6] = 1.0
    return ((train-mean)/std).astype(np.float32), ((val-mean)/std).astype(np.float32), mean, std


def size_bins(train_size, values_size):
    edges = [np.unique(np.quantile(train_size[:, j], [.25,.5,.75])) for j in range(2)]
    codes = np.zeros(len(values_size), np.int16)
    for j in range(2): codes = codes*4 + np.digitize(values_size[:,j], edges[j], right=True)
    return codes


def conditional_shuffle(values, size, train_size, rng):
    output = values.copy(); codes = size_bins(train_size, size)
    for code in np.unique(codes):
        index = np.flatnonzero(codes == code)
        if len(index) > 1: output[index] = values[rng.permutation(index)]
    return output


def augment_features(data, train_index, val_index, arm, seed):
    mapping = witness_map(); train_ids = data['ids'][train_index]; val_ids = data['ids'][val_index]
    size_train = raw_matrix(train_ids, SIZE_COLS, mapping); size_val = raw_matrix(val_ids, SIZE_COLS, mapping)
    if arm == 'W1_WITNESS3':
        raw_train = raw_matrix(train_ids, WITNESS_COLS, mapping); raw_val = raw_matrix(val_ids, WITNESS_COLS, mapping)
        protocol = 'frozen CFG Witness values'
    elif arm == 'W2_SIZE3_CONTROL':
        raw_train, raw_val = size_train, size_val; protocol = 'same-width size/call-count control'
    elif arm == 'W3_SIZE_CONDITIONAL_SHUFFLE':
        witness_train = raw_matrix(train_ids, WITNESS_COLS, mapping); witness_val = raw_matrix(val_ids, WITNESS_COLS, mapping)
        raw_train = conditional_shuffle(witness_train, size_train, size_train, np.random.RandomState(seed+47001))
        raw_val = conditional_shuffle(witness_val, size_val, size_train, np.random.RandomState(seed+47002))
        protocol = 'label-free Witness shuffle within train-derived size strata'
    else: raise ValueError(arm)
    train, val, mean, std = standardize(raw_train, raw_val)
    return train, val, {'protocol':protocol,'mean':mean,'std':std,'columns':WITNESS_COLS if arm!='W2_SIZE3_CONTROL' else SIZE_COLS}
