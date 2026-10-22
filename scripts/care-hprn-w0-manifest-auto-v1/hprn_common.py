#!/usr/bin/env python3
"""Contracts and deterministic Host--Prototype sidecars for the W0 HPRN experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-hprn-w0-manifest-auto-v1'
OUT = ROOT / 'outputs/care_hprn_w0_manifest_auto_v1'
BFA = ROOT / 'scripts/care-background-functional-attribution-v1'
SCBE = ROOT / 'outputs/care_scbe_v1'
MYD = ROOT / 'scripts/care-multiyear-development-v1'
AUTO = ROOT / 'outputs/care_manifest_rf_autonomous_error_router_v1'
TBFRS = ROOT / 'outputs/care-tb-frs-source-v1'
sys.path[:0] = [str(HERE), str(BFA), str(ROOT/'scripts/care-scbe-v1'), str(MYD)]

SEEDS = (900000, 900001, 900002)
MANIFEST_SEEDS = (20261018, 20261019, 20261020)
SOURCE_SPLITS = ('S1', 'S2', 'S3', 'S4', 'S5')
TRAIN_SPLITS = SOURCE_SPLITS + ('T1',)
YEARS = (2017, 2018, 2019, 2020, 2021)
ARMS = ('R0_W0_FROZEN', 'R1_SCALAR_CAPACITY', 'R2_TRUE_RELATION',
        'R3_BINDING_SHUFFLE', 'R4_DISTANCE_CATEGORY')
RELATION_ARMS = ARMS[1:]
BASE_WIDTH = 142
PROTO_WIDTH = 107
BACKGROUND_COUNT = 3
SIDECAR_PER_BACKGROUND = PROTO_WIDTH + 1
SIDECAR_WIDTH = BACKGROUND_COUNT * SIDECAR_PER_BACKGROUND
MODEL_WIDTH = BASE_WIDTH + SIDECAR_WIDTH


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''): h.update(block)
    return {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def encode(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=encode) + '\n', encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, values):
    values = list(values); fields = list(dict.fromkeys(k for row in values for k in row)) or ['status']
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(values)
    temp.replace(path)


def write_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def read_pickle(path):
    with Path(path).open('rb') as handle: return pickle.load(handle)


def asset_paths():
    old = ROOT/'outputs/care_multiyear_development_v1'
    paths = [HERE/'hprn_common.py', HERE/'model.py', HERE/'train.py', HERE/'evaluate.py',
             HERE/'run.sh', HERE/'launch_tmux.sh', HERE/'README.md',
             BFA/'bfa_core.py', ROOT/'scripts/care-scbe-v1/train.py',
             ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
             ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
             ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',
             ROOT/'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py',
             ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py',
             old/'data.npz', old/'feature_rows.pkl', old/'sample_manifest.csv',
             old/'predictions/C0R_2017/expert.npz',
             AUTO/'per_apk_predictions.csv.gz', AUTO/'protocol.json',
             AUTO/'asset_integrity_check.json', AUTO/'final_decision.json',
             TBFRS/'final_decision.json', ROOT/'scripts/care-tb-frs-source-v1/tb_frs.py']
    paths += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl'
              for split in TRAIN_SPLITS for seed in SEEDS]
    return paths


def verify_assets():
    manifest = OUT/'asset_manifest_before.json'
    if not manifest.is_file(): return {'status': 'MISSING'}
    prior = json.loads(manifest.read_text(encoding='utf-8')); changed = []
    for item in prior:
        try: now = digest(item['path'])
        except OSError as exc: changed.append({'path': item['path'], 'error': repr(exc)}); continue
        if now != item: changed.append({'before': item, 'after': now})
    return {'status': 'PASS' if not changed else 'FAIL', 'checked': len(prior), 'changed': changed,
            'scope': 'named experiment code, frozen W0/prototype/data/router assets; no APK bytes'}


def preflight(resume=False):
    if OUT.exists() and not resume: raise RuntimeError('output exists; use --resume, never overwrite')
    if not OUT.exists():
        OUT.mkdir(); (OUT/'models').mkdir(); (OUT/'logs').mkdir(); (OUT/'tmp').mkdir()
        auto_integrity = json.loads((AUTO/'asset_integrity_check.json').read_text(encoding='utf-8'))
        if auto_integrity.get('status') != 'PASS': raise RuntimeError('autonomous Manifest router integrity failed')
        tb = json.loads((TBFRS/'final_decision.json').read_text(encoding='utf-8'))
        atomic_json(OUT/'asset_manifest_before.json', [digest(p) for p in asset_paths()])
        atomic_json(OUT/'protocol.json', {
            'status': 'FROZEN', 'baseline': 'W0 Direct-C43 without CFG Witness',
            'stage1': 'Manifest135 RF',
            'router': 'frozen autonomous Manifest error router; P(RF error)>=0.5; no fixed budget',
            'arms': ARMS, 'training_years': [2017], 'evaluation_years': [2017,2018,2019,2020,2021],
            'future_years_are_development': True, 'future_used_for_selection': False,
            'global_width': 24, 'base_host_width': BASE_WIDTH, 'augmented_host_width': MODEL_WIDTH,
            'prototype_width': PROTO_WIDTH, 'relation_hidden': 16, 'per_background_output': 8,
            'fusion_hidden': 16, 'host_embedding_width': 24,
            'frozen': ['W0 parameters','External24','Shared32','Internal construction','RoleGraph','Global24','Stage1','router'],
            'trainable': 'HPRN projections/MLPs/fusion/zero-initialized residual interface only',
            'loss': 'equal APK-level BCE', 'threshold': 0.5,
            'epochs': 'reuse each W0 checkpoint selected_epochs; no HPRN tuning',
            'R3_shuffle': 'within APK independently per background; preserves prototype multiset',
            'R4_control': 'old scalar distance/commonness plus matched prototype category only',
            'TB_FRS_status': tb.get('status'),
            'TB_FRS_equivalence': 'DIFFERENT: APK-level aggregated Fisher probe, no per-Host relation branch or attention integration',
            'support_rule': '2017 OOF R2 must exceed R0,R1,R3,R4 in BA with positive Net difference and APK-cluster bootstrap lower CI > 0 for every comparison',
        })
    result = verify_assets()
    if result['status'] != 'PASS': raise RuntimeError('asset integrity failed')
    atomic_json(OUT/'preflight.json', {**result, 'APKs': 2788, 'APK_files_opened': 0,
                'future_labels_used_for_training': False, 'model_training_authorized': True})
    return result


def pairwise_rms(a, b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    return np.sqrt(np.maximum(np.mean(np.square(a[:,None,:]-b[None,:,:]),axis=2),0.0))


def safe_scale(values, fallback=1.0):
    values=np.asarray(values,np.float64); selected=values[np.isfinite(values)&(values>1e-8)]
    return max(float(np.median(selected)),1e-6) if len(selected) else float(fallback)


def deterministic_permutation(n, seed, sid, background):
    token=hashlib.sha256(f'{seed}:{sid}:{background}:hprn-binding'.encode()).digest()
    return np.random.RandomState(int.from_bytes(token[:4],'big')).permutation(n)


def internal_matches(x, shared_assignment, coverage):
    counts=np.bincount(shared_assignment, minlength=int(np.max(shared_assignment))+1)
    ordered=np.argsort(-counts); target=max(1,int(math.ceil(float(coverage)*len(x))))
    selected=[]; covered=0
    for cluster in ordered:
        if counts[cluster]<=0: continue
        selected.append(int(cluster)); covered+=int(counts[cluster])
        if covered>=target: break
    centers=np.asarray([np.mean(x[shared_assignment==c],axis=0) for c in selected],np.float32)
    distances=pairwise_rms(x,centers); nearest=np.argmin(distances,axis=1); raw=distances[np.arange(len(x)),nearest]
    scale=safe_scale(raw); old=np.log1p(raw/scale).astype(np.float32)
    categories=np.asarray([selected[i] for i in nearest],np.int16)
    return centers[nearest],old,categories,scale


def build_relation_bags(base_bags, raw_bags, ids, background, arm, seed, coverage=.70):
    """Append fixed-width relation views without changing the first 142 W0 columns."""
    from haig_background import split, hard
    if arm not in RELATION_ARMS: raise ValueError(arm)
    if len(base_bags)!=len(raw_bags) or len(base_bags)!=len(ids): raise RuntimeError('bag alignment')
    raw107, roles, _ = split(raw_bags)
    mean=np.asarray(background['mean'],np.float32); std=np.asarray(background['std'],np.float32)
    state=background['state']; sp=np.asarray(background['sp'],np.float32); bp=np.asarray(background['bp'],np.float32)
    shared=np.asarray(state['shared_centers'],np.float32); benign=np.asarray(state['benign_centers'],np.float32)
    shared_scale=np.asarray(state['shared_scale'],np.float32); external_scale=max(float(state['external_scale']),1e-6)
    commonness=np.asarray(state['commonness'],np.float32)
    output=[]; audits=[]; max_replay=0.0; multiset_error=0.0
    for base,raw,w,sid in zip(base_bags,raw107,roles,map(str,ids)):
        base=np.asarray(base,np.float32); x=((np.asarray(raw,np.float32)-mean)/std).astype(np.float32)
        if base.shape!=(len(x),BASE_WIDTH) or not np.array_equal(base[:,:107],x):
            delta=float(np.max(np.abs(base[:,:107]-x))) if base.shape[0]==len(x) else float('inf')
            if delta>1e-6: raise RuntimeError(f'frozen Compact107 replay mismatch {sid}: {delta}')
        sd=pairwise_rms(x,shared); sa=hard(sd/np.maximum(shared_scale[None,:],1e-6),w,sp)
        sraw=sd[np.arange(len(x)),sa]; svalue=commonness[sa]*np.exp(-sraw/np.maximum(shared_scale[sa],1e-6))
        bd=pairwise_rms(x,benign); ba=hard(bd,w,bp); braw=bd[np.arange(len(x)),ba]
        evalue=np.log1p(braw/external_scale).astype(np.float32)
        iproto,ivalue,icat,iscale=internal_matches(x,sa,coverage)
        old=np.stack([evalue,ivalue,svalue.astype(np.float32)],axis=1)
        max_replay=max(max_replay,float(np.max(np.abs(old-base[:,[107,108,109]]))))
        if max_replay>2e-5: raise RuntimeError(f'background replay mismatch {sid}: {max_replay}')
        protos=[benign[ba],shared[sa],iproto]
        values=[evalue,svalue.astype(np.float32),ivalue]
        categories=[ba.astype(np.int16),sa.astype(np.int16),icat]
        if arm=='R3_BINDING_SHUFFLE':
            shuffled_p=[]; shuffled_v=[]; shuffled_c=[]
            for j,(p,c) in enumerate(zip(protos,categories)):
                order=deterministic_permutation(len(x),seed,sid,j); ps=p[order]; cs=c[order]
                before=np.sort(np.asarray(p,np.float64),axis=0); after=np.sort(np.asarray(ps,np.float64),axis=0)
                multiset_error=max(multiset_error,float(np.max(np.abs(before-after))))
                rawd=np.sqrt(np.maximum(np.mean(np.square(x-ps),axis=1),0.0))
                if j==0: val=np.log1p(rawd/external_scale)
                elif j==1: val=commonness[cs]*np.exp(-rawd/np.maximum(shared_scale[cs],1e-6))
                else: val=np.log1p(rawd/iscale)
                shuffled_p.append(ps); shuffled_v.append(np.asarray(val,np.float32)); shuffled_c.append(cs)
            protos,values,categories=shuffled_p,shuffled_v,shuffled_c
        elif arm=='R4_DISTANCE_CATEGORY':
            encoded=[]
            for c in categories:
                p=np.zeros((len(x),PROTO_WIDTH),np.float32); p[np.arange(len(x)),np.asarray(c,int)]=1.0; encoded.append(p)
            protos=encoded
        elif arm=='R1_SCALAR_CAPACITY':
            scalars=np.asarray(base[:,107:111],np.float32); encoded=[]
            for _ in range(BACKGROUND_COUNT):
                p=np.zeros((len(x),PROTO_WIDTH),np.float32); p[:,:4]=scalars; encoded.append(p)
            protos=encoded
        pieces=[]
        for p,v in zip(protos,values): pieces.extend([np.asarray(p,np.float32),np.asarray(v,np.float32)[:,None]])
        joined=np.concatenate([base,*pieces],axis=1).astype(np.float32)
        if joined.shape!=(len(x),MODEL_WIDTH) or not np.isfinite(joined).all(): raise RuntimeError('HPRN bag contract')
        output.append(joined); audits.append((len(x),len(np.unique(ba)),len(np.unique(sa)),len(np.unique(icat))))
    return output, {'arm':arm,'bags':len(output),'hosts':int(sum(v[0] for v in audits)),
        'max_background_replay_error':max_replay,'prototype_multiset_max_error':multiset_error,
        'mean_external_categories':float(np.mean([v[1] for v in audits])),
        'mean_shared_categories':float(np.mean([v[2] for v in audits])),
        'mean_internal_categories':float(np.mean([v[3] for v in audits])),
        'label_free_sidecar':True,'first_142_columns_unchanged':True,'width':MODEL_WIDTH}


def load_auto_routes():
    import pandas as pd
    frame=pd.read_csv(AUTO/'per_apk_predictions.csv.gz')
    frame=frame[frame.strategy=='AUTO_ERROR_LR'].copy()
    expected=2788*len(MANIFEST_SEEDS)
    if len(frame)!=expected or frame.sha256.nunique()!=2788: raise RuntimeError('autonomous route table membership')
    if frame.duplicated(['year','manifest_seed','sha256']).any(): raise RuntimeError('duplicate autonomous route rows')
    return frame


def branch_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith('hprn_')}


def frozen_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('hprn_')}


def state_max_difference(a,b):
    if set(a)!=set(b): return float('inf')
    return max((float(np.max(np.abs(a[k].detach().cpu().numpy()-b[k].detach().cpu().numpy()))) for k in a),default=0.0)
