#!/usr/bin/env python3
"""Shared contracts for the static Role-Host aligned-fusion experiment; reads frozen caches only."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT / 'scripts/care-role-host-aligned-fusion-v2'
OUT = ROOT / 'outputs/care_role_host_aligned_fusion_v2'
BFA = ROOT / 'scripts/care-background-functional-attribution-v1'
SCBE = ROOT / 'outputs/care_scbe_v1'
MYD = ROOT / 'scripts/care-multiyear-development-v1'
HBI = ROOT / 'outputs/care-host-control-binding-information-source-v2'
sys.path[:0] = [str(HERE), str(BFA), str(ROOT/'scripts/care-scbe-v1'), str(MYD)]

SEEDS = (900000, 900001, 900002)
SPLITS = ('S1','S2','S3','S4','S5','T1','T2','T3','T4')
ARMS = ('F0_FROZEN_DIRECT_C43','F1_ALIGNED_BIPARTITE',
        'F2_WITHIN_APK_BINDING_SHUFFLE','F3_APK_ROLE_MARGINAL_CONTROL')


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''): h.update(chunk)
    return {'path':str(path.resolve()),'size':path.stat().st_size,'sha256':h.hexdigest()}


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    def encode(item):
        if isinstance(item,np.generic): return item.item()
        if isinstance(item,Path): return str(item)
        raise TypeError(type(item).__name__)
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=encode)+'\n',encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, values):
    values=list(values); fields=list(dict.fromkeys(k for row in values for k in row)) or ['status']
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(values)
    temp.replace(path)


def write_pickle(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    with temp.open('wb') as handle: pickle.dump(value,handle,protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def asset_paths():
    old=ROOT/'outputs/care_multiyear_development_v1'
    paths=[HERE/'common.py',HERE/'rhaf_common.py',HERE/'model.py',HERE/'train.py',HERE/'evaluate.py',
           HERE/'selftest.py',HERE/'run.sh',
           BFA/'bfa_core.py',ROOT/'scripts/care-scbe-v1/train.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',
           ROOT/'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py',
           ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py',
           ROOT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
           MYD/'metrics.py',old/'data.npz',old/'feature_rows.pkl',old/'sample_manifest.csv',
           HBI/'final_decision.json',HBI/'REPORT.md']
    paths += [old/'predictions'/split/'router.npz' for split in SPLITS[5:]]
    paths += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl' for split in SPLITS for seed in SEEDS]
    return paths


def verify_assets():
    manifest=OUT/'asset_manifest_before.json'
    if not manifest.is_file(): return {'status':'MISSING'}
    prior=json.loads(manifest.read_text(encoding='utf-8')); changed=[]
    for item in prior:
        try: now=digest(item['path'])
        except OSError as exc: changed.append({'path':item['path'],'error':repr(exc)}); continue
        if now!=item: changed.append({'before':item,'after':now})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(prior),'changed':changed,
            'scope':'named experiment code, frozen features/routes/checkpoints and prior HBI result; no APK bytes'}


def preflight(resume=False):
    if OUT.exists() and not resume: raise RuntimeError('output exists; use --resume, never overwrite')
    if not OUT.exists():
        OUT.mkdir(); (OUT/'models').mkdir(); (OUT/'logs').mkdir(); (OUT/'tmp').mkdir()
        prior=json.loads((HBI/'final_decision.json').read_text(encoding='utf-8'))
        if prior.get('status')!='HOST_BINDING_INFORMATION_NOT_SUPPORTED':
            raise RuntimeError('prior Host-binding boundary changed')
        atomic_json(OUT/'asset_manifest_before.json',[digest(p) for p in asset_paths()])
        atomic_json(OUT/'protocol.json',{
            'status':'FROZEN','method':'static exact Host-own-Role31 node-level bidirectional residual fusion',
            'arms':ARMS,'development_years':[2017,2018,2019,2020,2021],
            'future_years_are_development':True,'dynamic_analysis':False,'new_APKs':False,
            'Stage1_changed':False,'Router_changed':False,'backgrounds_changed':False,
            'loss':'equal BCE','threshold':0.5,'contrastive_loss':False,
            'v1_failure':'float32 reduction-order false positive in F3 marginal audit at T3/900000 validation',
            'v2_numerical_fix':'float64 mean reference and float64 marginal audit; model input remains float32',
            'role_marginal_tolerance':1e-5,
            'boundary':'last distinct end-to-end exact-binding test after the compact HBI test failed; no tuning if controls win'})
    result=verify_assets()
    if result['status']!='PASS': raise RuntimeError('asset integrity failed')
    atomic_json(OUT/'preflight.json',{**result,'APKs':2788,'routed_APKs':697,'APK_files_opened':0})
    return result


def role_binding(raw_bag):
    raw=np.asarray(raw_bag,np.float32)
    if raw.ndim!=2 or raw.shape[1]!=169: raise RuntimeError(f'raw HAIG bag must be N x 169, got {raw.shape}')
    weight=np.maximum(raw[:,43:74],0.0); denominator=weight.sum(1,keepdims=True)
    binding=np.divide(weight,denominator,out=np.zeros_like(weight),where=denominator>0)
    if binding.shape!=(len(raw),31) or not np.isfinite(binding).all(): raise RuntimeError('invalid Host-own-Role31 binding')
    return binding.astype(np.float32)


def deterministic_permutation(n,seed,sid):
    token=hashlib.sha256(f'{seed}:{sid}:role-host-binding'.encode()).digest()
    return np.random.RandomState(int.from_bytes(token[:4],'big')).permutation(n)


def augment_bags(bags,raw_bags,ids,arm,seed):
    if len(bags)!=len(raw_bags) or len(bags)!=len(ids): raise RuntimeError('bag alignment length mismatch')
    output=[]; changed=0; nonzero=0; marginal_error=0.0
    for bag,raw,sid in zip(bags,raw_bags,map(str,ids)):
        bag=np.asarray(bag,np.float32); binding=role_binding(raw)
        if bag.ndim!=2 or bag.shape[1]!=142 or len(bag)!=len(binding): raise RuntimeError('transformed/raw Host order mismatch')
        original=binding.copy()
        target_marginal=original.sum(axis=0,dtype=np.float64)
        if arm=='F1_ALIGNED_BIPARTITE': pass
        elif arm=='F2_WITHIN_APK_BINDING_SHUFFLE':
            binding=binding[deterministic_permutation(len(binding),seed,sid)]
        elif arm=='F3_APK_ROLE_MARGINAL_CONTROL':
            # Construct the class-blind APK-level marginal control with a
            # float64 reference, then return the float32 tensor expected by
            # the frozen training pipeline.
            mean_binding=original.astype(np.float64).mean(0,keepdims=True)
            binding=np.repeat(mean_binding,len(binding),axis=0).astype(np.float32)
        else: raise ValueError(arm)
        observed_marginal=binding.sum(axis=0,dtype=np.float64)
        marginal_error=max(marginal_error,float(np.max(np.abs(target_marginal-observed_marginal))))
        changed+=int(not np.array_equal(original,binding)); nonzero+=int(np.any(original>0))
        joined=np.concatenate([bag,binding],1)
        if joined.shape!=(len(bag),173) or not np.isfinite(joined).all(): raise RuntimeError('augmented Host bag contract')
        output.append(joined.astype(np.float32))
    # The invariant is evaluated in float64 so its result does not depend
    # on float32 reduction order or on the number of Hosts in an APK.
    marginal_tolerance=1e-5
    if marginal_error>marginal_tolerance:
        raise RuntimeError(f'APK Role marginals changed beyond float64-audited tolerance: {marginal_error}')
    return output,{'arm':arm,'bags':len(output),'binding_nonzero_bags':nonzero,'changed_bags':changed,
                   'role_marginal_max_abs_error':marginal_error,
                   'role_marginal_tolerance':marginal_tolerance,
                   'role_marginal_audit_dtype':'float64','width':173,'label_free':True}
