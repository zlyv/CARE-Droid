#!/usr/bin/env python3
"""Read-only DEX identity trace for frozen v7 Host-sensitivity candidates.

This maps opaque cached Host IDs to exact DEX methods and retained endpoint
invokes. It never assigns malicious intent or treats a Host-row ablation as a
semantics-preserving program intervention.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
PREV=ROOT/'outputs/care_host_region_attribution_v7'
OUT=ROOT/'outputs/care_host_region_attribution_v8'
LINEAGE=ROOT/'artifacts/care-p31rf-haig-source-v1/lineage/future_2018_2021.csv'
ENDPOINT=ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(name,data):
    path=OUT/name
    if path.exists():raise RuntimeError('refuse overwrite '+str(path))
    data=list(data);fields=list(dict.fromkeys(k for d in data for k in d)) or ['status']
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def write_json(name,data):
    path=OUT/name
    if path.exists():raise RuntimeError('refuse overwrite '+str(path))
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def stat_asset(path):
    path=Path(path).resolve(strict=True);s=path.stat()
    return dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=digest(path))

def canonical(x):return re.sub(r'\s+','',str(x or '')).casefold()
def stable_id(x):return int.from_bytes(hashlib.sha256(canonical(x).encode()).digest()[:8],'big')
REF=re.compile(r'L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+',re.I)
def reference(text):
    hits=REF.findall(re.sub(r'\s+','',str(text)))
    return canonical(hits[-1]) if hits else ''

def main():
    if not OUT.is_dir() or (OUT/'final_decision.json').exists():raise RuntimeError('output not fresh')
    start=time.monotonic()
    previous=json.loads((PREV/'final_decision.json').read_text())
    if previous['status']!='CONDITIONAL_HOST_PATH_SENSITIVITY_AUDIT_COMPLETE' or previous['formal_integrity']!='PASS':
        raise RuntimeError('v7 audit unavailable')
    cand=rows(PREV/'error_host_candidates.csv')
    per_apk=rows(PREV/'per_apk_host_sensitivity.csv')
    error_apks={r['sha256']:r for r in per_apk if r['category'] in ('UNFIXED_FN','BROKEN_TN')}
    if len(error_apks)!=27:raise RuntimeError('2020 error cohort count changed')
    ensemble=rows(PREV/'all_host_ensemble_effects.csv')
    for sid,apk in error_apks.items():
        if any(r['sha256']==sid for r in cand):continue
        fallback=next(r for r in ensemble if r['sha256']==sid and r['host_index']==apk['strongest_any_host_index'])
        cand.append(dict(sha256=sid,category=apk['category'],pair=apk['pair'],harmful_rank=0,
            host_index=fallback['host_index'],host_id=fallback['host_id'],primary_role=fallback['primary_role'],
            attention_rank=fallback['attention_rank'],attention_mean=fallback['attention_mean'],
            harmful_delta_logit=apk['strongest_any_harmful_delta'],seed_deltas=fallback['seed_deltas'],
            seed_flips=fallback['seed_flips'],scope='UNSTABLE_FALLBACK_NO_CONSISTENT_HARMFUL_HOST'))
    if not 27<=len(cand)<=135:raise RuntimeError('unexpected candidate count')
    wanted={r['sha256'] for r in cand}
    if len(wanted)!=27:raise RuntimeError('error cohort count changed')
    lineage={r['sha256']:r for r in rows(LINEAGE) if r['sha256'] in wanted}
    if len(lineage)!=27 or any(r['metadata_year']!='2020' for r in lineage.values()):raise RuntimeError('lineage mismatch')
    paths={sid:Path(lineage[sid]['absolute_path']).resolve(strict=True) for sid in wanted}
    for sid,path in paths.items():
        if path.suffix.lower()!='.apk' or path.stem.casefold()!=sid:raise RuntimeError('APK path SHA basename mismatch')
    selected={sid:{r['host_id'] for r in cand if r['sha256']==sid} for sid in wanted}
    frozen={}
    with gzip.open(ENDPOINT,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);sid=row.get('sha256')
            if sid in wanted and row.get('status')=='OK':
                if sid in frozen:raise RuntimeError('duplicate endpoint cache')
                frozen[sid]=row['payload']
    if len(frozen)!=27:raise RuntimeError('endpoint cache missing')
    assets=[PREV/'final_decision.json',PREV/'all_host_ensemble_effects.csv',PREV/'per_apk_host_sensitivity.csv',
            PREV/'error_host_candidates.csv',LINEAGE,ENDPOINT,
            ROOT/'scripts/care-local-control-constraint-feasibility-v2/dex_adapter.py']
    before=[stat_asset(p) for p in assets]
    apk_before={sid:stat_asset(path) for sid,path in paths.items()}
    if any(apk_before[sid]['sha256']!=sid for sid in wanted):raise RuntimeError('APK content SHA mismatch')
    write_json('asset_manifest_before.json',dict(reused=before,original_apks=list(apk_before.values())))
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    method_rows=[];callsite_rows=[];failures=[]
    for number,sid in enumerate(sorted(wanted),1):
        payload=frozen[sid]
        nodeids=list(map(str,payload['node_ids']))
        edge=payload['edge_index'];base=payload['node_base_features']
        retained={}
        for src,dst in zip(edge[0],edge[1]):
            if float(base[int(dst)][4])>.5:
                retained.setdefault(nodeids[int(src)],set()).add(nodeids[int(dst)])
        if any(hid not in retained for hid in selected[sid]):
            raise RuntimeError('selected Host missing frozen sensitive edge '+sid)
        hits=defaultdict(list)
        apk=APK(str(paths[sid]))
        for dex_name in sorted(apk.get_dex_names()):
            dexbytes=apk.get_file(dex_name);dexhash=hashlib.sha256(dexbytes).hexdigest();dex=DEX(dexbytes)
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    signature=canonical(str(cls.get_name())+'->'+str(method.get_name())+str(method.get_descriptor()))
                    hid=str(stable_id(signature))
                    if hid not in selected[sid]:continue
                    hits[hid].append((method,signature,dex_name,dexhash))
        for source in (r for r in cand if r['sha256']==sid):
            hid=source['host_id'];matches=hits.get(hid,[])
            if len(matches)!=1:
                failures.append(dict(sha256=sid,host_id=hid,matches=len(matches),reason='IDENTITY_AMBIGUITY' if matches else 'IDENTITY_NOT_FOUND'))
                method_rows.append(dict(**source,identity_status='IDENTITY_AMBIGUITY' if matches else 'IDENTITY_NOT_FOUND',
                                        method_signature='',dex_name='',dex_sha256='',retained_callsite_count='',all_invoke_count=''))
                continue
            method,signature,dexname,dexhash=matches[0]
            instructions=list(method.get_instructions_idx())
            invokes=[(int(off),ins) for off,ins in instructions if str(ins.get_name()).startswith('invoke-')]
            retained_count=0
            for off,ins in invokes:
                target=reference(ins.get_output())
                if target and str(stable_id(target)) in retained[hid]:
                    retained_count+=1
                    callsite_rows.append(dict(sha256=sid,category=source['category'],host_id=hid,method_signature=signature,
                        dex_name=dexname,dex_sha256=dexhash,invoke_offset_bytes=off,invoke_opcode=str(ins.get_name()),
                        invoked_api_signature=target,endpoint_id=str(stable_id(target)),
                        trace_status='EXACT_FROZEN_HOST_ENDPOINT_HASH_EDGE'))
            method_rows.append(dict(**source,identity_status='EXACT_UNIQUE_DEX_METHOD',method_signature=signature,
                dex_name=dexname,dex_sha256=dexhash,retained_callsite_count=retained_count,all_invoke_count=len(invokes),
                instruction_count=len(instructions),has_branch=int(any(str(ins.get_name()).startswith(('if-','goto','packed-switch','sparse-switch')) for _,ins in instructions))))
        print('HRA_V8_DEX',number,'/27',sid,'matched',sum(len(v)==1 for v in hits.values()),flush=True)
    write_csv('error_host_method_trace.csv',method_rows)
    write_csv('retained_sensitive_callsites.csv',callsite_rows)
    write_csv('identity_failures.csv',failures)
    # Compare the same suppression direction in the existing 22 FN/TN pairs.
    bysha=defaultdict(list)
    for r in ensemble:bysha[r['sha256']].append(r)
    pairrows=[]
    apks=rows(PREV/'per_apk_host_sensitivity.csv')
    for pair in range(1,23):
        fn=next(r for r in apks if r['pair']==str(pair) and r['category']=='UNFIXED_FN')
        tn=next(r for r in apks if r['pair']==str(pair) and r['category']=='STABLE_TN')
        def max_suppression(sid):
            a=[r for r in bysha[sid] if r['suppressive_consistent']=='True']
            return max((float(r['delta_mean']) for r in a),default=0.0)
        pairrows.append(dict(pair=pair,fn_sha256=fn['sha256'],tn_sha256=tn['sha256'],
            fn_max_consistent_suppression=max_suppression(fn['sha256']),
            tn_max_consistent_suppression=max_suppression(tn['sha256']),
            fn_minus_tn=max_suppression(fn['sha256'])-max_suppression(tn['sha256'])))
    write_csv('matched_same_direction_suppression.csv',pairrows)
    after=[stat_asset(p) for p in assets]
    apk_after={sid:stat_asset(path) for sid,path in paths.items()}
    changes=[dict(before=b,after=a) for b,a in zip(before,after) if b!=a]
    changes += [dict(before=apk_before[sid],after=apk_after[sid]) for sid in sorted(wanted) if apk_before[sid]!=apk_after[sid]]
    write_json('baseline_integrity_check.json',dict(status='PASS' if not changes else 'FAIL',
        reused_asset_count=len(assets),original_apk_count=len(wanted),changed=changes))
    if changes:raise RuntimeError('reused input changed')
    cats=Counter(r['category'] for r in method_rows)
    flips=Counter()
    for r in cand:
        if int(r['seed_flips'])==3:flips[r['category']]+=1
    decision=dict(status='HOST_REGIONS_MAPPED_MODEL_SENSITIVITY_ONLY',
        future_labels_used_for_training=False,source_or_future_model_changed=False,
        method_candidates=len(cand),unique_error_apks=len(wanted),identity_exact_unique=sum(r['identity_status']=='EXACT_UNIQUE_DEX_METHOD' for r in method_rows),
        identity_ambiguities=len(failures),retained_sensitive_callsites=len(callsite_rows),
        candidates_by_category=dict(cats),candidates_flipping_all_three_seeds=dict(flips),
        matched_pairs_fn_suppression_gt_tn=sum(float(r['fn_minus_tn'])>0 for r in pairrows),
        matched_pairs=22,formal_integrity='PASS',seconds=time.monotonic()-start,
        root_cause_verified=False,
        interpretation='Exact DEX method/callsite identity plus frozen model sensitivity, not a causal proof that code behavior caused a misclassification. Frozen RoleGraph/Global24 were held fixed.')
    write_json('final_decision.json',decision)
    print('HRA_V8_DONE',json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('no Python bytecode writes allowed')
    main()
