#!/usr/bin/env python3
"""Exact frozen three-seed ensemble decision effects of all v7 Host ablations."""
from __future__ import annotations
import csv,json,os,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
V7=ROOT/'outputs/care_host_region_attribution_v7'
BASE=ROOT/'outputs/care_multiyear_development_v1'
OUT=ROOT/'outputs/care_host_region_attribution_v10'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,data):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    data=list(data);fields=list(dict.fromkeys(k for r in data for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def dump(name,obj):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    if not OUT.is_dir() or (OUT/'final_decision.json').exists():raise RuntimeError('output not fresh')
    start=time.monotonic()
    decision=json.loads((V7/'final_decision.json').read_text())
    if decision['status']!='CONDITIONAL_HOST_PATH_SENSITIVITY_AUDIT_COMPLETE':raise RuntimeError('v7 incomplete')
    original={r['sha256']:r for r in read(BASE/'per_apk_predictions.csv') if r['window']=='T3' and r['arm']=='U3_FULL_ROLLING_C1'}
    groups=defaultdict(list)
    for r in read(V7/'all_host_effects.csv'):
        groups[(r['sha256'],int(r['host_index']))].append(r)
    if len(groups)!=20418:raise RuntimeError('Host count mismatch')
    out=[];baseline_by_sha={}
    for (sid,index),rows in groups.items():
        if sorted(int(r['seed']) for r in rows)!=[900000,900001,900002]:raise RuntimeError('seed completeness')
        before=sum(float(r['probability_before']) for r in rows)/3
        after=sum(float(r['probability_after']) for r in rows)/3
        frozen=float(original[sid]['expert_probability'])
        if abs(before-frozen)>1e-6:raise RuntimeError('ensemble replay mismatch '+sid)
        if sid in baseline_by_sha and abs(before-baseline_by_sha[sid])>1e-10:raise RuntimeError('baseline drift '+sid)
        baseline_by_sha[sid]=before
        category=rows[0]['category'];true=int(rows[0]['label'])
        if (before>=.5)!=bool(int(original[sid]['final_prediction'])):raise RuntimeError('final prediction mismatch '+sid)
        after_pred=int(after>=.5)
        corrective=bool(category in ('UNFIXED_FN','BROKEN_TN') and after_pred==true)
        damaging=bool(category in ('FIXED_FN','STABLE_TN') and after_pred!=true)
        signs=[float(r['delta_logit']) for r in rows]
        out.append(dict(sha256=sid,category=category,host_index=index,host_id=rows[0]['host_id'],
            primary_role=rows[0]['primary_role'],host_count=rows[0]['host_count'],
            ensemble_before=before,ensemble_after=after,delta_probability=after-before,
            ensemble_prediction_after=after_pred,corrective=int(corrective),damaging=int(damaging),
            three_seed_logit_direction_consistent=int(all(s>0 for s in signs) or all(s<0 for s in signs)),
            individual_seed_flip_count=sum(int(r['flipped']) for r in rows),
            seed_probability_after=json.dumps([float(r['probability_after']) for r in rows]),
            seed_probability_before=json.dumps([float(r['probability_before']) for r in rows])))
    write('all_host_ensemble_decision_effects.csv',out)
    corrections=[r for r in out if r['corrective']]
    damages=[r for r in out if r['damaging']]
    corrections.sort(key=lambda r:(r['category'],r['sha256'],-abs(r['delta_probability'])))
    write('ensemble_corrective_host_candidates.csv',corrections)
    write('ensemble_damaging_host_candidates.csv',damages)
    apk=[]
    for sid in sorted(baseline_by_sha):
        rr=[r for r in out if r['sha256']==sid];category=rr[0]['category']
        cc=[r for r in rr if r['corrective']];dd=[r for r in rr if r['damaging']]
        best=max(cc,key=lambda r:abs(r['delta_probability'])) if cc else None
        apk.append(dict(sha256=sid,category=category,hosts=len(rr),ensemble_before=rr[0]['ensemble_before'],
            corrective_host_count=len(cc),damaging_host_count=len(dd),
            strongest_corrective_host_index='' if best is None else best['host_index'],
            strongest_corrective_host_id='' if best is None else best['host_id'],
            strongest_corrective_delta_probability=0 if best is None else best['delta_probability'],
            strongest_corrective_consistent_direction=0 if best is None else best['three_seed_logit_direction_consistent']))
    write('per_apk_ensemble_decision_effects.csv',apk)
    result=dict(status='ENSEMBLE_HOST_CONDITIONAL_DECISION_AUDIT_COMPLETE',apks=len(apk),hosts=len(out),
        category_counts=dict(Counter(r['category'] for r in apk)),
        error_apks_with_any_corrective_host=dict(Counter(r['category'] for r in apk if r['corrective_host_count']>0)),
        corrective_host_count_by_category=dict(Counter(r['category'] for r in corrections)),
        correct_apks_with_any_damaging_host=dict(Counter(r['category'] for r in apk if r['damaging_host_count']>0)),
        damaging_host_count_by_category=dict(Counter(r['category'] for r in damages)),
        corrective_hosts_three_seed_direction_consistent=sum(r['three_seed_logit_direction_consistent'] for r in corrections),
        future_labels_used_for_training=False,model_changed=False,threshold_changed=False,
        interpretation='One-row Host-path ablation with frozen RoleGraph/Global24: ensemble model sensitivity only. A correction under artificial input removal is not a proof that the DEX method causes the real misclassification.',
        seconds=time.monotonic()-start)
    dump('final_decision.json',result)
    print('HRA_V10_DONE',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('no bytecode writes allowed')
    main()
