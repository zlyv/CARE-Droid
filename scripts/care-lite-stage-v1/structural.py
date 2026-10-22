#!/usr/bin/env python3
"""LiteStage-V1 frozen source structural/collision diagnostics. Reads existing Direct-C43 factorial OOF and new LiteStage predictions/shards; writes new diagnostic CSV only; no APK or training."""
from __future__ import annotations
import hashlib
import numpy as np
from common import *

RPF=ROOT/'outputs/care-direct-c43-rolepath-factorial-v1/source'
FACTOR=['R00_ROLE_NEITHER','R01_ROLE_READOUT_ONLY','R10_ROLE_QUERY_ONLY','R11_ROLE_QUERY_READOUT']
def expert_col(row):
    cols=[c for c in row if c.endswith('__expert_prob')]
    if len(cols)!=1:raise RuntimeError('factorial expert column '+repr(cols))
    return cols[0]
def main():
    source=readcsv(OUT/'two_stage_source2017_predictions.csv');chosen={r['sha256']:r for r in source if r['method']=='T7_LITE_C43'}
    if len(chosen)!=799:raise RuntimeError('T7 identity')
    ids=sorted(chosen);y=np.array([int(chosen[s]['label']) for s in ids]);base=np.array([float(chosen[s]['stage1_probability']) for s in ids]);route=np.array([int(chosen[s]['routed']) for s in ids],bool)
    rows=[]
    for arm in FACTOR:
        p=RPF/arm/'source_predictions.csv';raw=readcsv(p);indexed={r['sha256']:r for r in raw}
        if set(indexed)!=set(ids):raise RuntimeError('factorial SHA mismatch '+arm)
        col=expert_col(raw[0]);expert=[]
        for sid in ids:
            r=indexed[sid]
            if int(r['fold'])!=int(chosen[sid]['fold']) or int(r['label'])!=int(chosen[sid]['label']):raise RuntimeError('factorial fold/label mismatch '+sid)
            expert.append(float(r[col]))
        m,_=cascade(y,base,np.array(expert),route)
        rows.append(dict(arm=arm,scope='supplementary within RPF factorial branch, not capacity-matched S1/S2/S4 comparison',**m))
    rows.append(dict(arm='S1_ROLE_ONLY',scope='NOT_RUN: available A0 role-only checkpoint uses bounded/other baseline; incompatible',status='UNVERIFIED'))
    rows.append(dict(arm='S2_ROLE_HOST',scope='NOT_RUN: no matched route-independent direct checkpoint',status='UNVERIFIED'))
    rows.append(dict(arm='S4_FULL_HDBC_M0',scope='see T7 source metrics; different checkpoint family from RPF factorial',status='REFERENCE_ONLY'))
    writecsv('stage2_structural_ablation.csv',rows)
    shards,missing=load_shards('source')
    if missing:raise RuntimeError('source shards missing')
    buckets={}
    for sid in ids:
        role=(np.asarray(shards[sid]['role_method_count'])>0).astype(int)
        key=''.join(map(str,role));buckets.setdefault(key,[]).append(sid)
    cases=[]
    for key,group in buckets.items():
        if len(group)<2:continue
        labels={int(chosen[s]['label']) for s in group}
        if labels!={0,1}:continue
        cases.append(dict(role_presence=key,n=len(group),benign=sum(int(chosen[s]['label'])==0 for s in group),malware=sum(int(chosen[s]['label'])==1 for s in group),stage1_errors=sum((float(chosen[s]['stage1_probability'])>=.5)!=int(chosen[s]['label']) for s in group),fixed=sum(int(chosen[s]['fixed']) for s in group),broken=sum(int(chosen[s]['broken']) for s in group),sha_examples=';'.join(sorted(group)[:4]),interpretation='identical observed L4 Role presence; Host/graph semantic distinction not proven'))
    writecsv('role_collision_analysis.csv',cases)
    print('LITESTAGE_STRUCTURAL_DIAGNOSTIC_DONE collisions',len(cases),flush=True)
if __name__=='__main__':main()
