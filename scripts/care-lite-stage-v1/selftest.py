#!/usr/bin/env python3
"""LiteStage-V1 smoke tests. Reads one source APK shard, frozen P31/C43, mapping and source contract; writes only new smoke JSON; fits tiny RF; no original data writes."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
from sklearn.ensemble import RandomForestClassifier
from common import *
from extract import match,api_roles

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lineage,pred=source_contract();expert=frozen_expert_source();ids=sorted(lineage)
    if set(expert)!=set(ids):raise RuntimeError('C43 SHA mismatch')
    p=OUT/'shards/source'/(ids[0]+'.json')
    if not p.is_file():raise RuntimeError('smoke shard missing; run extract.py --cohort source --limit 1')
    v=json.loads(p.read_text(encoding='utf-8'))
    if len(v['manifest'])!=7 or len(v['dex_meta'])!=5 or len(v['role_ref_count'])!=31 or len(v['role_prevalence'])!=31:raise RuntimeError('feature width')
    if v['callgraph_constructed'] or v['host_graph_constructed']:raise RuntimeError('hidden graph construction')
    with np.load(P31,allow_pickle=False) as z:
        official={str(s).lower():i for i,s in enumerate(z['sha256'])};i=official[ids[0]];ref=z['role_prevalence'][i].astype(float);diff=float(np.max(np.abs(np.asarray(v['role_prevalence'])-ref)))
    if diff>1e-5:raise RuntimeError('source Role31 invoke/prevalence mismatch '+str(diff))
    with MAPPING.open(encoding='utf-8') as f:m=yaml.safe_load(f)
    if len(m['roles'])!=31:raise RuntimeError('mapping width')
    from androguard.core.apk import APK
    from loguru import logger
    logger.remove()
    if not Path(lineage[ids[0]]['absolute_path']).is_file():raise RuntimeError('APK path')
    # Tiny RF and fold-only role selection interfaces.
    tiny=[v,dict(v,role_method_count=[0]*31,role_ref_count=[0]*31),v,dict(v,role_method_count=[0]*31,role_ref_count=[0]*31)]
    y=np.array([1,0,1,0]);x=matrix(tiny,'L4_COUNT');roles=choose_roles(tiny,y,'L4_TOP8')
    if x.shape[1]!=128+7+5+31 or len(roles)!=8:raise RuntimeError('model input')
    rf=RandomForestClassifier(n_estimators=5,min_samples_leaf=1,random_state=1).fit(x,y)
    if rf.predict_proba(x).shape!=(4,2):raise RuntimeError('RF predict')
    if route_by_score(['a','b','c','d'],[.1,.4,.3,.2],.25).sum()!=1:raise RuntimeError('route budget')
    base=np.array([.2,.8,.3,.7]);c43=np.array([.9,.1,.6,.4]);mm,final=cascade(y,base,c43,[1,1,0,0]);
    if mm['stage1_errors']!=4 or mm['fixed']!=2 or mm['broken']!=0:raise RuntimeError('fixed/broken')
    out=dict(status='PASS',source=799,folds=[160,160,160,160,159],official_p31_max_abs_diff=diff,smoke_apk=ids[0],feature_width=x.shape[1],manifest_and_dex_smoke=True,call_graph_used=False,C43_OOF_aligned=True)
    p=OUT/'smoke_test.json'
    if p.exists():
        previous=json.loads(p.read_text(encoding='utf-8'))
        if previous!=out:raise RuntimeError('existing smoke report differs; refuse overwrite')
    else:
        p.write_text(json.dumps(out,indent=2)+'\n')
    print('LITESTAGE_SELFTEST_PASS',out,flush=True)
if __name__=='__main__':main()
