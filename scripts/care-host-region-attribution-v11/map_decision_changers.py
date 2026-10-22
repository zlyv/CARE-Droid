#!/usr/bin/env python3
"""Read-only DEX trace for exact ensemble decision-changing Host ablations."""
from __future__ import annotations
import csv,gzip,hashlib,json,os,re,time
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=ROOT/'outputs/care_host_region_attribution_v11'
V7=ROOT/'outputs/care_host_region_attribution_v7'
V10=ROOT/'outputs/care_host_region_attribution_v10'
LINEAGE=ROOT/'artifacts/care-p31rf-haig-source-v1/lineage/future_2018_2021.csv'
ENDPOINT=ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def info(path):
    path=path.resolve(strict=True);s=path.stat();return dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=digest(path))
def put(name,obj):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def csvout(name,rr):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    fields=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def canonical(s):return re.sub(r'\s+','',str(s or '')).casefold()
def hid(s):return str(int.from_bytes(hashlib.sha256(canonical(s).encode()).digest()[:8],'big'))
REF=re.compile(r'L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+',re.I)
def ref(output):
    hits=REF.findall(re.sub(r'\s+','',str(output)));return canonical(hits[-1]) if hits else ''
def main():
    if not OUT.is_dir() or (OUT/'final_decision.json').exists():raise RuntimeError('output not fresh')
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    try:
        from loguru import logger
        logger.remove()
    except ImportError:pass
    t=time.monotonic()
    candidates=read(V10/'ensemble_corrective_host_candidates.csv')+read(V10/'ensemble_damaging_host_candidates.csv')
    if len(candidates)!=4:raise RuntimeError('decision-changing candidate count changed')
    wanted={r['sha256'] for r in candidates}
    lineage={r['sha256']:r for r in read(LINEAGE) if r['sha256'] in wanted}
    if len(lineage)!=2 or any(r['metadata_year']!='2020' for r in lineage.values()):raise RuntimeError('lineage mismatch')
    apkpaths={sid:Path(lineage[sid]['absolute_path']).resolve(strict=True) for sid in wanted}
    assets=[V10/'ensemble_corrective_host_candidates.csv',V10/'ensemble_damaging_host_candidates.csv',LINEAGE,ENDPOINT,*apkpaths.values()]
    before=[info(p) for p in assets]
    if any(info(apkpaths[sid])['sha256']!=sid for sid in wanted):raise RuntimeError('APK hash mismatch')
    frozen={}
    with gzip.open(ENDPOINT,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row.get('sha256') in wanted and row.get('status')=='OK':frozen[row['sha256']]=row['payload']
    if len(frozen)!=2:raise RuntimeError('endpoint missing')
    result=[];calls=[]
    for r in candidates:
        sid=r['sha256'];payload=frozen[sid];nodeids=list(map(str,payload['node_ids']))
        endpoint_ids={nodeids[int(v)] for u,v in zip(*payload['edge_index']) if nodeids[int(u)]==r['host_id'] and float(payload['node_base_features'][int(v)][4])>.5}
        if not endpoint_ids:raise RuntimeError('candidate Host lacks retained endpoint')
        matching=[];apk=APK(str(apkpaths[sid]))
        for dexname in sorted(apk.get_dex_names()):
            raw=apk.get_file(dexname);dexhash=hashlib.sha256(raw).hexdigest();dex=DEX(raw)
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    sig=canonical(str(cls.get_name())+'->'+str(method.get_name())+str(method.get_descriptor()))
                    if hid(sig)==r['host_id']:matching.append((method,sig,dexname,dexhash))
        if len(matching)!=1:raise RuntimeError('Host identity not unique '+sid)
        method,sig,dexname,dexhash=matching[0];count=0
        for off,ins in method.get_instructions_idx():
            if not str(ins.get_name()).startswith('invoke-'):continue
            target=ref(ins.get_output())
            if target and hid(target) in endpoint_ids:
                count+=1;calls.append(dict(sha256=sid,category=r['category'],host_index=r['host_index'],
                    host_id=r['host_id'],method_signature=sig,dex_name=dexname,dex_sha256=dexhash,
                    invoke_offset_bytes=int(off),opcode=str(ins.get_name()),sensitive_target=target,
                    endpoint_id=hid(target),trace_status='EXACT_FROZEN_HOST_ENDPOINT_HASH_EDGE'))
        result.append(dict(sha256=sid,category=r['category'],host_index=r['host_index'],host_id=r['host_id'],
            decision_effect='CORRECTIVE' if r['corrective']=='1' else 'DAMAGING',
            ensemble_before=r['ensemble_before'],ensemble_after=r['ensemble_after'],
            delta_probability=r['delta_probability'],three_seed_direction_consistent=r['three_seed_logit_direction_consistent'],
            individual_seed_flip_count=r['individual_seed_flip_count'],
            method_signature=sig,dex_name=dexname,dex_sha256=dexhash,
            retained_sensitive_callsites=count,identity_status='EXACT_UNIQUE_DEX_METHOD'))
    csvout('decision_changing_host_methods.csv',result);csvout('decision_changing_callsites.csv',calls)
    after=[info(p) for p in assets];changed=[(a,b) for a,b in zip(before,after) if a!=b]
    put('baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(assets),changed=changed))
    if changed:raise RuntimeError('input changed')
    decision=dict(status='ENSEMBLE_DECISION_CHANGING_HOSTS_MAPPED',hosts=len(result),callsites=len(calls),
        formal_integrity='PASS',seconds=time.monotonic()-t,
        interpretation='Four conditional ensemble decision-changing Host rows have exact DEX identity; none is a three-seed robust or semantically coherent full-model intervention.')
    put('final_decision.json',decision);print('HRA_V11_DONE',json.dumps(decision),flush=True)
if __name__=='__main__':
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('no bytecode writes allowed')
    main()
