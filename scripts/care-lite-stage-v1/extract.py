#!/usr/bin/env python3
"""LiteStage-V1 read-only APK extraction. Inputs: frozen source/future lineages and Role31 YAML. Writes only outputs/care_lite_stage_v1/shards; no model training or call graph."""
from __future__ import annotations
import argparse,csv,gc,json,os,re,sys,time,zipfile
from pathlib import Path
import numpy as np,yaml
from loguru import logger
logger.remove()
from androguard.core.apk import APK
from androguard.core.dex import DEX

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=ROOT/'outputs/care_lite_stage_v1'
SOURCE=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv')
FUTURE=ROOT/'artifacts/care-litelocal-p31-diagnostics-v1/preflight/future_lineage_label_free.csv'
MAPPING=ROOT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml'
TARGET=re.compile(r'(L[^;]+;)->([^\s(]+)\(')

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def canon(v):return str(v or '').strip().casefold().replace(' ','')
def match(rule,cls,name):
    if rule.get('rule_kind','api')!='api':return False
    cls,name=canon(cls),canon(name)
    pre=[canon(x) for x in rule.get('class_prefixes',[])];con=[canon(x) for x in rule.get('class_contains',[])];names=[canon(x) for x in rule.get('method_names',[])];mp=[canon(x) for x in rule.get('method_prefixes',[])]
    return bool(pre or con) and bool(names or mp) and (any(cls.startswith(x) for x in pre) or any(x in cls for x in con)) and (name in names or any(name.startswith(x) for x in mp))
def api_roles(cls,name,rules,role_index):return [role_index[r['role']] for r in rules if match(r,cls,name)]
def safe(fn,default=None):
    try:return fn()
    except Exception:return default

def extract(row,rules,role_index,level='L4'):
    path=Path(row['absolute_path']);sid=canon(row['sha256'])
    if not path.is_file():raise FileNotFoundError(str(path))
    t0=time.perf_counter();apk=APK(str(path));permissions=sorted(set(map(str,apk.get_permissions() or [])))
    manifest=[len(permissions),len(apk.get_activities()),len(apk.get_services()),len(apk.get_receivers()),len(apk.get_providers()),int(apk.get_min_sdk_version() or 0),int(apk.get_target_sdk_version() or 0)]
    t1=time.perf_counter();ref=np.zeros(31,np.int64);role_methods=np.zeros(31,np.int64);invokes=np.zeros(31,np.int64)
    dex_count=class_count=method_ids=defined_methods=invoke_n=sensitive_methods=0
    if level=='L1':
        return dict(sha256=sid,source_path=str(path),apk_bytes=path.stat().st_size,permissions=permissions,manifest=manifest,dex_meta=[0]*5,
          role_ref_count=ref.tolist(),role_invoke_count=invokes.tolist(),role_method_count=role_methods.tolist(),role_prevalence=[0.]*31,
          sensitive_method_count=0,timing=dict(manifest_s=t1-t0,dex_metadata_s=0.,reference_s=0.,invoke_s=0.,total_s=t1-t0),
          extracted_level=level,cost_classes=['MANIFEST_ONLY'],callgraph_constructed=False,host_graph_constructed=False)
    # One ZipFile read for DEX blobs; APK() above only parses manifest. DEX() parses
    # sections but never constructs Analysis, call graph, Host or RoleGraph.
    with zipfile.ZipFile(path) as z:
        dex_names=sorted(n for n in z.namelist() if re.fullmatch(r'classes(?:\d+)?\.dex',Path(n).name) and '/' not in n)
        dex_count=len(dex_names)
        for dex_name in dex_names:
            vm=DEX(z.read(dex_name));classes=list(vm.get_classes());methods=list(vm.get_methods())
            class_count+=len(classes);method_ids+=len(methods)
            t2=time.perf_counter()
            if level in ('L3','L4'):
                for m in methods:
                    for rid in api_roles(safe(m.get_class_name,''),safe(m.get_name,''),rules,role_index):ref[rid]+=1
            t3=time.perf_counter()
            for cls in classes:
                for method in cls.get_methods():
                    defined_methods+=1
                    if level!='L4' or safe(method.get_code) is None:continue
                    found=set();getter=getattr(method,'get_instructions_idx',None)
                    iterator=getter() if callable(getter) else ((0,x) for x in method.get_instructions())
                    for _,ins in iterator:
                        opname=str(safe(ins.get_name,'')).casefold()
                        if not opname.startswith('invoke-'):continue
                        invoke_n+=1;hits=TARGET.findall(str(safe(ins.get_output,'')))
                        if not hits:continue
                        clsname,name=hits[-1]
                        for rid in api_roles(clsname,name,rules,role_index):invokes[rid]+=1;found.add(rid)
                    if found:
                        sensitive_methods+=1
                        for rid in found:role_methods[rid]+=1
            t4=time.perf_counter()
            # Stage timings are incremental for this shared-parse implementation.
            # DEX parse time precedes reference iteration; invocation time is body iteration.
            if 'meta_s' not in locals():meta_s=ref_s=invoke_s=0.
            meta_s+=t2-(t1 if dex_name==dex_names[0] else previous_t4)
            ref_s+=t3-t2;invoke_s+=t4-t3;previous_t4=t4
            del vm,classes,methods;gc.collect()
    t5=time.perf_counter();prevalence=(role_methods/max(sensitive_methods,1)).astype(float)
    return dict(sha256=sid,source_path=str(path),apk_bytes=path.stat().st_size,permissions=permissions,manifest=manifest,
        dex_meta=[dex_count,class_count,method_ids,defined_methods,invoke_n],role_ref_count=ref.tolist(),
        role_invoke_count=invokes.tolist(),role_method_count=role_methods.tolist(),role_prevalence=prevalence.tolist(),
        sensitive_method_count=sensitive_methods,timing=dict(manifest_s=t1-t0,dex_metadata_s=meta_s if dex_count else 0.,reference_s=ref_s if dex_count else 0.,invoke_s=invoke_s if dex_count else 0.,total_s=t5-t0),
        extracted_level=level,cost_classes=['MANIFEST_ONLY']+(['DEX_METADATA'] if level in ('L2','L3','L4') else [])+(['DEX_REFERENCE_SCAN'] if level in ('L3','L4') else [])+(['DEX_INVOKE_SCAN'] if level=='L4' else []),callgraph_constructed=False,host_graph_constructed=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['source','future'],required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--level',choices=['L1','L2','L3','L4'],default='L4');args=ap.parse_args()
    with MAPPING.open(encoding='utf-8') as f:mapping=yaml.safe_load(f)
    roles=list(mapping['roles']);role_index={r:i for i,r in enumerate(roles)};rules=[r for r in mapping['rules'] if r.get('rule_kind','api')=='api']
    if len(roles)!=31:raise RuntimeError('Role31 mapping changed')
    records=rows(SOURCE if args.cohort=='source' else FUTURE)
    expect=799 if args.cohort=='source' else 1989
    if len(records)!=expect or len({canon(r['sha256']) for r in records})!=expect:raise RuntimeError('lineage count/SHA')
    if args.limit:records=records[:args.limit]
    dest=OUT/'shards'/args.cohort;dest.mkdir(parents=True,exist_ok=True)
    failures=[]
    for n,row in enumerate(records,1):
        sid=canon(row['sha256']);p=dest/(sid+'.json')
        if p.exists():
            v=json.loads(p.read_text(encoding='utf-8'))
            if v.get('sha256')!=sid or v.get('callgraph_constructed') is not False:raise RuntimeError('shard integrity '+sid)
            levels={'L1':1,'L2':2,'L3':3,'L4':4}
            if levels.get(v.get('extracted_level','L4'),0)<levels[args.level]:raise RuntimeError('existing shard has insufficient level '+sid)
            continue
        try:
            v=extract(row,rules,role_index,args.level)
            with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'))
        except Exception as exc:
            failures.append(dict(sha256=sid,error=type(exc).__name__,detail=str(exc)))
        if n%10==0 or n==len(records):print('LITESTAGE_EXTRACT',args.cohort,n,'/',len(records),'failed',len(failures),flush=True)
    if failures:
        attempt=1
        while (OUT/f'{args.cohort}_failures_attempt{attempt}.json').exists():attempt+=1
        target=OUT/f'{args.cohort}_failures_attempt{attempt}.json'
        with target.open('x',encoding='utf-8') as f:json.dump(failures,f,indent=2)
        print('LITESTAGE_EXTRACTION_FAILURES',len(failures),flush=True)
    print('LITESTAGE_EXTRACTION_COMPLETE',args.cohort,'available',sum((dest/(canon(r['sha256'])+'.json')).exists() for r in records),flush=True)
if __name__=='__main__':main()
