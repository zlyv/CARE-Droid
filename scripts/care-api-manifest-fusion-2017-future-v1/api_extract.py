#!/usr/bin/env python3
"""Recover exact sensitive API signatures for frozen CARE Hosts without rebuilding graphs.

Inputs: verified APK paths and per-APK frozen Host→Endpoint stable-ID views. Outputs:
resumable per-APK Host/API event caches under the new artifact. APKs are read-only.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import OUT, atomic_json, atomic_json_gz, canonical, lineage, read_json_gz, sha256_file, stable_id, status


def endpoint_family(reference,role_ids):
    cls=canonical(reference).split('->',1)[0] if '->' in canonical(reference) else 'unknown'
    role_key='+'.join(map(str,sorted(set(map(int,role_ids))))) or 'UNKNOWN'
    return f'roles={role_key}|class={cls}',role_key


def scan_one(sid,year,apk_path,view_path,output_path):
    if sha256_file(apk_path)!=sid:raise RuntimeError('APK SHA256 mismatch '+sid)
    view=read_json_gz(view_path);host_set=set(view['host_ids'])
    if not host_set:
        atomic_json_gz(output_path,{'sha256':sid,'year':year,'status':'OK_NO_HOST','hosts':[],'apk_sha256_verified':True});return
    sys.path.insert(0,str(Path('/home/boyuanjie/projects/tasa-droid/scripts/care-local-control-constraint-feasibility-v2')))
    from dex_adapter import decode_method
    try:
        from loguru import logger;logger.remove()
    except Exception:pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    apk=APK(str(apk_path));methods=defaultdict(list);dex_count=0
    for dex_name in sorted(apk.get_dex_names()):
        dex_count+=1;dex=DEX(apk.get_file(dex_name))
        for cls in dex.get_classes():
            cn=str(cls.get_name())
            for method in cls.get_methods():
                sig=canonical(cn+'->'+str(method.get_name())+str(method.get_descriptor()));hid=str(stable_id(sig))
                if hid in host_set:methods[hid].append((method,dex_name,sig))
    hosts=[];recovered=expected=0;ambiguous=missing=decode_failed=0
    for hid in view['host_ids']:
        pairs=view['pairs_by_host'][hid];expected+=len(pairs);candidates=methods.get(hid,[])
        row={'host_stable_id':hid,'status':'OK','events':[],'expected_endpoint_ids':len(pairs)}
        if len(candidates)>1:
            row['status']='AMBIGUOUS_HOST_ID';ambiguous+=1;hosts.append(row);continue
        if not candidates:
            row['status']='HOST_NOT_FOUND';missing+=1;hosts.append(row);continue
        method,dex_name,host_sig=candidates[0]
        try:instructions,_=decode_method(method)
        except Exception as exc:
            row['status']='DECODE_FAILED';row['error_type']=type(exc).__name__;decode_failed+=1;hosts.append(row);continue
        events=Counter();role_for={}
        for ins in instructions:
            if not str(ins.get('name','')).startswith('invoke-'):continue
            ref=canonical(ins.get('reference',''))
            if not ref:continue
            target=str(stable_id(ref))
            if target not in pairs:continue
            roles=sorted(map(int,pairs[target]));family,role_key=endpoint_family(ref,roles)
            key=(target,ref,family,role_key,tuple(roles));events[key]+=1
        seen=set()
        for (target,ref,family,role_key,roles),count in sorted(events.items()):
            row['events'].append({'endpoint_stable_id':target,'endpoint':ref,'family':family,'role_key':role_key,'role_ids':list(roles),'count':int(count)});seen.add(target)
        recovered+=len(seen);row.update({'dex_name':dex_name,'host_signature_sha256':__import__('hashlib').sha256(host_sig.encode()).hexdigest(),'recovered_endpoint_ids':len(seen),'missing_endpoint_ids':sorted(set(pairs)-seen),'status':'OK' if set(pairs).issubset(seen) else 'PARTIAL'})
        hosts.append(row)
    atomic_json_gz(output_path,{'sha256':sid,'year':int(year),'status':'OK','dex_count':dex_count,'hosts':hosts,'expected_endpoint_ids':expected,'recovered_endpoint_ids':recovered,'ambiguous_hosts':ambiguous,'missing_hosts':missing,'decode_failed_hosts':decode_failed,'apk_sha256_verified':True})


def subprocess_one(row,smoke=False):
    folder=OUT/('smoke/api_events' if smoke else 'features/api_events');folder.mkdir(parents=True,exist_ok=True);out=folder/f"{row['sha256']}.json.gz"
    if out.is_file():return row['sha256'],'CACHED',0.
    cmd=[sys.executable,str(Path(__file__).resolve()),'--scan-one','--sha',row['sha256'],'--year',str(row['year']),'--apk',row['absolute_path'],'--view',str(OUT/'features/frozen_views'/f"{row['sha256']}.json.gz"),'--output',str(out)]
    start=time.monotonic();p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    if p.returncode:return row['sha256'],'ERROR',time.monotonic()-start,p.stdout[-8000:]
    return row['sha256'],'OK',time.monotonic()-start,''


def run(smoke_count=0,workers=2):
    rows=lineage()
    if smoke_count:
        candidates=[]
        for r in rows:
            v=read_json_gz(OUT/'features/frozen_views'/f"{r['sha256']}.json.gz")
            if v['host_ids']:candidates.append((len(v['host_ids']),r))
        selected=[r for _,r in sorted(candidates,key=lambda x:x[0])[:smoke_count]];results=[subprocess_one(r,True) for r in selected]
        if any(x[1] not in ('OK','CACHED') for x in results):raise RuntimeError('API smoke failed '+repr(results))
        atomic_json(OUT/'smoke/api_extraction.json',{'status':'PASS','results':results},replace=True);print('API_EXTRACTION_SMOKE_PASS',flush=True);return
    status('api_signature_extraction',completed=len(list((OUT/'features/api_events').glob('*.json.gz'))),total=2788)
    failed=[];done=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(subprocess_one,r,False):r['sha256'] for r in rows}
        for future in as_completed(futures):
            sid=futures[future]
            try:result=future.result()
            except Exception as exc:result=(sid,'ERROR',0.,repr(exc))
            done+=1
            if result[1] not in ('OK','CACHED'):
                failed.append({'sha256':sid,'status':result[1],'detail':result[3] if len(result)>3 else ''})
            if done%20==0 or done==2788:
                atomic_json(OUT/'logs/api_extraction_status.json',{'status':'RUNNING','completed':done,'failed':len(failed),'total':2788,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},replace=True)
                print('API_EXTRACTION_PROGRESS',done,'/2788 failed=',len(failed),flush=True)
    if failed:
        atomic_json(OUT/'logs/api_extraction_failures.json',failed,replace=True);raise RuntimeError(f'API extraction failed for {len(failed)} APKs')
    atomic_json(OUT/'logs/api_extraction_status.json',{'status':'COMPLETE','completed':2788,'failed':0,'total':2788},replace=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scan-one',action='store_true');p.add_argument('--sha');p.add_argument('--year',type=int);p.add_argument('--apk');p.add_argument('--view');p.add_argument('--output');p.add_argument('--smoke',type=int,default=0);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.scan_one:scan_one(a.sha,a.year,a.apk,Path(a.view),Path(a.output))
    else:run(a.smoke,a.workers)
if __name__=='__main__':main()
