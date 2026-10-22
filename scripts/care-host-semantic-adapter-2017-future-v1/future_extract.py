#!/usr/bin/env python3
"""Extract the frozen v2 S1/S3 semantics from 2018--2021 APKs.

The extractor reads the formal multiyear SHA inventory, future endpoint cache,
and original APK paths. It writes resumable per-APK semantic caches under the
new experiment only. It never reads labels or mutates W0 features.
"""
from __future__ import annotations
import argparse,gzip,json,os,subprocess,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
from semantic_common import ROOT,OUT,canonical,stable_id,atomic_json
from semantic_core import analyze_method

ENDPOINT=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz')
MANIFEST=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021/temporal_eval_manifest.csv')
DATA=ROOT/'outputs/care_multiyear_development_v1/data.npz'; ROLE_MAP=[7,10,21,19,9,15,16,0,11,25,29,6,12,1,26,28,23,13,14,2,17,30,22,4,5,24,3,18,8,20,27]

def sha256(path):
 import hashlib; h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def write_gz(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,separators=(',',':'))
 tmp.replace(path)

def frozen_view(record):
 p=record['payload']; ids=list(map(int,p['node_ids'])); roles=p['node_role_ids']; base=np.asarray(p['node_base_features'],np.float32); edges=p['edge_index']
 if base.size==0: base=np.empty((0,15),np.float32)
 if base.shape!=(len(ids),15) or len(edges)!=2: raise RuntimeError(f'future endpoint schema mismatch {record.get("sha256")} {base.shape} ids={len(ids)}')
 model=[sorted({ROLE_MAP[int(v)] for v in (x or [])}) for x in roles]; endpoints=set(map(int,np.flatnonzero(base[:,4]>.5))); targets=defaultdict(dict)
 for l,r in zip(edges[0],edges[1]):
  l,r=int(l),int(r)
  if r in endpoints and model[r]: targets[l][str(ids[r])]=model[r]
 nodes=sorted(n for n,v in targets.items() if v); return {'sha256':record['sha256'],'host_ids':[str(ids[n]) for n in nodes],'pairs_by_host':{str(ids[n]):targets[n] for n in nodes}}

def inventory():
 z=np.load(DATA,allow_pickle=False); years=np.asarray(z['years'],int); ids=np.asarray(z['ids']).astype(str); real=np.asarray(z['real_host_count'],int); wanted={ids[i]:(int(years[i]),int(real[i])) for i in range(len(ids)) if years[i] in (2018,2019,2020,2021)}
 if {y:sum(v[0]==y for v in wanted.values()) for y in range(2018,2022)}!={2018:793,2019:399,2020:397,2021:400}: raise RuntimeError('future inventory count mismatch')
 frame=pd.read_csv(MANIFEST,dtype=str); key='sha256' if 'sha256' in frame else 'sample_id'; pathcol='apk_path' if 'apk_path' in frame else 'absolute_path'; frame[key]=frame[key].str.lower(); paths=dict(zip(frame[key],frame[pathcol])); missing=set(wanted)-set(paths)
 if missing: raise RuntimeError(f'missing APK paths {len(missing)}')
 return wanted,paths

def prepare():
 wanted,paths=inventory(); found={}
 with gzip.open(ENDPOINT,'rt',encoding='utf-8') as f:
  for line in f:
   if not line.strip(): continue
   r=json.loads(line); sid=str(r.get('sha256','')).lower()
   if sid in wanted: found[sid]=r
 if set(found)!=set(wanted): raise RuntimeError(f'future endpoint coverage {len(found)}/{len(wanted)}')
 rows=[]
 for sid,(year,count) in wanted.items():
  view=frozen_view(found[sid])
  if len(view['host_ids'])!=count: raise RuntimeError(f'future Host count mismatch {sid}: {len(view["host_ids"])} != {count}')
  p=OUT/'future_frozen'/str(year)/(sid+'.json'); atomic_json(p,view); rows.append({'sha256':sid,'year':year,'absolute_path':paths[sid]})
 atomic_json(OUT/'audit/future_host_alignment.json',{'status':'PASS','apks':len(rows),'counts':{str(y):sum(r['year']==y for r in rows) for y in range(2018,2022)},'labels_accessed':False})
 return rows

def scan_one(sid,year,apk_path,frozen_path,outpath):
 if sha256(apk_path)!=sid: raise RuntimeError('APK SHA mismatch')
 frozen=json.loads(Path(frozen_path).read_text());
 if not frozen['host_ids']: write_gz(outpath,{'sha256':sid,'year':year,'status':'OK_NO_HOST','hosts':[],'wall_seconds':0.}); return
 sys.path.insert(0,str(ROOT/'scripts/care-local-control-constraint-feasibility-v2')); from dex_adapter import decode_method
 try:
  from loguru import logger; logger.remove()
 except Exception: pass
 from androguard.core.apk import APK
 from androguard.core.dex import DEX
 start=time.monotonic(); apk=APK(str(apk_path)); wanted=set(frozen['host_ids']); methods=defaultdict(list)
 for dn in sorted(apk.get_dex_names()):
  dex=DEX(apk.get_file(dn))
  for cls in dex.get_classes():
   cn=str(cls.get_name())
   for m in cls.get_methods():
    hid=str(stable_id(canonical(cn+'->'+str(m.get_name())+str(m.get_descriptor()))))
    if hid in wanted: methods[hid].append((m,dn))
 hosts=[]
 for hid in frozen['host_ids']:
  row={'host_stable_id':hid,'parse_status':'METHOD_ID_NOT_FOUND','instruction_count':0,'sensitive_event_count':0,'s1_counts':{},'s2_counts':{},'s3_dependencies':[],'unknown_reasons':['HOST_ALIGNMENT_UNKNOWN']}; cand=methods.get(hid,[])
  if len(cand)!=1:
   if len(cand)>1: row.update(parse_status='AMBIGUOUS_STABLE_ID',unknown_reasons=['HOST_ALIGNMENT_UNKNOWN','MULTIDEX_OR_CANONICAL_DUPLICATE'])
   hosts.append(row); continue
  try:
   m,dn=cand[0]; ins,meta=decode_method(m); a=analyze_method(ins,frozen['pairs_by_host'][hid],tries_size=int(meta.get('tries_size',0)),maximum_instructions=4096); s1=Counter(c for e in a['events'] for c in e['s1_categories']); s2=Counter(e['s2_primary'] for e in a['events']); row.update(parse_status=a['status'],instruction_count=a['instruction_count'],sensitive_event_count=len(a['events']),s1_counts=dict(s1),s2_counts=dict(s2),s3_dependencies=a['dependencies'],unknown_reasons=a['unknown_reasons'],dex_name=dn)
  except Exception as e: row.update(parse_status='UNPARSEABLE',unknown_reasons=['DEX_METHOD_DECODE_FAILED',type(e).__name__])
  hosts.append(row)
 write_gz(outpath,{'sha256':sid,'year':year,'status':'OK','hosts':hosts,'apk_sha256_verified':True,'wall_seconds':time.monotonic()-start})

def worker(row):
 sid,year=row['sha256'],row['year']; out=OUT/'future_cache'/str(year)/(sid+'.json.gz')
 if out.is_file(): return sid,'CACHED'
 cmd=[sys.executable,__file__,'--one','--sha',sid,'--year',str(year),'--apk',row['absolute_path'],'--frozen',str(OUT/'future_frozen'/str(year)/(sid+'.json')),'--out',str(out)]; p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 if p.returncode: atomic_json(OUT/'future_status'/str(year)/(sid+'.json'),{'status':'ERROR','output':p.stdout[-6000:]}); return sid,'ERROR'
 return sid,'OK'

def run():
 rows=prepare(); done=fail=0; atomic_json(OUT/'logs/run_status.json',{'status':'FUTURE_EXTRACTION_RUNNING','total':len(rows),'completed':0})
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(worker,r):r for r in rows}
  for f in as_completed(fs):
   _,s=f.result(); done+=1; fail+=s=='ERROR'
   if done%10==0 or done==len(rows): atomic_json(OUT/'logs/run_status.json',{'status':'FUTURE_EXTRACTION_RUNNING','total':len(rows),'completed':done,'failed':fail}); print('FUTURE_SCAN',done,'/',len(rows),'failed',fail,flush=True)
 if fail: raise RuntimeError(f'future extraction failures {fail}')
 atomic_json(OUT/'audit/future_semantic_extraction.json',{'status':'PASS','apks':len(rows),'failed':0,'labels_accessed':False})

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--one',action='store_true'); ap.add_argument('--sha'); ap.add_argument('--year',type=int); ap.add_argument('--apk'); ap.add_argument('--frozen'); ap.add_argument('--out'); a=ap.parse_args()
 if a.one: scan_one(a.sha,a.year,a.apk,a.frozen,a.out)
 else: run()
