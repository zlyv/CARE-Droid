from __future__ import annotations
import csv,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np

NAME='care-direct-c43-route-conditioned-diagnostics-v1'
HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('RCTD_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT=PROJECT/'outputs'/NAME
PARENT_SCRIPT=PROJECT/'scripts/care-direct-c43-route-conditioned-training-v2'
PARENT_OUTPUT=PROJECT/'outputs/care-direct-c43-route-conditioned-training-v2'
FAMILY=PROJECT/'scripts/care-p31-classifier-family-stage2-v1'
OLD_FUTURE=PROJECT/'scripts/care-svm-direct-c43-future-v1'
FORMAL_DIAG=PROJECT/'artifacts/care-droid-formal-diagnostics-v1'
YEARS=(2018,2019,2020,2021)
COUNTS={2018:793,2019:399,2020:397,2021:400}
ARMS=('B0_ORIGINAL','B1_ROUTE_WEIGHTED','B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE')

def norm(x): return (x.decode() if isinstance(x,bytes) else str(x)).strip().casefold()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def safe(p):
    p=Path(p).resolve()
    if p!=OUTPUT and OUTPUT not in p.parents:raise RuntimeError('write outside independent diagnostic output')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def dump(p,v):
    p=safe(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_csv(p,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status'];p=safe(p)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save_npz(p,**values):
    with safe(p).open('wb') as f:np.savez_compressed(f,**values)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return dict(absolute_path=str(p),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p),purpose=purpose)
def module(name,path):
    path=Path(path).resolve();existing=sys.modules.get(name)
    if existing is not None and getattr(existing,'__file__',None) and Path(existing.__file__).resolve()==path:return existing
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m);return m
def parent_modules():
    os.environ['RCT_PROJECT']=str(PROJECT)
    os.environ['PCFS2_PROJECT']=str(PROJECT)
    os.environ['SDC43F_PROJECT']=str(PROJECT)
    core=module('pilot_common',PARENT_SCRIPT/'pilot_common.py')
    core.OUTPUT=OUTPUT
    runtime=module('_rctd_source_runtime',PARENT_SCRIPT/'runtime.py')
    weight=module('_rctd_source_weighting',PARENT_SCRIPT/'weighting.py')
    adapter=module('_rctd_source_train',PARENT_SCRIPT/'train_adapter.py')
    return core,runtime,weight,adapter
def integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');after=[];changed=[]
    for r in before:
        p=Path(r['absolute_path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='MISSING'));continue
        a=asset(p,r['purpose']);after.append(a)
        if any(a[k]!=r[k] for k in ('size_bytes','mtime_ns','sha256')):changed.append(dict(before=r,after=a))
    dump(OUTPUT/'baseline_asset_manifest_after.json',after);dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused assets, not the whole server'))
    if changed:raise RuntimeError('protected reused asset changed')
