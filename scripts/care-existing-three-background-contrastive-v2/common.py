"""Read-only access to completed CARE data and exclusive writes to this independent experiment."""
from __future__ import annotations
import csv,hashlib,json,os,pickle,sys
from pathlib import Path
import numpy as np

PROJECT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=Path(__file__).resolve().parent
SOURCE=PROJECT/'scripts/care-multiyear-development-v1'
BASE=PROJECT/'outputs/care_multiyear_development_v1'
OUTPUT=Path(os.environ.get('ETBC_OUTPUT',str(PROJECT/'outputs/care_existing_three_background_contrastive_v1'))).resolve()
if OUTPUT.parent!=(PROJECT/'outputs').resolve() or not OUTPUT.name.startswith('care_existing_three_background_contrastive_v'):
    raise RuntimeError('unsafe experiment output path')

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return dict(path=str(p),purpose=purpose,size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write_json(p,x):
    p=Path(p).resolve()
    if OUTPUT not in p.parents:raise RuntimeError('write outside isolated output')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(plain(x),f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
def write_text(p,x):
    p=Path(p).resolve()
    if OUTPUT not in p.parents:raise RuntimeError('write outside isolated output')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(x)
def write_csv(p,rows):
    rows=list(rows);p=Path(p).resolve()
    if OUTPUT not in p.parents:raise RuntimeError('write outside isolated output')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def original():
    sys.path.insert(0,str(SOURCE))
    import myd_common as mc
    from audit_data import load_data
    v4,_,args=mc.runtime()
    data=load_data()
    return mc,v4,args,data
def checkpoint(path,mc,data):
    with Path(path).open('rb') as f:o=pickle.load(f)
    if o['signature']!=mc.signature(o['metadata']) or o['metadata']['dataset']!=data['signature'] or o['metadata']['code']!=mc.implementation_hash():raise RuntimeError('checkpoint signature mismatch')
    return o
def integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');changed=[]
    for a in before:
        p=Path(a['path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='missing'));continue
        b=asset(p,a['purpose'])
        if any(a[k]!=b[k] for k in ['size','mtime_ns','sha256']):changed.append(dict(before=a,after=b))
    write_json(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused assets, not all server files'))
    if changed:raise RuntimeError('protected original asset changed')
