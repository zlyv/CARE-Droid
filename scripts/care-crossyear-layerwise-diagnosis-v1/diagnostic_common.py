"""Read-only CARE dependencies and exclusive output writers for the layer audit."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

PROJECT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=Path(__file__).resolve().parent
OUTPUT=Path(os.environ.get('CLD_OUTPUT',str(PROJECT/'outputs/care_crossyear_layerwise_diagnosis_v1'))).resolve()
if OUTPUT.parent != (PROJECT/'outputs').resolve() or not OUTPUT.name.startswith('care_crossyear_layerwise_diagnosis_v'):
    raise RuntimeError('unsafe output directory')
PREVIOUS=PROJECT/'scripts/care-existing-three-background-contrastive-v2'
sys.path.insert(0,str(PREVIOUS))
import common as old
from contrastive_training import inputs
BASE=old.BASE
TAGS=('T2_C1','T3_C1','T4_C1')
SEEDS=(900000,900001,900002)

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat()
    return dict(path=str(p),purpose=purpose,size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=digest(p))
def safe(p):
    p=Path(p).resolve()
    if OUTPUT not in p.parents:raise RuntimeError('write outside output '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p
def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write_json(p,x):
    with safe(p).open('x',encoding='utf-8') as f:json.dump(serial(x),f,ensure_ascii=False,indent=2,allow_nan=False)
def write_csv(p,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with safe(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(serial(rows))
def write_text(p,s):
    with safe(p).open('x',encoding='utf-8') as f:f.write(s)
def write_npz(p,**values):
    with safe(p).open('xb') as f:np.savez_compressed(f,**values)
def original():return old.original()
def checkpoint(p,mc,data):return old.checkpoint(p,mc,data)
def integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');after=[];changed=[]
    for a in before:
        p=Path(a['path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='missing'));continue
        b=asset(p,a['purpose']);after.append(b)
        if any(a[k]!=b[k] for k in ('size','mtime_ns','sha256')):changed.append(dict(before=a,after=b))
    write_json(OUTPUT/'baseline_asset_manifest_after.json',after)
    write_json(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused files only; not all files or original APKs'))
    if changed:raise RuntimeError('protected asset integrity changed')
