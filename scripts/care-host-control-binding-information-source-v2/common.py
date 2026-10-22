#!/usr/bin/env python3
from __future__ import annotations
import csv,functools,gzip,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np

NAME='care-host-control-binding-information-source-v2'
HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('HBI_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT=PROJECT/'outputs'/NAME
OBS=Path(os.environ.get('HBI_OBSERVABILITY',str(PROJECT/'outputs/care-host-control-binding-observability-v1'))).resolve()
STORE=PROJECT/'outputs/care-svm-c43-representation-collision-audit-v1/representation_store.npz'
REPLAY=STORE.with_name('replay_validation.json')
FORMAL=PROJECT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
DIRECT=PROJECT/'outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv'

def norm(x):return (x.decode() if isinstance(x,bytes) else str(x)).strip().casefold()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return dict(absolute_path=str(p),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p),purpose=purpose)
def safe(p):
    p=Path(p).resolve();root=(PROJECT/'outputs').resolve()
    if OUTPUT.resolve()!=root/NAME or (p!=OUTPUT.resolve() and OUTPUT.resolve() not in p.parents):raise RuntimeError('write outside independent HBI output')
    p.parent.mkdir(parents=True,exist_ok=True);return p
def dump(p,obj):
    p=safe(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_csv(p,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with safe(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def protocol():return read_json(HERE/'protocol.json')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@functools.lru_cache(maxsize=1)
def reference():
    m=module('_hbi_reference',HERE/'reference_core.py')
    m.FORMAL=FORMAL;m.DIRECT=DIRECT;m.STORE=STORE
    return m
def load_reference():return reference().load_reference()
def load_store(ids,y,folds):return reference().load_store(ids,y,folds)
def auc(y,p):return reference().auc(y,p)
def metrics(y,base,p,route):return reference().metrics(y,base,p,route)

def integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');after=[];changed=[]
    for i,old in enumerate(before,1):
        p=Path(old['absolute_path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='MISSING'));continue
        now=asset(p,old['purpose']);after.append(now)
        if any(now[k]!=old[k] for k in ('size_bytes','mtime_ns','sha256')):changed.append(dict(before=old,after=now))
        if i%200==0:print(f'HBI_ASSET_AFTER {i}/{len(before)}',flush=True)
    dump(OUTPUT/'baseline_asset_manifest_after.json',after)
    dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,
        scope='actual reused source files only; original APKs never opened; not the whole project'))
    if changed:raise RuntimeError('protected asset integrity failure')
