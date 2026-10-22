"""Independent forensic outputs, protected-asset hashes, and read-only access to completed development models."""
from __future__ import annotations
import csv,gzip,hashlib,json,os,pickle,re,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('MEEA_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
BASE=PROJECT/'outputs/care_multiyear_development_v1'
SOURCE=PROJECT/'scripts/care-multiyear-development-v1'
OUTPUT=Path(os.environ.get('MEEA_OUTPUT',str(PROJECT/'outputs/care_multiyear_error_evidence_audit_v1'))).resolve()
if OUTPUT.parent!=(PROJECT/'outputs').resolve() or not OUTPUT.name.startswith('care_multiyear_error_evidence_audit_v'):
    raise RuntimeError('invalid independent audit output root')
ARMS=('U2_NEW_ROUTER_OLD_EXPERT','U3_FULL_ROLLING_C1','U4_FULL_YEAR_BALANCED_C2')
SEEDS=(900000,900001,900002)

def canonical(s):return re.sub(r'\s+','',str(s)).casefold()
def stable_id(s):return str(int.from_bytes(hashlib.sha256(canonical(s).encode()).digest()[:8],'big'))
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def safe(p):
    p=Path(p).resolve()
    if OUTPUT not in p.parents:raise RuntimeError('output outside independent audit '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True);return p
def plain(v):
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set)):return [plain(x) for x in v]
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):
    p=safe(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(plain(v),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def write_csv(p,rows):
    p=safe(p);rows=list(rows);names=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def signature(v):return hashlib.sha256(json.dumps(plain(v),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return dict(path=str(p),purpose=purpose,size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p))
def integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');after=[];changed=[]
    for r in before:
        p=Path(r['path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='missing'));continue
        a=asset(p,r['purpose']);after.append(a)
        if any(a[k]!=r[k] for k in ['size','mtime_ns','sha256']):changed.append(dict(before=r,after=a))
    dump(OUTPUT/'baseline_asset_manifest_after.json',after);dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused assets and inspected original APKs, not the entire server'))
    if changed:raise RuntimeError('protected original asset changed')
def old():
    sys.path.insert(0,str(SOURCE));import myd_common as mc
    from audit_data import load_data
    return mc,load_data()
def archive(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def load_checkpoint(path,mc,data):
    with Path(path).open('rb') as f:o=pickle.load(f)
    if o['signature']!=mc.signature(o['metadata']):raise RuntimeError('checkpoint metadata signature mismatch '+str(path))
    if o['metadata']['dataset']!=data['signature'] or o['metadata']['code']!=mc.implementation_hash():raise RuntimeError('checkpoint dataset/code mismatch '+str(path))
    return o
