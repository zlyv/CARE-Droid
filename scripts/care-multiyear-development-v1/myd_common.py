"""Multi-year independent paths, protected assets and original dependency loading."""
from __future__ import annotations
import csv,hashlib,importlib,importlib.util,json,os,pickle,sys,time
from pathlib import Path
import numpy as np

NAME='care-multiyear-development-v1'
HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('MYD_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT=PROJECT/'outputs/care_multiyear_development_v1'
FAMILY=PROJECT/'scripts/care-p31-classifier-family-stage2-v1'
DIAG=PROJECT/'artifacts/care-droid-formal-diagnostics-v1'
YEARS=(2017,2018,2019,2020,2021)
EXPECTED={2017:799,2018:793,2019:399,2020:397,2021:400}
MEMBERS=('LR','RF','XGB','SVM')
def norm(x):return (x.decode() if isinstance(x,bytes) else str(x)).strip().casefold()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def safe(p):
    p=Path(p).resolve()
    if p!=OUTPUT and OUTPUT not in p.parents:raise RuntimeError('write outside independent multi-year output: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True);return p
def clean_json(x):
    if isinstance(x,np.ndarray):return clean_json(x.tolist())
    if isinstance(x,np.generic):return clean_json(x.item())
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x
def dump(p,v):
    p=safe(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean_json(v),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_csv(p,rows):
    rows=list(rows);p=safe(p);t=p.with_suffix(p.suffix+'.tmp');fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with t.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    t.replace(p)
def save_npz(p,**data):
    p=safe(p);t=p.with_suffix('.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**data)
    t.replace(p)
def save_pickle(p,v):
    p=safe(p);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('wb') as f:pickle.dump(v,f,protocol=pickle.HIGHEST_PROTOCOL)
    t.replace(p)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return dict(path=str(p),purpose=purpose,size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p))
def signature(value):return hashlib.sha256(json.dumps(clean_json(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def implementation_hash():return signature({p.name:sha256(p) for p in HERE.glob('*.py')})
def module(name,path):
    path=Path(path).resolve();old=sys.modules.get(name)
    if old is not None and getattr(old,'__file__',None) and Path(old.__file__).resolve()==path:return old
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def family(name,alias=None):
    # Dependencies import a generic common; pin it to the original family module.
    if name!='common':family('common',alias='_myd_family_common')
    if name=='common':
        m=module(alias or '_myd_family_common',FAMILY/'common.py');sys.modules['common']=m;return m
    return module(alias or name,FAMILY/(name+'.py'))
def runtime():
    os.environ['PCFS2_PROJECT']=str(PROJECT);sys.path.insert(0,str(FAMILY))
    adapter=family('model_adapter');adapter.prepare_imports()
    contract=module('_myd_label_contract',PROJECT/'scripts/care-malben-host-v2/source_label_contract.py')
    v4=importlib.import_module('run_care_rolegraph_conditioned_dualstream_v4')
    contract.install_source_label_contract(v4);adapter.install_fold_safe_compact76(v4.audit)
    adapter.install_a0(v4,'MYD_DIRECT_C43')
    variant=module('_myd_direct_variant',PROJECT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py')
    variant.install_variant(v4,'U2_DIRECT_STAGE2')
    args=family('run_frozen_arm').build_parser().parse_args(['--arm','SVM','--device','cpu'])
    args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.;args.rebuild_future_inter_cache=False
    r=read_json(DIAG/'resolved_inputs.json');args.features=Path(r['sealed_features']);args.cache=Path(r['endpoint_cache']);args.future_inter_cache=Path(r['future_inter_cache']);args.region_cache=DIAG/'cache/a0_role_2017_2021.npz'
    return v4,adapter,args
def verify_assets():
    records=read_json(OUTPUT/'baseline_asset_manifest_before.json');changed=[];after=[]
    for r in records:
        p=Path(r['path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='missing'));continue
        a=asset(p,r['purpose']);after.append(a)
        if any(a[k]!=r[k] for k in ('size','mtime_ns','sha256')):changed.append(dict(before=r,after=a))
    dump(OUTPUT/'baseline_asset_manifest_after.json',after);dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',changed=changed,asset_count=len(records),scope='listed original files only'))
    if changed:raise RuntimeError('protected original asset changed')
def load_checkpoint(p,metadata):
    p=Path(p)
    if not p.is_file():return None
    with p.open('rb') as f:obj=pickle.load(f)
    if obj.get('signature')!=signature(metadata):raise RuntimeError('existing checkpoint contract differs: '+str(p))
    return obj
