from __future__ import annotations
import os, platform, subprocess, sys
from pathlib import Path
import numpy as np
from pilot_common import *
from runtime import load_runtime,arguments,parsed_args,validate_loaded_inputs,family_module,svm_nested_module

def main():
    v4,adapter,contract=load_runtime('B0_ORIGINAL'); args=parsed_args(v4,arguments('B0_ORIGINAL',os.environ.get('RCT_DEVICE','cuda')))
    required=[(FORMAL,'frozen SVM results'),(DIRECT,'frozen Direct expert'),(P31,'source P31'),
        (args.previous_result/'source_predictions.csv','frozen deployment reference')]
    SOURCE_MEMBER=family_module('common').SOURCE_MEMBER
    required.append((SOURCE_MEMBER,'frozen deployment committee verification only'))
    for k in ('features','schema','cache','p72_oof','source_multiscale_features','region_cache'):
        required.append((Path(getattr(args,k)).resolve(),'original input '+k))
    if args.care_reference.is_file(): required.append((args.care_reference,'optional source reference'))
    # Snapshot code before replay; also protect protocol files used by the original graph builder.
    roots={Path(s).resolve() for s in sys.path if s and str(Path(s).resolve()).startswith(str(PROJECT/'scripts')) and Path(s).is_dir()}
    for root in roots:
        iterator=root.glob('*.py') if root==PROJECT/'scripts' else root.rglob('*.py')
        required.extend((p,'original dependency source') for p in iterator)
    required.extend((p,'protocol configuration') for p in (PROJECT/'configs/protocol').rglob('*') if p.is_file())
    # Actual imported modules can reside in vendor folders not listed separately on sys.path.
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and PROJECT in Path(p).resolve().parents and OUTPUT not in Path(p).resolve().parents and HERE not in Path(p).resolve().parents:
            required.append((Path(p),'imported dependency'))
    missing=[str(p) for p,_ in required if not p.is_file()]
    if missing: raise RuntimeError('missing original source dependencies: '+repr(missing[:20]))
    records={str(Path(p).resolve()):asset(p,purpose) for p,purpose in required}
    dump(OUTPUT/'baseline_asset_manifest_before.json',list(records.values()))
    # Freeze this package separately; changes require a new version.
    dump(OUTPUT/'implementation_manifest.json',[asset(p,'Pilot implementation') for p in HERE.iterdir() if p.is_file()])
    nested_module=svm_nested_module()
    nested=nested_module.NestedStage1('SVM',OUTPUT/'audits/preflight_nested.json',int(os.environ.get('RCT_JOBS','4')))
    if nested.arm!='SVM' or len(nested.ids)!=799: raise RuntimeError('family SVM NestedStage1 capability check failed')
    dump(OUTPUT/'import_identity.json',dict(nested_stage1=str(Path(nested_module.__file__).resolve()),
        stage1_core=str(Path(family_module('stage1_core').__file__).resolve()),common=str(Path(family_module('common').__file__).resolve()),
        model_adapter=str(Path(adapter.__file__).resolve()),SVM_constructor_verified=True))
    loaded=v4.audit.load_inputs(args); ids,y,folds,base,route,expert=validate_loaded_inputs(loaded)
    stats=metrics(y,base,expert,route)
    if [int((folds==f).sum()) for f in range(1,6)]!=[160,160,160,160,159] or [int((route&(folds==f)).sum()) for f in range(1,6)]!=[40]*5:
        raise RuntimeError('frozen folds/routes changed')
    if (stats['fixed'],stats['broken'],stats['net'])!=(49,17,32) or int(((base>=.5)!=y).sum())!=97 or stats['routed_errors']!=84:
        raise RuntimeError('frozen SVM/Direct reference changed')
    bags,_,_=v4.get_source_host_bags(loaded[1],None,args)
    with np.load(P31,allow_pickle=False) as a:
        index={norm(s):i for i,s in enumerate(a['sha256'])}; values=np.asarray(a['role_prevalence'],float)
        if values.shape!=(799,31) or set(index)!=set(ids): raise RuntimeError('P31 membership/shape changed')
        p31=values[[index[s] for s in ids]]
    save_npz(OUTPUT/'source_index.npz',ids=ids,y=y,folds=folds,base=base,route=route,expert=expert,p31=p31,host_count=np.asarray([len(b) for b in bags]))
    # Ensure frozen deployment reference really implements the existing four-member rule.
    from weighting import opposition,top25
    mr={norm(r['sha256']):r for r in read_csv(SOURCE_MEMBER)}
    if len(mr)!=799 or set(mr)!=set(ids): raise RuntimeError('source member membership')
    member=np.asarray([[float(mr[s]['p_P31__'+n]) for n in ('LR','RF','XGB','SVM')] for s in ids])
    if not np.allclose(member[:,3],base,rtol=0,atol=1e-14): raise RuntimeError('SVM anchor changed')
    sc=opposition(member); rr=np.zeros(799,bool)
    for f in range(1,6):
        ix=np.flatnonzero(folds==f); rr[ix]=top25(sc[ix],ids[ix])
    if not np.array_equal(rr,route): raise RuntimeError('frozen four-member deployment route differs')
    refrows={norm(r['sha256']):r for r in read_csv(args.previous_result/'source_predictions.csv')}
    for i,s in enumerate(ids):
        r=refrows[s]
        if abs(float(r['Global24__prob'])-base[i])>1e-14 or int(r['A25__routed'])!=int(route[i]) or abs(float(r['A25__score'])-sc[i])>1e-14: raise RuntimeError('previous-result routing identity failed')
    # SOURCE_MEMBER is diagnostic of deployment only, never used for training masks.
    records[str(SOURCE_MEMBER.resolve())]=asset(SOURCE_MEMBER,'frozen deployment committee verification only')
    dump(OUTPUT/'baseline_asset_manifest_before.json',list(records.values()))
    dump(OUTPUT/'protocol.json',protocol())
    try: gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,timeout=10)
    except Exception as e: gpu=repr(e)
    import sklearn,scipy,xgboost
    dump(OUTPUT/'environment_inventory.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,torch=v4.torch.__version__,
        sklearn=sklearn.__version__,scipy=scipy.__version__,xgboost=xgboost.__version__,gpu=gpu,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        label_contract_path=str(Path(contract.__file__).resolve()),v4_path=str(Path(v4.__file__).resolve()),new_DEX_scans=0,future_rows_loaded=0))
    v4.resolve_device(args.device)
    dump(OUTPUT/'preflight.json',dict(status='PASS',frozen_reference=stats,source_rows=799,routed=200,training_distribution='all APK equal BCE',
        input_order_verified=True,cache_read_only=True,future_rows_loaded=0))
    print('RCT_PREFLIGHT_PASS source=799 route=200 direct_fixed=49 broken=17',flush=True)
if __name__=='__main__': main()
