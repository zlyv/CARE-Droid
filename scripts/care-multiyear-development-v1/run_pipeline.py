"""Audit-first CLI for smoke, selected rolling stages, checkpoint resume and independent outputs."""
from __future__ import annotations
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from myd_common import *

def smoke(data,device):
    from train_stage1 import fit_member,probability
    from train_stage2 import fit
    from protocol import mean_opposition,top_budget
    from metrics import cascade
    # Training subset selection is deterministic and confined to train labels.
    tr=np.concatenate([np.flatnonzero((data['years']==year)&(data['y']==lab))[:20] for year in [2017,2018] for lab in [0,1]])
    va=np.flatnonzero(data['years']==2019)[:8];start=time.perf_counter()
    m=fit_member('SVM',data['p31'][tr],data['y'][tr],data['ids'][tr]);p=probability(m,data['p31'][va])
    orig=family('stage1_core').fit_calibrated_member('SVM',data['p31'][tr],data['y'][tr],data['p31'][va],20260904)
    diff=float(np.max(np.abs(p-orig)))
    if diff>1e-12:raise RuntimeError('saved calibrated SVM adapter differs from original '+str(diff))
    save_pickle(OUTPUT/'smoke/stage1.pkl',m)
    with (OUTPUT/'smoke/stage1.pkl').open('rb') as f:again=__import__('pickle').load(f)
    if not np.array_equal(p,probability(again,data['p31'][va])):raise RuntimeError('SVM reload mismatch')
    # Exercise the actual original four-member router on these same train/val SHAs.
    models={n:fit_member(n,data['p31'][tr],data['y'][tr],data['ids'][tr]) for n in MEMBERS if n!='SVM'};models['SVM']=m
    member=np.column_stack([probability(models[n],data['p31'][va]) for n in MEMBERS]);score=mean_opposition(member,p);route=top_budget(score,data['ids'][va])
    tag='SMOKE_'+implementation_hash()[:12]+'_'+device
    c1,epochs=fit(data,tr,va,tag+'_C1',device,smoke=True,seeds=(900000,));c2,_=fit(data,tr,va,tag+'_C2',device,smoke=True,seeds=(900000,),epoch_reference=epochs)
    # Independently call the untouched original fit loop on the identical small
    # inputs, not merely our reloader, to verify the adapter's numerical semantics.
    from train_stage2 import raw_bags
    v4,_,args=runtime();args.max_epochs=1;dev=v4.resolve_device(device);fr=data['feature_rows'];rows=fr['rows']
    gx,gz,names=v4.audit.compact_matrix_variant(v4.VARIANT,[rows[i] for i in tr],[rows[i] for i in va],fr['compact_by_sha'],fr['schema'],fr['canonical_names']);gi,_=v4.evidence.split_view_indices(names)
    ref,_,_=v4.fit_predict_variant(np.asarray(gx[:,gi],np.float32),np.asarray(gz[:,gi],np.float32),data['s7'][tr],data['s7'][va],raw_bags(data,tr),raw_bags(data,va),np.full(len(tr),.5),np.full(len(va),.5),data['y'][tr],'RGCondAttn',[900000],args,dev)
    nn_diff=float(np.max(np.abs(ref-c1)))
    if nn_diff>1e-8:raise RuntimeError('Direct adapter differs from original fit loop '+str(nn_diff))
    results=[dict(arm='SMOKE_C1',**cascade(data['y'][va],p,c1,route)),dict(arm='SMOKE_C2',**cascade(data['y'][va],p,c2,route))]
    write_csv(OUTPUT/'smoke/cascade_results.csv',results)
    dump(OUTPUT/'smoke_test.json',dict(status='PASS',device=device,train_n=len(tr),validation_n=len(va),train_years=[2017,2018],validation_year=2019,max_epochs=1,seeds=[900000],smoke_only_not_performance_evidence=True,architecture_changed=False,svm_original_max_probability_difference=diff,direct_original_fit_loop_max_probability_difference=nn_diff,original_network_checkpoint_reload=True,seconds=time.perf_counter()-start,validation_labels_consumed_only_after_prediction=True))
    print('MYD_ORIGINAL_NETWORK_SMOKE_PASS',device,flush=True)

def main():
    cli=argparse.ArgumentParser(description=__doc__);m=cli.add_mutually_exclusive_group(required=True)
    m.add_argument('--audit-only',action='store_true');m.add_argument('--smoke-test',action='store_true');m.add_argument('--run-full',action='store_true');m.add_argument('--collect-only',action='store_true')
    cli.add_argument('--windows',nargs='+',choices=['T1','T2','T3','T4'],default=['T1','T2','T3','T4']);cli.add_argument('--stage',choices=['all','stage1','router','stage2','cascade'],default='all');cli.add_argument('--resume',action='store_true');cli.add_argument('--device',choices=['cpu','cuda'],default='cuda');cli.add_argument('--jobs',type=int,default=4);cli.add_argument('--bootstrap',type=int,default=2000)
    a=cli.parse_args();os.chdir(PROJECT);start=time.perf_counter()
    from audit_data import audit
    data=audit(a.resume)
    from baseline_replay import replay
    replay()
    if not a.resume and (OUTPUT/'models').exists():raise RuntimeError('models already exist; use --resume for signature-checked resume')
    if a.smoke_test:smoke(data,a.device)
    if a.run_full:
        from train_stage1 import run as stage1
        from evaluate_router import evaluate as router
        from train_stage2 import run as stage2
        from evaluate_cascade import evaluate
        for w in a.windows:
            if a.stage in ['all','stage1']:stage1(data,w,a.jobs)
            if a.stage in ['all','router']:router(data,w,a.jobs)
            if a.stage in ['all','stage2']:stage2(data,w,a.device)
            if a.stage in ['all','cascade']:evaluate(data,w,a.bootstrap)
    verify_assets()
    from collect_results import collect
    collect();dump(OUTPUT/'last_run.json',dict(mode=vars(a),seconds=time.perf_counter()-start,full_training_requested=a.run_full))

if __name__=='__main__':
    try:main()
    except Exception as e:
        OUTPUT.mkdir(parents=True,exist_ok=True);dump(OUTPUT/'last_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_silent_sample_drop=True))
        dump(OUTPUT/'final_decision.json',dict(status='FAILED',error=repr(e),automatic_continuation=False,previous_results_not_deleted=True))
        if (OUTPUT/'baseline_asset_manifest_before.json').exists():verify_assets()
        raise
