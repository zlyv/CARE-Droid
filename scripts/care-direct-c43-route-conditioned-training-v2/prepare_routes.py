from __future__ import annotations
import sys,time
import numpy as np
from pilot_common import *
from weighting import opposition,make_weights

def crossfit_committee(x,y,ids,seed,fit_committee,splitter,jobs=4):
    out=np.full((len(y),4),np.nan); assignment=np.full(len(y),-1,int); audits=[]
    for j,(fit,hold) in enumerate(splitter.split(x,y),1):
        if set(ids[fit])&set(ids[hold]): raise RuntimeError('crossfit fit/hold overlap')
        s=int(seed)+100000*j
        out[hold]=fit_committee(x[fit],y[fit],x[hold],s,inner_folds=3,jobs=jobs); assignment[hold]=j
        audits.append(dict(inner_fold=j,member_seed=s,fit_sha=ids[fit].tolist(),hold_sha=ids[hold].tolist()))
        print(f'RCT_ROUTER_INNER fold={j} fit={len(fit)} hold={len(hold)}',flush=True)
    if not np.isfinite(out).all() or np.any(assignment<0): raise RuntimeError('incomplete crossfit committee')
    return out,assignment,audits

def main():
    sys.path.insert(0,str(FAMILY))
    from runtime import family_module
    fit_calibrated_committee=family_module('stage1_core').fit_calibrated_committee
    from sklearn.model_selection import StratifiedKFold
    from runtime import load_runtime,arguments,parsed_args
    v4,_,_=load_runtime('B0_ORIGINAL'); args=parsed_args(v4,arguments('B0_ORIGINAL'))
    with np.load(OUTPUT/'source_index.npz',allow_pickle=False) as a: d={k:a[k] for k in a.files}
    timing=[]
    for f in range(1,6):
        t=time.monotonic(); ix=np.flatnonzero(d['folds']!=f); ids=d['ids'][ix]; y=d['y'][ix]; seed=int(args.split_seed)+1000*f
        member,assignment,audits=crossfit_committee(d['p31'][ix],y,ids,seed,fit_calibrated_committee,
            StratifiedKFold(4,shuffle=True,random_state=seed),int(os.environ.get('RCT_JOBS','4')))
        weights,route,wa=make_weights(y,ids,member,d['host_count'][ix],protocol()['shuffle_seed']+f,groups=assignment)
        if wa['changed_fraction']<protocol()['minimum_shuffle_changed_fraction']: raise RuntimeError('conditional shuffle lacks mixing; cannot relax strata after observing outcomes')
        save_npz(OUTPUT/'routes'/f'f{f}.npz',ids=ids,train_index=ix,member=member,inner_fold=assignment,route=route,**weights)
        dump(OUTPUT/'routes'/f'f{f}_audit.json',dict(outer_fold=f,excluded_outer_test_sha=d['ids'][d['folds']==f].tolist(),crossfit=audits,
            weight_audit=wa,seconds=time.monotonic()-t,future_rows_loaded=0))
        write_csv(OUTPUT/'routes'/f'f{f}_training_weights.csv',[dict(sha256=s,outer_fold=f,inner_fold=int(assignment[i]),label=int(y[i]),
            inner_SVM_probability=float(member[i,3]),inner_MeanOpposition=float(opposition(member)[i]),inner_routed=int(route[i]),**{a:float(w[i]) for a,w in weights.items()}) for i,s in enumerate(ids)])
        timing.append(dict(fold=f,seconds=time.monotonic()-t,train_n=len(ix),inner_routed=int(route.sum()),**{k:v for k,v in wa.items() if k!='groups'}))
        print(f'RCT_ROUTER_READY outer_fold={f} n={len(ix)} route={int(route.sum())} seconds={time.monotonic()-t:.1f}',flush=True)
    dump(OUTPUT/'routing_audit.json',dict(status='PASS',folds=timing,global_source_OOF_not_used_for_training=True,calibration_nested=True,future_rows_loaded=0))
if __name__=='__main__': main()
