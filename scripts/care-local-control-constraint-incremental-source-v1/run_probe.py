#!/usr/bin/env python3
from __future__ import annotations
from common import *
from model_core import fit_predict,matching_groups,conditional_shuffle,placebo

def main():
    ids,y,folds,base,route,expert,final=load_reference();store=load_store(ids,y,folds);p=protocol()
    with np.load(OUTPUT/'control_features.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids):raise RuntimeError('control feature SHA order mismatch')
        evidence=np.asarray(z['evidence'],float);coverage=np.asarray(z['coverage'],float)
    with np.load(OUTPUT/'p31_aligned.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids):raise RuntimeError('P31 order mismatch')
        p31=np.asarray(z['p31'],float)
    if evidence.shape!=(799,124) or coverage.shape!=(799,64) or not(np.isfinite(evidence).all() and np.isfinite(coverage).all()):raise RuntimeError('feature matrix contract')
    repeats=p['null_repeats'];scores={name:base.copy() for name in ('M1_HIDDEN_READOUT','M2_HIDDEN_CONTROL','M5_HIDDEN_COVERAGE_ONLY')}
    scores['M0_DIRECT_C43']=final
    for r in range(repeats):
        for arm in ('M3_CAPACITY_PLACEBO','M4_CONDITIONAL_SHUFFLE'):scores[f'{arm}__r{r:02d}']=base.copy()
    donors=[];fold_audit=[]
    for f in range(1,6):
        ti=store[f'f{f}_train_index'].astype(int);vi=store[f'f{f}_test_index'].astype(int)
        tr=store[f'f{f}_train_hidden'];te=store[f'f{f}_test_hidden'];train_base=store[f'f{f}_train_base'];test_base=store[f'f{f}_test_base']
        ce,ve=evidence[ti],evidence[vi];cc,vc=coverage[ti],coverage[vi];selection=route[vi]
        tg,vg,groupinfo=matching_groups(p31[ti],p31[vi],train_base,test_base,store[f'f{f}_train_host_count'],store[f'f{f}_test_host_count'],cc,vc,p['null_seed']+f)
        def predict(name,tx,vx):
            prob=fit_predict(tx,y[ti],vx,p['auxiliary_model']);scores[name][vi[selection]]=prob[selection]
        predict('M1_HIDDEN_READOUT',tr,te)
        predict('M5_HIDDEN_COVERAGE_ONLY',np.column_stack([tr,cc]),np.column_stack([te,vc]))
        predict('M2_HIDDEN_CONTROL',np.column_stack([tr,cc,ce]),np.column_stack([te,vc,ve]))
        movable_fraction=None
        for repeat in range(repeats):
            seed=p['null_seed']+f*10000+repeat*101
            rt,rv=placebo(ce,cc,vc,ids[ti],ids[vi],seed)
            predict(f'M3_CAPACITY_PLACEBO__r{repeat:02d}',np.column_stack([tr,cc,rt]),np.column_stack([te,vc,rv]))
            st,td,tm,tc=conditional_shuffle(ce,cc,tg,seed+1);sv,vd,vm,changed=conditional_shuffle(ve,vc,vg,seed+2)
            predict(f'M4_CONDITIONAL_SHUFFLE__r{repeat:02d}',np.column_stack([tr,cc,st]),np.column_stack([te,vc,sv]))
            if repeat==0:
                movable_fraction=float(vm[selection].mean());fold_audit.append(dict(fold=f,train_n=len(ti),test_n=len(vi),routed=int(selection.sum()),
                    routed_null_movable_fraction=movable_fraction,routed_shuffled_evidence_changed_fraction=float(changed[selection].mean()),
                    train_null_movable_fraction=float(tm.mean()),feature_width_real=tr.shape[1]+64+124,outer_test_used_for_training=False,
                    inner_hyperparameter_selection=False,train_hidden_in_sample_upstream=True,**groupinfo))
            for split,index,donor,movable in (('TRAIN',ti,td,tm),('TEST',vi,vd,vm)):
                for j in range(len(index)):
                    donors.append(dict(fold=f,repeat=repeat,split=split,sha256=ids[index[j]],donor_sha256=ids[index[donor[j]]],movable=bool(movable[j])))
        print(f'LCIP_PROBE_FOLD_DONE fold={f}/5 null_repeats={repeats} routed_null_movable={movable_fraction:.3f}',flush=True)
    if any(not np.array_equal(s[~route],base[~route]) for s in scores.values()):raise RuntimeError('un-routed final probabilities changed')
    np.savez_compressed(safe(OUTPUT/'probe_predictions.npz'),ids=ids,**scores)
    write_csv(OUTPUT/'null_donors.csv',donors);write_csv(OUTPUT/'fold_training_audit.csv',fold_audit)
    rows=[]
    for i,sid in enumerate(ids):
        rows.append(dict(sha256=sid,fold=int(folds[i]),label=int(y[i]),routed=int(route[i]),stage1_prob=float(base[i]),**{k:float(v[i]) for k,v in scores.items()}))
    write_csv(OUTPUT/'source_predictions.csv',rows)
    ready=sum(r['routed_null_movable_fraction']>=p['minimum_routed_null_movable_fraction'] for r in fold_audit)>=p['minimum_folds_null_ready']
    dump(OUTPUT/'probe_training_audit.json',dict(status='PASS',outer_fold_n=5,auxiliary_training_rows='ALL_OUTER_TRAIN',
        hyperparameter_search=False,nested_upstream_refits=False,global_oof_routed_training_selection=False,conditional_null_matching_ready=ready,
        future_rows_loaded=0,labels_used_for_auxiliary_training=True,independent_unit='APK',permission_features_used=False,
        coordinate_space='separate SAME upstream model per fold',matching_is_approximate_not_exact_conditional_randomization=True))
if __name__=='__main__':main()
