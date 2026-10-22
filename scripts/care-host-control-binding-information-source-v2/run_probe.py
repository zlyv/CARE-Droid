#!/usr/bin/env python3
from common import *
from model_core import fit_predict,placebo

def main():
    v=read_json(OUTPUT/'feature_validation.json')
    if not v.get('projection_ready'):
        dump(OUTPUT/'probe_training_audit.json',dict(status='NOT_RUN_PROJECTION_COVERAGE_INSUFFICIENT',model_training_performed=False,future_rows_loaded=0));return
    ids,y,folds,base,route,expert,final=load_reference();store=load_store(ids,y,folds);p=protocol()
    with np.load(OUTPUT/'binding_features.npz',allow_pickle=False) as z:
        fids=np.asarray([norm(s) for s in z['ids']]);e=np.asarray(z['evidence']);cov=np.asarray(z['coverage'])
        binding=np.asarray(z['binding']);shuffled=np.asarray(z['shuffle']);support=np.asarray(z['support'],bool)
    ix={s:i for i,s in enumerate(fids)}
    if len(ix)!=799 or set(ix)!=set(ids):raise RuntimeError('binding SHA membership differs')
    order=[ix[s] for s in ids];e=e[order];cov=cov[order];binding=binding[order];shuffled=shuffled[:,order];support=support[order]
    if binding.shape!=(799,64) or shuffled.shape!=(20,799,64):raise RuntimeError('binding width/repeats changed')
    parent=Path(read_json(OUTPUT/'input_paths.json')['parent'])
    with np.load(parent/'probe_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids):raise RuntimeError('previous comparator order differs')
        reference_aggregate=np.asarray(z['M2_HIDDEN_CONTROL'],float);reference_hidden=np.asarray(z['M1_HIDDEN_READOUT'],float)
    scores={a:base.copy() for a in ('B1_HIDDEN_AGGREGATE','B2_HOST_BINDING_PRIMARY','B5_HIDDEN_READOUT')}
    scores['B0_DIRECT_C43']=final.copy()
    for r in range(p['null_repeats']):
        for a in ('B3_WITHIN_APK_SHUFFLE','B4_CAPACITY_PLACEBO'):scores[f'{a}__r{r:02d}']=base.copy()
    audits=[];replay_folds=[];max_agg=0.;max_hidden=0.
    # First complete the replay pass; fit no new binding arm if it fails.
    for fold in range(1,6):
        ti=store[f'f{fold}_train_index'].astype(int);vi=store[f'f{fold}_test_index'].astype(int)
        tr=store[f'f{fold}_train_hidden'];te=store[f'f{fold}_test_hidden'];select=route[vi]
        tx=np.column_stack([tr,cov[ti],e[ti]]);vx=np.column_stack([te,cov[vi],e[vi]])
        h=fit_predict(tr,y[ti],te,p['auxiliary_model']);a=fit_predict(tx,y[ti],vx,p['auxiliary_model'])
        scores['B1_HIDDEN_AGGREGATE'][vi[select]]=a[select];scores['B5_HIDDEN_READOUT'][vi[select]]=h[select]
        da=float(np.max(np.abs(a[select]-reference_aggregate[vi[select]])))
        dh=float(np.max(np.abs(h[select]-reference_hidden[vi[select]])))
        max_agg=max(max_agg,da);max_hidden=max(max_hidden,dh)
        replay_folds.append(dict(fold=fold,routed=int(select.sum()),hidden_train_dtype=str(tr.dtype),
            hidden_test_dtype=str(te.dtype),aggregate_train_dtype=str(tx.dtype),aggregate_test_dtype=str(vx.dtype),
            aggregate_max_probability_difference=da,hidden_max_probability_difference=dh))
        print(f'HBI_REPLAY_FOLD fold={fold} hidden_dtype={tr.dtype} aggregate_dtype={tx.dtype} aggregate_diff={da:.12g} hidden_diff={dh:.12g}',flush=True)
    good=max(max_agg,max_hidden)<=p['parent_replay_tolerance']
    write_csv(OUTPUT/'aggregate_replay_fold_validation.csv',replay_folds)
    dump(OUTPUT/'aggregate_replay_validation.json',dict(status='PASS' if good else 'FAIL',max_aggregate_probability_difference=max_agg,
        max_hidden_probability_difference=max_hidden,tolerance=p['parent_replay_tolerance'],routed_predictions_compared=200,
        precision_strategy='preserve original input dtype; no tolerance relaxation',fold_details=replay_folds))
    if not good:raise RuntimeError('previous aggregate/readout not reproduced; binding fits prohibited')
    print('HBI_AGGREGATE_REPLAY_PASS',flush=True)
    for fold in range(1,6):
        ti=store[f'f{fold}_train_index'].astype(int);vi=store[f'f{fold}_test_index'].astype(int)
        tr=store[f'f{fold}_train_hidden'];te=store[f'f{fold}_test_hidden'];select=route[vi]
        tx=np.column_stack([tr,cov[ti],e[ti]]);vx=np.column_stack([te,cov[vi],e[vi]])
        def predict(name,btr,bte):
            out=fit_predict(np.column_stack([tx,btr]),y[ti],np.column_stack([vx,bte]),p['auxiliary_model'])
            scores[name][vi[select]]=out[select]
        predict('B2_HOST_BINDING_PRIMARY',binding[ti],binding[vi])
        for r in range(p['null_repeats']):
            predict(f'B3_WITHIN_APK_SHUFFLE__r{r:02d}',shuffled[r,ti],shuffled[r,vi])
            pt,pv=placebo(binding[ti],ids[ti],ids[vi],support[ti],support[vi],p['placebo_seed']+fold*10000+r*101)
            predict(f'B4_CAPACITY_PLACEBO__r{r:02d}',pt,pv)
        audits.append(dict(fold=fold,train_n=len(ti),test_n=len(vi),routed=int(select.sum()),
            aggregate_input_width=tx.shape[1],real_shuffle_placebo_input_width=tx.shape[1]+64,
            train_rows='ALL_OUTER_TRAIN',outer_test_used_for_training=False,inner_hyperparameter_search=False,
            upstream_training_hidden_in_sample=True,train_test_coordinates_same_upstream_model=True,
            countsketch_fitted_to_labels=False,placebo_test_moments_used=False,null_repeat_count=20))
        print(f'HBI_PROBE_FOLD_DONE fold={fold}/5 repeats=20',flush=True)
    if any(not np.array_equal(a[~route],base[~route]) for a in scores.values()):raise RuntimeError('un-routed predictions changed')
    np.savez_compressed(safe(OUTPUT/'probe_predictions.npz'),ids=ids,**scores)
    write_csv(OUTPUT/'source_predictions.csv',[dict(sha256=s,fold=int(folds[i]),label=int(y[i]),routed=int(route[i]),stage1_prob=float(base[i]),
        **{a:float(v[i]) for a,v in scores.items()}) for i,s in enumerate(ids)])
    write_csv(OUTPUT/'fold_training_audit.csv',audits)
    dump(OUTPUT/'probe_training_audit.json',dict(status='PASS',model_training_performed=True,folds=5,auxiliary_hyperparameter_search=False,
        upstream_inner_refits=False,upstream_hidden_train_in_sample=True,unit='APK',within_APK_null_not_independent_sample=True,
        same_feature_width_and_parameters_for_real_and_null=True,formal_stage1_changed=False,formal_stage2_changed=False,future_rows_loaded=0))
if __name__=='__main__':main()
