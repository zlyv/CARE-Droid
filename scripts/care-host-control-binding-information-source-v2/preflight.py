#!/usr/bin/env python3
from common import *

def main():
    import scipy,sklearn
    obs_names=('final_decision.json','binding_coverage.json','baseline_integrity_check.json',
        'baseline_asset_manifest_after.json','environment_inventory.json','null_repeat_apk_metrics.csv','protocol.json')
    for name in obs_names:
        if not (OBS/name).is_file():raise RuntimeError('missing passed observability file '+str(OBS/name))
    dec=read_json(OBS/'final_decision.json');intact=read_json(OBS/'baseline_integrity_check.json')
    if dec.get('status')!='HOST_CONTROL_BINDING_OBSERVABLE' or intact.get('status')!='PASS' or not dec.get('permit_specifying_separate_source_information_test') or dec.get('future_rows_loaded')!=0 or dec.get('labels_used') is not False:
        raise RuntimeError('observability gate not passed under label-free source scope')
    env=read_json(OBS/'environment_inventory.json');parent=Path(env['parent']).resolve()
    raw_core=PROJECT/'scripts/care-host-control-binding-observability-v1/binding_core.py'
    old_reference=PROJECT/'scripts/care-local-control-constraint-incremental-source-v1/common.py'
    if not raw_core.is_file() or sha256(raw_core)!=sha256(HERE/'binding_core.py'):raise RuntimeError('binding implementation differs from passed audit')
    if not old_reference.is_file() or sha256(old_reference)!=sha256(HERE/'reference_core.py'):raise RuntimeError('reference implementation differs from prior source test')
    prior=read_json(parent/'protocol.json')
    if prior.get('auxiliary_model')!=protocol()['auxiliary_model'] or prior.get('threshold')!=.5:raise RuntimeError('auxiliary parameters no longer match frozen aggregate comparator')
    required=[(OBS/n,'passed observability '+n) for n in obs_names]
    required += [(parent/n,'read-only incremental '+n) for n in ('control_features.npz','probe_predictions.npz','feature_scan_status.csv',
        'selection_label_free.csv','engine_identity.json','protocol.json','feature_validation.json','baseline_integrity_check.json')]
    required += [(FORMAL,'formal SVM source reference'),(DIRECT,'Direct-C43 source reference'),(STORE,'frozen fold representation store'),
        (REPLAY,'frozen nested C43 replay validation'),(raw_core,'passed binding source'),(old_reference,'prior reference source')]
    old=read_json(OBS/'baseline_asset_manifest_after.json')
    # A record of the previously passed data is not enough: check current contents.
    required += [(Path(r['absolute_path']),r['purpose']) for r in old]
    required += [(Path(env[k]),'observability input '+k) for k in ('endpoint','compact','manifest')]
    missing=[str(p) for p,purpose in required if not p.is_file()]
    if missing:raise RuntimeError('missing source input '+repr(missing[:10]))
    before=[];seen=set()
    for p,purpose in required:
        key=str(p.resolve())
        if key in seen:continue
        seen.add(key);before.append(asset(p,purpose))
        if len(before)%200==0:print(f'HBI_ASSET_BEFORE {len(before)}',flush=True)
    dump(OUTPUT/'baseline_asset_manifest_before.json',before)
    by={r['absolute_path']:r for r in before}
    for r in old:
        now=by[str(Path(r['absolute_path']).resolve())]
        if any(now[k]!=r[k] for k in ('size_bytes','mtime_ns','sha256')):raise RuntimeError('source input changed since observability audit '+r['absolute_path'])
    replay=read_json(REPLAY)
    if replay.get('status')!='PASS' or float(replay.get('max_abs_probability_difference',1))>1e-4 or not replay.get('checks',{}).get('nested_calls==5') or not replay.get('checks',{}).get('fold_calls==5'):raise RuntimeError('upstream replay/nested lineage failed')
    ids,y,folds,base,route,expert,final=load_reference();store=load_store(ids,y,folds)
    ref=metrics(y,base,final,route)
    if (ref['fixed'],ref['broken'],ref['net'])!=(49,17,32) or route.sum()!=200 or ((base>=.5)!=y).sum()!=97:raise RuntimeError('frozen SVM/Direct-C43 reference changed')
    rows=read_csv(parent/'selection_label_free.csv');membership={norm(r['sha256']):(int(r['fold']),int(r['routed'])) for r in rows}
    if len(rows)!=799 or set(membership)!=set(ids) or any(membership[s]!=(int(folds[i]),int(route[i])) for i,s in enumerate(ids)):raise RuntimeError('parent source membership differs')
    if [int((folds==f).sum()) for f in range(1,6)]!=[160,160,160,160,159] or [int((route&(folds==f)).sum()) for f in range(1,6)]!=[40]*5:raise RuntimeError('formal fold or route sizes changed')
    for f in range(1,6):
        if not np.array_equal(store[f'f{f}_test_base'],base[folds==f]):raise RuntimeError('stored outer-test Stage-I probability differs')
    with np.load(parent/'probe_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(np.asarray([norm(v) for v in z['ids']]),ids) or not np.array_equal(z['M0_DIRECT_C43'],final):raise RuntimeError('previous readout reference SHA or baseline mismatch')
        for key in ('M2_HIDDEN_CONTROL','M1_HIDDEN_READOUT'):
            a=np.asarray(z[key],float)
            if a.shape!=(799,) or not np.isfinite(a).all() or not np.array_equal(a[~route],base[~route]):raise RuntimeError('previous comparator output contract')
    obs_protocol=read_json(OBS/'protocol.json')
    for k,v in (('null_repeats',20),('null_seed',20261014),('context_width',74),('labels_used',False)):
        if obs_protocol.get(k)!=v:raise RuntimeError('passed binding contract differs: '+k)
    dump(OUTPUT/'protocol.json',protocol())
    dump(OUTPUT/'input_paths.json',dict(parent=str(parent),**{k:env[k] for k in ('endpoint','compact','manifest')}))
    dump(OUTPUT/'preflight.json',dict(status='PASS',source_rows=799,routed=200,frozen_reference=ref,read_only_inputs=len(before),
        no_C43_retraining=True,auxiliary_parameters_fixed=True,future_rows_loaded=0))
    dump(OUTPUT/'environment_inventory.json',dict(python=sys.version,python_executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,GPU_required=False,new_DEX_scans=0,projection_width=64,source_rows=799,routed=200,
        train_hidden_in_sample_upstream=True,auxiliary_inner_search=False,upstream_inner_refits=False,
        feature_construction_labels_used=False,auxiliary_training_labels_used=True,future_rows_loaded=0))
    print('HBI_PREFLIGHT_PASS source=799 route=200 projection=64 C43_retraining=false',flush=True)
if __name__=='__main__':main()
