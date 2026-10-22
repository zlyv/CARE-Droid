"""Freeze reused-asset hashes and replay all nine saved Direct-C43 checkpoints."""
from __future__ import annotations
import argparse, sys, time
from collections import Counter
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from diagnostic_common import *

def init_manifest():
    if OUTPUT.exists() and any(p.name not in ('tmp','cache') for p in OUTPUT.iterdir()):
        raise RuntimeError('output has experiment files; choose a new version, do not overwrite')
    paths={BASE/'data.npz':'prepared development data',BASE/'feature_rows.pkl':'prepared global features',BASE/'dataset_audit.json':'dataset contract',BASE/'final_decision.json':'original final status',BASE/'per_apk_predictions.csv':'original per APK result',PROJECT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py':'existing three backgrounds',PROJECT/'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py':'Role and Host evidence',PROJECT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py':'Direct C43 forward',PROJECT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py':'original architecture'}
    for p in (PROJECT/'scripts/care-multiyear-development-v1').glob('*.py'):paths[p]='original rolling protocol'
    for p in PREVIOUS.glob('*.py'):paths[p]='read-only diagnostic helper'
    for tag in TAGS:
        for seed in SEEDS:paths[BASE/'models/stage2'/tag/(str(seed)+'.pkl')]='frozen Stage-II checkpoint'
    mc,v4,args,data=original()
    for m in list(sys.modules.values()):
        q=getattr(m,'__file__',None)
        if q and str(PROJECT) in str(q) and Path(q).suffix=='.py' and HERE not in Path(q).resolve().parents:paths[Path(q).resolve()]='loaded dependency'
    manifest=[asset(p,purpose) for p,purpose in sorted(paths.items(),key=lambda x:str(x[0]))]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT/'baseline_asset_manifest_before.json',manifest)
    return mc,v4,args,data

def main():
    start=time.monotonic();mc,v4,args,data=init_manifest()
    from train_stage2 import raw_bags,transform_saved
    from protocol import mean_opposition
    cases=read_csv(BASE/'per_apk_predictions.csv')
    by={(r['window']+'_'+r['arm'],r['sha256']):r for r in cases}
    ix={str(s):i for i,s in enumerate(data['ids'])};rows=[];prov=[];replay=[];maxdiff=0.
    for tag in TAGS:
        saved=[checkpoint(BASE/'models/stage2'/tag/(str(seed)+'.pkl'),mc,data) for seed in SEEDS]
        meta=saved[0]['metadata'];va=np.array([ix[s] for s in meta['validation_sha']]);raw=raw_bags(data,va);ensemble=[]
        for seed,obj in zip(SEEDS,saved):
            m=obj['metadata']
            if m['validation_sha']!=meta['validation_sha'] or m['train_sha']!=meta['train_sha']:raise RuntimeError('checkpoint SHA ordering differs')
            a=argparse.Namespace(**m['args']);bags=transform_saved(v4,raw,obj['background'],a)
            graph=v4.transform_role_graphs(data['s7'][va],obj['graph_scaler'])
            model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',a);model.load_state_dict(obj['state_dict']);model.eval()
            p,_=v4.predict_model(model,np.asarray(obj['global_validation'],np.float32),np.full(len(va),.5),graph,bags,v4.torch.device('cpu'))
            diff=float(np.max(np.abs(p-obj['probability'])));maxdiff=max(maxdiff,diff)
            if diff>1e-6:raise RuntimeError(f'checkpoint replay mismatch {tag}/{seed}: {diff}')
            ensemble.append(p)
            train_years=sorted(set(int(y) for y in data['years'][[ix[s] for s in m['train_sha']]]))
            prov.append(dict(tag=tag,seed=seed,checkpoint=str(BASE/'models/stage2'/tag/(str(seed)+'.pkl')),checkpoint_sha256=digest(BASE/'models/stage2'/tag/(str(seed)+'.pkl')),train_n=len(m['train_sha']),validation_n=len(va),train_years=train_years,validation_year=int(data['years'][va[0]]),selected_epochs=obj['audit']['selected_epochs'],max_replay_diff=diff,host_width=int(bags[0].shape[1]),background_shared_shape=list(np.shape(obj['background']['state']['shared_centers'])),background_benign_shape=list(np.shape(obj['background']['state']['benign_centers']))))
            print('CLD_REPLAY',tag,seed,diff,flush=True)
        ensemble=np.mean(ensemble,axis=0)
        original_rows=[]
        for j,sid in enumerate(meta['validation_sha']):
            r=by.get((tag.replace('_C1','')+'_U3_FULL_ROLLING_C1',sid))
            if r is None:raise RuntimeError('missing original prediction '+sid)
            if abs(float(r['expert_probability'])-ensemble[j])>1e-6:raise RuntimeError('ensemble probability mismatch '+sid)
            y=int(r['label']);s=int(r['stage1_prediction']);routed=int(r['routed']);ep=int(ensemble[j]>=.5);final=ep if routed else s
            if final!=int(r['final_prediction']):raise RuntimeError('final prediction mismatch '+sid)
            category=('UNROUTED_ERROR' if s!=y else 'UNROUTED_CORRECT') if not routed else ('FIXED_FN' if y else 'FIXED_FP') if s!=y and ep==y else ('UNFIXED_FN' if y else 'UNFIXED_FP') if s!=y else ('BROKEN_TN' if y==0 else 'BROKEN_TP') if ep!=y else 'STABLE_CORRECT'
            row=dict(tag=tag,sha256=sid,year=int(r['year']),label=y,stage1_probability=float(r['stage1_probability']),stage1_prediction=s,mean_opposition_score='',routed=routed,expert_probability=float(ensemble[j]),expert_prediction=ep,final_probability=float(r['final_probability']),final_prediction=final,category=category,stage1_correct=int(s==y),expert_correct=int(ep==y),fixed=int(category.startswith('FIXED')),broken=int(category.startswith('BROKEN')),unfixed=int(category.startswith('UNFIXED')),stable_correct=int(category=='STABLE_CORRECT'))
            rows.append(row);original_rows.append(row)
        # Reconstruct unchanged MeanOpposition scores for the complete held-out year.
        score=mean_opposition(data['historical_members'][va],data['historical_base'][va])
        for row,value in zip(original_rows,score):row['mean_opposition_score']=float(value)
        routed=[r for r in original_rows if r['routed']];count=Counter(r['category'] for r in routed)
        yy=np.array([r['label'] for r in original_rows]);ff=np.array([r['final_prediction'] for r in original_rows])
        replay.append(dict(tag=tag,year=int(data['years'][va[0]]),n=len(va),routed=len(routed),BA=float(balanced_accuracy_score(yy,ff)),stage1_errors=sum(r['stage1_correct']==0 for r in original_rows),routed_stage1_errors=sum(r['stage1_correct']==0 for r in routed),fixed=sum(r['fixed'] for r in routed),broken=sum(r['broken'] for r in routed),unfixed=sum(r['unfixed'] for r in routed),categories=dict(count)))
    write_json(OUTPUT/'checkpoint_provenance.json',prov)
    write_json(OUTPUT/'baseline_replay.json',dict(status='PASS',tolerance=1e-6,max_probability_difference=maxdiff,checkpoint_replays=len(TAGS)*len(SEEDS),years=replay,development_data=True))
    write_csv(OUTPUT/'per_apk_error_contract.csv',rows)
    write_json(OUTPUT/'audit_runtime.json',dict(seconds=time.monotonic()-start,phase='asset inventory and 9-checkpoint replay'))
    print('CLD_BASELINE_PASS',[(r['year'],r['fixed'],r['broken'],r['unfixed']) for r in replay],flush=True)
if __name__=='__main__':main()
