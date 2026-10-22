"""Posthoc within-C1 selection interventions: encode all Hosts with C1; never import C0 hidden coordinates or fit weights."""
from __future__ import annotations
import argparse,dataclasses,math
from collections import defaultdict
import numpy as np
from audit_common import *

REPEATS=20

def weight_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def selected_cache(cache,local,k,weight,torch):
    summary=cache.scalar_summary.clone();summary[0]=cache.score[int(k)]
    summary[2]=-torch.sum(weight*torch.log(weight.clamp_min(1e-8)))/math.log(max(len(weight),2));summary[3]=torch.max(weight)
    return dataclasses.replace(cache,critical=local[int(k)],attended=torch.sum(weight[:,None]*local,0),scalar_summary=summary,attention=weight)

def main():
    if (OUTPUT/'selection_intervention_summary.json').exists():raise RuntimeError('selection audit already exists; do not overwrite')
    members=read_csv(OUTPUT/'private/cohort_key.csv');mc,data=old();v4,_,_=mc.runtime();torch=v4.torch
    from train_stage2 import raw_bags,transform_saved
    # Freeze this BEFORE intervention results exist. The cohort is still
    # retrospective; its labels/outcomes cannot select a deployment rule.
    dump(OUTPUT/'selection_intervention_protocol.json',dict(cohort='same 107 members from frozen case-control audit',retrospective=True,model='same frozen C1 weights, preprocessing, graph, global input and raw Host bag',policies=['ORIGINAL','OLD_CRITICAL_ONLY','OLD_ATTENDED_ONLY','OLD_CRITICAL_AND_ATTENDED'],old_information='C0 Host indices and attention weights only; NEVER C0 embeddings/scalers/hidden/head',critical_summary='use C1 score of selected Host',attended_summary='entropy/max recomputed from overridden attention',null='20 deterministic primary-Role-preserving old-attention permutations plus alternate same-primary-Role critical Host; degenerate controls marked',random_repeats=REPEATS,threshold=.5,training=False,policy_selection_authorized=False,semantic_causality_claim=False,program_execution=False,tolerance=1e-6))
    pos={s:i for i,s in enumerate(data['ids'])};result=defaultdict(lambda:defaultdict(list));null_result=defaultdict(lambda:np.full((3,REPEATS),np.nan));null_valid=defaultdict(lambda:np.zeros((3,REPEATS),bool));rows=[];replay_checks=[];hashchecks=[]
    for window in ['T2','T3','T4']:
        selected=[r for r in members if r['window']==window];ix=np.asarray([pos[r['sha256']] for r in selected]);bags=raw_bags(data,ix)
        for seed_index,seed in enumerate(SEEDS):
            obj=load_checkpoint(BASE/'models/stage2'/(window+'_C1')/(str(seed)+'.pkl'),mc,data);args=argparse.Namespace(**obj['metadata']['args']);vp={s:i for i,s in enumerate(obj['metadata']['validation_sha'])};vi=[vp[r['sha256']] for r in selected];gv=np.asarray(obj['global_validation'][vi],np.float32)
            b=transform_saved(v4,bags,obj['background'],args);graphs=v4.transform_role_graphs(data['s7'][ix],obj['graph_scaler']);graph=v4.graph_tensors(graphs,torch.device('cpu'))
            model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args);model.load_state_dict(obj['state_dict']);model.eval();before=weight_hash(model)
            with torch.no_grad():
                for j,r in enumerate(selected):
                    sid=r['sha256'];trace=read_json(OUTPUT/'private/traces'/(sid+'.json'));old_trace=next(a for a in trace['full_seed_traces'] if a['tag']=='C0R_2017' and a['seed']==seed)
                    region=torch.as_tensor(b[j],dtype=torch.float32);one={k:v[j] for k,v in graph.items()};cache,gg=model.select_local_evidence(one,region);local=model.region_encoder(region[:,:111]);g=torch.as_tensor(gv[j]);zero=torch.zeros(())
                    old_k=int(old_trace['critical_slot']);new_k=int(torch.argmax(cache.score));old_weight=torch.as_tensor(old_trace['attention'],dtype=torch.float32)
                    if len(old_weight)!=len(region) or not np.isclose(float(old_weight.sum()),1,atol=1e-6):raise RuntimeError('old/new frozen Host alignment differs')
                    def predict(c):
                        raw,_,_=model.delta_from_selected_evidence(g,zero,c,gg);return float(torch.sigmoid(raw))
                    original=predict(cache);difference=abs(original-float(obj['probability'][vi[j]]));replay_checks.append(difference)
                    if difference>1e-6:raise RuntimeError('within-model original replay differs '+str(difference))
                    candidates={'ORIGINAL':cache,'OLD_CRITICAL_ONLY':selected_cache(cache,local,old_k,cache.attention,torch),'OLD_ATTENDED_ONLY':selected_cache(cache,local,new_k,old_weight,torch),'OLD_CRITICAL_AND_ATTENDED':selected_cache(cache,local,old_k,old_weight,torch)}
                    for policy,c in candidates.items():
                        p=predict(c);result[sid][policy].append(p);rows.append(dict(sha256=sid,member_id=r['member_id'],window=window,category=r['category'],target=r['target'],seed=seed,policy=policy,probability=p,old_critical_slot=old_k,new_critical_slot=new_k,label=int(r['label'])))
                    counts=bags[j][:,43:74];primary=np.where(counts.sum(1)>0,np.argmax(counts,axis=1),-1);groups=[np.flatnonzero(primary==k) for k in np.unique(primary)];alternatives=np.flatnonzero((primary==primary[old_k])&(np.arange(len(primary))!=old_k))
                    for repeat in range(REPEATS):
                        rng=np.random.RandomState(int(signature([sid,seed,repeat,'MEEA_ROLE_NULL'])[:8],16));weight=np.asarray(old_weight).copy()
                        for group in groups:
                            if len(group)>1:weight[group]=weight[rng.permutation(group)]
                        k=int(rng.choice(alternatives)) if len(alternatives) else old_k
                        valid=bool(len(alternatives)>0 and not np.allclose(weight,np.asarray(old_weight),rtol=0,atol=1e-8));c=selected_cache(cache,local,k,torch.as_tensor(weight),torch)
                        null_result[sid][seed_index,repeat]=predict(c);null_valid[sid][seed_index,repeat]=valid
            after=weight_hash(model);hashchecks.append(dict(window=window,seed=seed,before=before,after=after,unchanged=before==after))
            if before!=after:raise RuntimeError('inference intervention changed model weights or buffers')
            del model,obj,graph
            print('MEEA_SELECTION_INTERVENTION',window,seed,len(selected),flush=True)
    per=[]
    for r in members:
        sid=r['sha256'];y=int(r['label']);out=dict(sha256=sid,member_id=r['member_id'],year=r['year'],category=r['category'],target=r['target'],label=y)
        for policy,prob in result[sid].items():
            p=float(np.mean(prob));out[policy+'_probability']=p;out[policy+'_correct']=int((p>=.5)==y)
        null=np.mean(null_result[sid],axis=0);valid=np.all(null_valid[sid],axis=0);out['role_null_valid_repeats']=int(valid.sum());out['role_null_correct_fraction']=float(np.mean((null[valid]>=.5)==y)) if valid.any() else None
        per.append(out)
    policies=['ORIGINAL','OLD_CRITICAL_ONLY','OLD_ATTENDED_ONLY','OLD_CRITICAL_AND_ATTENDED'];summary=[]
    for name,subset in [('REGRESSED_FN',[r for r in per if r['category']=='REGRESSED_FN']),('PERSISTENT_TARGETS',[r for r in per if r['target']=='True' and r['category']!='REGRESSED_FN']),('MATCHED_CORRECT_CONTROLS',[r for r in per if r['target']=='False'])]:
        baseline=sum(r['ORIGINAL_correct'] for r in subset)
        for policy in policies:
            summary.append(dict(cohort=name,n=len(subset),policy=policy,correct=sum(r[policy+'_correct'] for r in subset),repaired=sum(not r['ORIGINAL_correct'] and r[policy+'_correct'] for r in subset),damaged=sum(r['ORIGINAL_correct'] and not r[policy+'_correct'] for r in subset),net=sum(r[policy+'_correct'] for r in subset)-baseline))
        valid=[r['role_null_correct_fraction'] for r in subset if r['role_null_correct_fraction'] is not None]
        summary.append(dict(cohort=name,n=len(subset),policy='ROLE_MATCHED_RANDOM_JOINT',expected_correct_sum_valid_APKs=float(sum(valid)),valid_APKs=len(valid),note='mean across 20 repeats per APK, not independent APK replicates; degenerate controls excluded'))
    write_csv(OUTPUT/'private/selection_intervention_per_seed.csv',rows);write_csv(OUTPUT/'private/selection_intervention_per_apk.csv',per);write_csv(OUTPUT/'selection_intervention_results.csv',summary)
    dump(OUTPUT/'selection_intervention_summary.json',dict(status='WITHIN_MODEL_SELECTION_DIAGNOSTIC_COMPLETE',groups=summary,max_original_probability_difference=max(replay_checks),all_weight_hashes_unchanged=all(r['unchanged'] for r in hashchecks),weight_checks=hashchecks,independent_semantic_causes_confirmed=0,training_authorized=False,boundary='posthoc computational contribution of a chosen selection policy; not runtime intervention, true Host label, or a validated correction rule'))
    integrity();print('MEEA_SELECTION_DIAGNOSTIC_DONE',flush=True)

if __name__=='__main__':main()
