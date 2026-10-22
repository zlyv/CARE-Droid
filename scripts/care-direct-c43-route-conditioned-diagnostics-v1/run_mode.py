from __future__ import annotations
import argparse,os,sys,time
import numpy as np
from diag_common import *
from diagnostic_runtime import create_runtime

def aligned_p31(path,ids):
    with np.load(path,allow_pickle=False) as a:
        index={norm(s):i for i,s in enumerate(a['sha256'])}; x=np.asarray(a['role_prevalence'],float)
        if x.ndim!=2 or x.shape[1]!=31 or any(s not in index for s in ids):raise RuntimeError('P31 SHA/width mismatch')
        out=x[[index[s] for s in ids]]
    if not np.isfinite(out).all():raise RuntimeError('P31 nonfinite')
    return out

def context_seed(mode,year,fold,args):
    if mode=='frozen':return int(args.split_seed)+1701,20261017
    return int(args.split_seed)+int(year)*10000+int(fold)*100+1701,20261017+int(fold)

def replay_checks(expert,reference):
    diff=float(np.max(np.abs(reference-expert)));same=bool(np.array_equal(reference>=.5,expert>=.5))
    return dict(status='PASS' if diff<=1e-4 and same else 'FAIL',max_probability_difference=diff,all_decisions_exact=same)

class Expert:
    def __init__(self,mode,v4,rt,w,ta):
        self.mode=mode;self.v4=v4;self.rt=rt;self.w=w;self.ta=ta;self.original=v4.fit_predict_variant;self.count=0;self.source_ids=None
        self.contexts={};self.resolved=read_json(FORMAL_DIAG/'resolved_inputs.json')
        if mode=='within':
            rows=read_csv(FORMAL_DIAG/'within_2018_2019/predictions.csv')+read_csv(FORMAL_DIAG/'within_2020_2021/predictions.csv')
            self.meta={norm(r['sha256']):dict(year=int(r['year']),fold=int(r['fold']),label=int(r['label'])) for r in rows}
    def capture_inputs(self,result):
        manifest,source,clean=result[:3];self.source_ids=np.asarray([norm(r['sha256']) for r in source]);self.source_y=np.asarray([r['label'] for r in manifest],int)
        target=[r for r in clean if str(r.get('metadata_year','')).strip() in set(map(str,YEARS))]
        self.target_ids=np.asarray([norm(r['sha256']) for r in target]);self.target_year=np.asarray([int(r['metadata_year']) for r in target])
        if len(self.source_ids)!=799 or len(self.target_ids)!=1989 or len(set(self.target_ids))!=1989:raise RuntimeError('source/future cohort/order changed')
        if set(self.source_ids)&set(self.target_ids):raise RuntimeError('exact source/future SHA overlap; temporal diagnostic requires explicit review')
        if self.mode=='within' and set(self.meta)!=set(self.target_ids):raise RuntimeError('within-year SHA membership differs')
    def fit(self,gt,gv,s7t,s7v,bt,bv,baset,basev,y,variant,seeds,args,device):
        if self.source_ids is None:raise RuntimeError('uncaptured original feature order')
        if getattr(self.v4.RoleGraphConditionedEvidence,'correction_semantics',None)!='direct_stage2_logit_without_stage1_logit_input':
            raise RuntimeError('active expert is not Direct-C43; bounded or residual head is prohibited')
        self.count+=1;t=time.monotonic()
        if self.mode=='frozen':
            if self.count!=1:raise RuntimeError('one frozen training context required')
            year=2017;fold=0;tr_ids=self.source_ids;te_ids=self.target_ids;truth=self.source_y
            p31=aligned_p31(self.rt.family_module('common').SOURCE_P31,tr_ids);tag='train2017'
        else:
            year=YEARS[(self.count-1)//5];fold=(self.count-1)%5+1;ids=np.asarray(sorted(s for s,m in self.meta.items() if m['year']==year))
            tr_ids=np.asarray([s for s in ids if self.meta[s]['fold']!=fold]);te_ids=np.asarray([s for s in ids if self.meta[s]['fold']==fold])
            truth=np.asarray([self.meta[s]['label'] for s in tr_ids],int);p31=aligned_p31(self.resolved['future_p31_cache'],tr_ids);tag=f'{year}_f{fold}'
        if not np.array_equal(y,truth) or len(gt)!=len(tr_ids) or len(gv)!=len(te_ids) or set(tr_ids)&set(te_ids):raise RuntimeError('training context label/shape/SHA contract')
        if args.keep_weight!=1 or args.fix_weight!=1 or args.keep_consistency!=0:raise RuntimeError('Direct equal-BCE reference args changed')
        router_seed,shuffle_seed=context_seed(self.mode,year,fold,args)
        from sklearn.model_selection import StratifiedKFold
        fit_committee=self.rt.family_module('stage1_core').fit_calibrated_committee
        member=np.full((len(y),4),np.nan);assignment=np.full(len(y),-1,int);splits=[]
        for j,(fit,hold) in enumerate(StratifiedKFold(4,shuffle=True,random_state=router_seed).split(p31,y),1):
            member[hold]=fit_committee(p31[fit],y[fit],p31[hold],router_seed+100000*j,inner_folds=3,jobs=args.jobs);assignment[hold]=j
            splits.append(dict(inner_fold=j,fit_sha=tr_ids[fit].tolist(),hold_sha=tr_ids[hold].tolist()))
        if not np.isfinite(member).all() or not np.allclose(member[:,3],baset,atol=1e-14,rtol=0):raise RuntimeError('new committee SVM does not replay runner nested SVM')
        weights,train_route,wa=self.w.make_weights(y,tr_ids,member,np.asarray([len(b) for b in bt]),seed=shuffle_seed,groups=assignment)
        if wa['changed_fraction']<.10:raise RuntimeError('conditional shuffle mixing below frozen 10% threshold')
        save_npz(OUTPUT/'training_routes'/self.mode/(tag+'.npz'),ids=tr_ids,member=member,route=train_route,inner_fold=assignment,**weights)
        dump(OUTPUT/'training_routes'/self.mode/(tag+'.json'),dict(router_seed=router_seed,shuffle_seed=shuffle_seed,train_sha=tr_ids.tolist(),test_sha=te_ids.tolist(),crossfit=splits,weight_audit=wa))
        inputs=(gt,gv,s7t,s7v,bt,bv,baset,basev,y,variant)
        expert,diagnostic,audits=self.original(*inputs,seeds,args,device)
        self.ta.validate_epoch_records(audits,seeds,fold)
        audits=[dict(a,optimizer_steps=int(a['selected_epochs'])*int(np.ceil(len(y)/args.batch_size))) for a in audits]
        pred={'B0_ORIGINAL':expert}; records={'B0_ORIGINAL':audits}
        if self.mode=='frozen':
            old={norm(r['sha256']):r for r in read_csv(PROJECT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv')}
            ref=np.asarray([float(old[s]['direct_c43_prob']) for s in te_ids]);checks=replay_checks(expert,ref)
            dump(OUTPUT/'frozen_B0_replay.json',checks)
            if checks['status']!='PASS':raise RuntimeError('historical frozen B0 replay failed; weighted future fits prohibited')
        fp=self.ta.fingerprint([gt,gv,s7t,s7v,baset,basev,y,*bt,*bv])
        for arm in ARMS[1:]:
            p,_,a,ps=self.ta.fit_fixed_epochs(self.v4,inputs,weights[arm],audits,seeds,args,device)
            if self.ta.fingerprint([gt,gv,s7t,s7v,baset,basev,y,*bt,*bv])!=fp:raise RuntimeError('input changed during fit')
            for aa,bb in zip(a,audits):
                if aa['parameter_n']!=bb['parameter_n'] or aa['optimizer_steps']!=bb['optimizer_steps']:raise RuntimeError('capacity/exposure mismatch')
            pred[arm]=p;records[arm]=a
            save_npz(OUTPUT/'seed_sensitivity'/self.mode/(tag+'_'+arm+'.npz'),ids=te_ids,probabilities=ps)
            print(f'RCTD_ARM_CONTEXT_DONE mode={self.mode} context={tag} arm={arm}',flush=True)
        save_npz(OUTPUT/'experts'/self.mode/(tag+'.npz'),ids=te_ids,base=basev,**pred)
        dump(OUTPUT/'fit_audits'/self.mode/(tag+'.json'),dict(records=records,seeds=list(map(int,seeds)),train_n=len(y),test_n=len(te_ids),seconds=time.monotonic()-t,
            future_labels_used_for_frozen_fit=False,within_training_labels_used=self.mode=='within',test_labels_passed_to_model=False,input_sha256=fp))
        self.contexts[tag]=dict(ids=te_ids,base=np.asarray(basev),pred=pred)
        print(f'RCTD_CONTEXT_DONE mode={self.mode} context={tag} seconds={time.monotonic()-t:.1f}',flush=True)
        return expert,diagnostic,audits

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['frozen','within'],required=True);cli=p.parse_args()
    core,rt,w,ta,v4,adapter,runner=create_runtime(cli.mode)
    state=Expert(cli.mode,v4,rt,w,ta);v4.fit_predict_variant=state.fit
    original=v4.audit.load_inputs
    def capture(args):
        result=original(args);state.capture_inputs(result);return result
    v4.audit.load_inputs=capture
    options=['--device',os.environ.get('RCTD_DEVICE','cuda'),'--jobs',os.environ.get('RCTD_JOBS','4')]
    sys.argv=[sys.argv[0]]+(['--arm','SVM'] if cli.mode=='within' else [])+options
    runner.main()
    if state.count!=(1 if cli.mode=='frozen' else 20):raise RuntimeError('incomplete diagnostic contexts')
    path=OUTPUT/'runner-frozen/predictions_label_free.csv' if cli.mode=='frozen' else OUTPUT/'runner-within/within/SVM/predictions.csv'
    rows=read_csv(path);by={norm(r['sha256']):r for r in rows}
    if len(rows)!=1989 or len(by)!=1989:raise RuntimeError('runner output cohort')
    result=[]
    for tag,context in state.contexts.items():
        for i,s in enumerate(context['ids']):
            r=by[s];base=float(r['stage1_prob']);route=int(r['routed'])
            if abs(base-context['base'][i])>1e-14:raise RuntimeError('Stage-I output alignment')
            for arm in ARMS:
                ep=float(context['pred'][arm][i]);result.append(dict(sha256=s,year=int(r['year']),fold=int(r.get('fold',0)),arm=arm,stage1_prob=base,routed=route,
                    expert_prob=ep,final_prob=ep if route else base))
    write_csv(OUTPUT/(cli.mode+'_predictions_label_free.csv'),result)
    dump(OUTPUT/(cli.mode+'_completion.json'),dict(status='PASS',contexts=state.count,predictions=len(result),future_used_for_method_selection=False,
        mode=cli.mode,source_status_retained='ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED'))
    print('RCTD_MODE_DONE mode='+cli.mode,flush=True)
if __name__=='__main__':main()
