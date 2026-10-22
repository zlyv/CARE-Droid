from __future__ import annotations
import math
import numpy as np
from pilot_common import *

def fast_auc(y,p):
    from scipy.stats import rankdata
    n1=int(np.sum(y)); n0=len(y)-n1
    if n0==0 or n1==0: return None
    ranks=rankdata(p,method='average')
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def primary_values(y,base,p,route):
    final=np.where(route,p,base)
    ba=float(.5*(np.mean(final[y==0]<.5)+np.mean(final[y==1]>=.5)))
    n=route&(base<.5); m=route&(base>=.5)
    return np.asarray([ba,fast_auc(y[n],p[n]),fast_auc(y[m],p[m])],float)

def paired_bootstrap(y,folds,base,route,pred,config):
    groups={}
    for i,key in enumerate(zip(folds.tolist(),y.tolist(),route.tolist(),(base>=.5).tolist())): groups.setdefault(key,[]).append(i)
    groups=[np.asarray(v,int) for v in groups.values()]
    rng=np.random.RandomState(config['bootstrap_seed']); samples=[]
    for _ in range(config['bootstrap_repeats']):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        real=primary_values(y[ix],base[ix],pred['B1_ROUTE_WEIGHTED'][ix],route[ix])
        samples.append([real-primary_values(y[ix],base[ix],pred[a][ix],route[ix]) for a in ('B0_ORIGINAL','B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE')])
    a=np.asarray(samples); tail=config['simultaneous_alpha']/(2*config['primary_family_size'])
    result=[]; value=primary_values(y,base,pred['B1_ROUTE_WEIGHTED'],route)
    for j,arm in enumerate(('B0_ORIGINAL','B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE')):
        diff=value-primary_values(y,base,pred[arm],route)
        for k,name in enumerate(config['primary_endpoints']):
            lo,hi=np.quantile(a[:,j,k],[tail,1-tail]); l95,h95=np.quantile(a[:,j,k],[.025,.975])
            result.append(dict(comparison='B1-'+arm,endpoint=name,difference=float(diff[k]),
                ci95_low=float(l95),ci95_high=float(h95),family_adjusted_low=float(lo),family_adjusted_high=float(hi),
                family_size=config['primary_family_size'],bootstrap_repeats=config['bootstrap_repeats']))
    return result

def decide(pooled,fold_metrics,contrasts,p):
    criteria={}; targets=('B0_ORIGINAL','B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE')
    sides={name:True for name in ('FN_vs_TN_AUC','TP_vs_FP_AUC')}
    for target in targets:
        frows=[(r,next(t for t in fold_metrics if t['arm']==target and t['fold']==r['fold'])) for r in fold_metrics if r['arm']=='B1_ROUTE_WEIGHTED']
        criteria['BA_positive_folds_vs_'+target]=sum(r['BA']>b['BA']+1e-12 for r,b in frows)>=p['minimum_positive_folds']
        for endpoint,threshold in zip(p['primary_endpoints'],(p['minimum_BA_gain'],p['minimum_conditional_AUC_gain'],p['minimum_conditional_AUC_gain'])):
            c=next(c for c in contrasts if c['comparison']=='B1-'+target and c['endpoint']==endpoint)
            passed=c['difference']>=threshold and c['family_adjusted_low']>0
            criteria[endpoint+'_gain_and_interval_vs_'+target]=bool(passed)
            if endpoint in sides:
                evaluated=[(r,b) for r,b in frows if r[endpoint] is not None and b[endpoint] is not None]
                consistent=len(evaluated)>=p['minimum_evaluable_folds_per_side'] and sum(r[endpoint]>b[endpoint]+1e-12 for r,b in evaluated)>=math.ceil(p['minimum_positive_fold_fraction_per_side']*len(evaluated))
                criteria[endpoint+'_fold_consistency_vs_'+target]=bool(consistent); sides[endpoint]&=bool(passed and consistent)
    real=pooled['B1_ROUTE_WEIGHTED']
    criteria['fixed_retained']=real['fixed']>=p['minimum_fixed']; criteria['broken_controlled']=real['broken']<=p['maximum_broken']; criteria['net_improved']=real['net']>=p['minimum_net']
    success=all(criteria.values())
    status='ROUTE_CONDITIONED_TRAINING_SUPPORTED_IN_SOURCE' if success else ('ROUTE_CONDITIONED_TRAINING_PARTIALLY_SUPPORTED' if any(sides.values()) else 'ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED')
    return dict(status=status,criteria=criteria,directional_support=sides,pilot_passed=success,
        automatic_future_run=False,tuning_authorized=False,future_rows_loaded=0,
        next_action='freeze source evidence and independently specify replication; no automatic future run' if success else 'stop this branch; retain frozen Direct-C43; no weight or threshold search')

def main():
    p=protocol()
    with np.load(OUTPUT/'source_index.npz',allow_pickle=False) as a: d={k:a[k] for k in a.files}
    pred={}; rows=[]; folds=[]; per_seed=[]
    for arm in ARMS:
        with np.load(OUTPUT/'predictions'/f'{arm}.npz',allow_pickle=False) as a:
            if not np.array_equal(a['ids'],d['ids']): raise RuntimeError('prediction ID alignment')
            pred[arm]=np.asarray(a['expert'],float)
        rows.append(dict(arm=arm,scope='pooled',**metrics(d['y'],d['base'],pred[arm],d['route'])))
        for f in range(1,6):
            ix=d['folds']==f; folds.append(dict(arm=arm,fold=f,**metrics(d['y'][ix],d['base'][ix],pred[arm][ix],d['route'][ix])))
            if arm!='B0_ORIGINAL':
                with np.load(OUTPUT/'predictions'/arm/f'f{f}.npz',allow_pickle=False) as a:
                    lookup={s:i for i,s in enumerate(a['ids'])}; order=[lookup[s] for s in d['ids'][ix]]
                    for j,seed in enumerate(p['seeds']):
                        per_seed.append(dict(arm=arm,fold=f,seed=100000*f+seed,**metrics(d['y'][ix],d['base'][ix],a['seed_expert'][j,order],d['route'][ix])))
    contrasts=paired_bootstrap(d['y'],d['folds'],d['base'],d['route'],pred,p)
    decision=decide({r['arm']:r for r in rows},folds,contrasts,p)
    intact=read_json(OUTPUT/'baseline_integrity_check.json')
    if intact['status']!='PASS': raise RuntimeError('formal asset integrity failed')
    decision.update(formal_integrity='PASS',stopped_at='SOURCE_PREREGISTERED_PILOT',baseline_replay=read_json(OUTPUT/'baseline_replay_validation.json')['status'])
    write_csv(OUTPUT/'source_results.csv',rows); write_csv(OUTPUT/'source_fold_results.csv',folds); write_csv(OUTPUT/'seed_fold_sensitivity.csv',per_seed)
    write_csv(OUTPUT/'paired_comparisons.csv',contrasts)
    write_csv(OUTPUT/'source_predictions.csv',[dict(sha256=s,label=int(d['y'][i]),fold=int(d['folds'][i]),stage1_prob=float(d['base'][i]),routed=int(d['route'][i]),
        **{a+'_expert':float(pred[a][i]) for a in ARMS},**{a+'_final':float(pred[a][i] if d['route'][i] else d['base'][i]) for a in ARMS}) for i,s in enumerate(d['ids'])])
    dump(OUTPUT/'final_decision.json',decision)
    report=['# Direct-C43 route-conditioned training Pilot',f"\nFinal status: **{decision['status']}**.\n",
        'B0 uses the original all-APK equal BCE and inner early stopping. B1/B2/B3 use the same B0-selected fold/seed epoch counts, model, backgrounds, all training APKs and deployment route. Training masks are four-member nested crossfit predictions within each outer-train, with 3-fold calibration inside each member fit.\n',
        '| Arm | BA | AUC | Fixed | Broken | Net | FN fixed | FP fixed | FN/TN AUC | TP/FP AUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append('| '+r['arm']+' | '+' | '.join(str(r[k]) for k in ('BA','AUC','fixed','broken','net','fixed_FN','fixed_FP','FN_vs_TN_AUC','TP_vs_FP_AUC'))+' |')
    report += ['\n## Interpretation and statistical boundaries',
        'The bootstrap unit is APK, paired across arms, stratified by fixed outer fold, true label, deployment routing and Stage-I predicted side. The nine prespecified contrasts report ordinary 95% and Bonferroni percentile intervals. These approximate intervals condition on the fitted models; overlapping training sets, family/lineage dependence, refitting variation and earlier research selection are not covered. Seeds are sensitivity checks, not independent APKs. One conditional-shuffle draw is a matched control, not an exact conditional randomization test.\n',
        'B2 matches B1 class weight mass. B3 preserves the complete weight multiset within class/prediction-side/Host-size strata. The minibatch BCE uses the original batch-wise normalization, so class weight mass is a design check rather than an exact claim about realized gradient contributions. Background models continue to use all outer-train APKs. This Pilot isolates a fixed loss-weighting change; it does not test routed-only representation learning or optimal route-specific early stopping.\n',
        'All input fingerprints, epoch counts, parameter counts and unchanged input assets are audited. Empty Host bags follow the original cache policy; no failed APK is silently removed. Original APKs are not opened, rewritten or executed. No future data is loaded or scheduled.\n',
        'Passing justifies a separately specified replication, not replacement of the formal model or proof that training mismatch caused past errors. Failure stops this branch and forbids searching weights, thresholds or random seeds.\n',
        'Training time and routing preparation time are recorded per fold. See training/*/f*.json and routing_audit.json.\n',
        f"Formal reused-asset integrity: {intact['status']}.\n"]
    safe(OUTPUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    for r in rows: print(f"RCT_RESULT arm={r['arm']} BA={r['BA']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}",flush=True)
    print('RCT_FINAL_STATUS='+decision['status'],flush=True)
if __name__=='__main__': main()
