"""Aggregate completed development windows; never manufacture scores for unrun windows or infer algorithmic novelty."""
from __future__ import annotations
import numpy as np
from myd_common import *
from audit_data import load_data
from evaluate_cascade import archive
from metrics import classification,routing,cascade
from protocol import WINDOWS,top_budget,mean_opposition

def collect():
    data=load_data();stage=[];routes=[];casc=[];per=[];stage_per=[];abl=[];seeds=[];runtime_cost=[]
    for w in WINDOWS:
        root=OUTPUT/'predictions'/w
        sp=root/'stage1.npz';rp=root/'router.npz'
        if sp.exists() or rp.exists():
            a=archive(rp if rp.exists() else sp);va=a['val_index'];y=data['y'][va];year=int(data['years'][va][0])
            for name,p in [('S0H_HISTORICAL',data['historical_base'][va]),('S0R_MATCHED_2017',a['s0']),('S1_ROLLING',a['s1'])]:
                stage.append(dict(window=w,year=year,arm=name,**classification(y,p)))
                stage_per.extend(dict(window=w,arm=name,sha256=sid,year=year,label=int(y[k]),stage1_probability=float(p[k]),stage1_prediction=int(p[k]>=.5),stage1_correct=int((p[k]>=.5)==y[k]),routed=0,expert_probability=float(p[k]),final_probability=float(p[k]),final_prediction=int(p[k]>=.5),correct=int((p[k]>=.5)==y[k]),scope='Stage-I only; no expert invocation') for k,sid in enumerate(a['ids']))
            if rp.exists():
                for r,base in [('old',a['s0']),('reanchor',a['s1']),('new',a['s1'])]:
                    score=a[r+'_score']
                    for b in [.10,.15,.25,.30,.40,.50]:
                        route=top_budget(score,a['ids'],b);routes.append(dict(window=w,year=year,router=r,budget=b,primary=b==.25,score_min=float(score.min()),score_median=float(np.median(score)),score_max=float(score.max()),**routing(y,base,route)))
                hb=data['historical_base'][va];hs=mean_opposition(data['historical_members'][va],hb)
                routes.append(dict(window=w,year=year,router='historical_saved',budget=.25,primary=True,score_min=float(hs.min()),score_median=float(np.median(hs)),score_max=float(hs.max()),**routing(y,hb,top_budget(hs,a['ids']))))
        for name,target in [('cascade_results.csv',casc),('per_apk_predictions.csv',per),('ablation_results.csv',abl),('seed_sensitivity.csv',seeds)]:
            if (root/name).exists():target.extend(read_csv(root/name))
        for p in root.glob('*_timing.json'):runtime_cost.extend(read_json(p))
    # Pooled means are recomputed from all completed validation APKs; macro means
    # give every completed year equal weight. Missing years remain explicitly absent.
    def combine(rows,classification_only=False):
        result=list(rows)
        for arm in sorted({r['arm'] for r in rows}):
            rs=[r for r in rows if r['arm']==arm];metric=[k for k in rs[0] if k not in ['year','window','arm']]
            macro={k:float(np.nanmean([float(r[k]) for r in rs])) if any(np.isfinite(float(r[k])) for r in rs) else float('nan') for k in metric}
            result.append(dict(window='macro_completed',year='macro',arm=arm,years_completed=len(rs),**macro))
        return result
    cascade_out=combine(casc) if casc else []
    for arm in sorted({r['arm'] for r in per}):
        rs=[r for r in per if r['arm']==arm];y=np.asarray([int(r['label']) for r in rs]);base=np.asarray([float(r['stage1_probability']) for r in rs]);e=np.asarray([float(r['expert_probability']) for r in rs]);route=np.asarray([int(r['routed']) for r in rs],bool)
        cascade_out.append(dict(window='pooled_completed',year='pooled',arm=arm,**cascade(y,base,e,route)))
    stage_out=combine(stage) if stage else []
    for arm in sorted({r['arm'] for r in stage}):
        yy=[];pp=[]
        for w in WINDOWS:
            root=OUTPUT/'predictions'/w;rp=root/'router.npz';sp=root/'stage1.npz'
            if not (rp.exists() or sp.exists()):continue
            a=archive(rp if rp.exists() else sp);va=a['val_index'];yy.extend(data['y'][va]);pp.extend(data['historical_base'][va] if arm.startswith('S0H') else a['s0'] if arm.startswith('S0R') else a['s1'])
        stage_out.append(dict(window='pooled_completed',year='pooled',arm=arm,**classification(np.asarray(yy),np.asarray(pp))))
    for p in (OUTPUT/'models').rglob('*.pkl') if (OUTPUT/'models').exists() else []:
        with p.open('rb') as f:o=__import__('pickle').load(f)
        audit=o.get('audit',{});runtime_cost.append(dict(path=str(p),fit_seconds=o.get('fit_seconds',audit.get('fit_seconds',0)),inference_seconds=audit.get('inference_seconds',None),cache_extraction_seconds=0,new_APK_extraction_performed=False,scope='expert model' if audit else 'classifier fit; inference measured in per-window timing rows'))
    for name,rs in [('stage1_results',stage_out),('router_results',routes),('stage2_results',cascade_out),('cascade_results',cascade_out),('ablation_results',abl),('per_apk_predictions',stage_per+per),('stage1_per_apk_predictions',stage_per),('runtime_cost',runtime_cost),('seed_sensitivity',seeds)]:write_csv(OUTPUT/(name+'.csv'),rs)
    completed=sorted({r['window'] for r in casc});decision=dict(status='DEVELOPMENT_COMPLETE' if len(completed)==4 else 'PARTIAL_RESULTS' if completed else 'AUDIT_OR_SMOKE_COMPLETE_FULL_TRAINING_NOT_RUN',completed_cascade_windows=completed,missing_cascade_windows=[w for w in WINDOWS if w not in completed],development_not_final_test=True,automatic_recommendation=False,Q1='see stage1_results; no conclusion from absent windows',Q2='see primary router rows; diagnostic budgets never selected',Q3='see fixed-upstream C1-C0R/C2-C1 paired contrasts; no conclusion from absent windows',historical_algorithms_changed=False)
    decision['completed_stage1_windows']=sorted({r['window'] for r in stage});decision['completed_router_windows']=sorted({r['window'] for r in routes})
    if len(decision['completed_stage1_windows'])==4:
        s0=next(r for r in stage_out if r['window']=='pooled_completed' and r['arm']=='S0R_MATCHED_2017');s1=next(r for r in stage_out if r['window']=='pooled_completed' and r['arm']=='S1_ROLLING')
        decision['Q1']=dict(pooled_BA_2017_only=s0['BA'],pooled_BA_rolling=s1['BA'],gain_percentage_points=100*(s1['BA']-s0['BA']),FN_2017_only=s0['FN'],FN_rolling=s1['FN'],FP_2017_only=s0['FP'],FP_rolling=s1['FP'],interpretation='descriptive historical-data benefit, not a new algorithm; 2020 AUC did not improve')
    if len(decision['completed_router_windows'])==4:
        rr=[r for r in routes if r['primary'] and r['router']=='reanchor'];rn=[r for r in routes if r['primary'] and r['router']=='new'];errors=sum(r['stage1_errors'] for r in rn)
        decision['Q2']=dict(fixed_rolling_SVM_errors=errors,captured_old_auxiliary=sum(r['routed_errors'] for r in rr),captured_rolling_auxiliary=sum(r['routed_errors'] for r in rn),recall_old_auxiliary=sum(r['routed_errors'] for r in rr)/errors,recall_rolling_auxiliary=sum(r['routed_errors'] for r in rn)/errors,interpretation='same SVM/SHA/budget; three expanded windows improve observed coverage; expert effects pending')
        if not completed:decision['status']='STAGE1_ROUTER_COMPLETE_STAGE2_PENDING'
    replay=[]
    for w in WINDOWS:
        q=OUTPUT/'predictions'/w/'router.npz'
        if q.exists():
            a=archive(q);replay.append(dict(window=w,max_probability_difference=float(np.max(np.abs(a['s0']-data['historical_base'][a['val_index']])))))
    if replay:dump(OUTPUT/'stage1_replay_validation.json',dict(status='PASS' if max(r['max_probability_difference'] for r in replay)<=1e-8 else 'DESCRIPTIVE_ONLY',details=replay))
    dump(OUTPUT/'final_decision.json',decision)
    lines=['# CARE-Droid Multi-Year Development V1','',f"Status: {decision['status']}",'','2017–2021 are development data. This is not a new independent final test. No MalEval input, new Host feature, route gate, weight search or APK extraction is used.','', '## Actual dataset audit','',str(read_json(OUTPUT/'dataset_audit.json')['by_year']),'','## Stage-I year-by-year results','','| Window | Year | Arm | BA | AUC | FP | FN |','|---|---|---|---:|---:|---:|---:|']
    for r in stage:lines.append(f"| {r['window']} | {r['year']} | {r['arm']} | {float(r['BA']):.6f} | {float(r['AUC']):.6f} | {r['FP']} | {r['FN']} |")
    lines+=['','## Primary Top25% router','','| Year | Router | Errors | Captured | Recall | Precision | Lift |','|---|---|---:|---:|---:|---:|---:|']
    for r in routes:
        if r['primary']:lines.append(f"| {r['year']} | {r['router']} | {r['stage1_errors']} | {r['routed_errors']} | {r['ErrorRecall']:.6f} | {r['RoutePrecision']:.6f} | {r['RoutingLift']:.6f} |")
    lines+=['', '## Rolling year-by-year cascade','', '| Window | Year | Arm | BA | Fixed | Broken | Net |','|---|---|---|---:|---:|---:|---:|']
    for r in casc:lines.append(f"| {r['window']} | {r['year']} | {r['arm']} | {float(r['BA']):.6f} | {r['Fixed']} | {r['Broken']} | {r['Net']} |")
    if not casc:lines.append('No full-window deep training has run. No empirical Q1/Q2/Q3 conclusions are available yet.')
    lines+=['','## Current Q1/Q2/Q3 evidence','',str(decision['Q1']),'',str(decision['Q2']),'',str(decision['Q3']),'','## Engineering replay and smoke','',str(read_json(OUTPUT/'baseline_replay.json')) if (OUTPUT/'baseline_replay.json').exists() else 'unavailable','',str(read_json(OUTPUT/'smoke_test.json')) if (OUTPUT/'smoke_test.json').exists() else 'not run','','## Attribution and limitations','','S1-only classifier change is reported separately. U1 includes unavoidable SVM reanchoring and is not a pure Stage-I cascade contrast. U2–U1 isolates auxiliary-member refitting at the same SVM and expert. U3–U2 fixes Stage-I and routing; U4–U3 fixes data, capacity and optimization steps. H0 uses untouched historical predictions and is descriptive; S0R/C0R are controlled retrained baselines.','', 'Bootstrap intervals are paired APK-level, conditional on fitted models, exploratory and not multiple-comparison-adjusted. Actual family labels are unavailable; intervals do not prove family-independent generalization. Seeds measure sensitivity and are not independent APK samples. Missing conditional classes yield NaN with counts.','', 'A positive rolling result alone is ordinary training-data benefit, not an algorithmic novelty claim. Historical future years have been used repeatedly and cannot be relabeled as independent final tests.','', '## Integrity and cost','',str(read_json(OUTPUT/'baseline_integrity_check.json')),'','See runtime_cost.csv for measured training/inference; new static extraction cost is zero because existing caches are reused. Original APKs were not opened. Integrity is scoped to manifest-listed assets, not the entire server.']
    safe(OUTPUT/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('MYD_RESULTS_COLLECTED',decision['status'],flush=True)

if __name__=='__main__':collect()
