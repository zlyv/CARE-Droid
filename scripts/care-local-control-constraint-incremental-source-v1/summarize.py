#!/usr/bin/env python3
from __future__ import annotations
from common import *
from statistics_core import endpoints,paired_intervals

def main():
    ids,y,folds,base,route,expert,final=load_reference();p=protocol()
    with np.load(OUTPUT/'probe_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids):raise RuntimeError('OOF prediction SHA mismatch')
        scores={k:np.asarray(z[k],float) for k in z.files if k!='ids'}
    expected={'M0_DIRECT_C43','M1_HIDDEN_READOUT','M2_HIDDEN_CONTROL','M5_HIDDEN_COVERAGE_ONLY'}|{f'{arm}__r{r:02d}' for arm in ('M3_CAPACITY_PLACEBO','M4_CONDITIONAL_SHUFFLE') for r in range(p['null_repeats'])}
    if set(scores)!=expected:raise RuntimeError('arm/repeat count changed')
    if not np.array_equal(scores['M0_DIRECT_C43'],final):raise RuntimeError('M0 changed')
    if any(not(np.isfinite(s).all() and len(s)==799 and np.array_equal(s[~route],base[~route])) for s in scores.values()):raise RuntimeError('un-routed/OOF output mismatch')
    pooled=[];fold_rows=[];points={};fold_points={}
    for name,s in scores.items():
        ep=endpoints(y,s,base,route);points[name]=ep
        pooled.append(dict(arm=name,**metrics(y,base,s,route),**{k:v for k,v in ep.items() if k!='BA'},predicted_benign_n=int((route&(base<.5)).sum()),predicted_malware_n=int((route&(base>=.5)).sum())))
        for f in range(1,6):
            ix=np.flatnonzero(folds==f);value=endpoints(y[ix],s[ix],base[ix],route[ix]);fold_points[(name,f)]=value
            fold_rows.append(dict(arm=name,fold=f,**metrics(y[ix],base[ix],s[ix],route[ix]),**{k:v for k,v in value.items() if k!='BA'}))
    intervals=paired_intervals(y,folds,base,route,scores,p['bootstrap_repeats'],p['bootstrap_seed'])
    interval_map={(r['comparator'],r['endpoint']):r for r in intervals}
    null95={side:max(float(np.quantile([points[f'{a}__r{r:02d}'][side] for r in range(p['null_repeats'])],.95)) for a in ('M3_CAPACITY_PLACEBO','M4_CONDITIONAL_SHUFFLE')) for side in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')}
    direction={};fold_support={}
    for side in ('FN_vs_TN_AUC','TP_vs_FP_AUC'):
        evaluable=[f for f in range(1,6) if fold_points[('M2_HIDDEN_CONTROL',f)][side] is not None and fold_points[('M1_HIDDEN_READOUT',f)][side] is not None]
        positive=sum(fold_points[('M2_HIDDEN_CONTROL',f)][side]>fold_points[('M1_HIDDEN_READOUT',f)][side] for f in evaluable)
        fold_support[side]=dict(evaluable_folds=len(evaluable),positive_vs_readout=positive)
        gap=p['information_gate']['minimum_auc_gain_vs_readout_and_null95']
        checks=dict(real_at_least_readout_plus_margin=points['M2_HIDDEN_CONTROL'][side]>=points['M1_HIDDEN_READOUT'][side]+gap,
            real_at_least_null95_plus_margin=points['M2_HIDDEN_CONTROL'][side]>=null95[side]+gap,
            evaluable_folds=len(evaluable)>=p['information_gate']['minimum_evaluable_folds_per_side'],
            majority_folds=positive>=p['information_gate']['minimum_folds_positive_vs_readout_per_side'],
            adjusted_paired_intervals_positive=all(interval_map[(a,side)]['multiplicity_adjusted_low'] is not None and interval_map[(a,side)]['multiplicity_adjusted_low']>0 for a in ('M1_HIDDEN_READOUT','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00')))
        direction[side]=dict(supported=all(checks.values()),criteria=checks)
    pooled_by={r['arm']:r for r in pooled};real=pooled_by['M2_HIDDEN_CONTROL']
    fold_net=sum(next(r['net'] for r in fold_rows if r['arm']=='M2_HIDDEN_CONTROL' and r['fold']==f)>=next(r['net'] for r in fold_rows if r['arm']=='M0_DIRECT_C43' and r['fold']==f) for f in range(1,6))
    integration=dict(net_at_least_33=real['net']>=p['integration_gate']['net_at_least'],broken_at_most_17=real['broken']<=p['integration_gate']['broken_at_most'],
        at_least_four_folds_net_nondecreasing=fold_net>=p['integration_gate']['minimum_folds_net_nondecreasing_vs_direct'],
        BA_above_null95=real['BA']>null95['BA'],
        adjusted_BA_intervals_positive=all(interval_map[(a,'BA')]['multiplicity_adjusted_low'] is not None and interval_map[(a,'BA')]['multiplicity_adjusted_low']>0 for a in ('M0_DIRECT_C43','M1_HIDDEN_READOUT','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00')))
    training=json.loads((OUTPUT/'probe_training_audit.json').read_text());integrity=json.loads((OUTPUT/'baseline_integrity_check.json').read_text())
    both=all(d['supported'] for d in direction.values());one=any(d['supported'] for d in direction.values());valid=integrity['status']=='PASS' and training['conditional_null_matching_ready']
    if not valid:status='CONTROL_INCREMENT_INCONCLUSIVE'
    elif both and all(integration.values()):status='CONTROL_INCREMENT_SUPPORTED_SOURCE_ONLY'
    elif both:status='CONTROL_DIRECTION_SUPPORTED_INTEGRATION_NOT_CONFIRMED'
    elif one:status='PARTIALLY_SUPPORTED'
    else:status='CONTROL_INCREMENT_NOT_SUPPORTED'
    decision=dict(status=status,continue_to_specify_frozen_diagnostic=bool(valid and both and all(integration.values())),formal_stage1_changed=False,formal_stage2_changed=False,
        direction=direction,fold_direction_support=fold_support,integration_criteria=integration,null95=null95,conditional_null_ready=training['conditional_null_matching_ready'],
        formal_integrity=integrity['status'],future_rows_loaded=0,auxiliary_hyperparameter_search=False,
        inference_scope='source-only, fixed-model paired APK bootstrap; no family dependence or retraining uncertainty correction; historically examined source is not a pristine confirmatory holdout',
        next_action='specify separately frozen diagnostics; never select on future' if valid and both and all(integration.values()) else 'stop this version; report all arms and uncertainty; do not tune or launch future')
    write_csv(OUTPUT/'source_metrics.csv',pooled);write_csv(OUTPUT/'fold_metrics.csv',fold_rows);write_csv(OUTPUT/'paired_intervals.csv',intervals)
    primary_names=('M0_DIRECT_C43','M1_HIDDEN_READOUT','M2_HIDDEN_CONTROL','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00','M5_HIDDEN_COVERAGE_ONLY')
    write_csv(OUTPUT/'primary_source_metrics.csv',[pooled_by[a] for a in primary_names])
    write_csv(OUTPUT/'primary_fold_metrics.csv',[r for a in primary_names for r in fold_rows if r['arm']==a])
    dump(OUTPUT/'final_decision.json',decision)
    lines=['# Local Control Constraint Incremental Source Test v1','',f'Final status: **{status}**.','',
        '799 source APKs provide features/training; the fixed 200-APK route alone receives final expert decisions. No future APKs or labels are loaded.',
        'Auxiliary logistic parameters are fixed, with no inner search. Training hidden states are in-sample under the upstream outer-fold model; they are not claimed to be inner OOF. Each test representation is outer OOF, and coordinates are never mixed across folds.',
        '', '| Arm | BA | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('M0_DIRECT_C43','M1_HIDDEN_READOUT','M2_HIDDEN_CONTROL','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00','M5_HIDDEN_COVERAGE_ONLY'):
        r=pooled_by[name];lines.append(f'| {name} | {r["BA"]:.6f} | {r["fixed"]} | {r["broken"]} | {r["net"]} | {r["FN_vs_TN_AUC"]:.6f} | {r["TP_vs_FP_AUC"]:.6f} |')
    lines+=['','## Decision criteria','',json.dumps(dict(direction=direction,integration=integration,null95=null95),ensure_ascii=False,indent=2),
        '', '## Statistical boundary','',
        'The unit is APK, not Host/event/seed/null repetition. Paired bootstrap resamples preserve fold, label, route and Stage-I-side strata. Ordinary 95% intervals and Bonferroni-adjusted intervals for ten predefined differences are both exported. They condition on fitted models and this cohort; code-family dependence and upstream retraining variance are not captured.',
        'Conditional shuffle uses approximate label-free matching and receiver Role masking. Its null repeats are sensitivity controls, not independent samples or an exact conditional-randomization p-value. All 20 repeats are reported.',
        '', '## Semantics and protection','',
        'Zero proof coverage is not a no-guard claim. UNKNOWN/PARTIAL is not benign/malicious evidence. No permission fact, raw exact API, future selection, new gate, threshold tuning or bounded correction is used.',
        f'Formal/source/parent integrity: {integrity["status"]}. No historical output is overwritten.',
        'See feature_timing.json and phase_timing.json for measured cost; reused v2 facts are not counted as newly executed scans.',
        '',f'Next action: {decision["next_action"]}','']
    safe(OUTPUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('LCIP_FINAL_STATUS='+status,flush=True)
    for a in primary_names[:3]:
        r=pooled_by[a];print(f'LCIP_RESULT arm={r["arm"]} BA={r["BA"]:.6f} fixed={r["fixed"]} broken={r["broken"]} net={r["net"]}',flush=True)
if __name__=='__main__':main()
