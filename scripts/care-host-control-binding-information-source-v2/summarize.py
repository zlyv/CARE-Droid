#!/usr/bin/env python3
from common import *
from statistics_core import endpoints,paired_intervals,gate,PRIMARY

def main():
    training=read_json(OUTPUT/'probe_training_audit.json');intact=read_json(OUTPUT/'baseline_integrity_check.json')['status'];p=protocol()
    if training['status']=='NOT_RUN_PROJECTION_COVERAGE_INSUFFICIENT':
        d=dict(status='HOST_BINDING_PROJECTION_COVERAGE_INSUFFICIENT',formal_integrity=intact,model_training_performed=False,
            future_rows_loaded=0,permit_specifying_separate_frozen_diagnostic=False,next_action='stop; do not search projection width or seeds')
        dump(OUTPUT/'final_decision.json',d)
        safe(OUTPUT/'REPORT.md').write_text('# Host binding information test\n\n'+json.dumps(d,indent=2)+'\n',encoding='utf-8');return
    ids,y,folds,base,route,expert,final=load_reference()
    with np.load(OUTPUT/'probe_predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids):raise RuntimeError('OOF prediction order mismatch')
        scores={a:np.asarray(z[a],float) for a in z.files if a!='ids'}
    expected={'B0_DIRECT_C43','B1_HIDDEN_AGGREGATE','B2_HOST_BINDING_PRIMARY','B5_HIDDEN_READOUT'}|{f'{a}__r{r:02d}' for a in ('B3_WITHIN_APK_SHUFFLE','B4_CAPACITY_PLACEBO') for r in range(20)}
    if set(scores)!=expected or not np.array_equal(scores['B0_DIRECT_C43'],final) or any(v.shape!=(799,) or not np.isfinite(v).all() or np.any((v<0)|(v>1)) or not np.array_equal(v[~route],base[~route]) for v in scores.values()):raise RuntimeError('arm, probability or un-routed preservation mismatch')
    pooled=[];fr=[];points={};fp={};fm={}
    for a,v in scores.items():
        value=endpoints(y,v,base,route);points[a]=value
        pooled.append(dict(arm=a,**metrics(y,base,v,route),FN_vs_TN_AUC=value['FN_vs_TN_AUC'],TP_vs_FP_AUC=value['TP_vs_FP_AUC'],
            predicted_benign_n=int((route&(base<.5)).sum()),predicted_malware_n=int((route&(base>=.5)).sum())))
        for f in range(1,6):
            ix=folds==f;fp[(a,f)]=endpoints(y[ix],v[ix],base[ix],route[ix]);fm[(a,f)]=metrics(y[ix],base[ix],v[ix],route[ix])
            fr.append(dict(arm=a,fold=f,**fm[(a,f)],FN_vs_TN_AUC=fp[(a,f)]['FN_vs_TN_AUC'],TP_vs_FP_AUC=fp[(a,f)]['TP_vs_FP_AUC']))
    interval=paired_intervals(y,folds,base,route,scores,p['bootstrap_repeats'],p['bootstrap_seed'])
    null95={}
    for a in ('B3_WITHIN_APK_SHUFFLE','B4_CAPACITY_PLACEBO'):
        null95[a]={}
        for endpoint in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC'):
            values=[points[f'{a}__r{r:02d}'][endpoint] for r in range(20)]
            null95[a][endpoint]=float(np.quantile([v for v in values if v is not None],.95)) if any(v is not None for v in values) else None
    valid=intact=='PASS' and read_json(OUTPUT/'feature_validation.json')['projection_ready'] and read_json(OUTPUT/'aggregate_replay_validation.json')['status']=='PASS'
    d=gate(points,fp,fm,interval,null95,p,valid)
    d.update(formal_integrity=intact,null95=null95,future_rows_loaded=0,model_training_performed=True,new_DEX_scans=0,
        formal_stage1_changed=False,formal_stage2_changed=False,auxiliary_hyperparameter_search=False,primary_representation_width=64,
        inference_boundary='conditional fixed-model APK bootstrap on historically examined source; not pristine confirmatory or full-family/retraining uncertainty',
        next_action='specify separate frozen diagnostics without future selection' if d['permit_specifying_separate_frozen_diagnostic'] else 'stop this version; no tuning or automatic future run')
    names=('B0_DIRECT_C43','B5_HIDDEN_READOUT','B1_HIDDEN_AGGREGATE',PRIMARY,'B3_WITHIN_APK_SHUFFLE__r00','B4_CAPACITY_PLACEBO__r00')
    by={r['arm']:r for r in pooled}
    write_csv(OUTPUT/'source_metrics.csv',pooled);write_csv(OUTPUT/'fold_metrics.csv',fr);write_csv(OUTPUT/'paired_intervals.csv',interval)
    write_csv(OUTPUT/'primary_source_metrics.csv',[by[a] for a in names]);write_csv(OUTPUT/'primary_fold_metrics.csv',[r for a in names for r in fr if r['arm']==a])
    dump(OUTPUT/'final_decision.json',d)
    lines=['# CARE-Droid Host-Control Binding Conditional Information Test v2','',f'Final status: **{d["status"]}**.','',
        'All 799 source APKs provide label-free features and outer-train supervision; only the frozen 200 routed test APKs receive auxiliary final predictions. No C43, Stage-I, route, threshold or bounded-correction model is changed.',
        '', '| Arm | BA | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |','|---|---:|---:|---:|---:|---:|---:|']
    def show(v):return 'unavailable' if v is None else f'{v:.6f}'
    for a in names:
        r=by[a];lines.append(f'| {a} | {r["BA"]:.6f} | {r["fixed"]} | {r["broken"]} | {r["net"]} | {show(r["FN_vs_TN_AUC"])} | {show(r["TP_vs_FP_AUC"])} |')
    lines += ['', '## What changed','',
        'The new input is a single frozen 64-coordinate signed CountSketch of the previously audited Role-conditioned Host/proof covariance. Raw exact APIs, permission-as-benign rules and Host pseudo-labels are excluded. All 20 within-APK binding permutations are replayed, and previous 124+64 marginals remain exact. B1 and B5 must reproduce the previous aggregate and hidden-readout predictions before new binding fits.',
        '', '## Evaluation and statistical boundary','',
        'Each auxiliary fit uses all outer-train APKs and its own frozen upstream model coordinate space. Training hidden states are in-sample under that upstream model, not inner OOF; there is no hyperparameter selection, upstream inner refit or claim of newly fully nested training. Scalers and placebo moments use outer-train data only.',
        'The independent reporting/resampling unit is APK, not Host, edge, fold or null repeat. Primary paired bootstrap uses 2000 fold/label/route/Stage-I-side-stratified resamples; ordinary 95% and Bonferroni-adjusted intervals for ten contrasts are exported. Six fixed/broken/net intervals are exploratory. Twenty null realizations are sensitivity controls, not independent test samples or exact conditional-randomization p-values. Intervals condition on fitted models and this historically examined source cohort; family dependence and retraining variability are not captured. Within-version multiplicity correction does not correct the entire history of experiments.',
        '', '## Frozen gate','',json.dumps(dict(direction=d['direction'],integration=d['integration_criteria'],null95=null95),indent=2),
        '', '## Semantics, safety and cost','',
        'A proof concerns a local static CFG, not runtime execution, consent, data purpose or malware causality. UNKNOWN/PARTIAL never imply unguarded; absence of positive evidence never becomes a benign certificate. A negative result is scoped to this fixed compact representation/readout, not every possible Host vector.',
        f'No APK was opened, installed, executed or rewritten. New DEX scans: 0. Reused-asset integrity: {intact}; see manifests for actual scope. All output is isolated, and no historical result is overwritten. Future rows loaded: 0.',
        'Measured cost is in phase_timing.json. No server timing is inferred from synthetic local tests.',
        '', '## Next action','',d['next_action'],'']
    safe(OUTPUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('HBI_FINAL_STATUS='+d['status'],flush=True)
    for a in names:
        r=by[a];print(f'HBI_RESULT arm={a} BA={r["BA"]:.6f} fixed={r["fixed"]} broken={r["broken"]} net={r["net"]}',flush=True)
if __name__=='__main__':main()
