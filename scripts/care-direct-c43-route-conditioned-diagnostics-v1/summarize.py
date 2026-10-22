from __future__ import annotations
import argparse
import numpy as np
from diag_common import *

def compare_rows(mode,rows,labels,core):
    ids=sorted({norm(r['sha256']) for r in rows});by={(norm(r['sha256']),r['arm']):r for r in rows}
    if len(rows)!=4*1989 or len(by)!=len(rows) or set(ids)!=set(labels):raise RuntimeError('complete four-arm diagnostic cohort required')
    y=np.asarray([labels[s] for s in ids]);r0=[by[(s,'B0_ORIGINAL')] for s in ids]
    years=np.asarray([int(r['year']) for r in r0]);folds=np.asarray([int(r['fold']) for r in r0]);base=np.asarray([float(r['stage1_prob']) for r in r0]);route=np.asarray([int(r['routed']) for r in r0],bool)
    pred={a:np.asarray([float(by[(s,a)]['expert_prob']) for s in ids]) for a in ARMS}
    for arm in ARMS:
        rr=[by[(s,arm)] for s in ids]
        if not np.array_equal(np.asarray([float(r['stage1_prob']) for r in rr]),base) or not np.array_equal(np.asarray([int(r['routed']) for r in rr],bool),route):raise RuntimeError('arm deployment route/base differs')
        for i,r in enumerate(rr):
            if float(r['final_prob'])!=(pred[arm][i] if route[i] else base[i]):raise RuntimeError('direct decision mixture invalid')
    out=[];foldout=[]
    for scope,ix in [(str(z),years==z) for z in YEARS]+[('pooled',np.ones(len(y),bool))]:
        first=core.metrics(y[ix],base[ix],base[ix],route[ix])
        for arm in ARMS:
            second=core.metrics(y[ix],base[ix],pred[arm][ix],route[ix]);b0=core.metrics(y[ix],base[ix],pred['B0_ORIGINAL'][ix],route[ix])
            out.append(dict(protocol=mode,year=scope,arm=arm,n=int(ix.sum()),Stage1_BA=first['BA'],Stage2_BA=second['BA'],BA_gain_pp=100*(second['BA']-first['BA']),
                Stage1_accuracy=first['accuracy'],Stage2_accuracy=second['accuracy'],accuracy_gain_pp=100*(second['accuracy']-first['accuracy']),
                Stage1_AUC=first['AUC'],Stage2_AUC=second['AUC'],Stage1_F1=first['F1'],Stage2_F1=second['F1'],
                delta_BA_vs_B0_pp=100*(second['BA']-b0['BA']),delta_net_vs_B0=second['net']-b0['net'],**{k:v for k,v in second.items() if k not in ('n','BA','AUC','F1','accuracy')}))
        if mode=='within' and scope!='pooled':
            for f in range(1,6):
                mask=ix&(folds==f)
                for arm in ARMS:foldout.append(dict(protocol=mode,year=scope,fold=f,arm=arm,**core.metrics(y[mask],base[mask],pred[arm][mask],route[mask])))
    for arm in ARMS:
        annual=[r for r in out if r['arm']==arm and r['year']!='pooled']
        out.append(dict(protocol=mode,year='macro_year_mean',arm=arm,n=1989,**{k:float(np.mean([r[k] for r in annual])) for k in ('Stage1_BA','Stage2_BA','BA_gain_pp','Stage1_accuracy','Stage2_accuracy','accuracy_gain_pp','Stage1_AUC','Stage2_AUC','Stage1_F1','Stage2_F1','delta_BA_vs_B0_pp')}))
    return out,foldout,(ids,y,years,folds,base,route,pred)

def paired_intervals(mode,data,core,repeats=2000):
    ids,y,years,folds,base,route,pred=data;rng=np.random.RandomState(20261018);out=[]
    for scope,mask in [(str(z),years==z) for z in YEARS]+[('pooled',np.ones(len(y),bool))]:
        groups={}
        for i in np.flatnonzero(mask):groups.setdefault((int(years[i]),int(folds[i]),int(y[i]),bool(route[i]),bool(base[i]>=.5)),[]).append(i)
        buckets=[np.asarray(v,int) for v in groups.values()];samples=[]
        for _ in range(repeats):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in buckets]);b=outcome_totals(y[ix],base[ix],pred['B0_ORIGINAL'][ix],route[ix])
            values=[outcome_totals(y[ix],base[ix],pred[a][ix],route[ix]) for a in ARMS[1:]]
            samples.append([[v[k]-b[k] for k in ('BA','net','broken')] for v in values])
        samples=np.asarray(samples);ix=np.flatnonzero(mask);b=core.metrics(y[ix],base[ix],pred['B0_ORIGINAL'][ix],route[ix])
        for j,arm in enumerate(ARMS[1:]):
            m=core.metrics(y[ix],base[ix],pred[arm][ix],route[ix])
            for k,endpoint in enumerate(('BA','net','broken')):
                lo,hi=np.quantile(samples[:,j,k],[.025,.975]);out.append(dict(protocol=mode,year=scope,comparison=arm+'-B0',endpoint=endpoint,
                    difference=m[endpoint]-b[endpoint],ci95_low=float(lo),ci95_high=float(hi),bootstrap_repeats=repeats,
                    inference='exploratory unadjusted, conditional on fitted models; no method selection'))
    return out

def outcome_totals(y,base,expert,route):
    final=np.where(route,expert,base);before=(base>=.5)==y;after=(final>=.5)==y
    fixed=int((~before&after).sum());broken=int((before&~after).sum())
    return dict(BA=float(.5*(np.mean(final[y==0]<.5)+np.mean(final[y==1]>=.5))),net=fixed-broken,broken=broken)

def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen-only',action='store_true');cli=p.parse_args();core,*_=parent_modules()
    # This sealed label table is opened only after all four frozen predictions are written.
    labels={norm(r['sha256']):int(r['label']) for r in read_csv(FORMAL_DIAG/'frozen_predictions_with_labels.csv')}
    result,folds,data=compare_rows('frozen',read_csv(OUTPUT/'frozen_predictions_label_free.csv'),labels,core)
    write_csv(OUTPUT/'frozen_future_metrics.csv',result)
    intervals=paired_intervals('frozen',data,core)
    if not cli.frozen_only:
        within_meta=read_csv(FORMAL_DIAG/'within_2018_2019/predictions.csv')+read_csv(FORMAL_DIAG/'within_2020_2021/predictions.csv')
        wl={norm(r['sha256']):int(r['label']) for r in within_meta}
        if wl!=labels:raise RuntimeError('frozen and within label contract differs')
        wr,wf,wd=compare_rows('within',read_csv(OUTPUT/'within_predictions_label_free.csv'),wl,core);result+=wr;folds+=wf
        write_csv(OUTPUT/'within_year_metrics.csv',wr);write_csv(OUTPUT/'within_year_fold_metrics.csv',folds)
        intervals+=paired_intervals('within',wd,core)
    write_csv(OUTPUT/'all_year_comparison.csv',result);write_csv(OUTPUT/'paired_diagnostic_intervals.csv',intervals)
    decision=dict(status='FROZEN_DIAGNOSTIC_COMPLETE_WITHIN_PENDING' if cli.frozen_only else 'DIAGNOSTICS_COMPLETE_SOURCE_NOT_SUPPORTED_RETAINED',
        source_status='ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED',protocols_completed=['frozen'] if cli.frozen_only else ['frozen','within'],
        future_used_for_selection=False,within_year_used_for_selection=False,formal_model_changed=False,weight_or_threshold_tuning=False,
        user_authorized_future_diagnostics=True,source_conclusion_reversed=False,
        next_action='inspect diagnostic tables; no promotion, weight search or automatic new experiment')
    if not cli.frozen_only:
        decision['formal_integrity']=read_json(OUTPUT/'baseline_integrity_check.json')['status']
        if decision['formal_integrity']!='PASS':raise RuntimeError('reused asset integrity failed')
    dump(OUTPUT/'final_decision.json',decision)
    lines=['# Route-conditioned training: frozen future and within-year diagnostics','',
        'The source Pilot remains NOT_SUPPORTED. These explicitly requested diagnostic results do not select a method, weight, seed, threshold or new formal mainline. Within-year results are independently retrained OOF results, not temporal generalization.','',
        '| Protocol | Year | Arm | Stage-I BA | Stage-II BA | Gain pp | Fixed | Broken | Net |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result:lines.append(f"| {r['protocol']} | {r['year']} | {r['arm']} | {100*r['Stage1_BA']:.4f}% | {100*r['Stage2_BA']:.4f}% | {r['BA_gain_pp']:+.4f} | {r.get('fixed','')} | {r.get('broken','')} | {r.get('net','')} |")
    lines+=['','Accuracy, BA, AUC, F1, both conditional AUCs, FN/FP decompositions and routing coverage are in the CSV files. Macro-year mean weights each year equally; pooled metrics weight APKs. Counts are not averaged in the macro row.',
        'Paired bootstrap intervals resample APKs within year/fold/label/route/Stage-I-side strata. They are exploratory, unadjusted, conditional on fixed fitted models and do not account for refitting variation or app-family dependence. They do not rescue the failed source gate.',
        'B0 chooses epochs on training-only internal validation. All weighted arms reuse B0 epochs and original backgrounds/capacity. Full-source frozen models use 2017 labels only. Within-year models use only that outer fold training labels. Labels in the sealed frozen table are opened only after four frozen prediction vectors are complete.',
        'The combined legacy feature CSV may contain metadata labels; the frozen trainer does not use target labels in the fit. Within-year metadata labels are used for the explicitly requested same-year evaluation and fixed folds. Original APKs are never parsed, run, rewritten or installed; no cache rebuild is performed.']
    safe(OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RCTD_SUMMARY_DONE protocols='+','.join(decision['protocols_completed']),flush=True)
if __name__=='__main__':main()
