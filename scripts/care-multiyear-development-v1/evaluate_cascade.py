"""Fixed-upstream expert comparisons and anchor-compatible module attribution on identical validation SHAs."""
from __future__ import annotations
import numpy as np
from myd_common import *
from protocol import top_budget,mean_opposition
from metrics import *

def archive(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def expert(p,ids):
    a=archive(p);position={s:i for i,s in enumerate(a['ids'])}
    if any(s not in position for s in ids):raise RuntimeError('expert validation SHA missing')
    ix=[position[s] for s in ids];return a['probability'][ix],a['seed_probability'][:,ix]

def evaluate(data,window,bootstrap=2000):
    root=OUTPUT/'predictions'/window;router=archive(root/'router.npz');ids=router['ids'];va=router['val_index'];y=data['y'][va]
    if not np.array_equal(ids,data['ids'][va]):raise RuntimeError('router validation order mismatch')
    c0,s0seed=expert(OUTPUT/'predictions/C0R_2017/expert.npz',ids)
    c1,s1seed=expert(OUTPUT/'predictions'/(window+'_C1')/'expert.npz',ids)
    c2,s2seed=expert(OUTPUT/'predictions'/(window+'_C2')/'expert.npz',ids)
    arms={'U0_2017_REPLAY':(router['s0'],router['old_route'],c0,s0seed),
        'U1_STAGE1_PLUS_REQUIRED_REANCHOR':(router['s1'],router['reanchor_route'],c0,s0seed),
        'U2_NEW_ROUTER_OLD_EXPERT':(router['s1'],router['new_route'],c0,s0seed),
        'U3_FULL_ROLLING_C1':(router['s1'],router['new_route'],c1,s1seed),
        'U4_FULL_YEAR_BALANCED_C2':(router['s1'],router['new_route'],c2,s2seed)}
    hb=data['historical_base'][va];he=data['historical_expert'][va];hm=data['historical_members'][va]
    if not np.allclose(hm[:,3],hb,rtol=0,atol=1e-10):raise RuntimeError('historical reference SVM anchor differs')
    hr=top_budget(mean_opposition(hm,hb),ids)
    arms['H0_HISTORICAL_DESCRIPTIVE']=(hb,hr,he,None)
    rows=[];per=[];seedrows=[]
    for arm,(base,route,e,seeds) in arms.items():
        rows.append(dict(window=window,year=int(data['years'][va][0]),arm=arm,**cascade(y,base,e,route)))
        final=np.where(route,e,base)
        per.extend(dict(window=window,arm=arm,sha256=s,year=int(data['years'][i]),label=int(data['y'][i]),stage1_probability=float(base[k]),stage1_prediction=int(base[k]>=.5),stage1_correct=int((base[k]>=.5)==y[k]),routed=int(route[k]),expert_probability=float(e[k]),final_probability=float(final[k]),final_prediction=int(final[k]>=.5),correct=int((final[k]>=.5)==y[k])) for k,(s,i) in enumerate(zip(ids,va)))
        if seeds is not None:
            seedrows.extend(dict(window=window,arm=arm,seed=900000+k,**cascade(y,base,p,route)) for k,p in enumerate(seeds))
    diffs=[]
    for r in paired_intervals(y,router['s0'],np.ones(len(y),bool),router['s1'],router['s0'],repeats=bootstrap):
        diffs.append(dict(window=window,year=int(data['years'][va][0]),contrast='S1 minus S0R',scope='pure Stage-I; Fixed/Broken relative to S0R, not expert gain',**r))
    # Expert-only contrasts: upstream masks and probabilities are exactly identical.
    for a,b in [('U3_FULL_ROLLING_C1','U2_NEW_ROUTER_OLD_EXPERT'),('U4_FULL_YEAR_BALANCED_C2','U3_FULL_ROLLING_C1')]:
        ba,ra,ea,_=arms[a];bb,rb,eb,_=arms[b]
        if not np.array_equal(ba,bb) or not np.array_equal(ra,rb):raise RuntimeError('expert contrast upstream differs')
        for r in paired_intervals(y,ba,ra,ea,eb,repeats=bootstrap):diffs.append(dict(window=window,year=int(data['years'][va][0]),contrast=a+' minus '+b,scope='expert-only; fixed Stage-I/route/SHA',**r))
    # Router auxiliary-refit contrast: same base and expert, different route masks.
    base=router['s1'];ra=router['new_route'];rb=router['reanchor_route']
    finala=np.where(ra,c0,base);finalb=np.where(rb,c0,base)
    for r in paired_intervals(y,base,ra|rb,finala,finalb,repeats=bootstrap):
        if r['metric'].endswith('AUC'):continue
        diffs.append(dict(window=window,year=int(data['years'][va][0]),contrast='U2 minus U1',scope='aux-router refit; same SVM/old expert; union-mask bootstrap',**r))
    write_csv(root/'cascade_results.csv',rows);write_csv(root/'per_apk_predictions.csv',per);write_csv(root/'ablation_results.csv',diffs);write_csv(root/'seed_sensitivity.csv',seedrows)
    print('MYD_CASCADE_DONE',window,flush=True)
