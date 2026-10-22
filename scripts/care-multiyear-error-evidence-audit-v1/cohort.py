"""Freeze the 63 persistent cases, 10 regressed FNs, and explicit same-year matched controls without fitting."""
from __future__ import annotations
from collections import Counter
import numpy as np
from audit_common import *

def groups():
    by={}
    for r in read_csv(BASE/'per_apk_predictions.csv'):
        if r['arm'] in ARMS and int(r['year'])>=2019:
            by.setdefault(r['sha256'],{})[r['arm']]=r
    result={}
    for sid,g in by.items():
        if set(g)!=set(ARMS):raise RuntimeError('expert arm missing '+sid)
        rs=[g[a] for a in ARMS]
        if len({(r['label'],r['stage1_probability'],r['routed'],r['year']) for r in rs})!=1:raise RuntimeError('upstream differs '+sid)
        if not int(rs[0]['routed']):continue
        y=int(rs[0]['label']);b=int(rs[0]['stage1_prediction']);wrong=[int(r['final_prediction'])!=y for r in rs]
        category=None
        if all(wrong):category='PERSISTENT_FN' if b==0 and y==1 else 'PERSISTENT_FP' if b==1 and y==0 else 'PERSISTENT_BROKEN_TP' if y==1 else 'PERSISTENT_BROKEN_TN'
        elif y==1 and b==0 and not wrong[0] and wrong[1]:category='REGRESSED_FN'
        result[sid]=dict(sha256=sid,year=int(rs[0]['year']),window=rs[0]['window'],label=y,base_prediction=b,category=category or '',C0_probability=float(rs[0]['expert_probability']),C1_probability=float(rs[1]['expert_probability']),C2_probability=float(rs[2]['expert_probability']),C1_correct=not wrong[1])
    counts=Counter(r['category'] for r in result.values() if r['category'])
    expected={'PERSISTENT_FN':35,'PERSISTENT_FP':18,'PERSISTENT_BROKEN_TP':3,'PERSISTENT_BROKEN_TN':7,'REGRESSED_FN':10}
    if dict(counts)!=expected:raise RuntimeError('frozen failure cohort differs '+repr(dict(counts)))
    return result

def descriptors(data):
    # A matching/coverage sketch only, not a classifier or new model feature.
    # The reference scaler is fixed to 2017, so distances share one raw coordinate system.
    values=data['host_values'];offset=data['host_offsets'];rows=[]
    for i in range(len(data['ids'])):
        b=values[offset[i]:offset[i+1]]
        rows.append(np.r_[b[:,:43].mean(0),b[:,:43].std(0),b[:,43:74].mean(0),b[:,74:105].mean(0),data['p31'][i],np.log1p(data['real_host_count'][i])])
    x=np.asarray(rows,float);train=x[data['years']==2017];mean=train.mean(0);std=train.std(0);std[std<1e-6]=1
    if x.shape[1]!=180 or not np.isfinite(x).all():raise RuntimeError('invalid frozen raw descriptor')
    return (x-mean)/std

def prepare(data):
    g=groups();position={s:i for i,s in enumerate(data['ids'])};x=descriptors(data);targets=sorted([r for r in g.values() if r['category']],key=lambda r:(r['year'],r['category'],r['sha256']))
    pairs=[];selected={r['sha256'] for r in targets};lookup={r['sha256']:r for r in targets}
    for r in targets:
        pair=dict(target_sha256=r['sha256'],year=r['year'],category=r['category'])
        for kind in ['same_label_anchor','opposite_label_control']:
            label=r['label'] if kind=='same_label_anchor' else 1-r['label']
            eligible=[a for a in g.values() if a['sha256']!=r['sha256'] and a['year']==r['year'] and a['label']==label and a['C1_correct']]
            exact=[a for a in eligible if a['base_prediction']==r['base_prediction']]
            pool=exact or eligible
            if not pool:pair[kind]='';pair[kind+'_match']='NO_CORRECT_ROUTED_CONTROL';continue
            dist=[float(np.sqrt(np.mean((x[position[r['sha256']]]-x[position[a['sha256']]])**2))) for a in pool]
            choice=min(range(len(pool)),key=lambda j:(dist[j],pool[j]['sha256']));a=pool[choice];sid=a['sha256']
            pair[kind]=sid;pair[kind+'_distance']=dist[choice];pair[kind+'_match']='SAME_YEAR_LABEL_BASE_SIDE_ROUTED_C1_CORRECT' if exact else 'BASE_SIDE_RELAXED_EXPLICITLY'
            selected.add(sid);lookup[sid]=a
        pairs.append(pair)
    roles=read_json(BASE/'dataset_audit.json')['role_names']
    lineage={r['sha256']:r for r in read_csv(PROJECT/'artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv')}
    members=[]
    for sid in sorted(selected):
        r=dict(lookup[sid]);r['member_id']='M_'+signature(['MEEA_V1',sid])[:16];r['apk_path']=lineage.get(sid,{}).get('absolute_path','');r['raw_apk_present']=bool(r['apk_path'] and Path(r['apk_path']).is_file());r['target']=sid in {a['sha256'] for a in targets};members.append(r)
    write_csv(OUTPUT/'private/cohort_key.csv',members);write_csv(OUTPUT/'private/matched_pairs_key.csv',pairs)
    summary=dict(target_count=len(targets),unique_members=len(members),unique_controls=len(selected-{r['sha256'] for r in targets}),category_counts=dict(Counter(r['category'] for r in targets)),by_year=dict(Counter(r['year'] for r in targets)),raw_apks_present=sum(r['raw_apk_present'] for r in members),missing_raw_apks=[r['sha256'] for r in members if not r['raw_apk_present']],matching='fixed raw 180-D sketch; 2017-only unsupervised scaling; same year, routed, C1-correct controls; base-side relaxation explicit',control_reuse=True,control_reuse_does_not_create_independent_samples=True,matching_not_class_direction_proof=True,labels_used_only_for_diagnostic_selection=True,year2018_excluded_because_experts_identical=True)
    dump(OUTPUT/'cohort_audit.json',summary)
    return members,pairs,x,roles
