#!/usr/bin/env python3
"""Analyze frozen Bank support, cross-bank overlap, existing responses and errors.

Writes only to outputs/care_background_prototype_audit_v1. Uses fixed diagnostic
distance quantiles and pre-existing error/pair cohorts; never trains CARE-Droid.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter,defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_assets import OUT,ROOT,MODEL,TAGS,SEEDS,read_csv,write_csv,write_json
sys.path.insert(0,str(ROOT/'scripts'))
import run_care_joint_dualprototype_ablation as v2

EPS=0.5
RNG=np.random.RandomState(20261016)


def wilson(k,n,z=1.96):
    if n==0:return (None,None)
    p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0.,mid-rad),min(1.,mid+rad))


def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(y))==2 else None


def jaccard(a,b):
    union=int(np.sum(a|b));return float(np.sum(a&b)/union) if union else None


def finite_spearman(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<4 or np.std(a[ok])==0 or np.std(b[ok])==0:return None
    return float(spearmanr(a[ok],b[ok]).statistic)


def support_type(b,m,ci_b,ci_m,n_b,n_m):
    total=b+m;qb=b/n_b;qm=m/n_m
    if total<20:return 'INSUFFICIENT_SUPPORT'
    if qb>=.5 and qm>=.5 and abs(qb-qm)<.1:return 'HIGH_SUPPORT_SHARED'
    if qb-qm>=.1 and ci_b[0]>ci_m[1]:return 'BENIGN_ASSOCIATED'
    if qm-qb>=.1 and ci_m[0]>ci_b[1]:return 'MALWARE_ASSOCIATED'
    return 'LOW_SUPPORT_UNCERTAIN'


def load_bank(tag,seed):
    with np.load(OUT/'bank_cache'/(tag+'_'+str(seed)+'.npz'),allow_pickle=False) as a:
        d={k:a[k] for k in a.files}
    with (MODEL/'models/stage2'/tag/(str(seed)+'.pkl')).open('rb') as f:
        obj=pickle.load(f)
    if d['train_n']!=len(obj['metadata']['train_sha']):raise RuntimeError('Bank cache lineage mismatch')
    if list(d['ids'][:int(d['train_n'])])!=[str(x).lower() for x in obj['metadata']['train_sha']]:
        raise RuntimeError('Bank cache train SHA order mismatch')
    return d,obj


def support_and_overlap(tag,seed,d,obj):
    n=int(d['train_n']);y=d['labels'][:n];n_b=int(np.sum(y==0));n_m=int(np.sum(y==1))
    if n_b<100 or n_m<100:raise RuntimeError('unexpected training class counts')
    thr=d['thresholds'];state=obj['background']['state']
    sp=obj['background']['sp'];bp=obj['background']['bp']
    ss=d['min_shared'][:n]<=thr[1,0];eb=d['min_external'][:n]<=thr[1,1]
    support_rows=[];sensitivity=[];classes={}
    for bank,values,k,ind in [('Shared',d['min_shared'],32,0),('External',d['min_external'],24,1)]:
        profiles=sp if bank=='Shared' else bp
        for j in range(k):
            a=values[:n,j]<=thr[1,ind]
            bc=int(np.sum(a&(y==0)));mc=int(np.sum(a&(y==1)))
            cb=wilson(bc,n_b);cm=wilson(mc,n_m)
            kind=support_type(bc,mc,cb,cm,n_b,n_m);classes[(bank,j)]=kind
            row={'tag':tag,'seed':seed,'bank':bank,'prototype_id':j,
                'train_source':'balanced benign+malware KMeans' if bank=='Shared' else 'benign-only KMeans',
                'role':int(np.argmax(profiles[j])) if profiles[j].sum()>0 else 'UNKNOWN',
                'profile_nonempty':bool(profiles[j].sum()>0),
                'train_benign_APKs':n_b,'train_malware_APKs':n_m,
                'benign_support_APKs':bc,'malware_support_APKs':mc,
                'q_B':bc/n_b,'q_M':mc/n_m,
                'D_log_smoothed':float(np.log(((bc+EPS)/(n_b+2*EPS))/((mc+EPS)/(n_m+2*EPS)))),
                'q_B_CI_low':cb[0],'q_B_CI_high':cb[1],
                'q_M_CI_low':cm[0],'q_M_CI_high':cm[1],
                'support_type':kind,
                'threshold':float(thr[1,ind]),
                'self_source_exclusion':'UNAVAILABLE_KMEANS_CENTER_NOT_IDENTIFIABLE'}
            support_rows.append(row)
            for qidx,qname in enumerate(('q25','q50','q75')):
                ab=values[:n,j]<=thr[qidx,ind]
                sensitivity.append({'tag':tag,'seed':seed,'bank':bank,'prototype_id':j,
                    'threshold_quantile':qname,'threshold':float(thr[qidx,ind]),
                    'benign_support_APKs':int(np.sum(ab&(y==0))),
                    'malware_support_APKs':int(np.sum(ab&(y==1))),
                    'q_B':float(np.mean(ab[y==0])),'q_M':float(np.mean(ab[y==1]))})
    matrix=v2.pairwise_rms(state['shared_centers'],state['benign_centers'])
    if matrix.shape!=(32,24):raise RuntimeError('cross-bank matrix width changed')
    sr=np.argmax(sp,axis=1);br=np.argmax(bp,axis=1)
    compatible=(sr[:,None]==br[None,:])&(sp.sum(1)[:,None]>0)&(bp.sum(1)[None,:]>0)
    # Unlabeled geometry-only proximity scale from within-Bank nearest centers.
    ss_dist=v2.pairwise_rms(state['shared_centers'],state['shared_centers']);np.fill_diagonal(ss_dist,np.inf)
    bb_dist=v2.pairwise_rms(state['benign_centers'],state['benign_centers']);np.fill_diagonal(bb_dist,np.inf)
    scale=float(np.median(np.concatenate([ss_dist.min(1),bb_dist.min(1)])))
    matrix_rows=[];overlap=[];unique=[];controls=[];nearest={}
    for j in range(24):
        legal=np.flatnonzero(compatible[:,j]);
        if not len(legal):
            nearest[j]=None;continue
        k=int(legal[np.argmin(matrix[legal,j])]);nearest[j]=k
        e=eb[:,j];s=ss[:,k]
        metric=jaccard(e,s)
        compatible_union=np.any(ss[:,legal],axis=1)
        unique_count=int(np.sum(e & ~compatible_union))
        qdiff=abs(float(np.mean(e[y==0]))-float(np.mean(s[y==0])))+abs(float(np.mean(e[y==1]))-float(np.mean(s[y==1])))
        overlap.append({'tag':tag,'seed':seed,'external_id':j,'shared_id':k,
            'role':int(br[j]),'distance':float(matrix[k,j]),
            'distance_relative_to_within_bank_nearest_scale':float(matrix[k,j]/max(scale,1e-9)),
            'distance_close_unlabeled':bool(matrix[k,j]<=scale),
            'jaccard_train_APK_support':metric,
            'jaccard_benign':jaccard(e[y==0],s[y==0]),
            'jaccard_malware':jaccard(e[y==1],s[y==1]),
            'external_support_APKs':int(e.sum()),'shared_support_APKs':int(s.sum()),
            'both_support_APKs':int(np.sum(e&s)),
            'external_unique_vs_nearest_shared':int(np.sum(e&~s)),
            'external_unique_vs_any_role_compatible_shared':unique_count,
            'class_support_total_abs_difference':qdiff,
            'external_support_type':classes[('External',j)],
            'shared_support_type':classes[('Shared',k)]})
        unique.append({'tag':tag,'seed':seed,'external_id':j,
            'external_support_APKs':int(e.sum()),
            'unique_vs_any_role_compatible_shared_APKs':unique_count,
            'unique_fraction':unique_count/int(e.sum()) if e.sum() else None,
            'role_compatible_shared_candidates':len(legal)})
        for repeat in range(100):
            c=int(RNG.choice(legal));z=jaccard(e,ss[:,c])
            controls.append({'tag':tag,'seed':seed,'external_id':j,'repeat':repeat,
                'sampled_shared_id':c,'role_matched':True,
                'distance':float(matrix[c,j]),'jaccard':z,
                'true_nearest_shared_id':k})
    for k in range(32):
        for j in range(24):
            matrix_rows.append({'tag':tag,'seed':seed,'shared_id':k,'external_id':j,
                'shared_role':int(sr[k]) if sp[k].sum()>0 else 'UNKNOWN',
                'external_role':int(br[j]) if bp[j].sum()>0 else 'UNKNOWN',
                'role_compatible':bool(compatible[k,j]),'distance_RMS_107':float(matrix[k,j]),
                'within_bank_nearest_scale':scale,
                'training_support_jaccard':jaccard(ss[:,k],eb[:,j])})
    for row in support_rows:
        if row['bank']=='External' and row['prototype_id'] in nearest and nearest[row['prototype_id']] is not None:
            o=next(x for x in overlap if x['external_id']==row['prototype_id'])
            row.update(nearest_other_bank_id=o['shared_id'],nearest_other_bank_distance=o['distance'],
                       nearest_other_bank_support_jaccard=o['jaccard_train_APK_support'],
                       unique_coverage_APKs=o['external_unique_vs_any_role_compatible_shared'])
        elif row['bank']=='Shared':
            legal=np.flatnonzero(compatible[row['prototype_id']])
            if len(legal):
                j=int(legal[np.argmin(matrix[row['prototype_id'],legal])])
                row.update(nearest_other_bank_id=j,nearest_other_bank_distance=float(matrix[row['prototype_id'],j]),
                           nearest_other_bank_support_jaccard=jaccard(ss[:,row['prototype_id']],eb[:,j]),
                           unique_coverage_APKs=int(np.sum(ss[:,row['prototype_id']]&~eb[:,j])))
    return (support_rows,sensitivity,matrix_rows,overlap,unique,controls,classes,nearest,ss,eb)


def make_apk_rows(tag,seed,d,support_classes,nearest,errors):
    n=int(d['train_n']);thr=d['thresholds'];rows=[]
    for i in range(len(d['ids'])):
        sid=str(d['ids'][i]);s=d['min_shared'][i]<=thr[1,0];e=d['min_external'][i]<=thr[1,1]
        matched=[j for j in range(24) if e[j] and nearest.get(j) is not None]
        overlap=float(np.mean([s[nearest[j]] for j in matched])) if matched else None
        row={'tag':tag,'seed':seed,'split':'TRAIN' if i<n else 'VALIDATION',
            'sha256':sid,'year':int(d['years'][i]),'label':int(d['labels'][i]),
            'host_count':int(d['host_count'][i]),
            'role_mean_dominant':int(np.argmax(d['role_mean'][i])),
            'mean_external':float(d['response'][i,0]),
            'mean_internal':float(d['response'][i,1]),
            'mean_commonness':float(d['response'][i,2]),
            'mean_residual':float(d['response'][i,3]),
            'shared_supported_count':int(np.sum(s)),
            'external_supported_count':int(np.sum(e)),
            'high_support_shared_count':sum(bool(s[j]) and support_classes[('Shared',j)]=='HIGH_SUPPORT_SHARED' for j in range(32)),
            'benign_associated_external_count':sum(bool(e[j]) and support_classes[('External',j)]=='BENIGN_ASSOCIATED' for j in range(24)),
            'external_shared_overlap_fraction':overlap,
            'mean_shared_min_distance':float(np.min(d['min_shared'][i])),
            'mean_external_min_distance':float(np.min(d['min_external'][i]))}
        if i>=n and tag in errors:
            z=errors[tag].get(sid)
            if z is None:raise RuntimeError('frozen error contract missing '+tag+' '+sid)
            row.update({'routed':int(z['routed']),'stage1_probability':float(z['stage1_probability']),
                        'C43_probability':float(z['expert_probability']),
                        'category':z['category'],'stage1_prediction':int(z['stage1_prediction']),
                        'C43_prediction':int(z['expert_prediction'])})
        rows.append(row)
    return rows


def score_probe(xtr,ytr,xte,yte):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='liblinear',max_iter=1000))
    model.fit(xtr,ytr);pred=model.predict_proba(xte)[:,1]
    return {'AUC':safe_auc(yte,pred),'BA':float(balanced_accuracy_score(yte,pred>=.5))}


def incremental_probes(tag,seed,d):
    n=int(d['train_n']);ytr=d['labels'][:n];yte=d['labels'][n:]
    size=np.log1p(d['host_count'])[:,None]
    p0=np.concatenate([d['role_mean'],size],axis=1)
    c=d['response']
    feats={'P0_ROLE_SIZE':p0,'P1_PLUS_SHARED_COMMONNESS':np.concatenate([p0,c[:,2:3]],1),
           'P2_PLUS_EXTERNAL':np.concatenate([p0,c[:,2:3],c[:,0:1]],1),
           'P3_PLUS_INTERNAL_RESIDUAL':np.concatenate([p0,c[:,2:3],c[:,0:1],c[:,1:2],c[:,3:4]],1)}
    rows=[];ctrl=[]
    for arm,x in feats.items():
        stat=score_probe(x[:n],ytr,x[n:],yte)
        rows.append({'tag':tag,'seed':seed,'arm':arm,'train_n':n,'validation_n':len(yte),
                     'feature_width':x.shape[1],**stat,
                     'interpretation':'diagnostic readout, not original C43 causal ablation'})
    # Fixed same-width, no-information column and conditional train-only shuffles.
    p1=feats['P1_PLUS_SHARED_COMMONNESS']
    for rep in range(20):
        rng=np.random.RandomState(20261016+rep)
        null=np.zeros(len(d['ids']))
        bins=np.digitize(size[:n,0],np.quantile(size[:n,0],[.25,.5,.75]))
        role=np.argmax(d['role_mean'][:n],axis=1)
        keys=role*4+bins
        for key in np.unique(keys):
            idx=np.flatnonzero(keys==key)
            null[idx]=c[rng.permutation(idx),0]
        null[n:]=c[n:,0]
        xx=np.concatenate([p1,null[:,None]],axis=1)
        result=score_probe(xx[:n],ytr,xx[n:],yte)
        ctrl.append({'tag':tag,'seed':seed,'control':'TRAIN_EXTERNAL_ROLE_SIZE_SHUFFLE',
                     'repeat':rep,'n':len(yte),**result})
        noise=np.random.RandomState(20262016+rep).normal(size=len(d['ids']))
        xx=np.concatenate([p1,noise[:,None]],axis=1)
        result=score_probe(xx[:n],ytr,xx[n:],yte)
        ctrl.append({'tag':tag,'seed':seed,'control':'MATCHED_ONE_COLUMN_NOISE',
                     'repeat':rep,'n':len(yte),**result})
    return rows,ctrl


def dependency(tag,seed,rows):
    out=[]
    for split in ('TRAIN','VALIDATION'):
        part=[r for r in rows if r['split']==split]
        def vec(key):return np.asarray([r[key] if r[key] is not None else np.nan for r in part],float)
        ext=vec('mean_external');common=vec('mean_commonness');res=vec('mean_residual')
        shared=vec('shared_supported_count');size=np.log1p(vec('host_count'))
        role=np.asarray([r['role_mean_dominant'] for r in part])
        base=np.column_stack([size,np.eye(31)[role]])
        def partial(a,b):
            if len(a)<4:return None
            ra=a-LinearRegression().fit(base,a).predict(base)
            rb=b-LinearRegression().fit(base,b).predict(base)
            return finite_spearman(ra,rb)
        out += [{'tag':tag,'seed':seed,'split':split,'n':len(part),'pair':'commonness_vs_external',
                 'spearman':finite_spearman(common,ext),'partial_spearman_size_dominant_role':partial(common,ext)},
                {'tag':tag,'seed':seed,'split':split,'n':len(part),'pair':'commonness_vs_shared_support_count',
                 'spearman':finite_spearman(common,shared),'partial_spearman_size_dominant_role':partial(common,shared)},
                {'tag':tag,'seed':seed,'split':split,'n':len(part),'pair':'residual_vs_external',
                 'spearman':finite_spearman(res,ext),'partial_spearman_size_dominant_role':partial(res,ext)}]
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collect',action='store_true',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.device!='cpu' or args.workers!=1:raise RuntimeError('diagnostic fixed to one CPU worker')
    contracts=read_csv(ROOT/'outputs/care_crossyear_layerwise_diagnosis_v2/per_apk_error_contract.csv')
    errors={tag:{r['sha256']:r for r in contracts if r['tag']==tag} for tag in TAGS[1:]}
    outputs=defaultdict(list);all_apk=[];manifests=[]
    for tag in TAGS:
        for seed in SEEDS:
            d,obj=load_bank(tag,seed)
            derived=support_and_overlap(tag,seed,d,obj)
            names=('prototype_class_support','prototype_support_sensitivity',
                   'shared_external_distance_matrix','shared_external_overlap',
                   'prototype_unique_coverage','role_matched_overlap_controls')
            for name,part in zip(names,derived[:6]):outputs[name].extend(part)
            _,_,_,_,_,_,classes,nearest,_,_=derived
            rows=make_apk_rows(tag,seed,d,classes,nearest,errors)
            all_apk.extend(rows)
            outputs['background_dependency_metrics'].extend(dependency(tag,seed,rows))
            probe,ctrl=incremental_probes(tag,seed,d)
            outputs['background_incremental_information'].extend(probe)
            outputs['background_probe_controls'].extend(ctrl)
            manifests.append(json.loads((OUT/'bank_cache'/(tag+'_'+str(seed)+'_manifest.json')).read_text()))
            print('PROTOTYPE_ANALYSIS_BANK_DONE',tag,seed,flush=True)
    for name,part in outputs.items():write_csv(OUT/(name+'.csv'),part)
    write_csv(OUT/'per_apk_background_responses.csv',all_apk)
    write_json(OUT/'prototype_bank_manifest.json',{'banks':manifests,
        'feature_space':'same train-fitted standardized 107-dimensional Compact+Host space within each tag/seed; no cross-checkpoint coordinate comparison',
        'prototype_identity':'KMeans centers, not original Hosts'})
    provenance=[]
    for tag in TAGS:
        for seed in SEEDS:
            provenance += read_csv(OUT/'bank_cache'/(tag+'_'+str(seed)+'_provenance.csv'))
    write_csv(OUT/'prototype_provenance.csv',provenance)
    write_json(OUT/'analysis_phase.json',{'status':'PASS','banks':len(manifests),
        'train_only_support':True,'future_labels_used_for_thresholds':False,
        'new_model_training':False,'per_apk_rows':len(all_apk)})
    print('PROTOTYPE_ANALYSIS_DONE',len(all_apk),flush=True)


if __name__=='__main__':main()
