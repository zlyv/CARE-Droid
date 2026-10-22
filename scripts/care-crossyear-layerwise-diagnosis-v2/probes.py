"""Training-window-only linear/MLP readability probes, controls and geometry."""
from __future__ import annotations
import time,warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from diagnostic_common import *
from extract_layers import LAYERS

def auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
def measures(y,p,route=None,stage=None):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    d=dict(n=len(y),benign=int(sum(y==0)),malware=int(sum(y==1)),BA=float(balanced_accuracy_score(y,pred)),AUC=auc(y,p),F1=float(f1_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),specificity=float(tn/max(tn+fp,1)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),probability_mean=float(np.mean(p)),probability_std=float(np.std(p)))
    if route is not None:
        m=np.asarray(route,bool);d['routed_n']=int(m.sum());d['routed_AUC']=auc(y[m],p[m]) if m.any() else float('nan');d['routed_errors']=int(np.sum(pred[m]!=y[m]))
        if stage is not None:
            s=np.asarray(stage,int)
            for name,mask in [('FN_vs_TN',m&(s==0)),('TP_vs_FP',m&(s==1))]:
                d[name+'_n']=int(mask.sum());d[name+'_positive']=int(np.sum(y[mask]==1));d[name+'_negative']=int(np.sum(y[mask]==0));d[name+'_AUC']=auc(y[mask],p[mask]) if mask.any() else float('nan')
    return d
def linear(width=None):
    steps=[('scale',StandardScaler())]
    if width is not None:steps.append(('pca',PCA(n_components=width,svd_solver='full')))
    steps.append(('fit',LogisticRegression(C=1.0,max_iter=1000,solver='liblinear',random_state=0)))
    return Pipeline(steps)
def mlp(seed):return Pipeline([('scale',StandardScaler()),('fit',MLPClassifier(hidden_layer_sizes=(16,),alpha=.01,max_iter=120,random_state=seed,early_stopping=False))])
def train_cv(x,y):
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=1127);p=np.full(len(y),np.nan)
    for tr,te in cv.split(x,y):m=linear();m.fit(x[tr],y[tr]);p[te]=m.predict_proba(x[te])[:,1]
    return auc(y,p)
def geometry(xtr,ytr,xva,yva):
    scaler=StandardScaler().fit(xtr);a=scaler.transform(xtr);b=scaler.transform(xva)
    # The reference set and both class centroids are training-only.
    centers=np.stack([a[ytr==c].mean(0) for c in (0,1)])
    dist=((b[:,None,:]-centers[None,:,:])**2).sum(2)**.5
    nearest=(dist[:,1]<dist[:,0]).astype(int)
    from sklearn.metrics import pairwise_distances
    ds=pairwise_distances(b,a);ix=np.argsort(ds,axis=1)[:,:5]
    purity=float(np.mean(ytr[ix]==yva[:,None]))
    _,s,_=np.linalg.svd(a-a.mean(0),full_matrices=False);p=s*s/max(np.sum(s*s),1e-12);rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))
    return dict(train_n=len(a),validation_n=len(b),centroid_distance=float(np.linalg.norm(centers[1]-centers[0])),validation_nearest_train_centroid_accuracy=float(np.mean(nearest==yva)),validation_nearest5_train_label_purity=purity,train_effective_rank=rank,train_mean_norm=float(np.mean(np.linalg.norm(a,axis=1))),validation_mean_norm=float(np.mean(np.linalg.norm(b,axis=1))),mean_shift_norm=float(np.linalg.norm(b.mean(0)-a.mean(0))),same_model_coordinate_system=True,train_embedding_in_sample=True,validation_embedding_out_of_sample=True)

def main():
    if read_json(OUTPUT/'layer_extraction_audit.json')['status']!='PASS':raise RuntimeError('layer extraction gate failed')
    start=time.monotonic();cases=read_csv(OUTPUT/'per_apk_error_contract.csv');metrics=[];conditional=[];gaps=[];geoms=[];controls=[];background=[];predictions=[];runtimes=[]
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    for tag in TAGS:
        with np.load(OUTPUT/'embeddings'/tag/'train.npz',allow_pickle=False) as z:train={k:z[k] for k in z.files}
        with np.load(OUTPUT/'embeddings'/tag/'validation.npz',allow_pickle=False) as z:valid={k:z[k] for k in z.files}
        trids=list(map(str,train['ids']));vaids=list(map(str,valid['ids']))
        if set(trids)&set(vaids) or int(train['year'].max())>=int(valid['year'].min()):raise RuntimeError('temporal or SHA leakage')
        source={r['sha256']:r for r in cases if r['tag']==tag}
        if set(vaids)!=set(source):raise RuntimeError('validation SHA contract mismatch')
        ordered=[source[s] for s in vaids];route=np.array([int(r['routed']) for r in ordered],bool);stage=np.array([int(r['stage1_prediction']) for r in ordered]);ytr=train['y'].astype(int);yva=valid['y'].astype(int)
        if not np.array_equal(yva,np.array([int(r['label']) for r in ordered])):raise RuntimeError('label alignment mismatch')
        original=1/(1+np.exp(-valid['L6_DIRECT_LOGIT'][:,0].astype(float)))
        ensemble=np.array([float(r['expert_probability']) for r in ordered])
        baseline=measures(yva,original,route,stage)
        metrics.append(dict(tag=tag,year=int(valid['year'][0]),layer='L6_DIRECT_LOGIT',probe='FROZEN_ORIGINAL',capacity='frozen',train_n=len(ytr),input_width=1,params=0,train_cv_auc=float('nan'),seconds=0.,**baseline))
        metrics.append(dict(tag=tag,year=int(valid['year'][0]),layer='L6_DIRECT_ENSEMBLE',probe='FROZEN_3SEED',capacity='frozen',train_n=len(ytr),input_width=1,params=0,train_cv_auc=float('nan'),seconds=0.,**measures(yva,ensemble,route,stage)))
        predictions.extend(dict(tag=tag,year=int(valid['year'][i]),sha256=s,layer='L6_DIRECT_LOGIT',probe='FROZEN_ORIGINAL',probability=float(original[i])) for i,s in enumerate(vaids))
        predictions.extend(dict(tag=tag,year=int(valid['year'][i]),sha256=s,layer='L6_DIRECT_ENSEMBLE',probe='FROZEN_3SEED',probability=float(ensemble[i])) for i,s in enumerate(vaids))
        for layer in LAYERS[:-1]:
            xtr=train[layer];xva=valid[layer]
            if xtr.ndim!=2 or xtr.shape[1]!=xva.shape[1] or not np.isfinite(xtr).all() or not np.isfinite(xva).all():raise RuntimeError('invalid matrix '+tag+'/'+layer)
            t=time.monotonic();cv_auc=train_cv(xtr,ytr)
            for capacity in ('raw','pca32'):
                if capacity=='pca32' and xtr.shape[1]<=32:continue
                n=32 if capacity=='pca32' else None;m=linear(n);m.fit(xtr,ytr);prob=m.predict_proba(xva)[:,1]
                d=measures(yva,prob,route,stage);params=(n if n else xtr.shape[1])+1
                metrics.append(dict(tag=tag,year=int(valid['year'][0]),layer=layer,probe='LR',capacity=capacity,train_n=len(ytr),input_width=xtr.shape[1],params=params,train_cv_auc=cv_auc if capacity=='raw' else float('nan'),seconds=time.monotonic()-t,**d))
                if capacity=='raw':
                    conditional.append(dict(tag=tag,year=int(valid['year'][0]),layer=layer,probe='LR',FN_vs_TN_AUC=d['FN_vs_TN_AUC'],TP_vs_FP_AUC=d['TP_vs_FP_AUC'],original_FN_vs_TN_AUC=baseline['FN_vs_TN_AUC'],original_TP_vs_FP_AUC=baseline['TP_vs_FP_AUC'],routed_AUC=d['routed_AUC']))
                    gaps.append(dict(tag=tag,year=int(valid['year'][0]),layer=layer,train_internal_3fold_AUC=cv_auc,future_AUC=d['AUC'],gap=cv_auc-d['AUC'],upstream_train_embeddings_in_sample=True))
                    predictions.extend(dict(tag=tag,year=int(valid['year'][i]),sha256=s,layer=layer,probe='LR',probability=float(prob[i])) for i,s in enumerate(vaids))
            geoms.append(dict(tag=tag,year=int(valid['year'][0]),layer=layer,**geometry(xtr,ytr,xva,yva)))
            if layer in ('P0_P31_SIZE','L0_FIXED_SUMMARY','L1_HOST_ENCODER_SUMMARY','L4_ATTENTION_APK','L5_EXACT_HEAD_INPUT'):
                for seed in (11,29,47):
                    t2=time.monotonic();mm=mlp(seed);mm.fit(xtr,ytr);pr=mm.predict_proba(xva)[:,1]
                    metrics.append(dict(tag=tag,year=int(valid['year'][0]),layer=layer,probe='MLP16',capacity='fixed',seed=seed,train_n=len(ytr),input_width=xtr.shape[1],params=(xtr.shape[1]+1)*16+17,train_cv_auc=float('nan'),seconds=time.monotonic()-t2,**measures(yva,pr,route,stage)))
            runtimes.append(dict(tag=tag,layer=layer,linear_seconds=time.monotonic()-t,train_n=len(ytr),validation_n=len(yva)))
            print('CLD_PROBE',tag,layer,cv_auc,flush=True)
        # Fixed controls: labels permuted only in training; test labels used only for metrics.
        for layer in ('P0_P31_SIZE','L0_FIXED_SUMMARY','L5_EXACT_HEAD_INPUT'):
            xtr=train[layer];xva=valid[layer];real=auc(yva,linear().fit(xtr,ytr).predict_proba(xva)[:,1]);null=[]
            for repeat in range(20):
                yp=np.random.RandomState(8171+repeat).permutation(ytr);mm=linear().fit(xtr,yp);null.append(auc(yva,mm.predict_proba(xva)[:,1]))
            controls.append(dict(tag=tag,layer=layer,control='train_label_shuffle20',real_AUC=real,null_mean=float(np.mean(null)),null_95pct=float(np.quantile(null,.95)),real_exceeds_null95=bool(real>np.quantile(null,.95)),repeats=20))
        # Four original Host-background responses summarized by mean/std/max.
        bx=train['background'];bv=valid['background'];lh=train['L1_HOST_ENCODER_SUMMARY'];lv=valid['L1_HOST_ENCODER_SUMMARY']
        raw=train['L0_FIXED_SUMMARY'];rawv=valid['L0_FIXED_SUMMARY']
        base_areas={'background_only':(bx,bv),'L0_raw_only':(raw,rawv),'L0_plus_background':(np.c_[raw,bx],np.c_[rawv,bv]),'L1_encoded_only':(lh,lv),'L1_plus_background':(np.c_[lh,bx],np.c_[lv,bv])}
        rng=np.random.RandomState(9227);bt=bx[rng.permutation(len(bx))];bvv=bv[rng.permutation(len(bv))]
        base_areas['L1_plus_unpaired_background']=(np.c_[lh,bt],np.c_[lv,bvv])
        for name,(xx,zz) in base_areas.items():
            p=linear().fit(xx,ytr).predict_proba(zz)[:,1];d=measures(yva,p,route,stage)
            background.append(dict(tag=tag,year=int(valid['year'][0]),arm=name,input_width=xx.shape[1],**d))
        print('CLD_WINDOW_PROBES_DONE',tag,flush=True)
    write_csv(OUTPUT/'layerwise_information_metrics.csv',metrics)
    write_csv(OUTPUT/'conditional_auc_by_layer.csv',conditional)
    write_csv(OUTPUT/'train_vs_future_probe_gap.csv',gaps)
    write_csv(OUTPUT/'representation_geometry.csv',geoms)
    write_csv(OUTPUT/'probe_control_results.csv',controls)
    write_csv(OUTPUT/'background_response_controls.csv',background)
    write_csv(OUTPUT/'probe_predictions.csv',predictions)
    write_csv(OUTPUT/'runtime_cost.csv',runtimes+[dict(phase='all_probes',seconds=time.monotonic()-start)])
    print('CLD_PROBES_DONE',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
