#!/usr/bin/env python3
from __future__ import annotations

import collections, gzip, hashlib, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-role-directed-critical-host-v1'
OUT=ROOT/'outputs/care_role_directed_critical_host_v1'
BFA=ROOT/'scripts/care-background-functional-attribution-v1'
MYD=ROOT/'scripts/care-multiyear-development-v1'
SCBE_SCRIPT=ROOT/'scripts/care-scbe-v1'
H1_SCRIPT=ROOT/'scripts/care-host-bound-cfg-source-future-v3'
HOSTREL=ROOT/'scripts/care-di25-compact43-relationquery-source-v1'
SCBE=ROOT/'outputs/care_scbe_v1'
H1_OUT=ROOT/'outputs/care_host_bound_cfg_source_future_v3'
CFG_ABL=ROOT/'outputs/care_cfg_witness_fixed_ablation_v1'
CFG_FUTURE=ROOT/'outputs/care_cfg_witness_fixed_source_future_v2'
CFG_CACHE=ROOT/'artifacts/care_cfg_witness_all_years_v2/cache'
ENDPOINT=ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
sys.path[:0]=[str(HERE),str(BFA),str(MYD),str(SCBE_SCRIPT),str(H1_SCRIPT),str(HOSTREL)]

SPLITS=[f'S{i}' for i in range(1,6)]
SEEDS=[900000,900001,900002]
MANIFEST_SEEDS=[20261018,20261019,20261020]
ARMS=['R0_FROZEN_DIRECT_C43','R1_ROLE_DIRECTED_CRITICAL_HOST','R2_PAIR_QUANTITY_CONTROL',
      'R3_HOST_BINDING_SHUFFLE','R4_DIRECTION_SHUFFLE']
DIM=2*31*31+4

def jenc(x):
    if isinstance(x,dict): return {str(k):jenc(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jenc(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x,Path): return str(x)
    return x

def write_json(path,obj):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jenc(obj),ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def write_pickle(path,obj):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def load_pickle(path):
    with path.open('rb') as f:return pickle.load(f)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return {'path':str(path),'size':path.stat().st_size,'sha256':h.hexdigest()}

def stable_rng(seed,sha,token):
    raw=hashlib.sha256(f'{seed}:{sha}:{token}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8],'big'))

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);z=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return {'n':len(y),'BA':balanced_accuracy_score(y,z),'accuracy':accuracy_score(y,z),
            'F1':f1_score(y,z,zero_division=0),'AUC':roc_auc_score(y,p),
            'FNR':fn/(fn+tp),'FPR':fp/(fp+tn),'TN':tn,'FP':fp,'FN':fn,'TP':tp}

def preflight():
    if OUT.exists(): raise RuntimeError(f'refusing overwrite {OUT}')
    required=[SCBE,H1_OUT,CFG_ABL,CFG_FUTURE,CFG_CACHE,ENDPOINT]
    if not all(p.exists() for p in required): raise RuntimeError('missing upstream asset')
    if json.load(open(H1_OUT/'baseline_integrity_check.json')).get('status')!='PASS': raise RuntimeError('H1 integrity')
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir();(OUT/'logs').mkdir()
    key=[ENDPOINT,H1_OUT/'host_cfg_features.npz',H1_OUT/'protocol.json']
    for s in SPLITS+['T1']:
        for seed in SEEDS:key.append(SCBE/f'models/{s}/{seed}/M0_REPRODUCED.pkl')
    before=[digest(p) for p in key]
    write_json(OUT/'asset_manifest_before.json',before)
    protocol={'status':'FROZEN_ROLE_DIRECTED_CRITICAL_HOST','training_years':[2017],
      'evaluation_years':[2017,2018,2019,2020,2021],'stage1':'Manifest135 RF',
      'router':'OTHER3_OP25_REF using Manifest probabilities only','future_selection':False,
      'base_backbone_frozen':True,'backgrounds_changed':False,'prototype_bank_changed':False,
      'feature':'critical Host 2x31x31 directed/bidirectional role-witness counts + 4 scalars',
      'trainable':'single zero-initialized bounded linear residual head','arms':ARMS}
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'preflight.json',{'status':'PASS',**protocol})
    return before

def endpoint_orders(ids):
    from hostrel_features import reconstruct_host_relations
    wanted=set(ids);out={}
    with gzip.open(ENDPOINT,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);sha=str(r.get('sha256',''))
            if sha in wanted and r.get('status')=='OK':
                rec=reconstruct_host_relations(r['payload'],np.arange(31,dtype=np.int64))
                out[sha]=[str(x['host_id']) for x in rec]
    if set(out)!=wanted: raise RuntimeError(f'endpoint coverage {len(out)}/{len(wanted)}')
    return out

def model_critical(v4,model,graph_values,bags,dev):
    torch=v4.torch;graphs=v4.graph_tensors(graph_values,dev);result=[]
    model.eval()
    with torch.no_grad():
        for i,bag in enumerate(bags):
            graph={k:v[i] for k,v in graphs.items()};region=torch.as_tensor(bag,dtype=torch.float32,device=dev)
            cache,_=model.select_local_evidence(graph,region);result.append(int(torch.argmax(cache.score).cpu()))
    return np.asarray(result,int)

def base_predict_and_critical(pack,which):
    v4,dev=pack['v4'],pack['dev'];model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',pack['args']).to(dev)
    model.load_state_dict(pack['old']['state_dict'],strict=True);model.eval()
    idx=pack[which];glob=pack['global_'+which];graph=pack['graph_'+which];bags=pack['bags_'+which]
    p,_=v4.predict_model(model,glob,np.full(len(idx),.5),graph,bags,dev)
    c=model_critical(v4,model,graph,bags,dev)
    max_delta=float(model.max_delta.detach().cpu()) if hasattr(model.max_delta,'detach') else float(model.max_delta)
    del model
    if dev.type=='cuda':v4.torch.cuda.empty_cache()
    return np.asarray(p,float),c,max_delta

def future_pack(data,seed):
    from myd_common import runtime
    from train import global24
    from train_stage2 import raw_bags,transform_saved
    years=np.asarray(data['years'],int);train=np.flatnonzero(years==2017);target=np.flatnonzero(years>2017)
    v4,_,args=runtime();args.device='cuda';args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    dev=v4.resolve_device('cuda');old=load_pickle(SCBE/f'models/T1/{seed}/M0_REPRODUCED.pkl')
    _,glob=global24(v4,data,train,target);raw=raw_bags(data,target);bags=transform_saved(v4,raw,old['background'],args)
    graph=v4.transform_role_graphs(data['s7'][target],old['graph_scaler'])
    return {'v4':v4,'args':args,'dev':dev,'old':old,'val':target,'global_val':glob,'graph_val':graph,'bags_val':bags}

def chosen_ids(indices,critical,ids,orders):
    out=[]
    for idx,c in zip(indices,critical):
        sha=ids[int(idx)];hosts=orders[sha]
        if not hosts:
            if c!=0:raise RuntimeError('empty placeholder critical')
            out.append(None)
        else:
            if c>=len(hosts):raise RuntimeError('critical out of range')
            out.append(hosts[int(c)])
    return out

def donor_id(sha,hosts,critical_id,seed):
    if len(hosts)<=1:return critical_id
    c=hosts.index(critical_id);k=int(stable_rng(seed,sha,'host-donor').integers(0,len(hosts)-1))
    if k>=c:k+=1
    return hosts[k]

def build_base_contract(data,ids,orders):
    from bfa_core import formal_split
    contracts={};requested=collections.defaultdict(set);replay=[]
    for split in SPLITS:
        for seed in SEEDS:
            pack=formal_split(data,split,seed,'cuda');tp,tc,maxd=base_predict_and_critical(pack,'train');vp,vc,_=base_predict_and_critical(pack,'val')
            train_h=chosen_ids(pack['train'],tc,ids,orders);val_h=chosen_ids(pack['val'],vc,ids,orders)
            replay.append({'split':split,'seed':seed,'max_saved_val_probability_diff':float(np.max(np.abs(vp-np.asarray(pack['old']['probability']))))})
            for idx,h in zip(pack['train'],train_h):
                sha=ids[int(idx)]
                if h is not None:requested[sha].update([h,donor_id(sha,orders[sha],h,seed)])
            for idx,h in zip(pack['val'],val_h):
                sha=ids[int(idx)]
                if h is not None:requested[sha].update([h,donor_id(sha,orders[sha],h,seed)])
            contracts[(split,seed)]={'train':np.asarray(pack['train']),'val':np.asarray(pack['val']),
                'train_p':tp,'val_p':vp,'train_h':train_h,'val_h':val_h,'max_delta':maxd,
                'epochs':int(pack['old']['selected_epochs']),'lr':float(pack['args'].learning_rate),'wd':float(pack['args'].weight_decay)}
            print('RD_BASE_SOURCE',split,seed,flush=True)
    for seed in SEEDS:
        pack=formal_split(data,'T1',seed,'cuda');tp,tc,maxd=base_predict_and_critical(pack,'train')
        train_h=chosen_ids(pack['train'],tc,ids,orders)
        fp=future_pack(data,seed);vp,vc,_=base_predict_and_critical(fp,'val');val_h=chosen_ids(fp['val'],vc,ids,orders)
        for idx,h in list(zip(pack['train'],train_h))+list(zip(fp['val'],val_h)):
            sha=ids[int(idx)]
            if h is not None:requested[sha].update([h,donor_id(sha,orders[sha],h,seed)])
        contracts[('T1',seed)]={'train':np.asarray(pack['train']),'val':np.asarray(fp['val']),
            'train_p':tp,'val_p':vp,'train_h':train_h,'val_h':val_h,'max_delta':maxd,
            'epochs':int(pack['old']['selected_epochs']),'lr':float(pack['args'].learning_rate),'wd':float(pack['args'].weight_decay)}
        print('RD_BASE_FUTURE',seed,flush=True)
    write_json(OUT/'h0_replay_check.json',{'status':'PASS' if max(r['max_saved_val_probability_diff'] for r in replay)<=1e-6 else 'FAIL',
        'tolerance':1e-6,'source_checks':replay})
    write_pickle(OUT/'base_contract.pkl',contracts)
    return contracts,requested

def host_vector(pairs):
    uni=np.zeros((31,31),np.float32);bi=np.zeros((31,31),np.float32)
    for p in pairs:
        a,b=int(p['role_a']),int(p['role_b']);rel=p['relation']
        if rel=='A_TO_B':uni[a,b]+=1
        elif rel=='B_TO_A':uni[b,a]+=1
        elif rel=='BIDIRECTIONAL':
            bi[a,b]+=1
            if a!=b:bi[b,a]+=1
    total=float(uni.sum()+bi.sum());unique=float(np.count_nonzero(uni)+np.count_nonzero(bi));btotal=float(bi.sum())
    return np.concatenate([np.log1p(uni).reshape(-1),np.log1p(bi).reshape(-1),
                           np.asarray([math.log1p(total),math.log1p(unique),math.log1p(btotal),float(total>0)],np.float32)])

def extract_requested(requested):
    store={};stats=[]
    for n,(sha,hosts) in enumerate(requested.items(),1):
        with gzip.open(CFG_CACHE/f'{sha}.json.gz','rt',encoding='utf-8') as f:d=json.load(f)
        pm=collections.defaultdict(list)
        for p in d.get('pairs',[]):
            hid=str(p['host_node_id'])
            if hid in hosts:pm[hid].append(p)
        missing=0
        for h in hosts:
            if h not in {str(x['host_node_id']) for x in d.get('hosts',[])}:missing+=1
            store[(sha,h)]=host_vector(pm[h])
        stats.append({'sha256':sha,'requested_hosts':len(hosts),'missing_cfg_hosts':missing})
        if n%200==0:print('RD_CFG_CACHE',n,len(requested),flush=True)
    write_pickle(OUT/'critical_host_features.pkl',store);pd.DataFrame(stats).to_csv(OUT/'feature_coverage.csv',index=False)
    return store

def transform(vec,arm,sha,hid,seed,orders,store):
    if arm=='R1_ROLE_DIRECTED_CRITICAL_HOST':return vec
    if arm=='R3_HOST_BINDING_SHUFFLE':
        if hid is None:return vec
        return store[(sha,donor_id(sha,orders[sha],hid,seed))]
    z=vec.copy();body=z[:1922].reshape(2,31,31)
    if arm=='R2_PAIR_QUANTITY_CONTROL':
        rng=stable_rng(seed,sha,'quantity-control');out=np.zeros_like(body)
        for ch in range(2):
            vals=body[ch][body[ch]!=0];slots=rng.choice(961,size=len(vals),replace=False) if len(vals) else []
            out[ch].reshape(-1)[slots]=vals
        z[:1922]=out.reshape(-1);return z
    if arm=='R4_DIRECTION_SHUFFLE':
        rng=stable_rng(seed,sha,'direction-shuffle');out=body.copy()
        for a in range(31):
            for b in range(a+1,31):
                if rng.random()<.5:out[0,a,b],out[0,b,a]=body[0,b,a],body[0,a,b]
        z[:1922]=out.reshape(-1);return z
    raise KeyError(arm)

def matrix(indices,hids,arm,seed,ids,orders,store):
    rows=[]
    for idx,h in zip(indices,hids):
        sha=ids[int(idx)];v=np.zeros(DIM,np.float32) if h is None else store[(sha,h)]
        rows.append(transform(v,arm,sha,h,seed,orders,store))
    return np.asarray(rows,np.float32)

def train_head(contract,arm,seed,data,ids,orders,store):
    import torch
    torch.manual_seed(seed+991);dev=torch.device('cuda')
    x=matrix(contract['train'],contract['train_h'],arm,seed,ids,orders,store);xv=matrix(contract['val'],contract['val_h'],arm,seed,ids,orders,store)
    scale=np.maximum(np.sqrt(np.mean(np.square(x,dtype=np.float64),axis=0)),1.).astype(np.float32)
    xt=torch.as_tensor(x/scale,dtype=torch.float32,device=dev);yt=torch.as_tensor(data['y'][contract['train']],dtype=torch.float32,device=dev)
    base=np.clip(contract['train_p'],1e-6,1-1e-6);bt=torch.as_tensor(np.log(base/(1-base)),dtype=torch.float32,device=dev)
    head=torch.nn.Linear(DIM,1).to(dev);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    opt=torch.optim.Adam(head.parameters(),lr=contract['lr'],weight_decay=contract['wd'])
    started=time.perf_counter()
    for _ in range(contract['epochs']):
        opt.zero_grad(set_to_none=True);delta=contract['max_delta']*torch.tanh(head(xt).reshape(-1));loss=torch.nn.functional.binary_cross_entropy_with_logits(bt+delta,yt);loss.backward();opt.step()
    train_s=time.perf_counter()-started
    with torch.no_grad():
        vx=torch.as_tensor(xv/scale,dtype=torch.float32,device=dev);vp=np.clip(contract['val_p'],1e-6,1-1e-6)
        vb=torch.as_tensor(np.log(vp/(1-vp)),dtype=torch.float32,device=dev);pred=torch.sigmoid(vb+contract['max_delta']*torch.tanh(head(vx).reshape(-1))).cpu().numpy()
    state={k:v.detach().cpu() for k,v in head.state_dict().items()}
    return pred,state,scale,{'training_seconds':train_s,'epochs':contract['epochs'],'trainable_parameters':DIM+1,'final_BCE':float(loss.detach().cpu())}

def fit_all(data,ids,orders,contracts,store):
    predictions={a:{seed:{} for seed in SEEDS} for a in ARMS}
    for split in SPLITS+['T1']:
        for seed in SEEDS:
            c=contracts[(split,seed)]
            for sha,p in zip([ids[int(i)] for i in c['val']],c['val_p']):predictions['R0_FROZEN_DIRECT_C43'][seed][sha]=float(p)
            for arm in ARMS[1:]:
                pred,state,scale,cost=train_head(c,arm,seed,data,ids,orders,store)
                for sha,p in zip([ids[int(i)] for i in c['val']],pred):predictions[arm][seed][sha]=float(p)
                write_pickle(OUT/f'models/{split}/{seed}/{arm}.pkl',{'arm':arm,'split':split,'seed':seed,
                    'ids':[ids[int(i)] for i in c['val']],'probability':pred,'state_dict':state,'scale':scale,'cost':cost,
                    'training_years':[2017],'base_frozen':True,'backgrounds_changed':False,'prototype_bank_changed':False})
                print('RD_MODEL_DONE',split,seed,arm,flush=True)
    return predictions

def evaluate(data,ids,predictions):
    years=np.asarray(data['years'],int);labels=np.asarray(data['y'],int)
    stage=pd.read_csv(H1_OUT/'per_apk_predictions.csv.gz');route={}
    h0rows=stage[stage.arm=='H0_FROZEN_DIRECT_C43']
    for ms,g in h0rows.groupby('manifest_seed'):
        route[int(ms)]={str(r.sha256):(float(r.stage1_probability),bool(r.routed)) for r in g.itertuples()}
    perf=[];apk=[]
    for year in range(2017,2022):
        selected=np.flatnonzero(years==year);ss=[ids[i] for i in selected];y=labels[selected]
        for seed in SEEDS:
            for arm in ARMS:
                p=np.asarray([predictions[arm][seed][s] for s in ss])
                perf.append({'level':'stage2_all_apk','year':year,'arm':arm,'model_seed':seed,'manifest_seed':None,**metrics(y,p)})
                for ms in MANIFEST_SEEDS:
                    final=np.asarray([p[j] if route[ms][s][1] else route[ms][s][0] for j,s in enumerate(ss)])
                    perf.append({'level':'manifest_routed','year':year,'arm':arm,'model_seed':seed,'manifest_seed':ms,**metrics(y,final)})
                    for s,yy,ep,fp in zip(ss,y,p,final):apk.append({'year':year,'sha256':s,'label':int(yy),'arm':arm,'model_seed':seed,'manifest_seed':ms,
                        'routed':route[ms][s][1],'stage1_probability':route[ms][s][0],'expert_probability':ep,'final_probability':fp})
    frame=pd.DataFrame(perf);frame.to_csv(OUT/'performance_per_seed.csv',index=False)
    frame.groupby(['level','year','arm'],as_index=False).agg({c:'mean' for c in ['BA','accuracy','F1','AUC','FNR','FPR']}).to_csv(OUT/'performance_seed_mean.csv',index=False)
    pd.DataFrame(apk).to_csv(OUT/'per_apk_predictions.csv.gz',index=False,compression='gzip')
    source=frame[(frame.year==2017)&(frame.level=='stage2_all_apk')]
    source_idx=np.flatnonzero(years==2017);source_ids=[ids[i] for i in source_idx];source_y=labels[source_idx]
    fold_map={}
    for split in SPLITS:
        ck=load_pickle(SCBE/f'models/{split}/900000/M0_REPRODUCED.pkl')
        for sha in ck['ids']:fold_map[str(sha)]=split
    folds=np.asarray([fold_map[s] for s in source_ids])
    comparisons=[]
    for ref in ARMS:
        if ref=='R1_ROLE_DIRECTED_CRITICAL_HOST':continue
        for seed in SEEDS:
            pa=np.asarray([predictions['R1_ROLE_DIRECTED_CRITICAL_HOST'][seed][s] for s in source_ids]);pb=np.asarray([predictions[ref][seed][s] for s in source_ids])
            a=float(source[(source.arm=='R1_ROLE_DIRECTED_CRITICAL_HOST')&(source.model_seed==seed)].BA.iloc[0]);b=float(source[(source.arm==ref)&(source.model_seed==seed)].BA.iloc[0])
            rng=np.random.default_rng(20261020+seed+ARMS.index(ref));strata=[np.where((folds==f)&(source_y==c))[0] for f in SPLITS for c in (0,1)]
            diffs=[]
            for _ in range(5000):
                z=np.concatenate([rng.choice(q,len(q),replace=True) for q in strata if len(q)])
                diffs.append(balanced_accuracy_score(source_y[z],pa[z]>=.5)-balanced_accuracy_score(source_y[z],pb[z]>=.5))
            comparisons.append({'candidate':'R1_ROLE_DIRECTED_CRITICAL_HOST','reference':ref,'model_seed':seed,'delta_BA':a-b,
                'ci95_low':float(np.quantile(diffs,.025)),'ci95_high':float(np.quantile(diffs,.975)),'bootstrap':5000,'stratification':'fixed fold x class'})
    pd.DataFrame(comparisons).to_csv(OUT/'source_comparisons.csv',index=False)
    return frame,pd.DataFrame(comparisons)

def finalize(before,frame,comp):
    after=[digest(Path(x['path'])) for x in before];changed=[a['path'] for a,b in zip(before,after) if a!=b]
    write_json(OUT/'integrity_check.json',{'status':'PASS' if not changed else 'FAIL','checked_assets':len(before),'changed_assets':changed,'before':before,'after':after})
    s=frame[(frame.year==2017)&(frame.level=='stage2_all_apk')].groupby('arm').BA.mean()
    r=frame[(frame.year==2017)&(frame.level=='manifest_routed')].groupby('arm').BA.mean()
    deltas=comp.groupby('reference').delta_BA.mean().to_dict()
    controls=['R2_PAIR_QUANTITY_CONTROL','R3_HOST_BINDING_SHUFFLE','R4_DIRECTION_SHUFFLE']
    supported=all(bool((comp[comp.reference==x].ci95_low>0).all()) for x in controls)
    decision='ROLE_DIRECTED_SIGNAL_CANDIDATE' if supported else 'ROLE_DIRECTED_SIGNAL_NOT_SUPPORTED'
    write_json(OUT/'final_decision.json',{'status':'COMPLETE','decision':decision,'training_years':[2017],
        'future_years_development_only':[2018,2019,2020,2021],'future_used_for_selection':False,'source_mean_delta_BA':deltas})
    lines=['# CARE-Droid H1-RD full experiment','',f'Final source-only decision: **{decision}**.','',
      'Training used 2017 only. Manifest135 RF and OTHER3_OP25_REF were frozen. Future years are reported only after freezing.','',
      '| Arm | 2017 stage2 BA | 2017 routed BA |','|---|---:|---:|']
    for arm in ARMS:lines.append(f'| {arm} | {s[arm]:.4f} | {r[arm]:.4f} |')
    lines+=['','Source mean delta BA for R1:']+[f'- R1 - {k}: {v:+.4f}' for k,v in deltas.items()]
    lines+=['','Integrity: **PASS**.' if not changed else 'Integrity: **FAIL**.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    started=time.time();before=preflight()
    from bfa_core import load_data_contract
    data,_=load_data_contract();ids=list(map(str,data['ids']))
    if int(np.sum(np.asarray(data['years'])==2017))!=799:raise RuntimeError('source contract')
    orders=endpoint_orders(ids);contracts,requested=build_base_contract(data,ids,orders)
    store=extract_requested(requested);pred=fit_all(data,ids,orders,contracts,store)
    frame,comp=evaluate(data,ids,pred);finalize(before,frame,comp)
    print('CARE_ROLE_DIRECTED_CRITICAL_HOST_COMPLETE',time.time()-started,flush=True)

if __name__=='__main__':main()
