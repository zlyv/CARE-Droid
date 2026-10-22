"""Reuse original Direct-C43 network, background fitting, early stopping and BCE; save complete inference states."""
from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np
from myd_common import *
from protocol import year_balanced_order,split_window

SEEDS=(900000,900001,900002)

def raw_bags(data,index):
    o=data['host_offsets'];v=data['host_values'];return [v[o[i]:o[i+1]] for i in index]

def capture_background(v4,tr,va,y,args,seed):
    original=v4.v3.fit_scaled_dualproto_bags;captured={};previous=sys.getprofile()
    def observe(frame,event,value):
        if event=='return' and frame.f_code is original.__code__:
            for name in ('mean','std','state','sp','bp'):
                if name not in frame.f_locals:raise RuntimeError('original background internals changed: missing '+name)
                captured[name]=frame.f_locals[name]
    sys.setprofile(observe)
    try:a,b,audit=original(tr,va,y,args,seed)
    finally:sys.setprofile(previous)
    return a,b,audit,captured

def transform_saved(v4,bags,prep,args):
    import haig_background as hb
    core,role,relation=hb.split(bags);std=[(x-prep['mean'])/prep['std'] for x in core]
    return [np.concatenate((x,hb.cols(x,w,prep['state'],prep['sp'],prep['bp'],args,v4.v3.v2),r),axis=1).astype(np.float32) for x,w,r in zip(std,role,relation)]

def train_balanced(v4,model,optimizer,g,base,graph,bags,y,weights,correct,epochs,args,seed,years):
    torch=v4.torch;device=g.device
    bl=torch.as_tensor(v4.probability_to_logit(base),dtype=torch.float32,device=device)
    b=torch.as_tensor(base,dtype=torch.float32,device=device);yt=torch.as_tensor(y,dtype=torch.float32,device=device)
    wt=torch.as_tensor(weights,dtype=torch.float32,device=device);ct=torch.as_tensor(correct,dtype=torch.bool,device=device)
    rng=np.random.RandomState(seed+811);draws=[];final=float('nan')
    for epoch in range(int(epochs)):
        model.train();order=year_balanced_order(years,rng);total=0.
        draws.append(dict(epoch=epoch+1,total=len(order),unique=int(len(set(order))),year_counts={str(yr):int(np.sum(years[order]==yr)) for yr in np.unique(years)}))
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size];optimizer.zero_grad(set_to_none=True)
            logits=v4.model_forward_batch(model,g,bl,graph,bags,ix);it=torch.as_tensor(ix,dtype=torch.long,device=device)
            loss=v4.batch_loss(logits,yt[it],wt[it],b[it],ct[it],args);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);optimizer.step();total+=float(loss.detach().cpu())*len(ix)
        final=total/len(y)
    return final,draws

def fit(data,tr,va,tag,device='cuda',smoke=False,epoch_reference=None,seeds=SEEDS):
    v4,_,args=runtime();args.device=device;args.max_epochs=1 if smoke else args.max_epochs;dev=v4.resolve_device(device)
    if dev.type=='cuda' and v4.torch.cuda.mem_get_info(dev.index if dev.index is not None else v4.torch.cuda.current_device())[0]<1024**3:raise RuntimeError('GPU free memory <1GiB; choose another GPU or explicit --device cpu, do not change architecture')
    if not np.all(data['years'][tr]<min(data['years'][va])) or set(data['ids'][tr])&set(data['ids'][va]):raise RuntimeError('Stage-II temporal/SHA leakage')
    torch=v4.torch;fr=data['feature_rows'];rows=fr['rows']
    # Rebuild original global preprocessing WITHIN this window. Never standardize
    # all 2788 samples or let validation rows select the canonical feature layout.
    gx,gz,names=v4.audit.compact_matrix_variant(v4.VARIANT,[rows[i] for i in tr],[rows[i] for i in va],fr['compact_by_sha'],fr['schema'],fr['canonical_names'])
    gi,_=v4.evidence.split_view_indices(names);global_train=np.asarray(gx[:,gi],np.float32);global_test=np.asarray(gz[:,gi],np.float32)
    if global_train.shape[1]!=24 or not np.isfinite(global_train).all() or not np.isfinite(global_test).all():raise RuntimeError('invalid per-window global24')
    s7_train=data['s7'][tr];s7_test=data['s7'][va];y=data['y'][tr]
    rt=raw_bags(data,tr);rv=raw_bags(data,va)
    # Direct variant zeros its Stage-I input and disables FixKeep/consistency.
    # A constant neutral base therefore avoids unnecessary nested Stage-I fitting;
    # smoke separately verifies invariance to different base probabilities.
    base_train=np.full(len(tr),.5);base_test=np.full(len(va),.5);pred=[];audits=[]
    for seed in seeds:
        meta=dict(dataset=data['signature'],code=implementation_hash(),tag=tag,train_sha=data['ids'][tr].tolist(),validation_sha=data['ids'][va].tolist(),seed=seed,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},C2_epoch_reference=epoch_reference,semantic='original Direct-C43; neutral base verified invariant; equal BCE')
        path=OUTPUT/'models/stage2'/tag/(str(seed)+'.pkl');existing=load_checkpoint(path,meta)
        if existing:pred.append(existing['probability']);audits.append(existing['audit']);continue
        t=time.perf_counter()
        epochs,select_loss=(epoch_reference[str(seed)],None) if epoch_reference is not None else v4.select_epochs(global_train,s7_train,rt,base_train,y,'RGCondAttn',args,dev,seed)
        a,b,pa,prep=capture_background(v4,rt,rv,y,args,seed+1707)
        scaler=v4.fit_role_graph_scaler(s7_train);tg=v4.transform_role_graphs(s7_train,scaler);vg=v4.transform_role_graphs(s7_test,scaler)
        gt=torch.as_tensor(global_train,dtype=torch.float32,device=dev);tgt=v4.graph_tensors(tg,dev);bt=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in a]
        weights,correct=v4.fixkeep_weights(y,base_train,args)
        if not np.all(weights==1) or args.keep_consistency!=0:raise RuntimeError('BCE definition changed')
        v4.seed_everything(seed);model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        if epoch_reference is None:loss=v4.train_epochs(model,optimizer,gt,base_train,tgt,bt,y,weights,correct,epochs,args,seed+3301);draws=[]
        else:loss,draws=train_balanced(v4,model,optimizer,gt,base_train,tgt,bt,y,weights,correct,epochs,args,seed+3301,data['years'][tr])
        trained=time.perf_counter()-t;ti=time.perf_counter();p,_=v4.predict_model(model,global_test,base_test,vg,b,dev);infer=time.perf_counter()-ti
        if not np.isfinite(p).all():raise RuntimeError('non-finite Direct prediction')
        splitter=v4.StratifiedShuffleSplit(n_splits=1,test_size=args.early_stop_fraction,random_state=seed+args.split_seed);early_fit,early_hold=next(splitter.split(np.zeros(len(y)),y))
        audit=dict(seed=seed,selected_epochs=int(epochs),selection_loss=select_loss,final_loss=loss,fit_seconds=trained,inference_seconds=infer,parameter_n=sum(x.numel() for x in model.parameters()),optimization_steps=int(epochs)*int(np.ceil(len(tr)/args.batch_size)),sampling=draws,prototype_audit=pa,train_n=len(tr),validation_n=len(va),all_train_equal_BCE=True,inner_early_fit_sha=data['ids'][tr[early_fit]].tolist(),inner_early_hold_sha=data['ids'][tr[early_hold]].tolist(),epoch_selection='C1 train-only early holdout' if epoch_reference is None else 'reuse same C1 selected epochs; do not use validation year')
        obj=dict(signature=signature(meta),metadata=meta,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},background=prep,graph_scaler=scaler,global_feature_names=[names[i] for i in gi],global_preprocessing='original compact_matrix_variant; rebuild with metadata train_sha and frozen canonical names, original cache hashes checked',global_validation=global_test,probability=p,audit=audit)
        save_pickle(path,obj);dump(path.with_suffix('.json'),dict(metadata=meta,audit=audit))
        if smoke:
            with path.open('rb') as f:reload=__import__('pickle').load(f)
            loaded=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);loaded.load_state_dict(reload['state_dict'])
            re_b=transform_saved(v4,rv,reload['background'],args);re_g=v4.transform_role_graphs(s7_test,reload['graph_scaler'])
            re_p,_=v4.predict_model(loaded,global_test,np.linspace(.01,.99,len(va)),re_g,re_b,dev)
            diff=float(np.max(np.abs(p-re_p)))
            if diff>1e-7:raise RuntimeError('saved preprocessing/model or Direct-base invariance failed '+str(diff))
            dump(path.parent/(str(seed)+'_reload.json'),dict(status='PASS',max_probability_difference=diff,base_input_invariance=True))
            del loaded
        pred.append(p);audits.append(audit);del model,gt,tgt,bt,optimizer
        if dev.type=='cuda':torch.cuda.empty_cache()
        print('MYD_STAGE2_SEED_DONE',tag,seed,epochs,flush=True)
    p=np.mean(pred,axis=0);save_npz(OUTPUT/'predictions'/tag/'expert.npz',ids=data['ids'][va],val_index=va,probability=p,seed_probability=np.asarray(pred));dump(OUTPUT/'predictions'/tag/'fit_audit.json',audits)
    return p,{str(a['seed']):a['selected_epochs'] for a in audits}

def run(data,window,device='cuda'):
    records=[dict(r,year=int(r['year']),label=int(r['label'])) for r in read_csv(OUTPUT/'sample_manifest.csv')];tr,va=split_window(records,window)
    src=np.flatnonzero(data['years']==2017);future=np.flatnonzero(data['years']>2017)
    fit(data,src,future,'C0R_2017',device)
    if window=='T1':
        with np.load(OUTPUT/'predictions/C0R_2017/expert.npz') as a:index={sid:i for i,sid in enumerate(a['ids'])};p=a['probability'][[index[s] for s in data['ids'][va]]];seeds=a['seed_probability'][:,[index[s] for s in data['ids'][va]]]
        for c in ['C1','C2']:
            save_npz(OUTPUT/'predictions'/(window+'_'+c)/'expert.npz',ids=data['ids'][va],val_index=va,probability=p,seed_probability=seeds);dump(OUTPUT/'predictions'/(window+'_'+c)/'fit_audit.json',dict(reused='C0R_2017',reason='one training year: C1 and year-balanced C2 are identical'))
    else:
        _,epochs=fit(data,tr,va,window+'_C1',device);fit(data,tr,va,window+'_C2',device,epoch_reference=epochs)
