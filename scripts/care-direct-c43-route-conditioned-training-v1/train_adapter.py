from __future__ import annotations
import hashlib,time
import numpy as np
from pilot_common import *

def fingerprint(values):
    h=hashlib.sha256()
    for x in values:
        a=np.ascontiguousarray(x); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def validate_epoch_records(records,seeds,fold):
    if len(records)!=len(seeds) or [int(a['seed']) for a in records]!=[int(s) for s in seeds]: raise RuntimeError('B0 epoch seed contract')
    for a in records:
        if not 1<=int(a['selected_epochs'])<=30: raise RuntimeError('B0 epoch bound')
    return {int(a['seed']):int(a['selected_epochs']) for a in records}

def fit_fixed_epochs(v4,inputs,weights,epoch_records,seeds,args,device):
    gt,gv,s7t,s7v,bt,bv,baset,basev,y,variant=inputs; torch=v4.torch
    probs=[]; diags=[]; audits=[]
    epochs={int(a['seed']):int(a['selected_epochs']) for a in epoch_records}
    for seed in seeds:
        seed=int(seed); n_epochs=epochs[seed]
        train_bags,test_bags,background=v4.v3.fit_scaled_dualproto_bags(bt,bv,y,args,seed+1707)
        scaler=v4.fit_role_graph_scaler(s7t); train_graph=v4.transform_role_graphs(s7t,scaler); test_graph=v4.transform_role_graphs(s7v,scaler)
        global_tensor=torch.as_tensor(gt,dtype=torch.float32,device=device); graph_t=v4.graph_tensors(train_graph,device)
        bag_t=[torch.as_tensor(b,dtype=torch.float32,device=device) for b in train_bags]
        correct=(np.asarray(baset)>=.5)==y
        v4.seed_everything(seed)
        model=v4.RoleGraphConditionedEvidence(gt.shape[1],variant,args).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
        loss=v4.train_epochs(model,optimizer,global_tensor,baset,graph_t,bag_t,y,np.asarray(weights,np.float32),correct,n_epochs,args,seed+3301)
        p,d=v4.predict_model(model,gv,basev,test_graph,test_bags,device)
        probs.append(p); diags.append(d)
        audits.append(dict(seed=seed,variant=variant,selected_epochs=n_epochs,epoch_policy='fixed B0 inner-selected epochs',
            selection_loss=None,final_training_loss=float(loss),parameter_n=int(sum(x.numel() for x in model.parameters())),
            optimizer_steps=n_epochs*int(np.ceil(len(y)/args.batch_size)),**background))
        del model,graph_t,bag_t
        if device.type=='cuda': torch.cuda.empty_cache()
    return np.mean(probs,axis=0),np.mean(diags,axis=0),audits,np.asarray(probs)

class TrainingState:
    def __init__(self,v4,arm,index):
        self.v4=v4; self.arm=arm; self.d=index; self.original=v4.fit_predict_variant; self.fold=0

    def fit_predict(self,global_train,global_test,s7_train,s7_test,raw_train_bags,raw_test_bags,base_train,base_test,
                    labels,variant,seeds,args,device):
        self.fold+=1; f=self.fold; start=time.monotonic(); v4=self.v4
        tr=np.flatnonzero(self.d['folds']!=f); te=np.flatnonzero(self.d['folds']==f)
        if not np.array_equal(labels,self.d['y'][tr]) or not np.array_equal(base_test,self.d['base'][te]) or len(global_train)!=len(tr) or len(global_test)!=len(te):
            raise RuntimeError('outer fold input order differs from frozen manifest')
        with np.load(OUTPUT/'routes'/f'f{f}.npz',allow_pickle=False) as a: route={k:a[k] for k in a.files}
        if not np.array_equal(route['ids'],self.d['ids'][tr]) or not np.array_equal(route['train_index'],tr): raise RuntimeError('outer-train route membership')
        if not np.allclose(base_train,route['member'][:,3],atol=1e-14,rtol=0): raise RuntimeError('nested SVM differs from new committee SVM')
        fp=fingerprint([global_train,global_test,s7_train,s7_test,base_train,base_test,labels,*raw_train_bags,*raw_test_bags])
        inputs=(global_train,global_test,s7_train,s7_test,raw_train_bags,raw_test_bags,base_train,base_test,labels,variant)
        if self.arm=='B0_ORIGINAL':
            p,d,audits=self.original(*inputs,seeds,args,device)
            validate_epoch_records(audits,seeds,f)
            audits=[dict(a,optimizer_steps=int(a['selected_epochs'])*int(np.ceil(len(labels)/args.batch_size))) for a in audits]
            dump(OUTPUT/'B0_epochs'/f'f{f}.json',dict(fold=f,inputs_sha256=fp,records=audits,seeds=list(map(int,seeds)),
                feature_order_and_dtype_preserved=True,seconds=time.monotonic()-start))
            per_seed=None
        else:
            original=read_json(OUTPUT/'B0_epochs'/f'f{f}.json'); validate_epoch_records(original['records'],seeds,f)
            if original['inputs_sha256']!=fp: raise RuntimeError('input features/base/Host order changed relative to B0')
            p,d,audits,per_seed=fit_fixed_epochs(v4,inputs,route[self.arm],original['records'],seeds,args,device)
            for a,b in zip(audits,original['records']):
                if a['parameter_n']!=b['parameter_n'] or a['optimizer_steps']!=b['optimizer_steps']: raise RuntimeError('network capacity or final optimization exposure differs')
        values=dict(ids=self.d['ids'][te],expert=p)
        if per_seed is not None: values['seed_expert']=per_seed
        save_npz(OUTPUT/'predictions'/self.arm/f'f{f}.npz',**values)
        dump(OUTPUT/'training'/self.arm/f'f{f}.json',dict(fold=f,arm=self.arm,inputs_sha256=fp,
            records=audits,all_training_rows_retained=len(tr),seconds=time.monotonic()-start,
            sample_weight_min=float(route[self.arm].min()),sample_weight_max=float(route[self.arm].max()),
            effective_sample_size=float(route[self.arm].sum()**2/np.square(route[self.arm]).sum()),future_rows_loaded=0))
        print(f'RCT_FOLD_DONE arm={self.arm} fold={f} seconds={time.monotonic()-start:.1f}',flush=True)
        return p,d,audits
