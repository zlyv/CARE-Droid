#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

class CaptureState:
    def __init__(self, v4, ids, folds, output):
        self.v4=v4; self.ids=np.asarray(ids); self.folds=np.asarray(folds,int); self.output=output
        self.next_fold=1; self.records={}

    def predict(self, model, global_values, base_probability, graph_values, bags, device):
        v4=self.v4; torch=v4.torch
        gt=torch.as_tensor(global_values,dtype=torch.float32,device=device)
        bl=torch.as_tensor(v4.probability_to_logit(base_probability),dtype=torch.float32,device=device)
        graph=v4.graph_tensors(graph_values,device)
        bt=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in bags]
        probability=[]; diagnostic=[]; fused=[]; hidden=[]; model.eval()
        with torch.no_grad():
            for i in range(len(bags)):
                one={k:x[i] for k,x in graph.items()}; logit,diag,trace=model.forward_with_trace(gt[i],bl[i],one,bt[i])
                z=trace["correction_input"].reshape(-1); h=model.delta_head[:-1](z).reshape(-1)
                probability.append(float(torch.sigmoid(logit).cpu())); diagnostic.append(diag.detach().cpu().numpy())
                fused.append(z.detach().cpu().numpy()); hidden.append(h.detach().cpu().numpy())
        return np.asarray(probability),np.asarray(diagnostic),np.asarray(fused),np.asarray(hidden)

    def fit_predict(self, global_train, global_test, s7_train, s7_test, raw_train_bags, raw_test_bags,
                    base_train, base_test, labels, variant, seeds, args, device):
        v4=self.v4; torch=v4.torch; fold=self.next_fold; self.next_fold+=1
        test_index=np.flatnonzero(self.folds==fold); train_index=np.flatnonzero(self.folds!=fold)
        if len(test_index)!=len(global_test) or len(train_index)!=len(global_train):
            raise RuntimeError("BLOCKED_BY_CONTRACT: replay fold order mismatch")
        probs=[]; diags=[]; audits=[]; train_fused=[]; test_fused=[]; train_hidden=[]; test_hidden=[]
        for seed in seeds:
            selected_epochs,selection_loss=v4.select_epochs(global_train,s7_train,raw_train_bags,base_train,labels,variant,args,device,int(seed))
            train_bags,test_bags,prototype_audit=v4.v3.fit_scaled_dualproto_bags(raw_train_bags,raw_test_bags,labels,args,int(seed)+1707)
            scaler=v4.fit_role_graph_scaler(s7_train); train_graph=v4.transform_role_graphs(s7_train,scaler); test_graph=v4.transform_role_graphs(s7_test,scaler)
            global_tensor=torch.as_tensor(global_train,dtype=torch.float32,device=device); train_graph_t=v4.graph_tensors(train_graph,device)
            train_bag_t=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in train_bags]
            weights,correct=v4.fixkeep_weights(labels,base_train,args); v4.seed_everything(seed)
            model=v4.RoleGraphConditionedEvidence(global_train.shape[1],variant,args).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
            final_loss=v4.train_epochs(model,optimizer,global_tensor,base_train,train_graph_t,train_bag_t,labels,weights,correct,selected_epochs,args,int(seed)+3301)
            p,d,z,h=self.predict(model,global_test,base_test,test_graph,test_bags,device)
            _,_,zt,ht=self.predict(model,global_train,base_train,train_graph,train_bags,device)
            probs.append(p); diags.append(d); test_fused.append(z); test_hidden.append(h); train_fused.append(zt); train_hidden.append(ht)
            audits.append({"seed":int(seed),"variant":variant,"selected_epochs":int(selected_epochs),"selection_loss":float(selection_loss),
                "final_training_loss":float(final_loss),"parameter_n":int(sum(x.numel() for x in model.parameters())),**prototype_audit})
            del model,train_graph_t,train_bag_t
            if device.type=="cuda":torch.cuda.empty_cache()
        self.records[fold]={"train_index":train_index,"test_index":test_index,
            "train_base":np.asarray(base_train,float),"test_base":np.asarray(base_test,float),
            "train_y":np.asarray(labels,int),"train_host_count":np.asarray([len(x) for x in raw_train_bags],float),
            "test_host_count":np.asarray([len(x) for x in raw_test_bags],float),
            "train_s7":np.asarray(s7_train,float),"test_s7":np.asarray(s7_test,float),
            "train_fused":np.concatenate(train_fused,axis=1),"test_fused":np.concatenate(test_fused,axis=1),
            "train_hidden":np.concatenate(train_hidden,axis=1),"test_hidden":np.concatenate(test_hidden,axis=1)}
        return np.mean(np.asarray(probs),axis=0),np.mean(np.asarray(diags),axis=0),audits

    def save(self, path):
        payload={"ids":self.ids,"folds":self.folds}
        for fold,record in self.records.items():
            for key,value in record.items():payload[f"f{fold}_{key}"]=value
        np.savez_compressed(path,**payload)
