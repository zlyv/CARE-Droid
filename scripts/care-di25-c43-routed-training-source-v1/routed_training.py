#!/usr/bin/env python3
"""Monkey-patched C43-RQ fitting with routed-only neural optimization."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch

LOCKED_FIT=Path("artifacts/care-di25-compact43-relationquery-source-v1/arms/C43_RELATION_QUERY/fit_audit.csv")

def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))

def locked_epochs():
    rows=read(LOCKED_FIT);result={}
    for row in rows:
        key=(int(float(row["fold"])),int(float(row["seed"])))
        result[key]=(int(float(row["selected_epochs"])),float(row["selection_loss"]))
    if len(result)!=15:raise RuntimeError("BLOCKED_BY_CONTRACT: locked selected-epoch grid is not 5x3")
    return result

def install(v4,arm):
    if arm not in ("T1_ROUTED_LOSS","T2_FULLY_ROUTED"):raise KeyError(arm)
    state={"call":0};epochs=locked_epochs();original_reference=v4.care_v1.reference_arrays

    def capture_reference(path,manifest,*args,**kwargs):
        result=original_reference(path,manifest,*args,**kwargs)
        state["route"]=np.asarray(result["route"],bool)
        state["fold"]=np.asarray([int(row["fold"]) for row in manifest],int)
        state["ids"]=np.asarray([str(row["sha256"]).strip().lower() for row in manifest])
        state["base"]=np.asarray(result["coarse"],float)
        state["seed_outputs"]={offset:np.full(len(manifest),np.nan) for offset in (0,1,2)}
        return result
    v4.care_v1.reference_arrays=capture_reference

    def fit_predict(global_train,global_test,s7_train,s7_test,raw_train_bags,raw_test_bags,
                    base_train,base_test,labels,variant,seeds,args,device):
        state["call"]+=1;fold=state["call"]
        if fold not in (1,2,3,4,5):raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected fit call order")
        train_global=np.flatnonzero(state["fold"]!=fold);route_mask=state["route"][train_global]
        if len(route_mask)!=len(labels) or int(route_mask.sum())!=160:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: fold {fold} routed outer-train is not 160")
        probabilities=[];diagnostics=[];audits=[]
        for seed in seeds:
            selected,selection_loss=epochs[(fold,int(seed))]
            if arm=="T1_ROUTED_LOSS":
                train_bags,test_bags,prototype_audit=v4.v3.fit_scaled_dualproto_bags(raw_train_bags,raw_test_bags,labels,args,int(seed)+1707)
                scaler=v4.fit_role_graph_scaler(s7_train);train_graph_all=v4.transform_role_graphs(s7_train,scaler)
                test_graph=v4.transform_role_graphs(s7_test,scaler);idx=np.flatnonzero(route_mask)
                train_global_values=np.asarray(global_train)[idx];train_base=np.asarray(base_train)[idx];train_labels=np.asarray(labels)[idx]
                train_bag_values=[train_bags[i] for i in idx];train_graph={k:v[idx] for k,v in train_graph_all.items()}
                background_scope="all_outer_train";scaler_scope="all_outer_train"
            else:
                idx=np.flatnonzero(route_mask);raw_routed=[raw_train_bags[i] for i in idx];train_labels=np.asarray(labels)[idx]
                train_bag_values,test_bags,prototype_audit=v4.v3.fit_scaled_dualproto_bags(raw_routed,raw_test_bags,train_labels,args,int(seed)+1707)
                scaler=v4.fit_role_graph_scaler(np.asarray(s7_train)[idx]);train_graph=v4.transform_role_graphs(np.asarray(s7_train)[idx],scaler)
                test_graph=v4.transform_role_graphs(s7_test,scaler);train_global_values=np.asarray(global_train)[idx];train_base=np.asarray(base_train)[idx]
                background_scope="routed_outer_train";scaler_scope="routed_outer_train"
            global_tensor=torch.as_tensor(train_global_values,dtype=torch.float32,device=device)
            graph_t=v4.graph_tensors(train_graph,device);bag_t=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in train_bag_values]
            weights,correct=v4.fixkeep_weights(train_labels,train_base,args);v4.seed_everything(seed)
            model=v4.RoleGraphConditionedEvidence(global_train.shape[1],variant,args).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
            final_loss=v4.train_epochs(model,optimizer,global_tensor,train_base,graph_t,bag_t,train_labels,weights,correct,selected,args,int(seed)+3301)
            probability,diagnostic=v4.predict_model(model,global_test,base_test,test_graph,test_bags,device)
            probabilities.append(probability);diagnostics.append(diagnostic)
            offset=int(seed)%100000;val_global=np.flatnonzero(state["fold"]==fold)
            if offset not in state["seed_outputs"] or len(val_global)!=len(probability):raise RuntimeError("BLOCKED_BY_CONTRACT: seed/fold prediction alignment")
            state["seed_outputs"][offset][val_global]=probability
            audits.append({"seed":int(seed),"variant":variant,"selected_epochs":selected,"selection_loss":selection_loss,
                           "final_training_loss":float(final_loss),"parameter_n":int(sum(p.numel() for p in model.parameters())),
                           "optimization_train_n":int(len(idx)),"outer_train_n":int(len(labels)),"routed_train_n":int(route_mask.sum()),
                           "neural_loss_scope":"routed_outer_train","background_scope":background_scope,"stage2_scaler_scope":scaler_scope,
                           "selected_epochs_reused_from_locked_T0":True,**prototype_audit})
            del model,graph_t,bag_t
            if device.type=="cuda":torch.cuda.empty_cache()
        return np.mean(np.asarray(probabilities),axis=0),np.mean(np.asarray(diagnostics),axis=0),audits
    v4.fit_predict_variant=fit_predict
    return state

