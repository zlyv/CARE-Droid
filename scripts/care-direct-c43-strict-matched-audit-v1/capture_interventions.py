#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from interventions import ARMS, intervene_bag, strict_matched_donor


class InterventionState:
    def __init__(self, v4, ids, folds, p31, output, repeats=12):
        self.v4=v4; self.ids=np.asarray(ids); self.folds=np.asarray(folds,int); self.output=output
        self.p31=np.asarray(p31,float)
        if self.p31.shape != (len(self.ids),31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 alignment failed")
        self.repeats=int(repeats); self.next_fold=1
        self.sum={arm:np.zeros(len(ids),float) for arm in ARMS}
        self.sumsq={arm:np.zeros(len(ids),float) for arm in ARMS}
        self.count={arm:np.zeros(len(ids),int) for arm in ARMS}
        self.donor_rows=[]; self.fit_rows=[]; self.strict_match=np.zeros(len(ids),bool)

    def _one(self, model, global_value, graph, bag, device):
        v4=self.v4; torch=v4.torch
        g=torch.as_tensor(global_value,dtype=torch.float32,device=device)
        z=torch.zeros((),dtype=torch.float32,device=device)
        one={k:torch.as_tensor(x,dtype=torch.float32,device=device) for k,x in graph.items()}
        b=torch.as_tensor(bag,dtype=torch.float32,device=device)
        with torch.no_grad():
            logit,_,_=model.forward_with_trace(g,z,one,b)
        return float(torch.sigmoid(logit).cpu())

    def _query_only(self, model, global_value, own_graph, donor_graph, bag, device):
        """Break RoleGraph-to-local selection while restoring own readout context."""
        v4=self.v4; torch=v4.torch
        g=torch.as_tensor(global_value,dtype=torch.float32,device=device)
        z=torch.zeros((),dtype=torch.float32,device=device)
        own={k:torch.as_tensor(x,dtype=torch.float32,device=device) for k,x in own_graph.items()}
        donor={k:torch.as_tensor(x,dtype=torch.float32,device=device) for k,x in donor_graph.items()}
        b=torch.as_tensor(bag,dtype=torch.float32,device=device)
        with torch.no_grad():
            own_cache,own_gg=model.select_local_evidence(own,b)
            donor_cache,_=model.select_local_evidence(donor,b)
            Cache=type(own_cache)
            hybrid=Cache(role_token=own_cache.role_token,critical=donor_cache.critical,
                attended=donor_cache.attended,scalar_summary=donor_cache.scalar_summary,
                attention=donor_cache.attention,score=donor_cache.score,
                graph_entropy=own_cache.graph_entropy)
            raw,_,_=model.delta_from_selected_evidence(g,z,hybrid,own_gg)
        return float(torch.sigmoid(raw).cpu())

    @staticmethod
    def _graph_at(values,index): return {k:np.asarray(v[index]) for k,v in values.items()}

    def _record(self, arm, global_index, value):
        self.sum[arm][global_index]+=value; self.sumsq[arm][global_index]+=value*value
        self.count[arm][global_index]+=1

    def fit_predict(self, global_train, global_test, s7_train, s7_test, raw_train_bags, raw_test_bags,
                    base_train, base_test, labels, variant, seeds, args, device):
        v4=self.v4; torch=v4.torch; fold=self.next_fold; self.next_fold+=1
        test_index=np.flatnonzero(self.folds==fold); train_index=np.flatnonzero(self.folds!=fold)
        if len(test_index)!=len(global_test) or len(train_index)!=len(global_train):
            raise RuntimeError("BLOCKED_BY_CONTRACT: fold order mismatch")
        full_seed=[]; diag_seed=[]; audits=[]
        for seed in seeds:
            selected_epochs,selection_loss=v4.select_epochs(global_train,s7_train,raw_train_bags,base_train,labels,variant,args,device,int(seed))
            train_bags,test_bags,proto=v4.v3.fit_scaled_dualproto_bags(raw_train_bags,raw_test_bags,labels,args,int(seed)+1707)
            if any(np.asarray(x).ndim!=2 or np.asarray(x).shape[1]!=142 for x in train_bags+test_bags):
                raise RuntimeError("BLOCKED_BY_CONTRACT: formal A0 augmented Host width is not 142")
            haig_max=max([float(np.max(np.abs(np.asarray(x)[:,43:107]))) if len(x) else 0.0 for x in train_bags+test_bags])
            if haig_max>1e-6:
                raise RuntimeError("BLOCKED_BY_CONTRACT: A0 HAIG compatibility slots are not zero")
            scaler=v4.fit_role_graph_scaler(s7_train); train_graph=v4.transform_role_graphs(s7_train,scaler); test_graph=v4.transform_role_graphs(s7_test,scaler)
            gt=torch.as_tensor(global_train,dtype=torch.float32,device=device); tg=v4.graph_tensors(train_graph,device)
            tb=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in train_bags]
            weights,correct=v4.fixkeep_weights(labels,base_train,args); v4.seed_everything(seed)
            model=v4.RoleGraphConditionedEvidence(global_train.shape[1],variant,args).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
            loss=v4.train_epochs(model,opt,gt,base_train,tg,tb,labels,weights,correct,selected_epochs,args,int(seed)+3301)
            model.eval(); pfull=[]; drows=[]
            for local,global_i in enumerate(test_index):
                own_graph=self._graph_at(test_graph,local); own_bag=test_bags[local]
                value=self._one(model,global_test[local],own_graph,own_bag,device)
                pfull.append(value); self._record("FULL",global_i,value)
            for repeat in range(self.repeats):
                donor,donor_diagnostics=strict_matched_donor(base_test,[len(x) for x in raw_test_bags],
                    self.p31[test_index],repeat)
                for local,global_i in enumerate(test_index):
                    sid=str(self.ids[global_i]); own_graph=self._graph_at(test_graph,local); own_bag=test_bags[local]
                    bag=intervene_bag(own_bag,sid,"N0_ROW_ORDER",repeat)
                    self._record("N0_ROW_ORDER",global_i,self._one(model,global_test[local],own_graph,bag,device))
                    if int(donor[local]) < 0:
                        for arm in ARMS[2:]: self._record(arm,global_i,pfull[local])
                        if seed==seeds[0]:
                            self.donor_rows.append({"fold":fold,"repeat":repeat,"sha256":sid,
                                "matched":0,"stage1_direction":int(base_test[local]>=.5)})
                        continue
                    self.strict_match[global_i]=True
                    donor_graph=self._graph_at(test_graph,int(donor[local]))
                    self._record("I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT",global_i,
                        self._query_only(model,global_test[local],own_graph,donor_graph,own_bag,device))
                    self._record("I6_ROLEGRAPH_WHOLE_ALIGNMENT",global_i,
                        self._one(model,global_test[local],donor_graph,own_bag,device))
                    self._record("I7_GLOBAL24_LOCAL_ALIGNMENT",global_i,
                        self._one(model,global_test[int(donor[local])],own_graph,own_bag,device))
                    if seed==seeds[0]:
                        diag=donor_diagnostics[local]
                        self.donor_rows.append({"fold":fold,"repeat":repeat,"sha256":sid,"matched":1,
                            "donor_sha256":str(self.ids[test_index[int(donor[local])]]),
                            "stage1_direction":int(base_test[local]>=.5),
                            "donor_stage1_direction":int(base_test[int(donor[local])]>=.5), **diag})
            full_seed.append(np.asarray(pfull)); diag_seed.append(np.zeros((len(pfull),len(v4.DIAGNOSTIC_NAMES)),float))
            audits.append({"seed":int(seed),"variant":variant,"selected_epochs":int(selected_epochs),
                "selection_loss":float(selection_loss),"final_training_loss":float(loss),
                "parameter_n":int(sum(x.numel() for x in model.parameters())),"a0_haig_zero_slot_max_abs":haig_max,**proto})
            del model,tg,tb
            if device.type=="cuda":torch.cuda.empty_cache()
        return np.mean(np.asarray(full_seed),axis=0),np.mean(np.asarray(diag_seed),axis=0),audits

    def save(self,path):
        payload={"ids":self.ids,"folds":self.folds,"repeats":np.asarray([self.repeats],int),
            "STRICT_MATCH__eligible":self.strict_match.astype(np.int8)}
        for arm in ARMS:
            if np.any(self.count[arm]==0): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete intervention "+arm)
            mean=self.sum[arm]/self.count[arm]
            var=np.maximum(self.sumsq[arm]/self.count[arm]-mean*mean,0.0)
            payload[arm+"__mean"]=mean; payload[arm+"__std"]=np.sqrt(var); payload[arm+"__n"]=self.count[arm]
        np.savez_compressed(path,**payload)

