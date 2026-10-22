#!/usr/bin/env python3
"""Final frozen M1 replay with per-seed persistence and reloadable state."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/"care-di25-hostego-within-year-v1"
MECH=HERE.parent/"care-di25-hostego-stage2-mechanism-v1"
MATRIX=ORIGINAL/"within_matrix"
for p in (HERE,MECH,ORIGINAL,MATRIX,MATRIX/"vendor"):sys.path.insert(0,str(p))
import run_m1_within_year as frozen
import run_within_year_arm as within
from host_ego_encoder import make_fit_adapter, pack, unpack, repack_augmented
from instrumented_model import instrumented_class

v4=within.v4
ROOT=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/final_replay")
ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
PRIOR=Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/instrumented_rerun/apk_internal.csv")
SEEDS=(900000,900001,900002); TOL=1e-12

def normal(x):return str(x or "").strip().lower()
def read_csv(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    q=p.with_suffix(p.suffix+".tmp")
    with q.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    q.replace(p)
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def sha_ids(values):return hashlib.sha256(("\n".join(values)+"\n").encode()).hexdigest()
def sign(x):return 0 if abs(float(x))<=TOL else 1 if float(x)>0 else -1

def fit_anomaly_state(raw_train,raw_test,labels,args,seed):
    """Arithmetic-identical expansion of the frozen adapter, now retaining state."""
    host_train=[unpack(x)[0] for x in raw_train];host_test=[unpack(x)[0] for x in raw_test]
    module=sys.modules[within.future.ORIGINAL_ANOMALY_FIT.__module__]
    mean,std=module.care_v1.fit_equal_apk_scaler(host_train,np.arange(len(host_train),dtype=np.int64))
    scaled_train=[(np.asarray(x,dtype=np.float32)-mean)/std for x in host_train]
    scaled_test=[(np.asarray(x,dtype=np.float32)-mean)/std for x in host_test]
    state=module.v2.fit_prototype_state(scaled_train,labels,args,int(seed))
    aug_train,train_summary=module.v2.augment_bags(scaled_train,state,"JointDualPrototype",args)
    aug_test,test_summary=module.v2.augment_bags(scaled_test,state,"JointDualPrototype",args)
    audit={"region_scaler_zero_std_n":int(np.sum(std==1.0)),"shared_prototype_n":int(state["shared_k"]),
      "benign_prototype_n":int(state["benign_k"]),"sampled_balanced_region_n":int(state["sampled_balanced_n"]),
      "sampled_benign_region_n":int(state["sampled_benign_n"]),"shared_commonness_mean":float(np.mean(state["commonness"])),
      "shared_commonness_max":float(np.max(state["commonness"])),"external_distance_scale":float(state["external_scale"]),
      "train_external_mean":float(np.mean(train_summary[:,0])),"test_external_mean":float(np.mean(test_summary[:,0])),
      "train_internal_mean":float(np.mean(train_summary[:,1])),"test_internal_mean":float(np.mean(test_summary[:,1])),
      "train_commonness_mean":float(np.mean(train_summary[:,2])),"test_commonness_mean":float(np.mean(test_summary[:,2])),
      "ego_changes_scaler":False,"ego_changes_prototypes":False,"ego_edge_scope":"directed Host-graph one-hop"}
    return repack_augmented(aug_train,raw_train),repack_augmented(aug_test,raw_test),audit,{"region_mean":mean,"region_std":std,"prototype_state":state}

def role_names(args):
    p=args.future_ego.parent/"local_asset_manifest.json"
    if not p.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: local_asset_manifest.json missing")
    value=json.loads(p.read_text(encoding="utf-8"))["model_role_order"]
    if len(value)!=31:raise RuntimeError("BLOCKED_BY_CONTRACT: role order changed")
    return value

def infer(model,x,base,graphs,bags,metas,ids,labels,routed,names,device,seed,year,fold):
    gt=torch.as_tensor(x,dtype=torch.float32,device=device)
    bl=torch.as_tensor(v4.probability_to_logit(base),dtype=torch.float32,device=device)
    gr=v4.graph_tensors(graphs,device);bt=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in bags]
    apps=[];hosts=[];model.eval()
    with torch.no_grad():
      for i in range(len(ids)):
        logit,diag=model(gt[i],bl[i],{k:v[i] for k,v in gr.items()},bt[i]);c=model.last_internal_capture
        raw=float(c["raw_delta"]);bounded=float(c["bounded_delta"]);prob=float(torch.sigmoid(logit).cpu())
        apps.append({"year":year,"fold":fold,"sha256":ids[i],"label":int(labels[i]),"seed":seed,
          "raw_delta":raw,"bounded_delta":bounded,"probability":prob,"raw_delta_sign":sign(raw),
          "bounded_delta_sign":sign(bounded),"routed":int(routed[i])})
        if not routed[i]:continue
        n=metas[i].real_host_count
        for h in range(n):
          inc=metas[i].incoming[h];out=metas[i].outgoing[h]
          rn=lambda z:names[int(z)] if 0<=int(z)<len(names) else str(z)
          hosts.append({"year":year,"fold":fold,"sha256":ids[i],"label":int(labels[i]),"seed":seed,
            "host_index":h,"host_id":metas[i].host_ids[h],"host_role":rn(metas[i].primary_roles[h]),
            "raw_delta_if_host_level_available":"","raw_delta_is_apk_level":True,
            "gamma":float(c["gamma"][h]),"anomaly_score":float(c["anomaly_score"][h]),
            "attention_logit":float(c["attention_logit"][h]),"attention_weight":float(c["attention_weight"][h]),
            "host_embedding_norm":float(c["host_embedding_norm"][h]),"ego_embedding_norm":float(c["ego_embedding_norm"][h]),
            "local_embedding_norm":float(c["local_embedding_norm"][h]),"critical_host_flag":int(h==c["critical_index"]),
            "incoming_neighbor_count":len(inc),"outgoing_neighbor_count":len(out),"neighbor_count":len(set(inc+out)),
            "incoming_neighbor_indices":json.dumps(inc),"outgoing_neighbor_indices":json.dumps(out),
            "incoming_neighbor_roles":json.dumps([rn(metas[i].primary_roles[j]) for j in inc]),
            "outgoing_neighbor_roles":json.dumps([rn(metas[i].primary_roles[j]) for j in out]),
            "external_background_value":float(c["external_background_value"][h]),
            "internal_background_value":float(c["internal_background_value"][h]),
            "shared_commonness_value":float(c["shared_commonness_value"][h]),
            "structure_shared_across_seeds":True,"background_fitted_per_seed":True})
    return apps,hosts

def checkpoint_path(year,fold,seed):return ROOT/"checkpoints"/str(year)/f"fold_{fold}"/f"seed_{seed}.pt"
def output_paths(year,fold,seed):
    d=ROOT/"seed_outputs"/str(year)/f"fold_{fold}";return d/f"seed_{seed}_apk.csv",d/f"seed_{seed}_host.csv"

def reload_probability(path,Model,args,device):
    state=torch.load(path,map_location="cpu")
    model=Model(int(state["architecture_config"]["global_dim"]),"RGCondAttn",args).to(device)
    model.load_state_dict(state["model_state_dict"]);model.eval()
    p=state["validation_inference_state"];x=np.asarray(p["global_values"],np.float32);base=np.asarray(p["base_probability"],float)
    graphs={k:np.asarray(v) for k,v in p["graphs"].items()};bags=[np.asarray(v,np.float32) for v in p["bags"]]
    gt=torch.as_tensor(x,dtype=torch.float32,device=device);bl=torch.as_tensor(v4.probability_to_logit(base),dtype=torch.float32,device=device)
    gr=v4.graph_tensors(graphs,device);out=[]
    with torch.no_grad():
      for i,b in enumerate(bags):
        value,_=model(gt[i],bl[i],{k:v[i] for k,v in gr.items()},torch.as_tensor(b,dtype=torch.float32,device=device))
        out.append(float(torch.sigmoid(value).cpu()))
    return np.asarray(out),state

def train_seed(data,train,val,args,device,Model,metas,names,route,year,fold,seed):
    xtr,xva=within.fold_global24(data,train,val);base_train=within.inner_oof_base(data,train,int(args.split_seed)+year*10+fold)
    rawtr=within.subset_bags(data.bags,train);rawva=within.subset_bags(data.bags,val)
    selected,selection_loss=v4.select_epochs(xtr,data.s7[train],rawtr,base_train,data.labels[train],"RGCondAttn",args,device,seed)
    train_bags,test_bags,pa,state=fit_anomaly_state(rawtr,rawva,data.labels[train],args,seed+1707)
    graph_scaler=v4.fit_role_graph_scaler(data.s7[train]);train_graph=v4.transform_role_graphs(data.s7[train],graph_scaler);test_graph=v4.transform_role_graphs(data.s7[val],graph_scaler)
    global_tensor=torch.as_tensor(xtr,dtype=torch.float32,device=device);train_graph_t=v4.graph_tensors(train_graph,device)
    train_bag_t=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in train_bags]
    weights,correct=v4.fixkeep_weights(data.labels[train],base_train,args);v4.seed_everything(seed)
    model=Model(xtr.shape[1],"RGCondAttn",args).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    final_loss=v4.train_epochs(model,optimizer,global_tensor,base_train,train_graph_t,train_bag_t,data.labels[train],weights,correct,selected,args,seed+3301)
    ids=[data.ids[i] for i in val];apps,hosts=infer(model,xva,data.base[val],test_graph,test_bags,[metas[i] for i in val],ids,data.labels[val],route[val],names,device,seed,year,fold)
    resolved=json.loads(args.resolved.read_text(encoding="utf-8"));schema=Path(resolved["schema"])
    payload={"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"seed":seed,"year":year,"fold":fold,
      "best_epoch":int(selected),"early_stop_epoch_if_available":None,"selection_loss":float(selection_loss),"final_training_loss":float(final_loss),
      "architecture_config":{"global_dim":int(xtr.shape[1]),"hidden_dim":int(args.hidden_dim),"embedding_dim":int(args.embedding_dim),
        "graph_dim":int(args.graph_dim),"graph_heads":int(args.graph_heads),"graph_layers":int(args.graph_layers),"dropout":float(args.dropout)},
      "max_delta":2.25,"router_config":{"name":"DI25","budget":0.25,"route_column":"route_DI"},"feature_dimension":int(xtr.shape[1]),
      "feature_schema_path":str(schema),"feature_schema_sha256":sha_file(schema),"training_sample_sha256":sha_ids([data.ids[i] for i in train]),
      "validation_sample_sha256":sha_ids(ids),"n_train":len(train),"n_val":len(val),"role_graph_scaler_state":graph_scaler,
      "anomaly_fitting_state":state,"prototype_audit":pa,"validation_inference_state":{"ids":ids,"global_values":xva,
        "base_probability":data.base[val],"graphs":test_graph,"bags":test_bags}}
    cp=checkpoint_path(year,fold,seed);cp.parent.mkdir(parents=True,exist_ok=True);tmp=cp.with_suffix(".tmp");torch.save(payload,tmp);tmp.replace(cp)
    ap,hp=output_paths(year,fold,seed);write_csv(ap,apps);write_csv(hp,hosts)
    audit={"year":year,"fold":fold,"seed":seed,"checkpoint_path":str(cp),"checkpoint_sha256":sha_file(cp),"best_epoch":selected,
      "n_train":len(train),"n_val":len(val),"selection_loss":selection_loss,"final_training_loss":final_loss,
      "prob_mean":float(np.mean([r["probability"] for r in apps]))}
    return audit,apps

def valid_saved(year,fold,seed,Model,args,device,expected_hash=None):
    cp=checkpoint_path(year,fold,seed);ap,hp=output_paths(year,fold,seed)
    if not(cp.is_file() and ap.is_file() and hp.is_file()):return None
    if expected_hash and sha_file(cp)!=expected_hash:
      raise RuntimeError("BLOCKED_BY_CHECKPOINT_HASH_MISMATCH: "+str(cp))
    rows=read_csv(ap);prob=np.asarray([float(r["probability"]) for r in rows]);reload,state=reload_probability(cp,Model,args,device)
    if len(prob)!=int(state["n_val"]) or np.max(np.abs(prob-reload))>1e-7 or np.any((prob>=.5)!=(reload>=.5)):return None
    return {"year":year,"fold":fold,"seed":seed,"checkpoint_path":str(cp),"checkpoint_sha256":sha_file(cp),
      "best_epoch":state["best_epoch"],"n_train":state["n_train"],"n_val":state["n_val"],"selection_loss":state["selection_loss"],
      "final_training_loss":state["final_training_loss"],"prob_mean":float(np.mean(prob))}

def main():
    p=frozen.parser();p.add_argument("--mode",choices=("dryrun","full"),required=True);args=p.parse_args()
    if float(args.max_delta)!=2.25 or tuple(v4.parse_ints(args.seeds))!=SEEDS:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen seeds/max_delta changed")
    device=v4.resolve_device(args.device);data=within.load_data(args)
    # Reuse the already-audited exact loader from the previous instrumentation package.
    import run_instrumented_m1 as previous
    data.bags,metas=previous.load_packed_full(data.ids,args.future_host,args.future_ego)
    router={normal(r["sha256"]):r for r in read_csv(ROUTER)};route=np.asarray([int(float(router[x]["route_DI"]))==1 for x in data.ids])
    names=role_names(args);Model=instrumented_class(within.anomaly_bounded,v4);v4.AUGMENTED_REGION_DIM=within.future.ORIGINAL_AUGMENTED_DIM
    v4.RoleGraphConditionedEvidence=Model
    v4.v3.fit_scaled_dualproto_bags=make_fit_adapter(within.future.ORIGINAL_ANOMALY_FIT)
    years=[2018] if args.mode=="dryrun" else [2018,2019,2020,2021];folds=[1] if args.mode=="dryrun" else range(1,6)
    manifest=[];manifest_path=ROOT/("checkpoint_manifest_dryrun.csv" if args.mode=="dryrun" else "checkpoint_manifest.csv")
    existing={}
    if manifest_path.is_file():
      existing={(int(r["year"]),int(r["fold"]),int(r["seed"])):r["checkpoint_sha256"] for r in read_csv(manifest_path)}
    for year in years:
      for fold in folds:
        val=np.flatnonzero((data.years==year)&(data.folds==fold));train=np.flatnonzero((data.years==year)&(data.folds!=fold))
        for seed in SEEDS:
          started=time.perf_counter();audit=valid_saved(year,fold,seed,Model,args,device,existing.get((year,fold,seed)))
          if audit is None:audit,_=train_seed(data,train,val,args,device,Model,metas,names,route,year,fold,seed)
          audit["elapsed_seconds_this_invocation"]=time.perf_counter()-started;manifest.append(audit)
          print(f"[FINAL REPLAY] year={year} fold={fold} seed={seed} best_epoch={audit['best_epoch']} checkpoint_saved={audit['checkpoint_path']} checkpoint_sha256={audit['checkpoint_sha256']} n_val={audit['n_val']} prob_mean={audit['prob_mean']:.9f} elapsed={audit['elapsed_seconds_this_invocation']:.3f}",flush=True)
        print(f"year={year} fold {fold}/5 done",flush=True)
    write_csv(manifest_path,manifest)
    # Mandatory reload audit: dryrun all three; full one fixed seed/fold per year.
    selected=manifest if args.mode=="dryrun" else [r for r in manifest if int(r["fold"])==1 and int(r["seed"])==900000]
    reload_rows=[]
    for r in selected:
      ap,_=output_paths(int(r["year"]),int(r["fold"]),int(r["seed"]));saved=np.asarray([float(x["probability"]) for x in read_csv(ap)])
      value,_=reload_probability(Path(r["checkpoint_path"]),Model,args,device);mx=float(np.max(np.abs(saved-value)))
      reload_rows.append({"year":r["year"],"fold":r["fold"],"seed":r["seed"],"max_abs_prob_diff":mx,
        "prediction_difference_count":int(np.sum((saved>=.5)!=(value>=.5))),"passed":int(mx<=1e-7)})
    write_csv(ROOT/("checkpoint_reload_audit_dryrun.csv" if args.mode=="dryrun" else "checkpoint_reload_audit.csv"),reload_rows)
    if not all(int(r["passed"]) for r in reload_rows):raise RuntimeError("CHECKPOINT_RELOAD_FAIL")
    if args.mode=="dryrun":
      if len(manifest)!=3:raise RuntimeError("FINAL_REPLAY_DRYRUN_FAIL: expected 3 seeds")
      required_apk={"raw_delta","bounded_delta","probability","raw_delta_sign","bounded_delta_sign"}
      required_host={"gamma","anomaly_score","attention_logit","attention_weight","host_embedding_norm","ego_embedding_norm",
                     "local_embedding_norm","critical_host_flag","incoming_neighbor_roles","outgoing_neighbor_roles",
                     "external_background_value","internal_background_value","shared_commonness_value"}
      for seed in SEEDS:
        ap,hp=output_paths(2018,1,seed);ar=read_csv(ap);hr=read_csv(hp)
        if not ar or not required_apk.issubset(ar[0]) or not hr or not required_host.issubset(hr[0]):
          raise RuntimeError(f"FINAL_REPLAY_DRYRUN_FAIL: missing persisted fields for seed {seed}")
      print("FINAL_REPLAY_DRYRUN_PASS")
    else:
      if len(manifest)!=60 or len({r["checkpoint_sha256"] for r in manifest})!=60:raise RuntimeError("BLOCKED_BY_CONTRACT: expected 60 unique checkpoints")
      print("FINAL_REPLAY_TRAINING_AND_CHECKPOINTS_DONE")

if __name__=="__main__":main()
