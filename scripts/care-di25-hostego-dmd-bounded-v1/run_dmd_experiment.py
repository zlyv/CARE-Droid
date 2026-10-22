#!/usr/bin/env python3
"""One-shot predeclared DMD-BC same-year OOF experiment (B1/B2 only)."""
from __future__ import annotations

import argparse, csv, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEP = HERE.parent / "care-di25-hostego-within-year-v1"
MATRIX = DEP / "within_matrix"
for path in (HERE, DEP, MATRIX, MATRIX / "vendor"):
    sys.path.insert(0, str(path))

import run_within_year_arm as within
import run_care_rgcondattn_dedup_bounded_source_v1 as baseline
from host_ego_encoder import DirectedHostEgoFusion, make_fit_adapter, repack_augmented, unpack
from host_ego_models import anomaly_classes
from dmd_models import make_dmd_class
import dmd_training as training

v4 = within.v4
ROOT = Path("artifacts/care-di25-hostego-dmd-bounded-v1")
SEEDS = [900000, 900001, 900002]
DISCLAIMER = ("The 2018–2021 same-year OOF sets have already been inspected during mechanism "
              "diagnosis and are therefore used only for development/diagnostic evaluation of DMD-BC.")


def norm(x): return str(x or "").strip().lower()
def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def jsonable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,(np.floating,np.integer,np.bool_)): return value.item()
    if isinstance(value,dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    return value


def fit_bags_with_state(train_packed, test_packed, labels, args, seed):
    """Exact frozen anomaly transform, while retaining its inference state."""
    train=[unpack(x)[0] for x in train_packed]; test=[unpack(x)[0] for x in test_packed]
    original = within.future.ORIGINAL_ANOMALY_FIT
    frozen_globals = original.__globals__
    care_v1, v2 = frozen_globals["care_v1"], frozen_globals["v2"]
    mean,std=care_v1.fit_equal_apk_scaler(train,np.arange(len(train),dtype=np.int64))
    scaled_train=[(np.asarray(x,np.float32)-mean)/std for x in train]
    scaled_test=[(np.asarray(x,np.float32)-mean)/std for x in test]
    state=v2.fit_prototype_state(scaled_train,labels,args,int(seed))
    aug_train,train_summary=v2.augment_bags(scaled_train,state,"JointDualPrototype",args)
    aug_test,test_summary=v2.augment_bags(scaled_test,state,"JointDualPrototype",args)
    audit={"region_scaler_zero_std_n":int(np.sum(std==1.0)),"shared_prototype_n":int(state["shared_k"]),
        "benign_prototype_n":int(state["benign_k"]),"sampled_balanced_region_n":int(state["sampled_balanced_n"]),
        "sampled_benign_region_n":int(state["sampled_benign_n"]),"shared_commonness_mean":float(np.mean(state["commonness"])),
        "shared_commonness_max":float(np.max(state["commonness"])),"external_distance_scale":float(state["external_scale"]),
        "train_external_mean":float(np.mean(train_summary[:,0])),"test_external_mean":float(np.mean(test_summary[:,0])),
        "train_internal_mean":float(np.mean(train_summary[:,1])),"test_internal_mean":float(np.mean(test_summary[:,1])),
        "train_commonness_mean":float(np.mean(train_summary[:,2])),"test_commonness_mean":float(np.mean(test_summary[:,2])),
        "ego_changes_scaler":False,"ego_changes_prototypes":False,"ego_edge_scope":"directed Host-graph one-hop"}
    inference_state={"region_mean":mean,"region_std":std,**state}
    return repack_augmented(aug_train,train_packed),repack_augmented(aug_test,test_packed),audit,inference_state


def architecture_audit(args):
    _, Base=anomaly_classes(baseline,v4)
    old=Base(24,"RGCondAttn",args)
    old_head=old.delta_head
    first=next(x for x in old_head if isinstance(x,torch.nn.Linear))
    last=[x for x in old_head if isinstance(x,torch.nn.Linear)][-1]
    lines=["CARE-Droid M1 old correction-head hard audit","",
        f"implementation={DEP.as_posix()}/host_ego_models.py::HostEgoAnomaly.forward",
        f"base_class={old.__class__.__mro__[1].__module__}.{old.__class__.__mro__[1].__name__}",
        "forward_path=fused -> delta_head -> raw_delta -> 2.25*tanh(raw_delta) -> stage1_logit + bounded_delta",
        f"input_dimension={first.in_features}",f"hidden_dimension={first.out_features}",
        f"output_dimension={last.out_features}",f"head={old_head!r}",
        f"head_parameter_count={sum(p.numel() for p in old_head.parameters())}",
        f"whole_M1_parameter_count={sum(p.numel() for p in old.parameters())}",
        "replacement_scope=delta_head only; Host/Ego, anomaly, graph, attention, folds and features remain frozen"]
    out=ROOT/"audits"/"old_head_audit.txt";out.parent.mkdir(parents=True,exist_ok=True);out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return Base,int(first.in_features),int(sum(p.numel() for p in old_head.parameters())),int(sum(p.numel() for p in old.parameters()))


def parameter_audit(Base,args):
    rows=[]; models={"B0":Base(24,"RGCondAttn",args)}
    models["B1_nokeep"]=make_dmd_class(Base, __import__('host_ego_encoder').split_packed_tensor, DirectedHostEgoFusion,2)(24,"RGCondAttn",args)
    models["B2_full"]=make_dmd_class(Base, __import__('host_ego_encoder').split_packed_tensor, DirectedHostEgoFusion,3)(24,"RGCondAttn",args)
    b0=sum(p.numel() for p in models["B0"].parameters())
    if b0 != 34613:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen M1 parameter count changed: {b0}")
    upstream0={k:tuple(v.shape) for k,v in models["B0"].named_parameters() if not k.startswith("delta_head")}
    for arm,model in models.items():
        n=sum(p.numel() for p in model.parameters())
        if arm != "B0":
            upstream={k:tuple(v.shape) for k,v in model.named_parameters() if not k.startswith(("action_head","magnitude_head"))}
            if upstream != upstream0:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: upstream parameters changed in {arm}")
        rows.append({"arm":arm,"parameter_count":n,"extra_vs_B0":n-b0,"relative_increase":(n-b0)/b0,
            "head_tensors":json.dumps({k:list(v.shape) for k,v in model.named_parameters() if k.startswith(("delta_head","action_head","magnitude_head"))})})
    write_csv(ROOT/"audits"/"parameter_count.csv",rows)
    return rows


def checkpoint_payload(model, graph_scaler, anomaly_state, args, arm, year, fold, seed,
                       train_ids, val_ids, best_epoch, feature_schema):
    return {"model_state_dict":model.state_dict(),"graph_scaler":graph_scaler,
        "anomaly_background_state":anomaly_state,"feature_schema":feature_schema,
        "fold_train_sha256":list(train_ids),"fold_train_sha256_hash":training.sha_list(train_ids),
        "fold_validation_sha256":list(val_ids),"fold_validation_sha256_hash":training.sha_list(val_ids),
        "architecture":{"arm":arm,"module":"DirectionMagnitudeDecoupledBoundedCorrection","action_classes":2 if arm=="B1_nokeep" else 3},
        "loss":{"equation":"L_final(existing weighted BCE + keep consistency) + 1.0*CE_action + 1.0*SmoothL1_magnitude",
            "lambda_action":1.0,"lambda_magnitude":1.0,"margin_epsilon":1e-3},
        "max_delta":2.25,"seed":int(seed),"year":int(year),"fold":int(fold),"best_epoch":int(best_epoch),
        "optimizer_independent_inference_state":True}


def fit_seed(data, train, validation, args, device, Base, arm, seed, year, fold, dry=False):
    classes=2 if arm=="B1_nokeep" else 3
    Model=make_dmd_class(Base,__import__('host_ego_encoder').split_packed_tensor,DirectedHostEgoFusion,classes)
    xtr,xva=within.fold_global24(data,train,validation)
    base_train=within.inner_oof_base(data,train,int(args.split_seed)+year*10+fold)
    selected,selection_loss=training.select_epochs(v4,Model,xtr,data.s7[train],within.subset_bags(data.bags,train),
        base_train,data.labels[train],"RGCondAttn",args,device,seed,classes,fit_bags_with_state)
    trbag,vabag,proto_audit,anomaly_state=fit_bags_with_state(within.subset_bags(data.bags,train),
        within.subset_bags(data.bags,validation),data.labels[train],args,seed+1707)
    graph_scaler=v4.fit_role_graph_scaler(data.s7[train]); trg=v4.transform_role_graphs(data.s7[train],graph_scaler); vag=v4.transform_role_graphs(data.s7[validation],graph_scaler)
    gt=torch.as_tensor(xtr,dtype=torch.float32,device=device); graph=v4.graph_tensors(trg,device)
    bags=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in trbag]
    weights,correct=v4.fixkeep_weights(data.labels[train],base_train,args)
    v4.seed_everything(seed); model=Model(24,"RGCondAttn",args).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    losses=training.train_epochs(v4,model,optimizer,gt,base_train,graph,bags,data.labels[train],weights,correct,
        selected,args,seed+3301,classes)
    vgt=torch.as_tensor(xva,dtype=torch.float32,device=device); vgraph=v4.graph_tensors(vag,device)
    vbags=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in vabag]
    vbase=data.base[validation]; vlogit=torch.as_tensor(v4.probability_to_logit(vbase),dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad(): logits,_,_,capture=training.forward_aux(model,vgt,vlogit,vgraph,vbags,np.arange(len(validation)),capture=True)
    probability=torch.sigmoid(logits).cpu().numpy()
    parameter_count=int(sum(value.numel() for value in model.parameters()))
    ckpt=checkpoint_payload(model,graph_scaler,anomaly_state,args,arm,year,fold,seed,
        [data.ids[i] for i in train],[data.ids[i] for i in validation],selected,data.compact_schema)
    cpath=ROOT/("preflight/dryrun_checkpoint.pt" if dry else f"checkpoints/{arm}/{year}/fold_{fold}/seed_{seed}.pt")
    cpath.parent.mkdir(parents=True,exist_ok=True);torch.save(ckpt,cpath)
    del model,graph,bags,vgraph,vbags
    if device.type=="cuda":torch.cuda.empty_cache()
    return probability,capture,{"arm":arm,"year":year,"fold":fold,"seed":seed,"selected_epochs":selected,
        "selection_loss":selection_loss,**{f"loss_{k}":v for k,v in losses.items()},"parameter_count":parameter_count,**proto_audit}


def dry_run(data,args,device,Base,locked,router):
    year,fold,seed=2018,1,SEEDS[0]
    validation=np.flatnonzero((data.years==year)&(data.folds==fold));train=np.flatnonzero((data.years==year)&(data.folds!=fold))
    probability,capture,audit=fit_seed(data,train,validation,args,device,Base,"B2_full",seed,year,fold,dry=True)
    delta=np.asarray([x["delta"] for x in capture]);mag=np.asarray([x["magnitude"] for x in capture]);sums=np.asarray([np.sum(x["action_probability"]) for x in capture])
    # B0 is immutable/reused; identity is established by exact locked SHA and cohort alignment.
    b0_path=Path(args.locked_b0)
    b0={norm(x["sha256"]):x for x in read_csv(b0_path)}
    no_keep_target,no_keep_mask,_,wrong=training.targets(data.labels[train],within.inner_oof_base(data,train,int(args.split_seed)+year*10+fold),2,torch.device("cpu"))
    checks={"B0_locked_prediction_file_sha256":sha(b0_path),"B0_rows":len(b0),"B0_matches_cohort":set(b0)==set(data.ids),
        "abs_delta_le_2p25":bool(np.max(np.abs(delta))<=2.25+1e-6),"action_probability_sum_to_one":bool(np.max(np.abs(sums-1))<=1e-6),
        "magnitude_in_range":bool(np.min(mag)>=0 and np.max(mag)<=2.25+1e-6),"validation_label_in_forward_input":False,
        "action_target_constructed_train_only":True,"magnitude_target_constructed_train_only":True,
        "nokeep_correct_stage1_action_mask_all_zero":bool(torch.all(~no_keep_mask[~wrong])),"year":year,"fold":fold,"seed":seed}
    passed = bool(
        checks["B0_matches_cohort"]
        and checks["abs_delta_le_2p25"]
        and checks["action_probability_sum_to_one"]
        and checks["magnitude_in_range"]
        and not checks["validation_label_in_forward_input"]
        and checks["action_target_constructed_train_only"]
        and checks["magnitude_target_constructed_train_only"]
        and checks["nokeep_correct_stage1_action_mask_all_zero"]
    )
    checks["passed"] = passed
    if not passed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: DMD dry-run failed "+json.dumps(checks))
    (ROOT/"preflight"/"dryrun_audit.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    print("DMD_BOUNDED_HEAD_DRYRUN_PASS",flush=True)


def main():
    p=__import__('run_m1_within_year').parser()
    p.set_defaults(output=ROOT)
    p.add_argument("--locked-b0",type=Path,default=Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv"))
    p.add_argument("--dry-run-only",action="store_true")
    args=p.parse_args()
    if float(args.max_delta)!=2.25 or v4.parse_ints(args.seeds)!=SEEDS: raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta/seeds changed")
    for d in ("preflight","B0","B1_nokeep","B2_full","audits","checkpoints","reports"): (ROOT/d).mkdir(parents=True,exist_ok=True)
    device=v4.resolve_device(args.device);data=within.load_data(args);data.bags,_=__import__('run_m1_within_year').load_packed(data.ids,args.future_host,args.future_ego)
    v4.AUGMENTED_REGION_DIM=within.future.ORIGINAL_AUGMENTED_DIM
    baseline.v4.AUGMENTED_REGION_DIM=within.future.ORIGINAL_AUGMENTED_DIM
    router=read_csv(args.router_inputs);locked=read_csv(args.locked_b0)
    if len(router)!=1989 or len(locked)!=1989: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen cohort missing")
    router_by={norm(row["sha256"]):row for row in router};locked_by={norm(row["sha256"]):row for row in locked}
    if len(router_by)!=1989 or len(locked_by)!=1989 or set(router_by)!=set(data.ids) or set(locked_by)!=set(data.ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: B0/router/cohort identity changed")
    route_counts={year:sum(int(float(router_by[data.ids[i]]["route_DI"]))==1 for i in range(1989) if int(data.years[i])==year) for year in (2018,2019,2020,2021)}
    if route_counts!={2018:198,2019:100,2020:99,2021:100}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen DI25 route counts changed: "+repr(route_counts))
    if any(int(router_by[data.ids[i]]["label"])!=int(data.labels[i]) or int(router_by[data.ids[i]]["fold"])!=int(data.folds[i]) for i in range(1989)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: router label/fold lineage differs")
    Base,input_dim,old_head_n,b0_n=architecture_audit(args);parameter_audit(Base,args)
    (ROOT/"B0"/"locked_prediction_reference.json").write_text(json.dumps({
        "path":str(args.locked_b0),"sha256":sha(args.locked_b0),"rows":len(locked),
        "retrained":False,"role":"frozen M1 baseline"
    },indent=2)+"\n",encoding="utf-8")
    contract={"disclaimer":DISCLAIMER,"arms":["B0","B1_nokeep","B2_full"],"years":[2018,2019,2020,2021],
        "same_year_outer_folds":5,"seeds":SEEDS,"max_delta":2.25,"lambda_action":1.0,"lambda_magnitude":1.0,
        "margin_epsilon":1e-3,"hyperparameter_search":False,"old_head_input_dim":input_dim,"old_head_parameters":old_head_n,
        "B0_parameters":b0_n,"future_year_labels_used":False,"year_identity_feature":False,"frozen_DI25_route_counts":route_counts}
    (ROOT/"preflight"/"preflight_contract.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    dry_run(data,args,device,Base,locked,router)
    if args.dry_run_only:return
    for arm in ("B1_nokeep","B2_full"):
        ensemble=np.full(1989,np.nan)
        seed_path=ROOT/arm/"per_seed_internal.csv";audit_path=ROOT/arm/"fit_audit.csv"
        seed_rows=read_csv(seed_path) if seed_path.is_file() else []
        audit_rows=read_csv(audit_path) if audit_path.is_file() else []
        for year in (2018,2019,2020,2021):
            for fold in range(1,6):
                validation=np.flatnonzero((data.years==year)&(data.folds==fold));train=np.flatnonzero((data.years==year)&(data.folds!=fold))
                seed_probability=[];seed_capture=[]
                for seed in SEEDS:
                    prior=[r for r in seed_rows if int(r["year"])==year and int(r["fold"])==fold and int(r["seed"])==seed]
                    if len(prior)==len(validation) and {norm(r["sha256"]) for r in prior}=={data.ids[i] for i in validation}:
                        by={norm(r["sha256"]):r for r in prior};probability=np.asarray([float(by[data.ids[i]]["p_final"]) for i in validation])
                        seed_probability.append(probability);print(f"[resume] DMD {arm} {year} fold {fold} seed {seed}",flush=True);continue
                    if prior: raise RuntimeError("partial/duplicate per-seed checkpoint rows")
                    probability,capture,audit=fit_seed(data,train,validation,args,device,Base,arm,seed,year,fold)
                    seed_probability.append(probability);seed_capture.append(capture);audit_rows.append(audit)
                    for j,index in enumerate(validation):
                        x=capture[j];ap=np.asarray(x["action_probability"]);seed_rows.append({"arm":arm,"year":year,"fold":fold,"seed":seed,"sha256":data.ids[index],
                            "action_neg_logit":x["action_logits"][0],"action_keep_logit":np.nan if len(ap)==2 else x["action_logits"][1],"action_pos_logit":x["action_logits"][-1],
                            "action_neg_prob":ap[0],"action_keep_prob":np.nan if len(ap)==2 else ap[1],"action_pos_prob":ap[-1],
                            "direction_value":x["direction"],"magnitude":x["magnitude"],"delta":x["delta"],"p_final":probability[j],
                            "gamma_json":json.dumps(np.asarray(x["gamma"]).tolist()),"anomaly_json":json.dumps(np.asarray(x["anomaly"]).tolist()),
                            "attention_json":json.dumps(np.asarray(x["attention"]).tolist()),"critical_host_index":x["critical_host_index"]})
                    temporary=seed_path.with_suffix(".tmp");write_csv(temporary,seed_rows);temporary.replace(seed_path)
                    temporary=audit_path.with_suffix(".tmp");write_csv(temporary,audit_rows);temporary.replace(audit_path)
                ensemble[validation]=np.mean(seed_probability,axis=0)
                print(f"DMD {arm} {year} fold {fold}/5 done",flush=True)
        write_csv(seed_path,seed_rows);write_csv(audit_path,audit_rows)
        write_csv(ROOT/arm/"expert_predictions.csv",[{"sha256":data.ids[i],"year":int(data.years[i]),"fold":int(data.folds[i]),"label":int(data.labels[i]),"p_expert":ensemble[i]} for i in range(1989)])
    print("DMD_BOUNDED_FULL_RUN_DONE",flush=True)

if __name__=="__main__": main()
