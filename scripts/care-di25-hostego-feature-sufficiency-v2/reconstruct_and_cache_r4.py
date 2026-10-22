#!/usr/bin/env python3
"""Reconstruct deterministic fold preprocessing and cache frozen R4 features."""
from __future__ import annotations
import numpy as np
import torch
from fs_common import *


def compare(name,a,b):
    a=np.asarray(a);b=np.asarray(b)
    row={"component":name,"shape_reconstructed":str(tuple(a.shape)),"shape_saved":str(tuple(b.shape))}
    if a.shape!=b.shape:
        row.update({"exact_equal":0,"max_abs_diff":float("inf"),"mean_abs_diff":float("inf"),"passed":0});return row
    if name in ("sha256","bag_offsets"):
        equal=np.array_equal(a.astype(str) if name=="sha256" else a,b.astype(str) if name=="sha256" else b)
        row.update({"exact_equal":int(equal),"max_abs_diff":0.0 if equal else float("inf"),"mean_abs_diff":0.0 if equal else float("inf"),"passed":int(equal)})
    else:
        d=np.abs(a.astype(np.float64)-b.astype(np.float64));mx=float(np.max(d)) if d.size else 0.;mean=float(np.mean(d)) if d.size else 0.
        row.update({"exact_equal":int(np.array_equal(a,b)),"max_abs_diff":mx,"mean_abs_diff":mean,"passed":int(mx<=1e-6)})
    return row


def save_cache(path,result,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,sha256=result["sha256"],label=np.asarray(meta["label"],np.int64),
      routed=np.asarray(meta["routed"],np.int8),p_stage1=np.asarray(meta["p_stage1"],np.float64),
      stage1_logit=v4.probability_to_logit(np.asarray(meta["p_stage1"],np.float64)),R4_pre_delta=result["R4"],
      raw_delta=result["raw_delta"],bounded_delta=result["bounded_delta"],probability=result["probability"])


def main():
    gate=json.loads((OUT/"checkpoint_validation_replay_gate.json").read_text(encoding="utf-8"))
    replay=read_csv(OUT/"validation_forward_replay_audit.csv")
    if not gate["passed"] or len(replay)!=60 or not all(int(r["passed"]) for r in replay):
        raise RuntimeError("BLOCKED_BY_CONTRACT: validation replay gate not passed")
    records=manifest_records(); first=torch_load(records[0]["checkpoint_path_resolved"]); args=checkpoint_args(first)
    data=load_full_data(args);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit=[]; max_change=0.0
    for r in records:
        y,f,s=r["year"],r["fold"],r["seed"];st=torch_load(r["checkpoint_path_resolved"]);args=checkpoint_args(st)
        validation=np.flatnonzero((data.years==y)&(data.folds==f));train=np.flatnonzero((data.years==y)&(data.folds!=f))
        prep=fit_preprocessing(data,train,validation,args,s);rv=input_dict(prep["validation"]);saved=load_saved_inputs(r["validation_inputs_path_resolved"])
        fold_rows=[]
        for name in ("sha256","global_values","base_probability","graph_node","graph_edge","graph_global","bag_values","bag_offsets"):
            one=compare(name,rv[name],saved[name]);one.update({"year":y,"fold":f,"seed":s});fold_rows.append(one)
        passed=all(int(x["passed"]) for x in fold_rows);audit.extend(fold_rows)
        if not passed:
            print(f"preprocessing reconstruction {y} fold={f} seed={s} FAIL",flush=True);continue
        model,_=build_model(st,device);before=state_hash(model)
        tr=input_dict(prep["train"]);tr_result=forward_r4(model,tr,device);va_result=forward_r4(model,rv,device);after=state_hash(model)
        max_change=max(max_change,0.0 if before==after else float("inf"))
        root=OUT/"cache"/str(y)/f"fold_{f}"/f"seed_{s}"
        save_cache(root/"train_r4.npz",tr_result,{"label":data.labels[train],"routed":data.route[train],"p_stage1":prep["train"]["base_probability"]})
        save_cache(root/"validation_r4.npz",va_result,{"label":data.labels[validation],"routed":data.route[validation],"p_stage1":data.base[validation]})
        print(f"preprocessing/cache {y} fold={f} seed={s} PASS",flush=True)
    write_csv(OUT/"preprocessing_reconstruction_audit.csv",audit)
    identities={(int(r["year"]),int(r["fold"]),int(r["seed"])) for r in audit if r["component"]=="sha256" and int(r["passed"])}
    all_pass=len(audit)==60*8 and all(int(r["passed"]) for r in audit) and identities==EXPECTED
    no_train={"stage2_neural_training_invoked":False,"optimizer_step_count":0,"checkpoint_weights_modified":False,
      "frozen_parameter_max_abs_change":max_change,"validation_labels_used_for_preprocessing":False,
      "validation_labels_used_for_probe_training":False,"preprocessing_reconstructed":True,"neural_model_retrained":False,
      "preprocessing_reconstruction_passed":all_pass}
    write_json(OUT/"NO_STAGE2_RETRAINING_AUDIT.json",no_train)
    write_json(OUT/"feature_leakage_audit.json",{
      "validation_label_used_in_representation":False,"validation_label_used_in_scaler":False,
      "validation_label_used_in_probe_training":False,"year_used_as_feature":False,
      "future_information_used":False,"frozen_model_parameter_change_max_abs":max_change,
      "scope":"independent same-year development/diagnostic OOF; not temporal generalization"})
    if not all_pass or max_change!=0:raise RuntimeError("BLOCKED_BY_PREPROCESSING_RECONSTRUCTION")
    print("PREPROCESSING_RECONSTRUCTION_AND_R4_CACHE_PASS")

if __name__=="__main__":main()
