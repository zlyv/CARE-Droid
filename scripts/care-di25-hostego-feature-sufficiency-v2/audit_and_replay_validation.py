#!/usr/bin/env python3
"""Manifest gate, exact schema audit, and validation-only R4 replay."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from fs_common import *


def main():
    OUT.mkdir(parents=True,exist_ok=True); records=manifest_records(); audit=[]; accepted={}
    for r in records:
        y,f,s=r["year"],r["fold"],r["seed"]; cp=r["checkpoint_path_resolved"]; vi=r["validation_inputs_path_resolved"]
        row={"year":y,"fold":f,"seed":s,"checkpoint_path":str(cp),"validation_inputs_path":str(vi)}
        try:
            if not cp.is_file() or not vi.is_file(): raise RuntimeError("checkpoint or validation input missing")
            csha,vsha=sha_file(cp),sha_file(vi)
            expected_c=r.get("checkpoint_sha256","").lower(); expected_v=r.get("validation_inputs_sha256","").lower()
            if expected_c and csha!=expected_c: raise RuntimeError("checkpoint SHA256 mismatch")
            if expected_v and vsha!=expected_v: raise RuntimeError("validation input SHA256 mismatch")
            st=torch_load(cp); keys=set(st["model_state_dict"])
            actual_parameter_count=sum(int(v.numel()) for v in st["model_state_dict"].values())
            checks={"identity":(int(st["year"]),int(st["fold"]),int(st["seed"]))==(y,f,s),
                    "declared_parameter_count":int(st["parameter_count"])==34613,
                    "actual_parameter_count":actual_parameter_count==34613,
                    "max_delta":float(st["max_delta_logit"])==2.25,
                    "delta_head":any(k.startswith("delta_head.") for k in keys),
                    "host_ego":any(k.startswith("host_ego_fusion.") for k in keys),
                    "no_action":not any(k.startswith("action_head.") for k in keys),
                    "no_magnitude":not any(k.startswith("magnitude_head.") for k in keys)}
            if not all(checks.values()): raise RuntimeError("identity/architecture contract failed: "+json.dumps(checks))
            row.update({"checkpoint_sha256":csha,"validation_inputs_sha256":vsha,
                        "status":"ACCEPT_FROZEN_M1_VALIDATION_REPLAY","capability":"VALIDATION_REPLAY_ONLY"})
            if (y,f,s) in accepted: raise RuntimeError("duplicate manifest identity")
            accepted[(y,f,s)]=(r,st)
        except Exception as e: row.update({"status":"REJECT","reason":repr(e)})
        audit.append(row)
    write_csv(OUT/"manifest_checkpoint_audit.csv",audit)
    rejected=sum(r.get("status")!="ACCEPT_FROZEN_M1_VALIDATION_REPLAY" for r in audit)
    missing=sorted(EXPECTED-set(accepted)); gate={"passed":len(records)==60 and len(accepted)==60 and not missing and rejected==0,"manifest":str(MANIFEST),
      "manifest_rows":len(records),"accepted_validation_replay":len(accepted),"expected":60,
      "rejected_rows":rejected,
      "missing":[{"year":y,"fold":f,"seed":s} for y,f,s in missing],
      "capability":"ACCEPT_FROZEN_M1_VALIDATION_REPLAY" if not missing else "BLOCKED",
      "complete_train_reconstruction_state":False}
    write_json(OUT/"checkpoint_validation_replay_gate.json",gate)
    if not gate["passed"]: raise RuntimeError("BLOCKED_BY_CONTRACT: manifest validation gate failed")

    first=next(iter(accepted.values()))[1]; gd=int(first["global_dim"]); graph=int(first["graph_dim"]); emb=int(first["embedding_dim"])
    components=[("global24_features",gd),("stage1_base_logit",1),("role_graph_token",graph),
                ("critical_hostego_embedding",emb),("attention_weighted_hostego_embedding",emb),("scalar_summary",8)]
    schema={"status":"IDENTIFIED_FROM_ORIGINAL_FORWARD_AND_CHECKPOINT_SHAPES","representation":"R4_pre_delta",
      "dimension":sum(v for _,v in components),"delta_head_first_weight_shape":list(first["model_state_dict"]["delta_head.0.weight"].shape),
      "components":[{"name":n,"dimension":d} for n,d in components],
      "contains":{"Global24":True,"base_probability":False,"stage1_logit":True,"graph_context":True,
                  "local_evidence":True,"anomaly_evidence":True,"attention_features":True},
      "source_class":"InstrumentedHostEgoAnomaly",
      "source_forward":"instrumented_model.py: fused=torch.cat([global_features,base_logit,graph_token,critical,attended,scalar_summary])",
      "important_interpretation":"R4_pre_delta is a mixed Global24+Stage-I+Stage-II representation, not pure Stage-II z."}
    if schema["dimension"]!=113 or tuple(first["model_state_dict"]["delta_head.0.weight"].shape)!=(48,113):
        schema["status"]="NOT_IDENTIFIED";write_json(OUT/"pre_delta_feature_schema.json",schema);raise RuntimeError("pre-delta schema mismatch")
    write_json(OUT/"pre_delta_feature_schema.json",schema)

    history={(int(r["year"]),int(r["fold"]),int(r["seed"]),norm(r["sha256"])):r for r in read_csv(HISTORY)}
    rows=[];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for identity,(r,st) in sorted(accepted.items()):
        y,f,s=identity; model,_=build_model(st,device); saved=load_saved_inputs(r["validation_inputs_path_resolved"])
        result=forward_r4(model,saved,device); d=OUT/"validation_r4"/f"year_{y}"/f"fold_{f}";d.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(d/f"seed_{s}.npz",sha256=result["sha256"],R4_pre_delta=result["R4"],raw_delta=result["raw_delta"],
          bounded_delta=result["bounded_delta"],replayed_probability=result["probability"],base_probability=result["base_probability"])
        diffs={"raw":[],"bounded":[],"prob":[]};pred_diff=0
        for i,sha in enumerate(result["sha256"]):
            key=(y,f,s,norm(sha));h=history.get(key)
            if h is None: raise RuntimeError("historical replay anchor row missing: "+str(key))
            diffs["raw"].append(abs(result["raw_delta"][i]-float(h["raw_delta"])))
            diffs["bounded"].append(abs(result["bounded_delta"][i]-float(h["bounded_delta"])))
            diffs["prob"].append(abs(result["probability"][i]-float(h["probability"])))
            pred_diff += int((result["probability"][i]>=.5)!=(float(h["probability"])>=.5))
        row={"year":y,"fold":f,"seed":s,"n":len(result["sha256"]),"R4_dimension":result["R4"].shape[1],
             "raw_delta_max_abs_diff":max(diffs["raw"]),"raw_delta_mean_abs_diff":float(np.mean(diffs["raw"])),
             "bounded_delta_max_abs_diff":max(diffs["bounded"]),"bounded_delta_mean_abs_diff":float(np.mean(diffs["bounded"])),
             "probability_max_abs_diff":max(diffs["prob"]),"probability_mean_abs_diff":float(np.mean(diffs["prob"])),
             "prediction_difference_count":pred_diff,"parameter_hash_unchanged":result["parameter_hash_before"]==result["parameter_hash_after"]}
        row["passed"]=int(row["probability_max_abs_diff"]<=1e-7 and row["raw_delta_max_abs_diff"]<=1e-7 and
                          row["bounded_delta_max_abs_diff"]<=1e-7 and not pred_diff and row["parameter_hash_unchanged"])
        rows.append(row);print(f"validation replay {y} fold={f} seed={s} pass={row['passed']}",flush=True)
    write_csv(OUT/"validation_forward_replay_audit.csv",rows)
    passed=all(int(r["passed"]) for r in rows)
    report=["CARE-Droid validation-only R4 replay",f"accepted checkpoints: {len(accepted)}/60",
            "capability: VALIDATION_REPLAY_ONLY; checkpoint alone is not complete train reconstruction state",
            f"all_forward_replays_passed: {str(passed).lower()}",
            f"max_probability_abs_difference: {max(float(r['probability_max_abs_diff']) for r in rows):.12g}",
            f"prediction_difference_count: {sum(int(r['prediction_difference_count']) for r in rows)}"]
    (OUT/"VALIDATION_R4_REPLAY_REPORT.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    if not passed: raise RuntimeError("BLOCKED_BY_REPLAY_EQUIVALENCE")
    print("VALIDATION_R4_REPLAY_GATE_PASS")

if __name__=="__main__":main()
