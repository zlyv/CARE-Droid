#!/usr/bin/env python3
"""Rerun frozen same-year M1 and persist otherwise unavailable internals."""
from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "care-di25-hostego-within-year-v1"
MATRIX = ORIGINAL / "within_matrix"
for value in (HERE, ORIGINAL, MATRIX, MATRIX / "vendor"):
    sys.path.insert(0, str(value))

import run_m1_within_year as frozen
import run_within_year_arm as within
from host_ego_encoder import ApkEgoMeta, make_fit_adapter, pack
from instrumented_model import instrumented_class

v4 = within.v4
ROOT = Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis")
RUN = ROOT / "instrumented_rerun"
OLD = Path("artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv")
ROUTER = Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
SEEDS = [900000, 900001, 900002]

ACTIVE: dict = {}
SEED_CALLS: list[list[dict]] = []


def normal(value): return str(value or "").strip().lower()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def load_packed_full(expected, host_path, ego_path):
    with np.load(host_path, allow_pickle=False) as h, np.load(ego_path, allow_pickle=False) as e:
        ids = [normal(x) for x in h["sha256"].astype(str)]
        eids = [normal(x) for x in e["sha256"].astype(str)]
        hl = {x:i for i,x in enumerate(ids)}; el = {x:i for i,x in enumerate(eids)}
        if set(hl) != set(expected) or set(el) != set(expected):
            raise RuntimeError("BLOCKED_BY_CONTRACT: future Host/Ego lineage differs")
        hv=np.asarray(h["host_values"],np.float32); ho=np.asarray(h["host_offsets"],np.int64)
        rc=np.asarray(h["real_host_count"],np.int64); eo=np.asarray(e["ego_offsets"],np.int64)
        host_ids=e["host_ids"].astype(str); roles=np.asarray(e["host_primary_role"],np.int64)
        status=np.asarray(e["ego_status"],np.int64); nodes=np.asarray(e["ego_node_count"],np.int64)
        edge_n=np.asarray(e["ego_edge_count"],np.int64)
        io=np.asarray(e["incoming_offsets"],np.int64); oo=np.asarray(e["outgoing_offsets"],np.int64)
        ip=np.asarray(e["incoming_host_positions"],np.int64); op=np.asarray(e["outgoing_host_positions"],np.int64)
        bags=[]; metas=[]
        for sha in expected:
            i,ei=hl[sha],el[sha]; hb,he=int(ho[i]),int(ho[i+1]); eb,ee=int(eo[ei]),int(eo[ei+1])
            host=hv[hb:he]; n=int(rc[i])
            if ee-eb != n or (n and len(host)!=n) or (not n and len(host)!=1):
                raise RuntimeError("BLOCKED_BY_CONTRACT: Host/Ego alignment failed: "+sha)
            incoming=[]; outgoing=[]; edge_set=set()
            for g in range(eb,ee):
                inc=(ip[io[g]:io[g+1]]-eb).astype(int).tolist()
                out=(op[oo[g]:oo[g+1]]-eb).astype(int).tolist()
                if len(inc)!=len(set(inc)) or len(out)!=len(set(out)) or (g-eb) in inc+out:
                    raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate/self Ego neighbor: "+sha)
                incoming.append(inc); outgoing.append(out); edge_set.update((g-eb,v) for v in out)
            bags.append(pack(host, sorted(edge_set)))
            metas.append(ApkEgoMeta(
                sha, host_ids[eb:ee].tolist() if n else ["__missing_host__"],
                roles[eb:ee].astype(int).tolist() if n else [-1], incoming if n else [[]],
                outgoing if n else [[]], status[eb:ee].astype(int).tolist() if n else [0],
                nodes[eb:ee].astype(int).tolist() if n else [1], edge_n[eb:ee].astype(int).tolist() if n else [0], n))
    return bags, metas


def capture_predict(model, global_values, base_probability, graph_values, bags, device):
    global_tensor=torch.as_tensor(global_values,dtype=torch.float32,device=device)
    base_logit=torch.as_tensor(v4.probability_to_logit(base_probability),dtype=torch.float32,device=device)
    graph=v4.graph_tensors(graph_values,device)
    bag_tensors=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in bags]
    probability=[]; diagnostic=[]; calls=[]; model.eval()
    with torch.no_grad():
        for index in range(len(bags)):
            one_graph={key:value[index] for key,value in graph.items()}
            logit,diag=model(global_tensor[index],base_logit[index],one_graph,bag_tensors[index])
            probability.append(float(torch.sigmoid(logit).cpu()))
            diagnostic.append(diag.detach().cpu().numpy())
            c=model.last_internal_capture
            compact={"probability": probability[-1], "raw_delta": c["raw_delta"],
                     "bounded_delta": c["bounded_delta"], "attention_entropy": c["attention_entropy"],
                     "max_attention": c["max_attention"], "graph_entropy": c["graph_entropy"]}
            if ACTIVE["routed"][index]:
                compact["hosts"]={key:np.asarray(value).tolist() for key,value in c.items()
                                  if isinstance(value,np.ndarray)}
                compact["critical_index"]=c["critical_index"]
            calls.append(compact)
    SEED_CALLS.append(calls)
    return np.asarray(probability),np.asarray(diagnostic)


def mean(values): return float(np.mean(np.asarray(values,float)))


def parser():
    p=frozen.parser(); p.set_defaults(output=RUN / "unused_original_output")
    return p


def main():
    args=parser().parse_args(); RUN.mkdir(parents=True,exist_ok=True)
    if float(args.max_delta)!=2.25 or v4.parse_ints(args.seeds)!=SEEDS:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen max_delta/seeds changed")
    if not OLD.is_file() or not ROUTER.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked M1/router prediction is missing")
    old={normal(r["sha256"]):r for r in read_csv(OLD)}
    router={normal(r["sha256"]):r for r in read_csv(ROUTER)}
    if len(old)!=1989 or len(router)!=1989 or set(old)!=set(router):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked same-year cohort differs")
    device=v4.resolve_device(args.device); data=within.load_data(args)
    data.bags,metas=load_packed_full(data.ids,args.future_host,args.future_ego)
    route=np.asarray([int(float(router[x]["route_DI"]))==1 for x in data.ids],bool)
    manifest_candidates=[args.future_ego.parent/"local_asset_manifest.json",
                         Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/local_asset_manifest.json")]
    manifest=next((p for p in manifest_candidates if p.is_file()),None)
    if manifest is None: raise RuntimeError("BLOCKED_BY_CONTRACT: role-order manifest missing")
    role_names=json.loads(manifest.read_text(encoding="utf-8"))["model_role_order"]

    Model=instrumented_class(within.anomaly_bounded,v4)
    v4.AUGMENTED_REGION_DIM=within.future.ORIGINAL_AUGMENTED_DIM
    v4.RoleGraphConditionedEvidence=Model
    v4.v3.fit_scaled_dualproto_bags=make_fit_adapter(within.future.ORIGINAL_ANOMALY_FIT)
    v4.predict_model=capture_predict
    app_rows=[]; host_rows=[]; fit_rows=[]; started=time.perf_counter()
    for year in sorted(set(data.years.tolist())):
        for fold in range(1,6):
            validation=np.flatnonzero((data.years==year)&(data.folds==fold)); train=np.flatnonzero((data.years==year)&(data.folds!=fold))
            xtr,xva=within.fold_global24(data,train,validation)
            base_train=within.inner_oof_base(data,train,int(args.split_seed)+year*10+fold)
            global ACTIVE,SEED_CALLS
            ACTIVE={"routed":route[validation]}; SEED_CALLS=[]
            probability,diagnostic,audits=v4.fit_predict_variant(
                xtr,xva,data.s7[train],data.s7[validation],within.subset_bags(data.bags,train),
                within.subset_bags(data.bags,validation),base_train,data.base[validation],data.labels[train],
                "RGCondAttn",SEEDS,args,device)
            if len(SEED_CALLS)!=3 or {int(a["parameter_n"]) for a in audits}!={34613}:
                raise RuntimeError("BLOCKED_BY_REPRODUCTION: seed/capacity contract changed")
            fit_rows.extend({"year":year,"fold":fold,**a} for a in audits)
            for j,idx in enumerate(validation):
                sha=data.ids[idx]; ps=[call[j]["probability"] for call in SEED_CALLS]
                raws=[call[j]["raw_delta"] for call in SEED_CALLS]; bounds=[call[j]["bounded_delta"] for call in SEED_CALLS]
                p=float(probability[j]); base=float(data.base[idx]); base_l=float(v4.probability_to_logit(np.asarray([base]))[0])
                final_l=float(v4.probability_to_logit(np.asarray([p]))[0]); eff=final_l-base_l
                if abs(eff)>=2.25: raise RuntimeError("BLOCKED_BY_INTERNAL_CONSISTENCY: ensemble delta outside bound")
                eff_raw=float(np.arctanh(eff/2.25))
                seed_tanh=max(abs(bounds[k]-2.25*np.tanh(raws[k])) for k in range(3))
                app={"year":int(year),"fold":int(fold),"sha256":sha,"label":int(data.labels[idx]),
                     "routed":int(route[idx]),"p_stage1":base,"rerun_M1_prob":p,
                     "locked_M1_prob":float(old[sha]["p_expert_bounded"]),
                     "raw_delta":eff_raw,"bounded_delta":eff,"seed_mean_raw_delta":mean(raws),
                     "seed_mean_bounded_delta":mean(bounds),"seed_probability_sd":float(np.std(ps)),
                     "seed_tanh_max_abs_diff":seed_tanh,"real_host_count":metas[idx].real_host_count,
                     "nontrivial_ego_count":int(sum(metas[idx].status))}
                if route[idx]:
                    host_fields=["external_background_value","internal_background_value","shared_commonness_value",
                                 "anomaly_score","host_embedding_norm","ego_embedding_norm","gamma","gamma_ego_norm",
                                 "local_embedding_norm","ego_used","incoming_count","outgoing_count","attention_logit","attention_weight"]
                    n=metas[idx].real_host_count; aggregate={}
                    for key in host_fields:
                        aggregate[key]=np.mean(np.asarray([c[j]["hosts"][key][:n] for c in SEED_CALLS],float),axis=0) if n else np.asarray([])
                    crit_votes=[int(c[j]["critical_index"]) for c in SEED_CALLS]
                    critical=int(np.argmax(aggregate["attention_logit"])) if n else -1
                    app.update({"mean_gamma":mean(aggregate["gamma"]) if n else 0.0,
                                "max_gamma":float(np.max(aggregate["gamma"])) if n else 0.0,
                                "mean_anomaly":mean(aggregate["anomaly_score"]) if n else 0.0,
                                "max_anomaly":float(np.max(aggregate["anomaly_score"])) if n else 0.0,
                                "attention_entropy":mean([c[j]["attention_entropy"] for c in SEED_CALLS]),
                                "max_attention":mean([c[j]["max_attention"] for c in SEED_CALLS]),
                                "critical_host_index":critical,
                                "critical_host_id":metas[idx].host_ids[critical] if critical>=0 else "__missing_host__",
                                "critical_role":(role_names[int(metas[idx].primary_roles[critical])] if critical>=0 and 0<=int(metas[idx].primary_roles[critical])<len(role_names) else "missing"),
                                "critical_anomaly":float(aggregate["anomaly_score"][critical]) if critical>=0 else 0.0,
                                "critical_gamma":float(aggregate["gamma"][critical]) if critical>=0 else 0.0,
                                "critical_attention_logit":float(aggregate["attention_logit"][critical]) if critical>=0 else 0.0,
                                "critical_attention_weight":float(aggregate["attention_weight"][critical]) if critical>=0 else 0.0,
                                "critical_vote_fraction":crit_votes.count(critical)/3 if critical>=0 else 0.0})
                    for h in range(n):
                        inc=metas[idx].incoming[h]; out=metas[idx].outgoing[h]
                        def rn(v): return role_names[int(v)] if 0<=int(v)<len(role_names) else str(v)
                        host_rows.append({"year":int(year),"fold":int(fold),"sha256":sha,"label":int(data.labels[idx]),
                            "host_index":h,"host_id":metas[idx].host_ids[h],"center_role":rn(metas[idx].primary_roles[h]),
                            "ego_status":int(metas[idx].status[h]),"ego_node_count":int(metas[idx].node_count[h]),
                            "ego_edge_count":int(metas[idx].edge_count[h]),"incoming_neighbor_indices":json.dumps(inc),
                            "neighbor_count":len(set(inc+out)),
                            "outgoing_neighbor_indices":json.dumps(out),"incoming_neighbor_roles":json.dumps([rn(metas[idx].primary_roles[v]) for v in inc]),
                            "outgoing_neighbor_roles":json.dumps([rn(metas[idx].primary_roles[v]) for v in out]),
                            **{key:float(aggregate[key][h]) for key in host_fields},
                            "critical_host":int(h==critical),"critical_host_flag":int(h==critical),
                            "critical_seed_vote_fraction":crit_votes.count(h)/3})
                app_rows.append(app)
            write_csv(RUN/"apk_internal_checkpoint.csv",app_rows); write_csv(RUN/"host_internal_checkpoint.csv",host_rows)
            write_csv(RUN/"fit_audit.csv",fit_rows)
            print(f"instrumented M1 {year} fold {fold}/5 done",flush=True)
    write_csv(RUN/"apk_internal.csv",app_rows); write_csv(RUN/"host_internal.csv",host_rows)
    (RUN/"run_protocol.json").write_text(json.dumps({"seeds":SEEDS,"max_delta":2.25,"device":str(device),
        "training_seconds":time.perf_counter()-started,"arithmetic_changed":False,
        "prototype_column_mapping":{"-4":"external_background_value","-3":"internal_background_value",
        "-2":"shared_commonness_value","-1":"anomaly_score/residual"}},indent=2)+"\n",encoding="utf-8")
    print("INSTRUMENTED_M1_RERUN_DONE",flush=True)


if __name__=="__main__": main()
