#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from audit_core import ROLE_COUNT,reconstruct

REG=Path("artifacts/care-source-local-assets-v2/regions_2017.npz");CACHE=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz");OUT=Path("artifacts/care-di25-host-information-loss-audit-v1")

TOPO=["log_host_n","log_role_assignments","role_diversity","mean_role_incidence","multi_role_host","log_internal_edges","log_boundary_edges","internal_density","reciprocal_ratio","role_entropy","is_host","is_ego"]
COLS=[*(f"role_presence_{i}" for i in range(31)),*(f"role_log_count_{i}" for i in range(31)),*TOPO]

def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def finite_rho(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:return float("nan")
    return float(spearmanr(x,y).statistic)

def main():
    with np.load(REG,allow_pickle=False) as z:
        sha=[str(x).lower() for x in z["sha256"]];values=np.asarray(z["host_values"],dtype=float);offsets=np.asarray(z["host_offsets"],dtype=int);real=np.asarray(z["real_host_count"],dtype=int)
    if len(sha)!=799 or len(set(sha))!=799 or values.shape[1]!=74 or len(offsets)!=800:raise RuntimeError("BLOCKED_BY_CONTRACT: source Host NPZ contract changed")
    real_index=[]
    for i,n in enumerate(real):real_index.extend(range(offsets[i],offsets[i]+int(n)))
    old=values[np.asarray(real_index,dtype=int)]
    if len(old)!=95314:raise RuntimeError(f"BLOCKED_BY_CONTRACT: real hosts={len(old)}")
    column_rows=[]
    for j,name in enumerate(COLS):
        x=old[:,j];column_rows.append({"index":j,"name":name,"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max()),"unique_rounded_1e8":int(len(np.unique(np.round(x,8)))),"constant":int(np.std(x)<1e-12)})
    duplicate_max=float(np.max(np.abs(old[:,31:62]-np.log1p(np.maximum(old[:,:31],0)))))
    standardized=old-old.mean(0,keepdims=True);scale=old.std(0)
    nonconstant=scale>1e-12
    standardized[:,nonconstant]=standardized[:,nonconstant]/scale[nonconstant]
    effective_rank=int(np.linalg.matrix_rank(standardized,tol=1e-7))

    payloads={};cache_row_count=0;duplicate_cache_ids=[]
    with gzip.open(CACHE,"rt",encoding="utf-8") as h:
        for line in h:
            if not line.strip():continue
            cache_row_count+=1
            row=json.loads(line);sid=str(row.get("sha256") or row.get("sample_id") or "").strip().lower()
            if sid and row.get("payload") is not None:
                if sid in payloads:duplicate_cache_ids.append(sid)
                payloads[sid]=row["payload"]
    if duplicate_cache_ids:raise RuntimeError(f"BLOCKED_BY_CONTRACT: duplicate endpoint-cache IDs={len(duplicate_cache_ids)}")
    if set(payloads)!=set(sha):raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint cache mismatch cache={len(payloads)} source={len(sha)} missing={len(set(sha)-set(payloads))} outside={len(set(payloads)-set(sha))}")
    all_hosts=[];apk_rows=[];examples=[];observed_counts=[]
    for idx,sid in enumerate(sha):
        hosts,node_n,edge_n=reconstruct(payloads[sid]);observed_counts.append(len(hosts))
        if len(hosts)!=int(real[idx]):raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host count mismatch sha={sid} cache={len(hosts)} npz={real[idx]}")
        for h in hosts:h["sha256"]=sid;all_hosts.append(h)
        def arr(k):return np.asarray([x[k] for x in hosts],dtype=float) if hosts else np.asarray([],dtype=float)
        ep=arr("unique_endpoint_count");mult=np.asarray([np.max(x["role_endpoint_counts"]) for x in hosts],dtype=float) if hosts else np.asarray([])
        row={"sha256":sid,"host_count":len(hosts),"graph_node_count":node_n,"graph_edge_count":edge_n,
          "mean_unique_endpoints":float(ep.mean()) if len(ep) else 0.0,"max_unique_endpoints":float(ep.max()) if len(ep) else 0.0,
          "fraction_hosts_multi_endpoint":float(np.mean(ep>1)) if len(ep) else 0.0,"fraction_hosts_role_multiplicity":float(np.mean(mult>1)) if len(mult) else 0.0}
        for k in ("incoming_host_degree","outgoing_host_degree","icc_ratio","reflection_ratio","non_direct_ratio","upstream_ratio","downstream_ratio"):
            v=arr(k);row["mean_"+k]=float(v.mean()) if len(v) else 0.0;row["fraction_positive_"+k]=float(np.mean(v>0)) if len(v) else 0.0
        apk_rows.append(row)
    if len(all_hosts)!=95314:raise RuntimeError("BLOCKED_BY_CONTRACT: reconstructed Host total changed")
    # Rich, label-free examples for manual schema inspection.
    ranked=sorted(all_hosts,key=lambda h:(h["unique_endpoint_count"]+h["incoming_host_degree"]+h["outgoing_host_degree"]+10*(h["reflection_ratio"]+h["icc_ratio"])),reverse=True)[:20]
    for h in ranked:
        examples.append({"sha256":h["sha256"],"host_id":h["host_id"],"unique_endpoint_count":h["unique_endpoint_count"],"role_assignment_count":h["role_assignment_count"],"nonzero_roles":int(np.count_nonzero(h["role_endpoint_counts"])),"max_endpoints_in_one_role":int(np.max(h["role_endpoint_counts"])),"incoming_host_degree":h["incoming_host_degree"],"outgoing_host_degree":h["outgoing_host_degree"],"icc_ratio":h["icc_ratio"],"reflection_ratio":h["reflection_ratio"],"non_direct_ratio":h["non_direct_ratio"],"upstream_ratio":h["upstream_ratio"],"downstream_ratio":h["downstream_ratio"]})
    def count(pred):return sum(bool(pred(h)) for h in all_hosts)
    coverage={"total_apks":799,"endpoint_cache_rows":cache_row_count,"endpoint_cache_unique_apks":len(payloads),"total_real_hosts":len(all_hosts),"current_constant_columns":sum(r["constant"] for r in column_rows),"current_effective_matrix_rank":effective_rank,"duplicate_role_block_max_abs_diff":duplicate_max,
      "hosts_unique_endpoint_gt1":count(lambda h:h["unique_endpoint_count"]>1),"hosts_role_multiplicity_gt1":count(lambda h:np.max(h["role_endpoint_counts"])>1),
      "hosts_incoming_gt0":count(lambda h:h["incoming_host_degree"]>0),"hosts_outgoing_gt0":count(lambda h:h["outgoing_host_degree"]>0),
      "hosts_icc_positive":count(lambda h:h["icc_ratio"]>0),"hosts_reflection_positive":count(lambda h:h["reflection_ratio"]>0),"hosts_non_direct_positive":count(lambda h:h["non_direct_ratio"]>0),
      "hosts_upstream_positive":count(lambda h:h["upstream_ratio"]>0),"hosts_downstream_positive":count(lambda h:h["downstream_ratio"]>0)}
    for k in list(coverage):
        if k.startswith("hosts_"):coverage[k+"_rate"]=coverage[k]/len(all_hosts)
    size_rows=[];numeric=[k for k in apk_rows[0] if k not in ("sha256","graph_node_count","graph_edge_count")]
    nodes=np.asarray([r["graph_node_count"] for r in apk_rows]);edges=np.asarray([r["graph_edge_count"] for r in apk_rows])
    for k in numeric:
        x=np.asarray([r[k] for r in apk_rows]);size_rows.append({"feature":k,"rho_graph_nodes":finite_rho(x,nodes),"rho_graph_edges":finite_rho(x,edges),"max_abs_rho":float(np.nanmax(np.abs([finite_rho(x,nodes),finite_rho(x,edges)]))) if np.std(x)>1e-12 else float("nan")})
    schema={"dimension":74,"capacity_matched":True,"block_0_30":"role presence (retained)","block_31_61":"log1p distinct sensitive endpoints per role (replaces deterministic duplicate)","block_62_73":["log1p unique endpoint count","log1p role assignment count","endpoint role diversity","endpoint role entropy","multi-role endpoint ratio","log1p incoming Host degree","log1p outgoing Host degree","reciprocal Host ratio","ICC ratio","reflection ratio","non-direct ratio","upstream/downstream balance"],"source":"existing frozen endpoint cache; no APK reanalysis","labels_used":False}
    write(OUT/"existing_host_column_audit.csv",column_rows);write(OUT/"per_apk_host_v2_summary.csv",apk_rows);write(OUT/"size_association.csv",size_rows);write(OUT/"manual_examples.csv",examples)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"host_v2_feature_schema.json").write_text(json.dumps(schema,indent=2)+"\n",encoding="utf-8");(OUT/"coverage.json").write_text(json.dumps(coverage,indent=2)+"\n",encoding="utf-8")
    lines=["CARE-Droid Host information-loss audit v1","","Label-free read-only audit of frozen 2017 Host NPZ and endpoint cache. No training, inference, APK reanalysis, Ego, or future row.","",f"real_hosts={len(all_hosts)}",f"current_constant_columns={coverage['current_constant_columns']}/74",f"current_effective_matrix_rank={effective_rank}/74",f"duplicate_role_block_max_abs_diff={duplicate_max:.12g}",""]
    for k,v in coverage.items():
        if k.startswith("hosts_") and not k.endswith("_rate"):lines.append(f"{k}={v}/{len(all_hosts)} ({coverage[k+'_rate']:.3%})")
    # host_count is retained in the CSV as a size-reference baseline, but it
    # must not hide the strongest association of a proposed Host-V2 feature.
    maxrho=max((r for r in size_rows if r["feature"]!="host_count" and np.isfinite(r["max_abs_rho"])),key=lambda r:r["max_abs_rho"])
    lines += ["",f"strongest_APK_size_association={maxrho['feature']} max_abs_rho={maxrho['max_abs_rho']:.6f}","", "Decision boundary:","- Multiplicity/context features are candidates only if they are nontrivial and not merely APK-size proxies.","- This audit does not authorize training; inspect coverage and size_association.csv first."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("HOST_INFORMATION_LOSS_AUDIT_DONE");print((OUT/"REPORT.txt").read_text())
if __name__=="__main__":main()
