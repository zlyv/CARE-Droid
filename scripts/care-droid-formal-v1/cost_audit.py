#!/usr/bin/env python3
from __future__ import annotations
import os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from common import ROOT,RANDOM_SEEDS,read_csv,write_csv,dump,deterministic_route

rows=read_csv(ROOT/"formal_predictions.csv");n=len(rows);sha=np.asarray([r["sha256"] for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows]);base=np.asarray([float(r["stage1_prob"]) for r in rows]);score=np.asarray([float(r["router_score"]) for r in rows]);formal_route=np.asarray([int(r["routed"]) for r in rows],bool);candidate=np.asarray([float(r["bounded_prob"]) for r in rows])
try:
    pip_freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=False,timeout=60).stdout.splitlines()
except Exception:pip_freeze=[]
try:
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True,check=False,timeout=10).stdout.strip().splitlines()
except Exception:gpu=[]
env={"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count(),"gpu":gpu,"thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUDA_VISIBLE_DEVICES")},"packages":pip_freeze,"timing_cache_state":"post-cache arrays already resident; extraction excluded","future_rows_loaded":0}
try:
    import psutil
    env["ram_bytes"]=int(psutil.virtual_memory().total)
except Exception:env["ram_bytes"]=None
dump(ROOT/"environment_manifest.json",env)

# Inventory only. Heterogeneous historical timing is never silently mixed into the formal comparison.
inventory=[]
for top in (Path("artifacts/care-litelocal-semantic-source-v1"),Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")):
    if not top.exists():continue
    for path in top.rglob("*"):
        if path.is_file() and any(term in path.name.lower() for term in ("timing","runtime","time")):inventory.append({"path":str(path),"bytes":path.stat().st_size,"used_in_formal_cost":0,"reason":"not a complete same-run C0-C3 timing under identical cache conditions"})
write_csv(ROOT/"runtime_evidence_inventory.csv",inventory)

repetitions=500;raw=[]
def measure(name,fn):
    values=[]
    for rep in range(repetitions):
        start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e9)
    for rep,value in enumerate(values):raw.append({"scope":"post_cache_decision_only","operation":name,"repetition":rep,"wall_seconds":value})
    return {"mean":float(np.mean(values)),"median":float(np.median(values)),"p95":float(np.quantile(values,.95))}
def stage1_decision():return base>=.5
def di25_route():
    out=np.zeros(n,bool)
    for f in range(1,6):
        idx=np.flatnonzero(fold==f);order=np.lexsort((sha[idx],-score[idx]));out[idx[order[:int(np.sum(formal_route[idx]))]]]=True
    return out
def random_route():
    out=np.zeros(n,bool)
    for f in range(1,6):
        idx=np.flatnonzero(fold==f);out|=deterministic_route(idx,sha,int(np.sum(formal_route[idx])),RANDOM_SEEDS[0]+f*1009)
    return out
def compose(route):return np.where(route,candidate,base)>=.5
bench={"stage1_threshold":measure("stage1_threshold",stage1_decision),"DI25_routing":measure("DI25_routing",di25_route),"random25_routing":measure("random25_routing",random_route),"C0_compose":measure("C0_compose",lambda:compose(np.zeros(n,bool))),"C1_compose":measure("C1_compose",lambda:compose(random_route())),"C2_compose":measure("C2_compose",lambda:compose(formal_route)),"C3_compose":measure("C3_compose",lambda:compose(np.ones(n,bool)))}
write_csv(ROOT/"runtime_raw.csv",raw)
deep={"C0_STAGE1_ONLY":0,"C1_RANDOM25":200,"C2_DI25":200,"C3_FULL100":799};summary=[]
for arm,count in deep.items():
    op={"C0_STAGE1_ONLY":"C0_compose","C1_RANDOM25":"C1_compose","C2_DI25":"C2_compose","C3_FULL100":"C3_compose"}[arm]
    summary.append({"arm":arm,"number_of_apks":799,"deep_analyzed_apks":count,"deep_analysis_ratio":count/799,"stage1_extraction_seconds":"","stage1_inference_seconds":"","routing_seconds_measured_post_cache":bench["DI25_routing"]["mean"] if arm=="C2_DI25" else (bench["random25_routing"]["mean"] if arm=="C1_RANDOM25" else 0.),"deep_extraction_seconds":"","deep_inference_seconds":"","bounded_correction_seconds_included_in_post_cache_compose":bench[op]["mean"],"total_wall_seconds":"","mean_time_per_APK":"","median_time_per_APK":"","p95_time_per_APK":"","relative_cost_vs_P31_only":"","relative_cost_vs_Full100":"","deep_analysis_work_reduction_vs_full":1-count/799,"actual_runtime_claim_authorized":0,"timing_status":"INCOMPLETE_NO_SAME_RUN_EXTRACTION_OR_MODEL_INFERENCE_TIMING"})
write_csv(ROOT/"cost_summary.csv",summary);dump(ROOT/"cost_analysis_status.json",{"status":"COST_TIMING_INCOMPLETE","post_cache_microbenchmark":bench,"actual_selective_runtime_reduction":None,"work_count_reduction_vs_full":1-200/799,"reason":"No complete same-machine same-cache timing for P31 extraction, C43 extraction and C43 inference was preserved. Work-count reduction is reported separately and is not called runtime savings.","future_rows_loaded":0})
print("COST_ANALYSIS_DONE")

