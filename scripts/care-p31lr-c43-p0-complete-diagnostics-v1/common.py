#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np

ROOT=Path("artifacts/care-p31lr-c43-p0-complete-diagnostics-v1")
METHOD="P31LR__Opp25__C43Prov6Zero__Bounded"
EXPECTED_YEARS={2018:793,2019:399,2020:397,2021:400}
EXPECTED_ROUTE={2018:198,2019:100,2020:99,2021:100}

def _base():
    path=Path("scripts/care-litelocal-p31-c43-complete-diagnostics-v1/common.py")
    spec=importlib.util.spec_from_file_location("_p31mean_diag_common",path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
BASE=None
def base():
    global BASE
    if BASE is None:BASE=_base()
    return BASE
def norm(v):return str(v or "").strip().casefold()
def read_csv(path):return base().read_csv(Path(path))
def write_csv(path,rows):return base().write_csv(Path(path),rows)
def aligned_rows(rows,path):return base().aligned_rows(rows,Path(path))
def subset_bags(bags,index):return [bags[int(i)] for i in index]
def align_fold_global24(*args):return base().align_fold_global24(*args)
def resolved():
    import json
    return json.loads((ROOT/"preflight/resolved_inputs.json").read_text())

def install_paths():
    scripts=Path.cwd()/"scripts"
    paths=(scripts/"care-p31lr-c43-p0-complete-diagnostics-v1",scripts/"care-litelocal-p31lr-c43-prov6-source-v1",scripts/"care-litelocal-p31-diagnostics-v1",scripts/"care-litelocal-p31-c43-complete-diagnostics-v1",scripts/"care-litelocal-p31-c43-source-v1",scripts/"care-di25-compact43-relationquery-source-v1",scripts/"care-di25-graphglobal-path-source-v1",scripts/"care-route-stage2-future-matrix-v1",scripts/"care-route-stage2-future-matrix-v1/vendor",scripts/"care-rolegraph-conditioned-v4-script",scripts/"care-rgcondattn-dedup-bounded-source-v1",scripts/"care-malben-host-v2",scripts)
    for path in reversed(paths):
        if str(path) not in sys.path:sys.path.insert(0,str(path))

def load_bags(path,target_rows,expected_width):return base().load_bags(Path(path),target_rows,expected_width)
def zero6(bags):
    out=[]
    for bag in bags:
        x=np.asarray(bag,np.float32)
        if x.ndim!=2 or x.shape[1] not in (105,111):raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache width must be 105 or 111")
        if x.shape[1]==111:
            if np.max(np.abs(x[:,105:]))>0:raise RuntimeError("BLOCKED_BY_CONTRACT: P0 cache tail is not zero")
            out.append(x)
        else:out.append(np.concatenate([x,np.zeros((len(x),6),np.float32)],axis=1))
    return out

def install_p0(v4):
    from prov6_background import make_fit
    from prov6_models import make_class
    v4.AUGMENTED_REGION_DIM=84;v4.RoleGraphConditionedEvidence=make_class(v4);v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3)

class NestedAnnualLR:
    def __init__(self,ids,features):
        from lr_core import crossfit_lr_probability
        self.crossfit=crossfit_lr_probability;self.lookup={norm(v):i for i,v in enumerate(ids)};self.x=np.asarray(features,float);self.calls=[]
    def inner_oof(self,train_rows,train_labels,compact_by_sha,compact_schema,original_names,inner_folds,split_seed,model_seed):
        del compact_by_sha,compact_schema,original_names,inner_folds,model_seed
        idx=np.asarray([self.lookup[norm(r["sha256"])] for r in train_rows]);p=self.crossfit(self.x[idx],np.asarray(train_labels,int),int(split_seed));self.calls.append({"train_n":len(idx),"target_labels_seen":False});return p

def metric_counts(y,base_p,final_p,route):
    y=np.asarray(y,int);b=np.asarray(base_p,float)>=.5;f=np.asarray(final_p,float)>=.5;r=np.asarray(route,bool)
    return {"routed":int(r.sum()),"fixed":int(np.sum(~(b==y)&(f==y))),"broken":int(np.sum((b==y)&~(f==y))),"net":int(np.sum(f==y)-np.sum(b==y)),"fixed_FN":int(np.sum((y==1)&~b&f)),"fixed_FP":int(np.sum((y==0)&b&~f)),"broken_TN_to_FP":int(np.sum((y==0)&~b&f)),"broken_TP_to_FN":int(np.sum((y==1)&b&~f))}
