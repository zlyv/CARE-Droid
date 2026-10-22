#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from lr_core import crossfit_lr_probability

FEATURES=Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")
AUDIT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1/audits/nested_lr_calls.json")
class NestedLR:
    def __init__(self):
        with np.load(FEATURES,allow_pickle=False) as z:self.ids=np.asarray([str(v).strip().casefold() for v in z["sha256"]]);self.p31=np.asarray(z["role_prevalence"],float)
        if self.p31.shape!=(799,31) or len(set(self.ids))!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: P31 feature cache changed")
        self.index={v:i for i,v in enumerate(self.ids)};self.calls=[]
    def inner_oof(self,train_rows,train_labels,compact_by_sha,compact_schema,original_names,inner_folds,split_seed,model_seed):
        del compact_by_sha,compact_schema,original_names,inner_folds,model_seed
        ids=[str(r["sha256"]).strip().casefold() for r in train_rows]
        if len(ids) not in (639,640) or any(v not in self.index for v in ids):raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected outer-train cohort")
        p=crossfit_lr_probability(self.p31[[self.index[v] for v in ids]],np.asarray(train_labels,int),int(split_seed))
        self.calls.append({"outer_train_n":len(ids),"probability":"nested inner-OOF calibrated P31-LR","labels_self_seen":False,"future_rows_loaded":0});AUDIT.parent.mkdir(parents=True,exist_ok=True);AUDIT.write_text(json.dumps(self.calls,indent=2)+"\n",encoding="utf-8")
        return p
