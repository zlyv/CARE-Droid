#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage1_core import crossfit_base_probability

FEATURES=Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")


class NestedStage1:
    def __init__(self, mode, audit_path, jobs=4):
        if mode not in {"RF","MEAN"}: raise ValueError(mode)
        self.mode=mode; self.audit_path=Path(audit_path); self.jobs=int(jobs); self.calls=[]
        with np.load(FEATURES,allow_pickle=False) as z:
            self.ids=np.asarray([str(v).strip().casefold() for v in z["sha256"]])
            self.p31=np.asarray(z["role_prevalence"],float)
        if self.p31.shape!=(799,31) or len(set(self.ids.tolist()))!=799:
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 cache changed")
        self.index={v:i for i,v in enumerate(self.ids)}

    def inner_oof(self,train_rows,train_labels,compact_by_sha,compact_schema,
                  original_names,inner_folds,split_seed,model_seed):
        del compact_by_sha,compact_schema,original_names,inner_folds,model_seed
        ids=[str(r["sha256"]).strip().casefold() for r in train_rows]
        if len(ids) not in (639,640) or any(v not in self.index for v in ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected outer-train cohort")
        p=crossfit_base_probability(self.p31[[self.index[v] for v in ids]],
            np.asarray(train_labels,int),int(split_seed),self.mode,self.jobs)
        self.calls.append({"outer_train_n":len(ids),"mode":self.mode,
            "strict_nested_oof":True,"labels_self_seen":False,"future_rows_loaded":0})
        self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        self.audit_path.write_text(json.dumps(self.calls,indent=2)+"\n",encoding="utf-8")
        return p
