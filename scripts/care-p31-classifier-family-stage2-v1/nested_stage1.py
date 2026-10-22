#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import ARMS, SOURCE_P31, dump_json, norm
from stage1_core import crossfit_arm_probability


class NestedStage1:
    def __init__(self, arm, audit_path, jobs=4):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm; self.audit_path = Path(audit_path); self.jobs = int(jobs); self.calls = []
        with np.load(SOURCE_P31, allow_pickle=False) as archive:
            self.ids = np.asarray([norm(value) for value in archive["sha256"]])
            self.p31 = np.asarray(archive["role_prevalence"], float)
        if self.p31.shape != (799, 31) or len(set(self.ids.tolist())) != 799:
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 cache changed")
        self.index = {sid: index for index, sid in enumerate(self.ids)}

    def inner_oof(self, train_rows, train_labels, compact_by_sha, compact_schema,
                  original_names, inner_folds, split_seed, model_seed):
        del compact_by_sha, compact_schema, original_names, inner_folds, model_seed
        ids = [norm(row["sha256"]) for row in train_rows]
        if len(ids) not in (639, 640) or any(sid not in self.index for sid in ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected outer-train cohort")
        probability = crossfit_arm_probability(self.p31[[self.index[sid] for sid in ids]],
            np.asarray(train_labels, int), int(split_seed), self.arm, self.jobs)
        self.calls.append({"outer_train_n": len(ids), "arm": self.arm, "strict_nested_oof": True,
            "labels_self_seen": False, "future_rows_loaded": 0})
        dump_json(self.audit_path, self.calls)
        return probability
