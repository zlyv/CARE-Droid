#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p31_core import crossfit_mean_probability

FEATURES = Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")
AUDIT_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1/audits")


def norm(value):
    return str(value or "").strip().casefold()


class NestedP31:
    def __init__(self, jobs=4):
        with np.load(FEATURES, allow_pickle=False) as values:
            self.ids = np.asarray([norm(value) for value in values["sha256"]])
            self.p31 = np.asarray(values["role_prevalence"], float)
        if self.p31.shape != (799, 31) or len(set(self.ids.tolist())) != 799:
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 source feature cache changed")
        self.index = {value: index for index, value in enumerate(self.ids)}
        self.jobs = jobs
        self.calls = []

    def inner_oof(self, train_rows, train_labels, compact_by_sha, compact_schema,
                  original_names, inner_folds, split_seed, model_seed):
        del compact_by_sha, compact_schema, original_names, inner_folds, model_seed
        ids = [norm(row["sha256"]) for row in train_rows]
        if len(ids) not in (639, 640) or any(value not in self.index for value in ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: unexpected Stage-II outer-train cohort")
        matrix = self.p31[[self.index[value] for value in ids]]
        probability = crossfit_mean_probability(
            matrix, np.asarray(train_labels, int), int(split_seed), jobs=self.jobs
        )
        self.calls.append({
            "outer_train_n": len(ids), "dimension": 31,
            "outer_train_probability_source": "nested inner-OOF P31 Mean Committee",
            "split_seed": int(split_seed), "labels_self_seen": False,
            "future_rows_loaded": 0,
        })
        AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
        (AUDIT_ROOT / "nested_p31_calls.json").write_text(
            json.dumps(self.calls, indent=2) + "\n", encoding="utf-8"
        )
        return probability
