#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from common import PROJECT, SCRIPTS, METHOD


def prepare_imports():
    paths = (
        SCRIPTS / "care-p31rf-c43-haig-host-ablation-source-v1",
        SCRIPTS / "care-di25-graphglobal-path-source-v1",
        SCRIPTS / "care-stage2only-anomaly-v1/vendor",
        SCRIPTS / "care-rolegraph-conditioned-v4-script",
        SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
        SCRIPTS / "care-malben-host-v2",
        SCRIPTS,
    )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def install_a0(v4):
    prepare_imports()
    from haig_background import make_fit
    from haig_models import make_class
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.AUGMENTED_REGION_DIM = 142
    v4.RoleGraphConditionedEvidence = make_class(v4)
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)


def install_fold_safe_compact76(audit_module):
    original_builder = audit_module.allratio.original_compact_matrix_named

    def restore_fold_safe(train_rows, test_rows, compact_by_sha, compact_schema, canonical_original_names):
        fold_train, fold_test, fold_names = original_builder(train_rows, test_rows, compact_by_sha, compact_schema)
        fold_train = np.asarray(fold_train, float); fold_test = np.asarray(fold_test, float)
        fold_names = list(fold_names); canonical_original_names = list(canonical_original_names)
        if len(canonical_original_names) != 76 or len(set(canonical_original_names)) != 76:
            raise RuntimeError("BLOCKED_BY_CONTRACT: canonical Compact76 changed")
        unexpected = sorted(set(fold_names) - set(canonical_original_names))
        invalid = [name for name in unexpected if not (name.startswith("role_node__") or name.startswith("role_edge__"))]
        if invalid:
            raise RuntimeError("BLOCKED_BY_CONTRACT: non-role columns outside canonical Compact76: " + repr(invalid))
        index = {name: i for i, name in enumerate(fold_names)}
        train = np.zeros((len(train_rows), 76), float); test = np.zeros((len(test_rows), 76), float)
        for target, name in enumerate(canonical_original_names):
            source = index.get(name)
            if source is not None:
                train[:, target] = fold_train[:, source]; test[:, target] = fold_test[:, source]
        return train, test, canonical_original_names

    audit_module.restore_raw_compact76 = restore_fold_safe


def load_bags(cache_path, rows):
    with np.load(cache_path, allow_pickle=False) as z:
        ids = [str(x).strip().casefold() for x in z["sha256"]]
        values = np.asarray(z["host_values"], np.float32)
        offsets = np.asarray(z["host_offsets"], int)
    if values.ndim != 2 or values.shape[1] != 169 or offsets.shape != (len(ids) + 1,):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: A0 cache shape changed values={values.shape} offsets={offsets.shape}")
    pos = {sid: i for i, sid in enumerate(ids)}
    result = []
    for row in rows:
        sid = str(row["sha256"]).strip().casefold()
        if sid not in pos:
            raise RuntimeError("BLOCKED_BY_CONTRACT: A0 cache missing SHA " + sid)
        i = pos[sid]
        result.append(values[offsets[i]:offsets[i + 1]])
    return result
