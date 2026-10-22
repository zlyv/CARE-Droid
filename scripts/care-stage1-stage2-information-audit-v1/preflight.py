#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from common import (
    FORMAL_METHOD, FORMAL_PREDICTIONS, HOST_CACHE, OUTPUT, P31_CACHE,
    ROUTE_FIELD, STAGE1_FIELD, dump, norm, read_csv, sha256_file,
)


def main():
    required = (FORMAL_PREDICTIONS, P31_CACHE, HOST_CACHE)
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty frozen inputs:\n- " + "\n- ".join(missing))
    rows = read_csv(FORMAL_PREDICTIONS)
    fields = set(rows[0]) if rows else set()
    needed = {"sha256", "fold", "label", STAGE1_FIELD, ROUTE_FIELD,
              FORMAL_METHOD + "__prob", FORMAL_METHOD + "__bounded_delta"}
    if len(rows) != 799 or not needed.issubset(fields):
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal prediction contract changed")
    ids = [norm(row["sha256"]) for row in rows]
    folds = np.asarray([int(float(row["fold"])) for row in rows])
    labels = np.asarray([int(float(row["label"])) for row in rows])
    route = np.asarray([int(float(row[ROUTE_FIELD])) for row in rows], bool)
    if len(set(ids)) != 799 or [int(np.sum(folds == f)) for f in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source SHA/fold contract changed")
    if (int(np.sum(labels == 0)), int(np.sum(labels == 1))) != (400, 399):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source class counts changed")
    if [int(np.sum(route & (folds == f))) for f in range(1, 6)] != [40] * 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: DI25 budget changed")
    with np.load(P31_CACHE, allow_pickle=False) as data:
        if not {"sha256", "role_prevalence"}.issubset(data.files):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 cache schema changed")
        p_ids = [norm(value) for value in data["sha256"]]
        p31 = np.asarray(data["role_prevalence"])
        if p31.shape != (799, 31) or set(p_ids) != set(ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 cohort/shape changed")
    with np.load(HOST_CACHE, allow_pickle=False) as data:
        needed_host = {"sha256", "host_values", "host_offsets"}
        if not needed_host.issubset(data.files):
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache schema changed")
        h_ids = [norm(value) for value in data["sha256"]]
        values = np.asarray(data["host_values"])
        offsets = np.asarray(data["host_offsets"])
        if len(h_ids) < 799 or values.ndim != 2 or values.shape[1] < 105 or offsets.shape[0] < 800:
            raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cache shape changed")
        if set(h_ids[:799]) != set(ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cohort changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path, purpose in ((FORMAL_PREDICTIONS, "frozen Stage-I/route/A0 predictions"),
                          (P31_CACHE, "frozen P31 source features"),
                          (HOST_CACHE, "frozen A0 Host content")):
        manifest.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path),
                         "purpose": purpose})
    dump(OUTPUT / "baseline_asset_manifest_before.json", manifest)
    protocol = {
        "name": "CARE Stage-I Robustness and Stage-II Static Information Ceiling Audit v1",
        "source_year": 2017, "source_rows": 799, "future_rows_loaded": 0,
        "future_used_for_selection": False,
        "formal_stage1": "calibrated P31-RF probability only",
        "router": "RF-anchored MeanOpposition DI25 Top25% (frozen)",
        "formal_stage2": "A0 Role C43-RQ bounded max_delta=2.25 (read-only reference)",
        "no_new_correction_model": True,
        "stage1_audits": ["capacity ladder", "P31 near-duplicate grouped CV", "feature masking", "calibration", "high-confidence errors"],
        "stage2_audits": ["conditional FN-vs-TN", "conditional FP-vs-TP", "count-only", "label shuffle", "nearest opposite-label collision"],
        "probe_model": "outer-fold StandardScaler + train-only PCA(max31) + balanced LogisticRegression",
        "stage2_train_scope": "all outer-train APKs within frozen Stage-I predicted side",
        "stage2_evaluation_scopes": ["all side-matched outer-test APKs", "formal DI25-routed side-matched outer-test APKs"],
        "seed": 20261007,
    }
    dump(OUTPUT / "protocol.json", protocol)
    dump(OUTPUT / "preflight.json", {
        "status": "PASS", "rows": 799, "fold_sizes": [160, 160, 160, 160, 159],
        "route_rows": int(route.sum()), "future_rows_loaded": 0,
        "input_hashes": {item["purpose"]: item["sha256"] for item in manifest},
    })
    print("CARE_INFORMATION_AUDIT_PREFLIGHT_PASS")
    print(json.dumps({"rows": 799, "route": 200, "future_rows_loaded": 0}))


if __name__ == "__main__":
    main()

