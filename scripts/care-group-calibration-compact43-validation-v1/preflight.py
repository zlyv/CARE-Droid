#!/usr/bin/env python3
from __future__ import annotations

import json
import numpy as np

from common import FORMAL, HOST, METHOD, OUTPUT, P31, dump, norm, read_csv, sha256_file


def main():
    missing = [str(path) for path in (FORMAL, P31, HOST) if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing frozen inputs:\n- " + "\n- ".join(missing))
    rows = read_csv(FORMAL)
    required = {"sha256", "fold", "label", "Global24__prob", "A25__routed",
                METHOD + "__prob", METHOD + "__bounded_delta"}
    if len(rows) != 799 or not required.issubset(rows[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal prediction contract changed")
    ids = [norm(row["sha256"]) for row in rows]
    fold = np.asarray([int(row["fold"]) for row in rows])
    route = np.asarray([int(float(row["A25__routed"])) for row in rows])
    if len(set(ids)) != 799 or [int(np.sum(fold == value)) for value in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fixed source folds changed")
    if int(route.sum()) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal DI25 route changed")
    with np.load(P31, allow_pickle=False) as data:
        if data["role_prevalence"].shape != (799, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 shape changed")
    with np.load(HOST, allow_pickle=False) as data:
        if data["host_values"].ndim != 2 or data["host_values"].shape[1] < 105:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache shape changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path, purpose in ((FORMAL, "formal source predictions"), (P31, "P31 source cache"), (HOST, "A0 Host cache")):
        manifest.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path), "purpose": purpose})
    dump(OUTPUT / "baseline_asset_manifest_before.json", manifest)
    dump(OUTPUT / "protocol.json", {
        "name": "Group-safe RF calibration and strict-nested Compact43 direction validation v1",
        "source_rows": 799, "source_year": 2017, "future_rows_loaded": 0,
        "formal_assets_read_only": True, "no_end_to_end_correction": True,
        "group_definition": "frozen prior definition: cosine>=0.9999 AND unit-L2<=0.015 connected components",
        "group_models": ["raw RF", "outer-train group-OOF Platt RF", "inverse-group-size weighted Platt RF"],
        "compact_representation_predeclared": "Compact43 per-APK mean+std+max_abs",
        "compact_controls": ["mean only", "std only", "max_abs only", "count only", "count-residualized", "host-count-stratified row shuffle"],
        "strict_context": "outer-train Stage-I side is produced by inner OOF calibrated P31-RF excluding outer validation",
        "fixed_test_context": "frozen formal outer-OOF P31-RF",
        "max_delta": 2.25, "max_delta_used": False, "seed": 20261008,
    })
    dump(OUTPUT / "preflight.json", {"status": "PASS", "rows": 799, "route": 200, "future_rows_loaded": 0})
    print("CARE_GROUP_COMPACT_VALIDATION_PREFLIGHT_PASS")
    print(json.dumps({"rows": 799, "route": 200, "future_rows_loaded": 0}))


if __name__ == "__main__":
    main()

