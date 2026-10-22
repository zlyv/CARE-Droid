#!/usr/bin/env python3
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import numpy as np


FEATURES = Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz")
ROUTER = Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv")
UPSTREAM = Path("scripts/care-litelocal-semantic-source-v1/run_source_oof.py")


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value):
    return str(value or "").strip().casefold()


def main():
    missing = [str(p) for p in (FEATURES, ROUTER, UPSTREAM) if not p.is_file() or p.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- " + "\n- ".join(missing))
    rows = read(ROUTER)
    ids = [norm(r.get("sha256")) for r in rows]
    years = {int(float(r.get("year") or 2017)) for r in rows}
    folds = Counter(int(float(r["fold"])) for r in rows)
    route = Counter(int(float(r["fold"])) for r in rows if int(float(r["route_DI"])) == 1)
    if len(rows) != 799 or len(set(ids)) != 799 or years != {2017}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
    if dict(folds) != {1: 160, 2: 160, 3: 160, 4: 160, 5: 159} or dict(route) != {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fold/DI25 contract changed")
    with np.load(FEATURES, allow_pickle=False) as z:
        required = {"sha256", "role_prevalence", "within_method_cooccurrence", "directed_adjacent_transitions", "audit_values", "audit_names"}
        if not required.issubset(z.files):
            raise RuntimeError("BLOCKED_BY_CONTRACT: LiteLocal feature schema changed")
        feature_ids = [norm(x) for x in z["sha256"]]
        if len(feature_ids) != 799 or set(feature_ids) != set(ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: LiteLocal feature cohort changed")
        if z["role_prevalence"].shape != (799, 31) or z["within_method_cooccurrence"].shape != (799, 465) or z["directed_adjacent_transitions"].shape != (799, 961):
            raise RuntimeError("BLOCKED_BY_CONTRACT: LiteLocal raw dimensions changed")
    audit_source = Path("scripts/care-litelocal-stage1-attribution-source-v1/run_audit.py").read_text(encoding="utf-8")
    if "seed = 20260925 + 100000 * fold" not in audit_source:
        raise RuntimeError("BLOCKED_BY_CONTRACT: original LiteLocal model seed anchor changed")
    print("LITELOCAL_STAGE1_ATTRIBUTION_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_DI=200")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("feature_cache=reuse_existing apk_reanalysis=false")
    print("model_seed_anchor=20260925 matches_original_LiteLocal=true")
    print("future_rows_loaded=0 stage2_training_invoked=false")


if __name__ == "__main__":
    main()
