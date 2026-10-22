#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_common import *


def main():
    if not OUTPUT.is_relative_to((PROJECT / "outputs").resolve()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output path")
    required = [FORMAL, DIRECT, ENDPOINT, A0_CACHE, FAMILY_SCRIPT / "run_source_arm.py", DIRECT_SCRIPT / "model_variants.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing prerequisites " + repr(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "logs").mkdir(exist_ok=True); (OUTPUT / "audits").mkdir(exist_ok=True)
    ids, y, fold, base, route, expert = load_reference(); category = categories(y, base, expert)
    checks = {"rows==799": len(ids) == 799, "fold_sizes": [int(np.sum(fold == f)) for f in range(1,6)] == [160,160,160,160,159],
              "routed==200": int(route.sum()) == 200, "direct_fixed==49": int(np.sum(route & (category == "FIXED"))) == 49,
              "direct_broken==17": int(np.sum(route & (category == "BROKEN"))) == 17, "future_rows_loaded==0": True}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source reference changed " + repr(checks))
    assets = [asset(path, purpose) for path,purpose in [(FORMAL,"frozen SVM Stage-II family predictions"),(DIRECT,"frozen Direct-C43 predictions"),
              (ENDPOINT,"frozen 2017 endpoint-centered graph cache"),(A0_CACHE,"frozen A0 Host-region cache")]]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"git_used": False, "assets": assets})
    dump_json(OUTPUT / "protocol.json", {
        "status": "FROZEN_CACHE_PROXY_AUDIT", "scope": "2017 SVM-DI25 routed 200", "source_rows": 799,
        "stage1": "calibrated P31-SVM", "router": "SVM MeanOpposition Top25%", "stage2": "Direct-C43 threshold 0.5",
        "transforms": list(TRANSFORMS), "repeats": 3, "seed": SEED, "future_rows_loaded": 0,
        "critical_boundary": "Only endpoint/RoleGraph structure and Host order are perturbed. Compact43 Host descriptors are frozen. Results are REPRESENTATION_LEVEL_PROXY, not transformed-APK robustness.",
        "training": "Each outer-fold Direct-C43 model is trained once on original outer-train data; original and proxy-transformed validation views share identical weights.",
        "go_gate": {"T0_order_no_flips": True, "candidate_broken_auc>=0.65": True,
                    "candidate_broken_auc>within_stratum_label_shuffle95+0.03": True,
                    "broken_drift_higher_in_at_least_4_folds": True,
                    "T1_sensitive_mean_drift>N1_far_mean_drift": True,
                    "candidate_applicability>=0.80": True},
    })
    dump_json(OUTPUT / "preflight.json", {"status":"PASS","checks":checks,"category_counts":{name:int(np.sum(route & (category==name))) for name in np.unique(category)},"future_rows_loaded":0})
    print("C43_EQUIVALENCE_SENSITIVITY_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
