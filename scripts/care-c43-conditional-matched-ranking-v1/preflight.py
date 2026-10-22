#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from common import (DIRECT_PREDICTIONS, DIRECT_ROOT, FOLDS, OUTPUT, PROJECT, SOURCE_BAGS, SOURCE_MEMBER,
                    SOURCE_P31, dump_json, file_record, host_summary, load_source)


def main():
    if not OUTPUT.is_relative_to((PROJECT / "outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    required = [(SOURCE_MEMBER, "frozen member OOF predictions"), (SOURCE_P31, "frozen P31 source cache"),
                (SOURCE_BAGS, "frozen A0 raw Host cache"), (DIRECT_PREDICTIONS, "existing direct-C43 source predictions")]
    missing = [str(path) for path, _ in required if not path.is_file() or not path.stat().st_size]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing prerequisites " + repr(missing))
    rows, ids, y, fold, member, base, route_score, route, p31, bags = load_source(); summary, counts = host_summary(bags)
    checks = {"rows==799": len(rows) == 799, "class_counts==400_399": (int((y==0).sum()), int((y==1).sum())) == (400,399),
              "fold_sizes_fixed": [int((fold==f).sum()) for f in FOLDS] == [160,160,160,160,159],
              "SVM_errors==97": int(((base>=.5)!=y).sum()) == 97, "route==200": int(route.sum()) == 200,
              "route_captured_errors==84": int(np.sum(route & ((base>=.5)!=y))) == 84,
              "P31_shape": p31.shape == (799,31), "Host_summary_shape": summary.shape == (799,507),
              "all_Host_counts_nonnegative": bool(np.all(counts >= 0))}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen source contract changed " + repr(checks))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = [file_record(path, purpose) for path, purpose in required]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": before})
    previous = {
        "care-p31-lr-pairwise-diagnostics-v1": "P31 feature interactions; not APK-to-APK opposite-label matching",
        "care-fp-count-matched-residual-v1": "count-matched FP/TP audit on one prediction side using SCS; not unified C43 ranking",
        "care-class-conditional-role-direction-v1": "class prototypes; explicitly excluded from this experiment",
        "care-unified-routed-residual-v1": "fixed residual formula; not pairwise conditional ranking",
        "care-svm-c43-remove-bounded-correction-v1": "direct-C43 baseline reused; objective is ordinary APK classification"
    }
    found = {name: {"distinction": description, "output_exists": (PROJECT / "outputs" / name).exists()}
             for name, description in previous.items()}
    dump_json(OUTPUT / "existing_experiment_audit.json", {"status": "PARTIAL_OVERLAP_NOT_EQUIVALENT", "experiments": found})
    protocol = {"name": "CARE-Droid C43 Conditional Matched Ranking v1", "primary_domain": "2017 fixed five-fold source OOF",
        "stage1": "calibrated P31-SVM frozen OOF", "router": "SVM-anchored MeanOpposition Top25% frozen",
        "representation": "label-free APK summary of frozen A0 raw Host bags: mean, q90 and max per stored dimension",
        "compression": "outer-train-only StandardScaler plus label-free PCA(24)",
        "matching": "outer-train only; deterministic global-nearest greedy opposite-label matching without replacement, separately within each Stage-I prediction side",
        "matching_distance": "outer-train-standardized [sqrt(P31), SVM logit, MeanOpposition] Euclidean distance",
        "objective": "one unified linear malware ranking score; pairwise logistic on both d=(malware-benign) and -d",
        "training_scope": "all outer-train APKs, never routed-only", "evaluation_scope": "frozen routed validation APKs",
        "arms": ["C43_MATCHED_REAL", "C43_RANDOM_PAIR", "C43_UNMATCHED", "P31_MATCHED", "COUNT_ONLY_MATCHED", "EXISTING_DIRECT_C43"],
        "null_repetitions": 100, "future_rows_loaded": 0, "end_to_end_correction": False,
        "stop_rule": "no future or integration unless both directions beat best null by 0.03, reach AUC 0.60, and are consistent in >=3/5 evaluable folds"}
    dump_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "protocol_hash.txt").write_text(hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()+"\n", encoding="utf-8")
    dump_json(OUTPUT / "preflight.json", {"status":"PASS", "checks":checks, "future_rows_loaded":0,
        "host_count_min":int(counts.min()), "host_count_median":float(np.median(counts)), "host_count_max":int(counts.max())})
    print("C43_CONDITIONAL_MATCHED_PREFLIGHT_PASS rows=799 routed=200 captured=84 future=0", flush=True)


if __name__ == "__main__": main()
