#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, BAG_CACHE, EPOCHS, NULL_REPETITIONS, OUTPUT, PROJECT,
                    REAL_SEEDS, SOURCE_STAGE2, asset_record, correction_metrics,
                    dump_json, load_source)


def main():
    output_root = (PROJECT / "outputs").resolve()
    if not OUTPUT.is_relative_to(output_root):
        raise RuntimeError("BLOCKED_BY_CONTRACT: output must stay below project outputs")
    missing = [str(path) for path in (SOURCE_STAGE2, BAG_CACHE) if not path.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing formal assets " + repr(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, ids, y, fold, base, c43, route, bags = load_source()
    stage1 = correction_metrics(y, base, base, route); formal = correction_metrics(y, base, c43, route)
    checks = {"rows==799": len(rows) == 799, "fold_sizes_fixed": [int(np.sum(fold == value)) for value in range(1, 6)] == [160, 160, 160, 160, 159],
        "route==200": int(route.sum()) == 200, "stage1_BA": abs(stage1["BA"] - .8785839598997494) < 1e-12,
        "formal_BA": abs(formal["BA"] - .9098527568922306) < 1e-12,
        "formal_fixed_broken_net": (formal["fixed"], formal["broken"], formal["net"]) == (29, 4, 25),
        "C43_width==43": all(bag.shape[1] == 43 for bag in bags)}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen SVM/C43 baseline changed " + repr(checks))
    manifest = [asset_record(SOURCE_STAGE2, "frozen SVM Stage-I and C43 OOF predictions"),
                asset_record(BAG_CACHE, "frozen A0 Host cache; first 43 columns are C43")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", manifest)
    protocol = {"status": "FROZEN", "source_rows": 799, "supervision_units": 799,
        "host_instances": int(sum(len(bag) for bag in bags)), "folds": [160, 160, 160, 160, 159],
        "stage1": "calibrated P31-SVM", "router": "SVM-anchored MeanOpposition Top25%", "routed": 200,
        "input": "per-Host C43 only", "arms": list(ARMS), "real_seeds": list(REAL_SEEDS),
        "epochs": EPOCHS, "null_repetitions": NULL_REPETITIONS, "hidden_dim": 16,
        "clam_instance_weight": .25, "max_delta": 2.25, "future_rows_loaded": 0,
        "selection_rule": "direction gate is frozen before correction integration"}
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "checks": checks,
        "stage1": stage1, "formal_c43": formal, "host_instances": protocol["host_instances"], "future_rows_loaded": 0})
    print("SIGNED_HOST_MIL_PREFLIGHT_PASS", flush=True)
    print({"stage1_BA": stage1["BA"], "formal_BA": formal["BA"], "fixed": 29, "broken": 4,
           "net": 25, "bags": 799, "hosts": protocol["host_instances"]}, flush=True)


if __name__ == "__main__": main()

