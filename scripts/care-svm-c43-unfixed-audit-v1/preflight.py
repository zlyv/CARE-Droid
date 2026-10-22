#!/usr/bin/env python3
import numpy as np

from common import OUTPUT, PROJECT, SOURCE, asset_record, dump_json, load_source, metrics


def main():
    if not OUTPUT.is_relative_to((PROJECT / "outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    if not SOURCE.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: source predictions missing")
    OUTPUT.mkdir(parents=True, exist_ok=True); rows, ids, y, fold, base, final, route = load_source()
    stage1 = metrics(y, base, base, route); c43 = metrics(y, base, final, route)
    routed_errors = int(np.sum(route & ((base >= .5) != y)))
    checks = {"rows==799": len(rows) == 799, "folds_fixed": [int(np.sum(fold == f)) for f in range(1, 6)] == [160,160,160,160,159],
        "route==200": int(route.sum()) == 200, "stage1_errors==97": stage1["errors"] == 97,
        "routed_errors==84": routed_errors == 84,
        "C43_fixed_broken_net": (c43["fixed"], c43["broken"], c43["net"]) == (29,4,25)}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: baseline changed " + repr(checks))
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", [asset_record(SOURCE, "frozen SVM/C43 source OOF predictions")])
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "checks": checks, "stage1": stage1, "formal_c43": c43,
        "future_rows_loaded": 0})
    dump_json(OUTPUT / "protocol.json", {"status": "FROZEN_DIAGNOSTIC", "near_zero_abs_delta": .05, "max_delta": 2.25,
        "scale_sweep": [0,.25,.5,.75,1,1.25,1.5,2,3,4], "scale_sweep_is_selection": False,
        "future_rows_loaded": 0, "interpretation": "labels are used only for retrospective failure anatomy"})
    print(f"SVM_C43_UNFIXED_PREFLIGHT_PASS routed_errors={routed_errors} routed_correct={int(route.sum())-routed_errors}", flush=True)


if __name__ == "__main__": main()

