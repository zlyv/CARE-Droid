#!/usr/bin/env python3
"""Run one strict-2017 frozen-DI25 Q-branch isolation arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEPENDENCY = PROJECT / "scripts/care-di25-graphglobal-path-source-v1"
for path in (HERE, DEPENDENCY, PROJECT / "scripts",
             PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
             PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
             PROJECT / "scripts/care-malben-host-v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_contract import MAX_DELTA, build_di_reference
from qbranch_models import ARM_SPECS, TRAIN_ARMS, correction_mask, make_qbranch_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=TRAIN_ARMS)
    p.add_argument("--output", type=Path, default=Path("artifacts/care-di25-qbranch-source-v1"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if MAX_DELTA != 2.25:
        raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta changed")
    arm_dir = a.output / "arms" / a.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    contract = build_di_reference(PROJECT, a.output)
    install_source_label_contract(v4)
    method = f"DI25__Host__Anomaly__QBranch__{a.arm}__Bounded"
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
    v4.RoleGraphConditionedEvidence = make_qbranch_class(v4, a.arm)
    sys.argv = [sys.argv[0],
        "--previous-result", str(a.output / "preflight/di25_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-source-local-assets-v2/regions_2017.npz",
        "--variants", "RGCondAttn", "--seeds", a.seeds,
        "--max-delta", str(MAX_DELTA), "--device", a.device, "--output", str(arm_dir)]
    v4.main()
    fit_path = arm_dir / "fit_audit.csv"
    import csv
    with fit_path.open("r", encoding="utf-8-sig", newline="") as h:
        fit = list(csv.DictReader(h))
    counts = {int(float(r["parameter_n"])) for r in fit}
    if counts != {33036}:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: {a.arm} parameter counts={counts}")
    architecture = {
        "experiment": "CARE-Droid Q-Query branch isolation source v1", "arm": a.arm,
        "base_path": ARM_SPECS[a.arm][0], "kept_blocks": list(ARM_SPECS[a.arm][1]),
        "active_input_dimensions": int(correction_mask(a.arm).sum().item()),
        "correction_input_width": 113, "parameter_count": 33036,
        "router": "frozen DI25", "route_DI": contract["route_DI"],
        "source_only": True, "future_rows_loaded": 0, "ego_consumed": False,
        "bounded_formula": "sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))",
        "claim_boundary": "Fixed input masks test branch sufficiency, not causal feature importance."
    }
    (arm_dir / "qbranch_architecture.json").write_text(json.dumps(architecture, indent=2)+"\n", encoding="utf-8")
    print("QBRANCH_SOURCE_ARM_DONE", a.arm, flush=True)


if __name__ == "__main__":
    main()
