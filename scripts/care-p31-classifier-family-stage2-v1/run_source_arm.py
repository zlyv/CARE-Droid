#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from common import ENDPOINT_SOURCE, NEW_ARMS, OUTPUT, PROJECT, SCRIPTS, SOURCE_A0_CACHE, method_name
from model_adapter import install_a0, prepare_imports
from nested_stage1 import NestedStage1


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=NEW_ARMS, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(); output = OUTPUT / "source" / args.arm
    if (output / "source_predictions.csv").is_file() and (output / "architecture.json").is_file():
        print(f"P31_FAMILY_STAGE2_SOURCE_RESUME arm={args.arm}", flush=True); return
    prepare_imports()
    stage1_script = SCRIPTS / "care-p31-stage1-rf-selection-source-v1"
    if str(stage1_script) not in sys.path:
        sys.path.insert(0, str(stage1_script))
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    method = method_name(args.arm); install_source_label_contract(v4); install_a0(v4, method)
    nested = NestedStage1(args.arm, OUTPUT / "audits" / f"source_{args.arm}_nested.json", args.jobs)
    v4.inner_oof_global24 = nested.inner_oof
    sys.argv = [sys.argv[0], "--previous-result", str(OUTPUT / "references" / args.arm),
        "--cache", str(ENDPOINT_SOURCE), "--region-cache", str(SOURCE_A0_CACHE),
        "--variants", "RGCondAttn", "--seeds", "0,1,2", "--max-delta", "2.25",
        "--device", args.device, "--output", str(output)]
    v4.main()
    if len(nested.calls) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested Stage-I calls")
    path = output / "architecture.json"; value = json.loads(path.read_text(encoding="utf-8"))
    value.update({"arm": args.arm, "stage1": f"P31 calibrated {args.arm}",
        "router": f"{args.arm}-anchored MeanOpposition Top25%", "stage2": "A0 Role C43-RQ",
        "max_delta": 2.25, "future_rows_loaded": 0, "formal_RF_unchanged": True})
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P31_FAMILY_STAGE2_SOURCE_ARM_DONE arm={args.arm}", flush=True)


if __name__ == "__main__":
    main()
