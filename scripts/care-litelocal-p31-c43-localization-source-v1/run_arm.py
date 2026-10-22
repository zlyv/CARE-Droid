#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SCRIPTS = PROJECT / "scripts"
P31 = SCRIPTS / "care-litelocal-p31-c43-source-v1"
COMPACT = SCRIPTS / "care-di25-compact43-relationquery-source-v1"
GRAPHGLOBAL = SCRIPTS / "care-di25-graphglobal-path-source-v1"
SOURCE_VENDOR = SCRIPTS / "care-stage2only-anomaly-v1/vendor"
for path in (
    HERE, P31, COMPACT, GRAPHGLOBAL, SOURCE_VENDOR, SCRIPTS,
    SCRIPTS / "care-rolegraph-conditioned-v4-script",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_background import make_fit
from localization_models import ARM_SPECS, make_localization_class
from nested_p31 import NestedP31
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT = Path("artifacts/care-litelocal-p31-c43-localization-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(ARM_SPECS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    method = f"P31Mean__DI25__C43Localization__{args.arm}__Bounded"
    install_source_label_contract(v4)
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
    v4.AUGMENTED_REGION_DIM = 78
    v4.RoleGraphConditionedEvidence = make_localization_class(v4, args.arm)
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)
    nested = NestedP31(jobs=args.jobs)
    v4.inner_oof_global24 = nested.inner_oof
    output = ROOT / "arms" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--previous-result", str(P31_ROOT / "preflight/p31_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
        "--variants", "RGCondAttn", "--seeds", args.seeds,
        "--max-delta", "2.25", "--device", args.device,
        "--output", str(output),
    ]
    v4.main()
    if len(nested.calls) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested P31 calls")
    with (output / "fit_audit.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        fit = list(csv.DictReader(handle))
    counts = {int(float(row["parameter_n"])) for row in fit}
    if len(fit) != 15 or counts != {32292}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fit/parameter audit failed " + repr(counts))
    architecture_path = output / "architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    architecture.update({
        "experiment": "strict-2017 P31-DI25 C43 localization 2x2",
        "arm": args.arm, "topk3_critical_pool": ARM_SPECS[args.arm]["topk"],
        "relation_local_alignment": ARM_SPECS[args.arm]["aligned"],
        "stage1": "P31 Mean Committee", "router": "P31 MeanOpposition top25%",
        "outer_train_stage1": "nested inner-OOF",
        "stage2_training_scope": "all outer-train APKs",
        "relation": "Relation31 enters Host attention only",
        "correction_summary_width": 8, "correction_input_width": 113,
        "parameter_count": 32292, "max_delta": 2.25,
        "future_rows_loaded": 0, "ego_consumed": False,
    })
    architecture_path.write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    print("P31_C43_LOCALIZATION_ARM_DONE", args.arm, flush=True)


if __name__ == "__main__":
    main()
