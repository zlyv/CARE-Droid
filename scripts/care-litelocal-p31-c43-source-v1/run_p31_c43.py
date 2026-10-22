#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SCRIPTS = PROJECT / "scripts"
COMPACT = SCRIPTS / "care-di25-compact43-relationquery-source-v1"
GRAPHGLOBAL = SCRIPTS / "care-di25-graphglobal-path-source-v1"
for path in (
    HERE, COMPACT, GRAPHGLOBAL, SCRIPTS,
    SCRIPTS / "care-rolegraph-conditioned-v4-script",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_background import make_fit
from compact_models import make_compact_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

from nested_p31 import NestedP31

ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
METHOD = "P31Mean__DI25__C43RelationQuery__Bounded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    install_source_label_contract(v4)
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.AUGMENTED_REGION_DIM = 78
    v4.RoleGraphConditionedEvidence = make_compact_class(v4, "C43_RELATION_QUERY")
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)
    nested = NestedP31(jobs=args.jobs)
    v4.inner_oof_global24 = nested.inner_oof
    output = ROOT / "arms/P31_C43_RQ"
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--previous-result", str(ROOT / "preflight/p31_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
        "--variants", "RGCondAttn", "--seeds", args.seeds,
        "--max-delta", "2.25", "--device", args.device,
        "--output", str(output),
    ]
    v4.main()
    if len(nested.calls) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested P31 calls")
    architecture_path = output / "architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    architecture.update({
        "experiment": "strict-2017 P31 Mean Committee -> P31-DI25 -> C43-RQ",
        "stage1_representation": "P31 only; FUSED63 is prohibited",
        "stage1_probability": "calibrated LR/RF/XGB/SVM mean",
        "outer_train_stage1": "nested inner-OOF P31 probabilities",
        "outer_validation_stage1": "locked strict outer-OOF P31 probabilities",
        "router": "P31 MeanOpposition fold-wise top 25%",
        "stage2": "C43_RELATION_QUERY",
        "stage2_training_scope": "all outer-train APKs",
        "global24_role": "routed Stage-II context only; not Stage-I and not Router",
        "relation_enters_attention": True,
        "relation_enters_delta": False,
        "max_delta_logit": 2.25,
        "future_rows_loaded": 0,
        "ego_consumed": False,
    })
    architecture_path.write_text(json.dumps(architecture, indent=2) + "\n",
                                 encoding="utf-8")
    print("P31_C43_SOURCE_ARM_DONE")


if __name__ == "__main__":
    main()
