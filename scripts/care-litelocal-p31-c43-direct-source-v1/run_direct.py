#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SCRIPTS = PROJECT / "scripts"
P31 = SCRIPTS / "care-litelocal-p31-c43-source-v1"
COMPACT = SCRIPTS / "care-di25-compact43-relationquery-source-v1"
GRAPHGLOBAL = SCRIPTS / "care-di25-graphglobal-path-source-v1"
for path in (
    HERE, P31, COMPACT, GRAPHGLOBAL, SCRIPTS,
    SCRIPTS / "care-rolegraph-conditioned-v4-script",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_background import make_fit
from compact_models import make_compact_class
from direct_model import make_direct_class
from nested_p31 import NestedP31
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
METHOD = "P31Mean__DI25__C43RelationQuery__Direct"


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
    compact_class = make_compact_class(v4, "C43_RELATION_QUERY")
    v4.RoleGraphConditionedEvidence = make_direct_class(compact_class)
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)
    nested = NestedP31(jobs=args.jobs)
    v4.inner_oof_global24 = nested.inner_oof
    output = ROOT / "arms/P31_C43_DIRECT"
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--previous-result", str(P31_ROOT / "preflight/p31_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
        "--variants", "RGCondAttn", "--seeds", args.seeds,
        # max-delta remains in the base object solely for capacity/contract
        # compatibility; DirectC43 bypasses the bounded residual formula.
        "--max-delta", "2.25",
        "--keep-weight", "1.0", "--fix-weight", "1.0",
        "--keep-consistency", "0.0",
        "--device", args.device, "--output", str(output),
    ]
    v4.main()
    if len(nested.calls) != 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected five nested P31 calls")
    architecture_path = output / "architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    architecture.update({
        "experiment": "strict-2017 P31-DI25 C43-RQ direct classification",
        "stage1": "P31 Mean Committee",
        "router": "P31 MeanOpposition fold-wise top 25%",
        "stage2_representation": "C43_RELATION_QUERY unchanged",
        "stage2_output": "direct raw classification logit",
        "bounded_residual_formula_used": False,
        "base_logit_added_to_output": False,
        "training_loss": "unweighted binary cross entropy",
        "fixkeep_weighting_used": False,
        "keep_consistency_used": False,
        "stage2_training_scope": "all outer-train APKs",
        "outer_train_stage1": "nested inner-OOF P31 probabilities",
        "global24_role": "routed Stage-II context only",
        "relation_enters_attention": True,
        "relation_enters_output_head": False,
        "future_rows_loaded": 0,
        "ego_consumed": False,
    })
    architecture_path.write_text(json.dumps(architecture, indent=2) + "\n",
                                 encoding="utf-8")
    print("P31_C43_DIRECT_SOURCE_ARM_DONE")


if __name__ == "__main__":
    main()
