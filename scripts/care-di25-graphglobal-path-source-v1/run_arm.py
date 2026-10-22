#!/usr/bin/env python3
"""Run one strict-2017 frozen-DI25 GraphGlobal-path ablation arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (
    HERE,
    PROJECT / "scripts",
    PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_contract import MAX_DELTA, build_di_reference
from graphglobal_path_models import ARMS, make_graphglobal_path_class
from source_label_contract import install_source_label_contract

import run_care_rolegraph_conditioned_dualstream_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/care-di25-graphglobal-path-source-v1"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if MAX_DELTA != 2.25:
        raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta changed")

    root = args.output
    arm_dir = root / "arms" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    contract = build_di_reference(PROJECT, root)
    install_source_label_contract(v4)

    method = "DI25__Host__Anomaly__GraphGlobalPath__" + args.arm + "__Bounded"
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
    v4.RoleGraphConditionedEvidence = make_graphglobal_path_class(v4, args.arm)
    sys.argv = [
        sys.argv[0],
        "--previous-result", str(root / "preflight/di25_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-source-local-assets-v2/regions_2017.npz",
        "--variants", "RGCondAttn",
        "--seeds", args.seeds,
        "--max-delta", str(MAX_DELTA),
        "--device", args.device,
        "--output", str(arm_dir),
    ]
    v4.main()

    architecture_path = arm_dir / "architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    q, d = ARMS[args.arm]
    architecture.update({
        "experiment": "CARE-Droid DI25 GraphGlobal path 2x2 source ablation v1",
        "arm": args.arm,
        "method": method,
        "router": "frozen DI25",
        "route_DI": contract["route_DI"],
        "source_only": True,
        "future_rows_loaded": 0,
        "local_representation": "Host-only; Ego is not consumed",
        "role_graph_encoder_inputs": ["role_node", "role_edge"],
        "graphglobal_encoder_separate": True,
        "graphglobal_to_attention": bool(q),
        "graphglobal_to_delta_direct": bool(d),
        "correction_input_width": 113,
        "bounded_formula": "sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))",
        "seeds": [int(x) for x in args.seeds.split(",")],
        "claim_boundary": (
            "Q-only means Stage-II GraphGlobal directly conditions Host attention "
            "but has no direct post-selection path to delta. Global24 and the "
            "Stage-I logit remain correction inputs in every arm."
        ),
    })
    architecture_path.write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("GRAPHGLOBAL_PATH_ARM_DONE", args.arm, flush=True)


if __name__ == "__main__":
    main()

