#!/usr/bin/env python3
"""Run one strict-2017 DI25 Host-only correction-input ablation arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (
    HERE,
    PROJECT,
    PROJECT / "src",
    PROJECT / "scripts",
    PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from masked_host_model import ARMS, BLOCKS, correction_input_mask, make_masked_host_class
from experiment_contract import MAX_DELTA, build_di_reference
from source_label_contract import install_source_label_contract

import run_care_rolegraph_gated_v4 as v4
import run_care_gatedgraph_source as anomaly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/care-di25-host-input-replay-v1"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if MAX_DELTA != 2.25:
        raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta must remain 2.25")

    root = args.output
    method = f"DI25__Host__Anomaly__{args.arm}__Bounded"
    method_dir = root / "arms" / args.arm
    method_dir.mkdir(parents=True, exist_ok=True)
    build_di_reference(PROJECT, root)
    install_source_label_contract(v4)

    Model = make_masked_host_class(anomaly, args.arm)
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
    v4.RoleGraphConditionedEvidence = Model

    # The route is semantically frozen DI25 although the legacy loader column is
    # named A25__routed. No future row is loaded by this runner.
    sys.argv = [sys.argv[0],
        "--previous-result", str(root / "preflight/di25_reference"),
        "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        "--region-cache", "artifacts/care-source-local-assets-v2/regions_2017.npz",
        "--variants", "RGCondAttn",
        "--seeds", args.seeds,
        "--max-delta", "2.25",
        "--device", args.device,
        "--output", str(method_dir),
    ]
    v4.main()

    fit_audit = method_dir / "fit_audit.csv"
    if not fit_audit.is_file():
        raise RuntimeError("Missing fit_audit.csv after run")
    architecture = json.loads((method_dir / "architecture.json").read_text(encoding="utf-8"))
    architecture.update({
        "experiment": "CARE-Droid DI25 Host-only correction-input block ablation v1",
        "arm": args.arm,
        "method": method,
        "router_semantics": "frozen DI25 top 25%; legacy loader field A25__routed",
        "local_representation": "Host-only; no Ego construction or fusion",
        "correction_input_width": 113,
        "correction_input_blocks": {key: [left, right] for key, (left, right) in BLOCKS.items()},
        "zero_masked_blocks": list(ARMS[args.arm]),
        "active_dimensions": int(torch.sum(correction_input_mask(args.arm)).item()),
        "capacity_matched": True,
        "mask_has_parameters": False,
        "max_delta_logit": 2.25,
        "future_data_loaded": False,
    })
    (method_dir / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"HOST_INPUT_ABLATION_ARM_DONE {args.arm}", flush=True)


if __name__ == "__main__":
    main()
