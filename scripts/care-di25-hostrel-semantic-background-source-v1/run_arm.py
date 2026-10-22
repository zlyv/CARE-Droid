#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BASE = PROJECT / "scripts/care-di25-graphglobal-path-source-v1"
for p in (HERE, BASE, PROJECT/"scripts", PROJECT/"scripts/care-rolegraph-conditioned-v4-script", PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1", PROJECT/"scripts/care-malben-host-v2"):
    sys.path.insert(0, str(p))
from semantic_background import make_fit
from graphglobal_path_models import make_graphglobal_path_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

ROOT = Path("artifacts/care-di25-hostrel-semantic-background-source-v1")
ARM = "R1_HOSTREL74_HARD_ROLE"
METHOD = "DI25__HostRel74__Anomaly__QQuery__HardRoleBackground__Bounded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default=ARM, choices=(ARM,))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    install_source_label_contract(v4)
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.RoleGraphConditionedEvidence = make_graphglobal_path_class(v4, "Q_QUERY_ONLY")
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3, "B1_HARD_ROLE")
    out = ROOT / "arms" / ARM
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--previous-result", str(ROOT/"preflight/di25_reference"), "--cache", "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz", "--region-cache", str(ROOT/"cache/regions_2017_hostrel_role_sidecar.npz"), "--variants", "RGCondAttn", "--seeds", args.seeds, "--max-delta", "2.25", "--device", args.device, "--output", str(out)]
    v4.main()
    with (out/"fit_audit.csv").open("r", encoding="utf-8-sig", newline="") as f:
        fit = list(csv.DictReader(f))
    if {int(float(r["parameter_n"])) for r in fit} != {33036}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: parameter count changed")
    if {int(float(r["shared_prototype_n"])) for r in fit} != {32} or {int(float(r["benign_prototype_n"])) for r in fit} != {24}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: prototype capacity changed")
    contract = {
        "arm": ARM,
        "method": METHOD,
        "source_only": True,
        "host_representation": "HostRel74 = SelfRole31 + DirectedNeighborRole31 + frozen tail12",
        "directed_neighbor_role_formula": "tanh(log1p(outgoing normalized-neighbor role mass)-log1p(incoming normalized-neighbor role mass))",
        "background_representation_matches_test_host": True,
        "semantic_background_mode": "hard dominant endpoint role",
        "role_sidecar_enters_neural_model": False,
        "future_rows_loaded": 0,
        "ego_tensor_consumed": False,
        "ego_encoder_instantiated": False,
        "model_host_width": 74,
        "parameter_count": 33036,
        "shared_prototypes": 32,
        "benign_prototypes": 24,
        "bounded_formula": "sigmoid(logit(p_stage1)+2.25*tanh(raw_delta))",
    }
    (out/"contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("HOSTREL_SEMANTIC_BACKGROUND_ARM_DONE", ARM, flush=True)


if __name__ == "__main__":
    main()
