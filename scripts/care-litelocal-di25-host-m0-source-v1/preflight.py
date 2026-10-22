#!/usr/bin/env python3
from pathlib import Path
import csv
import importlib.util
import inspect
import sys
import numpy as np

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
FEATURES = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
ROUTER = PROJECT / "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"
UPSTREAM = PROJECT / "scripts/care-litelocal-semantic-source-v1/run_source_oof.py"
HOST_ARM = PROJECT / "scripts/care-di25-host-input-replay-v1/run_arm.py"
HOST_DIR = HOST_ARM.parent
MASKED = HOST_DIR / "masked_host_model.py"
REQUIRED = [FEATURES, ROUTER, UPSTREAM, HOST_ARM, MASKED]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- " + "\n- ".join(missing))
    with ROUTER.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    folds = [int(float(r["fold"])) for r in rows]
    labels = [int(float(r["label"])) for r in rows]
    if len(rows) != 799 or len({str(r["sha256"]).strip().lower() for r in rows}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort changed")
    if [folds.count(i) for i in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source folds changed")
    if labels.count(0) != 400 or labels.count(1) != 399:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source labels changed")
    with np.load(FEATURES, allow_pickle=False) as z:
        if z["role_prevalence"].shape != (799, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 shape changed")
        if z["within_method_cooccurrence"].shape != (799, 465):
            raise RuntimeError("BLOCKED_BY_CONTRACT: cooccurrence shape changed")
        if z["directed_adjacent_transitions"].shape != (799, 961):
            raise RuntimeError("BLOCKED_BY_CONTRACT: transition shape changed")

    upstream = load_module("_litelocal_preflight", UPSTREAM)
    if not hasattr(upstream, "committee_nested"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: LiteLocal committee_nested missing")
    params = list(inspect.signature(upstream.committee_nested).parameters)
    needed = ["arm", "prevalence", "cooc", "transitions", "y", "tr", "va", "seed"]
    if params[:8] != needed:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: committee_nested signature changed: {params}")

    if str(HOST_DIR) not in sys.path:
        sys.path.insert(0, str(HOST_DIR))
    host = load_module("_host_arm_preflight", HOST_ARM)
    if not hasattr(host, "build_di_reference"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: run_arm.build_di_reference missing")
    if not hasattr(host, "v4") or not hasattr(host.v4, "fit_predict_variant"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: existing v4.fit_predict_variant missing")
    fit_params = list(inspect.signature(host.v4.fit_predict_variant).parameters)
    expected_prefix = [
        "global_train", "global_test", "s7_train", "s7_test",
        "raw_train_bags", "raw_test_bags", "base_train", "base_test", "labels",
    ]
    if fit_params[:9] != expected_prefix:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: fit_predict_variant signature changed: {fit_params}")

    print("LITELOCAL_DI25_HOST_M0_PREFLIGHT_PASS")
    print("source=799 benign=400 malware=399 folds=160,160,160,160,159")
    print("stage1=FUSED63(P31+COOC16+TRANS16) stage2=existing Host-M0 FULL bounded")
    print("committee_nested_contract=PASS host_fit_predict_contract=PASS")
    print("future_rows_loaded=0")


if __name__ == "__main__":
    main()
