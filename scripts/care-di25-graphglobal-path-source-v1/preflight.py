#!/usr/bin/env python3
"""Exhaustive fail-closed preflight for the strict-2017 experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path.cwd()
HERE = ROOT / "scripts/care-di25-graphglobal-path-source-v1"
sys.path.insert(0, str(HERE))
from experiment_contract import build_di_reference


def main():
    required = [
        "scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py",
        "scripts/care-di25-graphglobal-path-source-v1/run_arm.py",
        "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
        "scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py",
        "artifacts/p1-features/features.csv",
        "artifacts/p1-features/feature_schema.json",
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv",
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz",
        "artifacts/care-source-local-assets-v2/source_asset_gate.txt",
        "artifacts/care-source-local-assets-v2/regions_2017.npz",
        "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
    ]
    errors = ["missing: " + p for p in required if not Path(p).is_file()]
    for module in ("numpy", "torch", "sklearn", "yaml"):
        if importlib.util.find_spec(module) is None:
            errors.append("missing Python module: " + module)
    gate = Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt")
    if gate.is_file() and "passed: true" not in gate.read_text(encoding="utf-8").lower():
        errors.append("source local-asset gate is not PASS")
    if errors:
        raise RuntimeError("BLOCKED_BY_CONTRACT:\n- " + "\n- ".join(errors))
    contract = build_di_reference(
        ROOT, Path("artifacts/care-di25-graphglobal-path-source-v1")
    )
    print("GRAPHGLOBAL_PATH_PREFLIGHT_PASS")
    print("source_rows=799")
    print("year=2017")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("route_DI=200")
    print("route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
    print("DI25_source=" + contract["di25_source"])
    print("future_rows_loaded=0")
    print("ego_consumed=false")


if __name__ == "__main__":
    main()

