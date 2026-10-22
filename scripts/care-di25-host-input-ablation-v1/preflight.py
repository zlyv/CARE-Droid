#!/usr/bin/env python3
"""Exhaustive preflight before any correction-input ablation training."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
EXPECTED_N, EXPECTED_ROUTED = 799, 200
EXPECTED_FOLDS = [159, 160, 160, 160, 160]

required = [
    "scripts/care-di25-host-input-ablation-v1/masked_host_model.py",
    "scripts/care-di25-host-input-ablation-v1/run_arm.py",
    "scripts/care-di25-host-input-ablation-v1/experiment_contract.py",
    "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
    "scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py",
    "scripts/care-di25-host-input-ablation-v1/source_label_contract.py",
    "artifacts/care-source-local-assets-v2/source_asset_gate.txt",
    "artifacts/care-source-local-assets-v2/regions_2017.npz",
    "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
    "artifacts/p1-features/features.csv",
    "artifacts/p1-features/feature_schema.json",
    "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv",
    "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz",
]
route_candidates = [
    "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv",
    "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv",
    "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv",
    "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
    "artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv",
]
errors = ["missing file: " + value for value in required if not Path(value).is_file()]
for name in ("numpy", "torch", "sklearn", "yaml"):
    if importlib.util.find_spec(name) is None:
        errors.append("missing Python module: " + name)

gate = Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt")
if gate.is_file() and "passed: true" not in gate.read_text(encoding="utf-8").lower():
    errors.append("source local-asset hard gate is not PASS")

valid_route = None
diagnostics = []
for value in route_candidates:
    path = Path(value)
    if not path.is_file():
        diagnostics.append(value + ": absent")
        continue
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fields = set(rows[0]) if rows else set()
        id_key = "sha256" if "sha256" in fields else "sample_id" if "sample_id" in fields else None
        ids = [str(row.get(id_key, "")).strip().lower() for row in rows] if id_key else []
        routed = sum(int(float(row["route_DI"])) for row in rows) if "route_DI" in fields else -1
        folds = sorted(Counter(int(float(row["fold"])) for row in rows).values()) if "fold" in fields else []
        reasons = []
        if len(rows) != EXPECTED_N: reasons.append(f"rows={len(rows)}")
        if len(set(ids)) != EXPECTED_N or "" in ids: reasons.append(f"unique_ids={len(set(ids))}")
        if not {"label", "fold", "p_stage1", "route_DI"}.issubset(fields): reasons.append("missing contract columns")
        if routed != EXPECTED_ROUTED: reasons.append(f"routed={routed}")
        if folds != EXPECTED_FOLDS: reasons.append(f"fold_sizes={folds}")
        if reasons: diagnostics.append(value + ": " + ", ".join(reasons))
        else:
            valid_route = path
            break
    except Exception as exc:
        diagnostics.append(f"{value}: {type(exc).__name__}: {exc}")
if valid_route is None:
    errors.append("no exact frozen DI25 route:\n    " + "\n    ".join(diagnostics))

if not errors:
    for path in (
        ROOT / "scripts/care-di25-host-input-ablation-v1",
        ROOT / "scripts/care-rolegraph-conditioned-v4-script",
        ROOT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
        ROOT / "scripts/care-malben-host-v2",
        ROOT / "scripts",
    ):
        sys.path.insert(0, str(path))
    for name in (
        "masked_host_model", "experiment_contract", "source_label_contract",
        "run_care_rolegraph_conditioned_dualstream_v4",
        "run_care_rgcondattn_dedup_bounded_source_v1",
    ):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name} failed: {type(exc).__name__}: {exc}")

if errors:
    raise SystemExit("BLOCKED_BY_CONTRACT: preflight found all unresolved inputs:\n- " + "\n- ".join(errors))
print("HOST_INPUT_ABLATION_PREFLIGHT_PASS")
print(f"Frozen DI25 source: {valid_route}; n=799; routed=200")
