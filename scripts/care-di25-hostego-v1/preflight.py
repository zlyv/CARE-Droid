#!/usr/bin/env python3
"""One-shot exhaustive preflight for the source-only DI25 Host/Ego experiment.

Historical result CSVs are comparison artefacts, not model inputs. They are
therefore optional. Everything used by M0--M3 is checked here before training.
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path.cwd()
EXPECTED_N = 799
EXPECTED_ROUTED = 200
EXPECTED_FOLDS = [159, 160, 160, 160, 160]

required = [
    "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
    "scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py",
    "scripts/care-malben-contrast-v2/run_care_malben_role_source_v2.py",
    "scripts/care-malben-host-v2/shared_role_prototypes.py",
    "scripts/care-malben-host-v2/source_label_contract.py",
    "artifacts/care-source-local-assets-v2/source_asset_gate.txt",
    "artifacts/care-source-local-assets-v2/host_regions_2017.npz",
    "artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz",
    "artifacts/care-source-local-assets-v2/regions_2017.npz",
    "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
    "artifacts/p1-features/features.csv",
    "artifacts/p1-features/feature_schema.json",
    "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv",
    "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz",
]
di_candidates = [
    "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv",
    "artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv",
    "artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
    "artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv",
]

errors: list[str] = []
missing = [value for value in required if not Path(value).is_file()]
errors.extend("missing file: " + value for value in missing)
for module in ("numpy", "torch", "sklearn", "yaml"):
    if importlib.util.find_spec(module) is None:
        errors.append("missing Python module: " + module)

gate_path = Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt")
if gate_path.is_file():
    gate = gate_path.read_text(encoding="utf-8").lower()
    if "passed: true" not in gate:
        errors.append("source local-asset gate is not PASS")


def normal(value: object) -> str:
    return str(value).strip().lower()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# Select the first valid frozen DI table, rather than the first existing file.
valid_di: Path | None = None
di_diagnostics: list[str] = []
for candidate_text in di_candidates:
    candidate = Path(candidate_text)
    if not candidate.is_file():
        di_diagnostics.append(candidate_text + ": absent")
        continue
    try:
        rows = read_rows(candidate)
        fields = set(rows[0]) if rows else set()
        id_key = "sha256" if "sha256" in fields else "sample_id" if "sample_id" in fields else None
        needed = {"label", "fold", "p_stage1", "route_DI"}
        absent = sorted(needed - fields)
        ids = [normal(row.get(id_key, "")) for row in rows] if id_key else []
        routed = sum(int(float(row["route_DI"])) for row in rows) if not absent else -1
        folds = sorted(Counter(int(float(row["fold"])) for row in rows).values()) if "fold" in fields else []
        reasons = []
        if len(rows) != EXPECTED_N:
            reasons.append(f"rows={len(rows)}")
        if id_key is None:
            reasons.append("missing sha256/sample_id")
        elif len(set(ids)) != EXPECTED_N or "" in ids:
            reasons.append(f"unique_ids={len(set(ids))}")
        if absent:
            reasons.append("missing columns=" + repr(absent))
        if routed != EXPECTED_ROUTED:
            reasons.append(f"routed={routed}")
        if folds != EXPECTED_FOLDS:
            reasons.append(f"fold_sizes={folds}")
        if reasons:
            di_diagnostics.append(candidate_text + ": " + ", ".join(reasons))
        else:
            valid_di = candidate
            break
    except Exception as exc:
        di_diagnostics.append(candidate_text + f": unreadable ({type(exc).__name__}: {exc})")
if valid_di is None:
    errors.append("no valid frozen source DI25 table:\n    " + "\n    ".join(di_diagnostics))


# Validate the immutable local assets consumed by the Host/Ego loader.
if importlib.util.find_spec("numpy") is not None:
    try:
        import numpy as np

        asset_root = Path("artifacts/care-source-local-assets-v2")
        host_path = asset_root / "host_regions_2017.npz"
        ego_path = asset_root / "ego_1hop_regions_2017.npz"
        if host_path.is_file() and ego_path.is_file():
            with np.load(host_path, allow_pickle=False) as host, np.load(ego_path, allow_pickle=False) as ego:
                for name, archive in (("host", host), ("ego", ego)):
                    if "sha256" not in archive:
                        errors.append(f"{name} cache lacks sha256")
                    elif len(archive["sha256"]) != EXPECTED_N:
                        errors.append(f"{name} cache rows={len(archive['sha256'])}, expected 799")
                if "sha256" in host and "sha256" in ego:
                    host_ids = [normal(value) for value in host["sha256"]]
                    ego_ids = [normal(value) for value in ego["sha256"]]
                    if host_ids != ego_ids:
                        errors.append("Host and Ego cache SHA order differs")
                    if len(set(host_ids)) != EXPECTED_N:
                        errors.append(f"Host cache unique SHA count={len(set(host_ids))}, expected 799")
                for key in ("ego_status", "incoming_host_positions", "outgoing_host_positions"):
                    if key not in ego:
                        errors.append("Ego cache lacks " + key)
                if "ego_status" in ego and int(np.sum(np.asarray(ego["ego_status"]) == 1)) <= 0:
                    errors.append("Ego cache contains no nontrivial one-hop neighborhood")
    except Exception as exc:
        errors.append(f"local asset schema audit failed ({type(exc).__name__}: {exc})")


# Import the exact dependency chain now, before self-test/training.
if not missing:
    for path in (
        PROJECT / "scripts",
        PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
        PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
        PROJECT / "scripts/care-malben-contrast-v2",
        PROJECT / "scripts/care-malben-host-v2",
    ):
        sys.path.insert(0, str(path))
    for module in (
        "run_care_rolegraph_conditioned_dualstream_v4",
        "run_care_rgcondattn_dedup_bounded_source_v1",
        "run_care_malben_role_source_v2",
        "shared_role_prototypes",
        "source_label_contract",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"import {module} failed ({type(exc).__name__}: {exc})")

if errors:
    raise SystemExit(
        "BLOCKED_BY_CONTRACT: exhaustive preflight found all unresolved requirements:\n- "
        + "\n- ".join(errors)
    )

print("HOSTEGO_PREFLIGHT_PASS")
print(f"[FROZEN] DI25 source: {valid_di}; rows=799; routed=200; folds={EXPECTED_FOLDS}")
optional = {
    "historical_anomaly_predictions": [
        "artifacts/care-malben-host-v2/source/anomaly/source_predictions.csv",
        "artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv",
    ],
    "historical_malben_predictions": [
        "artifacts/care-malben-host-v2/source/malben/source_predictions.csv",
        "artifacts/care-malben-host-v2/source/model/source_predictions.csv",
    ],
}
for label, values in optional.items():
    found = next((value for value in values if Path(value).is_file()), None)
    print(f"[OPTIONAL] {label}: {found or 'not available; historical probability comparison will be recorded as unavailable'}")
