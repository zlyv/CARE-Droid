#!/usr/bin/env python3
"""Read-only hard gate for the Stage-II Feature Sufficiency Audit.

This program never trains a model. It inventories candidate checkpoints and
accepts only a complete 4 years x 5 folds x 3 seeds frozen M1 replay lineage.
DMD B1/B2 checkpoints are explicitly rejected as substitutes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

OUT = Path("artifacts/care-di25-hostego-dmd-bounded-v1/feature_sufficiency_audit")
YEARS = (2018, 2019, 2020, 2021)
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (900000, 900001, 900002)
EXPECTED = {(year, fold, seed) for year in YEARS for fold in FOLDS for seed in SEEDS}

CANDIDATE_ROOTS = [
    Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/final_replay/checkpoints"),
    Path("artifacts/care-di25-hostego-within-year-v1/checkpoints"),
    Path("artifacts/care-di25-hostego-v1/checkpoints"),
    # Inspected only to document that DMD states are not valid M1 substitutes.
    Path("artifacts/care-di25-hostego-dmd-bounded-v1/checkpoints"),
]


def candidate_roots() -> list[Path]:
    """Find CARE Host/Ego artifacts without treating DMD as original M1."""
    roots = list(CANDIDATE_ROOTS)
    artifact_root = Path("artifacts")
    if artifact_root.is_dir():
        roots.extend(path for path in artifact_root.glob("care-di25-hostego*") if path.is_dir())
    # Preserve order while eliminating aliases/duplicates.
    unique = []
    seen = set()
    for path in roots:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key); unique.append(path)
    return unique


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def infer_identity(path: Path, state: dict) -> tuple[int | None, int | None, int | None]:
    def integer(value):
        try: return int(value)
        except (TypeError, ValueError): return None
    year, fold, seed = integer(state.get("year")), integer(state.get("fold")), integer(state.get("seed"))
    parts = path.parts
    if year is None:
        for value in parts:
            if value in {str(x) for x in YEARS}: year = int(value)
    if fold is None:
        for value in parts:
            if value.startswith("fold_"):
                fold = integer(value.split("_", 1)[1])
    if seed is None and path.stem.startswith("seed_"):
        seed = integer(path.stem.split("_", 1)[1])
    return year, fold, seed


def classify(path: Path, state: dict) -> tuple[str, str]:
    text = " ".join([
        str(path).lower(),
        json.dumps(state.get("architecture", {}), default=str).lower(),
        json.dumps(state.get("architecture_config", {}), default=str).lower(),
    ])
    keys = set(state.get("model_state_dict", {}))
    if "b1_nokeep" in text or "b2_full" in text or any(key.startswith(("action_head.", "magnitude_head.")) for key in keys):
        return "REJECT_DMD_CHECKPOINT", "DMD B1/B2 is not the frozen original M1"
    required = {"model_state_dict", "role_graph_scaler_state", "anomaly_fitting_state"}
    if not required.issubset(state):
        # The later final replay used these exact keys. A checkpoint without
        # them cannot reconstruct frozen train/validation representations.
        return "REJECT_INCOMPLETE_INFERENCE_STATE", "missing " + ",".join(sorted(required-set(state)))
    keys = set(state["model_state_dict"])
    if not any(key.startswith("delta_head.") for key in keys):
        return "REJECT_NOT_ORIGINAL_SCALAR_HEAD", "delta_head parameters absent"
    if any(key.startswith(("action_head.", "magnitude_head.")) for key in keys):
        return "REJECT_DMD_CHECKPOINT", "DMD head parameters present"
    if float(state.get("max_delta", -1)) != 2.25:
        return "REJECT_CONTRACT", "max_delta is not 2.25"
    return "ACCEPT_FROZEN_M1_REPLAY", "complete frozen M1 inference state"


def blocked_outputs(inventory: list[dict], accepted: dict, missing: list[tuple[int,int,int]]) -> None:
    schema = {
        "status": "MISSING_FROZEN_M1_CHECKPOINTS",
        "R0_stage1": {"status": "AVAILABLE_FROM_LOCKED_OOF_CSV", "dimension": 1},
        "R1_host": {"status": "NOT_AVAILABLE_WITHOUT_FROZEN_FORWARD_STATE"},
        "R2_hostego": {"status": "NOT_AVAILABLE_WITHOUT_FROZEN_FORWARD_STATE"},
        "R3_anomaly": {"status": "NOT_AVAILABLE_WITHOUT_FROZEN_ANOMALY_STATE"},
        "R4_fused": {"status": "MISSING", "reason": "original scalar-head input requires frozen M1 checkpoint forward"},
    }
    (OUT/"representation_schema.json").write_text(json.dumps(schema,indent=2)+"\n",encoding="utf-8")
    leakage = {
        "status":"BLOCKED_BEFORE_FEATURE_EXTRACTION",
        "validation_label_used_in_representation":False,
        "validation_label_used_in_scaler":False,
        "validation_label_used_in_probe_training":False,
        "year_used_as_feature":False,
        "future_information_used":False,
        "frozen_model_parameter_change_max_abs":0.0,
    }
    (OUT/"feature_leakage_audit.json").write_text(json.dumps(leakage,indent=2)+"\n",encoding="utf-8")
    no_change = {
        "training_invoked":False,"stage2_representation_retrained":False,
        "model_weights_modified":False,"checkpoint_modified":False,
        "probe_training_invoked":False,"new_CARE_variant_created":False,
        "reason":"required frozen M1 checkpoint lineage is incomplete",
    }
    (OUT/"NO_MODEL_MODIFICATION_AUDIT.json").write_text(json.dumps(no_change,indent=2)+"\n",encoding="utf-8")
    missing_text = "\n".join(f"- year={y} fold={f} seed={s}" for y,f,s in missing)
    report = [
        "CARE-Droid Stage-II Feature Sufficiency Audit", "",
        "STATUS: MISSING / BLOCKED_BY_CONTRACT", "",
        f"Accepted frozen M1 checkpoints: {len(accepted)}/60.",
        f"Missing year-fold-seed states: {len(missing)}.", "",
        "The requested probes were not run because R4_fused cannot be extracted",
        "from locked probabilities or DMD checkpoints. Replaying/retraining M1 is",
        "explicitly prohibited by the current contract.", "",
        "Missing states:", missing_text if missing_text else "- none", "",
        "All requested probe result files are MISSING rather than fabricated.",
    ]
    (OUT/"FEATURE_SUFFICIENCY_REPORT.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    decision = [
        "FEATURE_SUFFICIENCY_DECISION", "",
        "DECISION: NOT_IDENTIFIABLE_FROM_AVAILABLE_ARTIFACTS", "",
        "No conclusion may be drawn between FEATURE_EXTRACTION_INSUFFICIENT and",
        "CORRECTION_MAPPING_INSUFFICIENT without the frozen pre-head representation.",
        "DMD B1/B2 checkpoints are not valid substitutes for original M1.",
    ]
    (OUT/"FEATURE_SUFFICIENCY_DECISION.txt").write_text("\n".join(decision)+"\n",encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    roots = candidate_roots()
    for root in roots:
        if root.is_dir():
            for pattern in ("*.pt", "*.pth", "*.ckpt"):
                paths.extend(root.rglob(pattern))
    paths = sorted(set(path.resolve() for path in paths))
    inventory=[];accepted={}
    for path in paths:
        row={"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size}
        try:
            try:
                state=torch.load(path,map_location="cpu",weights_only=False)
            except TypeError:
                # Compatibility with older PyTorch releases that predate the
                # explicit ``weights_only`` argument.
                state=torch.load(path,map_location="cpu")
            if not isinstance(state,dict):
                raise TypeError("checkpoint payload is not a dict")
            year,fold,seed=infer_identity(path,state);status,reason=classify(path,state)
            row.update({"year":year,"fold":fold,"seed":seed,"status":status,"reason":reason})
            identity=(year,fold,seed)
            if status=="ACCEPT_FROZEN_M1_REPLAY" and identity in EXPECTED:
                if identity in accepted:
                    row["status"]="REJECT_DUPLICATE_IDENTITY";row["reason"]="more than one accepted checkpoint for year/fold/seed"
                else:
                    accepted[identity]=path
        except Exception as exc:
            row.update({"year":"","fold":"","seed":"","status":"REJECT_UNREADABLE","reason":repr(exc)})
        inventory.append(row)
    write_csv(OUT/"checkpoint_inventory.csv",inventory)
    missing=sorted(EXPECTED-set(accepted))
    gate={"passed":not missing and len(accepted)==60,"expected_checkpoint_count":60,
          "accepted_checkpoint_count":len(accepted),"missing_count":len(missing),
          "missing":[{"year":y,"fold":f,"seed":s} for y,f,s in missing],
          "accepted":{f"{y}-{f}-{s}":str(path) for (y,f,s),path in sorted(accepted.items())},
          "candidate_roots":[str(path) for path in roots]}
    (OUT/"checkpoint_completeness_gate.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    if not gate["passed"]:
        blocked_outputs(inventory,accepted,missing)
        print(f"MISSING_FROZEN_M1_CHECKPOINTS accepted={len(accepted)}/60")
        print(f"Report: {OUT/'FEATURE_SUFFICIENCY_REPORT.txt'}")
        return
    print("FROZEN_M1_CHECKPOINT_COMPLETENESS_PASS")
    print("All 60 frozen M1 replay checkpoints are present. No probe has been run by this gate.")


if __name__=="__main__":
    main()
