#!/usr/bin/env python3
"""Strict 2017 Host-only MalBenContrast with shared fold-local role scale."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT, ROOT / "care-malben-contrast-v2", ROOT / "care-rolegraph-conditioned-v4-script"):
    sys.path.insert(0, str(path))

import run_care_malben_role_source_v2 as old
from shared_role_prototypes import evidence_columns, fit_role_prototypes, ROLE_COUNT
from source_label_contract import install_source_label_contract


METHOD = "Global24__A25__CARE_MalBenHostV2_FixKeep"
ORIGINAL_FIT_SCALED = old.fit_scaled_role_bags


def fit_scaled_shared_role_bags(*args, **kwargs):
    train, test, audit = ORIGINAL_FIT_SCALED(*args, **kwargs)
    audit = dict(audit)
    audit["prototype_similarity"] = "exp(-RMS-Euclidean/shared-role-or-global-scale)"
    audit["prototype_scale_scope"] = "outer-train only; one common tau for S_ben and S_mal"
    return train, test, audit


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def emit_scale_audit(output: Path) -> None:
    mapping_path = ROOT / "care-source-recovery-stage2only-v1" / "extractor_repo" / "configs" / "protocol" / "semantic_mapping_v1.yaml"
    roles = list(yaml.safe_load(mapping_path.read_text(encoding="utf-8"))["roles"])
    rows = []
    for record in old.FIT_RECORDS:
        if record.get("purpose") != "final_outer_fit" or record.get("variant") != "MalBenContrast":
            continue
        fold = int(record["group"]) + 1
        for value in record["quality"]:
            rows.append({
                "fold": fold,
                "seed_argument": record["seed_argument"],
                "role": int(value["role"]),
                "role_name": roles[int(value["role"])],
                "n_ben": int(value["num_ben_regions"]),
                "n_mal": int(value["num_mal_regions"]),
                "tau_role": float(value["tau_role"]),
                "tau_global": float(value["tau_global"]),
                "fallback_ben": int(value["fallback_ben"]),
                "fallback_mal": int(value["fallback_mal"]),
            })
    write_csv(output.parent / "prototype_scale_audit.csv", rows)
    quality = output.parent / "source_prototype_quality.csv"
    if quality.is_file():
        with quality.open("r", encoding="utf-8-sig", newline="") as handle:
            quality_rows = list(csv.DictReader(handle))
        for row in quality_rows:
            row["role_name"] = roles[int(float(row["role"]))]
        write_csv(output.parent / "prototype_support.csv", quality_rows)


def rewrite_contract(output: Path) -> None:
    architecture_path = output / "architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    architecture.update({
        "experiment": "CARE Host-only MalBenContrast v2",
        "future_data_loaded": False,
        "variants": ["MalBenContrast"],
        "role_prototype": "outer-train-only role x class K=1; min_support=20",
        "fallback": "Role-Class-K1 -> Global-Class-K1 independently for benign and malware",
        "similarity_scale": "one shared outer-train tau_role for S_ben and S_mal; tau_global if either class side falls back",
        "prototype_space": "fold-standardized raw 74-D Host region x_ir^proto",
        "local_embedding": "h_ir is the 24-D LocalEncoder output and is not the prototype distance space",
        "local_input": "Host-only; cached Ego values are not loaded",
        "correction": "logit(p_stage1) + 2.25*tanh(o_i)",
        "role_conditioning": {
            "role_source": "argmax of declared-order 31 role-fraction columns; smallest-index tie",
            "K": 1, "min_support": 20,
            "fallback": "Role-Class-K1 -> same-class Global-Class-K1 independently",
        },
        "similarity": "exp(-RMS-Euclidean/shared outer-train tau_role); shared tau_global if either class side falls back",
    })
    architecture_path.write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    install_source_label_contract(old.v4)
    old.fit_role_prototypes = fit_role_prototypes
    old.evidence_columns = evidence_columns
    old.fit_scaled_role_bags = fit_scaled_shared_role_bags
    old.ROLE_COUNT = ROLE_COUNT
    old.VARIANTS = ("MalBenContrast",)
    old.METHODS = {"MalBenContrast": METHOD}
    old.v4.MODEL_VARIANTS = old.VARIANTS
    old.v4.METHOD_FOR_VARIANT = old.METHODS
    old.v4.v3.fit_scaled_dualproto_bags = fit_scaled_shared_role_bags
    old.v4.AUGMENTED_REGION_DIM = old.FULL_REGION_DIM
    old.v4.RoleGraphConditionedEvidence = old.RoleConditionedEvidenceModel
    old.v4.predict_model = old.capture_predict_model
    old.v4.fit_predict_variant = old.grouped_fit_predict
    old.main()
    output = Path("artifacts/care-malben-host-v2/source/model")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    emit_scale_audit(output)
    rewrite_contract(output)
    fit_path = output / "fit_audit.csv"
    if fit_path.exists():
        with fit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            values = list(csv.DictReader(handle))
        parameters = {int(float(row["parameter_n"])) for row in values}
        if parameters != {33388}:
            raise RuntimeError(f"selective MalBen parameter contract failed: {parameters}")
    print("MalBen Host v2 contracts passed", flush=True)


if __name__ == "__main__":
    main()
