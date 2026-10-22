#!/usr/bin/env python3
"""Fixed L2-LR strict same-year OOF diagnostic probes for predeclared candidates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_common import FEATURES, FOLDS, PREDICTIONS, PREFLIGHT, SEED, YEARS, norm, read_csv, write_csv, write_json


def main():
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREFLIGHT / "source_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((PREFLIGHT / "contract.json").read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("diagnostic_L2_LR_allowed") is not True:
        raise RuntimeError("BLOCKED_BY_CONTRACT: diagnostic-probe contract did not pass")
    schema = json.loads((FEATURES / "candidate_feature_schema.json").read_text(encoding="utf-8"))
    group_columns = schema["candidate_groups"]
    set_groups = {**schema["fixed_primary_sets"], **schema["fixed_secondary_sets"]}
    set_columns = {name: [column for group in groups for column in group_columns[group]] for name, groups in set_groups.items()}
    all_columns = []
    for columns in group_columns.values(): all_columns.extend(columns)

    metadata, matrices = [], {group: [] for group in group_columns}
    with (FEATURES / "candidate_features_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(all_columns) - set(reader.fieldnames or [])
        if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: candidate feature CSV missing schema columns")
        for row in reader:
            metadata.append({"sha256": norm(row["sha256"]), "year": int(row["year"]), "fold": int(row["fold"])})
            for group, columns in group_columns.items(): matrices[group].append([float(row[column]) for column in columns])
    matrices = {group: np.asarray(values, dtype=np.float64) for group, values in matrices.items()}
    feature_ids = [row["sha256"] for row in metadata]
    if len(feature_ids) != 1989 or len(set(feature_ids)) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: feature cohort is not exact unique 1989")
    cohort = {norm(row["sha256"]): row for row in read_csv(Path(manifest["cohort_file"]))}
    if set(feature_ids) != set(cohort): raise RuntimeError("BLOCKED_BY_CONTRACT: feature/cohort lineage mismatch")
    y = np.asarray([int(float(cohort[sample]["label"])) for sample in feature_ids], dtype=int)
    p_stage1 = np.asarray([float(cohort[sample]["p_stage1"]) for sample in feature_ids])
    target = ((p_stage1 >= 0.5) != y).astype(int)
    route = np.asarray([int(float(cohort[sample]["route_DI"])) == 1 for sample in feature_ids])
    years = np.asarray([row["year"] for row in metadata]); folds = np.asarray([row["fold"] for row in metadata])
    expected_routed = int(manifest["routed_n"])
    if int(np.sum(route)) != expected_routed: raise RuntimeError("BLOCKED_BY_CONTRACT: routed cohort changed")

    candidate_matrices = {}
    for candidate, groups in set_groups.items():
        candidate_matrices[candidate] = np.concatenate([matrices[group] for group in groups], axis=1)
    predictions, coefficients, isolation = [], [], []
    seen = {candidate: set() for candidate in candidate_matrices}
    for year in YEARS:
        for fold in FOLDS:
            train = np.flatnonzero(route & (years == year) & (folds != fold))
            validation = np.flatnonzero(route & (years == year) & (folds == fold))
            if set(train) & set(validation) or not len(validation):
                raise RuntimeError("BLOCKED_BY_CONTRACT: OOF fold isolation failed")
            if len(np.unique(target[train])) != 2:
                raise RuntimeError("BLOCKED_BY_CONTRACT: diagnostic train fold has one target class")
            isolation.append({"year": year, "fold": fold, "train_n": len(train), "validation_n": len(validation),
                              "train_wrong_n": int(np.sum(target[train])), "validation_wrong_n": int(np.sum(target[validation])),
                              "train_validation_overlap_n": 0})
            for candidate, matrix in candidate_matrices.items():
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced",
                                       max_iter=2000, random_state=SEED),
                )
                model.fit(matrix[train], target[train])
                probability = model.predict_proba(matrix[validation])[:, 1]
                for index, sample_index in enumerate(validation):
                    sample = feature_ids[sample_index]
                    if sample in seen[candidate]: raise RuntimeError("duplicate OOF candidate prediction")
                    seen[candidate].add(sample)
                    predictions.append({"candidate": candidate, "sha256": sample, "year": year, "fold": fold,
                                        "need_correction": int(target[sample_index]), "probability": float(probability[index])})
                if candidate in ("A", "B", "C"):
                    lr = model.named_steps["logisticregression"]
                    for name, value in zip(set_columns[candidate], lr.coef_[0]):
                        coefficients.append({"candidate": candidate, "year": year, "fold": fold,
                                             "feature": name, "standardized_coefficient": float(value)})
            print(f"candidate probes {year} fold={fold}/5 done", flush=True)
    routed_ids = {feature_ids[index] for index in np.flatnonzero(route)}
    if any(values != routed_ids for values in seen.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete OOF candidate predictions")
    write_csv(PREDICTIONS / "candidate_oof_predictions.csv", predictions)
    write_csv(PREDICTIONS / "diagnostic_fold_coefficients.csv", coefficients)
    write_csv(PREDICTIONS / "oof_fold_isolation_audit.csv", isolation)
    write_json(PREDICTIONS / "probe_contract.json", {
        "classifier": "StandardScaler + L2 LogisticRegression", "C": 1.0, "solver": "liblinear",
        "class_weight": "balanced", "max_iter": 2000, "random_state": SEED,
        "threshold": 0.5, "hyperparameter_search": False, "feature_selection": False,
        "cross_year_training": False, "pooled_classifier_training": False,
        "CARE_Droid_model": False, "diagnostic_only": True,
    })
    print("HOST_COMPLEMENT_CANDIDATE_OOF_PROBES_DONE")


if __name__ == "__main__":
    main()
