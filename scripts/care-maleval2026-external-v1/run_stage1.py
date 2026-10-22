#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

import numpy as np

from common import FORMAL_DIAG, OUTPUT, PROJECT, SEED, SOURCE_P31, SOURCE_SVM, mean_opposition, norm, read_csv, read_json, top25, write_csv, dump_json


def labels_by_sha(rows):
    result = {}
    for row in rows:
        if "label" in row and str(row["label"]).strip() != "":
            result[norm(row["sha256"])] = int(float(row["label"]))
    return result


def main():
    sys.path.insert(0, str(PROJECT / "scripts/care-svm-train-window-expansion-v1"))
    from stage1_core import fit_calibrated_committee

    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        source_ids = [norm(x) for x in archive["sha256"]]; source_x = np.asarray(archive["role_prevalence"], float)
    source_labels = labels_by_sha(read_csv(SOURCE_SVM))
    if len(source_ids) != 799 or any(sid not in source_labels for sid in source_ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source SVM labels unavailable")
    source_y = np.asarray([source_labels[sid] for sid in source_ids], int)

    resolved = read_json(FORMAL_DIAG / "resolved_inputs.json")
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        future_ids = [norm(x) for x in archive["sha256"]]; future_year = np.asarray(archive["year"], int)
        future_x = np.asarray(archive["role_prevalence"], float)
    within = read_csv(FORMAL_DIAG / "within_2018_2019/predictions.csv")
    labels18 = {norm(row["sha256"]): int(float(row["label"])) for row in within if int(float(row["year"])) == 2018}
    mask18 = future_year == 2018; ids18 = [sid for sid, keep in zip(future_ids, mask18) if keep]
    if len(ids18) != 793 or any(sid not in labels18 for sid in ids18):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen 2018 training labels unavailable")
    train_x = np.concatenate([source_x, future_x[mask18]], axis=0)
    train_y = np.concatenate([source_y, np.asarray([labels18[sid] for sid in ids18], int)])

    with np.load(OUTPUT / "maleval_p31.npz", allow_pickle=False) as archive:
        target_ids = np.asarray([norm(x) for x in archive["sha256"]]); target_x = np.asarray(archive["role_prevalence"], float)
    member = fit_calibrated_committee(train_x, train_y, target_x, SEED, inner_folds=3, jobs=4)
    base = member[:, 3]; route_score = mean_opposition(member, base); route = top25(route_score, target_ids)
    manifest = {norm(row["sha256"]): row for row in read_csv(OUTPUT / "dataset_manifest_label_free.csv")}
    rows = []
    for i, sid in enumerate(target_ids):
        rows.append({"sha256": sid, "cohort": manifest[sid]["cohort"],
                     "p_LR": float(member[i, 0]), "p_RF": float(member[i, 1]),
                     "p_XGB": float(member[i, 2]), "p_SVM": float(member[i, 3]),
                     "stage1_prob": float(base[i]), "router_score": float(route_score[i]), "routed": int(route[i])})
    write_csv(OUTPUT / "stage1_predictions_label_free.csv", rows)
    routed_manifest = [manifest[norm(row["sha256"])] for row in rows if int(row["routed"])]
    write_csv(OUTPUT / "routed_manifest_label_free.csv", routed_manifest)
    dump_json(OUTPUT / "stage1_fit_audit.json", {"status": "PASS", "train_rows": len(train_y),
        "train_years": [2017, 2018], "target_rows": len(rows), "target_labels_loaded": False,
        "stage1": "calibrated P31-SVM", "router": "SVM-anchored MeanOpposition Top25%",
        "routed": int(route.sum()), "seed": SEED})
    print(f"MALEVAL_STAGE1_DONE train={len(train_y)} target=255 routed={int(route.sum())} labels_loaded=false", flush=True)


if __name__ == "__main__":
    main()

