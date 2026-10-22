#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(os.environ.get("PLRPD_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("PLRPD_OUTPUT", str(PROJECT / "outputs/care-p31-lr-pairwise-diagnostics-v1"))).resolve()
SOURCE_IMPL = PROJECT / "scripts/care-p31-lr-source-safe-fusion-v1"
sys.path.insert(0, str(SOURCE_IMPL))
from transforms import transform  # noqa: E402
from run_source import metrics, select_c_and_predict  # noqa: E402

FORMAL = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"
SOURCE_MEMBER = PROJECT / "artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"
SOURCE_P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
SOURCE_RESULT = PROJECT / "outputs/care-p31-lr-source-safe-fusion-v1"
PRIOR_FAMILY = PROJECT / "outputs/care-p31-classifier-family-v1"
YEARS = (2018, 2019, 2020, 2021)
ARM = "B5_PAIRWISE_HELLINGER"
SEED = 20261009


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value): return str(value or "").strip().lower()


def summarize(scope, y, probability):
    return {"scope": str(scope), "arm": ARM, **metrics(np.asarray(y, int), np.asarray(probability, float))}


def main():
    if OUTPUT.exists() and any(path.name != "logs" for path in OUTPUT.iterdir()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-empty output exists: " + str(OUTPUT))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decision = json.loads((SOURCE_RESULT / "final_decision.json").read_text(encoding="utf-8"))
    if decision.get("status") != "SOURCE_CANDIDATE_FOUND" or decision.get("source_leader") != ARM:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source-selected B5 candidate is not frozen")
    source_decision_hash = hashlib.sha256((SOURCE_RESULT / "final_decision.json").read_bytes()).hexdigest()
    resolved = json.loads((FORMAL / "resolved_inputs.json").read_text(encoding="utf-8"))
    source_rows = read_csv(SOURCE_MEMBER)
    source_ids = np.asarray([norm(row["sha256"]) for row in source_rows])
    source_y = np.asarray([int(float(row["label"])) for row in source_rows], int)
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        ids = [norm(value) for value in archive["sha256"]]; values = np.asarray(archive["role_prevalence"], float)
    position = {sid: i for i, sid in enumerate(ids)}
    source_x = values[[position[sid] for sid in source_ids]]
    future_reference = read_csv(FORMAL / "references/future_label_free.csv")
    label_map = {norm(row["sha256"]): int(float(row["label"])) for row in read_csv(FORMAL / "frozen_predictions_with_labels.csv")}
    future_ids = np.asarray([norm(row["sha256"]) for row in future_reference])
    future_year = np.asarray([int(float(row["year"])) for row in future_reference], int)
    future_y = np.asarray([label_map[sid] for sid in future_ids], int)
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        pids = [norm(value) for value in archive["sha256"]]; pvalues = np.asarray(archive["role_prevalence"], float)
    pposition = {sid: i for i, sid in enumerate(pids)}
    future_x = pvalues[[pposition[sid] for sid in future_ids]]
    if len(source_y) != 799 or len(future_y) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
    tx_source = transform(source_x, ARM); tx_future = transform(future_x, ARM)
    frozen_probability, chosen, audit = select_c_and_predict(tx_source, source_y, tx_future, SEED + 9100000)
    dump_json(OUTPUT / "frozen_fit_audit.json", {"selected_C": chosen, "all_C": audit,
        "train_rows": 799, "test_rows": 1989, "future_used_for_selection": False})
    frozen_rows = [{"sha256": future_ids[i], "year": int(future_year[i]), "label": int(future_y[i]),
                    "probability": float(frozen_probability[i])} for i in range(len(future_y))]
    write_csv(OUTPUT / "frozen_predictions.csv", frozen_rows)
    frozen_metrics = [summarize(year, future_y[future_year == year], frozen_probability[future_year == year]) for year in YEARS]
    frozen_metrics.append(summarize("pooled", future_y, frozen_probability))
    write_csv(OUTPUT / "frozen_metrics.csv", frozen_metrics)
    print("P31_LR_PAIRWISE_FROZEN_DONE", flush=True)

    metadata_rows = read_csv(FORMAL / "within_2018_2019/predictions.csv") + read_csv(FORMAL / "within_2020_2021/predictions.csv")
    metadata = {norm(row["sha256"]): {"year": int(row["year"]), "fold": int(row["fold"]), "label": int(row["label"])} for row in metadata_rows}
    if len(metadata) != 1989 or set(metadata) != set(future_ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: same-year metadata changed")
    same_rows = []; same_metrics = []; same_fold_metrics = []; regularization = []
    checkpoints = OUTPUT / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        selected_ids = np.asarray(sorted(sid for sid, value in metadata.items() if value["year"] == year))
        y = np.asarray([metadata[sid]["label"] for sid in selected_ids], int)
        folds = np.asarray([metadata[sid]["fold"] for sid in selected_ids], int)
        x = transform(pvalues[[pposition[sid] for sid in selected_ids]], ARM)
        probability = np.full(len(y), np.nan)
        for fold in range(1, 6):
            train = np.flatnonzero(folds != fold); test = np.flatnonzero(folds == fold)
            checkpoint = checkpoints / f"{year}_fold_{fold}.npz"
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as archive:
                    stored = [norm(value) for value in archive["sha256"]]
                    local = np.asarray(archive["probability"], float)
                if stored != selected_ids[test].tolist(): raise RuntimeError("incompatible checkpoint")
                probability[test] = local
            else:
                local, chosen_local, audit_local = select_c_and_predict(x[train], y[train], x[test],
                    SEED + year * 10000 + fold * 100)
                probability[test] = local
                np.savez_compressed(checkpoint, sha256=selected_ids[test], probability=local)
                for row in audit_local:
                    regularization.append({"year": year, "outer_fold": fold, **row,
                                           "selected": int(row["C"] == chosen_local["C"])})
            print(f"P31_LR_PAIRWISE_WITHIN year={year} fold={fold}/5", flush=True)
        if not np.isfinite(probability).all(): raise RuntimeError("incomplete same-year OOF")
        same_metrics.append(summarize(year, y, probability))
        for fold in range(1, 6):
            take = folds == fold; same_fold_metrics.append({"year": year, "fold": fold, **metrics(y[take], probability[take])})
        for i, sid in enumerate(selected_ids):
            same_rows.append({"sha256": sid, "year": year, "fold": int(folds[i]), "label": int(y[i]),
                              "probability": float(probability[i])})
    all_y = np.asarray([row["label"] for row in same_rows], int)
    all_p = np.asarray([row["probability"] for row in same_rows], float)
    same_metrics.append(summarize("pooled", all_y, all_p))
    write_csv(OUTPUT / "within_predictions.csv", same_rows); write_csv(OUTPUT / "within_metrics.csv", same_metrics)
    write_csv(OUTPUT / "within_fold_metrics.csv", same_fold_metrics); write_csv(OUTPUT / "within_regularization_audit.csv", regularization)
    print("P31_LR_PAIRWISE_WITHIN_DONE", flush=True)
    prior_frozen = PRIOR_FAMILY / "frozen_future_metrics.csv"
    prior_within = PRIOR_FAMILY / "within_year_metrics.csv"
    references = {"frozen_classifier_family_metrics": str(prior_frozen), "within_classifier_family_metrics": str(prior_within),
                  "both_present": prior_frozen.exists() and prior_within.exists()}
    source_metric = decision["source_leader_metrics"]
    pooled_frozen = frozen_metrics[-1]; pooled_within = same_metrics[-1]
    result = {"status": "FROZEN_DIAGNOSTICS_COMPLETE", "arm": ARM, "source_decision_hash": source_decision_hash,
        "source": source_metric, "frozen_future_pooled": pooled_frozen, "same_year_pooled": pooled_within,
        "future_used_for_selection": False, "formal_stage1_changed": False, "stage2_run": False,
        "references": references, "interpretation": "Future and same-year results are diagnostics and cannot alter B5."}
    dump_json(OUTPUT / "final_decision.json", result)
    protocol = {"source_candidate": ARM, "source_decision_hash": source_decision_hash,
        "frozen_train": "2017 all 799", "frozen_test": "2018-2021 1989", "same_year": "independent fixed five-fold OOF",
        "future_used_for_selection": False, "stage2_run": False, "threshold": 0.5, "seed": SEED}
    dump_json(OUTPUT / "protocol.json", protocol)
    lines = ["# P31-LR pairwise Hellinger frozen diagnostics", "", f"Source BA: {source_metric['BA']:.6f}",
        f"Frozen future pooled BA: {pooled_frozen['BA']:.6f}", f"Same-year pooled BA: {pooled_within['BA']:.6f}", "",
        "B5 was selected using 2017 only. These diagnostics do not authorize retrospective tuning."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_P31_LR_PAIRWISE_DIAGNOSTICS_DONE", flush=True)


if __name__ == "__main__": main()

