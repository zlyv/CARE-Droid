#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from p31_core import MEMBERS, foldwise_top25, mean_opposition

ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
PREDICTIONS = Path(
    "artifacts/care-litelocal-stage1-attribution-source-v1/"
    "source_member_oof_predictions.csv"
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normal(value):
    return str(value or "").strip().casefold()


def main():
    rows = read(PREDICTIONS)
    required = {"sha256", "year", "fold", "label",
                *(f"p_P31__{name}" for name in MEMBERS),
                "p_P31__MEAN_COMMITTEE"}
    if len(rows) != 799 or not rows or not required.issubset(rows[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked P31 OOF table changed")
    ids = np.asarray([normal(row["sha256"]) for row in rows])
    year = np.asarray([int(float(row["year"])) for row in rows])
    fold = np.asarray([int(float(row["fold"])) for row in rows])
    label = np.asarray([int(float(row["label"])) for row in rows])
    members = np.asarray([
        [float(row[f"p_P31__{name}"]) for name in MEMBERS] for row in rows
    ])
    probability = np.asarray([
        float(row["p_P31__MEAN_COMMITTEE"]) for row in rows
    ])
    if len(set(ids.tolist())) != 799 or set(year.tolist()) != {2017}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not strict 2017")
    if Counter(fold.tolist()) != Counter({1: 160, 2: 160, 3: 160, 4: 160, 5: 159}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: outer folds changed")
    if np.max(np.abs(probability - members.mean(axis=1))) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 mean committee mismatch")
    score = mean_opposition(members, probability)
    route = foldwise_top25(score, fold, ids)
    route_counts = Counter(fold[route].tolist())
    if int(route.sum()) != 200 or route_counts != Counter({1: 40, 2: 40, 3: 40, 4: 40, 5: 40}):
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 DI25 budget mismatch")
    prediction = probability >= .5
    ba = float(balanced_accuracy_score(label, prediction))
    auc = float(roc_auc_score(label, probability))
    f1 = float(f1_score(label, prediction))
    errors = prediction != label
    captured = int(np.sum(errors & route))
    if not (abs(ba - 0.9261309523809524) < 1e-12 and
            abs(auc - 0.971860902255639) < 1e-12 and
            int(errors.sum()) == 59 and captured == 46):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked P31 result mismatch")
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append({
            "sha256": ids[index], "label": int(label[index]),
            "fold": int(fold[index]),
            # Compatibility fields required by the frozen v4 loader. They are P31.
            "Global24__prob": float(probability[index]),
            "Global24__A25__JointS7_ordinary__prob": float(probability[index]),
            "A25__score": float(score[index]),
            "A25__routed": int(route[index]),
            "P31_MEAN_COMMITTEE__prob": float(probability[index]),
            "P31_DI25__score": float(score[index]),
            "P31_DI25__routed": int(route[index]),
        })
    target = ROOT / "preflight/p31_reference/source_predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    contract = {
        "stage1_representation": "P31 only (31-D sensitive-role method prevalence)",
        "stage1_definition": "mean of calibrated LR/RF/XGB/SVM",
        "router": "fold-wise top-25% MeanOpposition",
        "source_rows": 799, "year": 2017,
        "fold_sizes": {str(k): v for k, v in Counter(fold.tolist()).items()},
        "routed": 200,
        "route_per_fold": {str(k): v for k, v in route_counts.items()},
        "stage1_BA": ba, "stage1_AUC": auc, "stage1_F1": f1,
        "stage1_errors": int(errors.sum()), "routed_errors": captured,
        "error_recall": captured / int(errors.sum()),
        "routing_precision": captured / 200,
        "compatibility_probability_column": "Global24__prob",
        "compatibility_route_column": "A25__routed",
        "compatibility_fields_semantically_are_P31": True,
    }
    (ROOT / "preflight/p31_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    print("P31_C43_REFERENCE_PASS")
    print("source_rows=799 year=2017 route_P31_DI25=200 routed_errors=46/59")
    print(f"P31_MEAN_COMMITTEE BA={ba:.12f} AUC={auc:.12f} F1={f1:.12f}")


if __name__ == "__main__":
    main()
