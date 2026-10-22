#!/usr/bin/env python3
"""Create the predeclared four-method table after both tmux jobs finish."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def align(rows, order, name):
    table = {row["sha256"].strip().lower(): row for row in rows}
    if len(table) != 799 or set(table) != set(order):
        raise RuntimeError(name + " is not the exact source cohort")
    return [table[value] for value in order]


def metric(name, y, p, invocation):
    prediction = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"method": name, "BA": balanced_accuracy_score(y, prediction),
            "AUC": roc_auc_score(y, p), "F1": f1_score(y, prediction, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "StageII_invocation_rate": float(invocation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective", type=Path, default=Path("artifacts/care-malben-host-v2/source/oof_predictions.csv"))
    parser.add_argument("--stage2", type=Path, default=Path("artifacts/care-stage2only-malben-v1/source/oof_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/care-malben-host-v2/source"))
    args = parser.parse_args()
    selected = read(args.selective)
    if len(selected) != 799 or len({row["sha256"].lower() for row in selected}) != 799:
        raise RuntimeError("selective comparison is incomplete")
    order = [row["sha256"].lower() for row in selected]
    stage2 = align(read(args.stage2), order, "Stage-II-only")
    y = np.asarray([int(row["label"]) for row in selected])
    if any(int(row["label"]) != int(y[index]) or int(row["fold"]) != int(selected[index]["fold"])
           for index, row in enumerate(stage2)):
        raise RuntimeError("Stage-II-only label/fold alignment mismatch")
    p1 = np.asarray([float(row["Global24__prob"]) for row in selected])
    route = np.asarray([int(row["A25__routed"]) for row in selected], dtype=bool)
    p_anomaly = np.asarray([float(row["A25__AnomalyBaseline__prob"]) for row in selected])
    p_malben = np.asarray([float(row["A25__MalBenContrast__prob"]) for row in selected])
    p_stage2 = np.asarray([float(row["StageIIOnly_MalBen__prob"]) for row in stage2])
    rows = [metric("Global24-only", y, p1, 0.0),
            metric("StageIIOnly-MalBen", y, p_stage2, 1.0),
            metric("A25 + AnomalyBaseline", y, p_anomaly, np.mean(route)),
            metric("A25 + MalBenContrast", y, p_malben, np.mean(route))]
    write(args.output / "main_comparison.csv", rows)
    lines = ["CARE-Droid predeclared four-method source comparison", "",
             "Strict 2017 OOF only; 2018-2021 are not read."]
    for row in rows:
        lines.append(f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} FP={row['FP']} FN={row['FN']} StageII={100*row['StageII_invocation_rate']:.2f}%")
    (args.output / "main_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
