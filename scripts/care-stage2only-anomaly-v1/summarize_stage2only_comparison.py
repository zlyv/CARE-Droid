#!/usr/bin/env python3
"""Paired OOF comparison of independent anomaly and MalBen representations."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric(name, y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "method": name,
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "StageII_invocation_rate": 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly", type=Path, required=True)
    parser.add_argument("--malben", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anomaly = read(args.anomaly)
    malben_by_sha = {row["sha256"].strip().lower(): row for row in read(args.malben)}
    if len(anomaly) != 799 or len(malben_by_sha) != 799:
        raise RuntimeError("Stage-II-only comparison requires exact 799-row inputs")
    aligned = []
    for row in anomaly:
        sha = row["sha256"].strip().lower()
        other = malben_by_sha.get(sha)
        if other is None or int(row["label"]) != int(other["label"]) or int(row["fold"]) != int(other["fold"]):
            raise RuntimeError("Stage-II-only alignment mismatch: " + sha)
        aligned.append(other)
    y = np.asarray([int(row["label"]) for row in anomaly])
    p_anomaly = np.asarray([float(row["StageIIOnly_Anomaly__prob"]) for row in anomaly])
    p_malben = np.asarray([float(row["StageIIOnly_MalBen__prob"]) for row in aligned])
    rows = [
        metric("StageIIOnly-Anomaly", y, p_anomaly),
        metric("StageIIOnly-MalBen", y, p_malben),
    ]
    delta = {
        "method": "MalBen-minus-Anomaly",
        "BA": rows[1]["BA"] - rows[0]["BA"],
        "AUC": rows[1]["AUC"] - rows[0]["AUC"],
        "F1": rows[1]["F1"] - rows[0]["F1"],
        "TN": rows[1]["TN"] - rows[0]["TN"],
        "FP": rows[1]["FP"] - rows[0]["FP"],
        "FN": rows[1]["FN"] - rows[0]["FN"],
        "TP": rows[1]["TP"] - rows[0]["TP"],
        "StageII_invocation_rate": 0.0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows + [delta])
    lines = [
        "CARE-Droid Stage-II-only representation comparison", "",
        "Strict paired 2017 OOF; both methods invoke Stage II for all 799 APKs.",
    ]
    for row in rows:
        lines.append(
            f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} FP={row['FP']} FN={row['FN']}"
        )
    lines.append(
        f"MalBen-Anomaly: dBA={delta['BA']:+.6f} dAUC={delta['AUC']:+.6f} "
        f"dF1={delta['F1']:+.6f} dFP={delta['FP']:+d} dFN={delta['FN']:+d}"
    )
    (args.output / "comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
