#!/usr/bin/env python3
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path("artifacts/care-di25-hostrel-semantic-background-source-v1")
H0 = Path("artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY")
B1 = Path("artifacts/care-di25-roleconditioned-background-source-v1/arms/B1_HARD_ROLE")
R1 = ROOT/"arms/R1_HOSTREL74_HARD_ROLE"
ARMS = ("B0_H0_LOCKED", "B1_HARD_ROLE_LOCKED", "R1_HOSTREL74_HARD_ROLE")
PATHS = {"B0_H0_LOCKED": H0, "B1_HARD_ROLE_LOCKED": B1, "R1_HOSTREL74_HARD_ROLE": R1}
METHODS = {
    "B0_H0_LOCKED": "DI25__Host__Anomaly__GraphGlobalPath__Q_QUERY_ONLY__Bounded",
    "B1_HARD_ROLE_LOCKED": "DI25__Host__Anomaly__QQuery__HardRoleBackground__Bounded",
    "R1_HOSTREL74_HARD_ROLE": "DI25__HostRel74__Anomaly__QQuery__HardRoleBackground__Bounded",
}


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def exact(a, b):
    n = a + b
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / (2 ** n))


def metrics(y, probability, base):
    pred = probability >= .5; base_pred = base >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum((base_pred != y) & (pred == y)))
    broken = int(np.sum((base_pred == y) & (pred != y)))
    return {"BA": float(balanced_accuracy_score(y, pred)), "AUC": float(roc_auc_score(y, probability)), "F1": float(f1_score(y, pred)), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "fixed": fixed, "broken": broken, "net": fixed-broken}


def main():
    ref = read(ROOT/"preflight/di25_reference/source_predictions.csv")
    ref.sort(key=lambda r: r["sha256"].lower())
    ids = [r["sha256"].lower() for r in ref]
    y = np.asarray([int(float(r["label"])) for r in ref])
    fold = np.asarray([int(float(r["fold"])) for r in ref])
    route = np.asarray([int(float(r["A25__routed"])) for r in ref])
    base = np.asarray([float(r["Global24__prob"]) for r in ref])
    probabilities = {}
    route_rows = []
    for arm in ARMS:
        rows = read(PATHS[arm]/"source_predictions.csv")
        by_id = {r["sha256"].lower(): r for r in rows}
        if set(by_id) != set(ids):
            raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohort " + arm)
        column = METHODS[arm] + "__prob"
        probabilities[arm] = np.asarray([float(by_id[s][column]) for s in ids])
        for i, sample_id in enumerate(ids):
            arm_route = int(float(by_id[sample_id]["A25__routed"]))
            route_rows.append({"sha256": sample_id, "arm": arm, "route_DI": int(route[i]), "route_arm": arm_route, "different": int(int(route[i]) != arm_route)})
    if sum(r["different"] for r in route_rows):
        raise RuntimeError("BLOCKED_BY_CONTRACT: route changed")
    aggregate = [{"arm": arm, **metrics(y, probabilities[arm], base)} for arm in ARMS]
    fold_rows = [{"arm": arm, "fold": f, **metrics(y[fold == f], probabilities[arm][fold == f], base[fold == f])} for arm in ARMS for f in range(1, 6)]
    h0_correct = (probabilities[ARMS[0]] >= .5) == y
    paired = []
    for arm in ARMS[1:]:
        correct = (probabilities[arm] >= .5) == y
        n01 = int(np.sum(~h0_correct & correct)); n10 = int(np.sum(h0_correct & ~correct))
        paired.append({"comparison": arm + "_minus_B0_H0_LOCKED", "candidate_only_correct": n01, "H0_only_correct": n10, "paired_net": n01-n10, "exact_p": exact(n01, n10)})
    write(ROOT/"results/aggregate_metrics.csv", aggregate)
    write(ROOT/"results/per_fold_metrics.csv", fold_rows)
    write(ROOT/"results/paired_vs_H0.csv", paired)
    write(ROOT/"results/route_integrity.csv", route_rows)
    lines = [
        "CARE-Droid fixed-width HostRel74 semantic-background source v1", "",
        "Strict 2017 five-fold OOF; frozen DI25=200/799; Q-Query; no Ego tensor or future row.",
        "R1 replaces the near-duplicate second 31-D Host role block with directed one-hop neighbor-role compression.",
        "R1 backgrounds are rebuilt in the identical HostRel74 space and hard-conditioned by the Host endpoint role.",
        "B0 and B1 are locked controls and are not retrained; only R1 is trained.", "",
    ]
    for row in aggregate:
        lines.append(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    lines += ["", "Paired versus H0:"] + [f"{r['comparison']}: candidate_only={r['candidate_only_correct']} H0_only={r['H0_only_correct']} net={r['paired_net']} p={r['exact_p']:.6f}" for r in paired]
    lines += ["", "Do not run future or same-year diagnostics until the strict source result is inspected."]
    (ROOT/"results/REPORT.txt").write_text("\n".join(lines) + "\n")
    print("HOSTREL_SEMANTIC_BACKGROUND_SOURCE_SUMMARY_PASS")
    print((ROOT/"results/REPORT.txt").read_text())


if __name__ == "__main__":
    main()
