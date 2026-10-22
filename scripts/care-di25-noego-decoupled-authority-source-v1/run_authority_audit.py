#!/usr/bin/env python3
"""Evaluate a fixed decoupled correction rule on strict 2017 OOF scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from authority_core import CROSSING_MARGIN, GATE_THRESHOLD, MAX_DELTA, decoupled_correction
from preflight import ARMS, GATES, ROUTER, main as preflight

OUTPUT = Path("artifacts/care-di25-noego-decoupled-authority-source-v1")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metrics(y, p):
    pred = p >= 0.5
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(np.sum((y == 0) & (~pred))), "FP": int(np.sum((y == 0) & pred)),
        "FN": int(np.sum((y == 1) & (~pred))), "TP": int(np.sum((y == 1) & pred)),
    }


def main():
    preflight()
    router = read_csv(ROUTER); gates = {r["sha256"].strip().lower(): r for r in read_csv(GATES)}
    ids = [r["sha256"].strip().lower() for r in router]
    y = np.asarray([int(float(r["label"])) for r in router])
    folds = np.asarray([int(float(r["fold"])) for r in router])
    p0 = np.asarray([float(r["p_stage1"]) for r in router])
    route = np.asarray([int(float(r["route_DI"])) == 1 for r in router])
    base_correct = (p0 >= 0.5) == y
    prediction_rows = []
    summary_rows = []

    base_metrics = metrics(y, p0)
    summary_rows.append({"scope": "aggregate", "arm": "Global24", **base_metrics,
                         "routed": 0, "gate_active": 0, "routed_errors": 0,
                         "gate_error_recall": 0.0, "fixed": 0, "broken": 0, "net": 0,
                         "correction_precision": 0.0, "bound_limited": 0})
    results = {}
    for arm in ARMS:
        q = np.zeros(len(ids), dtype=np.float64)
        for i, sha in enumerate(ids):
            if route[i]: q[i] = float(gates[sha][f"score_{arm}"])
        out = decoupled_correction(p0, q, route)
        p = out["probability"]; final_correct = (p >= 0.5) == y
        fixed_mask = (~base_correct) & final_correct
        broken_mask = base_correct & (~final_correct)
        routed_error = route & (~base_correct)
        active_error = out["active"] & (~base_correct)
        fixed = int(fixed_mask.sum()); broken = int(broken_mask.sum())
        row = {"scope": "aggregate", "arm": arm, **metrics(y, p),
               "routed": int(route.sum()), "gate_active": int(out["active"].sum()),
               "routed_errors": int(routed_error.sum()),
               "gate_error_recall": float(active_error.sum() / max(1, routed_error.sum())),
               "fixed": fixed, "broken": broken, "net": fixed - broken,
               "correction_precision": float(fixed / max(1, fixed + broken)),
               "bound_limited": int(out["bound_limited"].sum())}
        summary_rows.append(row); results[arm] = (q, out, p)
        for fold in range(1, 6):
            m = folds == fold; bc = base_correct[m]; fc = (p[m] >= 0.5) == y[m]
            summary_rows.append({"scope": f"fold_{fold}", "arm": arm, **metrics(y[m], p[m]),
                "routed": int(route[m].sum()), "gate_active": int(out["active"][m].sum()),
                "routed_errors": int((route[m] & (~base_correct[m])).sum()),
                "gate_error_recall": float((out["active"][m] & (~base_correct[m])).sum() / max(1, (route[m] & (~base_correct[m])).sum())),
                "fixed": int(((~bc) & fc).sum()), "broken": int((bc & (~fc)).sum()),
                "net": int(((~bc) & fc).sum() - (bc & (~fc)).sum()),
                "correction_precision": float(((~bc) & fc).sum() / max(1, ((~bc) & fc).sum() + (bc & (~fc)).sum())),
                "bound_limited": int(out["bound_limited"][m].sum())})

    for i, sha in enumerate(ids):
        row = {"sha256": sha, "year": 2017, "fold": int(folds[i]), "label": int(y[i]),
               "route_DI": int(route[i]), "p_stage1": float(p0[i]), "stage1_correct": int(base_correct[i])}
        for arm, (q, out, p) in results.items():
            row.update({f"gate_{arm}": float(q[i]), f"active_{arm}": int(out["active"][i]),
                        f"delta_{arm}": float(out["delta"][i]), f"p_{arm}": float(p[i]),
                        f"bound_limited_{arm}": int(out["bound_limited"][i])})
        prediction_rows.append(row)
    write_csv(OUTPUT / "oof_predictions.csv", prediction_rows)
    write_csv(OUTPUT / "metrics.csv", summary_rows)
    contract = {"year": 2017, "source_rows": 799, "route_DI": 200, "future_rows_loaded": 0,
                "ego_consumed": False, "training_invoked": False, "threshold_search": False,
                "gate_threshold": GATE_THRESHOLD, "max_delta_logit": MAX_DELTA,
                "crossing_margin_logit": CROSSING_MARGIN,
                "direction": "opposite frozen Stage-I hard decision",
                "magnitude": "min(abs(stage1_logit)+0.05,2.25) when gate is active"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    lines = ["CARE-Droid strict-2017 no-Ego decoupled correction-authority audit",
             "No training, no threshold search, no future row, and no Ego tensor was used.", ""]
    for row in summary_rows:
        if row["scope"] == "aggregate":
            lines.append(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
                         f"active={row['gate_active']} fixed={row['fixed']} broken={row['broken']} "
                         f"net={row['net']} gate_error_recall={row['gate_error_recall']:.6f} "
                         f"bound_limited={row['bound_limited']}")
    lines += ["", "Interpretation: this isolates correction authorization. It is not a new learned Stage-II model.",
              "Advance to a trainable magnitude head only if an OOF gate yields positive net correction without excessive broken samples."]
    (OUTPUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("DECOUPLED_AUTHORITY_SOURCE_AUDIT_DONE")


if __name__ == "__main__":
    main()

