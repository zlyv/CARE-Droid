#!/usr/bin/env python3
"""Summarize the strict same-year OOF 2 x 2 x 2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROUTERS = ("A25", "DI25")
REPRESENTATIONS = ("anomaly", "malben")
HEADS = ("direct", "bounded")
ROUTE_COLUMNS = {"A25": "route_A25", "DI25": "route_DI"}


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(y, p) -> dict:
    pred = np.asarray(p) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-inputs", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    router_rows = read_csv(args.router_inputs)
    ids = [norm(row["sha256"]) for row in router_rows]
    labels = np.asarray([int(row["label"]) for row in router_rows], dtype=np.int64)
    years = np.asarray([int(row["year"]) for row in router_rows], dtype=np.int64)
    base = np.asarray([float(row["p_stage1"]) for row in router_rows], dtype=np.float64)
    if len(ids) != 1989 or len(set(ids)) != 1989:
        raise RuntimeError("router input contract failed")

    expert = {}
    for representation in REPRESENTATIONS:
        for head in HEADS:
            path = args.arms_root / f"{representation}_{head}" / "within_year_expert_predictions.csv"
            rows = read_csv(path)
            by_id = {norm(row["sha256"]): row for row in rows}
            if len(rows) != len(by_id) or set(by_id) != set(ids):
                raise RuntimeError("expert prediction alignment failed: " + str(path))
            expert[(representation, head)] = np.asarray(
                [float(by_id[value]["p_expert"]) for value in ids], dtype=np.float64)

    annual = []
    pooled = []
    sample = []
    methods = [("Global24", base, np.zeros(len(base), dtype=bool))]
    for router in ROUTERS:
        route_column = ROUTE_COLUMNS[router]
        if route_column not in router_rows[0]:
            raise RuntimeError(
                f"router input is missing {route_column}; available columns="
                + repr(sorted(router_rows[0]))
            )
        route = np.asarray(
            [int(row[route_column]) == 1 for row in router_rows], dtype=bool
        )
        for representation in REPRESENTATIONS:
            for head in HEADS:
                name = f"{router}__{'AnomalyBaseline' if representation == 'anomaly' else 'MalBenContrast'}__{'DirectStageII' if head == 'direct' else 'BoundedCorrection'}"
                final = np.where(route, expert[(representation, head)], base)
                methods.append((name, final, route))

    base_pred = base >= 0.5
    for name, probability, route in methods:
        pred = probability >= 0.5
        base_correct = base_pred == labels
        final_correct = pred == labels
        fixed = (~base_correct) & final_correct
        broken = base_correct & (~final_correct)
        value = metrics(labels, probability)
        pooled.append({
            "method": name, **value,
            "fixed": int(np.sum(fixed)), "broken": int(np.sum(broken)),
            "net": int(np.sum(fixed) - np.sum(broken)),
            "routed_count": int(np.sum(route)), "routed_rate": float(np.mean(route)),
        })
        for year in sorted(set(years.tolist())):
            mask = years == year
            item = metrics(labels[mask], probability[mask])
            annual.append({
                "method": name, "year": year, **item,
                "fixed": int(np.sum(fixed & mask)), "broken": int(np.sum(broken & mask)),
                "net": int(np.sum(fixed & mask) - np.sum(broken & mask)),
                "routed_count": int(np.sum(route & mask)), "year_n": int(np.sum(mask)),
            })
        for i, value_p in enumerate(probability):
            sample.append({
                "sha256": ids[i], "year": int(years[i]), "label": int(labels[i]),
                "method": name, "p_final": float(value_p), "routed": int(route[i]),
                "correct": int(pred[i] == labels[i]),
            })

    macro = []
    for row in pooled:
        subset = [item for item in annual if item["method"] == row["method"]]
        macro.append({
            "method": row["method"],
            "mean_BA": float(np.mean([item["BA"] for item in subset])),
            "min_BA": float(np.min([item["BA"] for item in subset])),
            "mean_AUC": float(np.mean([item["AUC"] for item in subset])),
            "mean_F1": float(np.mean([item["F1"] for item in subset])),
            "mean_net": float(np.mean([item["net"] for item in subset])),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "within_year_annual_metrics.csv", annual)
    write_csv(args.output / "within_year_pooled_metrics.csv", pooled)
    write_csv(args.output / "within_year_macro_metrics.csv", macro)
    write_csv(args.output / "within_year_sample_predictions.csv", sample)

    lines = [
        "CARE-Droid independent within-year 2 x 2 x 2 matrix v1", "",
        "Each year uses its own strict five-fold OOF training/evaluation.",
        "These are same-distribution diagnostics, not temporal-generalization results.",
        "Budget=25%; max_delta(logit)=2.25 remains frozen from 2017.", "",
        "Pooled same-year OOF:",
    ]
    for row in pooled:
        lines.append(
            f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} FP={row['FP']} FN={row['FN']} "
            f"fixed={row['fixed']} broken={row['broken']} net={row['net']} "
            f"routed={row['routed_count']}/1989")
    lines.extend(["", "Annual same-year OOF:"])
    for year in sorted(set(years.tolist())):
        lines.append(str(year))
        for row in [item for item in annual if item["year"] == year]:
            lines.append(
                f"  {row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
                f"F1={row['F1']:.6f} net={row['net']}")
    lines.extend(["", "Interpretation boundary:",
                  "Do not describe these results as future-year, zero-day, or temporal performance."])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "protocol.json").write_text(json.dumps({
        "same_year_five_fold_oof": True,
        "temporal_generalization_result": False,
        "annual_models_are_independent": True,
        "budget": 0.25, "max_delta_logit": 2.25,
    }, indent=2) + "\n", encoding="utf-8")
    print("WITHIN_YEAR_MATRIX_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
