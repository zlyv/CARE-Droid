#!/usr/bin/env python3
"""Unseal future labels after all four experts freeze and evaluate eight routes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


YEARS = (2018, 2019, 2020, 2021)
ARMS = (("anomaly", "direct"), ("anomaly", "bounded"),
        ("malben", "direct"), ("malben", "bounded"))
DISPLAY = {"anomaly": "AnomalyBaseline", "malben": "MalBenContrast"}


def norm(value): return str(value or "").strip().lower()


def row_id(row):
    return norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id"))


def parse_label(value):
    text = norm(value)
    if text in {"1", "1.0", "malware", "malicious", "true"}: return 1
    if text in {"0", "0.0", "benign", "goodware", "false"}: return 0
    raise RuntimeError("unknown post-hoc future label: " + repr(value))


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metric(labels, probability, base, route, method, scope, year=""):
    labels = np.asarray(labels, dtype=np.int64); probability = np.asarray(probability)
    base = np.asarray(base); route = np.asarray(route, dtype=bool)
    pred = probability >= 0.5; base_ok = (base >= 0.5) == labels; final_ok = pred == labels
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum(route & ~base_ok & final_ok)); broken = int(np.sum(route & base_ok & ~final_ok))
    errors = ~base_ok
    return {
        "scope": scope, "year": year, "method": method, "n": len(labels),
        "routed_count": int(np.sum(route)), "routed_rate": float(np.mean(route)),
        "BA": float(balanced_accuracy_score(labels, pred)),
        "AUC": float(roc_auc_score(labels, probability)),
        "F1": float(f1_score(labels, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "error_recall": float(np.sum(route & errors) / max(np.sum(errors), 1)),
        "diagnostic_only": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolved", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"))
    p.add_argument("--arms-root", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/arms"))
    p.add_argument("--output", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/results"))
    args = p.parse_args()
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    frozen = read_csv(resolved["future_router_inputs_label_free"])
    labeled = read_csv(resolved["future_label_file_posthoc_only"])
    label_by_sha = {row_id(row): parse_label(row.get("label") or row.get("metadata_label"))
                    for row in labeled}
    if len(label_by_sha) != 1989: raise RuntimeError("future label seal cohort changed")
    sha = [norm(row["sha256"]) for row in frozen]
    years = np.asarray([int(row["year"]) for row in frozen], dtype=np.int64)
    labels = np.asarray([label_by_sha[value] for value in sha], dtype=np.int64)
    base = np.asarray([float(row["p_stage1"]) for row in frozen])
    routes = {
        "A25": np.asarray([int(row["route_A25"]) for row in frozen], dtype=bool),
        "DI25": np.asarray([int(row["route_DI"]) for row in frozen], dtype=bool),
    }
    experts = {}
    for representation, head in ARMS:
        path = args.arms_root / f"{representation}_{head}" / "future_expert_predictions.csv"
        rows = read_csv(path)
        by_sha = {norm(row["sha256"]): float(row["p_expert"]) for row in rows}
        if len(by_sha) != 1989 or set(by_sha) != set(sha):
            raise RuntimeError("expert cohort mismatch: " + str(path))
        experts[(representation, head)] = np.asarray([by_sha[value] for value in sha])

    metrics, sample = [], []
    final_map = {"Global24": base}
    zero = np.zeros(len(base), dtype=bool)
    for year in YEARS:
        idx = years == year
        metrics.append(metric(labels[idx], base[idx], base[idx], zero[idx], "Global24", "annual", year))
    metrics.append(metric(labels, base, base, zero, "Global24", "pooled_2018_2021"))
    for router, route in routes.items():
        for representation, head in ARMS:
            expert = experts[(representation, head)]
            final = np.where(route, expert, base)
            name = f"{router}__{DISPLAY[representation]}__{'DirectStageII' if head == 'direct' else 'BoundedCorrection'}"
            final_map[name] = final
            for year in YEARS:
                idx = years == year
                metrics.append(metric(labels[idx], final[idx], base[idx], route[idx], name, "annual", year))
            metrics.append(metric(labels, final, base, route, name, "pooled_2018_2021"))
    for i, value in enumerate(sha):
        row = {
            "sha256": value, "year": int(years[i]), "label": int(labels[i]),
            "p_stage1": float(base[i]), "route_A25": int(routes["A25"][i]),
            "route_DI25": int(routes["DI25"][i]),
        }
        for (representation, head), probability in experts.items():
            row[f"p_expert_{representation}_{head}"] = float(probability[i])
        for name, probability in final_map.items():
            row[name + "__prob"] = float(probability[i])
        sample.append(row)
    pairwise = []
    for router in routes:
        for representation in ("AnomalyBaseline", "MalBenContrast"):
            direct = f"{router}__{representation}__DirectStageII"
            bounded = f"{router}__{representation}__BoundedCorrection"
            a = final_map[direct] >= 0.5; b = final_map[bounded] >= 0.5
            for scope, idx in [("pooled_2018_2021", np.ones(len(labels), bool))] + [(str(y), years == y) for y in YEARS]:
                ca = a[idx] == labels[idx]; cb = b[idx] == labels[idx]
                pairwise.append({
                    "scope": scope, "comparison": f"{direct} vs {bounded}",
                    "direct_only_correct": int(np.sum(ca & ~cb)),
                    "bounded_only_correct": int(np.sum(~ca & cb)),
                    "net_direct_minus_bounded": int(np.sum(ca & ~cb) - np.sum(~ca & cb)),
                    "prediction_disagreement": float(np.mean(a[idx] != b[idx])),
                })
        for head in ("DirectStageII", "BoundedCorrection"):
            anomaly = f"{router}__AnomalyBaseline__{head}"
            malben = f"{router}__MalBenContrast__{head}"
            a = final_map[anomaly] >= 0.5; b = final_map[malben] >= 0.5
            for scope, idx in [("pooled_2018_2021", np.ones(len(labels), bool))] + [(str(y), years == y) for y in YEARS]:
                ca = a[idx] == labels[idx]; cb = b[idx] == labels[idx]
                pairwise.append({
                    "scope": scope, "comparison": f"{malben} vs {anomaly}",
                    "malben_only_correct": int(np.sum(cb & ~ca)),
                    "anomaly_only_correct": int(np.sum(ca & ~cb)),
                    "net_malben_minus_anomaly": int(np.sum(cb & ~ca) - np.sum(ca & ~cb)),
                    "prediction_disagreement": float(np.mean(a[idx] != b[idx])),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "future_metrics.csv", metrics)
    write_csv(args.output / "future_sample_predictions.csv", sample)
    write_csv(args.output / "future_pairwise.csv", pairwise)
    pooled = [row for row in metrics if row["scope"] == "pooled_2018_2021"]
    annual = [row for row in metrics if row["scope"] == "annual"]
    macro = []
    for method in dict.fromkeys(row["method"] for row in annual):
        selected = [row for row in annual if row["method"] == method]
        macro.append({
            "method": method,
            **{f"mean_{name}": float(np.mean([float(row[name]) for row in selected]))
               for name in ("BA", "AUC", "F1", "fixed", "broken", "net")},
            "min_BA": float(np.min([float(row["BA"]) for row in selected])),
            "diagnostic_only": True,
        })
    write_csv(args.output / "future_pooled_metrics.csv", pooled)
    write_csv(args.output / "future_annual_metrics.csv", annual)
    write_csv(args.output / "future_macro_mean_metrics.csv", macro)
    overlap = []
    for scope, idx in [("pooled_2018_2021", np.ones(len(labels), bool))] + [(str(y), years == y) for y in YEARS]:
        left, right = routes["A25"][idx], routes["DI25"][idx]
        intersection = int(np.sum(left & right)); union = int(np.sum(left | right))
        overlap.append({
            "scope": scope, "A25_n": int(np.sum(left)), "DI25_n": int(np.sum(right)),
            "intersection_n": intersection, "union_n": union,
            "jaccard": float(intersection / max(union, 1)),
        })
    write_csv(args.output / "future_route_overlap.csv", overlap)
    lines = [
        "CARE-Droid frozen future 2(router) x 2(Stage-II representation) x 2(decision head) matrix v1", "",
        "All experts were trained on the fixed 799-APK 2017 source cohort only.",
        "A25 and DI25 are frozen equal-budget per-year routes. Future labels were unsealed only here.",
        "DirectStageII replaces p_stage1 on routed APKs; BoundedCorrection applies the frozen logit-space correction.",
        "2018-2021 are diagnostic-only and must not change the method.", "", "Pooled 2018-2021:",
    ]
    for row in pooled:
        lines.append(
            "{method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} "
            "FP={FP} FN={FN} fixed={fixed} broken={broken} net={net} routed={routed_count}/{n}".format(**row))
    lines.extend(["", "Macro mean across 2018-2021:"])
    for row in macro:
        lines.append(
            "{method}: mean_BA={mean_BA:.6f} min_BA={min_BA:.6f} "
            "mean_AUC={mean_AUC:.6f} mean_F1={mean_F1:.6f} mean_net={mean_net:.2f}".format(**row))
    lines.extend(["", "Annual results:"])
    for year in YEARS:
        lines.append(str(year))
        for row in annual:
            if int(row["year"]) == year:
                lines.append("  {method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} net={net}".format(**row))
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol = {
        "matrix": {"routers": ["A25", "DI25"], "representations": ["AnomalyBaseline", "MalBenContrast"],
                   "heads": ["DirectStageII", "BoundedCorrection"]},
        "budget": 0.25, "max_delta_logit": 2.25,
        "future_labels_used_for_training_routing_model_selection_or_ensemble": False,
        "future_diagnostic_only": True,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__": main()
