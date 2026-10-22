#!/usr/bin/env python3
"""Unseal future labels after prediction freeze and summarize diagnostics."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from freeze_final_method import ROOT, blocked, read_csv, sha256_file


OUT = ROOT / "diagnostic_future"
PRE = OUT / "future_protocol_preunseal.json"
PRED = OUT / "future_predictions_label_free.csv"
RESOLVED = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
YEARS = (2018, 2019, 2020, 2021)
DISCLAIMER = "These 2018–2021 sets were previously inspected during method development. Therefore, the following experiment is diagnostic-only and is not treated as an untouched final temporal evaluation."


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def parse_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "malware", "malicious"}: return 1
    if text in {"0", "0.0", "benign", "goodware"}: return 0
    blocked(f"unknown label during post-hoc unseal: {value}")
    return -1


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if not n: return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / (2 ** n))


def metrics(y: np.ndarray, p: np.ndarray, base: np.ndarray, route: np.ndarray, scope: str, year: object, method: str) -> dict:
    pred, base_pred = p >= .5, base >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum(route & (base_pred != y) & (pred == y)))
    broken = int(np.sum(route & (base_pred == y) & (pred != y)))
    return {
        "scope": scope, "year": year, "method": method, "n": len(y), "routed": int(route.sum()), "routed_n": int(route.sum()),
        "routing_rate": float(route.mean()), "routing_ratio": float(route.mean()),
        "BA": float(balanced_accuracy_score(y, pred)), "AUC": float(roc_auc_score(y, p)), "F1": float(f1_score(y, pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "fixed": fixed, "broken": broken, "net": fixed-broken,
        "correction_precision": fixed/(fixed+broken) if fixed+broken else float("nan"), "diagnostic_only": True,
    }


def main() -> None:
    started = time.perf_counter()
    if not PRE.is_file() or not PRED.is_file() or not RESOLVED.is_file():
        blocked("label-free future predictions/protocol must be frozen before unsealing")
    protocol = json.loads(PRE.read_text(encoding="utf-8"))
    if not protocol.get("probabilities_frozen_before_label_access") or sha256_file(PRED) != protocol.get("label_free_predictions_sha256"):
        blocked("label-free future prediction checksum changed before unsealing")
    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
    labels_path = Path(resolved["future_label_file_posthoc_only"])
    if not labels_path.is_file():
        blocked(f"post-hoc future label file missing: {labels_path}")
    labels = {}
    for row in read_csv(labels_path):
        sha = normal(row.get("sha256", row.get("sample_id")))
        if sha in labels: blocked(f"duplicate future label: {sha}")
        labels[sha] = parse_label(row.get("label", row.get("metadata_label")))
    rows = read_csv(PRED)
    ids = [normal(row["sha256"]) for row in rows]
    if len(rows) != 1989 or len(set(ids)) != 1989 or set(ids) != set(labels):
        blocked(f"label unseal cohort mismatch predictions={len(rows)} labels={len(labels)}")
    y = np.asarray([labels[sha] for sha in ids], int)
    years = np.asarray([int(row["year"]) for row in rows], int)
    base = np.asarray([float(row["p_stage1"]) for row in rows], float)
    route = np.asarray([int(row["route_DI25"]) == 1 for row in rows], bool)
    p0e = np.asarray([float(row["p_M0_Host_Anomaly_Bounded"]) for row in rows], float)
    p1e = np.asarray([float(row["p_M1_HostEgo_Anomaly_Bounded"]) for row in rows], float)
    final = {"Global24": base, "M0_Host_Anomaly_Bounded": np.where(route, p0e, base), "M1_HostEgo_Anomaly_Bounded": np.where(route, p1e, base)}
    metric_rows = []
    for method, probability in final.items():
        applied = np.zeros_like(route) if method == "Global24" else route
        for year in YEARS:
            take = years == year
            metric_rows.append(metrics(y[take], probability[take], base[take], applied[take], "annual", year, method))
        metric_rows.append(metrics(y, probability, base, applied, "pooled_2018_2021", "2018-2021", method))
    write_csv(OUT / "future_metrics_by_year.csv", [row for row in metric_rows if row["scope"] == "annual"])
    write_csv(OUT / "future_metrics_pooled.csv", [row for row in metric_rows if row["scope"] == "pooled_2018_2021"])
    write_csv(OUT / "future_fixed_broken.csv", [{key: row[key] for key in ("scope", "year", "method", "n", "routed", "fixed", "broken", "net", "correction_precision")} for row in metric_rows])

    labeled_rows, paired = [], []
    for i, row in enumerate(rows):
        item = dict(row); item["label"] = int(y[i]); item["p_final_M0"] = float(final["M0_Host_Anomaly_Bounded"][i]); item["p_final_M1"] = float(final["M1_HostEgo_Anomaly_Bounded"][i])
        labeled_rows.append(item)
    write_csv(OUT / "future_predictions.csv", labeled_rows)
    for scope, take in [(str(year), years == year) for year in YEARS] + [("pooled_2018_2021", np.ones(len(y), bool))]:
        c0 = (final["M0_Host_Anomaly_Bounded"][take] >= .5) == y[take]
        c1 = (final["M1_HostEgo_Anomaly_Bounded"][take] >= .5) == y[take]
        n01, n10 = int(np.sum(~c0 & c1)), int(np.sum(c0 & ~c1))
        paired.append({"scope": scope, "n01_M0_wrong_M1_correct": n01, "n10_M0_correct_M1_wrong": n10, "net_M1_minus_M0": n01-n10, "exact_mcnemar_two_sided_p": exact_p(n01, n10)})
    write_csv(OUT / "future_M0_M1_paired.csv", paired)

    usage = []
    for label, take in [(str(year), years == year) for year in YEARS] + [("pooled_2018_2021", np.ones(len(y), bool))]:
        selected = np.flatnonzero(take); routed = selected[route[selected]]
        hosts = np.asarray([int(rows[i]["M1_real_host_count"]) for i in routed], int)
        egos = np.asarray([int(rows[i]["M1_nontrivial_ego_count"]) for i in routed], int)
        usage.append({
            "scope": label, "apks": int(np.sum(take)), "routed_apks": len(routed),
            "routed_apks_with_any_nontrivial_ego": int(np.sum(egos > 0)), "routed_apk_ego_coverage": float(np.mean(egos > 0)),
            "routed_real_hosts": int(hosts.sum()), "routed_nontrivial_ego_hosts": int(egos.sum()),
            "host_ego_usage_ratio": float(egos.sum()/hosts.sum()) if hosts.sum() else 0.0,
            "mean_hosts_per_routed_apk": float(np.mean(hosts)), "median_hosts_per_routed_apk": float(np.median(hosts)),
            "mean_nontrivial_ego_hosts_per_routed_apk": float(np.mean(egos)),
        })
    write_csv(OUT / "future_ego_usage.csv", usage)

    by_key = {(row["scope"], str(row["year"]), row["method"]): row for row in metric_rows}
    deltas = []
    for scope, year in [("annual", str(yv)) for yv in YEARS] + [("pooled_2018_2021", "2018-2021")]:
        for left, right in (("Global24", "M0_Host_Anomaly_Bounded"), ("Global24", "M1_HostEgo_Anomaly_Bounded"), ("M0_Host_Anomaly_Bounded", "M1_HostEgo_Anomaly_Bounded")):
            a, b = by_key[(scope, year, left)], by_key[(scope, year, right)]
            deltas.append({"scope": scope, "year": year, "comparison": f"{right} minus {left}", **{f"delta_{key}": float(b[key])-float(a[key]) for key in ("BA", "AUC", "F1", "fixed", "broken", "net")}})
    write_csv(OUT / "future_method_deltas.csv", deltas)

    final_protocol = dict(protocol)
    final_protocol.update({
        "future_labels_unsealed_only_by": "summarize_temporal_diagnostic.py",
        "label_file": str(labels_path), "label_file_sha256": sha256_file(labels_path),
        "posthoc_summary_seconds": time.perf_counter()-started,
        "future_labels_used_for_training_routing_scaling_anomaly_fitting_early_stopping_threshold_or_model_selection": False,
        "diagnostic_only": True,
    })
    (OUT / "future_protocol.json").write_text(json.dumps(final_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [DISCLAIMER, "", "CARE-Droid frozen temporal diagnostic", "", "No 2018-2021 label was available to training, routing, scaling, anomaly fitting, early stopping, threshold selection or model selection.", ""]
    for scope, year in [("annual", str(v)) for v in YEARS] + [("pooled_2018_2021", "2018-2021")]:
        lines.append(year)
        for method in final:
            row = by_key[(scope, year, method)]
            lines.append(f"  {method}: routed={row['routed']}/{row['n']} BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
        pair = next(row for row in paired if row["scope"] == (year if year != "2018-2021" else "pooled_2018_2021"))
        lines.append(f"  M0 vs M1 exact McNemar: n01={pair['n01_M0_wrong_M1_correct']} n10={pair['n10_M0_correct_M1_wrong']} p={pair['exact_mcnemar_two_sided_p']:.6f}")
        if scope == "annual":
            m0 = by_key[(scope, year, "M0_Host_Anomaly_Bounded")]
            m1 = by_key[(scope, year, "M1_HostEgo_Anomaly_Bounded")]
            eu = next(row for row in usage if row["scope"] == year)
            print("[FUTURE DIAGNOSTIC]", flush=True)
            print(f"year={year}", flush=True); print(f"n={m1['n']}", flush=True)
            print(f"Global24 BA={by_key[(scope, year, 'Global24')]['BA']:.9f}", flush=True)
            print(f"M0 BA={m0['BA']:.9f}", flush=True); print(f"M1 BA={m1['BA']:.9f}", flush=True)
            print(f"M1-M0 delta={m1['BA']-m0['BA']:+.9f}", flush=True)
            print(f"fixed={m1['fixed']}", flush=True); print(f"broken={m1['broken']}", flush=True); print(f"net={m1['net']}", flush=True)
            print(f"ego_coverage={eu['routed_apk_ego_coverage']:.9f}", flush=True)
            print(f"elapsed={time.perf_counter()-started:.3f}", flush=True)
    lines.extend(["", "Interpretation boundary: these years are diagnostic only; none of these results may revise the frozen method."])
    (OUT / "TEMPORAL_DIAGNOSTIC_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
