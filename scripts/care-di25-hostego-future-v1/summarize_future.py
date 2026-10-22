#!/usr/bin/env python3
"""Unseal labels only after M1 predictions freeze; report future diagnostics."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path("artifacts/care-di25-hostego-future-v1")
RESOLVED = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
FUTURE_INPUT = Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
M0_PATH = Path("artifacts/care-route-stage2-future-matrix-v1/arms/anomaly_bounded/future_expert_predictions.csv")
M1_PATH = ROOT / "m1_hostego_anomaly/future_m1_predictions_label_free.csv"
M1_PROTOCOL = ROOT / "m1_hostego_anomaly/protocol.json"
YEARS = (2018, 2019, 2020, 2021)


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_label(value: object) -> int:
    text = normal(value)
    if text in {"1", "1.0", "malware", "malicious", "true"}: return 1
    if text in {"0", "0.0", "benign", "goodware", "false"}: return 0
    raise RuntimeError("unknown post-hoc future label: " + repr(value))


def unique(rows: list[dict], expected=1989) -> dict[str, dict]:
    result = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id", "")))
        if not sha or sha in result:
            raise RuntimeError("missing/duplicate SHA: " + sha)
        result[sha] = row
    if len(result) != expected or len(rows) != expected:
        raise RuntimeError(f"cohort size changed: rows={len(rows)}, unique={len(result)}")
    return result


def metric(labels, probability, base, route, method, scope, year=""):
    labels = np.asarray(labels, dtype=int); probability = np.asarray(probability, dtype=float)
    base = np.asarray(base, dtype=float); route = np.asarray(route, dtype=bool)
    pred = probability >= .5; base_ok = (base >= .5) == labels; final_ok = pred == labels
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum(route & ~base_ok & final_ok)); broken = int(np.sum(route & base_ok & ~final_ok))
    return {
        "scope": scope, "year": year, "method": method, "n": len(labels),
        "routed": int(route.sum()), "routing_rate": float(route.mean()),
        "BA": float(balanced_accuracy_score(labels, pred)),
        "AUC": float(roc_auc_score(labels, probability)),
        "F1": float(f1_score(labels, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fixed": fixed, "broken": broken, "net": fixed-broken,
        "correction_precision": fixed/(fixed+broken) if fixed+broken else float("nan"),
        "diagnostic_only": True,
    }


def exact_p(n01: int, n10: int) -> float:
    n = n01+n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0*sum(math.comb(n, i) for i in range(k+1))/(2**n))


def main() -> None:
    protocol = json.loads(M1_PROTOCOL.read_text(encoding="utf-8"))
    if not protocol.get("source_only_training") or protocol.get("future_labels_loaded_by_model"):
        raise RuntimeError("M1 probabilities did not freeze under the label-free contract")
    resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
    frozen = read_csv(FUTURE_INPUT)
    future_index = unique(frozen)
    labels_file = Path(resolved["future_label_file_posthoc_only"])
    labeled = read_csv(labels_file)
    label_by_sha = {normal(row.get("sha256", row.get("sample_id", ""))):
                    parse_label(row.get("label", row.get("metadata_label"))) for row in labeled}
    if len(label_by_sha) != 1989 or set(label_by_sha) != set(future_index):
        raise RuntimeError("post-hoc future label cohort changed")
    m0, m1 = unique(read_csv(M0_PATH)), unique(read_csv(M1_PATH))
    if set(m0) != set(future_index) or set(m1) != set(future_index):
        raise RuntimeError("M0/M1 future prediction cohort differs")

    sha = [normal(row["sha256"]) for row in frozen]
    years = np.asarray([int(row["year"]) for row in frozen], dtype=int)
    labels = np.asarray([label_by_sha[value] for value in sha], dtype=int)
    base = np.asarray([float(future_index[value]["p_stage1"]) for value in sha])
    route = np.asarray([int(float(future_index[value]["route_DI"])) == 1 for value in sha])
    p0 = np.asarray([float(m0[value]["p_expert"]) for value in sha])
    p1 = np.asarray([float(m1[value]["p_expert_bounded"]) for value in sha])
    p1_direct = np.asarray([float(m1[value]["p_expert_direct_same_correction_head"]) for value in sha])
    if not all(np.isfinite(values).all() for values in (base, p0, p1, p1_direct)):
        raise RuntimeError("future probability input contains non-finite values")
    final = {
        "Global24": base,
        "DI25__M0_Host_Anomaly_Bounded": np.where(route, p0, base),
        "DI25__M1_HostEgo_Anomaly_Bounded": np.where(route, p1, base),
        "DI25__M1_HostEgo_Anomaly_DirectSecondary": np.where(route, p1_direct, base),
    }
    metrics = []
    for method, probability in final.items():
        use_route = np.zeros(len(route), bool) if method == "Global24" else route
        for year in YEARS:
            index = years == year
            metrics.append(metric(labels[index], probability[index], base[index], use_route[index], method, "annual", year))
        metrics.append(metric(labels, probability, base, use_route, method, "pooled_2018_2021"))

    paired, sample = [], []
    for scope, index in [(str(year), years == year) for year in YEARS] + [("pooled_2018_2021", np.ones(len(years), bool))]:
        c0 = (final["DI25__M0_Host_Anomaly_Bounded"][index] >= .5) == labels[index]
        c1 = (final["DI25__M1_HostEgo_Anomaly_Bounded"][index] >= .5) == labels[index]
        n01 = int(np.sum(~c0 & c1))  # M0 wrong, M1 correct
        n10 = int(np.sum(c0 & ~c1))  # M0 correct, M1 wrong
        paired.append({"scope": scope, "comparison": "M0_Bounded_vs_M1_Bounded",
                       "n01_M0_wrong_M1_correct": n01,
                       "n10_M0_correct_M1_wrong": n10,
                       "net_M1_minus_M0": n01-n10,
                       "exact_McNemar_two_sided_p_value": exact_p(n01, n10)})
    for i, value in enumerate(sha):
        c0 = int((final["DI25__M0_Host_Anomaly_Bounded"][i] >= .5) == bool(labels[i]))
        c1 = int((final["DI25__M1_HostEgo_Anomaly_Bounded"][i] >= .5) == bool(labels[i]))
        status = "same_correct" if c0 and c1 else "same_wrong" if not c0 and not c1 else \
                 "M0_wrong_M1_correct" if not c0 and c1 else "M0_correct_M1_wrong"
        sample.append({
            "sha256": value, "year": int(years[i]), "label": int(labels[i]),
            "p_stage1": float(base[i]), "route_DI25": int(route[i]),
            "p_expert_M0_bounded": float(p0[i]), "p_expert_M1_bounded": float(p1[i]),
            "p_expert_M1_direct_secondary": float(p1_direct[i]),
            "p_final_M0_bounded": float(final["DI25__M0_Host_Anomaly_Bounded"][i]),
            "p_final_M1_bounded": float(final["DI25__M1_HostEgo_Anomaly_Bounded"][i]),
            "M0_to_M1_status": status,
        })

    usage = []
    for year in YEARS:
        rows = [m1[value] for value in sha if int(m1[value]["year"]) == year]
        routed_values = [m1[sha[i]] for i in range(len(sha)) if years[i] == year and route[i]]
        usage.append({
            "year": year, "apks": len(rows), "routed_apks": len(routed_values),
            "total_real_hosts": sum(int(row["real_host_count"]) for row in rows),
            "total_nontrivial_ego_hosts": sum(int(row["nontrivial_ego_count"]) for row in rows),
            "routed_real_hosts": sum(int(row["real_host_count"]) for row in routed_values),
            "routed_nontrivial_ego_hosts": sum(int(row["nontrivial_ego_count"]) for row in routed_values),
            "routed_apks_with_any_ego": sum(int(row["has_any_nontrivial_ego"]) for row in routed_values),
            "routed_apk_ego_coverage": sum(int(row["has_any_nontrivial_ego"]) for row in routed_values)/max(len(routed_values), 1),
        })

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "future_metrics.csv", metrics)
    write_csv(results / "future_annual_metrics.csv", [row for row in metrics if row["scope"] == "annual"])
    write_csv(results / "future_pooled_metrics.csv", [row for row in metrics if row["scope"] == "pooled_2018_2021"])
    write_csv(results / "future_M0_M1_paired.csv", paired)
    write_csv(results / "future_sample_predictions.csv", sample)
    write_csv(results / "future_ego_usage.csv", usage)
    pooled = [row for row in metrics if row["scope"] == "pooled_2018_2021"]
    annual = [row for row in metrics if row["scope"] == "annual"]
    lines = [
        "CARE-Droid frozen DI25 Host/Ego future diagnostic v1", "",
        "M0 future probabilities are reused from the completed source-only future matrix; M0 was not retrained.",
        "M1 was fitted on the fixed 799-APK 2017 cohort only. Future labels were unsealed only by this summarizer.",
        "Future years are diagnostic-only and must not change routing, representation, bound, or head.", "",
        "Pooled 2018-2021:",
    ]
    for row in pooled:
        lines.append("{method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} FP={FP} FN={FN} fixed={fixed} broken={broken} net={net}".format(**row))
    lines.extend(["", "Annual:"])
    for year in YEARS:
        lines.append(str(year))
        for row in annual:
            if int(row["year"]) == year:
                lines.append(
                    "  {method}: routed={routed}/{n} BA={BA:.6f} AUC={AUC:.6f} "
                    "F1={F1:.6f} FP={FP} FN={FN} fixed={fixed} broken={broken} net={net}".format(**row)
                )
    lines.extend(["", "Paired M0 vs M1:"])
    for row in paired:
        lines.append("  {scope}: n01={n01_M0_wrong_M1_correct} n10={n10_M0_correct_M1_wrong} net={net_M1_minus_M0} p={exact_McNemar_two_sided_p_value:.6f}".format(**row))
    (results / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "unseal_audit.json").write_text(json.dumps({
        "model_protocol": str(M1_PROTOCOL.resolve()),
        "probabilities_frozen_before_label_access": True,
        "future_labels_used_for_training_routing_selection_or_early_stopping": False,
        "future_labels_unsealed_by": "summarize_future.py only",
        "future_diagnostic_only": True,
        "year_counts": dict(Counter(int(value) for value in years)),
        "route_counts": {str(year): int(np.sum(route & (years == year))) for year in YEARS},
    }, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
