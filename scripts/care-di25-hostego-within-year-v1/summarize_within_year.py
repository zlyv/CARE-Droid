#!/usr/bin/env python3
"""Compare frozen same-year M0 and M1 OOF predictions under identical DI25 routes."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path("artifacts/care-di25-hostego-within-year-v1")
ROUTER = Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
M0 = Path("artifacts/care-route-stage2-within-year-matrix-v1/arms/anomaly_bounded/within_year_expert_predictions.csv")
M1 = ROOT / "m1_hostego_anomaly_bounded/within_year_m1_predictions.csv"
M1_PROTOCOL = M1.parent / "protocol.json"
TEMPORAL = Path("artifacts/care-di25-hostego-future-v1/results/future_annual_metrics.csv")
YEARS = (2018, 2019, 2020, 2021)


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index(rows: list[dict], name: str) -> dict[str, dict]:
    result = {normal(row["sha256"]): row for row in rows}
    if len(rows) != 1989 or len(result) != 1989:
        raise RuntimeError(f"{name} is not the exact 1989-APK cohort")
    return result


def metric(y, p, base, route, method, scope, year="") -> dict:
    y = np.asarray(y, int); p = np.asarray(p, float); base = np.asarray(base, float); route = np.asarray(route, bool)
    prediction = p >= 0.5
    base_correct = (base >= 0.5) == y
    final_correct = prediction == y
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    fixed = int(np.sum(route & ~base_correct & final_correct))
    broken = int(np.sum(route & base_correct & ~final_correct))
    return {
        "scope": scope, "year": year, "method": method, "n": len(y),
        "routed": int(route.sum()), "routing_rate": float(route.mean()),
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, p)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "diagnostic_only": True,
    }


def exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    protocol = json.loads(M1_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "independent per-year five-fold OOF" or protocol.get("parameter_count") != 34613:
        raise RuntimeError("M1 same-year protocol is not frozen")
    router_rows = read_csv(ROUTER)
    ids = [normal(row["sha256"]) for row in router_rows]
    if len(ids) != 1989 or len(set(ids)) != 1989:
        raise RuntimeError("router cohort changed")
    m0, m1 = index(read_csv(M0), "M0"), index(read_csv(M1), "M1")
    if set(m0) != set(ids) or set(m1) != set(ids):
        raise RuntimeError("M0/M1/router cohort alignment failed")

    years = np.asarray([int(row["year"]) for row in router_rows], int)
    labels = np.asarray([int(row["label"]) for row in router_rows], int)
    base = np.asarray([float(row["p_stage1"]) for row in router_rows], float)
    route = np.asarray([int(float(row["route_DI"])) == 1 for row in router_rows], bool)
    p0 = np.asarray([float(m0[sha]["p_expert"]) for sha in ids], float)
    p1 = np.asarray([float(m1[sha]["p_expert_bounded"]) for sha in ids], float)
    pd = np.asarray([float(m1[sha]["p_expert_direct_secondary"]) for sha in ids], float)
    if not all(np.isfinite(value).all() for value in (base, p0, p1, pd)):
        raise RuntimeError("same-year probability contains non-finite values")
    final = {
        "Global24": base,
        "DI25__M0_Host_Anomaly_Bounded": np.where(route, p0, base),
        "DI25__M1_HostEgo_Anomaly_Bounded": np.where(route, p1, base),
        "DI25__M1_HostEgo_Anomaly_DirectSecondary": np.where(route, pd, base),
    }

    metrics: list[dict] = []
    for method, probability in final.items():
        method_route = np.zeros(len(route), bool) if method == "Global24" else route
        for year in YEARS:
            selected = years == year
            metrics.append(metric(labels[selected], probability[selected], base[selected], method_route[selected], method, "annual", year))
        metrics.append(metric(labels, probability, base, method_route, method, "pooled_2018_2021"))

    paired = []
    for scope, selected in [(str(year), years == year) for year in YEARS] + [("pooled_2018_2021", np.ones(len(years), bool))]:
        c0 = (final["DI25__M0_Host_Anomaly_Bounded"][selected] >= 0.5) == labels[selected]
        c1 = (final["DI25__M1_HostEgo_Anomaly_Bounded"][selected] >= 0.5) == labels[selected]
        n01 = int(np.sum(~c0 & c1))
        n10 = int(np.sum(c0 & ~c1))
        paired.append({
            "scope": scope, "comparison": "M0_Bounded_vs_M1_Bounded",
            "n01_M0_wrong_M1_correct": n01, "n10_M0_correct_M1_wrong": n10,
            "net_M1_minus_M0": n01 - n10,
            "exact_McNemar_two_sided_p_value": exact_p(n01, n10),
        })

    sample = []
    final_m0 = final["DI25__M0_Host_Anomaly_Bounded"]
    final_m1 = final["DI25__M1_HostEgo_Anomaly_Bounded"]
    for i, sha in enumerate(ids):
        c0 = bool((final_m0[i] >= 0.5) == labels[i])
        c1 = bool((final_m1[i] >= 0.5) == labels[i])
        status = (
            "same_correct" if c0 and c1 else
            "same_wrong" if not c0 and not c1 else
            "M0_wrong_M1_correct" if not c0 and c1 else
            "M0_correct_M1_wrong"
        )
        sample.append({
            "sha256": sha, "year": int(years[i]), "fold": int(router_rows[i]["fold"]),
            "label": int(labels[i]), "p_stage1": float(base[i]), "route_DI25": int(route[i]),
            "p_expert_M0_bounded": float(p0[i]), "p_expert_M1_bounded": float(p1[i]),
            "p_expert_M1_direct_secondary": float(pd[i]),
            "p_final_M0_bounded": float(final_m0[i]), "p_final_M1_bounded": float(final_m1[i]),
            "M0_correct": int(c0), "M1_correct": int(c1), "M0_to_M1_status": status,
        })

    usage = []
    for year in YEARS:
        chosen = np.flatnonzero((years == year) & route)
        rows = [m1[ids[i]] for i in chosen]
        usage.append({
            "year": year, "routed_apks": len(rows),
            "routed_real_hosts": sum(int(row["real_host_count"]) for row in rows),
            "routed_nontrivial_ego_hosts": sum(int(row["nontrivial_ego_count"]) for row in rows),
            "routed_apks_with_any_nontrivial_ego": sum(int(row["has_any_nontrivial_ego"]) for row in rows),
            "routed_apk_ego_coverage": sum(int(row["has_any_nontrivial_ego"]) for row in rows) / max(len(rows), 1),
        })

    results = ROOT / "results"
    annual = [row for row in metrics if row["scope"] == "annual"]
    pooled = [row for row in metrics if row["scope"] == "pooled_2018_2021"]
    write_csv(results / "within_year_metrics.csv", metrics)
    write_csv(results / "within_year_annual_metrics.csv", annual)
    write_csv(results / "within_year_pooled_metrics.csv", pooled)
    write_csv(results / "within_year_M0_M1_paired.csv", paired)
    write_csv(results / "within_year_sample_predictions.csv", sample)
    write_csv(results / "within_year_ego_usage.csv", usage)

    if TEMPORAL.is_file():
        temporal = read_csv(TEMPORAL)
        temporal_index = {(row["method"], int(row["year"])): row for row in temporal}
        gap = []
        for row in annual:
            if row["method"] not in {"Global24", "DI25__M0_Host_Anomaly_Bounded", "DI25__M1_HostEgo_Anomaly_Bounded"}:
                continue
            key = (row["method"], int(row["year"]))
            if key not in temporal_index:
                continue
            old = temporal_index[key]
            gap.append({
                "method": row["method"], "year": row["year"],
                "within_year_BA": row["BA"], "temporal_BA": old["BA"],
                "within_minus_temporal_BA": float(row["BA"]) - float(old["BA"]),
                "within_year_AUC": row["AUC"], "temporal_AUC": old["AUC"],
                "within_minus_temporal_AUC": float(row["AUC"]) - float(old["AUC"]),
                "within_year_F1": row["F1"], "temporal_F1": old["F1"],
                "within_minus_temporal_F1": float(row["F1"]) - float(old["F1"]),
            })
        if gap:
            write_csv(results / "within_vs_temporal_gap.csv", gap)

    lines = [
        "CARE-Droid frozen DI25 Host/Ego independent within-year diagnostic v1", "",
        "Each year is trained and evaluated independently with strict five-fold OOF.",
        "M0 is reused from the completed same-year matrix; only M1 is newly trained.",
        "These are same-distribution diagnostics, not temporal-generalization results.",
        "Budget=25%; max_delta(logit)=2.25; representation/head remain source-frozen.", "",
        "Pooled same-year OOF:",
    ]
    for row in pooled:
        lines.append(
            "{method}: BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} FP={FP} FN={FN} "
            "fixed={fixed} broken={broken} net={net}".format(**row))
    lines.extend(["", "Annual same-year OOF:"])
    for year in YEARS:
        lines.append(str(year))
        for row in annual:
            if int(row["year"]) == year:
                lines.append(
                    "  {method}: routed={routed}/{n} BA={BA:.6f} AUC={AUC:.6f} F1={F1:.6f} "
                    "FP={FP} FN={FN} fixed={fixed} broken={broken} net={net}".format(**row))
    lines.extend(["", "Paired M0 vs M1:"])
    for row in paired:
        lines.append(
            "  {scope}: n01={n01_M0_wrong_M1_correct} n10={n10_M0_correct_M1_wrong} "
            "net={net_M1_minus_M0} p={exact_McNemar_two_sided_p_value:.6f}".format(**row))
    lines.extend(["", "Interpretation boundary:",
                  "Do not describe these same-year OOF results as future-year, zero-day, or temporal performance."])
    (results / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
