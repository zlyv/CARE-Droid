#!/usr/bin/env python3
"""Assemble bounded/direct source metrics, integrity audits and paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

from experiment_contract import MAX_DELTA, read_csv, write_csv
from run_care_di25_hostego_v1 import ARMS


def exact_mcnemar(a_correct, b_correct):
    n01 = int(np.sum(a_correct & ~b_correct)); n10 = int(np.sum(~a_correct & b_correct))
    n = n01 + n10
    statistic = ((abs(n01 - n10) - 1) ** 2 / n) if n else 0.0
    if not n:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return n01, n10, statistic, p


def metrics(name, y, p, base, route, parameter_count, ego_usage, inference_per_apk):
    pred = p >= .5; base_pred = base >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fixed = int(np.sum((base_pred != y) & (pred == y)))
    broken = int(np.sum((base_pred == y) & (pred != y)))
    routed = int(np.sum(route)); changed = fixed + broken
    representation, evidence, head = name.split("__")[-3:]
    return {
        "method": name, "router": "DI25", "local_representation": representation,
        "evidence": evidence, "decision_head": head, "n": len(y), "routed": routed,
        "routing_rate": routed / len(y), "BA": balanced_accuracy_score(y, pred),
        "AUC": roc_auc_score(y, p), "F1": f1_score(y, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "correction_precision": fixed / changed if changed else float("nan"),
        "net_per_routed": (fixed-broken)/routed, "fixed_per_routed": fixed/routed,
        "broken_per_routed": broken/routed, "parameter_count": parameter_count,
        "ego_usage_rate": ego_usage,
        "stage2_seconds_per_routed_apk": inference_per_apk,
    }


def find_method_columns(rows):
    prob = [key for key in rows[0] if key.endswith("__prob") and key.startswith("DI25__")]
    if len(prob) != 1:
        raise RuntimeError("expected one arm probability column, got " + repr(prob))
    prefix = prob[0][:-6]
    direct = prefix + "__direct_stage2_probability"
    if direct not in rows[0]:
        raise RuntimeError("missing same-model direct Stage-II probability: " + direct)
    return prefix, prob[0], direct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/care-di25-hostego-v1"))
    args = p.parse_args()
    data = {}
    route_rows = []
    parameter_rows = []
    efficiency_rows = []
    for arm, (representation, evidence, _) in ARMS.items():
        directory = args.root / "source" / f"{arm}_{representation}_{evidence}_bounded"
        rows = read_csv(directory / "source_predictions.csv")
        prefix, prob_col, direct_col = find_method_columns(rows)
        sha = [r["sha256"].lower() for r in rows]
        y = np.asarray([int(r["label"]) for r in rows])
        base = np.asarray([float(r["Global24__prob"]) for r in rows])
        route = np.asarray([int(r["A25__routed"]) == 1 for r in rows])
        bounded = np.asarray([float(r[prob_col]) for r in rows])
        direct_expert = np.asarray([float(r[direct_col]) for r in rows])
        direct = np.where(route, direct_expert, base)
        write_csv(args.root / "direct_ablation" / f"{arm}_{representation}_{evidence}_direct" / "source_predictions.csv", [
            {"sha256": row["sha256"], "label": row["label"], "fold": row["fold"],
             "route_DI25": int(route[i]), "Global24__prob": float(base[i]),
             "direct_stage2_expert_probability": float(direct_expert[i]),
             "final_probability": float(direct[i])}
            for i, row in enumerate(rows)
        ])
        fit = read_csv(directory / "fit_audit.csv")
        params = sorted({int(float(r["parameter_n"])) for r in fit})
        if len(params) != 1:
            raise RuntimeError("parameter count changed across folds/seeds")
        arch = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
        wall = float(arch.get("wall_seconds_including_training", float("nan")))
        inference_per_apk = float(arch.get("model_inference_seconds_per_model_apk", float("nan")))
        feature_load = float(arch.get("feature_cache_loading_seconds", float("nan")))
        ego_usage = 0.0
        ego_diag = args.root / f"audits/ego_apk_diagnostic_{arm}.csv"
        if ego_diag.is_file():
            diag = read_csv(ego_diag)
            routed_diag = [r for r in diag if int(r["routed"]) == 1]
            numerator = sum(int(r["nontrivial_ego_count"]) for r in routed_diag)
            denominator = sum(max(1, int(r["host_count"])) for r in routed_diag)
            ego_usage = numerator / denominator if denominator else 0.0
        data[arm] = dict(sha=sha, y=y, base=base, route=route, bounded=bounded, direct=direct,
                         prefix=prefix, params=params[0], ego_usage=ego_usage, wall=wall,
                         inference_per_apk=inference_per_apk, feature_load=feature_load)
        parameter_rows.append({"arm": arm, "parameter_count": params[0]})
        efficiency_rows.append({"arm": arm, "total_apks": len(rows), "routed_apks": int(route.sum()),
                                "routing_rate": float(route.mean()), "stage2_invocation_rate": float(route.mean()),
                                "ego_usage_rate": ego_usage, "feature_cache_loading_seconds": feature_load,
                                "model_inference_seconds_per_routed_apk": inference_per_apk,
                                "wall_seconds_including_training": wall,
                                "timing_note": "cache loading and model inference are separate; one-time APK extraction excluded"})
    base_sha = data["M0"]["sha"]
    base_route = data["M0"]["route"]
    asset_root = Path("artifacts/care-source-local-assets-v2")
    with np.load(asset_root / "host_regions_2017.npz", allow_pickle=False) as h, np.load(asset_root / "ego_1hop_regions_2017.npz", allow_pickle=False) as e:
        if h["sha256"].astype(str).tolist() != base_sha:
            raise RuntimeError("efficiency cache lineage differs")
        real_hosts = np.asarray(h["real_host_count"], dtype=int)
        eoff = np.asarray(e["ego_offsets"], dtype=int); status = np.asarray(e["ego_status"], dtype=int)
        nontrivial_per_apk = np.asarray([np.sum(status[eoff[i]:eoff[i+1]] == 1) for i in range(len(base_sha))], dtype=int)
    routed_hosts = real_hosts[base_route]
    routed_egos = nontrivial_per_apk[base_route]
    efficiency_common = {
        "total_hosts": int(np.sum(routed_hosts)),
        "mean_hosts_per_routed_apk": float(np.mean(routed_hosts)),
        "median_hosts_per_routed_apk": float(np.median(routed_hosts)),
        "nontrivial_ego_hosts": int(np.sum(routed_egos)),
        "host_only_no_neighbor": int(np.sum(routed_hosts) - np.sum(routed_egos)),
        "mean_nontrivial_egos_per_routed_apk": float(np.mean(routed_egos)),
    }
    for row in efficiency_rows:
        row.update(efficiency_common)
    for i, sha in enumerate(base_sha):
        item = {"sha256": sha, "fold": read_csv(args.root / "source/M0_host_anomaly_bounded/source_predictions.csv")[i]["fold"]}
        for arm in ARMS:
            if data[arm]["sha"] != base_sha:
                raise RuntimeError("arm source order differs")
            item[f"route_DI25_{arm}"] = int(data[arm]["route"][i])
        route_rows.append(item)
    route_difference_count = sum(len({r[f"route_DI25_{arm}"] for arm in ARMS}) != 1 for r in route_rows)
    write_csv(args.root / "audits/route_integrity.csv", route_rows)
    if route_difference_count:
        raise RuntimeError(f"route_difference_count={route_difference_count}")

    host_params = data["M0"]["params"]; ego_params = data["M1"]["params"]
    parameter_rows.extend([
        {"metric": "host_only_parameter_count", "value": host_params},
        {"metric": "host_ego_parameter_count", "value": ego_params},
        {"metric": "extra_parameter_count", "value": ego_params-host_params},
        {"metric": "relative_parameter_increase", "value": (ego_params-host_params)/host_params},
    ])
    write_csv(args.root / "audits/parameter_count.csv", parameter_rows)
    write_csv(args.root / "audits/efficiency.csv", efficiency_rows)

    summary = []
    probabilities = {}
    for arm in ARMS:
        d = data[arm]
        bounded_name = d["prefix"]
        direct_name = bounded_name.replace("__Bounded", "__Direct")
        summary.append(metrics(bounded_name, d["y"], d["bounded"], d["base"], d["route"], d["params"], d["ego_usage"], d["inference_per_apk"]))
        summary.append(metrics(direct_name, d["y"], d["direct"], d["base"], d["route"], d["params"], d["ego_usage"], d["inference_per_apk"]))
        probabilities[arm+"_bounded"] = d["bounded"]
        probabilities[arm+"_direct"] = d["direct"]
    write_csv(args.root / "results/source_summary.csv", summary)
    write_csv(args.root / "results/source_comparison.csv", summary)
    write_csv(args.root / "results/direct_vs_bounded.csv", [r for r in summary if r["decision_head"] in ("Direct", "Bounded")])
    write_csv(args.root / "results/host_vs_ego.csv", [r for r in summary if r["decision_head"] == "Bounded"])

    comparisons = [("M0 vs M1", "M0_bounded", "M1_bounded"), ("M2 vs M3", "M2_bounded", "M3_bounded"),
                   ("M0 Direct vs M0 Bounded", "M0_direct", "M0_bounded"),
                   ("M1 Direct vs M1 Bounded", "M1_direct", "M1_bounded")]
    mc = []
    y = data["M0"]["y"]
    for label, left, right in comparisons:
        values = exact_mcnemar((probabilities[left] >= .5) == y, (probabilities[right] >= .5) == y)
        mc.append({"comparison": label, "n01": values[0], "n10": values[1], "statistic": values[2], "p_value": values[3]})
    write_csv(args.root / "results/mcnemar_results.csv", mc)
    # Consolidate the per-arm Ego diagnostics and add the required paired APK
    # prediction fields after all four source arms are available.
    all_host, all_fallback, all_apk = [], [], []
    for ego_arm, host_arm in (("M1", "M0"), ("M3", "M2")):
        hp = args.root / f"audits/ego_host_diagnostic_{ego_arm}.csv"
        fp = args.root / f"audits/ego_fallback_audit_{ego_arm}.csv"
        ap = args.root / f"audits/ego_apk_diagnostic_{ego_arm}.csv"
        all_host.extend(read_csv(hp)); all_fallback.extend(read_csv(fp))
        host_index = {sha: i for i, sha in enumerate(data[host_arm]["sha"])}
        ego_index = {sha: i for i, sha in enumerate(data[ego_arm]["sha"])}
        for row in read_csv(ap):
            item = dict(row)
            sha = row["sha256"].lower(); hi = host_index[sha]; ei = ego_index[sha]
            item["prediction_hostonly"] = float(data[host_arm]["bounded"][hi])
            item["prediction_hostego"] = float(data[ego_arm]["bounded"][ei])
            item["prediction_changed"] = int((item["prediction_hostonly"] >= .5) != (item["prediction_hostego"] >= .5))
            all_apk.append(item)
    write_csv(args.root / "audits/ego_host_diagnostic.csv", all_host)
    write_csv(args.root / "audits/ego_fallback_audit.csv", all_fallback)
    write_csv(args.root / "audits/ego_apk_diagnostic.csv", all_apk)
    lines = ["CARE-Droid frozen DI25 Host/Ego source ablation v1", "", "Strict 2017 five-fold OOF; no future labels are loaded.",
             f"router: DI25; routed=200/799; route_difference_count={route_difference_count}",
             f"max_delta_logit={MAX_DELTA}; correction_space=logit", "",
             "Host is a caller retained in the bounded sensitive-endpoint graph. Ego is the directed one-hop neighborhood in the induced Host graph.",
             "Ego contains adjacent Host methods only, not arbitrary ordinary method nodes. Host edges are retained raw call edges after Host filtering; no path contraction is used.", ""]
    for row in summary:
        lines.append(f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']} corr_precision={row['correction_precision']:.6f}")
    lines.extend(["", f"Host-only parameters={host_params}; Host+Ego parameters={ego_params}; extra={ego_params-host_params}.",
                  "Future labels used: NO."])
    (args.root / "results/REPORT.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print((args.root / "results/REPORT.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
