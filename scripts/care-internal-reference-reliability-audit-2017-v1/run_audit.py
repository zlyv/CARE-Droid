#!/usr/bin/env python3
"""Audit CARE-Droid Internal reference reliability on frozen 2017 OOF data.

This script trains no model, opens no APK, changes no frozen route, and writes
only a new audit artifact directory. It reconstructs the exact fold/seed
background tensors, measures dominant-cluster support and concentration,
defines label-free train-only background agreement states, and aligns them with
the existing BFA-V1 B0/B3 OOF predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-internal-reference-reliability-audit-2017-v1"
OUT = ROOT / "artifacts/care_internal_reference_reliability_audit_2017_v1"
BFA_CODE = ROOT / "scripts/care-background-functional-attribution-v1"
BFA_OUT = ROOT / "outputs/care_background_functional_attribution_v1"
HAIG = ROOT / "scripts/care-p31rf-c43-haig-host-ablation-source-v1"
ISR2 = ROOT / "outputs/care_internal_self_reference_audit_v2"
sys.path[:0] = [str(BFA_CODE), str(HAIG)]

from bfa_core import (SEEDS, asset_paths, formal_split, load_data_contract,
                      transform_arm)  # noqa: E402
import haig_background as hb  # noqa: E402

SPLITS = ("S1", "S2", "S3", "S4", "S5")
B0 = "B0_ORIGINAL"
B3 = "B3_INTERNAL_NEUTRAL"


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": h.hexdigest()}


def write_json(name, value):
    path = OUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2,
                  allow_nan=False, default=scalar)
        handle.write("\n")


def write_csv(name, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with (OUT / name).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_clusters(assignment, coverage=0.70):
    counts = np.bincount(np.asarray(assignment, int))
    ordered = np.argsort(-counts, kind="stable")
    target = max(1, int(math.ceil(coverage * len(assignment))))
    selected, covered = [], 0
    for cluster in ordered:
        if counts[cluster] <= 0:
            continue
        selected.append(int(cluster))
        covered += int(counts[cluster])
        if covered >= target:
            break
    return selected, counts, covered


def pairwise_rms(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).mean(2), 0.0))


def hard_assignment(pack, bag, raw):
    state = pack["old"]["background"]["state"]
    sp = pack["old"]["background"]["sp"]
    x = np.asarray(bag[:, :107], np.float32)
    role = np.maximum(np.asarray(raw[:, 43:74], np.float32), 0)
    denominator = role.sum(1, keepdims=True)
    role = np.divide(role, denominator, out=np.zeros_like(role), where=denominator > 0)
    distance = pairwise_rms(x, state["shared_centers"])
    assignment = hb.hard(distance / np.maximum(state["shared_scale"][None, :], 1e-6),
                         role, sp)
    return x, np.asarray(assignment, int)


def background_thresholds(train_bags):
    # Equal-APK, class-blind, training-fold-only medians. These thresholds are
    # descriptive sign anchors and are never optimized against labels.
    values = {}
    for name, column in (("external", 107), ("internal", 108), ("commonness", 109)):
        apk_medians = [float(np.median(np.asarray(bag)[:, column])) for bag in train_bags]
        values[name] = float(np.median(apk_medians))
    return values


def entropy_effective(counts):
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    if not len(counts):
        return 0.0, 0.0, 0.0
    p = counts / counts.sum()
    hhi = float(np.sum(p * p))
    effective = float(1.0 / hhi)
    entropy = 0.0 if len(p) == 1 else float(-np.sum(p * np.log(p)) / np.log(len(p)))
    return hhi, effective, entropy


def rank_average(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    ra, rb = rank_average(a[mask]), rank_average(b[mask])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def quantiles(values):
    values = np.asarray(values, float)
    return {"min": float(np.min(values)), "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "p75": float(np.quantile(values, .75)),
            "max": float(np.max(values)), "mean": float(np.mean(values))}


def mean_ci_paired(rows, key0, key1, repeats=4000, seed=20261021):
    a = np.asarray([float(row[key0]) for row in rows], float)
    b = np.asarray([float(row[key1]) for row in rows], float)
    delta = b - a
    rng = np.random.RandomState(seed)
    boot = np.empty(repeats, float)
    for i in range(repeats):
        indices = rng.randint(0, len(delta), len(delta))
        boot[i] = float(np.mean(delta[indices]))
    return {"n": len(delta), "B0_error_rate": float(np.mean(a)),
            "B3_error_rate": float(np.mean(b)),
            "B3_minus_B0_error_rate": float(np.mean(delta)),
            "bootstrap_95_low": float(np.quantile(boot, .025)),
            "bootstrap_95_high": float(np.quantile(boot, .975))}


def add_feature_quartile(rows, key, output_key):
    values = np.asarray([float(row[key]) for row in rows], float)
    cuts = np.quantile(values, [.25, .5, .75])
    for row, value in zip(rows, values):
        row[output_key] = "Q" + str(1 + int(value > cuts[0]) + int(value > cuts[1]) + int(value > cuts[2]))
    return [float(x) for x in cuts]


def status(label, probability):
    prediction = int(float(probability) >= .5)
    if int(label) == 1:
        return "TP" if prediction else "FN"
    return "FP" if prediction else "TN"


def immutable_assets():
    items = list(asset_paths())
    items += [HERE / "run_audit.py",
              BFA_CODE / "bfa_core.py", BFA_CODE / "bfa_evaluate.py",
              HAIG / "haig_background.py",
              BFA_OUT / "per_apk_predictions.csv",
              BFA_OUT / "final_decision.json",
              BFA_OUT / "asset_integrity_final.json",
              ISR2 / "final_decision.json", ISR2 / "baseline_integrity_check.json",
              ISR2 / "REPORT.md"]
    return list(dict.fromkeys(Path(x) for x in items))


def verify_assets(manifest):
    changed = []
    for item in manifest:
        try:
            now = digest(item["path"])
        except OSError as exc:
            changed.append({"path": item["path"], "error": str(exc)})
            continue
        if now != item:
            changed.append({"path": item["path"], "before": item, "after": now})
    return {"status": "PASS" if not changed else "FAIL",
            "checked": len(manifest), "changed": changed,
            "scope": "named frozen code, cached features, routes, checkpoints, BFA predictions and corrected ISR-v2 result; no APK bytes"}


def main():
    if OUT.exists():
        raise RuntimeError("output directory exists; refusing to overwrite: " + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = [digest(path) for path in immutable_assets()]
    write_json("asset_manifest.json", manifest)
    write_json("protocol.json", {
        "status": "FROZEN_BEFORE_LABEL_DIAGNOSTICS",
        "scope": "2017 five-fold OOF only",
        "models_trained": 0, "APK_files_opened": 0,
        "Stage1_changed": False, "Router_changed": False,
        "BFA_models_changed": False, "threshold_changed": False,
        "background_state": "exact fold/seed frozen SCBE M0 background state",
        "conflict_definition": {
            "thresholds": "training-fold-only equal-APK medians; class-blind",
            "anomaly_directions": "External high, Internal high, Commonness low",
            "internal_high_conflict": "Internal high while External low and Commonness high",
            "internal_low_conflict": "Internal low while External high and Commonness low",
            "consistent_anomalous": "all three anomaly directions high",
            "consistent_normal": "all three anomaly directions low"
        },
        "claim_boundary": "mechanism diagnosis on reused 2017 OOF labels; no rule selection or independent validation claim"
    })
    print("IRRA_PREFLIGHT_PASS assets=" + str(len(manifest)), flush=True)

    predictions = read_csv(BFA_OUT / "per_apk_predictions.csv")
    pred = defaultdict(dict)
    for row in predictions:
        if row["split"] in SPLITS and row["arm"] in (B0, B3):
            pred[(row["split"], row["sha256"])][row["arm"]] = row

    data, _ = load_data_contract()
    seed_rows = []
    threshold_rows = []
    for split in SPLITS:
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device="cpu")
            thresholds = background_thresholds(pack["bags_train"])
            _, b3_val, b3_protocol = transform_arm(
                pack["bags_train"], pack["bags_val"], pack["raw_train"],
                pack["raw_val"], B3, seed)
            threshold_rows.append({"split": split, "seed": seed, **thresholds,
                                   "b3_neutral_internal": b3_protocol["neutral_value"]})
            for local, (bag, changed, raw, index) in enumerate(zip(
                    pack["bags_val"], b3_val, pack["raw_val"], pack["val"])):
                sid = str(data["ids"][index])
                x, assignment = hard_assignment(pack, bag, raw)
                selected, counts, covered = selected_clusters(assignment)
                selected_mask = np.isin(assignment, np.asarray(selected, int))
                selected_counts = counts[np.asarray(selected, int)]
                hhi, effective, entropy = entropy_effective(selected_counts)
                occupied = int(np.sum(counts > 0))
                reference_x = np.ascontiguousarray(x[selected_mask])
                unique_reference = int(len(np.unique(reference_x, axis=0)))
                centroids = np.asarray([x[assignment == cluster].mean(0) for cluster in selected], np.float32)
                if len(centroids) > 1:
                    d = pairwise_rms(centroids, centroids)
                    centroid_separation = float(np.median(d[np.triu_indices(len(centroids), 1)]))
                else:
                    centroid_separation = 0.0

                external = np.asarray(bag[:, 107], float)
                internal = np.asarray(bag[:, 108], float)
                common = np.asarray(bag[:, 109], float)
                residual = np.asarray(bag[:, 110], float)
                e_high = external > thresholds["external"]
                i_high = internal > thresholds["internal"]
                c_low = common < thresholds["commonness"]
                high_conflict = i_high & ~e_high & ~c_low
                low_conflict = ~i_high & e_high & c_low
                consistent_anomaly = i_high & e_high & c_low
                consistent_normal = ~i_high & ~e_high & ~c_low
                assigned = high_conflict | low_conflict | consistent_anomaly | consistent_normal
                mixed = ~assigned
                n = len(bag)
                seed_rows.append({
                    "split": split, "seed": seed, "sha256": sid,
                    "label": int(data["y"][index]), "routed": int(pack["route"][local]),
                    "hosts": n, "occupied_shared_clusters": occupied,
                    "selected_reference_clusters": len(selected),
                    "reference_support_hosts": int(covered),
                    "reference_realized_coverage": float(covered / n),
                    "dominant_cluster_hosts": int(np.max(counts)),
                    "dominant_cluster_fraction": float(np.max(counts) / n),
                    "reference_cluster_hhi": hhi,
                    "reference_cluster_effective_n": effective,
                    "reference_cluster_normalized_entropy": entropy,
                    "reference_exact_unique_hosts": unique_reference,
                    "reference_exact_unique_fraction": float(unique_reference / max(covered, 1)),
                    "selected_centroid_pairwise_rms_median": centroid_separation,
                    "internal_high_conflict_fraction": float(np.mean(high_conflict)),
                    "internal_low_conflict_fraction": float(np.mean(low_conflict)),
                    "consistent_anomalous_fraction": float(np.mean(consistent_anomaly)),
                    "consistent_normal_fraction": float(np.mean(consistent_normal)),
                    "mixed_background_fraction": float(np.mean(mixed)),
                    "background_agreement_fraction": float(np.mean(consistent_anomaly | consistent_normal)),
                    "external_mean": float(np.mean(external)),
                    "internal_mean": float(np.mean(internal)),
                    "commonness_mean": float(np.mean(common)),
                    "residual_mean": float(np.mean(residual)),
                    "b3_internal_mean": float(np.mean(changed[:, 108])),
                    "b3_residual_mean": float(np.mean(changed[:, 110])),
                    "b3_minus_b0_internal_mean": float(np.mean(changed[:, 108] - bag[:, 108])),
                    "b3_minus_b0_residual_mean": float(np.mean(changed[:, 110] - bag[:, 110]))
                })
            print(f"IRRA_SPLIT_SEED_DONE split={split} seed={seed} apks={len(pack['val'])}", flush=True)

    by_apk = defaultdict(list)
    for row in seed_rows:
        by_apk[(row["split"], row["sha256"])].append(row)
    apk_rows = []
    identity = {"split", "sha256", "label", "routed", "hosts"}
    numeric = [key for key in seed_rows[0] if key not in identity | {"seed"}]
    for key, rows in by_apk.items():
        if len(rows) != 3:
            raise RuntimeError("missing seed rows: " + repr(key))
        base = {name: rows[0][name] for name in identity}
        for name in numeric:
            base[name] = float(np.mean([float(row[name]) for row in rows]))
        group = pred[key]
        if set(group) != {B0, B3}:
            raise RuntimeError("missing aligned B0/B3 prediction: " + repr(key))
        p0, p3 = group[B0], group[B3]
        if int(p0["label"]) != base["label"] or int(p0["routed"]) != base["routed"]:
            raise RuntimeError("prediction/feature identity mismatch: " + repr(key))
        base.update({
            "stage1_probability": float(p0["stage1_probability"]),
            "b0_expert_probability": float(p0["expert_probability"]),
            "b3_expert_probability": float(p3["expert_probability"]),
            "b3_minus_b0_expert_probability": float(p3["expert_probability"]) - float(p0["expert_probability"]),
            "b0_final_probability": float(p0["final_probability"]),
            "b3_final_probability": float(p3["final_probability"]),
            "b0_correct": int(p0["final_correct"]),
            "b3_correct": int(p3["final_correct"]),
            "b0_status": status(base["label"], p0["final_probability"]),
            "b3_status": status(base["label"], p3["final_probability"]),
        })
        if base["b0_correct"] == 0 and base["b3_correct"] == 1:
            base["transition"] = "B3_FIXED"
        elif base["b0_correct"] == 1 and base["b3_correct"] == 0:
            base["transition"] = "B3_BROKEN"
        else:
            base["transition"] = "UNCHANGED_CORRECT" if base["b0_correct"] else "UNCHANGED_WRONG"
        base["b0_error"] = 1 - base["b0_correct"]
        base["b3_error"] = 1 - base["b3_correct"]
        apk_rows.append(base)

    apk_rows.sort(key=lambda row: (row["split"], row["sha256"]))
    routed = [row for row in apk_rows if row["routed"] == 1]
    conflict_cuts = add_feature_quartile(routed, "internal_high_conflict_fraction", "high_conflict_quartile")
    agreement_cuts = add_feature_quartile(routed, "background_agreement_fraction", "agreement_quartile")

    summary_rows = []
    feature_names = (
        "hosts", "selected_reference_clusters", "reference_support_hosts",
        "reference_realized_coverage", "dominant_cluster_fraction",
        "reference_cluster_effective_n", "reference_cluster_normalized_entropy",
        "reference_exact_unique_fraction", "selected_centroid_pairwise_rms_median",
        "internal_high_conflict_fraction", "internal_low_conflict_fraction",
        "background_agreement_fraction", "residual_mean")
    for group_name, subset in (("ALL_2017_OOF", apk_rows), ("ROUTED", routed),
                               ("BENIGN", [r for r in apk_rows if r["label"] == 0]),
                               ("MALICIOUS", [r for r in apk_rows if r["label"] == 1])):
        for name in feature_names:
            summary_rows.append({"group": group_name, "metric": name,
                                 **quantiles([row[name] for row in subset])})

    group_rows = []
    for grouping in ("high_conflict_quartile", "agreement_quartile"):
        for group in sorted(set(row[grouping] for row in routed)):
            subset_all = [row for row in routed if row[grouping] == group]
            for label, class_name in ((0, "BENIGN_FPR"), (1, "MALICIOUS_FNR")):
                subset = [row for row in subset_all if row["label"] == label]
                if not subset:
                    continue
                ci = mean_ci_paired(subset, "b0_error", "b3_error",
                                    seed=20261021 + label + len(group_rows))
                group_rows.append({"grouping": grouping, "group": group,
                    "class": class_name, **ci,
                    "B3_fixed": sum(row["transition"] == "B3_FIXED" for row in subset),
                    "B3_broken": sum(row["transition"] == "B3_BROKEN" for row in subset),
                    "mean_high_conflict_fraction": float(np.mean([row["internal_high_conflict_fraction"] for row in subset])),
                    "mean_agreement_fraction": float(np.mean([row["background_agreement_fraction"] for row in subset]))})

    flip_rows = []
    for b0_kind in ("TN", "FP", "FN", "TP"):
        for transition in ("B3_FIXED", "B3_BROKEN", "UNCHANGED_CORRECT", "UNCHANGED_WRONG"):
            subset = [row for row in routed if row["b0_status"] == b0_kind and row["transition"] == transition]
            if not subset:
                continue
            flip_rows.append({"b0_status": b0_kind, "transition": transition, "n": len(subset),
                "high_conflict_fraction_mean": float(np.mean([r["internal_high_conflict_fraction"] for r in subset])),
                "high_conflict_fraction_median": float(np.median([r["internal_high_conflict_fraction"] for r in subset])),
                "agreement_fraction_mean": float(np.mean([r["background_agreement_fraction"] for r in subset])),
                "dominant_cluster_fraction_mean": float(np.mean([r["dominant_cluster_fraction"] for r in subset])),
                "reference_effective_clusters_mean": float(np.mean([r["reference_cluster_effective_n"] for r in subset])),
                "b3_minus_b0_probability_mean": float(np.mean([r["b3_minus_b0_expert_probability"] for r in subset])),
                "b3_minus_b0_residual_mean": float(np.mean([r["b3_minus_b0_residual_mean"] for r in subset]))})

    correlation_rows = []
    for group_name, subset in (("ROUTED_ALL", routed),
                               ("ROUTED_BENIGN", [r for r in routed if r["label"] == 0]),
                               ("ROUTED_MALICIOUS", [r for r in routed if r["label"] == 1])):
        for x in ("hosts", "dominant_cluster_fraction", "reference_cluster_effective_n",
                  "reference_exact_unique_fraction", "internal_high_conflict_fraction",
                  "background_agreement_fraction", "b3_minus_b0_residual_mean"):
            for y in ("internal_mean", "residual_mean", "b0_expert_probability",
                      "b3_minus_b0_expert_probability"):
                correlation_rows.append({"group": group_name, "x": x, "y": y,
                                         "spearman": spearman([r[x] for r in subset], [r[y] for r in subset]),
                                         "n": len(subset)})

    q4_benign = next((row for row in group_rows if row["grouping"] == "high_conflict_quartile"
                      and row["group"] == "Q4" and row["class"] == "BENIGN_FPR"), None)
    q4_agreement = next((row for row in group_rows if row["grouping"] == "agreement_quartile"
                         and row["group"] == "Q4" and row["class"] == "BENIGN_FPR"), None)
    overall = mean_ci_paired(routed, "b0_error", "b3_error", seed=20261099)
    per_fold = []
    for split in SPLITS:
        subset = [row for row in routed if row["split"] == split and row["high_conflict_quartile"] == "Q4" and row["label"] == 0]
        if subset:
            per_fold.append({"split": split, "n": len(subset),
                "fixed": sum(r["transition"] == "B3_FIXED" for r in subset),
                "broken": sum(r["transition"] == "B3_BROKEN" for r in subset)})
    positive_folds = sum(row["fixed"] > row["broken"] for row in per_fold)
    conflict_net = None if q4_benign is None else q4_benign["B3_fixed"] - q4_benign["B3_broken"]
    agreement_b0_advantage = None if q4_agreement is None else q4_agreement["B3_broken"] - q4_agreement["B3_fixed"]
    if conflict_net is not None and conflict_net > 0 and positive_folds >= 3 and agreement_b0_advantage is not None and agreement_b0_advantage > 0:
        audit_status = "PARTIAL_INTERNAL_RELIABILITY_MECHANISM_SIGNAL"
    elif conflict_net is not None and conflict_net > 0:
        audit_status = "CONFLICT_ASSOCIATION_ONLY"
    else:
        audit_status = "INTERNAL_CONFLICT_MECHANISM_NOT_SUPPORTED"

    isr2_decision = json.loads((ISR2 / "final_decision.json").read_text(encoding="utf-8"))
    decision = {
        "status": audit_status,
        "2017_OOF_APKs": len(apk_rows), "routed_APKs": len(routed),
        "models_trained": 0, "APK_files_opened": 0,
        "overall_routed_B3_minus_B0_error": overall,
        "high_conflict_Q4_benign": q4_benign,
        "high_agreement_Q4_benign": q4_agreement,
        "high_conflict_positive_folds": positive_folds,
        "high_conflict_fold_details": per_fold,
        "feature_only_quartile_cuts": {"high_conflict": conflict_cuts, "agreement": agreement_cuts},
        "corrected_ISR_v2_cross_evidence": {
            "status": isr2_decision["status"],
            "error_association_supported": isr2_decision["error_association_supported"],
            "formal_integrity": isr2_decision["formal_integrity"]},
        "interpretation": "Any subgroup association is descriptive on reused OOF labels. B0/B3 are separately retrained models, so their probability difference cannot isolate Internal from retraining or recomputed Residual.",
        "formal_mainline_changed": False
    }

    write_csv("internal_reference_seed_metrics.csv", seed_rows)
    write_csv("internal_reference_per_apk.csv", apk_rows)
    write_csv("internal_reference_summary.csv", summary_rows)
    write_csv("background_thresholds_by_fold_seed.csv", threshold_rows)
    write_csv("conflict_group_performance.csv", group_rows)
    write_csv("b0_b3_flip_analysis.csv", flip_rows)
    write_csv("probability_residual_correlations.csv", correlation_rows)
    write_json("decision.json", decision)

    metrics = {row["metric"]: row for row in summary_rows if row["group"] == "ALL_2017_OOF"}
    b0_errors = sum(row["b0_error"] for row in apk_rows)
    b3_errors = sum(row["b3_error"] for row in apk_rows)
    fixed = sum(row["transition"] == "B3_FIXED" for row in routed)
    broken = sum(row["transition"] == "B3_BROKEN" for row in routed)
    lines = [
        "# CARE-Droid Internal 参考可靠性审计（2017 OOF）", "",
        f"最终状态：**{audit_status}**。", "",
        "本审计只读取冻结2017五折OOF特征、BFA-V1预测及checkpoint；未训练模型、未读取APK、未改变Manifest135第一层、路由、阈值或正式三背景。", "",
        "## Internal 主导参考的构造与代表性", "",
        "Internal先按Shared原型分配Host，再按簇大小从大到小选择簇，直到覆盖至少70% Host；每个入选簇的APK内均值成为内部参考中心。70%是Host覆盖目标，并不保证语义覆盖或独立代码来源。", "",
        f"- APK Host数中位数：{metrics['hosts']['median']:.1f}。",
        f"- 实际参考Host覆盖率中位数：{metrics['reference_realized_coverage']['median']:.3f}。",
        f"- 入选参考簇数中位数：{metrics['selected_reference_clusters']['median']:.1f}；有效簇数中位数：{metrics['reference_cluster_effective_n']['median']:.2f}。",
        f"- 最大单簇Host占比中位数：{metrics['dominant_cluster_fraction']['median']:.3f}。",
        f"- 参考Host精确唯一比例中位数：{metrics['reference_exact_unique_fraction']['median']:.3f}。",
        "这些量衡量几何支持和集中度，不能直接识别广告、框架或恶意组件来源。", "",
        "## 三背景一致与冲突", "",
        "冲突状态使用每个训练折内、类别盲、等APK加权的中位数定义：External高、Internal高、Commonness低均指向更异常。重点冲突为Internal高，但External低且Commonness高。该定义在查看OOF标签前固定，不用于选择分类规则。", "",
        f"- Internal高冲突Host比例中位数：{metrics['internal_high_conflict_fraction']['median']:.3f}。",
        f"- 三背景完全一致Host比例中位数：{metrics['background_agreement_fraction']['median']:.3f}。", "",
        "## B0 与 B3", "",
        f"2017 OOF全体APK错误数：B0={b0_errors}，B3={b3_errors}。在冻结路由APK上，B3修复{fixed}个、破坏{broken}个。",
        f"路由APK的B3-B0错误率差为{overall['B3_minus_B0_error_rate']:+.4f}，APK bootstrap 95%区间[{overall['bootstrap_95_low']:+.4f}, {overall['bootstrap_95_high']:+.4f}]。", ""
    ]
    if q4_benign is not None:
        lines += [
            "### Internal高冲突最高四分位中的良性APK", "",
            f"- APK数：{q4_benign['n']}。",
            f"- B0 FP率：{q4_benign['B0_error_rate']:.4f}；B3 FP率：{q4_benign['B3_error_rate']:.4f}。",
            f"- B3修复/破坏：{q4_benign['B3_fixed']}/{q4_benign['B3_broken']}。",
            f"- 配对FP率差：{q4_benign['B3_minus_B0_error_rate']:+.4f}，95%区间[{q4_benign['bootstrap_95_low']:+.4f}, {q4_benign['bootstrap_95_high']:+.4f}]。",
            f"- 五折中修复数严格多于破坏数的折数：{positive_folds}/5。", ""
        ]
    if q4_agreement is not None:
        lines += [
            "### 三背景一致性最高四分位中的良性APK", "",
            f"- APK数：{q4_agreement['n']}。",
            f"- B0 FP率：{q4_agreement['B0_error_rate']:.4f}；B3 FP率：{q4_agreement['B3_error_rate']:.4f}。",
            f"- B3修复/破坏：{q4_agreement['B3_fixed']}/{q4_agreement['B3_broken']}。", ""
        ]
    lines += [
        "## 裁决边界", "",
        "B3是重新训练的Internal中性化模型，并同时重算Residual，因此B3-B0概率差不能分解为纯Internal直接效应与Residual效应，也不能排除重新训练容量效应。分组使用同一2017 OOF标签，只能作为机制诊断。", "",
        f"修正后的ISR-v2交叉证据为 `{isr2_decision['status']}`，资产完整性 `{isr2_decision['formal_integrity']}`，其错误关联判定为 `{isr2_decision['error_association_supported']}`。早期重放失败的ISR-v1未作为证据。", "",
        "本审计不授权替换Internal、增加第四背景、按冲突比例建立新门控，或把分组中的开发集变化声明为独立分类收益。", ""
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    integrity = verify_assets(manifest)
    write_json("baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("frozen asset changed during audit")
    print("IRRA_DONE status=" + audit_status, flush=True)


if __name__ == "__main__":
    main()
