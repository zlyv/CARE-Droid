#!/usr/bin/env python3
"""Recompose frozen B0/B3 experts under the Manifest135 autonomous route.

No model is trained. The exact 2017 OOF Manifest RF probabilities and
autonomous route decisions are preserved. Existing BFA B0/B3 ensemble expert
probabilities are substituted only as a diagnostic endpoint to test whether
Internal-neutralization repairs the APKs selected by the current router.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-internal-manifest-auto-route-audit-2017-v2"
OUT = ROOT / "artifacts/care_internal_manifest_auto_route_audit_2017_v2"
AUTO = ROOT / "outputs/care_manifest_rf_autonomous_error_router_v1"
BFA = ROOT / "outputs/care_background_functional_attribution_v1"
INTERNAL = ROOT / "artifacts/care_internal_reference_reliability_audit_2017_v2"
M1 = ROOT / "outputs/care_manifest_rf_m1_freeze_v1"
B0 = "B0_ORIGINAL"
B3 = "B3_INTERNAL_NEUTRAL"
SEEDS = (20261018, 20261019, 20261020)


def digest(path):
    path = Path(path); h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": h.hexdigest()}


def write_json(name, value):
    path = OUT / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(name, frame, compression=None):
    path = OUT / name
    frame.to_csv(path, index=False, mode="x", compression=compression)


def metric(y, probability):
    y = np.asarray(y, int); p = np.asarray(probability, float); z = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    return {"n": len(y), "BA": balanced_accuracy_score(y, z),
            "AUC": roc_auc_score(y, p), "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "FPR": float(fp / max(fp + tn, 1)),
            "FNR": float(fn / max(fn + tp, 1))}


def bootstrap_apk_delta(frame, repeats=5000, seed=20261021):
    per = frame.groupby("sha256", as_index=False).agg(
        b0_error=("b0_error", "mean"), b3_error=("b3_error", "mean"))
    delta = (per.b3_error - per.b0_error).to_numpy(float)
    rng = np.random.RandomState(seed)
    boot = np.empty(repeats, float)
    for i in range(repeats):
        idx = rng.randint(0, len(delta), len(delta)); boot[i] = delta[idx].mean()
    return {"independent_APKs": len(delta), "B3_minus_B0_error_rate": float(delta.mean()),
            "bootstrap_95_low": float(np.quantile(boot, .025)),
            "bootstrap_95_high": float(np.quantile(boot, .975))}


def verify(manifest):
    changed = []
    for item in manifest:
        try: now = digest(item["path"])
        except OSError as exc:
            changed.append({"path": item["path"], "error": str(exc)}); continue
        if now != item: changed.append({"before": item, "after": now})
    return {"status": "PASS" if not changed else "FAIL", "checked": len(manifest),
            "changed": changed,
            "scope": "named Manifest autonomous-route, BFA expert, Internal diagnostic and M1 contract assets; no APK bytes"}


def main():
    if OUT.exists(): raise RuntimeError("refusing overwrite: " + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    assets = [
        HERE / "run_audit.py",
        AUTO / "protocol.json", AUTO / "final_decision.json",
        AUTO / "asset_integrity_check.json", AUTO / "per_apk_predictions.csv.gz",
        BFA / "per_apk_predictions.csv", BFA / "final_decision.json",
        BFA / "asset_integrity_final.json",
        INTERNAL / "internal_reference_per_apk.csv", INTERNAL / "decision.json",
        INTERNAL / "baseline_integrity_check.json", INTERNAL / "protocol.json",
        M1 / "model_contract.json", M1 / "baseline_integrity_check.json",
    ]
    manifest = [digest(path) for path in assets]
    write_json("asset_manifest.json", manifest)

    auto_protocol = json.loads((AUTO / "protocol.json").read_text(encoding="utf-8"))
    if auto_protocol.get("stage1") != "Manifest135 RF" or auto_protocol.get("fixed_budget") is not False:
        raise RuntimeError("autonomous Manifest protocol changed")
    if auto_protocol.get("route_rule") != "predicted_error_probability >= 0.5":
        raise RuntimeError("route threshold changed")
    for path in (AUTO / "asset_integrity_check.json", BFA / "asset_integrity_final.json",
                 INTERNAL / "baseline_integrity_check.json", M1 / "baseline_integrity_check.json"):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("upstream integrity failed: " + str(path))
    write_json("protocol.json", {
        "status": "FROZEN_BEFORE_RECOMPOSITION",
        "years": [2017], "stage1": "Manifest135 RF",
        "router": auto_protocol["router"], "route_rule": auto_protocol["route_rule"],
        "fixed_budget": False, "route_threshold_tuned": False,
        "router_inputs": auto_protocol["router_inputs"],
        "source_router_protocol": "2017 five-fold OOF",
        "official_stage2_reference": "frozen M1=W1_WITNESS3",
        "diagnostic_experts": [B0, B3],
        "diagnostic_expert_source": "BFA-V1 three-seed ensemble probabilities",
        "models_trained": 0, "APK_files_opened": 0,
        "claim_boundary": "Endpoint recomposition diagnostic. B0/B3 were separately trained historical experts; this does not replace official M1 or isolate a causal Internal effect."
    })

    auto = pd.read_csv(AUTO / "per_apk_predictions.csv.gz")
    auto = auto[(auto.year == 2017) & (auto.strategy == "AUTO_ERROR_LR")].copy()
    if len(auto) != 799 * 3 or set(auto.manifest_seed) != set(SEEDS):
        raise RuntimeError("Manifest autonomous OOF coverage mismatch")
    bfa = pd.read_csv(BFA / "per_apk_predictions.csv")
    bfa = bfa[bfa.split.str.startswith("S") & bfa.arm.isin([B0, B3])].copy()
    experts = bfa.pivot(index="sha256", columns="arm", values="expert_probability").reset_index()
    labels = bfa.groupby("sha256", as_index=False).agg(bfa_label=("label", "first"))
    experts = experts.merge(labels, on="sha256", validate="one_to_one")
    if len(experts) != 799: raise RuntimeError("BFA source expert coverage mismatch")

    internal = pd.read_csv(INTERNAL / "internal_reference_per_apk.csv")
    if len(internal) != 799 or internal.sha256.nunique() != 799:
        raise RuntimeError("Internal feature coverage mismatch")
    keep = ["sha256", "routed", "internal_high_conflict_fraction",
            "background_agreement_fraction", "dominant_cluster_fraction",
            "reference_cluster_effective_n", "reference_exact_unique_fraction",
            "b3_minus_b0_internal_mean", "b3_minus_b0_residual_mean"]
    internal = internal[keep].rename(columns={"routed": "historical_p31_top25_route"})
    q_conflict = internal.internal_high_conflict_fraction.quantile([.25, .5, .75]).to_numpy()
    q_agree = internal.background_agreement_fraction.quantile([.25, .5, .75]).to_numpy()
    conflict_index = 1 + (internal.internal_high_conflict_fraction.to_numpy()[:, None] > q_conflict).sum(1)
    agreement_index = 1 + (internal.background_agreement_fraction.to_numpy()[:, None] > q_agree).sum(1)
    internal["conflict_quartile"] = [f"Q{value}" for value in conflict_index]
    internal["agreement_quartile"] = [f"Q{value}" for value in agreement_index]

    frame = auto.merge(experts, on="sha256", validate="many_to_one").merge(
        internal, on="sha256", validate="many_to_one")
    if not np.array_equal(frame.label.astype(int), frame.bfa_label.astype(int)):
        raise RuntimeError("Manifest/BFA label mismatch")
    frame["b0_probability"] = np.where(frame.routed, frame[B0], frame.stage1_probability)
    frame["b3_probability"] = np.where(frame.routed, frame[B3], frame.stage1_probability)
    frame["b0_correct"] = ((frame.b0_probability >= .5).astype(int) == frame.label).astype(int)
    frame["b3_correct"] = ((frame.b3_probability >= .5).astype(int) == frame.label).astype(int)
    frame["b0_error"] = 1 - frame.b0_correct
    frame["b3_error"] = 1 - frame.b3_correct
    frame["b3_fixed"] = ((frame.b0_correct == 0) & (frame.b3_correct == 1)).astype(int)
    frame["b3_broken"] = ((frame.b0_correct == 1) & (frame.b3_correct == 0)).astype(int)
    frame["b3_minus_b0_expert_probability"] = frame[B3] - frame[B0]
    route_count = frame.groupby("sha256").routed.sum().rename("auto_route_seed_count")
    frame = frame.merge(route_count, on="sha256", validate="many_to_one")

    performance = []
    for seed, q in frame.groupby("manifest_seed"):
        choices = {
            "STAGE1_ONLY": q.stage1_probability,
            "OFFICIAL_M1_W1": q.final_probability,
            "B0_RECOMPOSED": q.b0_probability,
            "B3_RECOMPOSED": q.b3_probability,
        }
        for name, probability in choices.items():
            row = {"manifest_seed": int(seed), "arm": name,
                   "route_rate": float(q.routed.mean()), "routed_APKs": int(q.routed.sum()),
                   **metric(q.label, probability)}
            if name in ("B0_RECOMPOSED", "B3_RECOMPOSED"):
                correct = (np.asarray(probability) >= .5) == q.label.to_numpy(int)
                stage1_correct = (q.stage1_probability.to_numpy() >= .5) == q.label.to_numpy(int)
                row.update({"Fixed_vs_stage1": int((~stage1_correct & correct).sum()),
                            "Broken_vs_stage1": int((stage1_correct & ~correct).sum())})
            performance.append(row)
    performance = pd.DataFrame(performance)
    mean_perf = performance.groupby("arm", as_index=False).mean(numeric_only=True)

    conflict_seed = []
    routed = frame[frame.routed].copy()
    for (seed, grouping, group, label), q in pd.concat([
        routed.assign(grouping="conflict_quartile", group=routed.conflict_quartile),
        routed.assign(grouping="agreement_quartile", group=routed.agreement_quartile),
    ]).groupby(["manifest_seed", "grouping", "group", "label"]):
        conflict_seed.append({"manifest_seed": int(seed), "grouping": grouping, "group": group,
            "class": "BENIGN_FPR" if int(label) == 0 else "MALICIOUS_FNR", "n": len(q),
            "B0_error_rate": float(q.b0_error.mean()), "B3_error_rate": float(q.b3_error.mean()),
            "B3_minus_B0_error_rate": float((q.b3_error-q.b0_error).mean()),
            "B3_fixed": int(q.b3_fixed.sum()), "B3_broken": int(q.b3_broken.sum())})
    conflict_seed = pd.DataFrame(conflict_seed)
    conflict_mean = conflict_seed.groupby(["grouping", "group", "class"], as_index=False).agg(
        manifest_seeds=("manifest_seed", "count"), n_mean=("n", "mean"),
        B0_error_rate=("B0_error_rate", "mean"), B3_error_rate=("B3_error_rate", "mean"),
        B3_minus_B0_error_rate=("B3_minus_B0_error_rate", "mean"),
        B3_fixed_mean=("B3_fixed", "mean"), B3_broken_mean=("B3_broken", "mean"))

    overlap = internal[["sha256", "historical_p31_top25_route"]].merge(
        route_count, on="sha256", validate="one_to_one")
    overlap["auto_any"] = overlap.auto_route_seed_count >= 1
    overlap["auto_majority"] = overlap.auto_route_seed_count >= 2
    overlap_rows = pd.DataFrame([{
        "APKs": len(overlap), "historical_P31_Top25": int(overlap.historical_p31_top25_route.sum()),
        "Manifest_auto_any_seed": int(overlap.auto_any.sum()),
        "Manifest_auto_majority": int(overlap.auto_majority.sum()),
        "majority_and_historical": int((overlap.auto_majority & overlap.historical_p31_top25_route.astype(bool)).sum()),
        "majority_new_vs_historical": int((overlap.auto_majority & ~overlap.historical_p31_top25_route.astype(bool)).sum()),
        "historical_not_majority": int((~overlap.auto_majority & overlap.historical_p31_top25_route.astype(bool)).sum()),
    }])

    primary = bootstrap_apk_delta(frame)
    q4_benign = conflict_mean[(conflict_mean.grouping == "conflict_quartile") &
                              (conflict_mean.group == "Q4") &
                              (conflict_mean["class"] == "BENIGN_FPR")].iloc[0].to_dict()
    q4_malicious = conflict_mean[(conflict_mean.grouping == "conflict_quartile") &
                                 (conflict_mean.group == "Q4") &
                                 (conflict_mean["class"] == "MALICIOUS_FNR")].iloc[0].to_dict()
    b0 = mean_perf[mean_perf.arm == "B0_RECOMPOSED"].iloc[0]
    b3 = mean_perf[mean_perf.arm == "B3_RECOMPOSED"].iloc[0]
    official = mean_perf[mean_perf.arm == "OFFICIAL_M1_W1"].iloc[0]
    if q4_benign["B3_minus_B0_error_rate"] < 0 and q4_benign["B3_fixed_mean"] > q4_benign["B3_broken_mean"]:
        status = "MANIFEST_ROUTE_CONFLICT_ASSOCIATION_ONLY"
    else:
        status = "MANIFEST_ROUTE_INTERNAL_CONFLICT_NOT_SUPPORTED"
    decision = {
        "status": status, "stage1": "Manifest135 RF",
        "router": "autonomous error LR", "route_rule": "P(error)>=0.5",
        "fixed_budget": False, "source_only": True,
        "manifest_seed_mean_route_rate": float(performance.route_rate.mean()),
        "B0_recomposed_BA": float(b0.BA), "B3_recomposed_BA": float(b3.BA),
        "B3_minus_B0_BA": float(b3.BA-b0.BA),
        "official_M1_W1_BA_reference": float(official.BA),
        "cluster_bootstrap_endpoint_error": primary,
        "high_conflict_Q4_benign": q4_benign,
        "high_conflict_Q4_malicious": q4_malicious,
        "official_M1_changed": False, "models_trained": 0,
        "interpretation": "B0/B3 recomposition is a mechanism audit, not a replacement evaluation for official M1/W1."
    }

    write_csv("per_apk_manifest_seed.csv.gz", frame, compression="gzip")
    write_csv("performance_per_seed.csv", performance)
    write_csv("performance_seed_mean.csv", mean_perf)
    write_csv("conflict_group_per_seed.csv", conflict_seed)
    write_csv("conflict_group_seed_mean.csv", conflict_mean)
    write_csv("route_overlap_summary.csv", overlap_rows)
    write_json("decision.json", decision)

    report = [
        "# Internal可靠性：Manifest135自主路由重组审计", "",
        f"最终状态：**{status}**。", "",
        "本审计使用当前Manifest135 RF第一层和2017五折OOF自主错误路由，规则固定为`P(error) >= 0.5`，没有固定预算。未训练模型、未读取APK、未修改正式M1/W1。", "",
        "B0/B3来自历史BFA三种子集成，仅作为同一路由样本上的Internal机制诊断。它们不是当前正式M1/W1，不能据此替换M1。", "",
        "| 第二层端点 | BA | AUC | FPR | FNR |", "|---|---:|---:|---:|---:|",
    ]
    for arm in ("STAGE1_ONLY", "OFFICIAL_M1_W1", "B0_RECOMPOSED", "B3_RECOMPOSED"):
        r = mean_perf[mean_perf.arm == arm].iloc[0]
        report.append(f"| {arm} | {r.BA:.4f} | {r.AUC:.4f} | {r.FPR:.4f} | {r.FNR:.4f} |")
    report += ["", f"Manifest自主路由率（三种子均值）：{performance.route_rate.mean():.3f}。",
        f"B3相对B0的BA差：{b3.BA-b0.BA:+.4f}。",
        f"按独立APK聚类bootstrap，B3-B0端点错误率差：{primary['B3_minus_B0_error_rate']:+.4f}，95%区间[{primary['bootstrap_95_low']:+.4f}, {primary['bootstrap_95_high']:+.4f}]。", "",
        "## Internal高冲突最高四分位", "",
        f"良性路由APK：B0 FP率{q4_benign['B0_error_rate']:.4f}，B3 FP率{q4_benign['B3_error_rate']:.4f}，每种子平均修复/破坏{q4_benign['B3_fixed_mean']:.2f}/{q4_benign['B3_broken_mean']:.2f}。",
        f"恶意路由APK：B0 FN率{q4_malicious['B0_error_rate']:.4f}，B3 FN率{q4_malicious['B3_error_rate']:.4f}，每种子平均修复/破坏{q4_malicious['B3_fixed_mean']:.2f}/{q4_malicious['B3_broken_mean']:.2f}。", "",
        "结论只回答：当前Manifest自主路由选择的样本中，历史B3中性化Internal是否比B0更稳定。它不分解Internal直接输入与Residual重算，也不构成新模型收益。", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    integrity = verify(manifest); write_json("baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS": raise RuntimeError("asset changed during audit")
    print("IMARA_DONE status=" + status, flush=True)


if __name__ == "__main__":
    main()
