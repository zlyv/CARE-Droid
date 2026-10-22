#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (ARMS, FUTURE_LABELS, FUTURE_RF, OUTPUT, WITHIN_RF, YEARS,
    classification_metrics, comparison_metrics, dump_json, file_record, norm, read_csv, write_csv)


def normalize_frozen(arm):
    path = FUTURE_RF if arm == "RF" else OUTPUT / "frozen" / arm / "predictions_label_free.csv"
    rows = read_csv(path); labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(FUTURE_LABELS)}
    result = []
    for row in rows:
        sid = norm(row["sha256"]); result.append({"sha256": sid, "year": int(row["year"]), "label": labels[sid],
            "arm": arm, "stage1_prob": float(row["stage1_prob"]), "routed": int(row["routed"]),
            "expert_prob": float(row["expert_prob"]),
            "stage2_prob": float(row["di25_final_prob"] if arm == "RF" else row["stage2_prob"])})
    return result


def normalize_within(arm):
    rows = (read_csv(WITHIN_RF[0]) + read_csv(WITHIN_RF[1])) if arm == "RF" else read_csv(OUTPUT / "within" / arm / "predictions.csv")
    result = []
    for row in rows:
        result.append({"sha256": norm(row["sha256"]), "year": int(row["year"]), "fold": int(row["fold"]),
            "label": int(row["label"]), "arm": arm, "stage1_prob": float(row["stage1_prob"]),
            "routed": int(row["routed"]), "expert_prob": float(row["expert_prob"]),
            "stage2_prob": float(row["di25_final_prob"] if arm == "RF" else row["stage2_prob"])})
    return result


def summarize(protocol, arm, rows, scope):
    y = np.asarray([row["label"] for row in rows], int); base = np.asarray([row["stage1_prob"] for row in rows], float)
    final = np.asarray([row["stage2_prob"] for row in rows], float); route = np.asarray([row["routed"] for row in rows], bool)
    stage1 = classification_metrics(y, base); stage2 = comparison_metrics(y, base, final, route)
    return {"protocol": protocol, "scope": scope, "arm": arm,
        **{f"stage1_{key}": value for key, value in stage1.items()},
        **{f"stage2_{key}": value for key, value in stage2.items()},
        "BA_gain_pp": 100 * (stage2["BA"] - stage1["BA"]), "accuracy_gain_pp": 100 * (stage2["accuracy"] - stage1["accuracy"]),
        "AUC_gain": stage2["AUC"] - stage1["AUC"], "F1_gain": stage2["F1"] - stage1["F1"]}


def main():
    source = read_csv(OUTPUT / "source_stage2_metrics.csv"); aggregate = list(source); fold_rows = []
    future_all = []; within_all = []
    for arm in ARMS:
        frozen = normalize_frozen(arm); within = normalize_within(arm)
        if len(frozen) != 1989 or len(within) != 1989:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: diagnostic rows changed arm={arm}")
        future_all.extend(frozen); within_all.extend(within)
        for year in YEARS:
            aggregate.append(summarize("frozen_2017_to_future", arm,
                [row for row in frozen if row["year"] == year], str(year)))
            aggregate.append(summarize("independent_same_year_fixed_5fold", arm,
                [row for row in within if row["year"] == year], str(year)))
            for fold in range(1, 6):
                selected = [row for row in within if row["year"] == year and row["fold"] == fold]
                fold_rows.append({**summarize("independent_same_year_fixed_5fold", arm, selected, str(year)), "fold": fold})
        aggregate.append(summarize("frozen_2017_to_future", arm, frozen, "pooled"))
        aggregate.append(summarize("independent_same_year_fixed_5fold", arm, within, "pooled"))
        annual_frozen = [row for row in aggregate if row["protocol"] == "frozen_2017_to_future" and row["arm"] == arm and row["scope"] in {str(y) for y in YEARS}]
        annual_within = [row for row in aggregate if row["protocol"] == "independent_same_year_fixed_5fold" and row["arm"] == arm and row["scope"] in {str(y) for y in YEARS}]
        aggregate.append({"protocol": "frozen_2017_to_future", "scope": "macro_year_mean", "arm": arm,
            "stage1_BA": float(np.mean([row["stage1_BA"] for row in annual_frozen])),
            "stage2_BA": float(np.mean([row["stage2_BA"] for row in annual_frozen])),
            "BA_gain_pp": float(np.mean([row["BA_gain_pp"] for row in annual_frozen]))})
        aggregate.append({"protocol": "independent_same_year_fixed_5fold", "scope": "macro_year_mean", "arm": arm,
            "stage1_BA": float(np.mean([row["stage1_BA"] for row in annual_within])),
            "stage2_BA": float(np.mean([row["stage2_BA"] for row in annual_within])),
            "BA_gain_pp": float(np.mean([row["BA_gain_pp"] for row in annual_within]))})
    write_csv(OUTPUT / "full_system_comparison.csv", aggregate); write_csv(OUTPUT / "same_year_stage2_fold_metrics.csv", fold_rows)
    write_csv(OUTPUT / "frozen_future_stage2_predictions.csv", future_all); write_csv(OUTPUT / "same_year_stage2_predictions.csv", within_all)
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))
    after = []; changed = []
    for old in before["assets"]:
        current = file_record(Path(old["absolute_path"]), old["purpose"]); after.append(current)
        if current["sha256"] != old["sha256"] or current["size_bytes"] != old["size_bytes"]:
            changed.append({"path": old["absolute_path"], "before": old["sha256"], "after": current["sha256"]})
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed,
        "asset_count": len(after)})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")
    lines = ["# CARE-Droid P31 classifier family + identical Stage-II comparison v1", "",
        "Every arm is: calibrated P31 classifier -> self-anchored MeanOpposition Top25% -> A0 Role C43-RQ -> bounded correction (max_delta=2.25).",
        "RF rows reuse the frozen formal results; LR/XGB/SVM/MEAN4 retrain the same Stage-II with strict arm-specific inner OOF context.",
        "Future is diagnostic only. Same-year OOF is not temporal generalization.", ""]
    for protocol, title in (("source_2017_fixed_oof", "2017 fixed OOF"),
            ("frozen_2017_to_future", "Frozen 2017 -> future"),
            ("independent_same_year_fixed_5fold", "Independent same-year OOF")):
        lines += [f"## {title}", "", "| Scope | Arm | Stage-I BA | +Stage-II BA | Gain (pp) | fixed | broken | net |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        scopes = ["pooled"] if protocol == "source_2017_fixed_oof" else [str(year) for year in YEARS] + ["pooled", "macro_year_mean"]
        for scope in scopes:
            for arm in ARMS:
                row = next(value for value in aggregate if value["protocol"] == protocol and value["scope"] == scope and value["arm"] == arm)
                lines.append(f"| {scope} | {arm} | {float(row['stage1_BA']):.6f} | {float(row['stage2_BA']):.6f} | {float(row['BA_gain_pp']):+.4f} | {row.get('stage2_fixed','')} | {row.get('stage2_broken','')} | {row.get('stage2_net','')} |")
        lines.append("")
    lines += ["## Evidence boundary", "", "- The formal RF system is unchanged.",
        "- Source 2017 is the primary comparison domain; future and same-year tables do not select the formal method.",
        "- A positive BA gain is not treated as sufficient unless fixed/broken/net and fold consistency agree.",
        "- Baseline asset integrity: PASS."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_by_arm = {row["arm"]: row for row in source}
    ranking = sorted(ARMS, key=lambda arm: (-float(source_by_arm[arm]["stage2_BA"]), arm))
    dump_json(OUTPUT / "final_decision.json", {"status": "COMPARISON_COMPLETE", "source_full_system_ranking": ranking,
        "formal_RF_changed": False, "future_used_for_selection": False, "same_year_used_for_selection": False,
        "baseline_integrity": "PASS"})
    print("CARE_P31_CLASSIFIER_FAMILY_STAGE2_COMPARISON_DONE", flush=True)
    print(f"report={OUTPUT / 'REPORT.md'}", flush=True); print(f"table={OUTPUT / 'full_system_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
