#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np

from common import OUTPUT, dump_json, metrics, norm, read_csv, write_csv


def wilson(success, total, z=1.96):
    if total <= 0: return [None, None]
    p = success / total; d = 1 + z*z/total
    c = (p + z*z/(2*total))/d; h = z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/d
    return [max(0., c-h), min(1., c+h)]


def main():
    labels = {norm(row["sha256"]): row for row in read_csv(OUTPUT / "dataset_manifest_label_sealed.csv")}
    path = OUTPUT / "stage2_predictions_label_free.csv"
    stage2_available = path.is_file()
    predictions = read_csv(path if stage2_available else OUTPUT / "stage1_predictions_label_free.csv")
    rows = []
    for row in predictions:
        sid = norm(row["sha256"]); truth = labels[sid]; base = float(row["stage1_prob"])
        final = float(row.get("stage2_prob", base)); y = int(truth["label"])
        bp = int(base >= .5); fp = int(final >= .5)
        rows.append({**row, "label": y, "care_overlap": truth["care_overlap"],
                     "stage1_pred": bp, "stage2_prob": final, "stage2_pred": fp,
                     "fixed": int(bp != y and fp == y), "broken": int(bp == y and fp != y)})
    write_csv(OUTPUT / "external_predictions.csv", rows)
    summaries = []
    scopes = [("all", np.ones(len(rows), bool)), ("sha_disjoint", np.asarray([int(r["care_overlap"]) == 0 for r in rows]))]
    scopes += [(cohort, np.asarray([r["cohort"] == cohort for r in rows])) for cohort in ("benign", "malradar", "new")]
    y_all = np.asarray([int(r["label"]) for r in rows]); base_all = np.asarray([float(r["stage1_prob"]) for r in rows]); final_all = np.asarray([float(r["stage2_prob"]) for r in rows])
    for scope, mask in scopes:
        if not mask.any(): continue
        arms = [("Stage1_SVM", base_all)]
        if stage2_available:
            arms.append(("Stage1_plus_C43", final_all))
        for arm, probability in arms:
            result = metrics(y_all[mask], probability[mask]); result.update({"scope": scope, "arm": arm})
            result["fixed"] = int(sum(int(r["fixed"]) for r, keep in zip(rows, mask) if keep)) if arm != "Stage1_SVM" else 0
            result["broken"] = int(sum(int(r["broken"]) for r, keep in zip(rows, mask) if keep)) if arm != "Stage1_SVM" else 0
            result["net"] = result["fixed"] - result["broken"]
            result["specificity_95ci"] = wilson(result["TN"], result["TN"] + result["FP"])
            result["sensitivity_95ci"] = wilson(result["TP"], result["TP"] + result["FN"])
            summaries.append(result)
    write_csv(OUTPUT / "external_metrics.csv", summaries)
    benign = [r for r in rows if r["cohort"] == "benign"]
    write_csv(OUTPUT / "benign_25_audit.csv", benign)
    status = "FULL_STAGE1_STAGE2_COMPLETE" if stage2_available else "STAGE1_COMPLETE_STAGE2_PENDING"
    primary = {(row["scope"], row["arm"]): row for row in summaries}
    dump_json(OUTPUT / "final_decision.json", {"status": status, "target_rows": len(rows),
        "stage2_available": stage2_available, "target_labels_used_for_fit": False,
        "primary_scope": "sha_disjoint", "sha_overlap_rows": int(sum(int(r["care_overlap"]) for r in rows)),
        "warning": "MalEval benign split has only 25 simulated false positives; cohort-specific BA/AUC are undefined for single-class cohorts"})
    all_s1 = primary.get(("all", "Stage1_SVM"), {})
    all_s2 = primary.get(("all", "Stage1_plus_C43"), {})
    report = [
        "# CARE-Droid MalEval2026 frozen external evaluation",
        "",
        f"Status: **{status}**.",
        "",
        "The target labels were joined only after frozen prediction. They were not used to fit the classifier, route, C43 model, threshold, or hyperparameters.",
        "",
        "## Dataset",
        "",
        "- benign: 25 simulated false-positive applications",
        "- malradar: 200 archived malware applications",
        "- new: 30 recent malware applications",
        f"- SHA overlap with CARE 2017--2021: {sum(int(r['care_overlap']) for r in rows)} / {len(rows)}",
        "",
        "## Overall results",
        "",
        f"- Stage-I SVM: BA={all_s1.get('BA')}, AUC={all_s1.get('AUC')}, F1={all_s1.get('F1')}, errors={all_s1.get('errors')}",
        (f"- Stage-I + C43: BA={all_s2.get('BA')}, AUC={all_s2.get('AUC')}, F1={all_s2.get('F1')}, errors={all_s2.get('errors')}, fixed={all_s2.get('fixed')}, broken={all_s2.get('broken')}, net={all_s2.get('net')}"
         if stage2_available else "- Stage-I + C43: pending; Stage-I metrics have already been checkpointed."),
        "",
        "## Interpretation boundary",
        "",
        "MalEval2026 was released for fine-grained malware-behavior report evaluation, not as a population-representative binary-detection benchmark. The 25 benign applications are a deliberately difficult false-positive cohort, so specificity is reported with a Wilson interval and every benign error changes specificity by four percentage points. Single-class cohort BA and AUC are intentionally undefined.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("MALEVAL2026_EXTERNAL_EVALUATION_DONE status=" + status, flush=True)
    for result in summaries:
        if result["scope"] in {"all", "sha_disjoint", "benign", "malradar", "new"}:
            print(result, flush=True)


if __name__ == "__main__":
    main()

