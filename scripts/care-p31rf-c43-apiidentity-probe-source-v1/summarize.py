#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path("artifacts/care-p31rf-c43-apiidentity-probe-source-v1")

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def f(row,key):return float(row[key])

def main():
    quality=read(ROOT/"direction_quality.csv");folds=read(ROOT/"per_fold_direction.csv");groups=read(ROOT/"fp_tp_fn_tn_metrics.csv")
    paired=read(ROOT/"paired_direction.csv");coverage=json.loads((ROOT/"coverage_cost.json").read_text());collapse=json.loads((ROOT/"representation_collapse.json").read_text());decision=json.loads((ROOT/"advance_decision.json").read_text())
    by={r["Representation"]:r for r in quality};h0=by["H0_ROLE"];h1=by["H1_ROLE_API"]
    lines=["CARE-Droid P31-RF C43 API-identity representation probe source v1","",
      "Strict 2017 five-fold OOF only. P31-RF, RF-anchored MeanOpposition DI25, route budget, threshold, and max_delta are unchanged.",
      "No neural training, future row, API pair, path, or APK reanalysis. Exact APIs are stable canonical-signature hashes from the frozen endpoint cache.",
      "The current run cannot recover readable API names or frozen per-Host C43 attention; it audits all routed Hosts and does not fabricate attention weights.","",
      f"routed={coverage['routed_rows']} routed_errors={coverage['routed_errors']} FN={coverage['FN']} FP={coverage['FP']}","","Direction quality:"]
    for r in quality:
        lines.append(f"{r['Representation']}: overall={f(r,'OverallDirectionAcc'):.6f} FN={f(r,'FNDirectionAcc'):.6f} FP={f(r,'FPDirectionAcc'):.6f} wrong={r['WrongDirection']} ambiguous={r['Ambiguous']} mean_aligned={f(r,'MeanAligned'):.6f} fold_std={f(r,'FoldStd'):.6f}")
    lines += ["","Representation sufficiency:",
      f"nearest C43 Host correction-sign disagreement={collapse.get('nearest_neighbor_correction_sign_disagreement',float('nan')):.6f}",
      f"nearest C43 Host API-set-difference fraction={collapse.get('nearest_neighbor_api_sets_differ_fraction',float('nan')):.6f}",
      f"exact-role/opposite-direction Hosts with different APIs={collapse.get('exact_role_opposite_direction_different_api_hosts',0)}/{collapse.get('exact_role_with_opposite_direction_hosts',0)}","",
      "FP-vs-TP and FN-vs-TN (routed only):"]
    for r in groups:
        if r["Representation"] in ("H0_ROLE","H1_ROLE_API","COUNT_ONLY"):
            lines.append(f"{r['Comparison']} {r['Representation']}: BA={f(r,'BA'):.6f} AUC={f(r,'AUC'):.6f} F1={f(r,'F1'):.6f} n={r['n']}")
    lines += ["","Paired direction:"]
    for r in paired:lines.append(f"{r['comparison']}: candidate_only={r['candidate_only_correct']} control_only={r['control_only_correct']} net={r['paired_net']} exact_p={f(r,'exact_p'):.6f}")
    lines += ["","Coverage/cost:",
      f"hosts={coverage['total_hosts']} API_occurrences={coverage['total_api_occurrences']} unique_API_hashes={coverage['unique_api_identity_hashes']} mean_UNK_rate={coverage['mean_unknown_rate']:.6f}",
      f"cache reconstruction mean={coverage['mean_reconstruction_seconds_per_apk']:.6f}s/APK failures={coverage['failure_count']}",
      "This timing is incremental reconstruction from an existing endpoint cache, not original APK call-graph extraction cost.","",
      "Predeclared E1 gate:",
      f"FP_gain={decision['observed']['FP_gain']:+.6f} FN_change={decision['observed']['FN_change']:+.6f} overall_gain={decision['observed']['overall_gain']:+.6f} nonnegative_folds={decision['observed']['nonnegative_folds']}/5",
      "E1_ADVANCE_DECISION="+("PASS" if decision["advance_api_identity_to_c43"] else "STOP"),
      "Future evaluation is not authorized by this source probe."]
    if not decision["advance_api_identity_to_c43"]:
        lines += ["","Stopping rule applied: do not build API Pair/Path or retrain C43 because API identity did not pass the source direction gate."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("P31RF_C43_APIIDENTITY_SUMMARY_PASS");print("\n".join(lines))

if __name__=="__main__":main()

