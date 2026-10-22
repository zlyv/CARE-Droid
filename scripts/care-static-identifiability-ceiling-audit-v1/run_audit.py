#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (OUTPUT, asset_record, discover_assets, dump_json, ensure_output_boundary,
                    fraction, norm, read_csv, write_csv)

ARMS = ("LR", "RF", "XGB", "SVM")
YEARS = (2017, 2018, 2019, 2020, 2021)


def family_map(source_path: Path, future_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    probabilities: dict[str, dict[str, float]] = defaultdict(dict); folds: dict[str, int] = {}
    for path in (source_path, future_path):
        for row in read_csv(path):
            arm = str(row.get("arm", "")).strip().upper()
            if arm not in ARMS: continue
            sid = norm(row.get("sha256")); probabilities[sid][arm] = float(row["probability"])
            if str(row.get("fold", "")).strip(): folds[sid] = int(float(row["fold"]))
    incomplete = [sid for sid, value in probabilities.items() if set(value) != set(ARMS)]
    if incomplete: raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete family probabilities {incomplete[:5]}")
    return dict(probabilities), folds


def build_sample_rows(assets: dict[str, Path]) -> list[dict[str, Any]]:
    probabilities, folds = family_map(assets["source_family"], assets["future_family"]); rows = []
    for raw in read_csv(assets["temporal"]):
        sid = norm(raw["sha256"])
        if sid not in probabilities: raise RuntimeError("BLOCKED_BY_CONTRACT: family prediction missing " + sid)
        y = int(float(raw["label"])); year = int(float(raw["year"])); route = bool(int(float(raw["routed"])))
        stage1 = float(raw["stage1_prob"]); expert = float(raw["expert_prob"])
        if abs(stage1 - probabilities[sid]["SVM"]) > 1e-8:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: SVM probability mismatch {sid}")
        family_wrong = {arm: int((value >= .5) != y) for arm, value in probabilities[sid].items()}
        svm_wrong = family_wrong["SVM"]; expert_wrong = int((expert >= .5) != y)
        current_wrong = expert_wrong if route else svm_wrong
        row = {"sha256": sid, "year": year, "fold": folds.get(sid, ""), "label": y,
               "confusion": raw.get("confusion", ""), "routed": int(route),
               "stage1_prob": stage1, "expert_prob": expert, "stage1_wrong": svm_wrong,
               "expert_wrong": expert_wrong, "current_wrong": current_wrong,
               "family_wrong_count": sum(family_wrong.values()),
               "family_unanimous_wrong": int(all(family_wrong.values())),
               "family_majority_wrong": int(sum(family_wrong.values()) >= 3),
               "family_plus_expert_unanimous_wrong": int(all(family_wrong.values()) and expert_wrong),
               "route_missed_error": int(svm_wrong and not route),
               "routed_error": int(svm_wrong and route),
               "fixed": int(route and svm_wrong and not expert_wrong),
               "broken": int(route and not svm_wrong and expert_wrong),
               "persistent_routed_error": int(route and svm_wrong and expert_wrong),
               "route_limited_oracle_wrong": int((not route and svm_wrong) or (route and svm_wrong and expert_wrong)),
               "route100_oracle_wrong": int(svm_wrong and expert_wrong),
               "family_oracle_wrong": int(all(family_wrong.values())),
               "family_plus_expert_oracle_wrong": int(all(family_wrong.values()) and expert_wrong)}
        for arm in ARMS:
            row[f"{arm}_prob"] = probabilities[sid][arm]; row[f"{arm}_wrong"] = family_wrong[arm]
        rows.append(row)
    if len(rows) != 2788: raise RuntimeError("BLOCKED_BY_CONTRACT: merged sample count changed")
    return rows


def annual(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for year in YEARS:
        local = [row for row in rows if row["year"] == year]; n = len(local)
        errors = sum(row["stage1_wrong"] for row in local); routed = sum(row["routed"] for row in local)
        routed_errors = sum(row["routed_error"] for row in local); fixed = sum(row["fixed"] for row in local)
        broken = sum(row["broken"] for row in local); current_errors = sum(row["current_wrong"] for row in local)
        consensus = sum(row["family_plus_expert_unanimous_wrong"] for row in local)
        result.append({"year": year, "n": n, "stage1_errors": errors, "routed": routed,
            "routed_stage1_errors": routed_errors, "route_error_recall": fraction(routed_errors, errors),
            "route_precision": fraction(routed_errors, routed), "unrouted_stage1_errors": errors-routed_errors,
            "fixed": fixed, "broken": broken, "net": fixed-broken, "current_errors": current_errors,
            "routed_error_fix_rate": fraction(fixed, routed_errors),
            "route_limited_oracle_errors": sum(row["route_limited_oracle_wrong"] for row in local),
            "route100_oracle_errors": sum(row["route100_oracle_wrong"] for row in local),
            "family_oracle_errors": sum(row["family_oracle_wrong"] for row in local),
            "family_plus_expert_oracle_errors": consensus,
            "consensus_error_share_of_stage1_errors": fraction(consensus, errors),
            "stage1_accuracy": fraction(n-errors, n), "current_accuracy": fraction(n-current_errors, n),
            "route_limited_oracle_accuracy": fraction(n-sum(row["route_limited_oracle_wrong"] for row in local), n),
            "route100_oracle_accuracy": fraction(n-sum(row["route100_oracle_wrong"] for row in local), n),
            "family_plus_expert_oracle_accuracy": fraction(n-consensus, n)})
    return result


def confusion_funnel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result=[]
    for year in YEARS:
        for kind in ("FN", "FP"):
            local=[row for row in rows if row["year"]==year and row["confusion"]==kind]
            result.append({"year":year,"error_type":kind,"stage1_errors":len(local),
                "routed":sum(row["routed"] for row in local),"fixed":sum(row["fixed"] for row in local),
                "persistent_routed":sum(row["persistent_routed_error"] for row in local),
                "unrouted":sum(row["route_missed_error"] for row in local),
                "family_plus_expert_unanimous_wrong":sum(row["family_plus_expert_unanimous_wrong"] for row in local)})
    return result


def consensus_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result=[]
    for year in YEARS:
        local=[row for row in rows if row["year"]==year]
        for wrong_count in range(5):
            selected=[row for row in local if row["family_wrong_count"]==wrong_count]
            result.append({"year":year,"family_wrong_count":wrong_count,"samples":len(selected),
                "expert_wrong":sum(row["expert_wrong"] for row in selected),
                "stage1_wrong":sum(row["stage1_wrong"] for row in selected),
                "current_wrong":sum(row["current_wrong"] for row in selected)})
    return result


def collision_strata(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows=[row for row in read_csv(path) if norm(row.get("representation"))=="hidden" and norm(row.get("distance"))=="cosine"]
    output=[]
    for category in ("FIXED","BROKEN","STABLE_CORRECT","PERSISTENT_ERROR"):
        local=[row for row in rows if row.get("category")==category]
        severe=[row for row in local if float(row["purity_k10"])<=.5 and float(row["collision_margin"])<=0]
        output.append({"category":category,"n":len(local),"mean_purity_k10":float(np.mean([float(r["purity_k10"]) for r in local])) if local else float("nan"),
            "opposite_nearer_rate":fraction(sum(float(r["collision_margin"])<=0 for r in local),len(local)),
            "severe_collision_n":len(severe),"severe_collision_rate":fraction(len(severe),len(local))})
    failures=[row for row in rows if row.get("category") in {"BROKEN","PERSISTENT_ERROR"}]
    successes=[row for row in rows if row.get("category") in {"FIXED","STABLE_CORRECT"}]
    failure_rate=fraction(sum(float(r["purity_k10"])<=.5 and float(r["collision_margin"])<=0 for r in failures),len(failures))
    success_rate=fraction(sum(float(r["purity_k10"])<=.5 and float(r["collision_margin"])<=0 for r in successes),len(successes))
    failure_folds=set(int(float(r["fold"])) for r in failures if float(r["purity_k10"])<=.5 and float(r["collision_margin"])<=0)
    return output,{"failure_n":len(failures),"success_n":len(successes),"failure_severe_collision_rate":failure_rate,
        "success_severe_collision_rate":success_rate,"enrichment":failure_rate-success_rate,
        "failure_collision_folds":len(failure_folds)}


def hash_assets(assets: dict[str, Path], phase: str) -> list[dict[str, Any]]:
    result=[]
    for key,path in assets.items():
        result.append(asset_record(path,key))
        print(f"SICA_ASSET_HASH phase={phase} asset={key}",flush=True)
    return result


def main() -> None:
    ensure_output_boundary(); OUTPUT.mkdir(parents=True,exist_ok=True); assets=discover_assets()
    before=hash_assets(assets,"before"); dump_json(OUTPUT/"baseline_asset_manifest_before.json",before)
    protocol={"schema_version":"1.0.0","task":"empirical static identifiability ceiling audit",
        "source_2017_is_primary":True,"future_2018_2021_is_frozen_diagnostic_only":True,
        "model_training":False,"threshold_tuning":False,"feature_selection":False,
        "family_members":ARMS,"severe_collision":"hidden cosine purity_k10<=0.5 and opposite-neighbor distance<=same-label distance",
        "source_route_limit_gate":"unrouted Stage-I errors / Stage-I errors >= 0.10",
        "source_static_overlap_gate":"all LR/RF/XGB/SVM and Direct-C43 wrong / SVM errors >= 0.20",
        "source_collision_gate":"failure severe-collision rate exceeds success by >=0.10 in >=3 folds",
        "interpretation":"empirical overlap is not a proof of Bayes error or theoretical unidentifiability"}
    dump_json(OUTPUT/"protocol.json",protocol)
    samples=build_sample_rows(assets); annual_rows=annual(samples); funnels=confusion_funnel(samples)
    consensus=consensus_summary(samples); collision_rows,collision_summary=collision_strata(assets["neighbors"])
    write_csv(OUTPUT/"sample_error_ceiling.csv",samples); write_csv(OUTPUT/"annual_error_funnel.csv",annual_rows)
    write_csv(OUTPUT/"confusion_error_funnel.csv",funnels); write_csv(OUTPUT/"model_consensus_summary.csv",consensus)
    write_csv(OUTPUT/"source_collision_strata.csv",collision_rows); dump_json(OUTPUT/"source_collision_summary.json",collision_summary)

    source=next(row for row in annual_rows if row["year"]==2017)
    route_limit=fraction(source["unrouted_stage1_errors"],source["stage1_errors"])>=.10
    overlap=source["consensus_error_share_of_stage1_errors"]>=.20
    collision=(collision_summary["enrichment"]>=.10 and collision_summary["failure_collision_folds"]>=3)
    active=sum((route_limit,overlap,collision))
    status=("MIXED_ROUTING_AND_STATIC_OVERLAP_SUPPORTED" if active>=2 else
            "SOURCE_ROUTING_LIMIT_SUPPORTED" if route_limit else
            "SOURCE_EMPIRICAL_STATIC_OVERLAP_SUPPORTED" if overlap or collision else
            "RESIDUAL_STATIC_LIMIT_NOT_IDENTIFIED")
    decision={"status":status,"criteria":{"source_route_limit":route_limit,"source_cross_model_static_overlap":overlap,
        "source_hidden_collision_enrichment":collision},"source":source,"source_collision":collision_summary,
        "future_rows_used_for_selection":False,"future_is_diagnostic_only":True,"model_change_authorized":False,
        "theoretical_unidentifiability_claimed":False,
        "next_action":"separate routing headroom from empirical representation overlap; do not add another C43 feature unless new independent observation or supervision is available"}
    dump_json(OUTPUT/"final_decision.json",decision)
    after=hash_assets(assets,"after"); dump_json(OUTPUT/"baseline_asset_manifest_after.json",after)
    by_before={row["absolute_path"]:row for row in before}; by_after={row["absolute_path"]:row for row in after}
    changed=[path for path in sorted(by_before) if path not in by_after or by_before[path]["sha256"]!=by_after[path]["sha256"]]
    integrity={"status":"PASS" if not changed else "FAIL","asset_count":len(before),"changed":changed}
    dump_json(OUTPUT/"baseline_integrity_check.json",integrity)
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")

    lines=["# CARE-Droid empirical static identifiability ceiling audit v1","",f"Final status: **{status}**.","",
        "This is an audit of frozen predictions and OOF representations. It does not train a classifier and does not prove a theoretical Bayes-error bound.","",
        "## Annual error funnel","","| Year | Stage-I errors | Routed errors | Route recall | Fixed | Broken | Current errors | Route-limited oracle errors | Route100 oracle errors | Family+expert oracle errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in annual_rows:
        lines.append(f"| {row['year']} | {row['stage1_errors']} | {row['routed_stage1_errors']} | {row['route_error_recall']:.3f} | {row['fixed']} | {row['broken']} | {row['current_errors']} | {row['route_limited_oracle_errors']} | {row['route100_oracle_errors']} | {row['family_plus_expert_oracle_errors']} |")
    lines += ["","## Source collision","",
        f"- Failure severe-collision rate: {collision_summary['failure_severe_collision_rate']:.3f}.",
        f"- Success severe-collision rate: {collision_summary['success_severe_collision_rate']:.3f}.",
        f"- Enrichment: {collision_summary['enrichment']:+.3f}; affected failure folds: {collision_summary['failure_collision_folds']}.","",
        "## Boundary","","The family/expert oracle is label-informed and unattainable; it only quantifies complementarity. Cross-model consensus errors and local opposite-label collisions are empirical overlap, not proof that static analysis can never solve a sample.","",
        "2018-2021 are frozen diagnostics and do not select a model, feature, threshold, route, or decision rule.","Formal asset integrity: PASS.",""]
    rendered="\n".join(lines); (OUTPUT/"REPORT.md").write_text(rendered,encoding="utf-8"); (OUTPUT/"report.md").write_text(rendered,encoding="utf-8")
    print("STATIC_IDENTIFIABILITY_CEILING_STATUS="+status,flush=True)
    print("report="+str(OUTPUT/"REPORT.md"),flush=True)


if __name__ == "__main__": main()
