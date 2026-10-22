#!/usr/bin/env python3
"""Post-hoc KEEP-collapse diagnosis. This module cannot train or modify a model."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path("artifacts/care-di25-hostego-dmd-bounded-v1")
REPORTS = ROOT / "reports"
MAX_DELTA = 2.25
EPS = 1e-12
MARGIN_EPSILON = 1e-3
YEARS = (2018, 2019, 2020, 2021)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def norm(value) -> str:
    return str(value or "").strip().lower()


def f(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_one(candidates: list[Path], label: str) -> Path:
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise RuntimeError(f"MISSING: {label}; checked: " + ", ".join(map(str, candidates)))
    return found[0]


def stats(values: list[float], percentiles=(25, 50, 75)) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    result = {"n": int(len(x))}
    if not len(x):
        return {**result, "mean": np.nan, "median": np.nan,
                **{f"p{p}": np.nan for p in percentiles}}
    result.update({"mean": float(np.mean(x)), "median": float(np.median(x))})
    result.update({f"p{p}": float(np.percentile(x, p)) for p in percentiles})
    return result


def stage1_logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def aggregate_seed_internal(path: Path) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(path):
        groups[norm(row["sha256"])].append(row)
    result = {}
    for sha, rows in groups.items():
        seeds = {int(float(row["seed"])) for row in rows}
        if seeds != {900000, 900001, 900002} or len(rows) != 3:
            raise RuntimeError(f"MISSING: exact three-seed internal output for {sha}")
        mean = lambda key: float(np.mean([f(row[key]) for row in rows]))
        result[sha] = {
            "q_neg": mean("action_neg_prob"),
            "q_keep": mean("action_keep_prob") if rows[0].get("action_keep_prob", "") not in ("", "nan", "NaN") else np.nan,
            "q_pos": mean("action_pos_prob"),
            "direction": mean("direction_value"),
            "magnitude": mean("magnitude"),
            # Mean of the actual per-seed products; do not multiply two means.
            "delta": mean("delta"),
        }
    return result


def target_action(row: dict) -> str:
    label = int(float(row["label"])); pred = int(float(row["pred_stage1"]))
    if pred == label:
        return "KEEP"
    return "NEG" if pred == 1 and label == 0 else "POS"


def argmax_action(qneg: float, qkeep: float, qpos: float) -> str:
    values = [(qneg, "NEG"), (qkeep, "KEEP"), (qpos, "POS")]
    return max(values, key=lambda item: item[0])[1]


def cohort(row: dict) -> str:
    s1 = int(float(row["stage1_correct"])) == 1
    final = int(float(row["final_correct"])) == 1
    if not s1:
        return "FIXED" if final else "UNFIXED"
    return "PRESERVED" if final else "BROKEN"


def main() -> None:
    per_path = resolve_one([
        REPORTS / "per_apk_predictions.csv",
        ROOT / "per_apk_predictions.csv",
    ], "per-APK B0/B1/B2 predictions")
    b1_seed_path = resolve_one([
        ROOT / "B1_nokeep" / "per_seed_internal.csv",
    ], "B1 per-seed internal output")
    b2_seed_path = resolve_one([
        ROOT / "B2_full" / "per_seed_internal.csv",
    ], "B2 per-seed internal output")
    input_paths = [per_path, b1_seed_path, b2_seed_path]
    hashes_before = {str(path): sha256_file(path) for path in input_paths}
    print("KEEP-collapse post-hoc inputs:")
    for path in input_paths:
        print(f"- {path}")

    all_rows = read_csv(per_path)
    by_arm = {arm: {} for arm in ("B0", "B1_nokeep", "B2_full")}
    for row in all_rows:
        arm = row.get("arm")
        if arm in by_arm:
            by_arm[arm][norm(row["sha256"])] = row
    cohorts = set(by_arm["B0"])
    if len(cohorts) != 1989 or any(set(rows) != cohorts for rows in by_arm.values()):
        raise RuntimeError("MISSING: exact aligned 1989-row B0/B1/B2 cohorts")
    b1_internal = aggregate_seed_internal(b1_seed_path)
    b2_internal = aggregate_seed_internal(b2_seed_path)
    if set(b1_internal) != cohorts or set(b2_internal) != cohorts:
        raise RuntimeError("MISSING: exact per-seed internal cohort")

    # Canonical B2 routed table with post-hoc targets and direct-head internals.
    routed = []
    for sha in sorted(cohorts):
        row = dict(by_arm["B2_full"][sha])
        if int(float(row["routed"])) != 1:
            continue
        internal = b2_internal[sha]
        row.update(internal)
        row["sha256"] = sha
        row["year"] = int(float(row["year"]))
        row["label"] = int(float(row["label"]))
        row["pred_stage1"] = int(float(row["pred_stage1"]))
        row["stage1_correct"] = int(float(row["stage1_correct"]))
        row["final_correct"] = int(float(row["final_correct"]))
        row["target_action"] = target_action(row)
        row["argmax_action"] = argmax_action(internal["q_neg"], internal["q_keep"], internal["q_pos"])
        row["correction_mass"] = internal["q_neg"] + internal["q_pos"]
        row["conditional_q_neg"] = internal["q_neg"] / (row["correction_mass"] + EPS)
        row["conditional_q_pos"] = internal["q_pos"] / (row["correction_mass"] + EPS)
        row["conditional_direction_pred"] = "POS" if internal["q_pos"] > internal["q_neg"] else "NEG"
        row["conditional_direction_correct"] = int(
            row["target_action"] in ("NEG", "POS") and row["conditional_direction_pred"] == row["target_action"])
        row["direction_strength"] = abs(internal["q_pos"] - internal["q_neg"])
        row["conditional_direction_confidence"] = row["direction_strength"] / (row["correction_mass"] + EPS)
        row["stage1_logit_abs"] = abs(stage1_logit(f(row["p_stage1"])))
        row["required_magnitude_unclipped"] = row["stage1_logit_abs"] + MARGIN_EPSILON
        row["required_magnitude"] = min(row["required_magnitude_unclipped"], MAX_DELTA)
        row["dmd_cohort"] = cohort(row)
        routed.append(row)
    errors = [row for row in routed if not row["stage1_correct"]]
    if len(errors) != 126:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected 126 routed Stage-I errors, found {len(errors)}")

    # Target and prediction distributions and full confusion matrices.
    target_rows = []; prediction_rows = []; confusion_rows = []
    scopes = [(str(year), [r for r in routed if r["year"] == year]) for year in YEARS] + [("pooled", routed)]
    for scope, rows in scopes:
        for action in ("NEG", "KEEP", "POS"):
            n = sum(r["target_action"] == action for r in rows)
            target_rows.append({"scope": scope, "year": scope, "target_action": action,
                                "count": n, "percentage": n / len(rows) if rows else np.nan})
            n_pred = sum(r["argmax_action"] == action for r in rows)
            prediction_rows.append({"scope": scope, "year": scope, "predicted_action": action,
                                    "count": n_pred, "percentage": n_pred / len(rows) if rows else np.nan})
        keep = sum(r["target_action"] == "KEEP" for r in rows)
        correction = len(rows) - keep
        target_rows.append({"scope": scope, "year": scope, "target_action": "KEEP_TO_NEG_POS_RATIO",
                            "count": "", "percentage": keep / correction if correction else np.nan})
        for target in ("NEG", "KEEP", "POS"):
            for pred in ("NEG", "KEEP", "POS"):
                confusion_rows.append({"scope": scope, "target": target, "prediction": pred,
                    "count": sum(r["target_action"] == target and r["argmax_action"] == pred for r in rows)})
    write_csv(REPORTS / "action_target_distribution.csv", target_rows)
    write_csv(REPORTS / "action_prediction_distribution.csv", prediction_rows)
    write_csv(REPORTS / "action_confusion_keep_diagnosis.csv", confusion_rows)

    outcome_rows = []
    for scope, rows0 in [(str(year), [r for r in errors if r["year"] == year]) for year in YEARS] + [("pooled", errors)]:
        for target in ("ALL", "NEG", "POS"):
            rows = rows0 if target == "ALL" else [r for r in rows0 if r["target_action"] == target]
            keep = sum(r["argmax_action"] == "KEEP" for r in rows)
            correct = sum(r["argmax_action"] == r["target_action"] for r in rows)
            wrong = sum(r["argmax_action"] not in ("KEEP", r["target_action"]) for r in rows)
            outcome_rows.append({"year": scope, "target_type": target, "n": len(rows),
                "pred_keep_n": keep, "pred_correct_direction_n": correct, "pred_wrong_direction_n": wrong,
                "pred_keep_rate": keep/len(rows) if rows else np.nan,
                "correct_direction_rate": correct/len(rows) if rows else np.nan,
                "wrong_direction_rate": wrong/len(rows) if rows else np.nan})
    write_csv(REPORTS / "stage1_error_action_outcomes.csv", outcome_rows)

    # Conditional direction diagnosis after mathematically removing KEEP competition.
    conditional_rows = []
    for scope, rows0 in [(str(year), [r for r in errors if r["year"] == year]) for year in YEARS] + [("pooled", errors)]:
        for target in ("ALL", "NEG", "POS"):
            rows = rows0 if target == "ALL" else [r for r in rows0 if r["target_action"] == target]
            correct = sum(r["conditional_direction_correct"] for r in rows)
            conditional_rows.append({"scope": scope, "target_type": target, "n": len(rows),
                "conditional_direction_correct_n": correct,
                "conditional_direction_accuracy": correct/len(rows) if rows else np.nan,
                "target_NEG_pred_NEG":sum(r["target_action"]=="NEG" and r["conditional_direction_pred"]=="NEG" for r in rows),
                "target_NEG_pred_POS":sum(r["target_action"]=="NEG" and r["conditional_direction_pred"]=="POS" for r in rows),
                "target_POS_pred_NEG":sum(r["target_action"]=="POS" and r["conditional_direction_pred"]=="NEG" for r in rows),
                "target_POS_pred_POS":sum(r["target_action"]=="POS" and r["conditional_direction_pred"]=="POS" for r in rows)})
    write_csv(REPORTS / "conditional_direction_diagnosis.csv", conditional_rows)

    # Confidence and KEEP probability by mutually interpretable cohorts.
    confidence_rows = []
    groups = {
        "STAGE1_CORRECT": [r for r in routed if r["stage1_correct"]],
        "STAGE1_WRONG": errors,
        "FIXED": [r for r in routed if r["dmd_cohort"] == "FIXED"],
        "UNFIXED": [r for r in routed if r["dmd_cohort"] == "UNFIXED"],
        "BROKEN": [r for r in routed if r["dmd_cohort"] == "BROKEN"],
        "PRESERVED": [r for r in routed if r["dmd_cohort"] == "PRESERVED"],
    }
    for name, rows in groups.items():
        for feature in ("direction_strength", "conditional_direction_confidence"):
            confidence_rows.append({"cohort": name, "feature": feature,
                                    **stats([f(r[feature]) for r in rows])})
    write_csv(REPORTS / "direction_confidence_by_cohort.csv", confidence_rows)

    y_need = np.asarray([int(not r["stage1_correct"]) for r in routed])
    scores = np.asarray([r["correction_mass"] for r in routed])
    wrong_scores = scores[y_need == 1]; correct_scores = scores[y_need == 0]
    separability = [{"scope":"pooled","n":len(routed),"stage1_wrong_n":int(np.sum(y_need)),
        "stage1_correct_n":int(np.sum(1-y_need)),"ROC_AUC":roc_auc_score(y_need,scores),
        "Average_Precision":average_precision_score(y_need,scores),
        "wrong_mean":np.mean(wrong_scores),"wrong_median":np.median(wrong_scores),
        "correct_mean":np.mean(correct_scores),"correct_median":np.median(correct_scores),
        "threshold_search":False}]
    write_csv(REPORTS / "correction_need_separability.csv", separability)

    keep_groups = {
        "STAGE1_CORRECT": [r for r in routed if r["stage1_correct"]],
        "STAGE1_WRONG": errors,
        "FP": [r for r in errors if r["target_action"] == "NEG"],
        "FN": [r for r in errors if r["target_action"] == "POS"],
        "STAGE1_CORRECT_BENIGN": [r for r in routed if r["stage1_correct"] and r["label"] == 0],
        "STAGE1_CORRECT_MALWARE": [r for r in routed if r["stage1_correct"] and r["label"] == 1],
    }
    keep_rows = []
    for name, rows in keep_groups.items():
        x=np.asarray([r["q_keep"] for r in rows],float)
        keep_rows.append({"group":name,"n":len(x),"mean":np.mean(x),"median":np.median(x),"std":np.std(x),
            "p10":np.percentile(x,10),"p25":np.percentile(x,25),"p50":np.percentile(x,50),"p75":np.percentile(x,75),"p90":np.percentile(x,90)})
    write_csv(REPORTS / "keep_probability_distribution.csv", keep_rows)

    magnitude_rows=[]
    for name, rows in [("ALL",errors),("FP",[r for r in errors if r["target_action"]=="NEG"]),("FN",[r for r in errors if r["target_action"]=="POS"])]:
        error=np.asarray([r["magnitude"]-r["required_magnitude"] for r in rows]);ae=np.abs(error)
        ratio=np.asarray([r["magnitude"]/(r["required_magnitude"]+EPS) for r in rows])
        magnitude_rows.append({"error_type":name,"n":len(rows),"MAE":np.mean(ae),"median_AE":np.median(ae),"mean_signed_error":np.mean(error),
            "mean_ratio":np.mean(ratio),"magnitude_ge_required_rate":np.mean(ratio>=1),"magnitude_ge_0p75_required_rate":np.mean(ratio>=.75),"magnitude_ge_0p50_required_rate":np.mean(ratio>=.5)})
    write_csv(REPORTS / "magnitude_head_diagnosis.csv", magnitude_rows)

    # Decompose only unresolved Stage-I errors; corrected rows are kept as an explicit audit class.
    decomposition=[]
    for row in errors:
        if row["final_correct"]:
            kind="CORRECTED"
        elif row["required_magnitude_unclipped"]>MAX_DELTA:
            kind="BOUND_LIMIT"
        elif not row["conditional_direction_correct"]:
            kind="WRONG_CONDITIONAL_DIRECTION"
        elif row["magnitude"]<row["required_magnitude"]:
            kind="MAGNITUDE_INSUFFICIENT"
        elif abs(row["delta"])<row["required_magnitude_unclipped"]:
            kind="KEEP_SUPPRESSION"
        else:
            kind="OTHER_NUMERIC_OR_ENSEMBLE_EFFECT"
        decomposition.append({"sha256":row["sha256"],"year":row["year"],"label":row["label"],"target_action":row["target_action"],
            "conditional_direction_pred":row["conditional_direction_pred"],"conditional_direction_correct":row["conditional_direction_correct"],
            "direction_factor_abs":row["direction_strength"],"correction_mass":row["correction_mass"],"magnitude":row["magnitude"],
            "required_magnitude":row["required_magnitude"],"required_magnitude_unclipped":row["required_magnitude_unclipped"],
            "effective_delta_abs":abs(row["delta"]),"B2_final_correct":row["final_correct"],"decomposition":kind})
    write_csv(REPORTS / "dmd_failure_decomposition.csv", decomposition)
    summary=[]
    for scope,rows0 in [(str(year),[r for r in decomposition if r["year"]==year]) for year in YEARS]+[("pooled",decomposition)]:
        counts=Counter(r["decomposition"] for r in rows0)
        for kind in ("CORRECTED","KEEP_SUPPRESSION","MAGNITUDE_INSUFFICIENT","WRONG_CONDITIONAL_DIRECTION","BOUND_LIMIT","OTHER_NUMERIC_OR_ENSEMBLE_EFFECT"):
            summary.append({"scope":scope,"decomposition":kind,"count":counts[kind],"percentage":counts[kind]/len(rows0) if rows0 else np.nan})
    write_csv(REPORTS / "dmd_failure_decomposition_summary.csv",summary)

    # Historical cohorts are anchored to B0 failure labels, never redefined by B2.
    b0 = by_arm["B0"]
    historical_wrong=[sha for sha,row in b0.items() if row.get("failure_type")=="A_WRONG_DIRECTION" and int(float(row["routed"]))==1]
    historical_weak=[sha for sha,row in b0.items() if row.get("failure_type")=="B_RIGHT_DIRECTION_TOO_WEAK" and int(float(row["routed"]))==1]
    if len(historical_wrong)!=27 or len(historical_weak)!=29:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: historical cohorts changed: wrong={len(historical_wrong)}, weak={len(historical_weak)}")
    routed_by={r["sha256"]:r for r in routed}
    def historical_rows(ids, weak=False):
        result=[]
        for sha in ids:
            r=routed_by[sha]
            if r["argmax_action"]=="KEEP":
                status="KEEP_BUT_CONDITIONAL_DIRECTION_CORRECT" if r["conditional_direction_correct"] else "KEEP_AND_CONDITIONAL_DIRECTION_WRONG"
            else:
                status="NONKEEP_CORRECT_DIRECTION" if r["conditional_direction_correct"] else "NONKEEP_WRONG_DIRECTION"
            row={"sha256":sha,"year":r["year"],"label":r["label"],"q_neg":r["q_neg"],"q_keep":r["q_keep"],"q_pos":r["q_pos"],
                "argmax_action":r["argmax_action"],"conditional_direction_pred":r["conditional_direction_pred"],
                "conditional_direction_correct":r["conditional_direction_correct"],"correction_mass":r["correction_mass"],
                "conditional_direction_confidence":r["conditional_direction_confidence"],"magnitude":r["magnitude"],
                "required_magnitude":r["required_magnitude"],"required_magnitude_unclipped":r["required_magnitude_unclipped"],
                "delta":r["delta"],"B2_final_correct":r["final_correct"],"status":status}
            if weak:
                row["keep_suppression"] = int(r["conditional_direction_correct"] and r["magnitude"]>=r["required_magnitude"] and abs(r["delta"])<r["required_magnitude_unclipped"])
            result.append(row)
        return result
    wrong_audit=historical_rows(historical_wrong);weak_audit=historical_rows(historical_weak,True)
    write_csv(REPORTS / "historical_wrong_direction_action_audit.csv",wrong_audit)
    write_csv(REPORTS / "historical_too_weak_action_audit.csv",weak_audit)

    # B1 acts only as a frozen NoKeep control.
    b1_control=[]
    for sha in sorted(cohorts):
        row=by_arm["B1_nokeep"][sha]
        if int(float(row["routed"]))!=1 or int(float(row["stage1_correct"]))==1:continue
        target=target_action(row);pred="POS" if b1_internal[sha]["q_pos"]>b1_internal[sha]["q_neg"] else "NEG"
        b1_control.append((target,pred))
    def direction_stats(pairs):
        return {"n":len(pairs),"direction_accuracy":np.mean([a==b for a,b in pairs]),
            "FP_to_NEG_accuracy":np.mean([b=="NEG" for a,b in pairs if a=="NEG"]),
            "FN_to_POS_accuracy":np.mean([b=="POS" for a,b in pairs if a=="POS"])}
    b2_pairs=[(r["target_action"],r["conditional_direction_pred"]) for r in errors]
    control=[{"arm":"B1_NoKeep",**direction_stats(b1_control)},{"arm":"B2_conditional_NEG_POS",**direction_stats(b2_pairs)}]
    for arm in ("B1_nokeep","B2_full"):
        rows=list(by_arm[arm].values());control_row=next(r for r in control if r["arm"].startswith("B1")==(arm=="B1_nokeep"))
        control_row["fixed"]=sum(int(float(r["routed"])) and not int(float(r["stage1_correct"])) and int(float(r["final_correct"])) for r in rows)
        control_row["broken"]=sum(int(float(r["routed"])) and int(float(r["stage1_correct"])) and not int(float(r["final_correct"])) for r in rows)
    write_csv(REPORTS / "B1_vs_B2_direction_control.csv",control)

    bins=[(0,.25,"[0,0.25)"),(.25,.5,"[0.25,0.5)"),(.5,1,"[0.5,1.0)"),(1,1.5,"[1.0,1.5)"),(1.5,2.25,"[1.5,2.25)"),(2.25,np.inf,"[2.25,+inf)")]
    margin_rows=[]
    for low,high,name in bins:
        rows=[r for r in routed if low<=r["stage1_logit_abs"]<high];wrong=[r for r in rows if not r["stage1_correct"]]
        margin_rows.append({"bin":name,"n":len(rows),"stage1_error_rate":np.mean([not r["stage1_correct"] for r in rows]) if rows else np.nan,
            "mean_q_keep":np.mean([r["q_keep"] for r in rows]) if rows else np.nan,"mean_correction_mass":np.mean([r["correction_mass"] for r in rows]) if rows else np.nan,
            "conditional_direction_accuracy":np.mean([r["conditional_direction_correct"] for r in wrong]) if wrong else np.nan,
            "fixed_rate":np.mean([r["dmd_cohort"]=="FIXED" for r in wrong]) if wrong else np.nan})
    write_csv(REPORTS / "stage1_margin_action_analysis.csv",margin_rows)

    hashes_after={str(path):sha256_file(path) for path in input_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("BLOCKED_BY_CONTRACT: an existing prediction artifact changed")
    no_retrain={"training_invoked":False,"model_weights_modified":False,"new_checkpoint_created":False,
        "threshold_search":False,"hyperparameter_search":False,"existing_prediction_only":True,
        "input_paths":list(hashes_before),"input_sha256_before":hashes_before,"input_sha256_after":hashes_after}
    (ROOT/"audits"/"NO_RETRAINING_AUDIT.json").write_text(json.dumps(no_retrain,indent=2)+"\n",encoding="utf-8")

    pooled_outcome=next(r for r in outcome_rows if r["year"]=="pooled" and r["target_type"]=="ALL")
    cond_all=next(r for r in conditional_rows if r["scope"]=="pooled" and r["target_type"]=="ALL")
    cond_fp=next(r for r in conditional_rows if r["scope"]=="pooled" and r["target_type"]=="NEG")
    cond_fn=next(r for r in conditional_rows if r["scope"]=="pooled" and r["target_type"]=="POS")
    mag_all=next(r for r in magnitude_rows if r["error_type"]=="ALL")
    dec=Counter(r["decomposition"] for r in decomposition)
    wrong_cond_correct=sum(r["conditional_direction_correct"] for r in wrong_audit)
    weak_keep=sum(r["keep_suppression"] for r in weak_audit)
    b1s,b2s=control
    # No tuned numerical cutoff is introduced. The largest mutually exclusive
    # unresolved-failure class determines the descriptive diagnosis; ties are
    # reported as mixed rather than resolved by an arbitrary threshold.
    diagnostic_counts={
        "KEEP_CLASS_COLLAPSE":dec["KEEP_SUPPRESSION"],
        "DIRECTION_HEAD_FAILURE":dec["WRONG_CONDITIONAL_DIRECTION"],
        "MAGNITUDE_HEAD_FAILURE":dec["MAGNITUDE_INSUFFICIENT"],
        "BOUND_OR_OTHER":dec["BOUND_LIMIT"]+dec["OTHER_NUMERIC_OR_ENSEMBLE_EFFECT"],
    }
    largest=max(diagnostic_counts.values());winners=[key for key,value in diagnostic_counts.items() if value==largest]
    if len(winners)!=1 or winners[0]=="BOUND_OR_OTHER":
        diagnosis="D. MIXED_FAILURE";priority="D. abandon DMD (for the present frozen formulation)"
    elif winners[0]=="KEEP_CLASS_COLLAPSE":
        diagnosis="A. KEEP_CLASS_COLLAPSE";priority="A. separate correction gate from direction"
    elif winners[0]=="DIRECTION_HEAD_FAILURE":
        diagnosis="B. DIRECTION_HEAD_FAILURE";priority="B. redesign direction representation"
    else:
        diagnosis="C. MAGNITUDE_HEAD_FAILURE";priority="C. redesign magnitude"
    lines=["CARE-Droid DMD-BC KEEP-collapse post-hoc diagnosis","",
        "NO TRAINING / NO MODEL MODIFICATION / NO NEW VARIANT / NO THRESHOLD SEARCH","",
        f"Q1. Routed Stage-I errors predicted KEEP: {pooled_outcome['pred_keep_n']}/126 ({pooled_outcome['pred_keep_rate']:.6f}).",
        f"Q2. Conditional NEG/POS direction accuracy: {cond_all['conditional_direction_accuracy']:.6f}.",
        f"Q3. FN -> POS conditional accuracy: {cond_fn['conditional_direction_accuracy']:.6f}.",
        f"Q4. FP -> NEG conditional accuracy: {cond_fp['conditional_direction_accuracy']:.6f}.",
        f"Q5. need_correction_score separability: ROC-AUC={separability[0]['ROC_AUC']:.6f}, PR-AUC={separability[0]['Average_Precision']:.6f}.",
        f"Q6. Magnitude adequacy: >=required {mag_all['magnitude_ge_required_rate']:.6f}; >=0.75*required {mag_all['magnitude_ge_0p75_required_rate']:.6f}; MAE={mag_all['MAE']:.6f}.",
        f"Q7. Unfixed decomposition: KEEP_SUPPRESSION={dec['KEEP_SUPPRESSION']}, MAGNITUDE_INSUFFICIENT={dec['MAGNITUDE_INSUFFICIENT']}, WRONG_CONDITIONAL_DIRECTION={dec['WRONG_CONDITIONAL_DIRECTION']}, BOUND_LIMIT={dec['BOUND_LIMIT']}. Corrected={dec['CORRECTED']}.",
        f"Q8. Historical 27 Wrong Direction with correct conditional direction: {wrong_cond_correct}/27; inspect status counts in historical_wrong_direction_action_audit.csv.",
        f"Q9. Historical 29 Too Weak classified as KEEP suppression: {weak_keep}/29.",
        f"Q10. B1 direction accuracy={b1s['direction_accuracy']:.6f}; B2 conditional direction accuracy={b2s['direction_accuracy']:.6f}; absolute gap={abs(b1s['direction_accuracy']-b2s['direction_accuracy']):.6f}.",
        f"Q11. Best-supported current diagnosis: {diagnosis}.",
        f"Q12. Diagnostic next-mechanism priority: {priority}. This is a judgment only; no implementation is authorized.","",
        "Required-magnitude note: capped required_magnitude is used for head adequacy; uncapped abs(stage1_logit)+1e-3 is retained separately so BOUND_LIMIT remains identifiable.",
        "All quantities are post-hoc diagnostics from saved OOF predictions and per-seed internal outputs."]
    (REPORTS/"KEEP_COLLAPSE_DIAGNOSIS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("KEEP_COLLAPSE_POSTHOC_DIAGNOSIS_PASS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
