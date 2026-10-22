#!/usr/bin/env python3
"""Read-only statistical analysis for the Host complement candidate audit."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_common import (
    AUDIT, BOOTSTRAPS, FEATURES, PREDICTIONS, REPORTS, ROOT, SEED, THRESHOLD,
    average_precision, binary_auc, clopper_pearson, ensure_dirs, load_json,
    classification_metrics, norm, read_csv, spearman, write_csv, write_json,
)


PRIMARY = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
CORE = ["A", "B", "C", "ABC"]


def col(rows, names, required=True):
    for name in names:
        if rows and name in rows[0]:
            return name
    if required:
        raise RuntimeError(f"Missing required column; tried {names}")
    return None


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_prior(manifest):
    score_rows = read_csv(Path(manifest["strict_oof_probe_file"]))
    failure_rows = read_csv(Path(manifest["failure_type_file"]))
    score_id = col(score_rows, ["sha256", "sample_id"])
    fail_id = col(failure_rows, ["sha256", "sample_id"])
    failure = col(failure_rows, ["failure_type", "failure", "cohort", "status"])
    prior = {}
    # The canonical R4 audit is long-form: one task/representation row per APK.
    if "representation" in score_rows[0]:
        probability = col(score_rows, ["probability", "oof_probability", "prob", "p", "score"])
        mapping = {
            "B0_graph_token32": "graph", "B1_critical_hostego24": "critical",
            "B2_attended_hostego24": "attended",
        }
        for r in score_rows:
            if r.get("task") != "need" or r.get("representation") not in mapping:
                continue
            prior.setdefault(norm(r[score_id]), {})[mapping[r["representation"]]] = float(r[probability])
    else:
        graph = col(score_rows, ["B0_graph_token32", "B0_prob", "p_B0", "graph_prob"])
        critical = col(score_rows, ["B1_critical_hostego24", "B1_prob", "p_B1", "critical_prob"])
        attended = col(score_rows, ["B2_attended_hostego24", "B2_prob", "p_B2", "attended_prob"])
        for r in score_rows:
            prior[norm(r[score_id])] = {
                "graph": float(r[graph]), "critical": float(r[critical]),
                "attended": float(r[attended]),
            }
    for r in failure_rows:
        prior.setdefault(norm(r[fail_id]), {})["failure"] = r[failure]
    return prior


def stratified_indices(y, years, rng):
    chunks = []
    for year in sorted(set(years)):
        for target in (0, 1):
            idx = np.flatnonzero((years == year) & (y == target))
            if len(idx):
                chunks.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(chunks)


def percentile_ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def resample_mask_stratified(mask, years, rng):
    chunks = []
    for year in sorted(set(years)):
        idx = np.flatnonzero(mask & (years == year))
        if len(idx): chunks.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(chunks)


def main():
    ensure_dirs()
    manifest = load_json(ROOT / "preflight" / "source_manifest.json")
    cohort = {norm(r["sha256"]): r for r in read_csv(Path(manifest["cohort_file"]))}
    prediction_rows = read_csv(PREDICTIONS / "candidate_oof_predictions.csv")
    prior = load_prior(manifest)
    schema = load_json(FEATURES / "candidate_feature_schema.json")
    ids = sorted(s for s, r in cohort.items() if int(float(r["route_DI"])) == 1)
    if len(ids) != int(manifest["routed_n"]):
        raise RuntimeError(f"Routed count changed: expected {manifest['routed_n']}, got {len(ids)}")
    missing_prior = sorted(set(ids) - set(prior))
    if missing_prior:
        raise RuntimeError(f"Prior diagnostic scores missing {len(missing_prior)} routed IDs")
    prediction_map = defaultdict(dict)
    for r in prediction_rows:
        prediction_map[r["candidate"]][norm(r["sha256"])] = float(r["probability"])
    candidate_names = list(schema["fixed_primary_sets"]) + list(schema["fixed_secondary_sets"])
    for name in candidate_names:
        if set(prediction_map[name]) != set(ids):
            raise RuntimeError(f"Candidate {name} does not have exact routed-cohort OOF predictions")

    labels = np.asarray([int(float(cohort[s]["label"])) for s in ids])
    stage1 = np.asarray([float(cohort[s]["p_stage1"]) for s in ids])
    y = ((stage1 >= THRESHOLD) != labels).astype(int)
    years = np.asarray([int(float(cohort[s]["year"])) for s in ids])
    probs = {name: np.asarray([prediction_map[name][s] for s in ids]) for name in candidate_names}
    graph = np.asarray([prior[s]["graph"] for s in ids])
    critical = np.asarray([prior[s]["critical"] for s in ids])
    attended = np.asarray([prior[s]["attended"] for s in ids])
    failure = np.asarray([prior[s].get("failure", "") for s in ids], dtype=object)

    # General OOF probe performance.
    metric_rows = []
    for candidate, p in probs.items():
        for year in [2018, 2019, 2020, 2021, "pooled"]:
            mask = np.ones(len(y), dtype=bool) if year == "pooled" else years == year
            metric_rows.append({"candidate": candidate, "year": year, **classification_metrics(y[mask], p[mask])})
    write_csv(REPORTS / "need_correction_metrics.csv", metric_rows)

    # Graph-negative conditional complementarity, exact intervals, and bootstrap CS.
    conditional_rows, separation_rows = [], []
    graph_negative = graph < THRESHOLD
    rng = np.random.default_rng(SEED)
    for candidate in PRIMARY:
        pred = probs[candidate] >= THRESHOLD
        err = graph_negative & (y == 1)
        safe = graph_negative & (y == 0)
        rescue_n, error_n = int(np.sum(pred & err)), int(np.sum(err))
        false_n, safe_n = int(np.sum(pred & safe)), int(np.sum(safe))
        rescue = rescue_n / error_n if error_n else np.nan
        false = false_n / safe_n if safe_n else np.nan
        rlo, rhi = clopper_pearson(rescue_n, error_n)
        flo, fhi = clopper_pearson(false_n, safe_n)
        cs = rescue - false
        conditional_rows.append({
            "candidate": candidate, "graph_negative_error_n": error_n,
            "graph_negative_error_rescued": rescue_n, "rescue_rate": rescue,
            "rescue_ci_low": rlo, "rescue_ci_high": rhi,
            "graph_negative_safe_correct_n": safe_n,
            "false_rescue_candidate_n": false_n, "false_rescue_rate": false,
            "false_rescue_ci_low": flo, "false_rescue_ci_high": fhi,
            "conditional_separation": cs,
        })
        boot = np.empty(BOOTSTRAPS, dtype=float)
        for b in range(BOOTSTRAPS):
            e = resample_mask_stratified(err, years, rng)
            s = resample_mask_stratified(safe, years, rng)
            boot[b] = float(np.mean(pred[e])) - float(np.mean(pred[s]))
        lo, hi = percentile_ci(boot)
        separation_rows.append({
            "candidate": candidate, "bootstrap_iterations": BOOTSTRAPS,
            "bootstrap_seed": SEED, "conditional_separation": cs,
            "ci_low": lo, "ci_high": hi,
        })
    write_csv(REPORTS / "conditional_rescue.csv", conditional_rows)
    write_csv(REPORTS / "conditional_separation_bootstrap.csv", separation_rows)

    # Historical failure-type rescue.
    failure_rows = []
    for candidate in PRIMARY:
        positive = probs[candidate] >= THRESHOLD
        for group, mask in [
            ("WrongDirection", failure == "A_WRONG_DIRECTION"),
            ("TooWeak", failure == "B_RIGHT_DIRECTION_TOO_WEAK"),
        ]:
            for graph_scope, base_mask in [
                ("GraphNegative", graph_negative), ("All", np.ones(len(y), dtype=bool))
            ]:
                m = mask & base_mask
                k, n = int(np.sum(positive & m)), int(np.sum(m))
                lo, hi = clopper_pearson(k, n)
                failure_rows.append({
                    "candidate": candidate, "failure_type": group,
                    "scope": graph_scope, "n": n, "rescued": k,
                    "rescue_rate": k / n if n else np.nan,
                    "ci_low": lo, "ci_high": hi,
                    "graph_or_candidate_captured": "",
                })
    # Add the all-cohort Graph OR candidate capture without changing candidate thresholds.
    for row in failure_rows:
        mask = failure == ("A_WRONG_DIRECTION" if row["failure_type"] == "WrongDirection" else "B_RIGHT_DIRECTION_TOO_WEAK")
        positive = probs[row["candidate"]] >= THRESHOLD
        row["graph_or_candidate_captured"] = int(np.sum(mask & ((graph >= THRESHOLD) | positive)))
        row["graph_or_candidate_denominator"] = int(np.sum(mask))
    write_csv(REPORTS / "failure_type_rescue.csv", failure_rows)
    write_csv(REPORTS / "wrong_direction_candidate_rescue.csv", [r for r in failure_rows if r["failure_type"] == "WrongDirection"])
    write_csv(REPORTS / "too_weak_candidate_rescue.csv", [r for r in failure_rows if r["failure_type"] == "TooWeak"])

    # Existing Host miss: Graph, Critical Host and Attended Host all negative.
    existing_miss = graph_negative & (critical < THRESHOLD) & (attended < THRESHOLD)
    existing_rows = []
    for candidate in CORE:
        positive = probs[candidate] >= THRESHOLD
        for group, group_mask in [
            ("NeedCorrection", y == 1),
            ("WrongDirection", failure == "A_WRONG_DIRECTION"),
        ]:
            m = existing_miss & group_mask
            k, n = int(np.sum(positive & m)), int(np.sum(m))
            lo, hi = clopper_pearson(k, n)
            existing_rows.append({
                "candidate": candidate, "cohort": group, "n": n,
                "rescued": k, "rescue_rate": k / n if n else np.nan,
                "ci_low": lo, "ci_high": hi,
            })
    write_csv(REPORTS / "existing_host_miss_rescue.csv", existing_rows)

    current_rows = []
    for name, p in [("CurrentCritical", critical), ("CurrentAttended", attended)]:
        positive = p >= THRESHOLD
        for group, mask in [
            ("GraphMissedNeedCorrection", graph_negative & (y == 1)),
            ("GraphMissedWrongDirection", graph_negative & (failure == "A_WRONG_DIRECTION")),
        ]:
            k, n = int(np.sum(positive & mask)), int(np.sum(mask))
            lo, hi = clopper_pearson(k, n)
            current_rows.append({"block": name, "cohort": group, "n": n, "captured": k,
                                 "capture_rate": k / n if n else np.nan, "ci_low": lo, "ci_high": hi})
    write_csv(REPORTS / "current_host_descriptive_comparison.csv", current_rows)

    # Per-sample Graph-missed Wrong Direction matrix.
    matrix_rows = []
    wd_missed = graph_negative & (failure == "A_WRONG_DIRECTION")
    for i in np.flatnonzero(wd_missed):
        row = {
            "sha256": ids[i], "year": int(years[i]), "fold": int(float(cohort[ids[i]]["fold"])),
            "label": int(float(cohort[ids[i]]["label"])), "failure_type": failure[i],
            "graph_score": graph[i], "critical_score": critical[i], "attended_score": attended[i],
            "graph_positive": int(graph[i] >= THRESHOLD),
            "critical_positive": int(critical[i] >= THRESHOLD),
            "attended_positive": int(attended[i] >= THRESHOLD),
            "existing_host_miss": int(existing_miss[i]),
        }
        for candidate in ["A", "B", "C", "ABC"]:
            row[f"{candidate}_score"] = probs[candidate][i]
            row[f"{candidate}_positive"] = int(probs[candidate][i] >= THRESHOLD)
            row[f"{candidate}_new_rescue"] = int(existing_miss[i] and probs[candidate][i] >= THRESHOLD)
        matrix_rows.append(row)
    write_csv(REPORTS / "graph_missed_wd_candidate_matrix.csv", matrix_rows)

    # Stratified paired bootstrap: general AUC/AP improvements.
    paired_rows = []
    general_series = {k: probs[k] for k in CORE}
    general_series.update({"CurrentCritical": critical, "CurrentAttended": attended})
    requested_pairs = [
        ("A", "B"), ("A", "C"), ("A", "ABC"), ("B", "C"),
        ("B", "ABC"), ("C", "ABC"),
        ("CurrentCritical", "ABC"), ("CurrentAttended", "ABC"),
    ]
    rng = np.random.default_rng(SEED + 1)
    for baseline, candidate in requested_pairs:
        pb, pc = general_series[baseline], general_series[candidate]
        point_auc = binary_auc(y, pc) - binary_auc(y, pb)
        point_ap = average_precision(y, pc) - average_precision(y, pb)
        ba, bp = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
        for b in range(BOOTSTRAPS):
            idx = stratified_indices(y, years, rng)
            ba[b] = binary_auc(y[idx], pc[idx]) - binary_auc(y[idx], pb[idx])
            bp[b] = average_precision(y[idx], pc[idx]) - average_precision(y[idx], pb[idx])
        alo, ahi = percentile_ci(ba)
        plo, phi = percentile_ci(bp)
        paired_rows.append({
            "baseline": baseline, "candidate": candidate,
            "bootstrap_iterations": BOOTSTRAPS, "bootstrap_seed": SEED + 1,
            "delta_auc": point_auc, "delta_auc_ci_low": alo, "delta_auc_ci_high": ahi,
            "delta_pr_auc": point_ap, "delta_pr_auc_ci_low": plo, "delta_pr_auc_ci_high": phi,
        })
    # Each candidate versus random baseline: AUC-0.5 and AP-prevalence.
    prevalence = float(np.mean(y))
    for candidate in CORE:
        p = probs[candidate]
        point_auc = binary_auc(y, p) - 0.5
        point_ap = average_precision(y, p) - prevalence
        ba, bp = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
        for b in range(BOOTSTRAPS):
            idx = stratified_indices(y, years, rng)
            ba[b] = binary_auc(y[idx], p[idx]) - 0.5
            bp[b] = average_precision(y[idx], p[idx]) - float(np.mean(y[idx]))
        alo, ahi = percentile_ci(ba)
        plo, phi = percentile_ci(bp)
        paired_rows.append({
            "baseline": "Random", "candidate": candidate,
            "bootstrap_iterations": BOOTSTRAPS, "bootstrap_seed": SEED + 1,
            "delta_auc": point_auc, "delta_auc_ci_low": alo, "delta_auc_ci_high": ahi,
            "delta_pr_auc": point_ap, "delta_pr_auc_ci_low": plo, "delta_pr_auc_ci_high": phi,
        })
    write_csv(REPORTS / "paired_bootstrap.csv", paired_rows)

    # Size shortcut audit over label-free features.
    # Stream the wide (role-pair/transition) CSV and retain only routed summary
    # columns; this avoids materialising millions of unused sparse cells.
    feature_rows = {}
    retained = {
        "sha256", "host_count", "method_count", "graph_node_count", "graph_edge_count",
        "gap1_relation_count", "gap2_relation_count",
    }
    for stored in ("A_summary", "B_summary", "C_summary"):
        retained.update(schema["candidate_groups"][stored])
    with (FEATURES / "candidate_features_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            sample = norm(r["sha256"])
            if sample in cohort and int(float(cohort[sample]["route_DI"])) == 1:
                feature_rows[sample] = {key: r[key] for key in retained}
    if set(feature_rows) != set(ids):
        raise RuntimeError("Routed feature summary lineage mismatch")
    size_columns = ["host_count", "method_count", "graph_node_count", "graph_edge_count"]
    size_rows = []
    primary_groups = {"A": "A_summary", "B": "B_summary", "C": "C_summary"}
    for candidate, stored in primary_groups.items():
        candidate_abs = []
        for feature_name in schema["candidate_groups"][stored]:
            feature = np.asarray([float(feature_rows[s][feature_name]) for s in ids])
            for size_name in size_columns:
                size = np.asarray([float(feature_rows[s][size_name]) for s in ids])
                rho = spearman(feature, size); candidate_abs.append(abs(rho))
                size_rows.append({
                    "candidate": candidate, "row_type": "feature_pair",
                    "feature_name": feature_name, "size_variable": size_name,
                    "spearman_rho": rho, "absolute_rho": abs(rho), "n": len(size),
                })
        size_rows.append({
            "candidate": candidate, "row_type": "candidate_summary",
            "feature_name": "ALL_SUMMARY_FEATURES", "size_variable": "ALL_SIZE_VARIABLES",
            "spearman_rho": "", "absolute_rho": "", "n": len(ids),
            "max_absolute_rho": max(candidate_abs), "median_absolute_rho": float(np.median(candidate_abs)),
            "possible_size_shortcut": int(float(np.median(candidate_abs)) >= 0.8),
        })
    write_csv(REPORTS / "size_shortcut_audit.csv", size_rows)

    # Complexity / coverage audit.
    coverage_rows = read_csv(FEATURES / "candidate_feature_coverage.csv")
    coverage = {r["candidate"]: r for r in coverage_rows if r["candidate"] != "C_relation_audit"}
    complexity_rows = []
    aliases = {"A": "A_summary", "B": "B_summary", "C": "C_summary", "A_pair": "A_pair", "C_transition": "C_transition"}
    for group, stored in aliases.items():
        complexity_rows.append({
            "feature_group": group,
            "feature_count": len(schema["candidate_groups"][stored]),
            "apk_nonzero_count": round(float(coverage[stored]["APK_coverage"]) * 1989),
            "apk_nonzero_rate": float(coverage[stored]["APK_coverage"]),
        })
    write_csv(REPORTS / "feature_complexity.csv", complexity_rows)
    routed_feature = [feature_rows[s] for s in ids]
    write_json(REPORTS / "coordination_coverage.json", {
        "routed_n": len(ids),
        "routed_gap1_any_n": int(sum(float(r["gap1_relation_count"]) > 0 for r in routed_feature)),
        "routed_gap2_any_n": int(sum(float(r["gap2_relation_count"]) > 0 for r in routed_feature)),
        "routed_gap1_APK_coverage": float(np.mean([float(r["gap1_relation_count"]) > 0 for r in routed_feature])),
        "routed_gap2_APK_coverage": float(np.mean([float(r["gap2_relation_count"]) > 0 for r in routed_feature])),
    })

    # Fold coefficient stability for atomic primary blocks.
    coefficient_rows = read_csv(PREDICTIONS / "diagnostic_fold_coefficients.csv")
    grouped = defaultdict(list)
    for r in coefficient_rows:
        grouped[(r["candidate"], r["feature"])].append(float(r["standardized_coefficient"]))
    stability_rows = []
    for (candidate, name), values in sorted(grouped.items()):
        arr = np.asarray(values)
        pos, neg = int(np.sum(arr > 0)), int(np.sum(arr < 0))
        consistency = max(pos, neg) / len(arr) if len(arr) else np.nan
        stability_rows.append({
            "candidate": candidate, "feature_name": name, "n_folds": len(arr),
            "median_coefficient": float(np.median(arr)),
            "mean_abs_coefficient": float(np.mean(np.abs(arr))),
            "positive_fold_count": pos, "negative_fold_count": neg,
            "sign_consistency": consistency,
            "stable_80pct": int(consistency >= 0.8),
        })
    write_csv(REPORTS / "feature_stability_exploratory.csv", stability_rows)

    write_json(AUDIT / "analysis_contract.json", {
        "routed_n": len(ids), "need_correction_n": int(np.sum(y)),
        "graph_negative_need_correction_n": int(np.sum(graph_negative & (y == 1))),
        "graph_negative_safe_correct_n": int(np.sum(graph_negative & (y == 0))),
        "wrong_direction_n": int(np.sum(failure == "A_WRONG_DIRECTION")),
        "too_weak_n": int(np.sum(failure == "B_RIGHT_DIRECTION_TOO_WEAK")),
        "threshold": THRESHOLD, "bootstrap_iterations": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "old_predictions_modified": False,
    })
    print("HOST_COMPLEMENT_CANDIDATE_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
