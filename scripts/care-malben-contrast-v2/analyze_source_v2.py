#!/usr/bin/env python3
"""Full source-only mechanism, repairability and significance audit for v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


METHOD_COLUMNS = {
    "M0_AnomalyBaseline": "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep__expert_prob",
    "M1_BenignOnly": "Global24__A25__CARE_RoleBenignOnly_FixKeep__expert_prob",
    "M2_MaliciousOnly": "Global24__A25__CARE_RoleMaliciousOnly_FixKeep__expert_prob",
    "M3_MalBenContrast": "Global24__A25__CARE_RoleMalBenContrast_FixKeep__expert_prob",
}


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm_sha(value) -> str:
    return str(value or "").strip().casefold()


def array(rows: list[dict], name: str, dtype=float) -> np.ndarray:
    return np.asarray([dtype(row[name]) for row in rows])


def find_column(rows: list[dict], preferred: str, tokens: tuple[str, ...]) -> str:
    if preferred in rows[0]:
        return preferred
    candidates = [name for name in rows[0] if all(token in name for token in tokens)]
    if len(candidates) != 1:
        raise RuntimeError(f"cannot resolve {preferred}; candidates={candidates}")
    return candidates[0]


def align(rows: list[dict], sha: list[str], context: str) -> list[dict]:
    by_sha = {norm_sha(row["sha256"]): row for row in rows}
    if len(by_sha) != len(rows) or set(by_sha) != set(sha):
        raise RuntimeError(context + " cohort/SHA mismatch")
    return [by_sha[item] for item in sha]


def safe_ratio(left: int, right: int) -> float:
    return float(left / right) if right else 0.0


def evaluate(y, base, expert, route) -> dict:
    y = np.asarray(y, dtype=np.int64)
    base = np.asarray(base, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    base_pred = base >= 0.5
    expert_pred = expert >= 0.5
    final = np.where(route, expert, base)
    pred = final >= 0.5
    base_correct = base_pred == y
    expert_correct = expert_pred == y
    fixed = int(np.sum(route & ~base_correct & expert_correct))
    broken = int(np.sum(route & base_correct & ~expert_correct))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    repairable = ~base_correct & expert_correct
    return {
        "n": int(len(y)),
        "routed_n": int(np.sum(route)),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, final)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "repairable_error_n": int(np.sum(repairable)),
        "routed_repairable_error_n": fixed,
        "RER_at_25": safe_ratio(fixed, int(np.sum(repairable))),
    }


def fp_fn_row(name, router, y, base, expert, route) -> dict:
    base_pred = np.asarray(base) >= 0.5
    expert_pred = np.asarray(expert) >= 0.5
    final_pred = np.where(route, expert_pred, base_pred)
    y = np.asarray(y, dtype=np.int64)
    stage1_fp = (y == 0) & base_pred
    stage1_fn = (y == 1) & ~base_pred
    stage1_tn = (y == 0) & ~base_pred
    stage1_tp = (y == 1) & base_pred
    return {
        "router": router,
        "method": name,
        "stage1_FP_n": int(np.sum(stage1_fp)),
        "stage1_FN_n": int(np.sum(stage1_fn)),
        "stage1_TN_n": int(np.sum(stage1_tn)),
        "stage1_TP_n": int(np.sum(stage1_tp)),
        "fixed_FP": int(np.sum(route & stage1_fp & ~final_pred)),
        "fixed_FN": int(np.sum(route & stage1_fn & final_pred)),
        "broken_to_FP": int(np.sum(route & stage1_tn & final_pred)),
        "broken_to_FN": int(np.sum(route & stage1_tp & ~final_pred)),
    }


def clipped_logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def distribution_row(router, method, group_type, group, delta) -> dict:
    value = np.asarray(delta, dtype=np.float64)
    return {
        "router": router,
        "method": method,
        "group_type": group_type,
        "group": group,
        "n": int(len(value)),
        "mean_delta": float(np.mean(value)) if len(value) else float("nan"),
        "median_delta": float(np.median(value)) if len(value) else float("nan"),
        "P10_delta": float(np.quantile(value, 0.10)) if len(value) else float("nan"),
        "P90_delta": float(np.quantile(value, 0.90)) if len(value) else float("nan"),
        "mean_abs_delta": float(np.mean(np.abs(value))) if len(value) else float("nan"),
        "median_abs_delta": float(np.median(np.abs(value))) if len(value) else float("nan"),
    }


def exact_mcnemar_p(n01: int, n10: int) -> float:
    total = int(n01 + n10)
    if total == 0:
        return 1.0
    lower = min(int(n01), int(n10))
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2.0 ** total)
    return min(1.0, 2.0 * probability)


def bootstrap_ba_delta(y, old_probability, new_probability, repeats=10000, seed=20260917):
    y = np.asarray(y, dtype=np.int64)
    old_pred = np.asarray(old_probability) >= 0.5
    new_pred = np.asarray(new_probability) >= 0.5
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    values = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        index = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in classes
        ])
        values[repeat] = (
            balanced_accuracy_score(y[index], new_pred[index])
            - balanced_accuracy_score(y[index], old_pred[index])
        )
    return {
        "point": float(
            balanced_accuracy_score(y, new_pred) - balanced_accuracy_score(y, old_pred)
        ),
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "repeats": int(repeats),
        "seed": int(seed),
    }


def add_region_groups(path: Path, group_by_sha: dict[str, dict]) -> list[dict]:
    rows = read_csv(path)
    for row in rows:
        group = group_by_sha[norm_sha(row["sample_id"])]
        row.update(group)
    write_csv(path, rows)
    return rows


def region_summary(rows: list[dict]) -> list[dict]:
    fields = ("S_ben", "S_mal", "margin", "E_ben", "E_mal", "evidence_strength")
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        group_specs = [
            ("stage1_confusion", name) for name in ("TN", "FP", "TP", "FN")
        ] + [
            ("special", name) for name in (
                "anomaly_baseline_broken", "malben_rescued", "malben_newly_broken"
            )
        ]
        for group_type, group_name in group_specs:
            if group_type == "stage1_confusion":
                selected = [row for row in variant_rows if row["stage1_confusion"] == group_name]
            else:
                selected = [row for row in variant_rows if int(row[group_name]) == 1]
            item = {
                "variant": variant, "group_type": group_type,
                "group": group_name, "region_n": len(selected),
                "apk_n": len({row["sample_id"] for row in selected}),
            }
            for field in fields:
                value = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
                item[field + "_mean"] = float(np.mean(value)) if len(value) else float("nan")
                item[field + "_median"] = float(np.median(value)) if len(value) else float("nan")
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", type=Path, default=Path(
        "artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv"
    ))
    parser.add_argument("--baseline", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv"
    ))
    parser.add_argument("--new", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/model/source_predictions.csv"
    ))
    parser.add_argument("--region", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/source_region_evidence.csv"
    ))
    parser.add_argument("--prototype-quality", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/source_prototype_quality.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source"
    ))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    router_rows = read_csv(args.router)
    if len(router_rows) != 799:
        raise RuntimeError("expected fixed 799-row source cohort")
    sha = [norm_sha(row["sha256"]) for row in router_rows]
    y = array(router_rows, "label", int).astype(np.int64)
    folds = array(router_rows, "fold", int).astype(np.int64)
    base = array(router_rows, "p_stage1")
    routes = {
        "A25": array(router_rows, "route_A25", int).astype(bool),
        "DI25": array(router_rows, "route_DI", int).astype(bool),
    }
    if any(int(np.sum(value)) != 200 for value in routes.values()):
        raise RuntimeError("fixed 25% source routes changed")

    baseline_rows = align(read_csv(args.baseline), sha, "M0 baseline")
    new_rows = align(read_csv(args.new), sha, "v2 models")
    baseline_col = find_column(
        baseline_rows, METHOD_COLUMNS["M0_AnomalyBaseline"], ("Dedup", "expert_prob")
    )
    experts = {"M0_AnomalyBaseline": array(baseline_rows, baseline_col)}
    for method in ("M1_BenignOnly", "M2_MaliciousOnly", "M3_MalBenContrast"):
        tokens = tuple(METHOD_COLUMNS[method].split("__")[2].split("_"))
        column = find_column(new_rows, METHOD_COLUMNS[method], tokens + ("expert", "prob"))
        experts[method] = array(new_rows, column)

    for name, rows in (("baseline", baseline_rows), ("new", new_rows)):
        if not np.array_equal(array(rows, "label", int), y):
            raise RuntimeError(name + " label alignment changed")
        if not np.array_equal(array(rows, "fold", int), folds):
            raise RuntimeError(name + " fold alignment changed")
        if not np.allclose(array(rows, "Global24__prob"), base, atol=2e-6, rtol=0.0):
            raise RuntimeError(name + " Global24 probabilities changed")

    aggregate_rows = []
    fold_rows = []
    fp_fn_rows = []
    repair_rows = []
    correction_rows = []
    final_probability = {}
    sample_rows = [{
        "sha256": sha[index],
        "label": int(y[index]),
        "fold": int(folds[index]),
        "p_stage1": float(base[index]),
        "route_A25": int(routes["A25"][index]),
        "route_DI25": int(routes["DI25"][index]),
    } for index in range(len(y))]
    for router_name, route in routes.items():
        baseline_metric = None
        for method, expert in experts.items():
            metric = evaluate(y, base, expert, route)
            metric.update({"router": router_name, "method": method})
            aggregate_rows.append(metric)
            if method == "M0_AnomalyBaseline":
                baseline_metric = metric
            fp_fn_rows.append(fp_fn_row(method, router_name, y, base, expert, route))
            repair_rows.append({
                "router": router_name,
                "method": method,
                "definition": "Stage-I hard error and Stage-II expert hard prediction correct at threshold 0.5",
                "sample_set": "all 799 fixed source OOF samples",
                "numerator_routed_repairable": metric["routed_repairable_error_n"],
                "denominator_all_repairable": metric["repairable_error_n"],
                "RER_at_25": metric["RER_at_25"],
            })
            final = np.where(route, expert, base)
            final_probability[(router_name, method)] = final
            logit_delta = clipped_logit(expert) - clipped_logit(base)
            for index, item in enumerate(sample_rows):
                item[method + "__expert_probability"] = float(expert[index])
                item[method + "__delta_logit"] = float(logit_delta[index])
                item[router_name + "__" + method + "__final_probability"] = float(final[index])
            base_pred = base >= 0.5
            delta = logit_delta
            selected = route
            stage1_confusion = np.where(
                y == 0, np.where(base_pred, "FP", "TN"), np.where(base_pred, "TP", "FN")
            )
            final_pred = final >= 0.5
            final_confusion = np.where(
                y == 0, np.where(final_pred, "FP", "TN"), np.where(final_pred, "TP", "FN")
            )
            for group in ("TN", "FP", "TP", "FN"):
                correction_rows.append(distribution_row(
                    router_name, method, "stage1_confusion", group,
                    delta[selected & (stage1_confusion == group)],
                ))
                correction_rows.append(distribution_row(
                    router_name, method, "final_confusion", group,
                    delta[selected & (final_confusion == group)],
                ))
            correct = base_pred == y
            for group, mask in (("correct", correct), ("wrong", ~correct)):
                correction_rows.append(distribution_row(
                    router_name, method, "stage1_status", group, delta[selected & mask]
                ))

            for fold in sorted(np.unique(folds)):
                chosen = folds == fold
                item = evaluate(y[chosen], base[chosen], expert[chosen], route[chosen])
                item.update({"fold": int(fold), "router": router_name, "method": method})
                fold_rows.append(item)

        for row in aggregate_rows:
            if row["router"] == router_name:
                row.update({
                    "dBA_vs_M0": float(row["BA"] - baseline_metric["BA"]),
                    "dAUC_vs_M0": float(row["AUC"] - baseline_metric["AUC"]),
                    "dF1_vs_M0": float(row["F1"] - baseline_metric["F1"]),
                    "dFixed_vs_M0": int(row["fixed"] - baseline_metric["fixed"]),
                    "dBroken_vs_M0": int(row["broken"] - baseline_metric["broken"]),
                    "dNet_vs_M0": int(row["net"] - baseline_metric["net"]),
                })

    # Add fold-level deltas within the same fold and router.
    by_fold_key = {
        (row["fold"], row["router"], row["method"]): row for row in fold_rows
    }
    for row in fold_rows:
        old = by_fold_key[(row["fold"], row["router"], "M0_AnomalyBaseline")]
        for metric in ("BA", "AUC", "F1"):
            row["d" + metric + "_vs_M0"] = float(row[metric] - old[metric])
        for metric in ("fixed", "broken", "net"):
            row["d" + metric.capitalize() + "_vs_M0"] = int(row[metric] - old[metric])

    write_csv(args.output / "source_ablation_metrics.csv", aggregate_rows)
    write_csv(args.output / "source_fold_metrics.csv", fold_rows)
    write_csv(args.output / "source_fp_fn_analysis.csv", fp_fn_rows)
    write_csv(args.output / "source_correction_delta.csv", correction_rows)
    write_csv(args.output / "source_repairability.csv", repair_rows)
    write_csv(args.output / "source_sample_predictions.csv", sample_rows)

    # RER definition audit: old committee expert versus selected max-delta=2.25 expert.
    historical = array(router_rows, "p_frozen_bounded_expert")
    current = experts["M0_AnomalyBaseline"]
    route = routes["A25"]
    historical_metric = evaluate(y, base, historical, route)
    current_metric = evaluate(y, base, current, route)
    rer_text = [
        "CARE-Droid RER definition audit",
        "",
        "Definition used for both rows:",
        "repairable(i) = [Stage-I hard prediction is wrong] AND",
        "                [the named Stage-II expert hard prediction is correct]",
        "RER@25 = routed repairable errors / all repairable errors for that expert.",
        "Hard threshold = 0.5; sample set = the same aligned 799 source OOF APKs.",
        "A25 route sample ids are read once from the same router artifact.",
        "",
        "Historical frozen bounded expert:",
        f"  numerator={historical_metric['routed_repairable_error_n']}",
        f"  denominator={historical_metric['repairable_error_n']}",
        f"  RER={historical_metric['RER_at_25']:.6f}",
        "  expert=p_frozen_bounded_expert stored by the earlier committee audit",
        "",
        "Current anomaly baseline:",
        f"  numerator={current_metric['routed_repairable_error_n']}",
        f"  denominator={current_metric['repairable_error_n']}",
        f"  RER={current_metric['RER_at_25']:.6f}",
        "  expert=deduplicated RGCondAttn retrained with max_delta(logit)=2.25",
        "",
        f"Expert probabilities differing by >1e-8: {int(np.sum(np.abs(historical-current)>1e-8))}/799",
        "Conclusion:",
        "The RER discrepancy is not a change in formula, cohort, route, or threshold.",
        "The denominator is expert-specific and changed because the Stage-II checkpoint/",
        "correction bound changed. The historical and current RER values must not be",
        "placed in one table as if they described the same expert.",
    ]
    (args.output / "rer_definition_audit.txt").write_text(
        "\n".join(rer_text) + "\n", encoding="utf-8"
    )

    # Paired source significance for the primary A25 comparison M0 vs M3.
    old_final = final_probability[("A25", "M0_AnomalyBaseline")]
    new_final = final_probability[("A25", "M3_MalBenContrast")]
    old_correct = (old_final >= 0.5) == y
    new_correct = (new_final >= 0.5) == y
    n01 = int(np.sum(old_correct & ~new_correct))
    n10 = int(np.sum(~old_correct & new_correct))
    p_value = exact_mcnemar_p(n01, n10)
    bootstrap = bootstrap_ba_delta(y, old_final, new_final)
    significance = [
        "CARE-Droid source-only paired significance audit",
        "",
        "Primary comparison: A25 M0 AnomalyBaseline vs M3 RoleMalBenContrast",
        "McNemar exact two-sided test:",
        f"  n01 (M0 correct, M3 wrong) = {n01}",
        f"  n10 (M0 wrong, M3 correct) = {n10}",
        f"  p-value = {p_value:.10g}",
        "",
        "Stratified paired bootstrap BA delta (M3-M0):",
        f"  point = {bootstrap['point']:.10f}",
        f"  mean = {bootstrap['mean']:.10f}",
        f"  2.5% = {bootstrap['lower']:.10f}",
        f"  97.5% = {bootstrap['upper']:.10f}",
        f"  repeats = {bootstrap['repeats']}",
        f"  seed = {bootstrap['seed']}",
        "",
        "These tests quantify source evidence; they are not additional model-selection rules.",
    ]
    (args.output / "source_significance.txt").write_text(
        "\n".join(significance) + "\n", encoding="utf-8"
    )

    # Prototype separation-to-dispersion diagnostics.
    quality_rows = read_csv(args.prototype_quality)
    for row in quality_rows:
        distance = float(row["prototype_distance"])
        dispersion = np.nanmean([
            float(row["within_ben_dispersion"]), float(row["within_mal_dispersion"])
        ])
        ratio = distance / max(float(dispersion), 1e-8) if np.isfinite(dispersion) else float("nan")
        row["separation_to_dispersion"] = ratio
        row["possible_collapse"] = int(
            int(row["fallback"]) == 0 and np.isfinite(ratio) and ratio < 0.5
        )
    write_csv(args.prototype_quality, quality_rows)

    # Attach source error groups to every region row and summarize the mechanism.
    base_pred = base >= 0.5
    old_pred = old_final >= 0.5
    new_pred = new_final >= 0.5
    group_by_sha = {}
    for index, item in enumerate(sha):
        confusion = (
            "FP" if y[index] == 0 and base_pred[index]
            else "TN" if y[index] == 0
            else "TP" if base_pred[index]
            else "FN"
        )
        group_by_sha[item] = {
            "stage1_confusion": confusion,
            "anomaly_baseline_broken": int(
                routes["A25"][index] and base_pred[index] == y[index] and old_pred[index] != y[index]
            ),
            "malben_rescued": int(old_pred[index] != y[index] and new_pred[index] == y[index]),
            "malben_newly_broken": int(old_pred[index] == y[index] and new_pred[index] != y[index]),
        }
    region_rows = add_region_groups(args.region, group_by_sha)
    write_csv(args.output / "source_region_group_summary.csv", region_summary(region_rows))

    # Source-side advance rule.
    aggregate = {(row["router"], row["method"]): row for row in aggregate_rows}
    old = aggregate[("A25", "M0_AnomalyBaseline")]
    new = aggregate[("A25", "M3_MalBenContrast")]
    fold_delta = [
        row for row in fold_rows
        if row["router"] == "A25" and row["method"] == "M3_MalBenContrast"
    ]
    net_positive = sum(int(row["dNet_vs_M0"] > 0) for row in fold_delta)
    net_zero = sum(int(row["dNet_vs_M0"] == 0) for row in fold_delta)
    net_negative = sum(int(row["dNet_vs_M0"] < 0) for row in fold_delta)
    ba_positive = sum(int(row["dBA_vs_M0"] > 0) for row in fold_delta)
    ba_negative = sum(int(row["dBA_vs_M0"] < 0) for row in fold_delta)
    fpfn = {(row["router"], row["method"]): row for row in fp_fn_rows}
    old_fpfn = fpfn[("A25", "M0_AnomalyBaseline")]
    new_fpfn = fpfn[("A25", "M3_MalBenContrast")]
    eligible_quality = [row for row in quality_rows if int(row["fallback"]) == 0]
    collapse_n = sum(int(row["possible_collapse"]) for row in eligible_quality)
    collapse_fraction = safe_ratio(collapse_n, len(eligible_quality))
    checks = {
        "aggregate_net_improves": int(new["net"] > old["net"]),
        "aggregate_BA_non_decreasing": int(new["BA"] >= old["BA"]),
        "broken_non_increasing": int(new["broken"] <= old["broken"]),
        "majority_folds_net_non_decreasing": int(net_positive + net_zero >= 3),
        "benign_mechanism_consistent": int(
            new_fpfn["broken_to_FP"] <= old_fpfn["broken_to_FP"]
        ),
        "rer_definition_explained": 1,
        # Individual roles may legitimately be non-discriminative.  Treat it
        # as system-level collapse only when most supported role/fold cells
        # have separation below half their within-class dispersion.
        "no_obvious_prototype_collapse": int(collapse_fraction <= 0.5),
    }
    advance = all(checks.values())
    summary = [
        "CARE-Droid role-conditioned MalBen source-only summary v2",
        "",
        "Primary router: A25. DI25 is secondary sensitivity only.",
        "No future-year sample is loaded by this experiment.",
        "",
        "A25 aggregate:",
    ]
    for method in METHOD_COLUMNS:
        row = aggregate[("A25", method)]
        summary.append(
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} "
            f"net={row['net']} RER@25={row['RER_at_25']:.6f}"
        )
    summary.extend([
        "",
        "DI25 sensitivity:",
    ])
    for method in METHOD_COLUMNS:
        row = aggregate[("DI25", method)]
        summary.append(
            f"{method}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} "
            f"F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}"
        )
    summary.extend([
        "",
        "A25 five-fold stability for M3 vs M0:",
        f"dNet >0 / =0 / <0: {net_positive} / {net_zero} / {net_negative}",
        f"dBA >0 / <0: {ba_positive} / {ba_negative}",
        "",
        "Source-side advance checks:",
        f"Supported role/fold cells flagged low-separation: {collapse_n}/"
        f"{len(eligible_quality)} ({collapse_fraction:.2%})",
    ])
    for name, passed in checks.items():
        summary.append(f"  {name}: {'PASS' if passed else 'FAIL'}")
    summary.extend([
        "",
        "SOURCE ADVANCE DECISION: " + ("PASS" if advance else "DO NOT ADVANCE"),
        "Future diagnostic must not run unless this decision is PASS and the source",
        "configuration is frozen without inspecting 2018-2021 labels.",
    ])
    text = "\n".join(summary) + "\n"
    (args.output / "source_summary.txt").write_text(text, encoding="utf-8")
    protocol = {
        "future_data_loaded": False,
        "primary_router": "A25",
        "secondary_router": "DI25",
        "advance": bool(advance),
        "advance_checks": checks,
        "min_support": 20,
        "K": 1,
        "similarity": "exp(-RMS-Euclidean/within-class-median-scale)",
    }
    (args.output / "source_decision.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
