#!/usr/bin/env python3
"""Read-only failure-type x R4-block audit over already saved strict OOF scores."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COMP = Path("artifacts/care-di25-hostego-dmd-bounded-v1/r4_component_audit")
GATE = Path("artifacts/care-di25-hostego-dmd-bounded-v1/gate_cohort_audit")
MECH = Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis")
OUT = Path("artifacts/care-di25-hostego-dmd-bounded-v1/failure_block_audit")

PROBES = COMP / "component_probe_oof_predictions.csv"
COHORTS = GATE / "historical_cohort_gate_scores.csv"
INTERNAL = MECH / "apk_stage2_internal.csv"
FAILURES = MECH / "routed_unfixed_failure_types.csv"

BLOCKS = (
    "B0_graph_token32",
    "B1_critical_hostego24",
    "B2_attended_hostego24",
    "B3_attention_anomaly_summary8",
)
REFERENCE_REPS = ("S2_stage2_tail88", "S3_tail88_plus_logit", "S4_full_R4")
ALL_REPS = BLOCKS + REFERENCE_REPS
GROUP_ORDER = (
    "C_FIXED",
    "D_ROUTED_UNFIXED",
    "E_PRESERVED",
    "F_BROKEN",
    "A_WRONG_DIRECTION",
    "B_RIGHT_DIRECTION_TOO_WEAK",
    "C_BOUND_LIMIT",
    "D_NEAR_ZERO",
)
PAIR_SPECS = (
    ("A_WRONG_DIRECTION", "B_RIGHT_DIRECTION_TOO_WEAK", "wrong-direction-vs-too-weak"),
    ("A_WRONG_DIRECTION", "C_FIXED", "wrong-direction-vs-fixed"),
    ("A_WRONG_DIRECTION", "E_PRESERVED", "wrong-direction-vs-preserved"),
    ("B_RIGHT_DIRECTION_TOO_WEAK", "C_FIXED", "too-weak-vs-fixed"),
    ("C_FIXED", "F_BROKEN", "fixed-vs-broken"),
)
MECHANISM_FIELDS = (
    "bounded_delta", "raw_delta", "mean_gamma", "max_gamma", "mean_anomaly", "max_anomaly",
    "attention_entropy", "max_attention", "graph_entropy", "real_host_count", "nontrivial_ego_count",
    "critical_anomaly", "critical_gamma", "critical_attention_logit", "critical_attention_weight",
    "critical_vote_fraction",
)
SEED = 20260920
ITERATIONS = 10000


def norm(value):
    return str(value or "").strip().lower()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "gate_positive_n_at_fixed_0p5": int(np.sum(values >= 0.5)),
        "gate_positive_rate_at_fixed_0p5": float(np.mean(values >= 0.5)),
    }


def standardized_difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / denom) if denom > 0 else 0.0


def binary_auc(target, score):
    """Mann-Whitney AUC with average ranks for ties; avoids fitting/importing ML code."""
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    positive_n = int(np.sum(target == 1))
    negative_n = int(np.sum(target == 0))
    if positive_n == 0 or negative_n == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[target == 1]))
    return (rank_sum - positive_n * (positive_n + 1) / 2.0) / (positive_n * negative_n)


def bootstrap_mean_difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(SEED)
    values = np.empty(ITERATIONS, dtype=float)
    for index in range(ITERATIONS):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        values[index] = np.mean(aa) - np.mean(bb)
    return {
        "mean_difference": float(np.mean(a) - np.mean(b)),
        "bootstrap_ci_low": float(np.percentile(values, 2.5)),
        "bootstrap_ci_high": float(np.percentile(values, 97.5)),
        "standardized_mean_difference": standardized_difference(a, b),
        "bootstrap_iterations": ITERATIONS,
        "bootstrap_seed": SEED,
    }


def group_name(row):
    failure = row.get("failure_type", "")
    if failure:
        return failure
    return {"C": "C_FIXED", "D": "D_ROUTED_UNFIXED", "E": "E_PRESERVED", "F": "F_BROKEN"}[row["cohort"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_rows = read_csv(COHORTS)
    cohort_by_sha = {norm(row["sha256"]): row for row in cohort_rows}
    if len(cohort_rows) != 497 or len(cohort_by_sha) != 497:
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical cohort file is not exact unique 497")
    expected_cohorts = Counter({"E": 360, "D": 74, "C": 52, "F": 11})
    if Counter(row["cohort"] for row in cohort_rows) != expected_cohorts:
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical cohort counts changed")
    failure_rows = read_csv(FAILURES)
    expected_failures = Counter({
        "B_RIGHT_DIRECTION_TOO_WEAK": 29,
        "A_WRONG_DIRECTION": 27,
        "C_BOUND_LIMIT": 14,
        "D_NEAR_ZERO": 4,
    })
    if Counter(row["failure_type"] for row in failure_rows) != expected_failures:
        raise RuntimeError("BLOCKED_BY_CONTRACT: failure-type counts changed")

    scores = defaultdict(dict)
    for row in read_csv(PROBES):
        if row["task"] != "need" or row["representation"] not in ALL_REPS:
            continue
        sha = norm(row["sha256"])
        representation = row["representation"]
        if representation in scores[sha]:
            raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate OOF block probability")
        scores[sha][representation] = float(row["probability"])
    if set(scores) != set(cohort_by_sha):
        raise RuntimeError("BLOCKED_BY_CONTRACT: block score/cohort APK mismatch")
    if any(set(row) != set(ALL_REPS) for row in scores.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete block probabilities")

    internal_by_sha = {norm(row["sha256"]): row for row in read_csv(INTERNAL)}
    if set(cohort_by_sha) - set(internal_by_sha):
        raise RuntimeError("BLOCKED_BY_CONTRACT: historical cohort missing mechanism internals")

    joined = []
    for sha, cohort in cohort_by_sha.items():
        row = {
            "sha256": sha,
            "year": int(cohort["year"]),
            "fold": int(cohort["fold"]),
            "label": int(cohort["label"]),
            "cohort": cohort["cohort"],
            "cohort_name": cohort["cohort_name"],
            "failure_type": cohort.get("failure_type", ""),
            "analysis_group": group_name(cohort),
        }
        row.update({name: scores[sha][name] for name in ALL_REPS})
        for field in MECHANISM_FIELDS:
            value = as_float(internal_by_sha[sha].get(field))
            if value is not None:
                row[field] = value
        if "critical_role" in internal_by_sha[sha]:
            row["critical_role"] = internal_by_sha[sha]["critical_role"]
        joined.append(row)
    write_csv(OUT / "failure_type_block_scores.csv", joined)

    groups = {
        "C_FIXED": [row for row in joined if row["cohort"] == "C"],
        "D_ROUTED_UNFIXED": [row for row in joined if row["cohort"] == "D"],
        "E_PRESERVED": [row for row in joined if row["cohort"] == "E"],
        "F_BROKEN": [row for row in joined if row["cohort"] == "F"],
    }
    for failure_type in expected_failures:
        groups[failure_type] = [row for row in joined if row["failure_type"] == failure_type]

    summary = []
    for group in GROUP_ORDER:
        for representation in ALL_REPS:
            summary.append({
                "group": group,
                "representation": representation,
                "n": len(groups[group]),
                **describe([row[representation] for row in groups[group]]),
            })
    write_csv(OUT / "block_score_summary_by_group.csv", summary)

    comparisons = []
    for positive, negative, label in PAIR_SPECS:
        for representation in ALL_REPS:
            a = [row[representation] for row in groups[positive]]
            b = [row[representation] for row in groups[negative]]
            target = np.asarray([1] * len(a) + [0] * len(b), dtype=int)
            probability = np.asarray(a + b, dtype=float)
            comparisons.append({
                "comparison": label,
                "positive_group": positive,
                "negative_group": negative,
                "representation": representation,
                "n_positive": len(a),
                "n_negative": len(b),
                "score_AUC": float(binary_auc(target, probability)),
                **bootstrap_mean_difference(a, b),
            })
    write_csv(OUT / "block_group_pairwise_comparisons.csv", comparisons)

    available_fields = [field for field in MECHANISM_FIELDS if all(field in row for row in joined)]
    mechanism_summary = []
    for group in GROUP_ORDER:
        for field in available_fields:
            mechanism_summary.append({
                "group": group,
                "mechanism_field": field,
                "n": len(groups[group]),
                **describe([row[field] for row in groups[group]]),
            })
    write_csv(OUT / "mechanism_internal_summary_by_group.csv", mechanism_summary)

    mechanism_comparisons = []
    for positive, negative, label in PAIR_SPECS[:3]:
        for field in available_fields:
            a = [row[field] for row in groups[positive]]
            b = [row[field] for row in groups[negative]]
            mechanism_comparisons.append({
                "comparison": label,
                "positive_group": positive,
                "negative_group": negative,
                "mechanism_field": field,
                "n_positive": len(a),
                "n_negative": len(b),
                **bootstrap_mean_difference(a, b),
            })
    write_csv(OUT / "mechanism_internal_pairwise_comparisons.csv", mechanism_comparisons)

    wrong_vs_weak = [row for row in comparisons if row["comparison"] == "wrong-direction-vs-too-weak"]
    ranked = sorted(wrong_vs_weak, key=lambda row: float(row["mean_difference"]))
    block_ranked = [row for row in ranked if row["representation"] in BLOCKS]
    most_negative = block_ranked[0]
    least_negative = block_ranked[-1]
    captures = {(row["group"], row["representation"]): row for row in summary}
    lines = [
        "CARE-Droid failure-type x R4-block diagnostic audit",
        "Read-only mapping of saved strict OOF need-correction probe probabilities to locked historical M1 cohorts.",
        "No CARE-Droid training, replay, threshold search, route change, or prediction change was performed.",
        "",
        "Fixed threshold: 0.5 (the original probe decision threshold).",
        "Separate block probes measure marginal OOF readability; they are not additive neural contributions and do not establish causal suppression.",
        "",
        "Wrong Direction vs Too Weak:",
    ]
    for row in block_ranked:
        lines.append(
            f"  {row['representation']}: AUC={float(row['score_AUC']):.6f} "
            f"mean_diff={float(row['mean_difference']):.6f} "
            f"CI=[{float(row['bootstrap_ci_low']):.6f},{float(row['bootstrap_ci_high']):.6f}]"
        )
    lines.extend([
        "",
        f"Largest negative marginal block separation: {most_negative['representation']} "
        f"(WrongDirection score lower than TooWeak by {abs(float(most_negative['mean_difference'])):.6f}).",
        f"Least negative/highest marginal block result: {least_negative['representation']} "
        f"(mean difference {float(least_negative['mean_difference']):.6f}).",
        "",
        "Fixed 0.5 capture by block:",
    ])
    for representation in BLOCKS:
        wd = captures[("A_WRONG_DIRECTION", representation)]
        tw = captures[("B_RIGHT_DIRECTION_TOO_WEAK", representation)]
        fx = captures[("C_FIXED", representation)]
        br = captures[("F_BROKEN", representation)]
        lines.append(
            f"  {representation}: WrongDirection={wd['gate_positive_n_at_fixed_0p5']}/27, "
            f"TooWeak={tw['gate_positive_n_at_fixed_0p5']}/29, "
            f"Fixed={fx['gate_positive_n_at_fixed_0p5']}/52, Broken={br['gate_positive_n_at_fixed_0p5']}/11"
        )
    lines.extend([
        "",
        "Decision boundary:",
        "- If all block probes score Wrong Direction below Too Weak, the available R4 structural views do not expose a generic correction-need signal for that cohort.",
        "- If one block materially reduces or reverses the gap, that block is a candidate source for a predeclared rescue mechanism; this audit alone does not authorize one.",
        "- Fixed/Broken associations are post-hoc historical-outcome diagnostics, not deployable performance estimates.",
    ])
    (OUT / "FAILURE_BLOCK_AUDIT_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "NO_MODEL_MODIFICATION_AUDIT.json", {
        "CARE_model_loaded": False,
        "stage2_forward_invoked": False,
        "training_invoked": False,
        "probe_fitting_invoked": False,
        "threshold_search_invoked": False,
        "fixed_threshold": 0.5,
        "routes_modified": False,
        "predictions_modified": False,
        "block_scores_source": str(PROBES),
        "historical_cohorts_source": str(COHORTS),
        "available_mechanism_fields": available_fields,
        "interpretation": "marginal OOF readability, not additive or causal block attribution",
    })
    print("FAILURE_BLOCK_AUDIT_DONE")


if __name__ == "__main__":
    main()
