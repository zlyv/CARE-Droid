#!/usr/bin/env python3
"""Read-only conditional complementarity audit over saved strict same-year OOF scores."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-di25-hostego-complementarity-audit-v1")
PREFLIGHT = ROOT / "preflight"
REPORTS = ROOT / "reports"
THRESHOLD = 0.5
MODES = ("critical", "attended", "union")
REPS = {
    "graph": "B0_graph_token32",
    "critical": "B1_critical_hostego24",
    "attended": "B2_attended_hostego24",
    "summary": "B3_attention_anomaly_summary8",
}
INTERNAL_FIELDS = (
    "mean_gamma", "max_gamma", "mean_anomaly", "max_anomaly", "attention_entropy", "max_attention",
    "critical_anomaly", "critical_gamma", "critical_attention_weight", "critical_vote_fraction",
)


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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, index) * p**index * (1.0 - p) ** (n - index) for index in range(k + 1))


def solve_decreasing_cdf(k, n, target):
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes, total, alpha=0.05):
    if total <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else solve_decreasing_cdf(successes - 1, total, 1.0 - alpha / 2.0)
    upper = 1.0 if successes == total else solve_decreasing_cdf(successes, total, alpha / 2.0)
    return lower, upper


def exact_mcnemar(graph_positive_host_negative, graph_negative_host_positive):
    discordant = graph_positive_host_negative + graph_negative_host_positive
    if discordant == 0:
        return 1.0
    smaller = min(graph_positive_host_negative, graph_negative_host_positive)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def describe(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def flags(row):
    graph = row["graph_score"] >= THRESHOLD
    critical = row["critical_hostego_score"] >= THRESHOLD
    attended = row["attended_hostego_score"] >= THRESHOLD
    return {"graph": graph, "critical": critical, "attended": attended, "union": critical or attended,
            "intersection": critical and attended}


def contingency(rows, mode):
    counts = Counter()
    for row in rows:
        state = flags(row)
        counts[(state["graph"], state[mode])] += 1
    return {
        "graph_positive_host_positive": counts[(True, True)],
        "graph_positive_host_negative": counts[(True, False)],
        "graph_negative_host_positive": counts[(False, True)],
        "graph_negative_host_negative": counts[(False, False)],
    }


def rescue_summary(rows, mode):
    graph_missed = [row for row in rows if not flags(row)["graph"]]
    rescued = [row for row in graph_missed if flags(row)[mode]]
    lower, upper = clopper_pearson(len(rescued), len(graph_missed))
    return {
        "host_mode": mode,
        "cohort_n": len(rows),
        "graph_captured_n": len(rows) - len(graph_missed),
        "graph_missed_n": len(graph_missed),
        "rescued_n": len(rescued),
        "rescue_rate_among_graph_missed": len(rescued) / len(graph_missed) if graph_missed else float("nan"),
        "rescue_rate_ci_low": lower,
        "rescue_rate_ci_high": upper,
        "graph_plus_host_total_capture_n": len(rows) - len(graph_missed) + len(rescued),
        "graph_plus_host_total_capture_rate": (len(rows) - len(graph_missed) + len(rescued)) / len(rows),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREFLIGHT / "source_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((PREFLIGHT / "contract.json").read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or contract.get("fixed_threshold") != THRESHOLD:
        raise RuntimeError("BLOCKED_BY_CONTRACT: preflight contract did not pass")
    for path_text, before in manifest["input_sha256"].items():
        if sha256(Path(path_text)) != before:
            raise RuntimeError("BLOCKED_BY_CONTRACT: input SHA256 changed before audit: " + path_text)

    probes = read_csv(Path(manifest["strict_oof_probe_file"]))
    cohort_rows = read_csv(Path(manifest["historical_cohort_file"]))
    failure_rows = read_csv(Path(manifest["failure_type_file"]))
    failure_by_sha = {norm(row["sha256"]): row["failure_type"] for row in failure_rows}
    score_by_sha = defaultdict(dict)
    for row in probes:
        if row["task"] == "need" and row["representation"] in set(REPS.values()):
            score_by_sha[norm(row["sha256"])][row["representation"]] = float(row["probability"])
    internal_by_sha = {}
    internal_path = manifest.get("optional_existing_mechanism_internal_file")
    if internal_path:
        internal_by_sha = {norm(row["sha256"]): row for row in read_csv(Path(internal_path))}

    rows = []
    for cohort in cohort_rows:
        sample = norm(cohort["sha256"])
        score = score_by_sha[sample]
        failure_type = failure_by_sha.get(sample, cohort.get("failure_type", ""))
        stage1_wrong = cohort.get("stage1_wrong")
        if stage1_wrong in (None, ""):
            stage1_wrong = int(cohort["cohort"] in ("C", "D"))
        row = {
            "sha256": sample,
            "year": int(cohort["year"]),
            "fold": int(cohort["fold"]),
            "label": int(cohort["label"]),
            "cohort": cohort["cohort"],
            "stage1_wrong": int(float(stage1_wrong)),
            "failure_type": failure_type,
            "graph_score": score[REPS["graph"]],
            "critical_hostego_score": score[REPS["critical"]],
            "attended_hostego_score": score[REPS["attended"]],
            "attention_anomaly_score": score[REPS["summary"]],
        }
        if sample in internal_by_sha:
            for field in INTERNAL_FIELDS:
                value = safe_float(internal_by_sha[sample].get(field))
                if value is not None:
                    row[field] = value
        rows.append(row)

    wrong_direction = [row for row in rows if row["failure_type"] == "A_WRONG_DIRECTION"]
    too_weak = [row for row in rows if row["failure_type"] == "B_RIGHT_DIRECTION_TOO_WEAK"]
    fixed = [row for row in rows if row["cohort"] == "C"]
    broken = [row for row in rows if row["cohort"] == "F"]
    safe_correct = [row for row in rows if row["stage1_wrong"] == 0]
    if (len(wrong_direction), len(too_weak), len(fixed), len(broken), len(safe_correct)) != (27, 29, 52, 11, 371):
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked cohort counts changed during audit")

    wd_missed = [row for row in wrong_direction if not flags(row)["graph"]]
    tw_missed = [row for row in too_weak if not flags(row)["graph"]]
    safe_graph_negative = [row for row in safe_correct if not flags(row)["graph"]]

    wd_sample_rows = []
    for row in wd_missed:
        state = flags(row)
        wd_sample_rows.append({**row,
            "graph_positive": int(state["graph"]), "critical_positive": int(state["critical"]),
            "attended_positive": int(state["attended"]), "host_union_positive": int(state["union"]),
            "critical_rescue": int(state["critical"]), "attended_rescue": int(state["attended"]),
            "union_rescue": int(state["union"]),
        })
    write_csv(REPORTS / "wrong_direction_graph_missed_samples.csv", wd_sample_rows)

    safe_sample_rows = []
    for row in safe_graph_negative:
        state = flags(row)
        safe_sample_rows.append({
            "sha256": row["sha256"], "year": row["year"], "fold": row["fold"], "label": row["label"],
            "graph_score": row["graph_score"], "critical_hostego_score": row["critical_hostego_score"],
            "attended_hostego_score": row["attended_hostego_score"],
            "critical_false_rescue_candidate": int(state["critical"]),
            "attended_false_rescue_candidate": int(state["attended"]),
            "union_false_rescue_candidate": int(state["union"]),
        })
    write_csv(REPORTS / "safe_correct_graph_negative_samples.csv", safe_sample_rows)

    wd_rescue = [rescue_summary(wrong_direction, mode) for mode in MODES]
    intersection = rescue_summary(wrong_direction, "intersection")
    wd_rescue.append(intersection)
    write_csv(REPORTS / "wrong_direction_rescue.csv", wd_rescue)
    tw_rescue = [rescue_summary(too_weak, mode) for mode in MODES]
    write_csv(REPORTS / "too_weak_rescue.csv", tw_rescue)

    false_rows = []
    for mode in MODES:
        flagged = sum(flags(row)[mode] for row in safe_graph_negative)
        low, high = clopper_pearson(flagged, len(safe_graph_negative))
        false_rows.append({
            "host_mode": mode, "safe_correct_n": len(safe_correct),
            "graph_negative_safe_correct_n": len(safe_graph_negative),
            "false_rescue_candidate_n": flagged,
            "false_rescue_candidate_rate": flagged / len(safe_graph_negative) if safe_graph_negative else float("nan"),
            "false_rescue_candidate_ci_low": low, "false_rescue_candidate_ci_high": high,
            "terminology": "probe-level false-rescue candidate; not actual Broken",
        })
    write_csv(REPORTS / "safe_correct_false_rescue.csv", false_rows)

    contingency_rows, test_rows = [], []
    for mode in MODES:
        table = contingency(wrong_direction, mode)
        union_size = table["graph_positive_host_positive"] + table["graph_positive_host_negative"] + table["graph_negative_host_positive"]
        graph_positive = table["graph_positive_host_positive"] + table["graph_positive_host_negative"]
        host_positive = table["graph_positive_host_positive"] + table["graph_negative_host_positive"]
        contingency_rows.append({
            "host_mode": mode, "cohort": "A_WRONG_DIRECTION", "n": len(wrong_direction), **table,
            "graph_positive_n": graph_positive, "host_positive_n": host_positive,
            "jaccard_overlap": table["graph_positive_host_positive"] / union_size if union_size else float("nan"),
            "host_unique_capture_rate": table["graph_negative_host_positive"] / len(wrong_direction),
        })
        p_value = exact_mcnemar(table["graph_positive_host_negative"], table["graph_negative_host_positive"])
        test_rows.append({
            "comparison": f"Graph vs {mode}", "cohort": "A_WRONG_DIRECTION", "n": len(wrong_direction),
            "graph_positive_host_negative": table["graph_positive_host_negative"],
            "graph_negative_host_positive": table["graph_negative_host_positive"],
            "discordant_n": table["graph_positive_host_negative"] + table["graph_negative_host_positive"],
            "exact_two_sided_p_value": p_value,
            "test": "exact McNemar / two-sided exact binomial on discordant pairs",
        })
    write_csv(REPORTS / "wrong_direction_contingency.csv", contingency_rows)
    write_csv(REPORTS / "paired_exact_tests.csv", test_rows)

    ci_rows = []
    for cohort_name, rescue_rows in (("WrongDirection", wd_rescue[:3]), ("TooWeak", tw_rescue)):
        for result in rescue_rows:
            ci_rows.append({
                "cohort": cohort_name, "host_mode": result["host_mode"],
                "successes": result["rescued_n"], "total": result["graph_missed_n"],
                "proportion": result["rescue_rate_among_graph_missed"],
                "exact_95_ci_low": result["rescue_rate_ci_low"], "exact_95_ci_high": result["rescue_rate_ci_high"],
                "method": "Clopper-Pearson exact",
            })
    for result in false_rows:
        ci_rows.append({
            "cohort": "SafeCorrect_GraphNegative", "host_mode": result["host_mode"],
            "successes": result["false_rescue_candidate_n"], "total": result["graph_negative_safe_correct_n"],
            "proportion": result["false_rescue_candidate_rate"],
            "exact_95_ci_low": result["false_rescue_candidate_ci_low"],
            "exact_95_ci_high": result["false_rescue_candidate_ci_high"],
            "method": "Clopper-Pearson exact",
        })
    write_csv(REPORTS / "binomial_ci.csv", ci_rows)

    summary_rows = []
    for mode in MODES:
        wd = next(row for row in wd_rescue if row["host_mode"] == mode)
        tw = next(row for row in tw_rescue if row["host_mode"] == mode)
        false = next(row for row in false_rows if row["host_mode"] == mode)
        numerator = wd["rescued_n"]
        denominator = false["false_rescue_candidate_n"]
        summary_rows.append({
            "host_mode": mode,
            "wrong_direction_graph_missed_n": wd["graph_missed_n"],
            "wrong_direction_rescued_n": numerator,
            "wrong_direction_rescue_rate": wd["rescue_rate_among_graph_missed"],
            "wrong_direction_rescue_ci_low": wd["rescue_rate_ci_low"],
            "wrong_direction_rescue_ci_high": wd["rescue_rate_ci_high"],
            "safe_correct_graph_negative_n": false["graph_negative_safe_correct_n"],
            "false_rescue_candidate_n": denominator,
            "false_rescue_candidate_rate": false["false_rescue_candidate_rate"],
            "false_rescue_candidate_ci_low": false["false_rescue_candidate_ci_low"],
            "false_rescue_candidate_ci_high": false["false_rescue_candidate_ci_high"],
            "conditional_rescue_ratio_numerator": numerator,
            "conditional_rescue_ratio_denominator": denominator,
            "conditional_rescue_ratio": numerator / denominator if denominator else float("inf") if numerator else float("nan"),
            "rescue_precision_numerator": numerator,
            "rescue_precision_denominator": numerator + denominator,
            "rescue_precision_graph_negative_diagnostic": numerator / (numerator + denominator) if numerator + denominator else float("nan"),
            "too_weak_graph_missed_n": tw["graph_missed_n"],
            "too_weak_rescued_n": tw["rescued_n"],
            "too_weak_rescue_rate": tw["rescue_rate_among_graph_missed"],
        })
    write_csv(REPORTS / "complementarity_summary.csv", summary_rows)

    score_rows = []
    for score_name in ("graph_score", "critical_hostego_score", "attended_hostego_score"):
        score_rows.append({"subset": "WrongDirection_GraphMissed", "score": score_name, "n": len(wd_missed), **describe([row[score_name] for row in wd_missed])})
    write_csv(REPORTS / "wrong_direction_graph_missed_continuous_scores.csv", score_rows)

    available_internal = [field for field in INTERNAL_FIELDS if all(field in row for row in wd_missed)]
    internal_rows = []
    for mode in MODES:
        for status, subset in (
            ("rescued", [row for row in wd_missed if flags(row)[mode]]),
            ("not_rescued", [row for row in wd_missed if not flags(row)[mode]]),
        ):
            for field in available_internal:
                internal_rows.append({"host_mode": mode, "rescue_status": status, "field": field, "n": len(subset), **describe([row[field] for row in subset])})
    if internal_rows:
        write_csv(REPORTS / "existing_mechanism_internal_by_rescue_status.csv", internal_rows)

    # Evidence-based decision: no newly invented score threshold. Exact interval separation is used.
    supported_modes = []
    any_unique = False
    for row in summary_rows:
        if row["host_mode"] not in ("critical", "attended"):
            continue
        any_unique = any_unique or row["wrong_direction_rescued_n"] > 0
        if (row["wrong_direction_rescued_n"] > 0 and
                row["wrong_direction_rescue_ci_low"] > row["false_rescue_candidate_ci_high"]):
            supported_modes.append(row["host_mode"])
    if supported_modes:
        decision = "HOST_COMPLEMENTARITY_SUPPORTED"
        rationale = "At least one atomic Host/Ego block has nonzero Graph-missed WD capture and non-overlapping exact intervals favoring rescue over false-rescue-candidate rate."
    elif any_unique:
        decision = "HOST_COMPLEMENTARITY_WEAK"
        rationale = "Host/Ego has unique Graph-missed WD captures, but exact intervals do not separate rescue from false-rescue-candidate cost."
    else:
        decision = "HOST_COMPLEMENTARITY_NOT_SUPPORTED"
        rationale = "Neither atomic Host/Ego block uniquely captures a Graph-missed Wrong-Direction sample."
    (REPORTS / "COMPLEMENTARITY_DECISION.txt").write_text(
        decision + "\n" + rationale + "\nThis is a post-hoc diagnostic decision, not authorization for a deployable model change.\n",
        encoding="utf-8",
    )

    after_hashes = {path: sha256(Path(path)) for path in manifest["input_sha256"]}
    if after_hashes != manifest["input_sha256"]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: an input file changed during the read-only audit")
    with (PREFLIGHT / "sha256_after.txt").open("w", encoding="utf-8") as handle:
        for path, digest in after_hashes.items():
            handle.write(f"{digest}  {path}\n")

    wd_graph_captured = len(wrong_direction) - len(wd_missed)
    lookup = {row["host_mode"]: row for row in summary_rows}
    inter = intersection
    p_lookup = {row["comparison"]: row for row in test_rows}
    lines = [
        "CARE-Droid Host Complementarity Audit",
        "Strict same-year OOF post-hoc diagnostic; not unseen, future, zero-day, or an independent temporal test.",
        "No model/probe training, replay, threshold search, route modification, or prediction modification.",
        "",
        "HOST_COMPLEMENTARITY_AUDIT_DONE",
        "",
        "WrongDirection:",
        f"N={len(wrong_direction)}",
        f"Graph captured={wd_graph_captured}",
        f"Graph missed={len(wd_missed)}",
        "",
        "Among Graph-missed WD:",
    ]
    for mode, label in (("critical", "Critical"), ("attended", "Attended"), ("union", "Union")):
        row = lookup[mode]
        lines.append(f"{label} rescue={row['wrong_direction_rescued_n']}/{row['wrong_direction_graph_missed_n']} ({row['wrong_direction_rescue_rate']:.3%}) exact95CI=[{row['wrong_direction_rescue_ci_low']:.3%},{row['wrong_direction_rescue_ci_high']:.3%}]")
    lines.append(f"Intersection={inter['rescued_n']}/{inter['graph_missed_n']} ({inter['rescue_rate_among_graph_missed']:.3%})")
    lines.extend(["", "SafeCorrect among Graph-negative:"])
    for mode, label in (("critical", "Critical"), ("attended", "Attended"), ("union", "Union")):
        row = lookup[mode]
        lines.append(f"{label} false-rescue candidate={row['false_rescue_candidate_n']}/{row['safe_correct_graph_negative_n']} ({row['false_rescue_candidate_rate']:.3%})")
    lines.extend(["", "TooWeak:"])
    for mode, label in (("critical", "Critical"), ("attended", "Attended"), ("union", "Union")):
        row = lookup[mode]
        lines.append(f"{label} rescue={row['too_weak_rescued_n']}/{row['too_weak_graph_missed_n']} ({row['too_weak_rescue_rate']:.3%})")
    lines.extend(["", "Paired exact tests:"])
    for mode, label in (("critical", "Critical"), ("attended", "Attended"), ("union", "Union")):
        row = p_lookup[f"Graph vs {mode}"]
        lines.append(f"Graph vs {label}: b={row['graph_positive_host_negative']} c={row['graph_negative_host_positive']} p={row['exact_two_sided_p_value']:.6f}")
    lines.extend([
        "",
        f"DECISION={decision}",
        rationale,
        "",
        "Interpretation boundary: Existing Host/Ego representations may contain complementary signal for a subset of Graph-missed Wrong-Direction cases. This audit cannot show that Host/Ego solves Wrong Direction, causes correction improvement, or will improve CARE-Droid.",
    ])
    report = "\n".join(lines) + "\n"
    (REPORTS / "COMPLEMENTARITY_REPORT.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
