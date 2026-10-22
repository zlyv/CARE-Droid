#!/usr/bin/env python3
"""Produce the frozen terminal/report decision for the diagnostic audit."""
from __future__ import annotations

from pathlib import Path

from audit_common import (
    AUDIT, BOOTSTRAPS, FEATURES, REPORTS, ROOT, ensure_dirs, hash_files,
    load_json, read_csv, write_json, write_text,
)


CORE = ["A", "B", "C", "ABC"]


def pooled_metrics(rows):
    return {r["candidate"]: r for r in rows if r["year"] == "pooled"}


def floatv(row, key):
    return float(row[key])


def main():
    ensure_dirs()
    manifest = load_json(ROOT / "preflight" / "source_manifest.json")
    metrics = pooled_metrics(read_csv(REPORTS / "need_correction_metrics.csv"))
    conditional = {r["candidate"]: r for r in read_csv(REPORTS / "conditional_rescue.csv")}
    separation = {r["candidate"]: r for r in read_csv(REPORTS / "conditional_separation_bootstrap.csv")}
    paired = read_csv(REPORTS / "paired_bootstrap.csv")
    failure = read_csv(REPORTS / "failure_type_rescue.csv")
    existing = read_csv(REPORTS / "existing_host_miss_rescue.csv")
    complexity = read_csv(REPORTS / "feature_complexity.csv")

    versus_random = {r["candidate"]: r for r in paired if r["baseline"] == "Random"}
    classifications = {}
    for candidate in CORE:
        random_row = versus_random[candidate]
        general = (
            floatv(random_row, "delta_auc_ci_low") > 0
            and floatv(random_row, "delta_pr_auc_ci_low") > 0
        )
        complementary = floatv(separation[candidate], "ci_low") > 0
        point_positive = floatv(conditional[candidate], "conditional_separation") > 0
        rescued = int(conditional[candidate]["graph_negative_error_rescued"]) > 0
        if general and complementary:
            label = "SUPPORTED"
        elif point_positive and rescued:
            label = "WEAK_SIGNAL"
        else:
            label = "NOT_SUPPORTED"
        classifications[candidate] = {
            "classification": label,
            "general_nonrandom": general,
            "conditional_complementarity": complementary,
            "conditional_point_positive": point_positive,
        }

    supported = [c for c in CORE if classifications[c]["classification"] == "SUPPORTED"]
    winner = "NO_UNIQUE_WINNER"
    paired_map = {(r["baseline"], r["candidate"]): r for r in paired}

    def paired_supports(left, right):
        """Return True only if left is CI-supported above right on AUC and PR."""
        if (right, left) in paired_map:
            row = paired_map[(right, left)]
            return float(row["delta_auc_ci_low"]) > 0 and float(row["delta_pr_auc_ci_low"]) > 0
        if (left, right) in paired_map:
            row = paired_map[(left, right)]
            return float(row["delta_auc_ci_high"]) < 0 and float(row["delta_pr_auc_ci_high"]) < 0
        return False

    if supported:
        ranked = sorted(
            supported,
            key=lambda c: (
                float(metrics[c]["ROC_AUC"]), float(metrics[c]["PR_AUC"]),
                float(conditional[c]["conditional_separation"]),
            ), reverse=True,
        )
        best = ranked[0]
        strictly_best = all(
            float(metrics[best]["ROC_AUC"]) > float(metrics[c]["ROC_AUC"])
            and float(metrics[best]["PR_AUC"]) > float(metrics[c]["PR_AUC"])
            and float(conditional[best]["conditional_separation"])
                >= float(conditional[c]["conditional_separation"])
            for c in supported if c != best
        )
        paired_best = all(paired_supports(best, c) for c in supported if c != best)
        if strictly_best and paired_best:
            winner = best

    wd = {}
    tw = {}
    for r in failure:
        if r["candidate"] in CORE and r["scope"] == "GraphNegative":
            target = wd if r["failure_type"] == "WrongDirection" else tw
            target[r["candidate"]] = r
    miss = {r["candidate"]: r for r in existing if r["cohort"] == "NeedCorrection"}
    miss_wd = {r["candidate"]: r for r in existing if r["cohort"] == "WrongDirection"}
    feature_count = {r["feature_group"]: int(r["feature_count"]) for r in complexity}
    coordination = load_json(REPORTS / "coordination_coverage.json")
    size_summary = {
        r["candidate"]: r for r in read_csv(REPORTS / "size_shortcut_audit.csv")
        if r["row_type"] == "candidate_summary"
    }

    lines = [
        "CARE-Droid Host Complement Candidate Audit v1",
        "",
        "Strict same-year five-fold OOF diagnostic on frozen routed samples only.",
        "No CARE-Droid/Global24/existing probe training, no neural replay, no APK reanalysis, and no old prediction modification.",
        "This is development/diagnostic evidence, not unseen-future, zero-day, or final-test evidence.",
        "",
        "HOST_COMPLEMENT_CANDIDATE_AUDIT_DONE",
        "",
        "Pooled NeedCorrection probes:",
    ]
    for candidate in ["A", "B", "C", "AB", "AC", "BC", "ABC"]:
        m = metrics[candidate]
        c = conditional[candidate]
        s = separation[candidate]
        lines.append(
            f"{candidate}: BA={float(m['BA']):.6f} AUC={float(m['ROC_AUC']):.6f} "
            f"PR-AUC={float(m['PR_AUC']):.6f}; Graph-negative rescue="
            f"{c['graph_negative_error_rescued']}/{c['graph_negative_error_n']} "
            f"({float(c['rescue_rate']):.3%}); false-rescue="
            f"{c['false_rescue_candidate_n']}/{c['graph_negative_safe_correct_n']} "
            f"({float(c['false_rescue_rate']):.3%}); CS={float(c['conditional_separation']):.6f} "
            f"95% bootstrap CI=[{float(s['ci_low']):.6f},{float(s['ci_high']):.6f}]"
        )
    lines.extend(["", "Required final questions:"])
    for i, candidate in enumerate(CORE, 1):
        lines.append(
            f"Q{i}. {candidate} classification: {classifications[candidate]['classification']} "
            f"(general_nonrandom={classifications[candidate]['general_nonrandom']}, "
            f"conditional_complementarity={classifications[candidate]['conditional_complementarity']})."
        )
    lines.append(f"Q5. Unique winning candidate set: {winner}.")
    lines.append(
        "Q6. Graph-missed WrongDirection rescue: "
        + "; ".join(f"{c}={wd[c]['rescued']}/{wd[c]['n']}" for c in CORE)
    )
    lines.append(
        "Q7. Graph-missed TooWeak rescue: "
        + "; ".join(f"{c}={tw[c]['rescued']}/{tw[c]['n']}" for c in CORE)
    )
    lines.append(
        "Q8. Existing-Host-miss NeedCorrection rescue: "
        + "; ".join(f"{c}={miss[c]['rescued']}/{miss[c]['n']}" for c in CORE)
    )
    lines.append(
        "Q9. Feature dimensions: "
        f"A={feature_count['A']}, B={feature_count['B']}, C={feature_count['C']}, "
        f"A_pair={feature_count['A_pair']}, C_transition={feature_count['C_transition']}."
    )
    lines.append(
        "Q10. Decision rule: SUPPORTED requires both non-random general OOF signal and "
        "positive Graph-negative conditional separation with a 95% bootstrap lower bound above zero; "
        "no arbitrary AUC cut-off is used."
    )
    lines.append(
        "Q11. Interpretation boundary: probe readability and rescue association are diagnostic; "
        "they do not establish causal contribution or authorize a CARE-Droid model change."
    )
    lines.append(f"Q12. Bootstrap iterations={BOOTSTRAPS}; confidence intervals are reported explicitly.")

    report = "\n".join(lines) + "\n"
    write_text(REPORTS / "HOST_COMPLEMENT_CANDIDATE_REPORT.txt", report)
    decision_lines = [
        f"A={classifications['A']['classification']}",
        f"B={classifications['B']['classification']}",
        f"C={classifications['C']['classification']}",
        f"ABC={classifications['ABC']['classification']}",
        f"PRIMARY_HOST_COMPLEMENT_CANDIDATE={winner}",
        "METHOD_CHANGE_AUTHORIZED=NO",
    ]
    write_text(REPORTS / "HOST_COMPLEMENT_CANDIDATE_DECISION.txt", "\n".join(decision_lines) + "\n")
    write_json(REPORTS / "decision.json", {
        "status": "HOST_COMPLEMENT_CANDIDATE_AUDIT_DONE",
        "candidate_classification": classifications,
        "unique_winner": winner,
        "old_predictions_modified": False,
        "method_change_authorized": False,
    })

    # Final immutable-input check.
    before = manifest["original_sha256"]
    after = hash_files([Path(p) for p in before])
    changed = [p for p in before if after.get(p) != before[p]]
    write_text(AUDIT / "original_sha256_after.txt", "".join(f"{digest}  {path}\n" for path, digest in after.items()))
    write_json(AUDIT / "no_original_modification.json", {
        "passed": not changed, "changed_files": changed,
        "old_artifacts_read_only": True,
    })
    if changed:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: old artifact hash changed: {changed}")
    concise = [
        "HOST_COMPLEMENT_CANDIDATE_AUDIT_DONE", "",
        "Routed cohort:", f"N={metrics['A']['N']}", "",
        "Need-Correction pooled OOF:",
    ]
    for candidate in CORE:
        concise.append(f"{candidate}: AUC={float(metrics[candidate]['ROC_AUC']):.6f} PR={float(metrics[candidate]['PR_AUC']):.6f}")
    concise += ["", "Graph-negative NeedCorrection:", f"N={conditional['A']['graph_negative_error_n']}"]
    for candidate in CORE:
        c, s = conditional[candidate], separation[candidate]
        concise.append(
            f"{candidate} rescue={c['graph_negative_error_rescued']}/{c['graph_negative_error_n']} ({float(c['rescue_rate']):.3%}); "
            f"false-rescue={c['false_rescue_candidate_n']}/{c['graph_negative_safe_correct_n']} ({float(c['false_rescue_rate']):.3%}); "
            f"conditional separation={float(c['conditional_separation']):.6f} CI=[{float(s['ci_low']):.6f},{float(s['ci_high']):.6f}]"
        )
    concise += ["", "Graph-missed WrongDirection:", f"N={wd['A']['n']}"]
    for candidate in CORE:
        concise.append(f"{candidate} rescue={wd[candidate]['rescued']}/{wd[candidate]['n']}")
    concise += ["", "ExistingHostMiss WrongDirection:", f"N={miss_wd['A']['n']}"]
    for candidate in CORE:
        concise.append(f"{candidate} new rescue={miss_wd[candidate]['rescued']}/{miss_wd[candidate]['n']}")
    concise += [
        "", "Coordination coverage:",
        f"gap1 APK coverage={coordination['routed_gap1_APK_coverage']:.3%}",
        f"gap2 APK coverage={coordination['routed_gap2_APK_coverage']:.3%}",
        "", "Size shortcut:",
    ]
    for candidate in ["A", "B", "C"]:
        concise.append(f"{candidate} max|rho|={float(size_summary[candidate]['max_absolute_rho']):.6f}")
    concise += ["", "DECISIONS:"] + decision_lines[:3] + [decision_lines[4], "", "All original dependency SHA256 values unchanged."]
    print("\n".join(concise))


if __name__ == "__main__":
    main()
