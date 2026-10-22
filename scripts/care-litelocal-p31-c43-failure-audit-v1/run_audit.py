#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_core import bounded_failure, logit

ROOT = Path("artifacts/care-litelocal-p31-c43-failure-audit-v1")
P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
DIRECT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
BOUNDED_METHOD = "P31Mean__DI25__C43RelationQuery__Bounded"
DIRECT_METHOD = "P31Mean__DI25__C43RelationQuery__Direct"


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    reference = sorted(read(P31 / "preflight/p31_reference/source_predictions.csv"),
                       key=lambda row: row["sha256"].casefold())
    bounded_rows = sorted(read(P31 / "arms/P31_C43_RQ/source_predictions.csv"),
                          key=lambda row: row["sha256"].casefold())
    direct_rows = sorted(read(DIRECT / "arms/P31_C43_DIRECT/source_predictions.csv"),
                         key=lambda row: row["sha256"].casefold())
    error_rows, correct_rows = [], []
    failure_counts = Counter()
    overlap_errors = Counter()
    overlap_correct = Counter()
    by_fold = defaultdict(Counter)
    by_type = defaultdict(Counter)
    max_effective_delta = 0.0
    for reference_row, bounded_row, direct_row in zip(reference, bounded_rows, direct_rows):
        sha = reference_row["sha256"].casefold()
        label = int(float(reference_row["label"]))
        fold = int(float(reference_row["fold"]))
        base = float(reference_row["P31_MEAN_COMMITTEE__prob"])
        route = int(float(reference_row["P31_DI25__routed"])) == 1
        if not route:
            continue
        bounded_expert = float(bounded_row[BOUNDED_METHOD + "__expert_prob"])
        direct_expert = float(direct_row[DIRECT_METHOD + "__expert_prob"])
        base_correct = int((base >= .5) == bool(label))
        bounded_correct = int((bounded_expert >= .5) == bool(label))
        direct_correct = int((direct_expert >= .5) == bool(label))
        if not base_correct:
            mechanism = bounded_failure(label, base, bounded_expert)
            max_effective_delta = max(max_effective_delta,
                                      abs(mechanism["effective_bounded_delta"]))
            overlap = ("BOTH_FIX" if bounded_correct and direct_correct
                       else "BOUNDED_ONLY_FIX" if bounded_correct
                       else "DIRECT_ONLY_FIX" if direct_correct
                       else "NEITHER_FIX")
            error_type = "FN" if label == 1 else "FP"
            row = {
                "sha256": sha, "fold": fold, "label": label,
                "stage1_error_type": error_type,
                "P31_probability": base,
                "P31_abs_logit": abs(logit(base)),
                "bounded_expert_probability": bounded_expert,
                "direct_expert_probability": direct_expert,
                "direct_correct": direct_correct,
                "bounded_direct_overlap": overlap,
                **mechanism,
            }
            error_rows.append(row)
            failure_counts[mechanism["primary_failure_type"]] += 1
            overlap_errors[overlap] += 1
            by_fold[fold][mechanism["primary_failure_type"]] += 1
            by_type[error_type][mechanism["primary_failure_type"]] += 1
        else:
            overlap = ("BOTH_PRESERVE" if bounded_correct and direct_correct
                       else "BOUNDED_ONLY_PRESERVE" if bounded_correct
                       else "DIRECT_ONLY_PRESERVE" if direct_correct
                       else "BOTH_BREAK")
            correct_rows.append({
                "sha256": sha, "fold": fold, "label": label,
                "P31_probability": base,
                "bounded_expert_probability": bounded_expert,
                "direct_expert_probability": direct_expert,
                "bounded_preserved": bounded_correct,
                "direct_preserved": direct_correct,
                "bounded_direct_overlap": overlap,
            })
            overlap_correct[overlap] += 1
    if len(error_rows) != 46 or len(correct_rows) != 154:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed cohort accounting failed")
    if max_effective_delta > 2.25 + 1e-6:
        raise RuntimeError("BLOCKED_BY_CONTRACT: recovered bounded delta exceeds 2.25")
    summary_rows = []
    for key, value in sorted(failure_counts.items()):
        summary_rows.append({"scope": "bounded_error_mechanism", "category": key,
                             "count": value, "denominator": 46,
                             "ratio": value / 46})
    for key, value in sorted(overlap_errors.items()):
        summary_rows.append({"scope": "error_fix_overlap", "category": key,
                             "count": value, "denominator": 46,
                             "ratio": value / 46})
    for key, value in sorted(overlap_correct.items()):
        summary_rows.append({"scope": "correct_preservation_overlap", "category": key,
                             "count": value, "denominator": 154,
                             "ratio": value / 154})
    fold_rows = [{"fold": fold, **dict(counts)}
                 for fold, counts in sorted(by_fold.items())]
    type_rows = [{"stage1_error_type": kind, **dict(counts)}
                 for kind, counts in sorted(by_type.items())]
    write(ROOT / "routed_errors_mechanism.csv", error_rows)
    write(ROOT / "routed_correct_safety.csv", correct_rows)
    write(ROOT / "mechanism_summary.csv", summary_rows)
    write(ROOT / "mechanism_by_fold.csv", fold_rows)
    write(ROOT / "mechanism_by_error_type.csv", type_rows)
    audit = {
        "source_rows": 799, "routed_rows": 200,
        "routed_stage1_errors": 46, "routed_stage1_correct": 154,
        "max_recovered_abs_bounded_delta": max_effective_delta,
        "bounded_failure_counts": dict(failure_counts),
        "error_fix_overlap": dict(overlap_errors),
        "correct_preservation_overlap": dict(overlap_correct),
        "training_invoked": False, "inference_invoked": False,
        "route_modified": False, "future_rows_loaded": 0,
    }
    (ROOT / "hard_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "CARE-Droid P31-DI25 C43-RQ failure-mechanism audit v1", "",
        "Read-only strict-2017 audit of 46 routed P31 errors and 154 routed P31-correct APKs.",
        "No training, inference, route change, threshold search, future row, or Ego tensor.", "",
        "Bounded mechanism among 46 routed errors:",
    ]
    lines += [f"  {key}={value}/46 ({value/46:.3%})"
              for key, value in sorted(failure_counts.items())]
    lines += ["", "Bounded/Direct fix overlap among errors:"]
    lines += [f"  {key}={value}/46 ({value/46:.3%})"
              for key, value in sorted(overlap_errors.items())]
    lines += ["", "Bounded/Direct preservation overlap among 154 routed-correct APKs:"]
    lines += [f"  {key}={value}/154 ({value/154:.3%})"
              for key, value in sorted(overlap_correct.items())]
    lines += ["", f"max_recovered_abs_bounded_delta={max_effective_delta:.12g}",
              "Primary interpretation must follow the counts; do not infer feature causality from probabilities alone."]
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P31_C43_FAILURE_MECHANISM_AUDIT_DONE")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
