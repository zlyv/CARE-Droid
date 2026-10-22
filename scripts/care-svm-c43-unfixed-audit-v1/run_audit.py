#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

import numpy as np

from common import (MAX_DELTA, NEAR_ZERO, OUTPUT, SCALES, dump_json, load_source,
                    logit, metrics, sigmoid, write_csv)


def error_category(fixed, aligned, required):
    if fixed: return "FIXED"
    if required > MAX_DELTA + 1e-9: return "STAGE1_MARGIN_EXCEEDS_MAX_DELTA"
    if abs(aligned) < NEAR_ZERO: return "NEAR_ZERO"
    if aligned < 0: return "WRONG_DIRECTION"
    return "CORRECT_DIRECTION_TOO_WEAK"


def correct_category(broken, harm_aligned):
    if broken: return "BROKEN"
    if abs(harm_aligned) < NEAR_ZERO: return "NEAR_ZERO"
    if harm_aligned > 0: return "TOWARD_WRONG_CLASS_BUT_NOT_FLIPPED"
    return "REINFORCED_OR_PRESERVED_CORRECT_CLASS"


def main():
    rows, ids, y, fold, base, final, route = load_source(); base_logit = logit(base)
    delta = np.where(route, np.clip(logit(final) - base_logit, -MAX_DELTA, MAX_DELTA), 0.0)
    bp = base >= .5; fp = final >= .5; before = bp == y; after = fp == y
    routed_error = route & (~before); routed_correct = route & before
    if int(routed_error.sum()) != 84 or int(routed_correct.sum()) != 116:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed SVM composition changed")
    error_rows = []; correct_rows = []
    for i in np.flatnonzero(routed_error):
        target_sign = 1.0 if y[i] == 1 else -1.0; aligned = target_sign * delta[i]; required = abs(base_logit[i])
        fixed = bool(after[i]); category = error_category(fixed, aligned, required)
        error_rows.append({"sha256": ids[i], "fold": int(fold[i]), "label": int(y[i]), "side": "FN" if y[i] else "FP",
            "stage1_prob": float(base[i]), "stage1_abs_logit_margin": float(required), "c43_prob": float(final[i]),
            "bounded_delta": float(delta[i]), "target_aligned_delta": float(aligned), "direction_correct": int(aligned > 0),
            "fixed": int(fixed), "max_delta_crossable": int(required <= MAX_DELTA), "delta_saturated": int(abs(delta[i]) >= MAX_DELTA - .01),
            "minimum_scale_to_cross": float(required / aligned) if aligned > 0 else None, "primary_category": category})
    for i in np.flatnonzero(routed_correct):
        true_sign = 1.0 if y[i] == 1 else -1.0; harm = -true_sign * delta[i]; required = abs(base_logit[i]); broken = bool(not after[i])
        correct_rows.append({"sha256": ids[i], "fold": int(fold[i]), "label": int(y[i]), "side": "TP" if y[i] else "TN",
            "stage1_prob": float(base[i]), "stage1_abs_logit_margin": float(required), "c43_prob": float(final[i]),
            "bounded_delta": float(delta[i]), "wrong_class_aligned_delta": float(harm), "broken": int(broken),
            "minimum_scale_to_break": float(required / harm) if harm > 0 else None,
            "primary_category": correct_category(broken, harm)})
    write_csv(OUTPUT / "routed_error_anatomy.csv", error_rows); write_csv(OUTPUT / "routed_correct_risk.csv", correct_rows)

    category_rows = []
    for scope, selected in [("ALL", error_rows), ("FP", [r for r in error_rows if r["side"] == "FP"]),
                            ("FN", [r for r in error_rows if r["side"] == "FN"])]:
        counts = Counter(row["primary_category"] for row in selected)
        for category in ("FIXED", "CORRECT_DIRECTION_TOO_WEAK", "WRONG_DIRECTION", "NEAR_ZERO", "STAGE1_MARGIN_EXCEEDS_MAX_DELTA"):
            category_rows.append({"scope": scope, "category": category, "count": counts[category],
                "fraction": counts[category] / max(len(selected), 1), "n": len(selected)})
    write_csv(OUTPUT / "routed_error_category_summary.csv", category_rows)

    fold_rows = []
    for current in range(1, 6):
        local = [row for row in error_rows if row["fold"] == current]; counts = Counter(row["primary_category"] for row in local)
        fold_rows.append({"fold": current, "routed_errors": len(local), "fixed": counts["FIXED"],
            "correct_too_weak": counts["CORRECT_DIRECTION_TOO_WEAK"], "wrong_direction": counts["WRONG_DIRECTION"],
            "near_zero": counts["NEAR_ZERO"], "margin_exceeds_cap": counts["STAGE1_MARGIN_EXCEEDS_MAX_DELTA"]})
    write_csv(OUTPUT / "routed_error_fold_summary.csv", fold_rows)

    scale_rows = []; scale_folds = []
    for scale in SCALES:
        scaled = np.where(route, np.clip(delta * scale, -MAX_DELTA, MAX_DELTA), 0.0); probability = sigmoid(base_logit + scaled)
        scale_rows.append({"scale": scale, **metrics(y, base, probability, route)})
        for current in range(1, 6):
            selected = fold == current
            scale_folds.append({"scale": scale, "fold": current, **metrics(y[selected], base[selected], probability[selected], route[selected])})
    write_csv(OUTPUT / "descriptive_scale_sweep.csv", scale_rows); write_csv(OUTPUT / "descriptive_scale_sweep_folds.csv", scale_folds)

    unresolved = [row for row in error_rows if not row["fixed"]]; unresolved_crossable = [row for row in unresolved if row["max_delta_crossable"]]
    correct_weak = [row for row in unresolved_crossable if row["primary_category"] == "CORRECT_DIRECTION_TOO_WEAK"]
    weak_fraction = len(correct_weak) / max(len(unresolved_crossable), 1)
    weak_majority_folds = sum(next(r for r in fold_rows if r["fold"] == current)["correct_too_weak"] >
                              next(r for r in fold_rows if r["fold"] == current)["wrong_direction"] + next(r for r in fold_rows if r["fold"] == current)["near_zero"]
                              for current in range(1, 6))
    baseline = next(row for row in scale_rows if row["scale"] == 1.0)
    exploratory_better = [row for row in scale_rows if row["scale"] > 1 and row["net"] > baseline["net"] and row["broken"] <= baseline["broken"]]
    calibration_candidate = weak_fraction >= .5 and weak_majority_folds >= 3 and bool(exploratory_better)
    if calibration_candidate: status = "SVM_C43_MAGNITUDE_CALIBRATION_WORTH_NESTED_TEST"
    elif weak_fraction >= .5: status = "CORRECT_DIRECTION_WEAK_BUT_GLOBAL_SCALING_NOT_SAFE"
    else: status = "SVM_C43_REMAINING_ERRORS_NOT_PRIMARILY_MAGNITUDE_LIMITED"
    decision = {"status": status, "future_rows_loaded": 0, "routed_errors": len(error_rows), "routed_correct": len(correct_rows),
        "fixed": sum(row["fixed"] for row in error_rows), "broken": sum(row["broken"] for row in correct_rows),
        "unresolved": len(unresolved), "unresolved_crossable": len(unresolved_crossable),
        "correct_direction_too_weak": len(correct_weak), "correct_weak_fraction_of_crossable_unresolved": weak_fraction,
        "weak_majority_folds": weak_majority_folds, "max_delta_uncrossable": sum(not row["max_delta_crossable"] for row in unresolved),
        "descriptive_scaling_is_model_selection": False, "exploratory_better_scales": [row["scale"] for row in exploratory_better],
        "continue_to_strict_nested_magnitude_test": calibration_candidate,
        "note": "Scale sweep is retrospective anatomy only and cannot change the formal model."}
    dump_json(OUTPUT / "final_decision.json", decision)
    print("SVM_C43_UNFIXED_AUDIT_DONE", flush=True); print(decision, flush=True)


if __name__ == "__main__": main()

