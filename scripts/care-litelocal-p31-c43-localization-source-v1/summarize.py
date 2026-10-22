#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path("artifacts/care-litelocal-p31-c43-localization-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
ARMS = ("L0_LOCKED", "L1_TOPK3_RAW", "L2_ARGMAX_ALIGNED", "L3_TOPK3_ALIGNED")
METHODS = {
    "L0_LOCKED": "P31Mean__DI25__C43RelationQuery__Bounded",
    **{arm: f"P31Mean__DI25__C43Localization__{arm}__Bounded" for arm in ARMS[1:]},
}


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def metric(y, probability):
    prediction = probability >= .5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"BA": float(balanced_accuracy_score(y, prediction)),
            "AUC": float(roc_auc_score(y, probability)),
            "F1": float(f1_score(y, prediction)), "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp), "errors": int(np.sum(prediction != y))}


def arm_path(arm):
    return (P31_ROOT / "arms/P31_C43_RQ/source_predictions.csv" if arm == "L0_LOCKED"
            else ROOT / "arms" / arm / "source_predictions.csv")


def mechanism(y, base, final, delta, route):
    base_correct = (base >= .5) == y
    final_correct = (final >= .5) == y
    result = {"routed_errors": int(np.sum(~base_correct & route)),
              "routed_correct": int(np.sum(base_correct & route)),
              "fixed": int(np.sum(~base_correct & final_correct & route)),
              "broken": int(np.sum(base_correct & ~final_correct & route))}
    for kind, mask in (("FN", (y == 1) & (base < .5) & route),
                       ("FP", (y == 0) & (base >= .5) & route)):
        fixed = mask & final_correct
        right = mask & ~final_correct & (((delta > 0) if kind == "FN" else (delta < 0)))
        wrong = mask & ~final_correct & ~right
        result[kind + "_n"] = int(mask.sum())
        result[kind + "_fixed"] = int(fixed.sum())
        result[kind + "_too_weak"] = int(right.sum())
        result[kind + "_wrong_or_zero"] = int(wrong.sum())
    result["wrong_or_zero"] = result["FN_wrong_or_zero"] + result["FP_wrong_or_zero"]
    result["too_weak"] = result["FN_too_weak"] + result["FP_too_weak"]
    result["net"] = result["fixed"] - result["broken"]
    return result


def main():
    reference = sorted(read(P31_ROOT / "preflight/p31_reference/source_predictions.csv"),
                       key=lambda row: row["sha256"].casefold())
    ids = [row["sha256"].casefold() for row in reference]
    y = np.asarray([int(float(row["label"])) for row in reference])
    fold = np.asarray([int(float(row["fold"])) for row in reference])
    base = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in reference])
    route = np.asarray([int(float(row["P31_DI25__routed"])) == 1 for row in reference])
    arm_data, aggregate, mech_rows, per_fold = {}, [], [], []
    aggregate.append({"arm": "P31_MEAN_COMMITTEE", **metric(y, base),
                      "routed": 0, "fixed": 0, "broken": 0, "net": 0,
                      "wrong_or_zero": 0, "too_weak": 0})
    for arm in ARMS:
        rows = sorted(read(arm_path(arm)), key=lambda row: row["sha256"].casefold())
        if [row["sha256"].casefold() for row in rows] != ids:
            raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch " + arm)
        method = METHODS[arm]
        final = np.asarray([float(row[method + "__prob"]) for row in rows])
        expert = np.asarray([float(row[method + "__expert_prob"]) for row in rows])
        delta = np.asarray([float(row[method + "__bounded_delta"]) for row in rows])
        observed_base = np.asarray([float(row["Global24__prob"]) for row in rows])
        observed_route = np.asarray([int(float(row["A25__routed"])) == 1 for row in rows])
        if np.max(np.abs(observed_base - base)) > 1e-12 or np.any(observed_route != route):
            raise RuntimeError("BLOCKED_BY_CONTRACT: base/route changed " + arm)
        if np.max(np.abs(final[~route] - base[~route])) > 1e-12:
            raise RuntimeError("BLOCKED_BY_CONTRACT: correction outside route " + arm)
        mech = mechanism(y, base, final, delta, route)
        arm_data[arm] = {"final": final, "expert": expert, "delta": delta,
                         "correct": (final >= .5) == y, "mech": mech}
        aggregate.append({"arm": arm, **metric(y, final), "routed": int(route.sum()), **mech})
        mech_rows.append({"arm": arm, **mech})
        for value in range(1, 6):
            selected = fold == value
            fold_mech = mechanism(y[selected], base[selected], final[selected],
                                  delta[selected], route[selected])
            per_fold.append({"arm": arm, "fold": value,
                             **metric(y[selected], final[selected]), **fold_mech})
    l0 = arm_data["L0_LOCKED"]["correct"]
    paired = []
    for arm in ARMS[1:]:
        candidate = arm_data[arm]["correct"]
        candidate_only = int(np.sum(candidate & ~l0))
        l0_only = int(np.sum(l0 & ~candidate))
        total = candidate_only + l0_only
        p = float(binomtest(min(candidate_only, l0_only), total, .5).pvalue) if total else 1.0
        fold_net = []
        for value in range(1, 6):
            selected = fold == value
            fold_net.append(int(np.sum(candidate[selected] & ~l0[selected])) -
                            int(np.sum(l0[selected] & ~candidate[selected])))
        paired.append({"comparison": arm + "_minus_L0_LOCKED",
                       "candidate_only_correct": candidate_only,
                       "L0_only_correct": l0_only, "paired_net": candidate_only-l0_only,
                       "exact_p": p, "nonnegative_paired_folds": sum(x >= 0 for x in fold_net),
                       "positive_paired_folds": sum(x > 0 for x in fold_net)})
    results = ROOT / "results"
    write(results / "aggregate_metrics.csv", aggregate)
    write(results / "failure_mechanisms.csv", mech_rows)
    write(results / "per_fold_metrics.csv", per_fold)
    write(results / "paired_vs_L0.csv", paired)
    l0m = arm_data["L0_LOCKED"]["mech"]
    candidates = [arm for arm in ARMS[1:]
                  if arm_data[arm]["mech"]["wrong_or_zero"] < l0m["wrong_or_zero"]
                  and arm_data[arm]["mech"]["net"] > l0m["net"]
                  and arm_data[arm]["mech"]["broken"] <= l0m["broken"]]
    audit = {"source_rows": 799, "year": 2017, "route_P31_DI25": 200,
             "parameter_count_all_arms": 32292, "future_rows_loaded": 0,
             "ego_consumed": False, "automatic_future_authorized": False,
             "source_candidates_meeting_predeclared_pareto_rule": candidates}
    (results / "hard_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    base_metric = metric(y, base)
    lines = ["CARE-Droid P31-C43 localization 2x2 source v1", "",
             "Strict 2017 five-fold OOF; P31-DI25=200/799; C43 Host-only; no Ego/future row.",
             "L0 is locked and not retrained. Candidates preserve Relation31, backgrounds, 113-D correction input, bounded head, and 32,292 parameters.",
             "TopK3 reuses normalized attention over the three highest-score Hosts. Aligned Relation uses a fixed mean-preserving local-evidence gate.", "",
             f"P31: BA={base_metric['BA']:.6f} AUC={base_metric['AUC']:.6f} errors={base_metric['errors']}"]
    for row in aggregate[1:]:
        lines.append(f"{row['arm']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']} wrong={row['wrong_or_zero']} weak={row['too_weak']}")
    lines.extend(["", "Predeclared Pareto rule: wrong-direction < L0, net > L0, and broken <= L0.",
                  "SOURCE_PARETO_CANDIDATES=" + (",".join(candidates) if candidates else "NONE"),
                  "No future diagnostic is automatically authorized; inspect folds and paired changes first."])
    (results / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P31_C43_LOCALIZATION_SUMMARY_PASS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
