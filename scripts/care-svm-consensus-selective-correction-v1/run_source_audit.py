#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

from common import (OUTPUT, SEED, SOURCE_MEMBERS, SOURCE_STAGE2, binary_auc, dump, metrics,
                    norm, read_csv, write_csv)

ARMS = ("S0_STAGE1_SVM", "S1_ALL_C43", "S2_MAJORITY_2_OF_3", "S3_UNANIMOUS_3_OF_3")


def final_for(candidate, base, flip, support_count, threshold):
    accept_flip = support_count >= threshold
    # Only class-changing corrections are filtered. Non-flipping probability refinements stay identical to S1.
    return np.where(flip & (~accept_flip), base, candidate), accept_flip


def main():
    stage2 = sorted([row for row in read_csv(SOURCE_STAGE2) if norm(row.get("arm")) == "svm"], key=lambda row: norm(row["sha256"]))
    members = sorted(read_csv(SOURCE_MEMBERS), key=lambda row: norm(row["sha256"]))
    sha = np.asarray([norm(row["sha256"]) for row in stage2]); y = np.asarray([int(row["label"]) for row in stage2])
    fold = np.asarray([int(row["fold"]) for row in stage2]); route = np.asarray([int(row["routed"]) for row in stage2], bool)
    base = np.asarray([float(row["stage1_prob"]) for row in stage2]); candidate = np.asarray([float(row["stage2_prob"]) for row in stage2])
    expert = np.asarray([float(row["expert_prob"]) for row in stage2])
    other = np.asarray([[float(row["p_P31__LR"]), float(row["p_P31__RF"]), float(row["p_P31__XGB"])] for row in members])
    base_label = base >= .5; candidate_label = candidate >= .5; flip = base_label != candidate_label
    support_count = np.sum((other >= .5) == candidate_label[:, None], axis=1)
    direction_support = np.where(candidate_label, other.mean(axis=1), 1 - other.mean(axis=1))
    expert_support = np.where(candidate_label, expert, 1 - expert)
    primary_trust = .5 * direction_support + .5 * expert_support
    before_ok = base_label == y; after_ok = candidate_label == y
    fixed = flip & (~before_ok) & after_ok; broken = flip & before_ok & (~after_ok)
    changed = flip
    if (int(fixed.sum()), int(broken.sum())) != (29, 4):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen SVM+C43 outcome changed fixed={fixed.sum()} broken={broken.sum()}")

    majority, majority_accept = final_for(candidate, base, flip, support_count, 2)
    unanimous, unanimous_accept = final_for(candidate, base, flip, support_count, 3)
    finals = {"S0_STAGE1_SVM": base, "S1_ALL_C43": candidate,
              "S2_MAJORITY_2_OF_3": majority, "S3_UNANIMOUS_3_OF_3": unanimous}
    aggregate, per_fold, predictions = [], [], []
    for arm in ARMS:
        value = finals[arm]; aggregate.append({"arm": arm, **metrics(y, base, value, route)})
        for current in range(1, 6):
            selected = fold == current
            per_fold.append({"arm": arm, "fold": current, **metrics(y[selected], base[selected], value[selected], route[selected])})
        accepted = np.ones(len(y), bool) if arm in ("S0_STAGE1_SVM", "S1_ALL_C43") else (majority_accept if arm.startswith("S2") else unanimous_accept)
        for i in range(len(y)):
            predictions.append({"sha256": sha[i], "fold": int(fold[i]), "label": int(y[i]), "arm": arm,
                "stage1_prob": float(base[i]), "candidate_prob": float(candidate[i]), "final_prob": float(value[i]),
                "routed": int(route[i]), "candidate_flip": int(flip[i]), "accepted_flip": int(accepted[i]) if flip[i] else "",
                "other_support_count": int(support_count[i]), "other_direction_support": float(direction_support[i]),
                "expert_direction_support": float(expert_support[i]), "primary_trust": float(primary_trust[i]),
                "candidate_fixed": int(fixed[i]), "candidate_broken": int(broken[i])})

    # Fixed orientation: larger score means more trustworthy. No feature or sign is selected from outcomes.
    signal_rows = []
    target = fixed[changed].astype(int)
    signals = {"OTHER_DIRECTION_SUPPORT": direction_support[changed], "EXPERT_DIRECTION_SUPPORT": expert_support[changed],
               "PRIMARY_EQUAL_WEIGHT_TRUST": primary_trust[changed], "OTHER_HARD_SUPPORT_COUNT": support_count[changed].astype(float)}
    for name, values in signals.items():
        signal_rows.append({"signal": name, "changed": int(changed.sum()), "fixed": int(target.sum()),
            "broken": int(len(target) - target.sum()), "fixed_vs_broken_AUC": binary_auc(target, values)})
        for current in range(1, 6):
            local = changed & (fold == current); local_target = fixed[local].astype(int)
            signal_rows.append({"signal": name, "fold": current, "changed": int(local.sum()),
                "fixed": int(local_target.sum()), "broken": int(len(local_target) - local_target.sum()),
                "fixed_vs_broken_AUC": binary_auc(local_target, (primary_trust if name == "PRIMARY_EQUAL_WEIGHT_TRUST" else
                    direction_support if name == "OTHER_DIRECTION_SUPPORT" else expert_support if name == "EXPERT_DIRECTION_SUPPORT" else support_count)[local])})

    # Matched random control: preserve the majority rule's accepted flip count within fold and proposed class.
    rng = np.random.RandomState(SEED); random_rows = []
    majority_metric = metrics(y, base, majority, route)
    for iteration in range(2000):
        accept = np.zeros(len(y), bool)
        for current in range(1, 6):
            for proposed in (0, 1):
                eligible = np.flatnonzero(flip & (fold == current) & (candidate_label == proposed))
                count = int(np.sum(majority_accept[eligible]))
                if count:
                    accept[rng.choice(eligible, count, replace=False)] = True
        final = np.where(flip & (~accept), base, candidate)
        random_rows.append({"iteration": iteration, **metrics(y, base, final, route)})
    null_net = np.asarray([row["net"] for row in random_rows], float)
    null_broken = np.asarray([row["broken"] for row in random_rows], float)
    p_net = float((1 + np.sum(null_net >= majority_metric["net"])) / (1 + len(null_net)))
    null_summary = {"iterations": len(random_rows), "matching": "fold x proposed correction class x accepted flip count",
        "majority_net": majority_metric["net"], "majority_broken": majority_metric["broken"],
        "random_net_mean": float(null_net.mean()), "random_net_95pct": float(np.quantile(null_net, .95)),
        "random_broken_mean": float(null_broken.mean()), "random_broken_5pct": float(np.quantile(null_broken, .05)),
        "one_sided_p_net": p_net}

    all_c43 = next(row for row in aggregate if row["arm"] == "S1_ALL_C43")
    maj = next(row for row in aggregate if row["arm"] == "S2_MAJORITY_2_OF_3")
    fixed_retention = maj["fixed"] / max(all_c43["fixed"], 1)
    all_by_fold = {row["fold"]: row for row in per_fold if row["arm"] == "S1_ALL_C43"}
    maj_by_fold = {row["fold"]: row for row in per_fold if row["arm"] == "S2_MAJORITY_2_OF_3"}
    noninferior = sum(maj_by_fold[f]["net"] >= all_by_fold[f]["net"] for f in range(1, 6))
    primary_auc = next(row["fixed_vs_broken_AUC"] for row in signal_rows if row["signal"] == "PRIMARY_EQUAL_WEIGHT_TRUST" and "fold" not in row)
    criteria = {"majority_net>S1_net": maj["net"] > all_c43["net"],
        "majority_broken<S1_broken": maj["broken"] < all_c43["broken"],
        "fixed_retention>=0.75": fixed_retention >= .75,
        "noninferior_folds>=3": noninferior >= 3,
        "primary_trust_AUC>=0.70": primary_auc >= .70,
        "majority_net>matched_random_95pct": maj["net"] > null_summary["random_net_95pct"]}
    supported = all(criteria.values())
    decision = {"status": "SUPPORTED_FOR_STRICT_NESTED_GATE" if supported else "SVM_CONSENSUS_GATE_NOT_SUPPORTED",
        "criteria": criteria, "future_rows_loaded": 0, "source_only": True, "fixed_retention": fixed_retention,
        "noninferior_folds": noninferior, "primary_trust_fixed_vs_broken_AUC": primary_auc,
        "reason": "A strict nested learned selector is authorized only if every predeclared source gate passes."}
    write_csv(OUTPUT / "source_results.csv", aggregate); write_csv(OUTPUT / "source_fold_results.csv", per_fold)
    write_csv(OUTPUT / "source_predictions.csv", predictions); write_csv(OUTPUT / "trust_signal_audit.csv", signal_rows)
    write_csv(OUTPUT / "matched_random_iterations.csv", random_rows)
    dump(OUTPUT / "matched_random_control.json", null_summary); dump(OUTPUT / "source_decision.json", decision)
    print("SVM_CONSENSUS_SELECTIVE_CORRECTION_SOURCE_DONE")
    for row in aggregate:
        print(f"{row['arm']}: BA={row['BA']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    print("SVM_CONSENSUS_SOURCE_DECISION=" + decision["status"])


if __name__ == "__main__":
    main()
