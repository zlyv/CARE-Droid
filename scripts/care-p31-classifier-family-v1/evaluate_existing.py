#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, FUTURE_LABELED, FUTURE_MEMBER, MEMBERS, OUTPUT, SOURCE_MEMBER, YEAR_COUNTS,
    binary_auc, dump_json, metric, norm, read_csv, route_metric, write_csv)


def probability_matrix(rows):
    member = np.asarray([[float(row[f"p_P31__{name}"]) for name in MEMBERS] for row in rows], float)
    if not np.isfinite(member).all():
        raise RuntimeError("nonfinite member probability")
    return member


def evaluate_source() -> None:
    rows = read_csv(SOURCE_MEMBER); sha = np.asarray([norm(row["sha256"]) for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows], int); fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    member = probability_matrix(rows); probabilities = {name: member[:, i] for i, name in enumerate(MEMBERS)}
    probabilities["MEAN4"] = member.mean(axis=1); aggregate = []; per_fold = []; predictions = []
    for arm in ARMS:
        probability = probabilities[arm]; value = {**metric(y, probability), **route_metric(y, probability, member, sha, fold)}
        aggregate.append({"protocol": "source_2017_fixed_oof", "scope": "pooled", "arm": arm, **value})
        for current_fold in range(1, 6):
            selected = fold == current_fold
            per_fold.append({"protocol": "source_2017_fixed_oof", "fold": current_fold, "arm": arm,
                **metric(y[selected], probability[selected]), **route_metric(y[selected], probability[selected], member[selected], sha[selected], np.zeros(selected.sum(), int))})
        for i in range(len(rows)):
            predictions.append({"protocol": "source_2017_fixed_oof", "sha256": sha[i], "fold": fold[i], "label": y[i],
                "arm": arm, "probability": probability[i], "prediction": int(probability[i] >= .5)})
    paired = []; rf_ok = (probabilities["RF"] >= .5) == y
    try:
        from scipy.stats import binomtest
    except Exception:
        binomtest = None
    for arm in ARMS:
        candidate_ok = (probabilities[arm] >= .5) == y; candidate_only = int(np.sum(candidate_ok & (~rf_ok))); rf_only = int(np.sum(rf_ok & (~candidate_ok)))
        p_value = float(binomtest(min(candidate_only, rf_only), candidate_only + rf_only, .5).pvalue) if binomtest and candidate_only + rf_only else 1.0
        paired.append({"arm": arm, "candidate_only_correct": candidate_only, "RF_only_correct": rf_only,
            "paired_net_vs_RF": candidate_only - rf_only, "mcnemar_exact_p": p_value})
    ranking = sorted(aggregate, key=lambda row: (-row["BA"], -row["AUC"], -row["F1"], row["arm"]))
    dump_json(OUTPUT / "source_ranking_frozen_before_future.json", {"ranking": [row["arm"] for row in ranking],
        "metric": "BA then AUC then F1", "leader": ranking[0]["arm"], "future_used": False,
        "formal_stage1_changed": False})
    write_csv(OUTPUT / "source_metrics.csv", aggregate); write_csv(OUTPUT / "source_fold_metrics.csv", per_fold)
    write_csv(OUTPUT / "source_predictions.csv", predictions); write_csv(OUTPUT / "source_paired_vs_rf.csv", paired)
    print("P31_MEMBER_SOURCE_DONE leader=" + ranking[0]["arm"], flush=True)


def evaluate_frozen() -> None:
    rows = read_csv(FUTURE_MEMBER); labels = {norm(row["sha256"]): int(float(row["label"])) for row in read_csv(FUTURE_LABELED)}
    sha = np.asarray([norm(row["sha256"]) for row in rows]); year = np.asarray([int(float(row["year"])) for row in rows], int)
    y = np.asarray([labels[sid] for sid in sha], int); member = probability_matrix(rows)
    probabilities = {name: member[:, i] for i, name in enumerate(MEMBERS)}; probabilities["MEAN4"] = member.mean(axis=1)
    aggregate = []; predictions = []
    for scope, selected in [(str(value), year == value) for value in YEAR_COUNTS] + [("pooled", np.ones(len(rows), bool))]:
        groups = year[selected]
        for arm in ARMS:
            probability = probabilities[arm][selected]
            aggregate.append({"protocol": "frozen_2017_to_future", "scope": scope, "arm": arm,
                **metric(y[selected], probability), **route_metric(y[selected], probability, member[selected], sha[selected], groups)})
    for arm in ARMS:
        for i in range(len(rows)):
            predictions.append({"protocol": "frozen_2017_to_future", "sha256": sha[i], "year": year[i], "label": y[i],
                "arm": arm, "probability": probabilities[arm][i], "prediction": int(probabilities[arm][i] >= .5)})
    write_csv(OUTPUT / "frozen_future_metrics.csv", aggregate); write_csv(OUTPUT / "frozen_future_predictions.csv", predictions)
    print("P31_MEMBER_FROZEN_FUTURE_DONE rows=1989 diagnostic_only=true", flush=True)


if __name__ == "__main__":
    evaluate_source(); evaluate_frozen()

