#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from common import (OUTPUT, SEED, diagnostic_assets, dump_json, metrics, parse_diagnostic_rows,
                    parse_source_formal, source_assets, write_csv)
from modeling import ARMS, correction_probability, crossfit_scores, fit_score, load_feature_matrix


def frozen_scores(source_rows, future_rows, arm, ordinal):
    source_x, _ = load_feature_matrix(source_rows, OUTPUT / "cache/features/source", arm)
    future_x, _ = load_feature_matrix(future_rows, OUTPUT / "cache/features/future_frozen", arm)
    source_y = np.asarray([row["label"] for row in source_rows], int)
    source_pred = np.asarray([row["stage1_pred"] for row in source_rows], int)
    future_pred = np.asarray([row["stage1_pred"] for row in future_rows], int)
    scores = np.full(len(future_rows), np.nan)
    for side in (0, 1):
        train = source_pred == side; test = future_pred == side
        target = (source_y[train] == (1 if side == 0 else 0)).astype(int)
        scores[test] = fit_score(source_x[train], target, future_x[test], SEED + ordinal * 1000 + side)
    if not np.isfinite(scores).all():
        raise RuntimeError("frozen direction scores incomplete")
    return scores


def tables(protocol, rows, probability_by_arm):
    output = []
    y = np.asarray([row["label"] for row in rows], int); base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool); formal = np.asarray([row["formal_prob"] for row in rows], float)
    scopes = [("POOLED", np.ones(len(rows), bool))] + [(str(year), np.asarray([row["year"] == year for row in rows])) for year in (2018, 2019, 2020, 2021)]
    for scope, mask in scopes:
        output.append({"protocol": protocol, "scope": scope, "arm": "STAGE1", **metrics(y[mask], base[mask], base[mask], np.zeros(mask.sum(), bool))})
        output.append({"protocol": protocol, "scope": scope, "arm": "FORMAL_A0", **metrics(y[mask], base[mask], formal[mask], route[mask])})
        for arm, probability in probability_by_arm.items():
            output.append({"protocol": protocol, "scope": scope, "arm": arm, **metrics(y[mask], base[mask], probability[mask], route[mask])})
    return output


def main() -> None:
    source_decision = json.loads((OUTPUT / "source_decision.json").read_text(encoding="utf-8"))
    source_status = source_decision.get("status")
    if source_status not in {"SUPPORTED_FOR_FROZEN_DIAGNOSTICS", "NOT_SUPPORTED"}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source protocol/decision is not frozen")
    assets = diagnostic_assets(required=True)
    source_rows = parse_source_formal(source_assets()["formal"])
    frozen_rows = parse_diagnostic_rows(assets["frozen"])
    within_rows = parse_diagnostic_rows(assets["within_a"]) + parse_diagnostic_rows(assets["within_b"])
    if len(frozen_rows) != 1989 or len(within_rows) != 1989:
        raise RuntimeError(f"future cohort mismatch frozen={len(frozen_rows)} within={len(within_rows)}")
    all_metrics, prediction_rows = [], []

    frozen_probabilities = {}
    for ordinal, arm in enumerate(ARMS):
        score = frozen_scores(source_rows, frozen_rows, arm, ordinal)
        probability, delta = correction_probability(frozen_rows, score); frozen_probabilities[arm] = probability
        for index, row in enumerate(frozen_rows):
            prediction_rows.append({"protocol": "FROZEN_2017_TO_FUTURE", "arm": arm, "sha256": row["sha256"],
                                    "year": row["year"], "fold": row["fold"], "label": row["label"],
                                    "stage1_prob": row["stage1_prob"], "routed": row["routed"],
                                    "direction_score": score[index], "bounded_delta": delta[index], "final_prob": probability[index]})
        print(f"CMW_FROZEN_ARM_DONE arm={arm}", flush=True)
    all_metrics.extend(tables("FROZEN_2017_TO_FUTURE", frozen_rows, frozen_probabilities))

    within_probabilities = {}
    for ordinal, arm in enumerate(ARMS):
        x, _ = load_feature_matrix(within_rows, OUTPUT / "cache/features/future_within", arm)
        score = np.full(len(within_rows), np.nan)
        for year in (2018, 2019, 2020, 2021):
            indexes = np.asarray([i for i, row in enumerate(within_rows) if row["year"] == year], int)
            local_rows = [within_rows[i] for i in indexes]
            local_folds = np.asarray([row["fold"] for row in local_rows], int)
            local_score, _ = crossfit_scores(local_rows, x[indexes], local_folds, SEED + 100000 + ordinal * 1000 + year)
            score[indexes] = local_score
        if not np.isfinite(score).all():
            raise RuntimeError("within-year scores incomplete")
        probability, delta = correction_probability(within_rows, score); within_probabilities[arm] = probability
        for index, row in enumerate(within_rows):
            prediction_rows.append({"protocol": "INDEPENDENT_SAME_YEAR_5FOLD", "arm": arm, "sha256": row["sha256"],
                                    "year": row["year"], "fold": row["fold"], "label": row["label"],
                                    "stage1_prob": row["stage1_prob"], "routed": row["routed"],
                                    "direction_score": score[index], "bounded_delta": delta[index], "final_prob": probability[index]})
        print(f"CMW_WITHIN_ARM_DONE arm={arm}", flush=True)
    all_metrics.extend(tables("INDEPENDENT_SAME_YEAR_5FOLD", within_rows, within_probabilities))
    write_csv(OUTPUT / "diagnostic_metrics.csv", all_metrics); write_csv(OUTPUT / "diagnostic_predictions.csv", prediction_rows)
    summary = {"status": "DIAGNOSTICS_COMPLETE", "source_status": source_status,
               "source_protocol_hash": source_decision["protocol_hash"],
               "future_used_for_selection": False, "future_rows": 1989, "within_rows": 1989,
               "negative_source_cannot_be_rescued_by_diagnostics": True, "metrics": all_metrics}
    dump_json(OUTPUT / "diagnostic_summary.json", summary)
    final_status = ("SUPPORTED_SOURCE_WITH_FROZEN_DIAGNOSTICS_COMPLETE" if source_status == "SUPPORTED_FOR_FROZEN_DIAGNOSTICS"
                    else "NOT_SUPPORTED_SOURCE_WITH_DIAGNOSTICS_COMPLETE")
    dump_json(OUTPUT / "final_decision.json", {"status": final_status, "source_status": source_status,
              "source_protocol_hash": source_decision["protocol_hash"], "future_used_for_selection": False,
              "negative_source_cannot_be_rescued_by_diagnostics": True,
              "future_rows_loaded_after_freeze": 1989})
    print("CARE_CMW_DIAGNOSTICS_DONE", flush=True)


if __name__ == "__main__":
    main()
