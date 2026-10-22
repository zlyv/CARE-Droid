#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (BASELINE_STAGE2, LABELS, OUTPUT, TARGET_YEARS, WINDOWS, YEAR_COUNTS,
    comparison, metrics, norm, read_csv, write_csv, dump_json)


def load_prediction_rows(window):
    path = BASELINE_STAGE2 if window == "W17" else OUTPUT / window / "predictions_label_free.csv"
    rows = [row for row in read_csv(path) if int(row["year"]) in TARGET_YEARS]
    if len(rows) != 1196:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: {window} target rows={len(rows)}")
    return rows


def main():
    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(LABELS)}
    aggregate = []; predictions = []
    for window in ("W17",) + WINDOWS:
        rows = load_prediction_rows(window); ids = np.asarray([norm(row["sha256"]) for row in rows])
        year = np.asarray([int(row["year"]) for row in rows], int); y = np.asarray([labels[sid] for sid in ids], int)
        base = np.asarray([float(row["stage1_prob"]) for row in rows]); final = np.asarray([float(row["stage2_prob"]) for row in rows])
        route = np.asarray([int(row["routed"]) for row in rows], bool)
        for scope in list(TARGET_YEARS) + ["pooled"]:
            selected = np.ones(len(rows), bool) if scope == "pooled" else year == scope
            if scope != "pooled" and int(selected.sum()) != YEAR_COUNTS[int(scope)]:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: {window}/{scope} count changed")
            stage1 = metrics(y[selected], base[selected]); stage2 = comparison(y[selected], base[selected], final[selected], route[selected])
            aggregate.append({"window": window, "train_years": "2017" if window == "W17" else ("2018" if window == "W18" else "2017+2018"),
                "scope": scope, **{"Stage1_" + key: value for key, value in stage1.items()},
                **{"Stage2_" + key: value for key, value in stage2.items()},
                "gain_pp": 100 * (stage2["BA"] - stage1["BA"])})
        for i, sid in enumerate(ids):
            bp = int(base[i] >= .5); fp = int(final[i] >= .5)
            predictions.append({"window": window, "sha256": sid, "year": int(year[i]), "label": int(y[i]),
                "stage1_prob": float(base[i]), "stage1_pred": bp, "routed": int(route[i]),
                "stage2_prob": float(final[i]), "stage2_pred": fp,
                "fixed": int(bp != y[i] and fp == y[i]), "broken": int(bp == y[i] and fp != y[i])})
    write_csv(OUTPUT / "train_window_results.csv", aggregate); write_csv(OUTPUT / "train_window_predictions.csv", predictions)
    pooled = {row["window"]: row for row in aggregate if row["scope"] == "pooled"}
    w17 = pooled["W17"]; w18 = pooled["W18"]; combined = pooled["W1718"]
    if float(w18["Stage2_BA"]) > float(w17["Stage2_BA"]) and float(combined["Stage2_BA"]) <= float(w18["Stage2_BA"]) + .001:
        diagnosis = "RECENCY_OR_DISTRIBUTION_DOMINATES"
    elif float(combined["Stage2_BA"]) > max(float(w17["Stage2_BA"]), float(w18["Stage2_BA"])) + .001:
        diagnosis = "ADDED_SAMPLE_DIVERSITY_SUPPORTED"
    elif float(combined["Stage2_BA"]) < max(float(w17["Stage2_BA"]), float(w18["Stage2_BA"])) - .001:
        diagnosis = "POOLING_NEGATIVE_TRANSFER_OR_NO_SAMPLE_SIZE_BENEFIT"
    else:
        diagnosis = "INCONCLUSIVE"
    dump_json(OUTPUT / "evaluation_decision.json", {"status": "EVALUATED", "diagnosis": diagnosis,
        "primary_endpoint": "2019-2021 pooled and per-year SVM+C43 BA plus FP-side net",
        "W17": w17, "W18": w18, "W1718": combined,
        "sample_size_claim_rule": "Only W1718 beating both W17 and size-matched W18 supports added sample diversity/size.",
        "target_labels_used_for_fit": False, "target_labels_loaded_for_evaluation": True,
        "future_is_diagnostic_only": True})
    print("SVM_TRAIN_WINDOW_EVALUATION_DONE diagnosis=" + diagnosis, flush=True)


if __name__ == "__main__":
    main()
