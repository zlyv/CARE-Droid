#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from common import (EXPECTED_ROUTE, EXPECTED_YEARS, METHOD, ROOT, metric_counts,
                    norm, read_csv, write_csv)


def rows_by_sha(rows):
    result = {norm(row["sha256"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate SHA in prediction table")
    return result


def metrics(protocol, rows):
    output = []
    year_values = np.asarray([int(float(row["year"])) for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows])
    base = np.asarray([float(row["p_stage1"]) for row in rows])
    final = np.asarray([float(row["p_final"]) for row in rows])
    route = np.asarray([int(float(row["route_DI"])) for row in rows], dtype=bool)
    for scope, selected in ([(str(year), year_values == year) for year in sorted(set(year_values))] +
                            [("pooled", np.ones(len(rows), dtype=bool))]):
        count = metric_counts(y[selected], base[selected], final[selected], route[selected])
        base_prediction = base[selected] >= .5
        final_prediction = final[selected] >= .5
        output.append({
            "protocol": protocol, "scope": scope, "n": int(selected.sum()),
            "routed": count["routed"],
            "P31_BA": float(balanced_accuracy_score(y[selected], base_prediction)),
            "P31_AUC": float(roc_auc_score(y[selected], base[selected])),
            "P31_F1": float(f1_score(y[selected], base_prediction, zero_division=0)),
            "CARE_BA": float(balanced_accuracy_score(y[selected], final_prediction)),
            "CARE_AUC": float(roc_auc_score(y[selected], final[selected])),
            "CARE_F1": float(f1_score(y[selected], final_prediction, zero_division=0)),
            "fixed": count["fixed"], "broken": count["broken"], "net": count["net"],
            "delta_BA": float(balanced_accuracy_score(y[selected], final_prediction) -
                              balanced_accuracy_score(y[selected], base_prediction)),
        })
    return output


def main():
    frozen_prediction = read_csv(ROOT / "frozen_future/label_free_predictions.csv")
    labels = read_csv(Path(
        "artifacts/care-litelocal-p31-diagnostics-v1/within_year/oof_predictions.csv"))
    label_lookup = rows_by_sha(labels)
    frozen = []
    for row in frozen_prediction:
        sha = norm(row["sha256"])
        if sha not in label_lookup:
            raise RuntimeError("cannot join frozen diagnostic label for " + sha)
        frozen.append({**row, "label": label_lookup[sha]["label"]})
    if len(frozen) != 1989:
        raise RuntimeError("frozen prediction count changed")

    within = []
    for group in ("2018_2019", "2020_2021"):
        within.extend(read_csv(ROOT / f"within_year/{group}/oof_predictions.csv"))
    if len(within) != 1989 or len(rows_by_sha(within)) != 1989:
        raise RuntimeError("within-year prediction count changed")
    frozen_metrics = metrics("frozen_2017_to_future", frozen)
    within_metrics = metrics("independent_same_year_5fold", within)
    all_metrics = frozen_metrics + within_metrics
    write_csv(ROOT / "results/metrics.csv", all_metrics)
    write_csv(ROOT / "results/frozen_predictions_with_posthoc_labels.csv", frozen)
    write_csv(ROOT / "results/within_year_oof_predictions.csv", within)

    lines = [
        "CARE-Droid P31-DI25 + C43-RQ complete diagnostic v1", "",
        "Stage-I: P31 calibrated LR/RF/XGB/SVM mean. Router: P31 MeanOpposition Top25%.",
        "Stage-II: source-selected C43_RELATION_QUERY with Argmax Critical Host and bounded correction.",
        "No Ego tensor is consumed. The failed TopK/Aligned localization candidates are excluded.", "",
        "FROZEN 2017 -> 2018-2021 (future labels opened only by summarizer):",
    ]
    for row in frozen_metrics:
        lines.append(
            f"{row['scope']}: P31_BA={row['P31_BA']:.6f} CARE_BA={row['CARE_BA']:.6f} "
            f"CARE_AUC={row['CARE_AUC']:.6f} CARE_F1={row['CARE_F1']:.6f} "
            f"routed={row['routed']} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    lines.extend(["", "INDEPENDENT SAME-YEAR FIVE-FOLD OOF (not temporal generalization):"])
    for row in within_metrics:
        lines.append(
            f"{row['scope']}: P31_BA={row['P31_BA']:.6f} CARE_BA={row['CARE_BA']:.6f} "
            f"CARE_AUC={row['CARE_AUC']:.6f} CARE_F1={row['CARE_F1']:.6f} "
            f"routed={row['routed']} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    lines.extend([
        "", "Evidence boundary:",
        "2018-2021 were already inspected during development. Frozen results are temporal diagnostics, not a new untouched test.",
        "Same-year OOF uses annual labels for training and is not temporal generalization.",
        "No future result may be used to retune P31, DI25, C43-RQ, or the 2.25 correction bound.",
    ])
    report = "\n".join(lines) + "\n"
    (ROOT / "results/REPORT.txt").write_text(report, encoding="utf-8")
    hard = {
        "method": METHOD,
        "source_locked_result": {"P31_BA": 0.9261309523809524,
                                 "CARE_BA": 0.9336434837092732,
                                 "fixed": 8, "broken": 2, "net": 6},
        "frozen_future_labels_loaded_by_model": False,
        "same_year_is_temporal_generalization": False,
        "ego_consumed": False,
        "future_route_counts": EXPECTED_ROUTE,
        "future_year_counts": EXPECTED_YEARS,
    }
    (ROOT / "results/hard_audit.json").write_text(
        json.dumps(hard, indent=2) + "\n", encoding="utf-8")
    print("P31_C43_COMPLETE_DIAGNOSTICS_SUMMARY_PASS")
    print(report)


if __name__ == "__main__":
    main()
