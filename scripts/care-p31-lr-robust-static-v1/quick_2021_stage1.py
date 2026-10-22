#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, FUTURE_LABELS, FUTURE_REFERENCE, OUTPUT, SOURCE_MEMBER,
    SOURCE_P31, classification_metrics, dump_json, norm, read_csv, write_csv)
from stage1_core import fit_calibrated_member
from transforms import feature_names, transform_p31


def fit_lr(x_train, y_train, x_target, arm, seed):
    return fit_calibrated_member("LR", transform_p31(x_train, arm), y_train,
        transform_p31(x_target, arm), seed, inner_folds=3, jobs=4)


def main():
    source_rows = read_csv(SOURCE_MEMBER)
    source_ids = np.asarray([norm(row["sha256"]) for row in source_rows])
    source_y = np.asarray([int(float(row["label"])) for row in source_rows], int)
    source_fold = np.asarray([int(float(row["fold"])) for row in source_rows], int)
    formal_source_lr = np.asarray([float(row["p_P31__LR"]) for row in source_rows], float)
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        ids = [norm(value) for value in archive["sha256"]]
        values = np.asarray(archive["role_prevalence"], float)
    position = {sid: index for index, sid in enumerate(ids)}
    source_x = values[[position[sid] for sid in source_ids]]

    future_rows = read_csv(FUTURE_REFERENCE)
    label_map = {norm(row["sha256"]): int(float(row["label"])) for row in read_csv(FUTURE_LABELS)}
    rows_2021 = [row for row in future_rows if int(float(row["year"])) == 2021]
    ids_2021 = np.asarray([norm(row["sha256"]) for row in rows_2021])
    y_2021 = np.asarray([label_map[sid] for sid in ids_2021], int)
    resolved = __import__("json").loads((__import__("pathlib").Path(
        "/home/boyuanjie/projects/tasa-droid/artifacts/care-droid-formal-diagnostics-v1/resolved_inputs.json"
    )).read_text(encoding="utf-8"))
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        future_ids = [norm(value) for value in archive["sha256"]]
        future_values = np.asarray(archive["role_prevalence"], float)
    future_position = {sid: index for index, sid in enumerate(future_ids)}
    x_2021 = future_values[[future_position[sid] for sid in ids_2021]]

    metrics = []; predictions = []; probability_audit = []
    for arm_index, arm in enumerate(ARMS):
        if arm == "L0_RAW":
            source_probability = formal_source_lr
            target_probability = np.asarray([float(row["p_P31__LR"]) for row in rows_2021], float)
        else:
            source_probability = np.full(len(source_y), np.nan)
            for fold in range(1, 6):
                train = np.flatnonzero(source_fold != fold); test = np.flatnonzero(source_fold == fold)
                source_probability[test] = fit_lr(source_x[train], source_y[train], source_x[test], arm,
                    20261009 + arm_index * 100000 + fold * 1000)
            target_probability = fit_lr(source_x, source_y, x_2021, arm,
                20261009 + arm_index * 100000 + 99000)
        source_metric = classification_metrics(source_y, source_probability)
        target_metric = classification_metrics(y_2021, target_probability)
        metrics.append({"arm": arm, "feature_width": len(feature_names(arm)),
            **{f"source_{key}": value for key, value in source_metric.items()},
            **{f"y2021_{key}": value for key, value in target_metric.items()},
            "y2021_BA_gain_vs_raw_pp": 100 * (target_metric["BA"] - .54)})
        for index, sid in enumerate(ids_2021):
            predictions.append({"sha256": sid, "year": 2021, "label": int(y_2021[index]),
                "arm": arm, "probability": float(target_probability[index]),
                "prediction": int(target_probability[index] >= .5)})
        for label in (0, 1):
            selected = y_2021 == label; values_local = target_probability[selected]
            probability_audit.append({"arm": arm, "label": label, "n": int(selected.sum()),
                "mean": float(values_local.mean()), "q05": float(np.quantile(values_local, .05)),
                "q25": float(np.quantile(values_local, .25)), "median": float(np.quantile(values_local, .5)),
                "q75": float(np.quantile(values_local, .75)), "q95": float(np.quantile(values_local, .95)),
                "above_0_5": float(np.mean(values_local >= .5))})
        print(f"P31_LR_ROBUST_QUICK_2021 arm={arm} source_BA={source_metric['BA']:.6f} "
              f"y2021_BA={target_metric['BA']:.6f} sensitivity={target_metric['TP']/max(target_metric['TP']+target_metric['FN'],1):.6f}", flush=True)
    ranking_source = [row["arm"] for row in sorted(metrics, key=lambda row: (-row["source_BA"], -row["source_AUC"], row["arm"]))]
    ranking_2021 = [row["arm"] for row in sorted(metrics, key=lambda row: (-row["y2021_BA"], -row["y2021_AUC"], row["arm"]))]
    write_csv(OUTPUT / "quick_2021_stage1_metrics.csv", metrics)
    write_csv(OUTPUT / "quick_2021_stage1_predictions.csv", predictions)
    write_csv(OUTPUT / "quick_2021_probability_audit.csv", probability_audit)
    dump_json(OUTPUT / "quick_2021_decision.json", {"status": "DIAGNOSTIC_COMPLETE",
        "source_ranking": ranking_source, "y2021_diagnostic_ranking": ranking_2021,
        "future_used_for_selection": False, "stage2_run": False,
        "interpretation": "2021 is already-seen frozen diagnostic evidence and cannot select the formal representation."})
    print("CARE_P31_LR_ROBUST_QUICK_2021_DONE", flush=True)


if __name__ == "__main__":
    main()
