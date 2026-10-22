#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, MEMBERS, OUTPUT, RESOLVED, SEED, WITHIN_INPUTS, YEARS, metric, norm, read_csv,
    read_json, route_metric, write_csv)
from stage1_core import fit_calibrated_committee


def main() -> None:
    metadata_rows = read_csv(WITHIN_INPUTS[0]) + read_csv(WITHIN_INPUTS[1])
    metadata = {norm(row["sha256"]): {"year": int(float(row["year"])), "fold": int(float(row["fold"])),
        "label": int(float(row["label"]))} for row in metadata_rows}
    path = __import__("pathlib").Path(read_json(RESOLVED)["future_p31_cache"])
    with np.load(path, allow_pickle=False) as archive:
        all_ids = [norm(value) for value in archive["sha256"]]; all_x = np.asarray(archive["role_prevalence"], float)
    position = {sid: index for index, sid in enumerate(all_ids)}; checkpoint = OUTPUT / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    aggregate = []; per_fold = []; predictions = []; audit = []
    for year in YEARS:
        ids = np.asarray(sorted(sid for sid, value in metadata.items() if value["year"] == year))
        x = all_x[[position[sid] for sid in ids]]; y = np.asarray([metadata[sid]["label"] for sid in ids], int)
        folds = np.asarray([metadata[sid]["fold"] for sid in ids], int); member = np.full((len(ids), len(MEMBERS)), np.nan)
        for fold in range(1, 6):
            train = np.flatnonzero(folds != fold); test = np.flatnonzero(folds == fold); target = checkpoint / f"within_{year}_fold_{fold}.npz"
            if target.is_file():
                with np.load(target, allow_pickle=False) as archive:
                    stored_ids = [norm(value) for value in archive["sha256"]]; stored_probability = np.asarray(archive["member_probability"], float)
                if stored_ids != ids[test].tolist() or stored_probability.shape != (len(test), 4):
                    raise RuntimeError("BLOCKED_BY_CONTRACT: incompatible checkpoint " + str(target))
                local = stored_probability; resumed = True
            else:
                seed = SEED + year * 10000 + fold * 100
                local = fit_calibrated_committee(x[train], y[train], x[test], seed, inner_folds=3, jobs=4)
                np.savez_compressed(target, sha256=ids[test], member_probability=local); resumed = False
            member[test] = local; audit.append({"year": year, "fold": fold, "train_n": len(train), "validation_n": len(test),
                "inner_calibration_folds": 3, "resumed": resumed, "test_excluded_from_fit": True})
            print(f"P31_MEMBER_WITHIN year={year} fold={fold}/5 resumed={str(resumed).lower()}", flush=True)
        if not np.isfinite(member).all():
            raise RuntimeError(f"incomplete same-year probabilities year={year}")
        probabilities = {name: member[:, i] for i, name in enumerate(MEMBERS)}; probabilities["MEAN4"] = member.mean(axis=1)
        for arm in ARMS:
            probability = probabilities[arm]
            aggregate.append({"protocol": "independent_same_year_fixed_5fold", "scope": str(year), "arm": arm,
                **metric(y, probability), **route_metric(y, probability, member, ids, folds)})
            for fold in range(1, 6):
                selected = folds == fold
                per_fold.append({"protocol": "independent_same_year_fixed_5fold", "year": year, "fold": fold, "arm": arm,
                    **metric(y[selected], probability[selected]), **route_metric(y[selected], probability[selected], member[selected], ids[selected], np.zeros(selected.sum(), int))})
            for index, sid in enumerate(ids):
                predictions.append({"protocol": "independent_same_year_fixed_5fold", "sha256": sid, "year": year,
                    "fold": folds[index], "label": y[index], "arm": arm, "probability": probability[index],
                    "prediction": int(probability[index] >= .5)})
    for arm in ARMS:
        selected_rows = [row for row in predictions if row["arm"] == arm]
        y = np.asarray([row["label"] for row in selected_rows], int); probability = np.asarray([row["probability"] for row in selected_rows], float)
        # Pooled route metrics are sums of the independently budgeted year/fold routes, so only classification metrics are pooled here.
        aggregate.append({"protocol": "independent_same_year_fixed_5fold", "scope": "pooled", "arm": arm, **metric(y, probability)})
    write_csv(OUTPUT / "within_year_metrics.csv", aggregate); write_csv(OUTPUT / "within_year_fold_metrics.csv", per_fold)
    write_csv(OUTPUT / "within_year_predictions.csv", predictions); write_csv(OUTPUT / "within_year_fit_audit.csv", audit)
    print("P31_MEMBER_WITHIN_YEAR_DONE rows=1989", flush=True)


if __name__ == "__main__":
    main()

