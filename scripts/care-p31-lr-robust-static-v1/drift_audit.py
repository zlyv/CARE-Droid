#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from common import (ACTIVE_ARMS, FORMAL_DIAG, FUTURE_LABELS, FUTURE_L0_STAGE2, OUTPUT, SOURCE_P31,
    YEARS, norm, read_csv, write_csv, dump_json)


def quantiles(values):
    values = np.asarray(values, float)
    return {f"q{int(q*100):02d}": float(np.quantile(values, q)) for q in (0, .05, .25, .5, .75, .95, 1)}


def psi(source, target):
    source = np.asarray(source, float); target = np.asarray(target, float)
    cuts = np.unique(np.quantile(source, np.linspace(0, 1, 11)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf; cuts[-1] = np.inf
    a = np.histogram(source, bins=cuts)[0] / len(source)
    b = np.histogram(target, bins=cuts)[0] / len(target)
    a = np.clip(a, 1e-6, None); b = np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def main():
    resolved = json.loads((FORMAL_DIAG / "resolved_inputs.json").read_text(encoding="utf-8"))
    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(FUTURE_LABELS)}
    probability_rows = []
    for arm in ACTIVE_ARMS:
        source = read_csv(OUTPUT / "references" / arm / "source_predictions.csv")
        future_path = FUTURE_L0_STAGE2 if arm == "L0_RAW" else OUTPUT / "frozen" / arm / "predictions_label_free.csv"
        future = read_csv(future_path)
        for scope, rows in [("2017", source)] + [(str(year), [row for row in future if int(row["year"]) == year]) for year in YEARS]:
            y = np.asarray([int(row["label"]) if scope == "2017" else labels[norm(row["sha256"])] for row in rows], int)
            p = np.asarray([float(row["Global24__prob"]) if scope == "2017" else float(row["stage1_prob"]) for row in rows])
            logit = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))
            for label in (0, 1):
                selected = y == label
                probability_rows.append({"arm": arm, "year": scope, "label": label, "n": int(selected.sum()),
                    **{f"prob_{k}": v for k, v in quantiles(p[selected]).items()},
                    **{f"logit_{k}": v for k, v in quantiles(logit[selected]).items()},
                    "above_0_5": float(np.mean(p[selected] >= .5))})
    write_csv(OUTPUT / "probability_drift_by_year_label.csv", probability_rows)

    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        source = np.asarray(archive["role_prevalence"], float)
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        future_ids = [norm(value) for value in archive["sha256"]]
        future = np.asarray(archive["role_prevalence"], float)
    future_year = {norm(row["sha256"]): int(row["year"]) for row in read_csv(FUTURE_L0_STAGE2)}
    keep = np.asarray([index for index, sid in enumerate(future_ids) if sid in future_year], int)
    future = future[keep]; future_ids = [future_ids[index] for index in keep]
    if len(future_ids) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future P31 diagnostic cohort={len(future_ids)}")
    years = np.asarray([future_year[sid] for sid in future_ids], int)
    try:
        from scipy.stats import wasserstein_distance
    except Exception:
        wasserstein_distance = None
    drift = []
    for year in YEARS:
        selected = years == year
        for role in range(31):
            drift.append({"year": year, "role_index": role, "source_mean": float(source[:, role].mean()),
                "target_mean": float(future[selected, role].mean()), "psi": psi(source[:, role], future[selected, role]),
                "wasserstein": float(wasserstein_distance(source[:, role], future[selected, role])) if wasserstein_distance else "UNAVAILABLE"})
    write_csv(OUTPUT / "p31_feature_drift.csv", drift)
    dump_json(OUTPUT / "drift_audit_summary.json", {"status": "DONE", "future_diagnostic_only": True,
        "future_used_for_selection": False, "largest_PSI": sorted(drift, key=lambda row: -float(row["psi"]))[:20]})
    print("P31_LR_ROBUST_DRIFT_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
