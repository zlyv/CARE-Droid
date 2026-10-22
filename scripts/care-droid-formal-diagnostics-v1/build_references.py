#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil

import numpy as np

from common import ROOT, SOURCE_P31, SOURCE_REFERENCE, load_future_cohort, norm, read_csv, write_csv, dump
from stage1_core import MEMBERS, fit_calibrated_committee, mean_opposition, top25

SEED = 20260925


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    out = ROOT / "references"
    frozen = out / "future_label_free.csv"
    if frozen.is_file() and not args.rebuild:
        print("[resume] frozen future P31-RF reference exists")
        return
    resolved = __import__("json").loads((ROOT / "resolved_inputs.json").read_text())
    future_cache = resolved["future_p31_cache"]
    source_ref = read_csv(SOURCE_REFERENCE)
    label = {norm(r["sha256"]): int(r["label"]) for r in source_ref}
    with np.load(SOURCE_P31, allow_pickle=False) as z:
        source_ids = np.asarray([norm(x) for x in z["sha256"]]); source_x = np.asarray(z["role_prevalence"], float)
    if source_x.shape != (799, 31) or set(source_ids) != set(label):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source P31/label contract changed")
    y = np.asarray([label[sid] for sid in source_ids])
    _, cohort = load_future_cohort()
    future_ids = np.asarray([sid for sid, _ in cohort]); years = np.asarray([year for _, year in cohort])
    with np.load(future_cache, allow_pickle=False) as z:
        ids = [norm(x) for x in z["sha256"]]; x = np.asarray(z["role_prevalence"], float)
    pos = {sid: i for i, sid in enumerate(ids)}
    future_x = x[[pos[sid] for sid in future_ids]]
    member = fit_calibrated_committee(source_x, y, future_x, SEED, inner_folds=4, jobs=args.jobs)
    rf = member[:, MEMBERS.index("RF")]
    opposition = mean_opposition(member, rf)
    route = np.zeros(len(rf), bool)
    for year in (2018, 2019, 2020, 2021):
        idx = np.flatnonzero(years == year)
        route[idx] = top25(opposition[idx], future_ids[idx])
    expected = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
    observed = {year: int(np.sum(route & (years == year))) for year in expected}
    if observed != expected:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future DI25 counts changed " + repr(observed))
    rows = []
    for i, sid in enumerate(future_ids):
        row = {"sha256": sid, "year": int(years[i]), "stage1_prob": float(rf[i]),
               "router_score": float(opposition[i]), "routed": int(route[i])}
        for j, name in enumerate(MEMBERS):
            row[f"p_P31__{name}"] = float(member[i, j])
        rows.append(row)
    write_csv(frozen, rows)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_REFERENCE, out / "source_predictions.csv")
    dump(out / "reference_audit.json", {
        "stage1": "P31 calibrated RF only", "committee_role": "routing only",
        "router": "RF-anchored MeanOpposition Top25% independently per year",
        "source_training_rows": 799, "future_rows": 1989, "route_by_year": observed,
        "future_labels_loaded": False, "seed": SEED,
    })
    print("FROZEN_P31RF_REFERENCE_PASS")
    print(f"future_rows=1989 routed={int(route.sum())} labels_loaded=false route_by_year={observed}")


if __name__ == "__main__":
    main()
