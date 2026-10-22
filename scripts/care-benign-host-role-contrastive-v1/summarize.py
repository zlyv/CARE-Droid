#!/usr/bin/env python3
"""Evaluate frozen Host-Role pilot models with unchanged Manifest-RF routes.

Reads only saved predictions. Writes exclusively into this experiment's output.
All 2018-2021 results are development diagnostics, not independent validation.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
BASE = PROJECT / "outputs/care_multiyear_development_v1"
OUTPUT = PROJECT / "outputs/care_benign_host_role_contrastive_v1"
MANIFEST = (
    PROJECT
    / "outputs/care_manifest_four_member_disagreement_v1"
    / "run_20261019_pilot01/per_apk_predictions.csv"
)
TAGS = {
    "2018": "C0R_2017",
    "2019": "T2_C1",
    "2020": "T3_C1",
    "2021": "T4_C1",
}
ARMS = ("B0_ORIGINAL", "H1_TRUE_ROLE", "H2_ROLE_SHUFFLE")
STRATEGIES = ("ANCHOR_UNC25_REF", "OTHER3_OP25_REF")
SEEDS = (900000, 900001, 900002)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Empty table {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metric(rows):
    y = np.array([r["label"] for r in rows], int)
    stage1 = np.array([r["stage1_probability"] for r in rows], float)
    expert = np.array([r["expert_probability"] for r in rows], float)
    routed = np.array([r["routed"] for r in rows], int)
    final = np.where(routed > 0, expert, stage1)
    base_pred = stage1 >= 0.5
    pred = final >= 0.5
    base_wrong = base_pred != y
    fixed = int(np.sum(base_wrong & (pred == y)))
    broken = int(np.sum(~base_wrong & (pred != y)))
    return dict(
        n=len(rows),
        routed=int(routed.sum()),
        stage1_errors=int(base_wrong.sum()),
        routed_stage1_errors=int(np.sum(base_wrong & (routed > 0))),
        BA=float(balanced_accuracy_score(y, pred)),
        AUC=float(roc_auc_score(y, final)),
        FN=int(np.sum((y == 1) & (~pred))),
        FP=int(np.sum((y == 0) & pred)),
        fixed=fixed,
        broken=broken,
        net=fixed - broken,
    )


def prediction_map(path, year):
    with np.load(path) as z:
        ids = list(map(str, z["sha256"] if "sha256" in z else z["ids"]))
        p = np.asarray(z["probability"], float)
    if len(ids) != len(set(ids)) or len(ids) != len(p):
        raise RuntimeError(f"Malformed prediction array: {path}")
    return dict(zip(ids, p))


def main():
    stage1_rows = [
        r for r in read_csv(MANIFEST)
        if r["anchor"] == "RF" and r["strategy"] in STRATEGIES
        and r["year"] in TAGS
    ]
    if len(stage1_rows) != 3 * 1989 * len(STRATEGIES):
        raise RuntimeError(f"Unexpected Manifest-RF row count {len(stage1_rows)}")
    expert_maps = {}
    for year, tag in TAGS.items():
        base = prediction_map(BASE / "predictions" / tag / "expert.npz", year)
        expert_maps[(year, "B0_ORIGINAL")] = base
        for arm in ARMS[1:]:
            preds = []
            for seed in SEEDS:
                path = (
                    OUTPUT / "full" / tag / str(seed) / f"{arm}_predictions.npz"
                )
                if not path.is_file():
                    raise RuntimeError(f"Missing training output {path}")
                preds.append(prediction_map(path, year))
            if any(set(p) != set(base) for p in preds):
                raise RuntimeError(f"Expert SHA mismatch {year}/{arm}")
            expert_maps[(year, arm)] = {
                sid: float(np.mean([p[sid] for p in preds])) for sid in base
            }
    grouped = defaultdict(list)
    per_apk = []
    for r in stage1_rows:
        year, sid = r["year"], r["sha256"]
        for arm in ARMS:
            expert = expert_maps[(year, arm)].get(sid)
            if expert is None:
                raise RuntimeError(f"Missing expert row {year}/{sid}/{arm}")
            row = dict(
                stage1_seed=r["seed"],
                year=year,
                sha256=sid,
                strategy=r["strategy"],
                arm=arm,
                label=int(r["label"]),
                stage1_probability=float(r["stage1_probability"]),
                routed=int(r["routed"]),
                expert_probability=expert,
            )
            per_apk.append(row)
            grouped[(r["seed"], year, r["strategy"], arm)].append(row)
    yearly = []
    for (seed, year, strategy, arm), rows in sorted(grouped.items()):
        yearly.append(dict(
            stage1_seed=seed,
            year=year,
            strategy=strategy,
            arm=arm,
            **metric(rows),
        ))
    pooled_groups = defaultdict(list)
    for row in per_apk:
        pooled_groups[(row["stage1_seed"], row["strategy"], row["arm"])].append(row)
    pooled = [
        dict(stage1_seed=seed, strategy=strategy, arm=arm, **metric(rows))
        for (seed, strategy, arm), rows in sorted(pooled_groups.items())
    ]
    comparisons = []
    for strategy in STRATEGIES:
        for arm in ARMS[1:]:
            for year in ("2018", "2019", "2020", "2021", "pooled"):
                source = pooled if year == "pooled" else yearly
                b = {
                    r["stage1_seed"]: r for r in source
                    if r["strategy"] == strategy and r["arm"] == "B0_ORIGINAL"
                    and (year == "pooled" or r["year"] == year)
                }
                a = {
                    r["stage1_seed"]: r for r in source
                    if r["strategy"] == strategy and r["arm"] == arm
                    and (year == "pooled" or r["year"] == year)
                }
                if set(a) != set(b) or len(a) != 3:
                    raise RuntimeError(f"Missing paired set {strategy}/{arm}/{year}")
                comparisons.append(dict(
                    year=year,
                    strategy=strategy,
                    arm=arm,
                    mean_delta_BA_pp=float(np.mean([
                        (a[s]["BA"] - b[s]["BA"]) * 100 for s in a
                    ])),
                    mean_delta_fixed=float(np.mean([
                        a[s]["fixed"] - b[s]["fixed"] for s in a
                    ])),
                    mean_delta_broken=float(np.mean([
                        a[s]["broken"] - b[s]["broken"] for s in a
                    ])),
                    positive_BA_seeds=sum(a[s]["BA"] > b[s]["BA"] for s in a),
                ))
    write_csv(OUTPUT / "yearly_metrics.csv", yearly)
    write_csv(OUTPUT / "pooled_metrics.csv", pooled)
    write_csv(OUTPUT / "paired_comparison.csv", comparisons)
    write_csv(OUTPUT / "per_apk_predictions.csv", per_apk)
    decision = dict(
        status="BENIGN_HOST_ROLE_CONTRASTIVE_DEVELOPMENT_COMPLETE",
        formal_model_changed=False,
        datasets="2017-train through 2017-2020-train; 2018-2021 development validation",
        independent_test=False,
        main_comparison="H1 vs B0 and H2 on the same Manifest-RF route",
        warning="Within-role compactness alone cannot establish benign class evidence.",
    )
    with (OUTPUT / "final_decision.json").open("x", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
        f.write("\n")
    print("BHRC_EVALUATION_DONE", len(yearly), len(per_apk), flush=True)


if __name__ == "__main__":
    main()
