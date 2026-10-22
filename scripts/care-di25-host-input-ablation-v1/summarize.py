#!/usr/bin/env python3
"""Summarize capacity-matched Host-only correction-input ablations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from masked_host_model import ARMS, BLOCKS, correction_input_mask

ROOT = Path("artifacts/care-di25-host-input-ablation-v1")
LOCKED_M0 = {
    "BA": 0.9111058897243107,
    "AUC": 0.9410213032581454,
    "F1": 0.9086229086229086,
    "fixed": 87,
    "broken": 14,
    "net": 73,
    "parameter_count": 33388,
}


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metric(y, p, base, route):
    pred, base_pred = p >= .5, base >= .5
    fixed = int(np.sum(route & (base_pred != y) & (pred == y)))
    broken = int(np.sum(route & (base_pred == y) & (pred != y)))
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    return {"BA": float(balanced_accuracy_score(y, pred)), "AUC": float(roc_auc_score(y, p)),
            "F1": float(f1_score(y, pred, zero_division=0)), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "fixed": fixed, "broken": broken, "net": fixed-broken}


def exact_p(n01, n10):
    n = n01+n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0*sum(math.comb(n, i) for i in range(k+1))/(2.0**n))


def bootstrap(y, full, candidate, iterations=10000, seed=20260921):
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("BA", "AUC", "F1")}
    n = len(y)
    for _ in range(iterations):
        take = rng.integers(0, n, n)
        yt, pf, pc = y[take], full[take], candidate[take]
        if len(np.unique(yt)) < 2: continue
        for name, fn in (
            ("BA", lambda yy, pp: balanced_accuracy_score(yy, pp >= .5)),
            ("AUC", roc_auc_score),
            ("F1", lambda yy, pp: f1_score(yy, pp >= .5, zero_division=0)),
        ):
            values[name].append(float(fn(yt, pf)-fn(yt, pc)))
    out = {}
    for name, vals in values.items():
        a = np.asarray(vals, float)
        out[f"full_minus_arm_{name}"] = float(np.mean(a))
        out[f"full_minus_arm_{name}_ci_low"] = float(np.quantile(a, .025))
        out[f"full_minus_arm_{name}_ci_high"] = float(np.quantile(a, .975))
    return out


def load_arm(arm):
    rows = read(ROOT / "arms" / arm / "source_predictions.csv")
    columns = [key for key in rows[0] if key.startswith("DI25__Host__Anomaly__") and key.endswith("__Bounded__prob")]
    if len(columns) != 1:
        raise RuntimeError(f"{arm}: expected one final method column, got {columns}")
    ids = [row["sha256"].strip().lower() for row in rows]
    if len(rows) != 799 or len(set(ids)) != 799:
        raise RuntimeError(f"{arm}: cohort integrity failure")
    parameter_values = {int(float(row["parameter_n"])) for row in read(ROOT / "arms" / arm / "fit_audit.csv")}
    if len(parameter_values) != 1:
        raise RuntimeError(f"{arm}: inconsistent parameter counts {parameter_values}")
    return rows, columns[0], parameter_values.pop()


def main():
    loaded = {arm: load_arm(arm) for arm in ARMS}
    canonical_ids = [row["sha256"].strip().lower() for row in loaded["FULL"][0]]
    for arm, (rows, _, params) in loaded.items():
        if [row["sha256"].strip().lower() for row in rows] != canonical_ids:
            raise RuntimeError(f"{arm}: SHA order differs from FULL")
        if params != loaded["FULL"][2]:
            raise RuntimeError(f"{arm}: parameter count {params} differs from FULL {loaded['FULL'][2]}")
    base_rows = loaded["FULL"][0]
    y = np.asarray([int(float(row["label"])) for row in base_rows], int)
    base = np.asarray([float(row["Global24__prob"]) for row in base_rows], float)
    route = np.asarray([int(float(row["A25__routed"])) == 1 for row in base_rows], bool)
    fold = np.asarray([int(float(row["fold"])) for row in base_rows], int)
    if int(np.sum(route)) != 200:
        raise RuntimeError("Frozen DI25 route is not 200/799")

    probs, aggregate, per_fold = {}, [], []
    for arm, (rows, column, params) in loaded.items():
        p = np.asarray([float(row[column]) for row in rows], float); probs[arm] = p
        aggregate.append({"arm": arm, "dropped_blocks": "+".join(ARMS[arm]) or "none",
                          "active_dimensions": int(correction_input_mask(arm).sum()),
                          "parameter_count": params, **metric(y, p, base, route)})
        for f in sorted(set(fold)):
            take = fold == f
            per_fold.append({"arm": arm, "fold": f, "n": int(np.sum(take)),
                             **metric(y[take], p[take], base[take], route[take])})

    full_correct = (probs["FULL"] >= .5) == y
    paired = []
    full_metric = metric(y, probs["FULL"], base, route)
    reproduction = {
        "expected": LOCKED_M0,
        "observed": {**full_metric, "parameter_count": loaded["FULL"][2]},
        "metric_tolerance": 5e-7,
    }
    reproduction["passed"] = bool(
        all(abs(float(full_metric[key])-float(LOCKED_M0[key])) <= 5e-7 for key in ("BA", "AUC", "F1"))
        and all(int(full_metric[key]) == int(LOCKED_M0[key]) for key in ("fixed", "broken", "net"))
        and loaded["FULL"][2] == LOCKED_M0["parameter_count"]
    )
    historical = Path("artifacts/care-di25-hostego-v1/source/M0_host_anomaly_bounded/source_predictions.csv")
    reproduction["historical_prediction_file"] = str(historical) if historical.is_file() else None
    if historical.is_file():
        old_rows = read(historical)
        old_by_id = {row["sha256"].strip().lower(): row for row in old_rows}
        old_cols = [key for key in old_rows[0] if "Host" in key and "HostEgo" not in key and key.endswith("__prob")]
        if len(old_by_id) == 799 and len(old_cols) == 1 and set(old_by_id) == set(canonical_ids):
            old = np.asarray([float(old_by_id[sha][old_cols[0]]) for sha in canonical_ids])
            reproduction["historical_probability_column"] = old_cols[0]
            reproduction["max_abs_probability_diff"] = float(np.max(np.abs(old-probs["FULL"])))
            reproduction["prediction_difference_count"] = int(np.sum((old >= .5) != (probs["FULL"] >= .5)))
        else:
            reproduction["historical_probability_comparison"] = "unavailable_schema_or_cohort_mismatch"
    for index, arm in enumerate(ARMS):
        if arm == "FULL": continue
        correct = (probs[arm] >= .5) == y
        n01 = int(np.sum(~full_correct & correct))
        n10 = int(np.sum(full_correct & ~correct))
        arm_metric = metric(y, probs[arm], base, route)
        boot = bootstrap(y, probs["FULL"], probs[arm], seed=20260921+index)
        paired.append({
            "comparison": f"FULL vs {arm}", "arm": arm,
            "n01_FULL_wrong_arm_correct": n01, "n10_FULL_correct_arm_wrong": n10,
            "exact_mcnemar_p": exact_p(n01, n10),
            "delta_net_FULL_minus_arm": full_metric["net"]-arm_metric["net"], **boot,
        })

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_reproduction_audit.json").write_text(
        json.dumps(reproduction, indent=2)+"\n", encoding="utf-8")
    if not reproduction["passed"]:
        raise RuntimeError("BLOCKED_BY_REPRODUCTION: FULL does not reproduce locked Host-only M0; see full_reproduction_audit.json")
    write(out / "aggregate_metrics.csv", aggregate)
    write(out / "per_fold_metrics.csv", per_fold)
    write(out / "paired_full_vs_ablation.csv", paired)
    contract = {"n": 799, "routed": 200, "router": "frozen DI25", "host_only": True,
                "ego_used": False, "max_delta": 2.25, "capacity_matched": True,
                "parameter_count": loaded["FULL"][2], "bootstrap_iterations": 10000,
                "future_rows_loaded": False}
    (out / "experiment_contract.json").write_text(json.dumps(contract, indent=2)+"\n", encoding="utf-8")

    by_arm = {row["arm"]: row for row in aggregate}
    by_pair = {row["arm"]: row for row in paired}
    lines = [
        "CARE-Droid Host-only correction-input block ablation v1", "",
        "Strict 2017 five-fold OOF only. Frozen DI25 route (200/799), fixed max_delta=2.25.",
        "All arms retain the same 113-wide head and identical parameter count; removed blocks are fixed to zero.",
        "No Ego feature/cache is consumed by the model. No 2018-2021 row is loaded.", "",
    ]
    for arm in ARMS:
        m = by_arm[arm]
        lines.append(f"{arm}: active={m['active_dimensions']}/113 BA={m['BA']:.6f} AUC={m['AUC']:.6f} F1={m['F1']:.6f} fixed={m['fixed']} broken={m['broken']} net={m['net']}")
        if arm != "FULL":
            q = by_pair[arm]
            lines.append(f"  FULL-arm: dBA={q['full_minus_arm_BA']:+.6f} CI=[{q['full_minus_arm_BA_ci_low']:+.6f},{q['full_minus_arm_BA_ci_high']:+.6f}] dAUC={q['full_minus_arm_AUC']:+.6f} CI=[{q['full_minus_arm_AUC_ci_low']:+.6f},{q['full_minus_arm_AUC_ci_high']:+.6f}] McNemar p={q['exact_mcnemar_p']:.6f}")
    lines += ["", "Interpretation rule:",
              "A positive FULL-minus-drop estimate means the removed block helped the complete Host-only model.",
              "A confidence interval crossing zero is inconclusive, not proof that the block is useless.",
              "LOO effects include interactions among blocks; they are not additive causal contributions."]
    (out / "REPORT.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("HOST_INPUT_ABLATION_SUMMARY_PASS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
