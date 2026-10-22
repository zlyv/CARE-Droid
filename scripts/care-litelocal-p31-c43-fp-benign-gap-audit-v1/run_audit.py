#!/usr/bin/env python3
from __future__ import annotations

import csv, gzip, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from feature_core import FEATURE_GROUPS, centroid_score, extract_features

ROOT = Path("artifacts/care-litelocal-p31-c43-fp-benign-gap-audit-v1")


def norm(value): return str(value or "").strip().casefold()


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows: raise RuntimeError("refusing empty output")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def auc_ci(positive, negative, seed, iterations=5000):
    positive = np.asarray(positive, float); negative = np.asarray(negative, float)
    estimate = float(roc_auc_score(np.r_[np.ones(len(positive)), np.zeros(len(negative))],
                                   np.r_[positive, negative]))
    rng = np.random.default_rng(seed); result = np.empty(iterations)
    for index in range(iterations):
        p = positive[rng.integers(0, len(positive), len(positive))]
        n = negative[rng.integers(0, len(negative), len(negative))]
        result[index] = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
    return estimate, float(np.quantile(result, .025)), float(np.quantile(result, .975))


def delta_ci(positive_a, negative_a, positive_b, negative_b, seed, iterations=5000):
    pa, na, pb, nb = map(lambda x: np.asarray(x, float), (positive_a, negative_a, positive_b, negative_b))
    rng = np.random.default_rng(seed); values = np.empty(iterations)
    for index in range(iterations):
        pi = rng.integers(0, len(pa), len(pa)); ni = rng.integers(0, len(na), len(na))
        label = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
        values[index] = (roc_auc_score(label, np.r_[pa[pi], na[ni]]) -
                         roc_auc_score(label, np.r_[pb[pi], nb[ni]]))
    point = roc_auc_score(np.r_[np.ones(len(pa)), np.zeros(len(na))], np.r_[pa, na]) - roc_auc_score(np.r_[np.ones(len(pb)), np.zeros(len(nb))], np.r_[pb, nb])
    return float(point), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def main():
    resolved = json.loads((ROOT / "resolved_inputs.json").read_text(encoding="utf-8"))
    rows = read(resolved["p31"]); remap = np.asarray(resolved["payload_to_model_role"], int)
    payloads = {}
    with gzip.open(resolved["endpoint"], "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); payloads[norm(row["sha256"])] = row["payload"]
    extracted, audits = {}, {}
    for index, row in enumerate(rows, 1):
        sha = norm(row["sha256"]); extracted[sha], audits[sha] = extract_features(payloads[sha], remap)
        if index % 200 == 0 or index == len(rows): print(f"FP benign-gap feature audit {index}/{len(rows)}", flush=True)

    ids = [norm(row["sha256"]) for row in rows]
    y = np.asarray([int(float(row["label"])) for row in rows])
    fold = np.asarray([int(float(row["fold"])) for row in rows])
    probability = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in rows])
    prediction = (probability >= .5).astype(int)
    route = np.asarray([int(float(row["P31_DI25__routed"])) for row in rows], bool)
    correct = prediction == y
    cohort = np.asarray([("FP" if yi == 0 else "TP") if pi == 1 else ("TN" if yi == 0 else "FN")
                         for yi, pi in zip(y, prediction)])
    semantic_names = list(FEATURE_GROUPS["ALL_SEMANTIC"])
    x = np.asarray([[extracted[sha][name] for name in semantic_names] for sha in ids], float)
    size_names = ["node_count", "edge_count", "endpoint_count", "host_count"]
    size = np.log1p(np.asarray([[audits[sha][name] for name in size_names] for sha in ids], float))
    scores = {group: np.full(len(rows), np.nan) for group in (*FEATURE_GROUPS, "SIZE_ONLY")}
    for outer in range(1, 6):
        train = (fold != outer) & route & correct
        validation = (fold == outer) & route
        if sum(train & (y == 0)) == 0 or sum(train & (y == 1)) == 0:
            raise RuntimeError("BLOCKED_BY_CONTRACT: class-symmetric controls absent")
        for group, names in FEATURE_GROUPS.items():
            columns = [semantic_names.index(name) for name in names]
            scores[group][validation] = centroid_score(x[train][:, columns], y[train], x[validation][:, columns])
        scores["SIZE_ONLY"][validation] = centroid_score(size[train], y[train], size[validation])

    detail = []
    for index in np.flatnonzero(route):
        row = {"sha256": ids[index], "fold": int(fold[index]), "label": int(y[index]),
               "P31_probability": float(probability[index]), "P31_prediction": int(prediction[index]),
               "cohort": cohort[index], "P31_correct": int(correct[index])}
        row.update(extracted[ids[index]])
        row.update({"audit_" + key: value for key, value in audits[ids[index]].items()})
        row.update({group + "__benign_direction_score": float(value[index]) for group, value in scores.items()})
        detail.append(row)

    fp = route & (cohort == "FP"); tp = route & (cohort == "TP"); tn = route & (cohort == "TN")
    comparisons = []
    for group, values in scores.items():
        for name, positive, negative in (("FP_vs_TP", fp, tp), ("TN_vs_TP", tn, tp), ("FP_vs_TN", fp, tn)):
            estimate, low, high = auc_ci(values[positive], values[negative], 20260928 + len(comparisons))
            comparisons.append({"comparison": name, "feature_group": group, "positive_n": int(positive.sum()),
                                "negative_n": int(negative.sum()), "AUC": estimate,
                                "bootstrap_CI_low": low, "bootstrap_CI_high": high})

    feature_rows = []
    for column, name in enumerate(semantic_names):
        estimate, low, high = auc_ci(x[fp, column], x[tp, column], 20261028 + column)
        signed = estimate
        feature_rows.append({"feature": name, "FP_mean": float(x[fp, column].mean()),
                             "TP_mean": float(x[tp, column].mean()),
                             "TN_mean": float(x[tn, column].mean()),
                             "FP_minus_TP": float(x[fp, column].mean() - x[tp, column].mean()),
                             "signed_AUC_FP_high": signed, "unsigned_separability_AUC": max(signed, 1-signed),
                             "bootstrap_CI_low_signed": low, "bootstrap_CI_high_signed": high})

    correlations = []
    for group in FEATURE_GROUPS:
        for column, size_name in enumerate(size_names):
            rho = float(spearmanr(scores[group][route], size[route, column]).statistic)
            if not np.isfinite(rho): rho = 0.0
            correlations.append({"feature_group": group, "audit_size": size_name,
                                 "spearman_rho": rho, "abs_rho": abs(rho)})
    all_auc = next(row["AUC"] for row in comparisons if row["comparison"] == "FP_vs_TP" and row["feature_group"] == "ALL_SEMANTIC")
    all_low = next(row["bootstrap_CI_low"] for row in comparisons if row["comparison"] == "FP_vs_TP" and row["feature_group"] == "ALL_SEMANTIC")
    size_auc = next(row["AUC"] for row in comparisons if row["comparison"] == "FP_vs_TP" and row["feature_group"] == "SIZE_ONLY")
    delta, delta_low, delta_high = delta_ci(scores["ALL_SEMANTIC"][fp], scores["ALL_SEMANTIC"][tp],
                                             scores["SIZE_ONLY"][fp], scores["SIZE_ONLY"][tp], 20261128)
    max_rho = max(row["abs_rho"] for row in correlations if row["feature_group"] == "ALL_SEMANTIC")
    if all_low > .5 and delta_low > 0 and max_rho < .8:
        decision = "BENIGN_COUNTEREVIDENCE_PRESENT_BEYOND_SIZE"
    elif all_low > .5:
        decision = "PARTIAL_BENIGN_COUNTEREVIDENCE"
    else:
        decision = "BENIGN_EXPLANATION_NOT_READABLE"

    write(ROOT / "routed_apk_features_and_scores.csv", detail)
    write(ROOT / "group_separability.csv", comparisons)
    write(ROOT / "individual_feature_audit.csv", feature_rows)
    write(ROOT / "size_association.csv", correlations)
    report = f"""CARE-Droid P31-C43 FP benign-explanation gap audit v1

Read-only strict-2017 diagnostic from frozen endpoint tensors and P31 OOF predictions.
No neural training/inference, route change, future row, Ego tensor, or APK reanalysis.
Class-direction references use only routed-correct APKs from the other outer folds.
All semantic model inputs are ratios/normalized organization measures; raw counts are audit-only.

Cohorts: FP=20 TP=71 TN=83 FN=26.
Primary ALL_SEMANTIC benign-direction FP-vs-TP AUC={all_auc:.6f} CI=[{all_low:.6f},{next(row['bootstrap_CI_high'] for row in comparisons if row['comparison']=='FP_vs_TP' and row['feature_group']=='ALL_SEMANTIC'):.6f}]
SIZE_ONLY FP-vs-TP AUC={size_auc:.6f}
Semantic-minus-size DeltaAUC={delta:+.6f} CI=[{delta_low:+.6f},{delta_high:+.6f}]
ALL_SEMANTIC max|rho with audit size|={max_rho:.6f}

DECISION={decision}
If benign counterevidence is readable, the next experiment should expose that direction to C43 without changing localization.
If it is not readable, redesign Host evidence before changing the gate or correction head.
This source diagnostic does not authorize a future experiment or deployable rule.
"""
    (ROOT / "REPORT.txt").write_text(report, encoding="utf-8")
    print("P31_C43_FP_BENIGN_GAP_AUDIT_DONE")
    print(report)


if __name__ == "__main__": main()
