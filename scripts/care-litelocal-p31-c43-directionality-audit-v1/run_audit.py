#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from direction_core import (REPRESENTATIONS, aggregate_host_values,
                            build_references, class_distances,
                            equal_apk_scaler, transform_bags)


ROOT = Path("artifacts/care-litelocal-p31-c43-directionality-audit-v1")
P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
FAIL = Path("artifacts/care-litelocal-p31-c43-failure-audit-v1")
CACHE = Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")


def norm(value):
    return str(value or "").strip().casefold()


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty output: " + str(path))
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def percentile(value, control):
    control = np.asarray([entry for entry in control if np.isfinite(entry)], float)
    if not np.isfinite(value) or len(control) == 0:
        return np.nan
    return float((np.sum(control < value) + .5 * np.sum(control == value)) / len(control))


def bootstrap_auc(positive, negative, iterations=3000, seed=20260928):
    positive = np.asarray(positive, float); positive = positive[np.isfinite(positive)]
    negative = np.asarray(negative, float); negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan, np.nan, np.nan
    label = np.r_[np.ones(len(positive), int), np.zeros(len(negative), int)]
    value = np.r_[positive, negative]
    estimate = float(roc_auc_score(label, value))
    rng = np.random.default_rng(seed)
    result = np.empty(iterations, float)
    for index in range(iterations):
        p = positive[rng.integers(0, len(positive), len(positive))]
        n = negative[rng.integers(0, len(negative), len(negative))]
        result[index] = roc_auc_score(
            np.r_[np.ones(len(p), int), np.zeros(len(n), int)], np.r_[p, n])
    return estimate, float(np.quantile(result, .025)), float(np.quantile(result, .975))


def load_cache():
    with np.load(CACHE, allow_pickle=False) as saved:
        ids = [norm(value) for value in saved["sha256"].astype(str)]
        matrix = np.asarray(saved["host_values"], np.float32)
        offsets = np.asarray(saved["host_offsets"], int)
        real = np.asarray(saved["real_host_count"], int)
        empty = (np.asarray(saved["host_was_empty"], int)
                 if "host_was_empty" in saved.files else (real == 0).astype(int))
    core, roles, relations = [], [], []
    for index in range(len(ids)):
        begin = int(offsets[index]); stored_end = int(offsets[index + 1])
        count = int(real[index])
        # True empty APKs retain their deterministic placeholder for the neural
        # model but are excluded from semantic direction estimation.
        end = min(begin + count, stored_end)
        value = matrix[begin:end]
        core.append(value[:, :43])
        roles.append(value[:, 43:74])
        relations.append(value[:, 74:105])
    return ids, core, roles, relations, real, empty


def representation_bags(core, relation, name):
    if name == "CORE43":
        return core
    if name == "RELATION31":
        return relation
    if name == "JOINT74":
        return [np.concatenate([left, right], axis=1)
                for left, right in zip(core, relation)]
    raise KeyError(name)


def cohort_name(label, prediction, correct, mechanism):
    if correct:
        return "CORRECT_PRED_BENIGN" if prediction == 0 else "CORRECT_PRED_MALWARE"
    return ("FN_" if label == 1 else "FP_") + mechanism


def main():
    reference = read(P31 / "preflight/p31_reference/source_predictions.csv")
    bounded = {norm(row["sha256"]): row for row in
               read(P31 / "arms/P31_C43_RQ/source_predictions.csv")}
    failure = {norm(row["sha256"]): row for row in
               read(FAIL / "routed_errors_mechanism.csv")}
    cache_ids, core, roles, relation, real_count, empty = load_cache()
    cache_index = {value: index for index, value in enumerate(cache_ids)}
    ids = [norm(row["sha256"]) for row in reference]
    order = np.asarray([cache_index[value] for value in ids], int)
    core = [core[index] for index in order]
    roles = [roles[index] for index in order]
    relation = [relation[index] for index in order]
    real_count = real_count[order]; empty = empty[order]
    labels = np.asarray([int(float(row["label"])) for row in reference])
    folds = np.asarray([int(float(row["fold"])) for row in reference])
    base = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in reference])
    route = np.asarray([int(float(row["P31_DI25__routed"])) for row in reference], bool)
    prediction = (base >= .5).astype(int)
    correct = prediction == labels
    if int(route.sum()) != 200 or int(np.sum(route & ~correct)) != 46:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed accounting changed")

    apk_rows = {index: {
        "sha256": ids[index], "fold": int(folds[index]), "label": int(labels[index]),
        "P31_prediction": int(prediction[index]), "P31_probability": float(base[index]),
        "P31_correct": int(correct[index]), "real_host_count": int(real_count[index]),
        "host_was_empty": int(empty[index]),
    } for index in np.flatnonzero(route)}
    for index, row in apk_rows.items():
        mechanism = "P31_CORRECT"
        if not correct[index]:
            value = failure[ids[index]]
            mechanism = value.get("primary_failure_type") or value.get("mechanism")
            if not mechanism:
                raise RuntimeError("failure mechanism column missing")
        row["mechanism"] = mechanism
        row["cohort"] = cohort_name(labels[index], prediction[index], correct[index], mechanism)
    observed_mechanisms = Counter(
        row["mechanism"] for row in apk_rows.values() if not row["P31_correct"])
    expected_mechanisms = Counter({
        "WRONG_OR_ZERO_DIRECTION": 32,
        "RIGHT_DIRECTION_TOO_WEAK": 6,
        "FIXED_BOUNDED": 8,
    })
    if observed_mechanisms != expected_mechanisms:
        raise RuntimeError("BLOCKED_BY_CONTRACT: failure cohorts changed: " +
                           repr(observed_mechanisms))

    host_rows = []
    scaler_audit = []
    for fold in range(1, 6):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero((folds == fold) & route)
        for representation in REPRESENTATIONS:
            raw = representation_bags(core, relation, representation)
            mean, std = equal_apk_scaler([raw[index] for index in train])
            scaled_train = transform_bags([raw[index] for index in train], mean, std)
            scaled_validation = transform_bags([raw[index] for index in validation], mean, std)
            train_roles = [roles[index] for index in train]
            references, global_references = build_references(
                scaled_train, train_roles, labels[train])
            scaler_audit.append({
                "fold": fold, "representation": representation,
                "dimension": int(len(mean)), "zero_std_n": int(np.sum(std == 1.0)),
                "benign_global_reference_n": int(len(global_references[0])),
                "malware_global_reference_n": int(len(global_references[1])),
                "benign_role_reference_n": int(sum(len(references[(0, r)]) for r in range(31))),
                "malware_role_reference_n": int(sum(len(references[(1, r)]) for r in range(31))),
            })
            for local, index in enumerate(validation):
                feature = scaled_validation[local]
                d_benign, d_malware, support_benign, support_malware = class_distances(
                    feature, roles[index], references, global_references)
                malware_margin = d_benign - d_malware
                sign_against_p31 = 1.0 if prediction[index] == 0 else -1.0
                counter = sign_against_p31 * malware_margin
                truth = (1.0 if labels[index] == 1 else -1.0) * malware_margin
                benign_summary = aggregate_host_values(-d_benign)
                malware_summary = aggregate_host_values(-d_malware)
                margin_summary = aggregate_host_values(malware_margin)
                counter_summary = aggregate_host_values(counter)
                truth_summary = aggregate_host_values(truth)
                target = apk_rows[index]
                prefix = representation + "__"
                target[prefix + "benign_distance_mean"] = float(np.mean(d_benign)) if len(d_benign) else np.nan
                target[prefix + "malware_distance_mean"] = float(np.mean(d_malware)) if len(d_malware) else np.nan
                target[prefix + "benign_closeness_max"] = benign_summary["max"]
                target[prefix + "malware_closeness_max"] = malware_summary["max"]
                for name, value in margin_summary.items():
                    target[prefix + "malware_margin_" + name] = value
                for name, value in counter_summary.items():
                    target[prefix + "counterevidence_" + name] = value
                for name, value in truth_summary.items():
                    target[prefix + "truth_support_" + name] = value
                target[prefix + "benign_reference_support_min"] = int(np.min(support_benign)) if len(support_benign) else 0
                target[prefix + "malware_reference_support_min"] = int(np.min(support_malware)) if len(support_malware) else 0
                for host_index in range(len(feature)):
                    host_rows.append({
                        "sha256": ids[index], "fold": fold,
                        "label": int(labels[index]), "P31_prediction": int(prediction[index]),
                        "cohort": target["cohort"], "representation": representation,
                        "host_index": host_index,
                        "d_benign": float(d_benign[host_index]),
                        "d_malware": float(d_malware[host_index]),
                        "malware_margin": float(malware_margin[host_index]),
                        "counterevidence_margin": float(counter[host_index]),
                        "truth_support_margin": float(truth[host_index]),
                        "primary_role": int(np.argmax(roles[index][host_index]))
                        if roles[index][host_index].sum() > 0 else -1,
                    })
        print(f"directionality fold {fold}/5 done", flush=True)

    rows = [apk_rows[index] for index in sorted(apk_rows)]
    rank_features = []
    for representation in REPRESENTATIONS:
        for statistic in ("max", "top3_mean", "mean", "fraction_positive"):
            rank_features.append(f"{representation}__counterevidence_{statistic}")
    for row in rows:
        controls = [candidate for candidate in rows
                    if candidate["fold"] == row["fold"]
                    and candidate["P31_prediction"] == row["P31_prediction"]
                    and candidate["P31_correct"] == 1]
        for feature in rank_features:
            row[feature + "__rank_vs_matched_correct"] = percentile(
                float(row[feature]), [float(candidate[feature]) for candidate in controls])

    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row["cohort"]].append(row)
    cohort_summary = []
    summary_features = []
    for representation in REPRESENTATIONS:
        summary_features.extend((
            f"{representation}__benign_distance_mean",
            f"{representation}__malware_distance_mean",
            f"{representation}__counterevidence_max",
            f"{representation}__counterevidence_top3_mean",
            f"{representation}__counterevidence_mean",
            f"{representation}__counterevidence_fraction_positive",
            f"{representation}__counterevidence_top3_mean__rank_vs_matched_correct",
        ))
    for cohort, values in sorted(cohorts.items()):
        output = {"cohort": cohort, "n": len(values)}
        for feature in summary_features:
            vector = np.asarray([float(row[feature]) for row in values], float)
            vector = vector[np.isfinite(vector)]
            output[feature + "__mean"] = float(np.mean(vector)) if len(vector) else np.nan
            output[feature + "__median"] = float(np.median(vector)) if len(vector) else np.nan
        cohort_summary.append(output)

    separability = []
    comparisons = {
        "ALL_ROUTED_ERROR_vs_CORRECT": (
            [row for row in rows if not row["P31_correct"]],
            [row for row in rows if row["P31_correct"]]),
        "FN_vs_CORRECT_PRED_BENIGN": (
            [row for row in rows if not row["P31_correct"] and row["label"] == 1],
            [row for row in rows if row["P31_correct"] and row["P31_prediction"] == 0]),
        "FP_vs_CORRECT_PRED_MALWARE": (
            [row for row in rows if not row["P31_correct"] and row["label"] == 0],
            [row for row in rows if row["P31_correct"] and row["P31_prediction"] == 1]),
        "FIXED_vs_WRONG_DIRECTION": (
            [row for row in rows if row["mechanism"] == "FIXED_BOUNDED"],
            [row for row in rows if row["mechanism"] == "WRONG_OR_ZERO_DIRECTION"]),
    }
    for comparison, (positive_rows, negative_rows) in comparisons.items():
        for representation in REPRESENTATIONS:
            for statistic in ("max", "top3_mean", "mean", "fraction_positive"):
                feature = f"{representation}__counterevidence_{statistic}"
                estimate, low, high = bootstrap_auc(
                    [float(row[feature]) for row in positive_rows],
                    [float(row[feature]) for row in negative_rows],
                    seed=20260928 + len(separability))
                separability.append({
                    "comparison": comparison, "representation": representation,
                    "statistic": statistic, "positive_n": len(positive_rows),
                    "negative_n": len(negative_rows), "AUC": estimate,
                    "bootstrap_CI_low": low, "bootstrap_CI_high": high,
                })

    primary = next(row for row in separability
                   if row["comparison"] == "ALL_ROUTED_ERROR_vs_CORRECT"
                   and row["representation"] == "JOINT74"
                   and row["statistic"] == "top3_mean")
    fn_primary = next(row for row in separability
                      if row["comparison"] == "FN_vs_CORRECT_PRED_BENIGN"
                      and row["representation"] == "JOINT74"
                      and row["statistic"] == "top3_mean")
    fp_primary = next(row for row in separability
                      if row["comparison"] == "FP_vs_CORRECT_PRED_MALWARE"
                      and row["representation"] == "JOINT74"
                      and row["statistic"] == "top3_mean")
    if (primary["bootstrap_CI_low"] > .5 and fn_primary["AUC"] > .6 and
            fp_primary["AUC"] > .6):
        decision = "CLASS_CONDITIONAL_DIRECTION_SIGNAL_SUPPORTED"
    elif primary["AUC"] > .6:
        decision = "PARTIAL_OR_UNCERTAIN_DIRECTION_SIGNAL"
    else:
        decision = "DIRECTION_SIGNAL_WEAK_OR_ABSENT"

    write(ROOT / "apk_directionality.csv", rows)
    write(ROOT / "routed_host_directionality.csv", host_rows)
    write(ROOT / "cohort_summary.csv", cohort_summary)
    write(ROOT / "separability.csv", separability)
    write(ROOT / "fold_background_audit.csv", scaler_audit)
    mechanism_counts = Counter(row["mechanism"] for row in rows if not row["P31_correct"])
    hard = {
        "source_rows": 799, "routed_rows": 200, "routed_errors": 46,
        "mechanism_counts": dict(mechanism_counts),
        "reference_protocol": "outer-train-only, class-symmetric, primary-role-conditioned, one centroid per APK-role",
        "distance": "mean RMS distance to five nearest reference APK-role centroids",
        "primary_feature": "JOINT74 counterevidence top3 mean",
        "primary_AUC": primary["AUC"],
        "primary_bootstrap_CI": [primary["bootstrap_CI_low"], primary["bootstrap_CI_high"]],
        "decision": decision,
        "training_invoked": False, "neural_inference_invoked": False,
        "route_modified": False, "future_rows_loaded": 0, "ego_consumed": False,
    }
    (ROOT / "hard_audit.json").write_text(json.dumps(hard, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CARE-Droid P31-C43 class-conditional Host directionality audit v1", "",
        "Read-only strict-2017 OOF diagnostic. No neural training/inference, route change, future row, or Ego.",
        "Every validation Host is compared only with outer-training benign and malware APK-role references.",
        "Positive counterevidence means the Host representation opposes the P31 predicted class.", "",
        f"routed=200 errors=46 correct_controls=154 mechanisms={dict(mechanism_counts)}", "",
        "Primary JOINT74 Top3 counterevidence:",
        f"all errors vs routed-correct AUC={primary['AUC']:.6f} CI=[{primary['bootstrap_CI_low']:.6f},{primary['bootstrap_CI_high']:.6f}]",
        f"FN vs correct-predicted-benign AUC={fn_primary['AUC']:.6f} CI=[{fn_primary['bootstrap_CI_low']:.6f},{fn_primary['bootstrap_CI_high']:.6f}]",
        f"FP vs correct-predicted-malware AUC={fp_primary['AUC']:.6f} CI=[{fp_primary['bootstrap_CI_low']:.6f},{fp_primary['bootstrap_CI_high']:.6f}]", "",
        "DECISION=" + decision,
        "This audit tests whether existing frozen Host features contain class direction; it does not authorize a new head, gate, threshold, or future experiment.",
        "Inspect CORE43, RELATION31, and JOINT74 rows in separability.csv before changing the representation.",
    ]
    report = "\n".join(lines) + "\n"
    (ROOT / "REPORT.txt").write_text(report, encoding="utf-8")
    print("P31_C43_DIRECTIONALITY_AUDIT_DONE")
    print(report)


if __name__ == "__main__":
    main()
