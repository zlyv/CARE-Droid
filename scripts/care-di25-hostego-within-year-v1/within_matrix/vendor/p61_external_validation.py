#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_compact_quantity_shared_caller_2x3_2017_eval as exp


OLD_FEATURES = ROOT / "artifacts/p1-features/features.csv"
OLD_SCHEMA = ROOT / "artifacts/p1-features/feature_schema.json"
OLD_CACHE = (
    ROOT
    / "artifacts/p25-endpoint-cache-full-cap256"
    / "endpoint_centered_tensor_cache.jsonl.gz"
)
NEW_FEATURES = ROOT / "artifacts/p54-new-compact-features/features.csv"
NEW_SCHEMA = ROOT / "artifacts/p54-new-compact-features/feature_schema.json"
NEW_CACHE = (
    ROOT
    / "artifacts/p54-new-endpoint-cache-full-cap256"
    / "endpoint_centered_tensor_cache.jsonl.gz"
    / "endpoint_centered_tensor_cache.jsonl.gz"
)
IDENTITY_AUDIT = (
    ROOT
    / "artifacts/p61-new-external-audit"
    / "stable_identity_overlap.csv"
)
OUTPUT = ROOT / "artifacts/p61-frozen-2017-new-external-validation"

YEARS = (2017, 2018, 2019, 2020, 2021)
MODEL_SEEDS = (0, 1, 2, 3, 4)
ITERATIONS = 1500
LEARNING_RATE = 0.05
L2 = 0.03
THRESHOLD = 0.5

VARIANTS = (
    "raw_count70_only",
    "raw_count70_plus_shared_caller",
    "top5_selective_shared_caller",
)
PROTOCOL_FIELD_ALIASES = {
    "sha_disjoint": (
        "eligible_sha_disjoint", "sha_disjoint", "is_sha_disjoint",
    ),
    "package_certificate_disjoint": (
        "eligible_lineage_disjoint", "eligible_package_certificate_disjoint",
        "lineage_disjoint", "package_certificate_disjoint",
    ),
    "package_disjoint": (
        "eligible_package_disjoint", "package_disjoint", "is_package_disjoint",
    ),
}
TOP5_ROLES = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)
TOP5_ROUTE_COLUMNS = tuple(
    f"role_node__{role}" for role in TOP5_ROLES
) + tuple(f"role_edge__{role}" for role in TOP5_ROLES)


def norm(value):
    return str(value).strip().casefold()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing to write empty CSV: " + str(path))
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def parse_year(row):
    value = row.get("metadata_year", row.get("year", ""))
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_label(row):
    value = norm(row.get("metadata_label", row.get("label", "")))
    if value in {"0", "0.0", "benign", "goodware", "false"}:
        return 0
    if value in {"1", "1.0", "malicious", "malware", "true"}:
        return 1
    raise RuntimeError("unknown label: " + repr(value))


def number(value):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if np.isfinite(output) else 0.0


def route_mask(rows, feature_by_sha):
    return np.asarray([
        any(
            number(feature_by_sha[norm(row["sha256"])].get(column, 0.0)) > 0.0
            for column in TOP5_ROUTE_COLUMNS
        )
        for row in rows
    ], dtype=bool)


def calculate_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= THRESHOLD).astype(np.int64)
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    tn = int(np.sum((labels == 0) & (prediction == 0)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
    accuracy = float(np.mean(prediction == labels))
    specificity = float(tn / (tn + fp)) if tn + fp else None
    fpr = float(fp / (tn + fp)) if tn + fp else None
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    precision = float(tp / (tp + fp)) if tp + fp else None
    f1 = (
        float(2.0 * precision * sensitivity / (precision + sensitivity))
        if precision is not None and sensitivity is not None and precision + sensitivity
        else None
    )
    if len(np.unique(labels)) == 2:
        values = exp.metrics(labels, probability, THRESHOLD)
        ba = float(values["balanced_accuracy"])
        auc = float(values["auc"])
    else:
        ba = None
        auc = None
    return {
        "accuracy": accuracy,
        "balanced_accuracy": ba,
        "auc": auc,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }, prediction


def mean_present(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return float(np.mean(values)) if values else None


def resolve_protocol_fields(identity_rows):
    if not identity_rows:
        raise RuntimeError("identity audit is empty")
    fields = set(identity_rows[0])
    resolved = {}
    for protocol, aliases in PROTOCOL_FIELD_ALIASES.items():
        matches = [field for field in aliases if field in fields]
        if len(matches) != 1:
            raise RuntimeError(
                f"cannot resolve one field for {protocol}; matches={matches}; "
                f"available={sorted(fields)}"
            )
        resolved[protocol] = matches[0]
    return resolved


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_features = read_csv(OLD_FEATURES)
    new_features = read_csv(NEW_FEATURES)
    identity_rows = read_csv(IDENTITY_AUDIT)
    protocols = resolve_protocol_fields(identity_rows)

    with OLD_SCHEMA.open("r", encoding="utf-8") as handle:
        old_schema = json.load(handle)
    with NEW_SCHEMA.open("r", encoding="utf-8") as handle:
        new_schema = json.load(handle)
    if old_schema.get("feature_fields") != new_schema.get("feature_fields"):
        raise RuntimeError("old/new feature schemas do not have identical feature_fields")

    train_rows = sorted([
        {"sha256": norm(row["sha256"]), "label": parse_label(row)}
        for row in old_features
        if parse_year(row) == 2017
    ], key=lambda row: row["sha256"])
    if len(train_rows) != 799:
        raise RuntimeError(f"expected 799 old source APKs, got {len(train_rows)}")

    identity_by_sha = {norm(row["sha256"]): row for row in identity_rows}
    test_rows = []
    for feature in new_features:
        sha = norm(feature["sha256"])
        if sha not in identity_by_sha:
            raise RuntimeError("identity audit missing SHA: " + sha)
        audit = identity_by_sha[sha]
        test_rows.append({
            "sha256": sha,
            "label": parse_label(feature),
            "year": parse_year(feature),
            **{
                protocol: int(float(audit[field]))
                for protocol, field in protocols.items()
            },
        })
    test_rows.sort(key=lambda row: (int(row["year"]), row["sha256"]))
    if len(test_rows) != 1089:
        raise RuntimeError(f"expected 1089 new APKs, got {len(test_rows)}")

    feature_by_sha = {
        norm(row["sha256"]): row
        for row in old_features + new_features
    }
    if len(feature_by_sha) != len(old_features) + len(new_features):
        raise RuntimeError("unexpected old/new SHA overlap in feature dictionaries")

    print("loading old endpoint cache", flush=True)
    old_cache = exp.read_cache(OLD_CACHE)
    print("loading new endpoint cache", flush=True)
    new_cache = exp.read_cache(NEW_CACHE)
    if set(old_cache) & set(new_cache):
        raise RuntimeError("old/new endpoint caches overlap by SHA")
    cache = dict(old_cache)
    cache.update(new_cache)

    print("building Compact matrices", flush=True)
    compact_modes, compact_names = exp.compact_quantity_matrices(
        train_rows, test_rows, feature_by_sha, old_schema
    )
    if len(compact_names) != 70:
        raise RuntimeError(f"expected 70 Compact fields, got {len(compact_names)}")
    raw_train, raw_test = compact_modes["raw_count70"]

    print("building shared-caller matrices", flush=True)
    hyper_train_raw, train_audit = exp.hypergraph_matrix(train_rows, cache)
    hyper_test_raw, test_audit = exp.hypergraph_matrix(test_rows, cache)
    if len(train_audit) != len(train_rows) or len(test_audit) != len(test_rows):
        raise RuntimeError("hypergraph audit row mismatch")

    hyper_train, hyper_test = exp.standardize_pair(hyper_train_raw, hyper_test_raw)
    shared_train = hyper_train[:, :exp.RELATION_DIMENSION]
    shared_test = hyper_test[:, :exp.RELATION_DIMENSION]
    train_route = route_mask(train_rows, feature_by_sha)
    test_route = route_mask(test_rows, feature_by_sha)
    x_raw = (raw_train, raw_test)
    x_full = (
        np.concatenate([raw_train, shared_train], axis=1),
        np.concatenate([raw_test, shared_test], axis=1),
    )
    y_train = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test_rows], dtype=np.int64)

    probability_by_variant = {variant: {} for variant in VARIANTS}
    for seed in MODEL_SEEDS:
        print("training seed", seed, flush=True)
        raw_probability = np.asarray(exp.predict_logistic(
            x_raw[0], y_train, x_raw[1], seed, ITERATIONS, LEARNING_RATE, L2
        ), dtype=float)
        full_probability = np.asarray(exp.predict_logistic(
            x_full[0], y_train, x_full[1], seed, ITERATIONS, LEARNING_RATE, L2
        ), dtype=float)
        selective_probability = np.where(
            test_route, full_probability, raw_probability
        )
        probability_by_variant["raw_count70_only"][seed] = raw_probability
        probability_by_variant["raw_count70_plus_shared_caller"][seed] = full_probability
        probability_by_variant["top5_selective_shared_caller"][seed] = selective_probability

    predictions = []
    per_seed = []
    for seed in MODEL_SEEDS:
        raw_probability = probability_by_variant["raw_count70_only"][seed]
        raw_prediction = (raw_probability >= THRESHOLD).astype(np.int64)
        for variant in VARIANTS:
            probability = probability_by_variant[variant][seed]
            prediction = (probability >= THRESHOLD).astype(np.int64)
            for index, row in enumerate(test_rows):
                predictions.append({
                    "year": row["year"],
                    "seed": seed,
                    "variant": variant,
                    "sha256": row["sha256"],
                    "label": row["label"],
                    "top5_routed": int(test_route[index]),
                    "sha_disjoint": row["sha_disjoint"],
                    "package_certificate_disjoint": row["package_certificate_disjoint"],
                    "package_disjoint": row["package_disjoint"],
                    "probability": float(probability[index]),
                    "prediction": int(prediction[index]),
                    "threshold": THRESHOLD,
                })
            for protocol in protocols:
                for year in YEARS:
                    mask = np.asarray([
                        row[protocol] == 1 and row["year"] == year
                        for row in test_rows
                    ], dtype=bool)
                    if not np.any(mask):
                        continue
                    metrics, selected_prediction = calculate_metrics(
                        y_test[mask], probability[mask]
                    )
                    selected_raw = raw_prediction[mask]
                    selected_labels = y_test[mask]
                    corrected = (selected_raw != selected_labels) & (
                        selected_prediction == selected_labels
                    )
                    regressed = (selected_raw == selected_labels) & (
                        selected_prediction != selected_labels
                    )
                    per_seed.append({
                        "protocol": protocol,
                        "year": year,
                        "variant": variant,
                        "seed": seed,
                        "n": int(mask.sum()),
                        "benign": int(np.sum(selected_labels == 0)),
                        "malware": int(np.sum(selected_labels == 1)),
                        "routed_apks": int(np.sum(test_route[mask])),
                        "route_coverage": float(np.mean(test_route[mask])),
                        "corrected_vs_raw": int(np.sum(corrected)),
                        "regressed_vs_raw": int(np.sum(regressed)),
                        "corrected_FP_vs_raw": int(np.sum(corrected & (selected_labels == 0))),
                        "corrected_FN_vs_raw": int(np.sum(corrected & (selected_labels == 1))),
                        "regressed_TN_vs_raw": int(np.sum(regressed & (selected_labels == 0))),
                        "regressed_TP_vs_raw": int(np.sum(regressed & (selected_labels == 1))),
                        **metrics,
                    })

    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["protocol"], row["year"], row["variant"])].append(row)
    per_year = []
    metric_fields = (
        "accuracy", "balanced_accuracy", "auc", "precision", "sensitivity",
        "specificity", "fpr", "f1", "TP", "TN", "FP", "FN",
        "corrected_vs_raw", "regressed_vs_raw", "corrected_FP_vs_raw",
        "corrected_FN_vs_raw", "regressed_TN_vs_raw", "regressed_TP_vs_raw",
    )
    for (protocol, year, variant), rows in sorted(grouped.items()):
        output = {
            "protocol": protocol,
            "year": year,
            "variant": variant,
            "seeds": len(rows),
            "n": rows[0]["n"],
            "benign": rows[0]["benign"],
            "malware": rows[0]["malware"],
            "routed_apks": rows[0]["routed_apks"],
            "route_coverage": rows[0]["route_coverage"],
        }
        for field in metric_fields:
            output[field] = mean_present(rows, field)
        per_year.append(output)

    summary = []
    for protocol in protocols:
        for variant in VARIANTS:
            rows = [
                row for row in per_year
                if row["protocol"] == protocol and row["variant"] == variant
            ]
            primary = [row for row in rows if row["year"] in (2018, 2019, 2020)]
            same_year = next(row for row in rows if row["year"] == 2017)
            benign_2021 = next(row for row in rows if row["year"] == 2021)
            ba = np.asarray([row["balanced_accuracy"] for row in primary], dtype=float)
            auc = np.asarray([row["auc"] for row in primary], dtype=float)
            summary.append({
                "protocol": protocol,
                "variant": variant,
                "primary_future_years": 3,
                "primary_future_apks": int(sum(row["n"] for row in primary)),
                "mean_BA_2018_2020": float(ba.mean()),
                "min_BA_2018_2020": float(ba.min()),
                "mean_AUC_2018_2020": float(auc.mean()),
                "min_AUC_2018_2020": float(auc.min()),
                "mean_ACC_2018_2020": float(np.mean([row["accuracy"] for row in primary])),
                "mean_route_coverage_2018_2020": float(np.mean([
                    row["route_coverage"] for row in primary
                ])),
                "total_mean_corrected_vs_raw_2018_2020": float(sum(
                    row["corrected_vs_raw"] for row in primary
                )),
                "total_mean_regressed_vs_raw_2018_2020": float(sum(
                    row["regressed_vs_raw"] for row in primary
                )),
                "external_2017_BA": same_year["balanced_accuracy"],
                "external_2017_AUC": same_year["auc"],
                "benign_2021_n": benign_2021["n"],
                "benign_2021_FP": benign_2021["FP"],
                "benign_2021_FPR": benign_2021["fpr"],
                "benign_2021_specificity": benign_2021["specificity"],
            })

    write_csv(OUTPUT / "predictions.csv", predictions)
    write_csv(OUTPUT / "per_seed.csv", per_seed)
    write_csv(OUTPUT / "per_year.csv", per_year)
    write_csv(OUTPUT / "summary.csv", summary)

    audit = {
        "phase": "P61_FROZEN_OLD2017_NEW_EXTERNAL_VALIDATION",
        "old_training_apks": len(train_rows),
        "old_training_benign": int(np.sum(y_train == 0)),
        "old_training_malware": int(np.sum(y_train == 1)),
        "new_external_apks": len(test_rows),
        "protocols": protocols,
        "model_seeds": list(MODEL_SEEDS),
        "threshold": THRESHOLD,
        "top5_roles": list(TOP5_ROLES),
        "training_rule": (
            "train raw-count and raw-count-plus-shared-caller models on all "
            "799 old-2017 APKs"
        ),
        "selective_inference_rule": (
            "use the full-model probability only when any TOP-5 role is "
            "present; otherwise retain the raw-count probability"
        ),
        "source_labels_used_for_training": True,
        "new_labels_used_for_training": False,
        "new_labels_used_for_routing": False,
        "new_labels_used_for_metrics_only": True,
        "primary_protocol": "package_certificate_disjoint",
        "primary_future_years": [2018, 2019, 2020],
        "year_2021_policy": "benign-only; report FP/FPR/specificity, not BA/AUC",
        "actual_new_cache": str(NEW_CACHE),
        "source_routed_apks": int(np.sum(train_route)),
        "source_route_coverage": float(np.mean(train_route)),
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nPRIMARY PACKAGE+CERTIFICATE-DISJOINT RESULTS")
    for row in per_year:
        if row["protocol"] != "package_certificate_disjoint":
            continue
        ba = "NA" if row["balanced_accuracy"] is None else f"{row['balanced_accuracy']:.6f}"
        auc = "NA" if row["auc"] is None else f"{row['auc']:.6f}"
        print(
            row["year"], row["variant"], f"n={row['n']}",
            f"BA={ba}", f"AUC={auc}", f"ACC={row['accuracy']:.6f}",
            f"FP={row['FP']:.1f}", f"FN={row['FN']:.1f}",
            f"route={row['route_coverage']:.4f}",
        )
    print("\nSUMMARY")
    for row in summary:
        if row["protocol"] == "package_certificate_disjoint":
            print(
                row["variant"],
                f"mean_BA={row['mean_BA_2018_2020']:.6f}",
                f"min_BA={row['min_BA_2018_2020']:.6f}",
                f"mean_AUC={row['mean_AUC_2018_2020']:.6f}",
                f"2021_FPR={row['benign_2021_FPR']:.6f}",
            )
    print("saved =", OUTPUT)


if __name__ == "__main__":
    main()
