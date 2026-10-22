#!/usr/bin/env python3
"""Evaluate the compact P1 features on the provisional annual splits.

This is a pilot signal check, not the formal TASA-Droid graph model.  It uses
only the numeric pilot whitelist, recomputes constant-column filtering inside
each training fold, and reports the same annual test years used by the split
manifest.  It never reads paths, package names, labels, years, certificates,
or parser status as model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "NumPy is required. Use the existing Androguard environment, for example "
        "/home/boyuanjie/envs/tif/bin/python."
    ) from exc


SCHEMA_VERSION = "1.0.0"
AUDIT_COLUMNS = {
    "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
    "metadata_label", "package_name", "certificate_sha256", "version_code",
    "status", "manifest_parse_ok", "error_type", "error_message",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def label_value(value: str) -> int:
    value = str(value or "").strip().lower()
    if value == "benign":
        return 0
    if value in {"malware", "malicious"}:
        return 1
    raise ValueError(f"unknown label: {value!r}")


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(x_train, y_train, seed, iterations=1500, learning_rate=0.05, l2=1e-3):
    rng = np.random.RandomState(seed)
    weights = rng.normal(0.0, 0.01, size=x_train.shape[1])
    bias = 0.0
    n = float(len(y_train))
    for _ in range(iterations):
        probabilities = sigmoid(x_train @ weights + bias)
        error = probabilities - y_train
        grad_w = (x_train.T @ error) / n + l2 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def auc_score(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc_score(y_true, scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def standardize(x_train, x_test):
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    return (x_train - means) / scales, (x_test - means) / scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--whitelist", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--reference-mean", type=float, default=0.9074)
    args = parser.parse_args()

    feature_path = args.features.expanduser().resolve(strict=True)
    split_path = args.splits.expanduser().resolve(strict=True)
    if bool(args.whitelist) == bool(args.schema):
        raise SystemExit("provide exactly one of --schema or --whitelist")
    whitelist_path = args.whitelist.expanduser().resolve(strict=True) if args.whitelist else None
    schema_path = args.schema.expanduser().resolve(strict=True) if args.schema else None
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(feature_path)
    split_rows = read_csv(split_path)
    if schema_path:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        candidate_features = [
            name for name in schema.get("feature_fields", [])
            if name not in AUDIT_COLUMNS
        ]
        feature_source = "FIXED_SCHEMA_COLUMNS_WITHIN_FOLD_FILTERING"
    else:
        whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))
        candidate_features = list(whitelist.get("pilot_numeric_nonconstant_features", []))
        feature_source = "GLOBAL_PILOT_WHITELIST_UNSAFE_FOR_STRICT_COMPARISON"
    if not feature_rows or not split_rows or not candidate_features:
        raise SystemExit("features, split manifest, or candidate feature definition is empty")

    feature_by_sha = {str(row["sha256"]).strip().lower(): row for row in feature_rows}
    feature_header = set(feature_rows[0])
    unknown_features = sorted(set(candidate_features) - feature_header)
    if unknown_features:
        raise SystemExit(f"whitelist features missing from CSV: {unknown_features}")

    split_names = sorted({row["split_name"] for row in split_rows})
    result_rows = []
    summary_rows = []

    for split_name in split_names:
        current = [row for row in split_rows if row["split_name"] == split_name]
        train_manifest = [row for row in current if row["partition"] == "train"]
        test_manifest = [row for row in current if row["partition"] == "test"]
        train_ids = [row["sha256"].strip().lower() for row in train_manifest]
        test_ids = [row["sha256"].strip().lower() for row in test_manifest]
        if not train_ids or not test_ids:
            raise SystemExit(f"empty train or test partition in {split_name}")

        train_rows = [feature_by_sha[sha] for sha in train_ids]
        test_rows = [feature_by_sha[sha] for sha in test_ids]
        usable_features = []
        for name in candidate_features:
            try:
                train_values = np.asarray([float(row[name]) for row in train_rows], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(train_values)):
                continue
            if np.unique(train_values).size > 1:
                usable_features.append(name)

        if not usable_features:
            raise SystemExit(f"no usable train-fold features in {split_name}")
        x_train = np.asarray(
            [[float(row[name]) for name in usable_features] for row in train_rows], dtype=float
        )
        x_test = np.asarray(
            [[float(row[name]) for name in usable_features] for row in test_rows], dtype=float
        )
        if not np.all(np.isfinite(x_test)):
            raise SystemExit(f"test partition contains non-finite values in {split_name}")
        x_train, x_test = standardize(x_train, x_test)
        y_train = np.asarray([label_value(row["metadata_label"]) for row in train_rows])
        y_test = np.asarray([label_value(row["metadata_label"]) for row in test_rows])

        split_metric_rows = []
        for seed in args.seeds:
            weights, bias = fit_logistic(x_train, y_train, seed)
            scores = sigmoid(x_test @ weights + bias)
            metric = metrics(y_test, scores)
            result = {
                "split_name": split_name,
                "seed": seed,
                "model": "compact_logistic_pilot",
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": len(usable_features),
                **metric,
            }
            result_rows.append(result)
            split_metric_rows.append(result)

        summary = {
            "split_name": split_name,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "feature_count_mean": float(np.mean([row["feature_count"] for row in split_metric_rows])),
        }
        for metric_name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"):
            values = [row[metric_name] for row in split_metric_rows if row[metric_name] is not None]
            summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric_name}_std"] = float(np.std(values)) if values else None
        summary_rows.append(summary)

    result_path = output / "pilot_results.csv"
    result_fields = list(result_rows[0].keys())
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    annual_accuracies = [row["accuracy_mean"] for row in summary_rows if row["accuracy_mean"] is not None]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_PILOT_COMPACT_FEATURE_EVALUATION",
        "formal_training_enabled": False,
        "model": "compact_logistic_pilot",
        "features_csv": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "split_manifest_csv": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "feature_source": feature_source,
        "schema_json": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "whitelist_json": str(whitelist_path) if whitelist_path else None,
        "whitelist_sha256": sha256_file(whitelist_path) if whitelist_path else None,
        "strict_test_feature_isolation": bool(schema_path),
        "seeds": list(args.seeds),
        "reference_mean": args.reference_mean,
        "reference_metric": "accuracy_mean_assumed_from_baseline_table",
        "split_summaries": summary_rows,
        "four_year_mean_accuracy": float(np.mean(annual_accuracies)) if annual_accuracies else None,
        "exceeds_reference_point_estimate": (
            bool(np.mean(annual_accuracies) > args.reference_mean)
            if annual_accuracies else False
        ),
        "outputs": {
            "results_csv": str(result_path),
        },
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "compact_summary_is_not_formal_graph_model",
            "tif_implementation_and_metric_definition_not_reproduced",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "Feature filtering is recomputed inside each training fold.",
            "Strict mode defines candidates from the fixed schema and does not inspect test values during filtering.",
            "Audit and identity fields are never used as model inputs.",
            "The legacy global whitelist mode is retained only for backward compatibility and is not strict.",
            "This result is a pilot signal check and cannot be reported as the final TASA-Droid result.",
        ],
    }
    (output / "pilot_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
