#!/usr/bin/env python3
"""Post-hoc error analysis for a locked fixed-year evaluation.

This script consumes already generated predictions and compact features.  It
does not fit a model, select parameters, or reopen APKs.  Test labels are used
only for post-hoc diagnosis; the outputs must not be used to tune the locked
protocol and then reported on the same test rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


AUDIT_COLUMNS = {
    "absolute_path", "certificate_sha256", "error_message", "error_type",
    "lineage_id", "metadata_label", "metadata_year", "package_name",
    "sha256", "source_id", "status", "version_code",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).lstrip("\ufeff"): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def variance(values: list[float], center: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def standardized_delta(error_values: list[float], reference_values: list[float]) -> float | None:
    if not error_values or not reference_values:
        return None
    error_mean = mean(error_values)
    reference_mean = mean(reference_values)
    assert error_mean is not None and reference_mean is not None
    numerator = (len(error_values) - 1) * variance(error_values, error_mean)
    numerator += (len(reference_values) - 1) * variance(reference_values, reference_mean)
    denominator = len(error_values) + len(reference_values) - 2
    pooled = math.sqrt(numerator / denominator) if denominator > 0 else 0.0
    if pooled <= 1e-12:
        return 0.0 if abs(error_mean - reference_mean) <= 1e-12 else None
    return (error_mean - reference_mean) / pooled


def parse_year(split_name: str) -> str:
    match = re.search(r"test_(\d{4})", split_name)
    return match.group(1) if match else "UNKNOWN"


def aggregate_predictions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split_name = row.get("split_name", "")
        sha256 = row.get("sha256", "").casefold()
        if split_name and sha256:
            grouped[(split_name, sha256)].append(row)

    output = []
    for (split_name, sha256), group in sorted(grouped.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"inconsistent labels for {split_name} {sha256}")
        probabilities = [float(row["probability"]) for row in group]
        thresholds = {float(row["threshold"]) for row in group}
        if len(thresholds) != 1:
            raise ValueError(f"inconsistent thresholds for {split_name} {sha256}")
        probability = sum(probabilities) / len(probabilities)
        label = next(iter(labels))
        threshold = next(iter(thresholds))
        predicted = int(probability >= threshold)
        if label == 1 and predicted == 1:
            outcome = "TP"
        elif label == 0 and predicted == 0:
            outcome = "TN"
        elif label == 0 and predicted == 1:
            outcome = "FP"
        else:
            outcome = "FN"
        output.append(
            {
                "split_name": split_name,
                "year": parse_year(split_name),
                "sha256": sha256,
                "label": label,
                "probability_mean": probability,
                "threshold": threshold,
                "predicted_label": predicted,
                "outcome": outcome,
                "seed_count": len(group),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prediction_path = args.predictions.expanduser().resolve(strict=True)
    feature_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    prediction_rows = aggregate_predictions(read_csv_rows(prediction_path))
    feature_rows = read_csv_rows(feature_path)
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_names = [
        name for name in schema.get("feature_fields", [])
        if name not in AUDIT_COLUMNS
    ]

    error_fields = [
        "split_name", "year", "sha256", "label", "probability_mean",
        "threshold", "predicted_label", "outcome", "seed_count",
    ]
    error_rows = [row for row in prediction_rows if row["outcome"] in {"FP", "FN"}]
    write_csv(output / "error_rows.csv", error_rows, error_fields)

    summary_rows = []
    effect_rows = []
    top_error_rows = []
    for split_name in sorted({str(row["split_name"]) for row in prediction_rows}):
        rows = [row for row in prediction_rows if row["split_name"] == split_name]
        year = parse_year(split_name)
        counts = {outcome: sum(row["outcome"] == outcome for row in rows)
                  for outcome in ("TP", "TN", "FP", "FN")}
        accuracy = (counts["TP"] + counts["TN"]) / len(rows) if rows else None
        summary_rows.append(
            {
                "split_name": split_name,
                "year": year,
                "row_count": len(rows),
                "accuracy": accuracy,
                **{f"{key.lower()}_count": value for key, value in counts.items()},
                "test_labels_used_post_hoc_only": True,
            }
        )

        for outcome in ("FP", "FN"):
            errors = [row for row in rows if row["outcome"] == outcome]
            errors.sort(
                key=lambda row: float(row["probability_mean"]),
                reverse=(outcome == "FP"),
            )
            for rank, row in enumerate(errors[:20], start=1):
                top_error_rows.append(
                    {
                        "split_name": split_name,
                        "year": year,
                        "outcome": outcome,
                        "rank": rank,
                        "sha256": row["sha256"],
                        "label": row["label"],
                        "probability_mean": row["probability_mean"],
                        "confidence_in_error": (
                            row["probability_mean"]
                            if outcome == "FP"
                            else 1.0 - float(row["probability_mean"])
                        ),
                    }
                )

        comparisons = (
            ("FP_VS_TN_BENIGN", "FP", "TN", 0),
            ("FN_VS_TP_MALWARE", "FN", "TP", 1),
        )
        for comparison, error_outcome, reference_outcome, label in comparisons:
            error_group = [
                row for row in rows
                if row["outcome"] == error_outcome and row["label"] == label
            ]
            reference_group = [
                row for row in rows
                if row["outcome"] == reference_outcome and row["label"] == label
            ]
            for feature in feature_names:
                error_values = []
                reference_values = []
                for row in error_group:
                    source = feature_by_sha.get(str(row["sha256"]))
                    if source is not None:
                        try:
                            value = float(source[feature])
                            if math.isfinite(value):
                                error_values.append(value)
                        except (KeyError, TypeError, ValueError):
                            pass
                for row in reference_group:
                    source = feature_by_sha.get(str(row["sha256"]))
                    if source is not None:
                        try:
                            value = float(source[feature])
                            if math.isfinite(value):
                                reference_values.append(value)
                        except (KeyError, TypeError, ValueError):
                            pass
                error_mean = mean(error_values)
                reference_mean = mean(reference_values)
                effect_rows.append(
                    {
                        "split_name": split_name,
                        "year": year,
                        "comparison": comparison,
                        "feature": feature,
                        "error_n": len(error_values),
                        "reference_n": len(reference_values),
                        "error_mean": error_mean,
                        "reference_mean": reference_mean,
                        "mean_delta": (
                            error_mean - reference_mean
                            if error_mean is not None and reference_mean is not None
                            else None
                        ),
                        "standardized_delta": standardized_delta(
                            error_values, reference_values
                        ),
                    }
                )

    write_csv(
        output / "error_summary.csv",
        summary_rows,
        [
            "split_name", "year", "row_count", "accuracy",
            "tp_count", "tn_count", "fp_count", "fn_count",
            "test_labels_used_post_hoc_only",
        ],
    )
    write_csv(
        output / "top_errors.csv",
        top_error_rows,
        [
            "split_name", "year", "outcome", "rank", "sha256", "label",
            "probability_mean", "confidence_in_error",
        ],
    )
    write_csv(
        output / "error_feature_effects.csv",
        effect_rows,
        [
            "split_name", "year", "comparison", "feature", "error_n",
            "reference_n", "error_mean", "reference_mean", "mean_delta",
            "standardized_delta",
        ],
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_LOCKED_POST_HOC_ERROR_ANALYSIS",
        "formal_training_enabled": False,
        "predictions": str(prediction_path),
        "features": str(feature_path),
        "schema": str(schema_path),
        "aggregated_prediction_row_count": len(prediction_rows),
        "error_row_count": len(error_rows),
        "feature_count": len(feature_names),
        "outputs": {
            "summary": str(output / "error_summary.csv"),
            "errors": str(output / "error_rows.csv"),
            "top_errors": str(output / "top_errors.csv"),
            "feature_effects": str(output / "error_feature_effects.csv"),
        },
        "blocking_reasons": [
            "post_hoc_test_label_analysis_must_not_be_used_for_retraining",
            "tif_per_apk_predictions_not_available",
        ],
        "notes": [
            "Predictions are averaged across seeds before assigning an outcome.",
            "The analysis does not fit a model or change the locked protocol.",
            "Feature effects are descriptive comparisons of test errors and correct rows.",
        ],
    }
    audit_path = output / "error_analysis_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
