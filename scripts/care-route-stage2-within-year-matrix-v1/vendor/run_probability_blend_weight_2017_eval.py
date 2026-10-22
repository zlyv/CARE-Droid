#!/usr/bin/env python3
"""Select entry/endpoint probability blend weight on 2017 rolling folds.

The script reuses P27 per-sample predictions, so no model, APK, or graph is
recomputed.  For endpoint weight w, probability is (1-w)*entry + w*endpoint.
Only 2017 validation rows are accepted.  Classification threshold is fixed at
0.5; weights are ranked by mean balanced accuracy minus fold standard
deviation, with mean BA and lower endpoint weight as deterministic tie-breaks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402


THRESHOLD = 0.5
DEFAULT_WEIGHTS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ENTRY_VARIANT = "entry_control"
ENDPOINT_VARIANT = "endpoint_expert"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blend_probability(
    entry_probability: np.ndarray,
    endpoint_probability: np.ndarray,
    endpoint_weight: float,
) -> np.ndarray:
    weight = float(endpoint_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("endpoint weight must be in [0, 1]")
    return (1.0 - weight) * entry_probability + weight * endpoint_probability


def prediction_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("split_name", "")),
        str(row.get("sha256", "")).casefold(),
        int(row.get("seed", 0)),
    )


def read_prediction_experts(path: Path) -> tuple[dict, dict]:
    entry = {}
    endpoint = {}
    for row in read_csv_rows(path):
        variant = str(row.get("variant", ""))
        if variant not in {ENTRY_VARIANT, ENDPOINT_VARIANT}:
            continue
        key = prediction_key(row)
        if not key[0] or not key[1]:
            raise ValueError("prediction row has an empty split or SHA256")
        target = entry if variant == ENTRY_VARIANT else endpoint
        if key in target:
            raise ValueError(f"duplicate {variant} prediction: {key}")
        target[key] = {
            "label": int(row["label"]),
            "probability": float(row["probability"]),
        }
    if not entry or not endpoint:
        raise ValueError("entry_control or endpoint_expert predictions are missing")
    if set(entry) != set(endpoint):
        raise ValueError(
            f"expert prediction keys differ: entry={len(entry)} endpoint={len(endpoint)}"
        )
    for key in entry:
        if entry[key]["label"] != endpoint[key]["label"]:
            raise ValueError(f"expert labels differ: {key}")
    return entry, endpoint


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--endpoint-weights",
        nargs="+",
        type=float,
        default=DEFAULT_WEIGHTS,
    )
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.endpoint_weights):
        raise SystemExit("all endpoint weights must be in [0, 1]")
    weights = sorted(set(float(value) for value in args.endpoint_weights))
    if not weights:
        raise SystemExit("at least one endpoint weight is required")

    paths = {
        "predictions": args.predictions.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry, endpoint = read_prediction_experts(paths["predictions"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    split_rows = read_split_rows(paths["splits"])

    predicted_keys = set(entry)
    predicted_shas = {key[1] for key in predicted_keys}
    missing_features = predicted_shas - set(feature_by_sha)
    if missing_features:
        raise SystemExit(f"predicted rows missing features: {len(missing_features)}")
    observed_years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in predicted_shas
    }
    if observed_years != {"2017"}:
        raise SystemExit(
            f"weight selection accepts only 2017 rows; observed={sorted(observed_years)}"
        )

    expected_split_shas = {
        split_name: {row["sha256"] for row in partitions["test"]}
        for split_name, partitions in split_rows.items()
    }
    prediction_split_shas: dict[str, set[str]] = defaultdict(set)
    seeds_by_split: dict[str, set[int]] = defaultdict(set)
    for split_name, sha256, seed in predicted_keys:
        prediction_split_shas[split_name].add(sha256)
        seeds_by_split[split_name].add(seed)
    if set(prediction_split_shas) != set(expected_split_shas):
        raise SystemExit("prediction and split-manifest split names differ")
    for split_name in expected_split_shas:
        if prediction_split_shas[split_name] != expected_split_shas[split_name]:
            raise SystemExit(
                f"{split_name}: prediction validation SHA256 set differs from split manifest"
            )

    per_fold_rows = []
    grid_prediction_rows = []
    per_weight_fold: dict[float, list[dict]] = defaultdict(list)
    for endpoint_weight in weights:
        variant = f"probability_blend_endpoint_w{endpoint_weight:g}"
        for split_name in sorted(expected_split_shas):
            seed_metrics = []
            ordered_shas = sorted(expected_split_shas[split_name])
            for seed in sorted(seeds_by_split[split_name]):
                keys = [(split_name, sha256, seed) for sha256 in ordered_shas]
                missing = [key for key in keys if key not in entry]
                if missing:
                    raise SystemExit(
                        f"{split_name} seed={seed}: missing predictions={len(missing)}"
                    )
                labels = np.asarray([entry[key]["label"] for key in keys], dtype=np.int64)
                entry_probability = np.asarray([
                    entry[key]["probability"] for key in keys
                ], dtype=np.float64)
                endpoint_probability = np.asarray([
                    endpoint[key]["probability"] for key in keys
                ], dtype=np.float64)
                probability = blend_probability(
                    entry_probability, endpoint_probability, endpoint_weight
                )
                seed_metrics.append(metrics(labels, probability, threshold=THRESHOLD))
                for key, label, value in zip(keys, labels, probability):
                    grid_prediction_rows.append({
                        "variant": variant,
                        "endpoint_weight": endpoint_weight,
                        "entry_weight": 1.0 - endpoint_weight,
                        "split_name": split_name,
                        "sha256": key[1],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })
            row = {
                "variant": variant,
                "endpoint_weight": endpoint_weight,
                "entry_weight": 1.0 - endpoint_weight,
                "split_name": split_name,
                "validation_count": len(ordered_shas),
                "seed_count": len(seed_metrics),
                "threshold": THRESHOLD,
            }
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                row[f"{name}_mean"] = float(np.mean([
                    result[name] for result in seed_metrics
                ]))
            per_fold_rows.append(row)
            per_weight_fold[endpoint_weight].append(row)

    summary_rows = []
    for endpoint_weight in weights:
        rows = per_weight_fold[endpoint_weight]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        summary_rows.append({
            "variant": f"probability_blend_endpoint_w{endpoint_weight:g}",
            "endpoint_weight": endpoint_weight,
            "entry_weight": 1.0 - endpoint_weight,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    summary_rows.sort(key=lambda row: (
        -row["stability_score"],
        -row["mean_balanced_accuracy"],
        row["endpoint_weight"],
    ))
    for rank, row in enumerate(summary_rows, 1):
        row["rank"] = rank

    summary_path = output / "probability_blend_weight_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "endpoint_weight", "entry_weight",
        "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, summary_rows)
    fold_path = output / "probability_blend_weight_2017_per_fold.csv"
    fold_fields = [
        "variant", "endpoint_weight", "entry_weight", "split_name",
        "validation_count", "seed_count", "threshold", "accuracy_mean",
        "balanced_accuracy_mean", "precision_mean", "recall_mean", "f1_mean",
        "auc_mean",
    ]
    write_csv(fold_path, fold_fields, per_fold_rows)
    prediction_path = output / "probability_blend_weight_2017_predictions.csv"
    prediction_fields = [
        "variant", "endpoint_weight", "entry_weight", "split_name", "sha256",
        "seed", "label", "probability", "threshold", "predicted_label",
    ]
    write_csv(prediction_path, prediction_fields, grid_prediction_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P31_2017_ROLLING_PROBABILITY_BLEND_WEIGHT_SELECTION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_WEIGHT_SELECTION_ONLY",
        "strict_future_year_isolation": True,
        "observed_years": sorted(observed_years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "endpoint_weights_compared": weights,
        "selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "tie_breakers": ["mean_balanced_accuracy_desc", "endpoint_weight_asc"],
        "source_predictions": str(paths["predictions"]),
        "source_predictions_sha256": sha256_file(paths["predictions"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "ranked_weights": summary_rows,
        "selected_endpoint_weight": summary_rows[0]["endpoint_weight"],
        "selected_entry_weight": summary_rows[0]["entry_weight"],
        "model_refit_performed": False,
        "outputs": {
            "summary": str(summary_path),
            "per_fold": str(fold_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "selected_weight_not_yet_locked_in_future_year_evaluator",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Only P27 predictions for chronological 2017 validation rows are used.",
            "No APK, graph cache, training model, or 2018--2021 row is read.",
            "The classification threshold remains fixed at 0.5 for all weights.",
            "The coarse grid limits overfitting to three development folds.",
        ],
    }
    audit_path = output / "probability_blend_weight_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank endpoint_weight entry_weight mean_BA min_BA std_BA stability AUC", flush=True)
    for row in summary_rows:
        print(
            row["rank"],
            f'{row["endpoint_weight"]:.2f}',
            f'{row["entry_weight"]:.2f}',
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
