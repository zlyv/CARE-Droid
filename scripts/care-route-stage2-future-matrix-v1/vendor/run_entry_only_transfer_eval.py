#!/usr/bin/env python3
"""Evaluate the Entry view alone on the new unpacked transfer test set.

The split manifest defines the transfer protocol.  Its training rows are used
as-is (for example, the existing partial-cache manifest may contain legacy
2017 plus new 2017 rows), and its test rows are scored without fitting or
feature selection.  Only the bounded Entry graph view is used; Compact and
the endpoint view are intentionally excluded.

The classification threshold is fixed at 0.5.  Reliability, graph pooling
and standardization are fitted from the outer training rows only.  No APK is
opened by this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    attach_audit_metadata,
    entry_model_args,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_locked_dual_view_module_ablation import predict_logistic  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def label_from_feature(row: dict) -> int:
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown feature label: {value!r}")


def source_counts(rows: list[dict], feature_by_sha: dict[str, dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        feature = feature_by_sha[row["sha256"]]
        source = str(feature.get("source_id", "unknown")).strip() or "unknown"
        counts[source] += 1
    return dict(sorted(counts.items()))


def year_counts(rows: list[dict], feature_by_sha: dict[str, dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        feature = feature_by_sha[row["sha256"]]
        year = str(feature.get("metadata_year", "unknown")).strip() or "unknown"
        counts[year] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not args.seeds:
        raise SystemExit("at least one seed is required")
    if args.graph_weight < 0.0:
        raise SystemExit("graph-weight must be nonnegative")

    paths = {
        "cache": args.cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(paths["cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)

    selected_shas = {
        row["sha256"]
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    }
    missing = {
        "cache": len(selected_shas - set(cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    per_split = []
    prediction_rows = []
    composition_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")

        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, include_reliability=True
        )
        entry_test, y_test = graph_matrix(
            test_rows, cache, evidence, entry_args, include_reliability=True
        )
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        entry_train *= args.graph_weight
        entry_test *= args.graph_weight

        seed_results = []
        for seed in args.seeds:
            probability = predict_logistic(
                entry_train,
                y_train,
                entry_test,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            result = metrics(y_test, probability, threshold=THRESHOLD)
            seed_results.append(result)
            for row, label, score in zip(test_rows, y_test, probability):
                prediction_rows.append({
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "seed": seed,
                    "label": int(label),
                    "probability": float(score),
                    "threshold": THRESHOLD,
                    "predicted_label": int(score >= THRESHOLD),
                })

        per_split.append({
            "split_name": split_name,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "entry_feature_count": int(entry_train.shape[1]),
            "threshold": THRESHOLD,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_results])),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in seed_results])),
            "precision_mean": float(np.mean([row["precision"] for row in seed_results])),
            "recall_mean": float(np.mean([row["recall"] for row in seed_results])),
            "f1_mean": float(np.mean([row["f1"] for row in seed_results])),
            "auc_mean": float(np.mean([row["auc"] for row in seed_results])),
            "training_source_counts": json.dumps(source_counts(train_rows, feature_by_sha), sort_keys=True),
            "training_year_counts": json.dumps(year_counts(train_rows, feature_by_sha), sort_keys=True),
            "test_source_counts": json.dumps(source_counts(test_rows, feature_by_sha), sort_keys=True),
            "test_year_counts": json.dumps(year_counts(test_rows, feature_by_sha), sort_keys=True),
        })
        train_labels = [label_from_feature(feature_by_sha[row["sha256"]]) for row in train_rows]
        test_labels = [label_from_feature(feature_by_sha[row["sha256"]]) for row in test_rows]
        composition_rows.append({
            "split_name": split_name,
            "train_count": len(train_rows),
            "train_benign": train_labels.count(0),
            "train_malicious": train_labels.count(1),
            "test_count": len(test_rows),
            "test_benign": test_labels.count(0),
            "test_malicious": test_labels.count(1),
            "train_source_counts": json.dumps(source_counts(train_rows, feature_by_sha), sort_keys=True),
            "test_source_counts": json.dumps(source_counts(test_rows, feature_by_sha), sort_keys=True),
            "test_year_counts": json.dumps(year_counts(test_rows, feature_by_sha), sort_keys=True),
        })
        print(
            split_name,
            f"ACC={per_split[-1]['accuracy_mean']:.4f}",
            f"BA={per_split[-1]['balanced_accuracy_mean']:.4f}",
            f"AUC={per_split[-1]['auc_mean']:.4f}",
            flush=True,
        )

    mean_accuracy = float(np.mean([row["accuracy_mean"] for row in per_split]))
    mean_ba = float(np.mean([row["balanced_accuracy_mean"] for row in per_split]))
    summary = [{
        "variant": "entry_graph_only_transfer",
        "evaluated_split_count": len(per_split),
        "mean_accuracy": mean_accuracy,
        "mean_balanced_accuracy": mean_ba,
        "minimum_split_balanced_accuracy": float(min(row["balanced_accuracy_mean"] for row in per_split)),
        "std_balanced_accuracy": float(np.std([row["balanced_accuracy_mean"] for row in per_split])),
        "mean_auc": float(np.mean([row["auc_mean"] for row in per_split])),
    }]

    summary_path = output / "entry_only_transfer_summary.csv"
    per_split_path = output / "entry_only_transfer_per_split.csv"
    composition_path = output / "entry_only_transfer_composition.csv"
    prediction_path = output / "entry_only_transfer_predictions.csv"
    write_csv(summary_path, list(summary[0]), summary)
    write_csv(per_split_path, list(per_split[0]), per_split)
    write_csv(composition_path, list(composition_rows[0]), composition_rows)
    write_csv(
        prediction_path,
        ["split_name", "sha256", "seed", "label", "probability", "threshold", "predicted_label"],
        prediction_rows,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P4_ENTRY_ONLY_CROSS_SOURCE_TRANSFER_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "PILOT_TRANSFER_NOT_FORMAL",
        "strict_test_isolation": True,
        "fixed_threshold": THRESHOLD,
        "graph_weight": args.graph_weight,
        "classifier": {
            "type": "custom_l2_logistic",
            "l2": args.l2,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
        },
        "entry_only": True,
        "compact_used": False,
        "endpoint_view_used": False,
        "split_summaries": per_split,
        "mean_accuracy": mean_accuracy,
        "mean_balanced_accuracy": mean_ba,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(per_split_path),
            "composition": str(composition_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "cross_source_transfer_pilot_not_formal",
            "audited_first_seen_utc_not_joined",
            "entry_cache_is_entry_conditioned_and_may_contain_coverage_shortcuts",
        ],
        "notes": [
            "Training and test partitions are taken exactly from the supplied transfer manifest.",
            "All reliability fitting, graph standardization, and classifier fitting use training rows only.",
            "Test labels are used only for post-hoc metrics.",
            "The classification threshold is fixed at 0.5.",
            "No APK is opened and no raw graph is persisted.",
        ],
    }
    audit_path = output / "entry_only_transfer_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "four_year_or_transfer_mean_ACC=" if len(per_split) != 3 else "three_year_mean_ACC=",
        f"{mean_accuracy:.4f}",
        "mean_BA=",
        f"{mean_ba:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
