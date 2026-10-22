#!/usr/bin/env python3
"""Nested annual pilot evaluation for the cached A3-MVP graph model.

The outer test year is touched only for final scoring.  Reliability is fit
from the corresponding training rows; inner folds refit reliability before
threshold selection.  The output is a pilot result and does not enable formal
training.
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

from tasa_droid.p3.a3_graph_model import (  # noqa: E402
    A3GraphClassifier,
    build_cached_graph,
)
from tasa_droid.p3.reliability import (  # noqa: E402
    fit_training_fold_components,
    read_cache,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_rows(path: Path) -> dict[str, dict[str, list[dict]]]:
    result = defaultdict(lambda: {"train": [], "test": []})
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {str(k).lstrip("\ufeff"): v for k, v in raw.items()}
            split = row.get("split_name", "")
            partition = row.get("partition", "").casefold()
            sha256 = row.get("sha256", "").casefold()
            if split and partition in {"train", "test"} and sha256:
                result[split][partition].append(
                    {
                        "sha256": sha256,
                        "blocking_group": row.get("blocking_group") or sha256,
                    }
                )
    return dict(result)


def label_of(row: dict) -> int:
    label = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if label in {"malicious", "malware", "1", "true"}:
        return 1
    if label in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {label!r}")


def group_folds(rows: list[dict], n_splits: int) -> list[list[int]]:
    groups = defaultdict(list)
    positives = defaultdict(int)
    for index, row in enumerate(rows):
        group = row["blocking_group"]
        groups[group].append(index)
        positives[group] += label_of(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), -positives[item[0]], item[0]),
    )
    actual = min(max(2, n_splits), len(ordered))
    if actual < 2:
        raise ValueError("not enough blocking groups for inner validation")
    folds = [[] for _ in range(actual)]
    totals = [0] * actual
    positive_totals = [0] * actual
    for fold_index, (_group, indices) in enumerate(ordered[:actual]):
        folds[fold_index].extend(indices)
        totals[fold_index] += len(indices)
        positive_totals[fold_index] += positives[_group]
    for group, indices in ordered[actual:]:
        chosen = min(
            range(actual),
            key=lambda fold: (
                abs((positive_totals[fold] + positives[group]) - sum(positive_totals) / actual),
                abs((totals[fold] + len(indices)) - sum(totals) / actual),
                fold,
            ),
        )
        folds[chosen].extend(indices)
        totals[chosen] += len(indices)
        positive_totals[chosen] += positives[group]
    if any(not fold for fold in folds):
        raise ValueError("inner grouped fold is empty")
    return folds


def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def balanced_accuracy(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    values = []
    for label in (0, 1):
        mask = y == label
        if np.any(mask):
            values.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(values)) if values else 0.0


def f1_score(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def auc_score(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability)
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if not positive or not negative:
        return None
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def rows_to_graphs(rows, cache, evidence, args):
    component_map = evidence["components_by_node_id"]
    prior = evidence["global_training_prior"]
    weights = {"D": args.weight_d, "T": args.weight_t, "one_minus_F": args.weight_one_minus_f}
    graphs = []
    labels = []
    used_rows = []
    for row in rows:
        sha256 = row["sha256"]
        cache_row = cache.get(sha256)
        if cache_row is None:
            raise ValueError(f"cache row missing: {sha256}")
        graphs.append(
            build_cached_graph(
                cache_row["payload"],
                component_map,
                prior,
                weights,
                args.r_min,
                args.r_max,
                args.role_vocabulary_size,
            )
        )
        labels.append(label_of(cache_row))
        used_rows.append(row)
    return graphs, np.asarray(labels, dtype=np.float64), used_rows


def select_threshold(train_rows, cache, args):
    folds = group_folds(train_rows, args.inner_folds)
    threshold_values = [float(value) for value in args.thresholds]
    scores = {
        threshold: {"accuracy": [], "balanced_accuracy": []}
        for threshold in threshold_values
    }
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        fit_rows = [row for index, row in enumerate(train_rows) if index not in validation_set]
        validation_rows = [train_rows[index] for index in validation_indices]
        fit_shas = {row["sha256"] for row in fit_rows}
        evidence = fit_training_fold_components(cache, fit_shas, args.definition)
        fit_graphs, fit_y, _ = rows_to_graphs(fit_rows, cache, evidence, args)
        validation_graphs, validation_y, _ = rows_to_graphs(validation_rows, cache, evidence, args)
        model = A3GraphClassifier(
            input_dim=4 + args.role_vocabulary_size + 1,
            hidden_dim=args.hidden_dim,
            seed=args.selection_seed + fold_index,
        )
        model.fit(
            fit_graphs,
            fit_y,
            learning_rate=args.learning_rate,
            l2=args.l2,
            epochs=args.epochs,
        )
        probability = model.predict_proba(validation_graphs)
        for threshold in threshold_values:
            pred = (probability >= threshold).astype(int)
            scores[threshold]["accuracy"].append(accuracy(validation_y, pred))
            scores[threshold]["balanced_accuracy"].append(
                balanced_accuracy(validation_y, pred)
            )
    selected = max(
        threshold_values,
        key=lambda threshold: (
            float(np.mean(scores[threshold]["balanced_accuracy"])),
            float(np.mean(scores[threshold]["accuracy"])),
            -abs(threshold - 0.5),
            -threshold,
        ),
    )
    return selected, {
        str(key): {
            "accuracy": float(np.mean(value["accuracy"])),
            "balanced_accuracy": float(np.mean(value["balanced_accuracy"])),
        }
        for key, value in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A3-MVP cached graph pilot evaluation")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", dest="split_names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[round(0.05 + 0.05 * index, 2) for index in range(19)],
    )
    parser.add_argument("--selection-seed", type=int, default=20260806)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--kappa-support", type=float, default=20.0)
    parser.add_argument("--kappa-context", type=float, default=20.0)
    parser.add_argument("--r-min", type=float, default=0.10)
    parser.add_argument("--r-max", type=float, default=0.95)
    parser.add_argument("--weight-d", type=float, default=1.0)
    parser.add_argument("--weight-t", type=float, default=1.0)
    parser.add_argument("--weight-one-minus-f", type=float, default=1.0)
    parser.add_argument("--role-vocabulary-size", type=int, default=31)
    args = parser.parse_args()
    from tasa_droid.p3.reliability import PilotReliabilityDefinition

    args.definition = PilotReliabilityDefinition(
        alpha_prior=args.alpha_prior,
        kappa_support=args.kappa_support,
        kappa_context=args.kappa_context,
        r_min=args.r_min,
        r_max=args.r_max,
    )
    args.definition.validate()
    if args.weight_d < 0 or args.weight_t < 0 or args.weight_one_minus_f < 0:
        raise SystemExit("reliability weights must be nonnegative")
    cache_path = args.cache.expanduser().resolve()
    split_path = args.splits.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = read_cache(cache_path)
    split_rows = read_split_rows(split_path)
    if args.split_names:
        unknown = set(args.split_names) - set(split_rows)
        if unknown:
            raise SystemExit(f"unknown split names: {sorted(unknown)}")
        split_rows = {name: split_rows[name] for name in args.split_names}
    for partitions in split_rows.values():
        for row in partitions["train"] + partitions["test"]:
            cache_row = cache.get(row["sha256"])
            if cache_row is None:
                raise SystemExit(f"split manifest row missing from cache: {row['sha256']}")
            row["metadata_label"] = cache_row.get("metadata_label", "")
    prediction_rows = []
    summaries = []
    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty outer partition: {split_name}")
        threshold, inner_scores = select_threshold(train_rows, cache, args)
        outer_shas = {row["sha256"] for row in train_rows}
        outer_evidence = fit_training_fold_components(cache, outer_shas, args.definition)
        train_graphs, train_y, _ = rows_to_graphs(train_rows, cache, outer_evidence, args)
        test_graphs, test_y, test_rows_used = rows_to_graphs(test_rows, cache, outer_evidence, args)
        seed_metrics = []
        seed_probabilities = []
        seed_predictions = []
        for seed in args.seeds:
            model = A3GraphClassifier(
                input_dim=4 + args.role_vocabulary_size + 1,
                hidden_dim=args.hidden_dim,
                seed=seed,
            )
            model.fit(
                train_graphs,
                train_y,
                learning_rate=args.learning_rate,
                l2=args.l2,
                epochs=args.epochs,
            )
            probability = model.predict_proba(test_graphs)
            predicted = (probability >= threshold).astype(int)
            seed_probabilities.append(probability)
            seed_predictions.append(predicted)
            metrics = {
                "accuracy": accuracy(test_y, predicted),
                "balanced_accuracy": balanced_accuracy(test_y, predicted),
                "f1": f1_score(test_y, predicted),
                "auc": auc_score(test_y, probability),
            }
            seed_metrics.append(metrics)
            for row, label, prob, pred in zip(test_rows_used, test_y, probability, predicted):
                prediction_rows.append(
                    {
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(prob),
                        "predicted_label": int(pred),
                        "threshold": float(threshold),
                    }
                )
        summaries.append(
            {
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "threshold": threshold,
                "inner_threshold_scores": inner_scores,
                "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_metrics])),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in seed_metrics])),
                "f1_mean": float(np.mean([row["f1"] for row in seed_metrics])),
                "auc_mean": float(np.mean([row["auc"] for row in seed_metrics if row["auc"] is not None])),
                "probability_min_mean": float(np.mean([values.min() for values in seed_probabilities])),
                "probability_max_mean": float(np.mean([values.max() for values in seed_probabilities])),
                "probability_mean": float(np.mean([values.mean() for values in seed_probabilities])),
                "predicted_positive_rate_mean": float(np.mean([values.mean() for values in seed_predictions])),
                "test_positive_rate": float(np.mean(test_y)),
                "test_rows_used_for_training_or_selection": False,
                "outer_training_node_count": outer_evidence["node_count"],
            }
        )

    prediction_path = output / "a3_mvp_pilot_predictions.csv"
    fields = ["split_name", "sha256", "seed", "label", "probability", "predicted_label", "threshold"]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_A3_MVP_CACHED_GRAPH_PILOT_EVALUATION",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "model": "two_layer_mean_message_passing_graph_classifier",
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "seeds": list(args.seeds),
        "inner_folds": args.inner_folds,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "thresholds": list(args.thresholds),
        "reliability_definition_status": "PILOT_PROVISIONAL_NOT_FORMAL",
        "split_summaries": summaries,
        "evaluated_split_count": len(summaries),
        "mean_accuracy_over_evaluated_splits": float(
            np.mean([row["accuracy_mean"] for row in summaries])
        ),
        "four_year_mean_accuracy": (
            float(np.mean([row["accuracy_mean"] for row in summaries]))
            if len(summaries) == 4
            else None
        ),
        "outputs": {"predictions": str(prediction_path)},
        "blocking_reasons": [
            "pilot_reliability_definition_not_resolved_by_nested_validation",
            "formal_training_config_not_resolved",
            "tif_predictions_not_available_for_paired_comparison",
        ],
        "notes": [
            "Outer test rows are not used for reliability, threshold, or weight selection.",
            "Inner folds refit reliability from inner-training rows only.",
            "Unseen test nodes use the explicit global prior from outer training.",
            "This is a graph-model pilot, not a final formal result.",
        ],
    }
    (output / "a3_mvp_pilot_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
