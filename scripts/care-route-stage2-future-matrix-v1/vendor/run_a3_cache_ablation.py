#!/usr/bin/env python3
"""Cache-only ablation for the A3-MVP graph pilot.

The script compares three graph inputs on the same outer split:

* ``graph_reliability``: current reliability-aware graph input;
* ``graph_no_reliability``: the same graph model without D/T/(1-F) node input;
* ``graph_compact_fused``: reliability-aware graph input plus fold-local
  standardized compact static features at the APK pooling layer.

All threshold selection and compact-feature normalization are performed inside
the outer training partition.  No APK is reopened and no outer test row is
used for fitting or selection.  The existing compact nested result should be
reported beside this audit as the compact-only reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tasa_droid.p3.a3_graph_model import A3GraphClassifier, CachedGraph, build_cached_graph  # noqa: E402
from tasa_droid.p3.reliability import fit_training_fold_components, read_cache  # noqa: E402


AUDIT_COLUMNS = {
    "absolute_path",
    "certificate_sha256",
    "error_message",
    "error_type",
    "lineage_id",
    "metadata_label",
    "metadata_year",
    "package_name",
    "sha256",
    "source_id",
    "status",
    "version_code",
}

VARIANTS = {
    "graph_reliability": {"include_reliability": True, "fuse_compact": False},
    "graph_no_reliability": {"include_reliability": False, "fuse_compact": False},
    "graph_compact_fused": {"include_reliability": True, "fuse_compact": True},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).lstrip("\ufeff"): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_split_rows(path: Path) -> dict[str, dict[str, list[dict]]]:
    result = defaultdict(lambda: {"train": [], "test": []})
    for row in read_csv_rows(path):
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
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def group_folds(rows: list[dict], n_splits: int) -> list[list[int]]:
    groups = defaultdict(list)
    positives = defaultdict(int)
    for index, row in enumerate(rows):
        group = str(row["blocking_group"])
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
    target_total = len(rows) / actual
    target_positive = sum(positives.values()) / actual
    for fold, (_group, indices) in enumerate(ordered[:actual]):
        folds[fold].extend(indices)
        totals[fold] += len(indices)
        positive_totals[fold] += positives[_group]
    for group, indices in ordered[actual:]:
        chosen = min(
            range(actual),
            key=lambda fold: (
                abs(positive_totals[fold] + positives[group] - target_positive),
                abs(totals[fold] + len(indices) - target_total),
                fold,
            ),
        )
        folds[chosen].extend(indices)
        totals[chosen] += len(indices)
        positive_totals[chosen] += positives[group]
    if any(not fold for fold in folds):
        raise ValueError("grouped inner split produced an empty fold")
    return folds


def accuracy(y, pred) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    values = []
    for label in (0, 1):
        mask = y == label
        if np.any(mask):
            values.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(values)) if values else 0.0


def f1_score(y, pred) -> float:
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


def candidate_features(schema: dict) -> list[str]:
    return [
        name for name in schema.get("feature_fields", []) if name not in AUDIT_COLUMNS
    ]


def select_features(rows: list[dict], candidates: list[str]) -> list[str]:
    selected = []
    for name in candidates:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)) and np.unique(values).size > 1:
            selected.append(name)
    return selected


def compact_matrices(
    fit_rows: list[dict], eval_rows: list[dict], feature_by_sha: dict[str, dict], candidates: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fit_table = [feature_by_sha[row["sha256"]] for row in fit_rows]
    eval_table = [feature_by_sha[row["sha256"]] for row in eval_rows]
    names = select_features(fit_table, candidates)
    if not names:
        raise ValueError("training fold has no usable compact features")
    x_fit = np.asarray([[float(row[name]) for name in names] for row in fit_table])
    x_eval = np.asarray([[float(row[name]) for name in names] for row in eval_table])
    if not np.all(np.isfinite(x_fit)) or not np.all(np.isfinite(x_eval)):
        raise ValueError("compact feature matrix contains non-finite values")
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_fit - mean) / scale, (x_eval - mean) / scale, names


def make_graphs(
    rows: list[dict],
    cache: dict[str, dict],
    evidence: dict,
    args,
    include_reliability: bool,
    extra_features: np.ndarray | None = None,
) -> tuple[list[CachedGraph], np.ndarray]:
    weights = {
        "D": args.weight_d,
        "T": args.weight_t,
        "one_minus_F": args.weight_one_minus_f,
    }
    graphs = []
    labels = []
    for index, row in enumerate(rows):
        cache_row = cache.get(row["sha256"])
        if cache_row is None:
            raise ValueError(f"cache row missing: {row['sha256']}")
        graph = build_cached_graph(
            cache_row["payload"],
            evidence["components_by_node_id"],
            evidence["global_training_prior"],
            weights,
            args.r_min,
            args.r_max,
            args.role_vocabulary_size,
            include_reliability=include_reliability,
        )
        if extra_features is not None:
            graph = replace(graph, extra_features=np.asarray(extra_features[index], dtype=np.float64))
        graphs.append(graph)
        labels.append(label_of(cache_row))
    return graphs, np.asarray(labels, dtype=np.float64)


def model_for(graphs: list[CachedGraph], args, seed: int, extra_dim: int) -> A3GraphClassifier:
    return A3GraphClassifier(
        input_dim=graphs[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        seed=seed,
        extra_dim=extra_dim,
    )


def select_threshold(
    train_rows: list[dict],
    variant: dict,
    cache: dict[str, dict],
    feature_by_sha: dict[str, dict],
    candidates: list[str],
    args,
) -> tuple[float, dict, list[int]]:
    folds = group_folds(train_rows, args.inner_folds)
    scores = {float(value): {"accuracy": [], "balanced_accuracy": []} for value in args.thresholds}
    selected_feature_counts = []
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        fit_rows = [row for index, row in enumerate(train_rows) if index not in validation_set]
        validation_rows = [train_rows[index] for index in validation_indices]
        evidence = fit_training_fold_components(cache, {row["sha256"] for row in fit_rows}, args.definition)
        fit_extra = validation_extra = None
        extra_dim = 0
        if variant["fuse_compact"]:
            fit_extra, validation_extra, names = compact_matrices(
                fit_rows, validation_rows, feature_by_sha, candidates
            )
            extra_dim = len(names)
            selected_feature_counts.append(extra_dim)
        fit_graphs, fit_y = make_graphs(
            fit_rows, cache, evidence, args, variant["include_reliability"], fit_extra
        )
        validation_graphs, validation_y = make_graphs(
            validation_rows, cache, evidence, args, variant["include_reliability"], validation_extra
        )
        model = model_for(fit_graphs, args, args.selection_seed + fold_index, extra_dim)
        model.fit(
            fit_graphs,
            fit_y,
            learning_rate=args.learning_rate,
            l2=args.l2,
            epochs=args.epochs,
        )
        probability = model.predict_proba(validation_graphs)
        for threshold in scores:
            predicted = (probability >= threshold).astype(int)
            scores[threshold]["accuracy"].append(accuracy(validation_y, predicted))
            scores[threshold]["balanced_accuracy"].append(
                balanced_accuracy(validation_y, predicted)
            )
    selected = max(
        scores,
        key=lambda threshold: (
            float(np.mean(scores[threshold]["balanced_accuracy"])),
            float(np.mean(scores[threshold]["accuracy"])),
            -abs(threshold - 0.5),
            -threshold,
        ),
    )
    report = {
        str(key): {
            "accuracy": float(np.mean(value["accuracy"])),
            "balanced_accuracy": float(np.mean(value["balanced_accuracy"])),
        }
        for key, value in scores.items()
    }
    return selected, report, selected_feature_counts


def evaluate_variant(
    variant_name: str,
    variant: dict,
    split_name: str,
    train_rows: list[dict],
    test_rows: list[dict],
    cache: dict[str, dict],
    feature_by_sha: dict[str, dict],
    candidates: list[str],
    args,
) -> tuple[dict, list[dict]]:
    threshold, inner_scores, inner_feature_counts = select_threshold(
        train_rows, variant, cache, feature_by_sha, candidates, args
    )
    evidence = fit_training_fold_components(
        cache, {row["sha256"] for row in train_rows}, args.definition
    )
    train_extra = test_extra = None
    feature_count = 0
    if variant["fuse_compact"]:
        train_extra, test_extra, names = compact_matrices(
            train_rows, test_rows, feature_by_sha, candidates
        )
        feature_count = len(names)
    train_graphs, train_y = make_graphs(
        train_rows, cache, evidence, args, variant["include_reliability"], train_extra
    )
    test_graphs, test_y = make_graphs(
        test_rows, cache, evidence, args, variant["include_reliability"], test_extra
    )
    seed_metrics = []
    prediction_rows = []
    for seed in args.seeds:
        model = model_for(train_graphs, args, seed, feature_count)
        model.fit(
            train_graphs,
            train_y,
            learning_rate=args.learning_rate,
            l2=args.l2,
            epochs=args.epochs,
        )
        probability = model.predict_proba(test_graphs)
        predicted = (probability >= threshold).astype(int)
        seed_metrics.append(
            {
                "accuracy": accuracy(test_y, predicted),
                "balanced_accuracy": balanced_accuracy(test_y, predicted),
                "f1": f1_score(test_y, predicted),
                "auc": auc_score(test_y, probability),
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
                "probability_mean": float(probability.mean()),
                "predicted_positive_rate": float(predicted.mean()),
            }
        )
        for row, label, prob, pred in zip(test_rows, test_y, probability, predicted):
            prediction_rows.append(
                {
                    "variant": variant_name,
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "seed": seed,
                    "label": int(label),
                    "probability": float(prob),
                    "predicted_label": int(pred),
                    "threshold": float(threshold),
                }
            )
    auc_values = [row["auc"] for row in seed_metrics if row["auc"] is not None]
    summary = {
        "variant": variant_name,
        "split_name": split_name,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "threshold": float(threshold),
        "inner_threshold_scores": inner_scores,
        "inner_fold_compact_feature_count": inner_feature_counts,
        "outer_compact_feature_count": feature_count,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_metrics])),
        "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in seed_metrics])),
        "f1_mean": float(np.mean([row["f1"] for row in seed_metrics])),
        "auc_mean": float(np.mean(auc_values)) if auc_values else None,
        "probability_min_mean": float(np.mean([row["probability_min"] for row in seed_metrics])),
        "probability_max_mean": float(np.mean([row["probability_max"] for row in seed_metrics])),
        "probability_mean": float(np.mean([row["probability_mean"] for row in seed_metrics])),
        "predicted_positive_rate_mean": float(np.mean([row["predicted_positive_rate"] for row in seed_metrics])),
        "test_positive_rate": float(np.mean(test_y)),
        "test_rows_used_for_training_or_selection": False,
        "outer_training_node_count": evidence["node_count"],
    }
    return summary, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", dest="split_names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[round(0.05 + i * 0.05, 2) for i in range(19)])
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
    paths = {
        "cache": args.cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = read_cache(paths["cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    candidates = candidate_features(schema)
    split_rows = read_split_rows(paths["splits"])
    if args.split_names:
        unknown = set(args.split_names) - set(split_rows)
        if unknown:
            raise SystemExit(f"unknown split names: {sorted(unknown)}")
        split_rows = {name: split_rows[name] for name in args.split_names}
    for partitions in split_rows.values():
        for row in partitions["train"] + partitions["test"]:
            cache_row = cache.get(row["sha256"])
            if cache_row is None:
                raise SystemExit(f"split row missing from cache: {row['sha256']}")
            row["metadata_label"] = cache_row.get("metadata_label", "")
            if row["sha256"] not in feature_by_sha:
                raise SystemExit(f"split row missing from compact features: {row['sha256']}")

    summaries = []
    predictions = []
    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty outer partition: {split_name}")
        for variant_name, variant in VARIANTS.items():
            print(f"running {variant_name} on {split_name}", flush=True)
            summary, rows = evaluate_variant(
                variant_name,
                variant,
                split_name,
                train_rows,
                test_rows,
                cache,
                feature_by_sha,
                candidates,
                args,
            )
            summaries.append(summary)
            predictions.extend(rows)

    prediction_path = output / "a3_cache_ablation_predictions.csv"
    fields = ["variant", "split_name", "sha256", "seed", "label", "probability", "predicted_label", "threshold"]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_A3_CACHE_LEVEL_ABLATION",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "variants": list(VARIANTS),
        "cache": str(paths["cache"]),
        "cache_sha256": sha256_file(paths["cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "seeds": list(args.seeds),
        "inner_folds": args.inner_folds,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "thresholds": list(args.thresholds),
        "compact_candidate_feature_count": len(candidates),
        "reliability_definition_status": "PILOT_PROVISIONAL_NOT_FORMAL",
        "split_summaries": summaries,
        "outputs": {"predictions": str(prediction_path)},
        "blocking_reasons": [
            "pilot_reliability_definition_not_resolved_by_nested_validation",
            "formal_training_config_not_resolved",
            "tif_predictions_not_available_for_paired_comparison",
        ],
        "notes": [
            "Outer test rows are not used for reliability, compact normalization, threshold selection, or fitting.",
            "Graph variants reuse the same bounded cache and outer split.",
            "graph_compact_fused standardizes compact features from each corresponding fit partition only.",
            "The compact-only nested result must be compared from its existing audit on the same split.",
            "This is a cache-level diagnostic ablation, not a final formal result.",
        ],
    }
    audit_path = output / "a3_cache_ablation_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
