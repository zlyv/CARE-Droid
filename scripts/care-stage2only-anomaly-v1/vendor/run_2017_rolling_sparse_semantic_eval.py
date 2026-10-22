#!/usr/bin/env python3
"""Evaluate cache-only sparse semantic structure features on rolling 2017.

The A3 cache contains only stable hashed node IDs, bounded edge indexes,
semantic role IDs, and compact edge flags.  This script derives sparse token
features without reading APKs or persisting raw graphs/signatures.  IDF,
filtering, scaling, and classifiers are fit on each rolling training fold only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
GRAPH_WEIGHT = 0.5
RELIABILITY_COMPONENTS = ("D", "one_minus_F")
SEEDS = (0, 1, 2, 3, 4)
C_VALUES = (0.1, 1.0, 10.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_labels(role_ids, tier: int) -> tuple[str, ...]:
    labels = tuple(f"r{int(value)}" for value in sorted(set(role_ids or [])))
    return labels or (f"u_t{tier}",)


def sparse_semantic_tokens(cache_row: dict, max_two_hop: int = 12000) -> dict[str, float]:
    payload = cache_row.get("payload") or {}
    node_ids = [str(value) for value in payload.get("node_ids") or []]
    node_roles = list(payload.get("node_role_ids") or [])
    base = list(payload.get("node_base_features") or [])
    edge_index = payload.get("edge_index") or [[], []]
    source = [int(value) for value in edge_index[0]]
    target = [int(value) for value in edge_index[1]]
    edge_features = list(payload.get("edge_features") or [])
    edge_type_ids = [str(value) for value in payload.get("edge_type_ids") or []]
    if not (len(node_ids) == len(node_roles) == len(base)):
        raise ValueError("cached node arrays have inconsistent lengths")
    if not (
        len(source) == len(target) == len(edge_features) == len(edge_type_ids)
    ):
        raise ValueError("cached edge arrays have inconsistent lengths")

    tokens: Counter[str] = Counter()
    tiers = []
    external = []
    labels_by_node = []
    for index, (node_id, roles, features) in enumerate(
        zip(node_ids, node_roles, base)
    ):
        if len(features) < 4:
            raise ValueError("node_base_features must contain four values")
        is_external = bool(float(features[0]) > 0.5)
        tier = int(round(float(features[2]) * 4.0))
        tier = min(4, max(0, tier))
        is_entry = bool(float(features[3]) > 0.5)
        role_values = tuple(sorted(set(int(value) for value in roles or [])))
        labels = role_labels(role_values, tier)
        tiers.append(tier)
        external.append(is_external)
        labels_by_node.append(labels)

        tokens[f"NODE_TIER:t{tier}"] += 1.0
        if is_external:
            tokens[f"EXT_API:{node_id}"] += 1.0
        if role_values:
            for role in role_values:
                tokens[f"ROLE:r{role}"] += 1.0
                tokens[f"ROLE_API:r{role}:{node_id}"] += 1.0
                tokens[f"ROLE_TIER:r{role}:t{tier}"] += 1.0
        if is_entry:
            tokens[f"ENTRY_TIER:t{tier}"] += 1.0
            for label in labels:
                tokens[f"ENTRY_ROLE:{label}"] += 1.0

    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    edge_descriptors = []
    for edge_number, (src, dst, features, edge_type) in enumerate(
        zip(source, target, edge_features, edge_type_ids)
    ):
        if src < 0 or dst < 0 or src >= len(node_ids) or dst >= len(node_ids):
            raise ValueError("cached edge index is outside node range")
        if len(features) < 3:
            raise ValueError("cached edge feature must contain three flags")
        flags = "".join("1" if float(value) > 0.5 else "0" for value in features[:3])
        descriptor = f"et{edge_type}:f{flags}"
        edge_descriptors.append(descriptor)
        incoming[dst].append(edge_number)
        outgoing[src].append(edge_number)
        tokens[f"EDGE_TYPE:{descriptor}"] += 1.0
        for src_label in labels_by_node[src]:
            for dst_label in labels_by_node[dst]:
                tokens[f"ROLE_EDGE:{src_label}>{dst_label}:{descriptor}"] += 1.0
        if external[src] and external[dst]:
            tokens[
                f"EXT_EDGE:{node_ids[src]}>{node_ids[dst]}:{edge_type}"
            ] += 1.0
        elif labels_by_node[dst][0].startswith("r"):
            for dst_label in labels_by_node[dst]:
                tokens[
                    f"CONTEXT_TO_ROLE:t{tiers[src]}>{dst_label}:{descriptor}"
                ] += 1.0

    two_hop_count = 0
    for middle in sorted(set(incoming) & set(outgoing)):
        for left_edge in incoming[middle][:32]:
            src = source[left_edge]
            for right_edge in outgoing[middle][:32]:
                dst = target[right_edge]
                for src_label in labels_by_node[src]:
                    for middle_label in labels_by_node[middle]:
                        for dst_label in labels_by_node[dst]:
                            tokens[
                                "ROLE_PATH2:"
                                f"{src_label}>{middle_label}>{dst_label}:"
                                f"{edge_descriptors[left_edge]}:"
                                f"{edge_descriptors[right_edge]}"
                            ] += 1.0
                            two_hop_count += 1
                            if two_hop_count >= max_two_hop:
                                return dict(tokens)
    return dict(tokens)


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    result = metrics(labels, probability, threshold=THRESHOLD)
    result["brier_score"] = float(brier_score_loss(labels, probability))
    result["log_loss"] = float(
        log_loss(labels, np.clip(probability, 1e-7, 1.0 - 1e-7))
    )
    return result


def mean_seed_metrics(rows: list[dict]) -> dict[str, float]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "brier_score",
        "log_loss",
    )
    return {
        f"{name}_mean": float(np.mean([row[name] for row in rows]))
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-dim", type=int, default=16384)
    args = parser.parse_args()
    if args.hash_dim < 1024:
        raise SystemExit("hash-dim must be at least 1024")

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
    split_map = read_split_rows(paths["splits"])

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.1,
        r_max=0.95,
    )
    definition.validate()
    model_args = SimpleNamespace(
        r_min=definition.r_min,
        r_max=definition.r_max,
        role_vocabulary_size=31,
        reliability_components=RELIABILITY_COMPONENTS,
        graph_edge_mode="all",
        graph_summary_mode="full",
        graph_pool_mode="role_stratified",
    )
    hasher = FeatureHasher(
        n_features=args.hash_dim,
        input_type="dict",
        alternate_sign=False,
    )
    token_cache: dict[str, dict[str, float]] = {}
    split_summaries = []

    for split_name in sorted(split_map):
        train_rows = split_map[split_name]["train"]
        validation_rows = split_map[split_name]["test"]
        print(
            f"processing {split_name}: train={len(train_rows)} "
            f"validation={len(validation_rows)}",
            flush=True,
        )
        for row in train_rows + validation_rows:
            sha256 = row["sha256"]
            if sha256 not in cache or sha256 not in feature_by_sha:
                raise SystemExit(f"missing cache or feature row: {sha256}")
            row["metadata_label"] = cache[sha256].get("metadata_label", "")
            row["metadata_year"] = cache[sha256].get("metadata_year", "")
            if sha256 not in token_cache:
                token_cache[sha256] = sparse_semantic_tokens(cache[sha256])

        evidence = fit_training_fold_components(
            cache,
            {row["sha256"] for row in train_rows},
            definition,
            active_components=RELIABILITY_COMPONENTS,
        )
        graph_train, y_train = graph_matrix(
            train_rows, cache, evidence, model_args, include_reliability=True
        )
        graph_validation, y_validation = graph_matrix(
            validation_rows, cache, evidence, model_args, include_reliability=True
        )
        compact_train, compact_validation, compact_count = compact_matrix(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        graph_train, graph_validation = standardize_pair(
            graph_train, graph_validation
        )
        baseline_train = np.concatenate(
            [GRAPH_WEIGHT * graph_train, compact_train], axis=1
        )
        baseline_validation = np.concatenate(
            [GRAPH_WEIGHT * graph_validation, compact_validation], axis=1
        )
        hashed_train = hasher.transform(
            [token_cache[row["sha256"]] for row in train_rows]
        )
        hashed_validation = hasher.transform(
            [token_cache[row["sha256"]] for row in validation_rows]
        )
        tfidf = TfidfTransformer(
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )
        hashed_train = tfidf.fit_transform(hashed_train)
        hashed_validation = tfidf.transform(hashed_validation)

        matrices = {
            "baseline": (
                csr_matrix(baseline_train),
                csr_matrix(baseline_validation),
            ),
            "sparse_semantic": (hashed_train, hashed_validation),
            "baseline_plus_sparse_semantic": (
                hstack([csr_matrix(baseline_train), hashed_train], format="csr"),
                hstack(
                    [csr_matrix(baseline_validation), hashed_validation],
                    format="csr",
                ),
            ),
        }
        for representation, (x_train, x_validation) in matrices.items():
            for c_value in C_VALUES:
                seed_rows = []
                for seed in SEEDS:
                    classifier = LogisticRegression(
                        C=c_value,
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    )
                    classifier.fit(x_train, y_train)
                    probability = classifier.predict_proba(x_validation)[:, 1]
                    seed_rows.append(score(y_validation, probability))
                split_summaries.append(
                    {
                        "split_name": split_name,
                        "variant": f"{representation}__C{c_value:g}",
                        "representation": representation,
                        "C": c_value,
                        "train_count": len(train_rows),
                        "validation_count": len(validation_rows),
                        "feature_count": int(x_train.shape[1]),
                        "compact_feature_count": int(compact_count),
                        "hashed_train_nnz": int(hashed_train.nnz),
                        **mean_seed_metrics(seed_rows),
                    }
                )
        print(f"completed {split_name}", flush=True)

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in split_summaries:
        by_variant[row["variant"]].append(row)
    aggregate = []
    for variant, rows in sorted(by_variant.items()):
        fold_ba = [row["balanced_accuracy_mean"] for row in rows]
        aggregate.append(
            {
                "variant": variant,
                "representation": rows[0]["representation"],
                "C": rows[0]["C"],
                "fold_count": len(rows),
                "mean_accuracy": float(
                    np.mean([row["accuracy_mean"] for row in rows])
                ),
                "mean_balanced_accuracy": float(np.mean(fold_ba)),
                "minimum_fold_balanced_accuracy": float(np.min(fold_ba)),
                "std_fold_balanced_accuracy": float(np.std(fold_ba)),
                "stability_score_mean_minus_std_ba": float(
                    np.mean(fold_ba) - np.std(fold_ba)
                ),
                "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
                "mean_brier_score": float(
                    np.mean([row["brier_score_mean"] for row in rows])
                ),
                "mean_log_loss": float(
                    np.mean([row["log_loss_mean"] for row in rows])
                ),
            }
        )
    aggregate.sort(
        key=lambda row: (
            -row["stability_score_mean_minus_std_ba"],
            -row["mean_balanced_accuracy"],
            row["variant"],
        )
    )

    summary_path = output / "rolling_sparse_semantic_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P5_2017_ROLLING_SPARSE_SEMANTIC_REPRESENTATION_DEVELOPMENT",
        "formal_training_enabled": False,
        "strict_validation_isolation": True,
        "fixed_threshold": THRESHOLD,
        "hash_dimension": args.hash_dim,
        "hash_alternate_sign": False,
        "idf_fit_scope": "corresponding_rolling_training_partition_only",
        "graph_weight": GRAPH_WEIGHT,
        "reliability_components": list(RELIABILITY_COMPONENTS),
        "C_values": list(C_VALUES),
        "seeds": list(SEEDS),
        "cache_sha256": sha256_file(paths["cache"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "split_summaries": split_summaries,
        "aggregate_summaries": aggregate,
        "selection_rule": (
            "maximize mean rolling balanced accuracy minus across-fold "
            "standard deviation; tie-break by mean balanced accuracy"
        ),
        "outputs": {"summary": str(summary_path)},
        "blocking_reasons": [
            "dex_date_is_not_audited_first_seen_utc",
            "representation_selection_is_development_not_confirmatory",
        ],
        "notes": [
            "Only rolling 2017 rows are read.",
            "The classification threshold is fixed at 0.5.",
            "Hashed node IDs are used only as anonymous stable tokens.",
            "IDF and classifiers are fit on each rolling training fold only.",
            "No APK, DEX, raw signature, or raw call graph is read or persisted.",
        ],
    }
    audit_path = output / "rolling_sparse_semantic_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_fold_BA std_BA stability AUC")
    for rank, row in enumerate(aggregate, start=1):
        print(
            rank,
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_fold_balanced_accuracy"]:.4f}',
            f'{row["stability_score_mean_minus_std_ba"]:.4f}',
            f'{row["mean_auc"]:.4f}',
        )


if __name__ == "__main__":
    main()
