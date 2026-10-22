#!/usr/bin/env python3
"""Evaluate global/local semantic alignment without reopening APKs.

The experiment separates compact features into topology-free base statistics
and global semantic marginals.  It also compares role-agnostic graph summaries
with the existing role-aligned graph summaries.  Role-agnostic graphs retain
the same bounded topology, generic sensitive/tier indicators, reachability,
provenance, edge types, and reliability; only role-specific dimensions are
removed.  Classification threshold and probability blend weights are fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    attach_audit_metadata,
    endpoint_matrix,
    entry_model_args,
)
from run_dual_view_error_attribution import (  # noqa: E402
    compact_matrix_named,
    endpoint_feature_names,
    entry_feature_names,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_GRAPH_WEIGHT = 0.25
ENTRY_PROBABILITY_WEIGHT = 0.75
ENDPOINT_PROBABILITY_WEIGHT = 0.25

SEMANTIC_COMPACT_EXACT = {
    "mapped_api_node_count",
    "unmapped_external_node_count",
    "mapped_external_ratio",
    "sensitive_node_count",
    "sensitive_edge_count",
    "manifest_context_match_count",
}

FIT_VARIANTS = (
    "compact_base_only",
    "compact_global_semantic_only",
    "compact_full",
    "role_agnostic_entry_plus_compact",
    "role_aligned_entry_plus_compact",
    "role_agnostic_endpoint_plus_compact",
    "role_aligned_endpoint_plus_compact",
)
BLEND_VARIANTS = (
    "role_agnostic_probability_blend",
    "role_aligned_probability_blend",
)
ALL_VARIANTS = FIT_VARIANTS + BLEND_VARIANTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    weights, bias = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
    )
    return sigmoid(x_eval @ weights + bias)


def is_global_semantic_compact(name: str) -> bool:
    return (
        name in SEMANTIC_COMPACT_EXACT
        or name.startswith("role_node__")
        or name.startswith("role_edge__")
        or name.startswith("context_role__")
        or name.startswith("permission_behavior__")
    )


def role_agnostic_graph_mask(names: list[str]) -> np.ndarray:
    """Remove only role-identity dimensions, retaining generic graph context."""

    keep = []
    for name in names:
        role_specific = (
            "__role__" in name
            or name.startswith("role_pool__")
            or name.startswith("role_edge__")
        )
        keep.append(not role_specific)
    result = np.asarray(keep, dtype=bool)
    if not np.any(result) or np.all(result):
        raise ValueError("role-agnostic mask did not separate graph dimensions")
    return result


def build_matrices(
    entry_train: np.ndarray,
    entry_eval: np.ndarray,
    endpoint_train: np.ndarray,
    endpoint_eval: np.ndarray,
    compact_train: np.ndarray,
    compact_eval: np.ndarray,
    compact_names: list[str],
    entry_weight: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    entry_names = entry_feature_names()
    endpoint_names = endpoint_feature_names()
    if entry_train.shape[1] != len(entry_names):
        raise ValueError(
            f"entry feature names differ: matrix={entry_train.shape[1]} names={len(entry_names)}"
        )
    if endpoint_train.shape[1] != len(endpoint_names):
        raise ValueError(
            "endpoint feature names differ: "
            f"matrix={endpoint_train.shape[1]} names={len(endpoint_names)}"
        )
    semantic_mask = np.asarray(
        [is_global_semantic_compact(name) for name in compact_names], dtype=bool
    )
    base_mask = ~semantic_mask
    if not np.any(semantic_mask) or not np.any(base_mask):
        raise ValueError("compact semantic/base split is empty")
    entry_agnostic = role_agnostic_graph_mask(entry_names)
    endpoint_agnostic = role_agnostic_graph_mask(endpoint_names)

    ce_train = compact_train[:, semantic_mask]
    ce_eval = compact_eval[:, semantic_mask]
    cb_train = compact_train[:, base_mask]
    cb_eval = compact_eval[:, base_mask]
    ea_train = entry_weight * entry_train[:, entry_agnostic]
    ea_eval = entry_weight * entry_eval[:, entry_agnostic]
    er_train = entry_weight * entry_train
    er_eval = entry_weight * entry_eval
    sa_train = endpoint_train[:, endpoint_agnostic]
    sa_eval = endpoint_eval[:, endpoint_agnostic]

    matrices = {
        "compact_base_only": (cb_train, cb_eval),
        "compact_global_semantic_only": (ce_train, ce_eval),
        "compact_full": (compact_train, compact_eval),
        "role_agnostic_entry_plus_compact": (
            np.concatenate([ea_train, compact_train], axis=1),
            np.concatenate([ea_eval, compact_eval], axis=1),
        ),
        "role_aligned_entry_plus_compact": (
            np.concatenate([er_train, compact_train], axis=1),
            np.concatenate([er_eval, compact_eval], axis=1),
        ),
        "role_agnostic_endpoint_plus_compact": (
            np.concatenate([sa_train, compact_train], axis=1),
            np.concatenate([sa_eval, compact_eval], axis=1),
        ),
        "role_aligned_endpoint_plus_compact": (
            np.concatenate([endpoint_train, compact_train], axis=1),
            np.concatenate([endpoint_eval, compact_eval], axis=1),
        ),
    }
    dimensions = {
        "compact_base": int(base_mask.sum()),
        "compact_global_semantic": int(semantic_mask.sum()),
        "compact_full": len(compact_names),
        "entry_role_agnostic": int(entry_agnostic.sum()),
        "entry_role_aligned": len(entry_names),
        "endpoint_role_agnostic": int(endpoint_agnostic.sum()),
        "endpoint_role_aligned": len(endpoint_names),
    }
    return matrices, dimensions


def blend(entry: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    if entry.shape != endpoint.shape:
        raise ValueError("expert probabilities do not align")
    return (
        ENTRY_PROBABILITY_WEIGHT * entry
        + ENDPOINT_PROBABILITY_WEIGHT * endpoint
    )


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--endpoint-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument(
        "--entry-graph-weight", type=float, default=ENTRY_GRAPH_WEIGHT
    )
    args = parser.parse_args()
    if not args.seeds:
        raise SystemExit("at least one seed is required")
    if args.entry_graph_weight < 0.0:
        raise SystemExit("entry graph weight must be nonnegative")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "endpoint_cache": args.endpoint_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    entry_cache = read_cache(paths["entry_cache"])
    endpoint_cache = read_cache(paths["endpoint_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(entry_cache, feature_by_sha)
    attach_audit_metadata(endpoint_cache, feature_by_sha)

    selected_shas = {
        row["sha256"]
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    }
    missing = {
        "entry_cache": selected_shas - set(entry_cache),
        "endpoint_cache": selected_shas - set(endpoint_cache),
        "features": selected_shas - set(feature_by_sha),
    }
    if any(missing.values()):
        raise SystemExit(
            "missing selected rows: "
            + str({name: len(values) for name, values in missing.items()})
        )
    observed_years = sorted({
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    })
    development_only = observed_years == ["2017"]

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    per_variant = {name: [] for name in ALL_VARIANTS}
    predictions = []
    dimensions = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        eval_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        entry_evidence = fit_training_fold_components(
            entry_cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        endpoint_evidence = fit_training_fold_components(
            endpoint_cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, entry_cache, entry_evidence, entry_args, True
        )
        entry_eval, y_eval = graph_matrix(
            eval_rows, entry_cache, entry_evidence, entry_args, True
        )
        endpoint_train, endpoint_y_train = endpoint_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        endpoint_eval, endpoint_y_eval = endpoint_matrix(
            eval_rows, endpoint_cache, endpoint_evidence
        )
        if not (
            np.array_equal(y_train, endpoint_y_train)
            and np.array_equal(y_eval, endpoint_y_eval)
        ):
            raise SystemExit(f"cache label mismatch: {split_name}")
        compact_train, compact_eval, compact_names = compact_matrix_named(
            train_rows, eval_rows, feature_by_sha, schema
        )
        entry_train, entry_eval = standardize_pair(entry_train, entry_eval)
        endpoint_train, endpoint_eval = standardize_pair(
            endpoint_train, endpoint_eval
        )
        matrices, split_dimensions = build_matrices(
            entry_train,
            entry_eval,
            endpoint_train,
            endpoint_eval,
            compact_train,
            compact_eval,
            compact_names,
            args.entry_graph_weight,
        )
        split_dimensions["split_name"] = split_name
        dimensions.append(split_dimensions)

        seed_metrics = {name: [] for name in ALL_VARIANTS}
        for seed in args.seeds:
            probability = {}
            for variant in FIT_VARIANTS:
                x_train, x_eval = matrices[variant]
                probability[variant] = predict_logistic(
                    x_train,
                    y_train,
                    x_eval,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
            probability["role_agnostic_probability_blend"] = blend(
                probability["role_agnostic_entry_plus_compact"],
                probability["role_agnostic_endpoint_plus_compact"],
            )
            probability["role_aligned_probability_blend"] = blend(
                probability["role_aligned_entry_plus_compact"],
                probability["role_aligned_endpoint_plus_compact"],
            )
            for variant in ALL_VARIANTS:
                score = probability[variant]
                seed_metrics[variant].append(
                    metrics(y_eval, score, threshold=THRESHOLD)
                )
                for row, label, value in zip(eval_rows, y_eval, score):
                    predictions.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })

        for variant in ALL_VARIANTS:
            row = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(eval_rows),
                "threshold": THRESHOLD,
            }
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                row[f"{name}_mean"] = float(np.mean([
                    item[name] for item in seed_metrics[variant]
                ]))
            per_variant[variant].append(row)
        print(f"completed {split_name}", flush=True)

    summary_rows = []
    for variant in ALL_VARIANTS:
        rows = per_variant[variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        summary_rows.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_split_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })

    summary_path = output / "semantic_alignment_ablation_summary.csv"
    write_csv(summary_path, list(summary_rows[0]), summary_rows)
    per_split_path = output / "semantic_alignment_ablation_per_split.csv"
    per_split_rows = [
        row for variant in ALL_VARIANTS for row in per_variant[variant]
    ]
    write_csv(per_split_path, list(per_split_rows[0]), per_split_rows)
    prediction_path = output / "semantic_alignment_ablation_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        predictions,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P40_GLOBAL_LOCAL_SEMANTIC_ALIGNMENT_ABLATION",
        "formal_training_enabled": False,
        "protocol_status": (
            "DEVELOPMENT_2017_ONLY" if development_only
            else "RETROSPECTIVE_FUTURE_ABLATION_NOT_FOR_SELECTION"
        ),
        "observed_years": observed_years,
        "fixed_threshold": THRESHOLD,
        "entry_graph_weight": args.entry_graph_weight,
        "entry_probability_weight": ENTRY_PROBABILITY_WEIGHT,
        "endpoint_probability_weight": ENDPOINT_PROBABILITY_WEIGHT,
        "semantic_compact_definition": {
            "prefixes": [
                "role_node__", "role_edge__", "context_role__",
                "permission_behavior__",
            ],
            "exact_fields": sorted(SEMANTIC_COMPACT_EXACT),
        },
        "role_agnostic_graph_definition": (
            "same graph cache and generic topology/context fields; "
            "role-specific node, role-pool, and role-edge dimensions removed"
        ),
        "module_dimensions": dimensions,
        "aggregate_results": summary_rows,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(per_split_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "legacy_future_years_are_retrospective_if_present",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "All filtering, normalization, reliability, and fitting use training rows only.",
            "Role-agnostic graphs retain topology and generic sensitive indicators.",
            "This ablation tests role identity alignment, not graph removal.",
            "No APK, raw graph, raw path, or method signature is opened or persisted.",
        ],
    }
    (output / "semantic_alignment_ablation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in summary_rows:
        print(
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
