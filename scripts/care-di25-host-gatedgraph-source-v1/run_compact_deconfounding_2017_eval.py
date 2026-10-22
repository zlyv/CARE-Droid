#!/usr/bin/env python3
"""Evaluate compact-feature deconfounding candidates on 2017 rolling folds.

The locked dual-view architecture is retained: an entry graph expert and an
endpoint graph expert each receive the same compact feature view, and their
probabilities are blended as 0.75 * entry + 0.25 * endpoint.  Only the compact
view changes between candidates.  The threshold is always 0.5.

This development evaluator accepts only 2017 rows.  It never opens an APK and
never reads a future-year split.  Candidate selection must therefore be made
from the chronological 2017 folds emitted by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    endpoint_matrix,
    entry_model_args,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    AUDIT_COLUMNS,
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
ENTRY_WEIGHT = 0.75
ENDPOINT_WEIGHT = 0.25

VARIANTS = (
    "control",
    "deduplicated_scale",
    "deduplicated_no_coverage",
    "context_additive",
    "context_replacement",
)

RAW_SCALE_FIELDS = {
    "method_count",
    "graph_node_count",
    "graph_edge_count",
    "internal_node_count",
    "internal_edge_count",
    "external_node_count",
    "external_edge_count",
    "mapped_api_node_count",
    "unmapped_external_node_count",
    "sensitive_node_count",
    "sensitive_edge_count",
}

COVERAGE_SHORTCUT_FIELDS = {
    "mapped_external_ratio",
}

CONTEXTUALIZED_ROLES = {
    "command_execution",
    "encoding_or_compression",
}

CONTEXTUALIZED_DECLARATIONS = {
    "permission_behavior__overlay__declared",
    "permission_behavior__install__declared",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(row: dict, name: str) -> float:
    try:
        value = float(row.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _log_count(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def _role_value(values: dict[str, float], prefix: str, role: str) -> float:
    return _log_count(values.get(f"{prefix}__{role}", 0.0))


def _cooccurrence(left: float, right: float) -> float:
    """Bounded AND-like evidence: both signals must be present."""

    return min(max(float(left), 0.0), max(float(right), 0.0))


def context_features(values: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    pairs = (
        ("command_execution", "dynamic_code_loading"),
        ("command_execution", "sensitive_file_or_content_write"),
        ("command_execution", "network_transmission_or_upload"),
        ("encoding_or_compression", "dynamic_code_loading"),
        ("encoding_or_compression", "sensitive_file_or_content_write"),
        ("encoding_or_compression", "network_transmission_or_upload"),
    )
    for prefix in ("role_node", "role_edge"):
        for left, right in pairs:
            result[f"deconfounded_context__{prefix}__{left}__and__{right}"] = (
                _cooccurrence(
                    _role_value(values, prefix, left),
                    _role_value(values, prefix, right),
                )
            )

    overlay_declared = float(
        values.get("permission_behavior__overlay__declared", 0.0) > 0.0
    )
    install_declared = float(
        values.get("permission_behavior__install__declared", 0.0) > 0.0
    )
    result["deconfounded_context__overlay_declared_and_api"] = (
        overlay_declared
        * float(
            values.get("role_node__overlay_or_ui_deception", 0.0) > 0.0
            or values.get("role_edge__overlay_or_ui_deception", 0.0) > 0.0
        )
    )
    result["deconfounded_context__install_declared_and_api"] = (
        install_declared
        * float(
            values.get("role_node__package_installation", 0.0) > 0.0
            or values.get("role_edge__package_installation", 0.0) > 0.0
        )
    )
    return result


def compact_variant_values(
    row: dict, base_names: list[str], variant: str,
) -> dict[str, float]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown compact variant: {variant}")
    original = {name: safe_float(row, name) for name in base_names}
    if variant == "control":
        return original

    values = {
        name: value
        for name, value in original.items()
        if name not in RAW_SCALE_FIELDS
    }

    # Replace many collinear absolute graph counts with one size term and a
    # small set of bounded ratios.  Role volumes are log-transformed so that a
    # large library cannot vote repeatedly merely because it has more methods.
    values["deconfounded_log_method_count"] = _log_count(
        original.get("method_count", 0.0)
    )
    values["deconfounded_graph_edge_per_node"] = (
        original.get("graph_edge_count", 0.0)
        / (original.get("graph_node_count", 0.0) + 1.0)
    )
    values["deconfounded_sensitive_node_share"] = (
        original.get("sensitive_node_count", 0.0)
        / (original.get("graph_node_count", 0.0) + 1.0)
    )
    values["deconfounded_sensitive_edge_share"] = (
        original.get("sensitive_edge_count", 0.0)
        / (original.get("graph_edge_count", 0.0) + 1.0)
    )
    values["deconfounded_external_node_share"] = (
        original.get("external_node_count", 0.0)
        / (original.get("graph_node_count", 0.0) + 1.0)
    )
    values["deconfounded_unmapped_external_share"] = (
        original.get("unmapped_external_node_count", 0.0)
        / (original.get("external_node_count", 0.0) + 1.0)
    )
    for name in list(values):
        if name.startswith("role_node__") or name.startswith("role_edge__"):
            values[name] = _log_count(values[name])

    if variant in {
        "deduplicated_no_coverage", "context_additive", "context_replacement"
    }:
        for name in COVERAGE_SHORTCUT_FIELDS:
            values.pop(name, None)
        values.pop("deconfounded_external_node_share", None)
        values.pop("deconfounded_unmapped_external_share", None)

    if variant in {"context_additive", "context_replacement"}:
        values.update(context_features(original))

    if variant == "context_replacement":
        for prefix in ("role_node", "role_edge"):
            for role in CONTEXTUALIZED_ROLES:
                values.pop(f"{prefix}__{role}", None)
        for name in CONTEXTUALIZED_DECLARATIONS:
            values.pop(name, None)
    return values


def compact_variant_matrix(
    train_rows: list[dict],
    eval_rows: list[dict],
    feature_by_sha: dict[str, dict],
    schema: dict,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base_names = [
        name
        for name in schema.get("feature_fields", [])
        if name not in AUDIT_COLUMNS
    ]
    train_table = [
        compact_variant_values(feature_by_sha[row["sha256"]], base_names, variant)
        for row in train_rows
    ]
    eval_table = [
        compact_variant_values(feature_by_sha[row["sha256"]], base_names, variant)
        for row in eval_rows
    ]
    candidates = sorted(train_table[0]) if train_table else []
    names = []
    for name in candidates:
        column = np.asarray([item.get(name, 0.0) for item in train_table], dtype=float)
        if np.all(np.isfinite(column)) and np.unique(column).size > 1:
            names.append(name)
    x_train = np.asarray(
        [[item.get(name, 0.0) for name in names] for item in train_table],
        dtype=float,
    )
    x_eval = np.asarray(
        [[item.get(name, 0.0) for name in names] for item in eval_table],
        dtype=float,
    )
    if not names:
        raise ValueError(f"{variant}: no usable compact feature")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_train - mean) / scale, (x_eval - mean) / scale, names


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
    return 1.0 / (
        1.0 + np.exp(-np.clip(x_eval @ weights + bias, -40.0, 40.0))
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
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()
    variants = list(dict.fromkeys(args.variants))
    if not args.seeds:
        raise SystemExit("at least one seed is required")

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
    observed_years = {
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    }
    if observed_years != {"2017"}:
        raise SystemExit(
            "P33 accepts only 2017 development rows; "
            f"observed={sorted(observed_years)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    per_variant_fold: dict[str, list[dict]] = {name: [] for name in variants}
    feature_counts: dict[str, Counter] = {name: Counter() for name in variants}
    selected_feature_names: dict[str, dict[str, list[str]]] = {
        name: {} for name in variants
    }
    prediction_rows: list[dict] = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        if not train_rows or not validation_rows:
            raise SystemExit(f"empty development partition: {split_name}")
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
        entry_validation, y_validation = graph_matrix(
            validation_rows, entry_cache, entry_evidence, entry_args, True
        )
        endpoint_train, endpoint_y_train = endpoint_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        endpoint_validation, endpoint_y_validation = endpoint_matrix(
            validation_rows, endpoint_cache, endpoint_evidence
        )
        if not (
            np.array_equal(y_train, endpoint_y_train)
            and np.array_equal(y_validation, endpoint_y_validation)
        ):
            raise SystemExit(f"cache label mismatch: {split_name}")
        entry_train, entry_validation = standardize_pair(
            entry_train, entry_validation
        )
        endpoint_train, endpoint_validation = standardize_pair(
            endpoint_train, endpoint_validation
        )

        for variant in variants:
            compact_train, compact_validation, names = compact_variant_matrix(
                train_rows,
                validation_rows,
                feature_by_sha,
                schema,
                variant,
            )
            selected_feature_names[variant][split_name] = names
            feature_counts[variant][len(names)] += 1
            entry_expert_train = np.concatenate(
                [0.5 * entry_train, compact_train], axis=1
            )
            entry_expert_validation = np.concatenate(
                [0.5 * entry_validation, compact_validation], axis=1
            )
            endpoint_expert_train = np.concatenate(
                [endpoint_train, compact_train], axis=1
            )
            endpoint_expert_validation = np.concatenate(
                [endpoint_validation, compact_validation], axis=1
            )
            seed_metrics = []
            for seed in args.seeds:
                entry_probability = predict_logistic(
                    entry_expert_train,
                    y_train,
                    entry_expert_validation,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                endpoint_probability = predict_logistic(
                    endpoint_expert_train,
                    y_train,
                    endpoint_expert_validation,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                probability = (
                    ENTRY_WEIGHT * entry_probability
                    + ENDPOINT_WEIGHT * endpoint_probability
                )
                seed_metrics.append(
                    metrics(y_validation, probability, threshold=THRESHOLD)
                )
                for row, label, score in zip(
                    validation_rows, y_validation, probability
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(score),
                        "threshold": THRESHOLD,
                        "predicted_label": int(score >= THRESHOLD),
                    })
            fold = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "compact_feature_count": len(names),
                "entry_expert_feature_count": int(entry_expert_train.shape[1]),
                "endpoint_expert_feature_count": int(endpoint_expert_train.shape[1]),
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                fold[f"{metric_name}_mean"] = float(np.mean([
                    item[metric_name] for item in seed_metrics
                ]))
            per_variant_fold[variant].append(fold)
        print(f"completed {split_name}", flush=True)

    summaries = []
    for variant in variants:
        folds = per_variant_fold[variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in folds])
        summaries.append({
            "variant": variant,
            "compact_feature_counts": sorted(feature_counts[variant]),
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in folds])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in folds])),
            "fold_summaries": folds,
        })
    summaries.sort(key=lambda row: (
        -row["stability_score"],
        -row["mean_balanced_accuracy"],
        row["variant"],
    ))
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank

    summary_path = output / "compact_deconfounding_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "compact_feature_counts", "mean_accuracy",
        "mean_balanced_accuracy", "minimum_fold_balanced_accuracy",
        "std_balanced_accuracy", "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, summaries)
    fold_path = output / "compact_deconfounding_2017_per_fold.csv"
    fold_fields = [
        "variant", "split_name", "train_count", "validation_count",
        "compact_feature_count", "entry_expert_feature_count",
        "endpoint_expert_feature_count", "threshold", "accuracy_mean",
        "balanced_accuracy_mean", "precision_mean", "recall_mean",
        "f1_mean", "auc_mean",
    ]
    write_csv(
        fold_path,
        fold_fields,
        [row for variant in variants for row in per_variant_fold[variant]],
    )
    prediction_path = output / "compact_deconfounding_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    control = next((row for row in summaries if row["variant"] == "control"), None)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P33_2017_ROLLING_COMPACT_DECONFOUNDING_ABLATION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(observed_years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_probability_weight": ENTRY_WEIGHT,
        "endpoint_probability_weight": ENDPOINT_WEIGHT,
        "variants": variants,
        "variant_definitions": {
            "control": "exact current compact feature view",
            "deduplicated_scale": "replace correlated raw graph volumes with one log size and bounded shares; log-transform role counts",
            "deduplicated_no_coverage": "deduplicated_scale without mapped/external coverage shares",
            "context_additive": "deduplicated_no_coverage plus label-free semantic co-occurrence features",
            "context_replacement": "context_additive with standalone command/compression and overlay/install declaration shortcuts removed",
        },
        "raw_scale_fields": sorted(RAW_SCALE_FIELDS),
        "coverage_shortcut_fields": sorted(COVERAGE_SHORTCUT_FIELDS),
        "contextualized_roles": sorted(CONTEXTUALIZED_ROLES),
        "contextualized_declarations": sorted(CONTEXTUALIZED_DECLARATIONS),
        "classifier": {
            "type": "two_independent_custom_l2_logistic_experts",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "ranked_variants": summaries,
        "selected_development_candidate": summaries[0]["variant"],
        "control_reproduction": {
            "observed_mean_balanced_accuracy": (
                control["mean_balanced_accuracy"] if control else None
            ),
            "target_mean_balanced_accuracy": 0.9213554444747093,
            "absolute_difference": (
                abs(control["mean_balanced_accuracy"] - 0.9213554444747093)
                if control else None
            ),
        },
        "selected_compact_features_by_fold": selected_feature_names,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_fold": str(fold_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "candidate_not_locked",
            "future_year_post_hoc_error_attributions_are_hypothesis_generation_only",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted.",
            "Every fold fits reliability, feature filtering, normalization, and model weights on its training partition only.",
            "No APK or raw graph is opened or persisted.",
            "The fixed 0.5 threshold and 0.75/0.25 expert probability blend are unchanged.",
            "P32 future-year labels motivate the candidate family but never enter this evaluator.",
        ],
    }
    audit_path = output / "compact_deconfounding_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in summaries:
        print(
            row["rank"],
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
