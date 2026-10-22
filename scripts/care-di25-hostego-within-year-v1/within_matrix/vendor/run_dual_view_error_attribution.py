#!/usr/bin/env python3
"""Attribute locked dual-view FP/FN decisions to interpretable features.

The script deterministically refits the locked P28 entry and endpoint logistic
experts from the 2017 partition, verifies every reconstructed probability
against the saved P28 prediction file, and computes exact integrated logistic
feature contributions relative to a zero-standardized-feature baseline.

No APK or raw call graph is opened.  Outputs are post-hoc diagnostics and must
not be used to tune the locked protocol on 2018--2021 labels.
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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_coverage_aware_eval import add_derived_features  # noqa: E402
from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    EDGE_TYPE_NAMES,
    ROLE_VOCABULARY_SIZE,
    attach_audit_metadata,
    endpoint_matrix,
    entry_model_args,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    AUDIT_COLUMNS,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
)
from run_pilot_compact_eval import fit_logistic  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_WEIGHT = 0.75
ENDPOINT_WEIGHT = 0.25
ENTRY_GRAPH_SCALE = 0.5
PREDICTION_VARIANT = "fixed_residual_blend_025"
ROLE_NAMES = sorted([
    "command_execution", "dynamic_code_loading", "native_library_loading",
    "reflective_invocation", "package_installation", "package_uninstallation",
    "device_administration", "accessibility_control", "sms_send_receive",
    "call_control_or_recording", "account_or_credential_access",
    "contacts_or_calendar_access", "device_identifier_access", "location_access",
    "microphone_access", "camera_access", "clipboard_access",
    "network_transmission_or_upload", "sensitive_file_or_content_write",
    "boot_persistence", "task_or_alarm_scheduling", "background_service_control",
    "overlay_or_ui_deception", "icon_or_component_hiding",
    "process_or_wakelock_control", "cryptography", "encoding_or_compression",
    "url_or_http_setup", "file_read", "database_access", "other",
])
if len(ROLE_NAMES) != ROLE_VOCABULARY_SIZE:
    raise RuntimeError("role-name vocabulary size mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def integrated_logistic_attribution(
    x: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact additive probability attribution along the zero-to-x path."""

    contribution = x * weights[None, :]
    delta_logit = contribution.sum(axis=1)
    baseline = float(sigmoid(float(bias)))
    probability = sigmoid(delta_logit + float(bias))
    factor = np.empty(len(x), dtype=np.float64)
    regular = np.abs(delta_logit) >= 1e-12
    factor[regular] = (probability[regular] - baseline) / delta_logit[regular]
    factor[~regular] = baseline * (1.0 - baseline)
    attribution = contribution * factor[:, None]
    if not np.allclose(
        attribution.sum(axis=1) + baseline, probability, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("integrated logistic attributions do not reconstruct probability")
    return attribution, probability, baseline


def standardize_named(
    x_fit: np.ndarray, x_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_fit - mean) / scale, (x_eval - mean) / scale


def compact_matrix_named(
    fit_rows: list[dict],
    eval_rows: list[dict],
    feature_by_sha: dict,
    schema: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base_names = [
        name for name in schema.get("feature_fields", []) if name not in AUDIT_COLUMNS
    ]
    fit_table = [
        add_derived_features(feature_by_sha[row["sha256"]], base_names, "base")
        for row in fit_rows
    ]
    eval_table = [
        add_derived_features(feature_by_sha[row["sha256"]], base_names, "base")
        for row in eval_rows
    ]
    candidates = sorted(fit_table[0]) if fit_table else []
    names = []
    for name in candidates:
        try:
            values = np.asarray([float(row[name]) for row in fit_table])
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)) and np.unique(values).size > 1:
            names.append(name)
    x_fit = np.asarray([
        [float(row[name]) for name in names] for row in fit_table
    ], dtype=np.float64)
    x_eval = np.asarray([
        [float(row[name]) for name in names] for row in eval_table
    ], dtype=np.float64)
    x_fit, x_eval = standardize_named(x_fit, x_eval)
    return x_fit, x_eval, names


def entry_feature_names() -> list[str]:
    node_names = (
        ["external", "mapped_sensitive", "tier_scaled", "manifest_entry"]
        + [f"role__{role}" for role in ROLE_NAMES]
        + ["reliability"]
    )
    names = []
    for statistic in ("mean", "max", "std"):
        names.extend([f"node_{statistic}__{name}" for name in node_names])
    names.extend([
        "edge_mean__icc", "edge_mean__reflection", "edge_mean__non_direct",
        "graph__log_node_count", "graph__log_edge_count",
    ])
    role_fields = (
        "support_log", "external_mean", "mapped_sensitive_mean",
        "tier_scaled_mean", "manifest_entry_mean", "reliability_mean",
    )
    for role in ROLE_NAMES:
        names.extend([f"role_pool__{role}__{field}" for field in role_fields])
    return names


def endpoint_feature_names() -> list[str]:
    base_names = [
        "external", "mapped_sensitive", "tier_scaled", "manifest_entry",
        "endpoint_seed", "provenance_package_aligned", "provenance_bundled",
        "provenance_external", "entry_reachable", "entry_unreachable",
        "reachability_unknown", "entry_distance_score", "upstream_context",
        "downstream_context", "endpoint_distance_score",
    ]
    names = []
    for statistic in ("mean", "max", "std"):
        names.extend([f"node_{statistic}__{name}" for name in base_names])
    names.extend([
        "edge_mean__icc", "edge_mean__reflection", "edge_mean__non_direct",
        "edge_mean__target_is_endpoint", "edge_mean__source_is_endpoint",
        "edge_mean__traversed_upstream", "edge_mean__traversed_downstream",
        "graph__log_node_count", "graph__log_edge_count",
        "graph__mean_reliability",
    ])
    role_fields = (
        "support_log", "endpoint_support_log", "entry_reachable_mean",
        "entry_unreachable_mean", "reachability_unknown_mean",
        "provenance_package_aligned_mean", "provenance_bundled_mean",
        "provenance_external_mean", "endpoint_distance_score_mean",
    )
    for role in ROLE_NAMES:
        names.extend([f"role_pool__{role}__{field}" for field in role_fields])
    role_edge_fields = (
        "outgoing_log", "incoming_log", "flagged_outgoing_ratio",
        "flagged_incoming_ratio",
    )
    for role in ROLE_NAMES:
        names.extend([f"role_edge__{role}__{field}" for field in role_edge_fields])
    for edge_type in EDGE_TYPE_NAMES:
        names.extend([
            f"edge_type__{edge_type}__count_log",
            f"edge_type__{edge_type}__ratio",
            f"edge_type__{edge_type}__flag_mean",
        ])
    return names


def attribution_family(feature: str) -> str:
    if feature.startswith("compact::"):
        return "compact"
    if feature.startswith("entry_graph::"):
        return "entry_graph"
    if "provenance_package_aligned" in feature:
        return "endpoint_package_aligned"
    if "provenance_bundled" in feature:
        return "endpoint_bundled"
    if "provenance_external" in feature or "::external" in feature:
        return "endpoint_external"
    if feature.startswith("endpoint_graph::"):
        return "endpoint_graph_other"
    return "other"


def read_saved_predictions(path: Path) -> dict[tuple[str, str, int], float]:
    result = {}
    for row in read_csv_rows(path):
        if row.get("variant") != PREDICTION_VARIANT:
            continue
        key = (
            row["split_name"], row["sha256"].casefold(), int(row["seed"])
        )
        if key in result:
            raise ValueError(f"duplicate saved prediction: {key}")
        result[key] = float(row["probability"])
    if not result:
        raise ValueError(f"no {PREDICTION_VARIANT} rows in saved predictions")
    return result


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
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", action="append", dest="split_names")
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("top-k must be positive")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "endpoint_cache": args.endpoint_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
        "predictions": args.predictions.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry_cache = read_cache(paths["entry_cache"])
    endpoint_cache = read_cache(paths["endpoint_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    if args.split_names:
        unknown = set(args.split_names) - set(split_rows)
        if unknown:
            raise SystemExit(f"unknown splits: {sorted(unknown)}")
        split_rows = {name: split_rows[name] for name in args.split_names}
    saved_predictions = read_saved_predictions(paths["predictions"])
    attach_audit_metadata(entry_cache, feature_by_sha)
    attach_audit_metadata(endpoint_cache, feature_by_sha)

    all_rows = [
        row for partitions in split_rows.values()
        for partition in ("train", "test") for row in partitions[partition]
    ]
    selected_shas = {row["sha256"] for row in all_rows}
    missing = {
        "entry_cache": len(selected_shas - set(entry_cache)),
        "endpoint_cache": len(selected_shas - set(endpoint_cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")
    for row in all_rows:
        feature = feature_by_sha[row["sha256"]]
        row["metadata_label"] = feature.get("metadata_label", "")
        row["metadata_year"] = feature.get("metadata_year", "")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    entry_graph_names = entry_feature_names()
    endpoint_graph_names = endpoint_feature_names()
    sample_rows = []
    top_rows = []
    prevalence_rows = []
    family_rows = []
    maximum_prediction_difference = 0.0
    reconstruction_errors = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        entry_evidence = fit_training_fold_components(
            entry_cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        endpoint_evidence = fit_training_fold_components(
            endpoint_cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_graph_train, y_train = graph_matrix(
            train_rows, entry_cache, entry_evidence, entry_args, True
        )
        entry_graph_test, y_test = graph_matrix(
            test_rows, entry_cache, entry_evidence, entry_args, True
        )
        endpoint_graph_train, endpoint_y_train = endpoint_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        endpoint_graph_test, endpoint_y_test = endpoint_matrix(
            test_rows, endpoint_cache, endpoint_evidence
        )
        if not (
            np.array_equal(y_train, endpoint_y_train)
            and np.array_equal(y_test, endpoint_y_test)
        ):
            raise SystemExit(f"cache label mismatch in {split_name}")
        if entry_graph_train.shape[1] != len(entry_graph_names):
            raise SystemExit(
                f"entry feature-name mismatch: matrix={entry_graph_train.shape[1]} names={len(entry_graph_names)}"
            )
        if endpoint_graph_train.shape[1] != len(endpoint_graph_names):
            raise SystemExit(
                f"endpoint feature-name mismatch: matrix={endpoint_graph_train.shape[1]} names={len(endpoint_graph_names)}"
            )
        entry_graph_train, entry_graph_test = standardize_named(
            entry_graph_train, entry_graph_test
        )
        endpoint_graph_train, endpoint_graph_test = standardize_named(
            endpoint_graph_train, endpoint_graph_test
        )
        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows, test_rows, feature_by_sha, schema
        )
        entry_x_train = np.concatenate([
            ENTRY_GRAPH_SCALE * entry_graph_train, compact_train
        ], axis=1)
        entry_x_test = np.concatenate([
            ENTRY_GRAPH_SCALE * entry_graph_test, compact_test
        ], axis=1)
        endpoint_x_train = np.concatenate([
            endpoint_graph_train, compact_train
        ], axis=1)
        endpoint_x_test = np.concatenate([
            endpoint_graph_test, compact_test
        ], axis=1)

        combined_names = (
            [f"entry_graph::{name}" for name in entry_graph_names]
            + [f"endpoint_graph::{name}" for name in endpoint_graph_names]
            + [f"compact::{name}" for name in compact_names]
        )
        seed_probabilities = []
        seed_attributions = []
        seed_baselines = []
        for seed in args.seeds:
            entry_weights, entry_bias = fit_logistic(
                entry_x_train, y_train, seed,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            endpoint_weights, endpoint_bias = fit_logistic(
                endpoint_x_train, y_train, seed,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            entry_attr, entry_probability, entry_baseline = (
                integrated_logistic_attribution(
                    entry_x_test, entry_weights, entry_bias
                )
            )
            endpoint_attr, endpoint_probability, endpoint_baseline = (
                integrated_logistic_attribution(
                    endpoint_x_test, endpoint_weights, endpoint_bias
                )
            )
            probability = (
                ENTRY_WEIGHT * entry_probability
                + ENDPOINT_WEIGHT * endpoint_probability
            )
            combined_attr = np.concatenate([
                ENTRY_WEIGHT * entry_attr[:, :len(entry_graph_names)],
                ENDPOINT_WEIGHT * endpoint_attr[:, :len(endpoint_graph_names)],
                (
                    ENTRY_WEIGHT * entry_attr[:, len(entry_graph_names):]
                    + ENDPOINT_WEIGHT * endpoint_attr[:, len(endpoint_graph_names):]
                ),
            ], axis=1)
            baseline = (
                ENTRY_WEIGHT * entry_baseline
                + ENDPOINT_WEIGHT * endpoint_baseline
            )
            if not np.allclose(
                combined_attr.sum(axis=1) + baseline,
                probability,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("combined attributions do not reconstruct blend")
            for row, value in zip(test_rows, probability):
                key = (split_name, row["sha256"], seed)
                saved = saved_predictions.get(key)
                if saved is None:
                    reconstruction_errors.append(f"missing saved prediction {key}")
                    continue
                maximum_prediction_difference = max(
                    maximum_prediction_difference, abs(float(value) - saved)
                )
            seed_probabilities.append(probability)
            seed_attributions.append(combined_attr)
            seed_baselines.append(baseline)

        mean_probability = np.mean(seed_probabilities, axis=0)
        mean_attribution = np.mean(seed_attributions, axis=0)
        mean_baseline = float(np.mean(seed_baselines))
        if not np.allclose(
            mean_attribution.sum(axis=1) + mean_baseline,
            mean_probability,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("mean attributions do not reconstruct mean probability")
        predicted = mean_probability >= THRESHOLD
        outcomes = np.where(
            (y_test == 1) & predicted, "TP",
            np.where(
                (y_test == 0) & (~predicted), "TN",
                np.where((y_test == 0) & predicted, "FP", "FN"),
            ),
        )
        families = [attribution_family(name) for name in combined_names]
        family_names = sorted(set(families))
        family_indices = {
            family: np.asarray([
                index for index, value in enumerate(families) if value == family
            ], dtype=np.int64)
            for family in family_names
        }

        for sample_index, (row, label, probability, outcome) in enumerate(
            zip(test_rows, y_test, mean_probability, outcomes)
        ):
            contribution = mean_attribution[sample_index]
            direction = 1.0 if outcome == "FP" else -1.0
            driving = direction * contribution
            top_indices = np.argsort(-driving)[:args.top_k] if outcome in {"FP", "FN"} else []
            top_features = [combined_names[index] for index in top_indices if driving[index] > 0.0]
            sample = {
                "split_name": split_name,
                "year": feature_by_sha[row["sha256"]].get("metadata_year", ""),
                "sha256": row["sha256"],
                "label": int(label),
                "outcome": outcome,
                "mean_probability": float(probability),
                "mean_baseline_probability": mean_baseline,
                "top_error_driving_features": "|".join(top_features),
            }
            for family in family_names:
                sample[f"contribution__{family}"] = float(
                    contribution[family_indices[family]].sum()
                )
            sample_rows.append(sample)
            if outcome in {"FP", "FN"}:
                for rank, index in enumerate(top_indices, 1):
                    if driving[index] <= 0.0:
                        continue
                    top_rows.append({
                        "split_name": split_name,
                        "year": sample["year"],
                        "sha256": row["sha256"],
                        "outcome": outcome,
                        "rank": rank,
                        "feature": combined_names[index],
                        "family": families[index],
                        "contribution": float(contribution[index]),
                        "error_direction_contribution": float(driving[index]),
                    })

        for error_outcome, reference_outcome in (("FP", "TN"), ("FN", "TP")):
            error_mask = outcomes == error_outcome
            reference_mask = outcomes == reference_outcome
            error_count = int(error_mask.sum())
            if not error_count or not np.any(reference_mask):
                continue
            direction = 1.0 if error_outcome == "FP" else -1.0
            top_counter = Counter()
            for contribution in mean_attribution[error_mask]:
                driving = direction * contribution
                for index in np.argsort(-driving)[:args.top_k]:
                    if driving[index] > 0.0:
                        top_counter[int(index)] += 1
            error_mean = mean_attribution[error_mask].mean(axis=0)
            reference_mean = mean_attribution[reference_mask].mean(axis=0)
            for index, count in top_counter.most_common():
                share = count / error_count
                prevalence_rows.append({
                    "split_name": split_name,
                    "year": feature_by_sha[test_rows[0]["sha256"]].get("metadata_year", ""),
                    "comparison": f"{error_outcome}_VS_{reference_outcome}",
                    "feature": combined_names[index],
                    "family": families[index],
                    "error_count": error_count,
                    "top_k_occurrence_count": count,
                    "top_k_error_share": share,
                    "majority_error_driver": share >= 0.5,
                    "mean_error_contribution": float(error_mean[index]),
                    "mean_reference_contribution": float(reference_mean[index]),
                    "mean_contribution_delta": float(error_mean[index] - reference_mean[index]),
                })
            for family in family_names:
                indices = family_indices[family]
                error_values = mean_attribution[error_mask][:, indices].sum(axis=1)
                reference_values = mean_attribution[reference_mask][:, indices].sum(axis=1)
                family_rows.append({
                    "split_name": split_name,
                    "year": feature_by_sha[test_rows[0]["sha256"]].get("metadata_year", ""),
                    "comparison": f"{error_outcome}_VS_{reference_outcome}",
                    "family": family,
                    "error_count": error_count,
                    "reference_count": int(reference_mask.sum()),
                    "mean_error_contribution": float(error_values.mean()),
                    "mean_reference_contribution": float(reference_values.mean()),
                    "mean_contribution_delta": float(error_values.mean() - reference_values.mean()),
                    "positive_error_rate": float(np.mean(direction * error_values > 0.0)),
                })
        print(f"completed {split_name}", flush=True)

    if reconstruction_errors:
        raise SystemExit(reconstruction_errors[0])
    if maximum_prediction_difference > 1e-8:
        raise SystemExit(
            f"reconstructed predictions differ from P28: max_abs={maximum_prediction_difference}"
        )

    sample_path = output / "sample_attributions.csv"
    sample_fields = sorted({key for row in sample_rows for key in row})
    write_csv(sample_path, sample_fields, sample_rows)
    top_path = output / "top_error_feature_attributions.csv"
    write_csv(
        top_path,
        [
            "split_name", "year", "sha256", "outcome", "rank", "feature",
            "family", "contribution", "error_direction_contribution",
        ],
        top_rows,
    )
    prevalence_path = output / "error_feature_prevalence.csv"
    write_csv(
        prevalence_path,
        [
            "split_name", "year", "comparison", "feature", "family",
            "error_count", "top_k_occurrence_count", "top_k_error_share",
            "majority_error_driver", "mean_error_contribution",
            "mean_reference_contribution", "mean_contribution_delta",
        ],
        prevalence_rows,
    )
    family_path = output / "error_family_attribution.csv"
    write_csv(
        family_path,
        [
            "split_name", "year", "comparison", "family", "error_count",
            "reference_count", "mean_error_contribution",
            "mean_reference_contribution", "mean_contribution_delta",
            "positive_error_rate",
        ],
        family_rows,
    )
    majority = [row for row in prevalence_rows if row["majority_error_driver"]]
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P32_LOCKED_DUAL_VIEW_POST_HOC_FEATURE_ATTRIBUTION",
        "formal_training_enabled": False,
        "post_hoc_test_label_analysis": True,
        "locked_protocol_modified": False,
        "attribution_method": "exact_integrated_logistic_probability_contribution_from_zero_standardized_baseline",
        "entry_probability_weight": ENTRY_WEIGHT,
        "endpoint_probability_weight": ENDPOINT_WEIGHT,
        "fixed_threshold": THRESHOLD,
        "top_k": args.top_k,
        "maximum_reconstructed_prediction_absolute_difference": maximum_prediction_difference,
        "sample_count": len(sample_rows),
        "error_sample_count": sum(row["outcome"] in {"FP", "FN"} for row in sample_rows),
        "majority_driver_row_count": len(majority),
        "majority_driver_rows": majority,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "samples": str(sample_path),
            "top_error_features": str(top_path),
            "feature_prevalence": str(prevalence_path),
            "family_attribution": str(family_path),
        },
        "blocking_reasons": [
            "post_hoc_future_test_labels_must_not_be_used_for_model_selection",
            "provenance_is_package_prefix_heuristic_not_an_audited_sdk_catalog",
        ],
        "notes": [
            "Attributions reconstruct the saved P28 probability exactly within tolerance.",
            "A top-k prevalence share answers whether the same feature drives many errors.",
            "Attribution indicates model influence, not real-world causal behavior.",
            "No APK, raw graph, raw path, or method signature is read or persisted.",
        ],
    }
    audit_path = output / "dual_view_error_attribution_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_count": audit["sample_count"],
        "error_sample_count": audit["error_sample_count"],
        "majority_driver_row_count": audit["majority_driver_row_count"],
        "maximum_prediction_difference": maximum_prediction_difference,
        "outputs": audit["outputs"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
