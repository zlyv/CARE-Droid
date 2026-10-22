#!/usr/bin/env python3
"""Evaluate coverage-aware and semantic-interaction compact features.

All derived features are computed from the already extracted compact feature
CSV.  Candidate filtering and standardization are fitted on each outer
training partition only.  No APK is reopened and no test value is used for
feature construction, filtering, or model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402


AUDIT_COLUMNS = {
    "absolute_path", "certificate_sha256", "error_message", "error_type",
    "lineage_id", "metadata_label", "metadata_year", "package_name",
    "sha256", "source_id", "status", "version_code",
}


INTERACTION_PAIRS = (
    ("network_transmission_or_upload", "boot_persistence"),
    ("network_transmission_or_upload", "dynamic_code_loading"),
    ("account_or_credential_access", "network_transmission_or_upload"),
    ("sms_send_receive", "network_transmission_or_upload"),
    ("location_access", "network_transmission_or_upload"),
    ("native_library_loading", "dynamic_code_loading"),
    ("command_execution", "dynamic_code_loading"),
    ("sensitive_file_or_content_write", "network_transmission_or_upload"),
)


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


def label_value(value: str) -> int:
    value = str(value).strip().casefold()
    if value in {"1", "malicious", "malware"}:
        return 1
    if value in {"0", "benign"}:
        return 0
    raise ValueError(f"unknown label: {value}")


def read_split_rows(path: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in read_csv_rows(path):
        split = row.get("split_name", "")
        partition = row.get("partition", "")
        sha256 = row.get("sha256", "").casefold()
        if not split or partition not in {"train", "test"} or not sha256:
            continue
        result.setdefault(split, {"train": [], "test": []})[partition].append(
            {"sha256": sha256, "label": row.get("metadata_label", "")}
        )
    return result


def safe_float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row.get(name, "0"))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def add_derived_features(
    row: dict[str, str], base_names: list[str], mode: str,
) -> dict[str, float]:
    values = {name: safe_float(row, name) for name in base_names}
    if mode == "base":
        return values

    graph_nodes = values.get("graph_node_count", 0.0)
    graph_edges = values.get("graph_edge_count", 0.0)
    external_nodes = values.get("external_node_count", 0.0)
    values.update(
        {
            "derived_sensitive_edge_density": values.get("sensitive_edge_count", 0.0) / (graph_edges + 1.0),
            "derived_sensitive_node_density": values.get("sensitive_node_count", 0.0) / (graph_nodes + 1.0),
            "derived_mapped_api_density": values.get("mapped_api_node_count", 0.0) / (graph_nodes + 1.0),
            "derived_unmapped_external_ratio": values.get("unmapped_external_node_count", 0.0) / (external_nodes + 1.0),
            "derived_external_node_ratio": external_nodes / (graph_nodes + 1.0),
            "derived_sensitive_edge_per_method": values.get("sensitive_edge_count", 0.0) / (values.get("method_count", 0.0) + 1.0),
            "derived_graph_edge_per_node": graph_edges / (graph_nodes + 1.0),
        }
    )
    if mode == "coverage":
        return values

    if mode in {"size_density_additive", "size_density_only"}:
        mapped_nodes = values.get("mapped_api_node_count", 0.0)
        sensitive_nodes = values.get("sensitive_node_count", 0.0)
        sensitive_edges = values.get("sensitive_edge_count", 0.0)
        component_count = values.get("manifest_component_count", 0.0)
        density_values: dict[str, float] = {}
        for name, value in values.items():
            if name.startswith("role_node__"):
                role = name[len("role_node__"):]
                density_values[f"derived_role_node_graph_density__{role}"] = (
                    value / (graph_nodes + 1.0)
                )
                density_values[f"derived_role_node_mapped_share__{role}"] = (
                    value / (mapped_nodes + 1.0)
                )
                density_values[f"derived_role_node_sensitive_share__{role}"] = (
                    value / (sensitive_nodes + 1.0)
                )
            elif name.startswith("role_edge__"):
                role = name[len("role_edge__"):]
                density_values[f"derived_role_edge_graph_density__{role}"] = (
                    value / (graph_edges + 1.0)
                )
                density_values[f"derived_role_edge_sensitive_share__{role}"] = (
                    value / (sensitive_edges + 1.0)
                )
        for component in ("activity", "service", "receiver", "provider"):
            name = f"manifest_{component}_count"
            if name in values:
                density_values[f"derived_manifest_{component}_share"] = (
                    values[name] / (component_count + 1.0)
                )
        values.update(density_values)
        if mode == "size_density_additive":
            return values

        # Remove absolute volume features while retaining ratios, SDK values,
        # parser indicators, context flags, and the density features above.
        absolute_names = {
            "method_count", "graph_node_count", "graph_edge_count",
            "internal_node_count", "internal_edge_count",
            "external_node_count", "external_edge_count",
            "mapped_api_node_count", "unmapped_external_node_count",
            "sensitive_node_count", "sensitive_edge_count",
            "manifest_permission_count", "manifest_permission_count_log",
            "manifest_activity_count", "manifest_service_count",
            "manifest_receiver_count", "manifest_provider_count",
            "manifest_component_count", "manifest_component_count_log",
        }
        return {
            name: value for name, value in values.items()
            if name not in absolute_names
            and not name.startswith("role_node__")
            and not name.startswith("role_edge__")
        }

    for prefix in ("role_node__", "role_edge__"):
        for left, right in INTERACTION_PAIRS:
            left_name = prefix + left
            right_name = prefix + right
            if left_name in values and right_name in values:
                values[f"derived_interaction__{prefix[:-2]}__{left}__x__{right}"] = (
                    values[left_name] * values[right_name]
                )
    return values


def build_matrix(
    rows: list[dict[str, str]], feature_by_sha: dict[str, dict[str, str]],
    base_names: list[str], mode: str, selected_names: list[str] | None = None,
):
    table = [
        add_derived_features(feature_by_sha[row["sha256"]], base_names, mode)
        for row in rows
    ]
    names = selected_names or sorted(table[0]) if table else []
    matrix = np.asarray([[float(item.get(name, 0.0)) for name in names] for item in table], dtype=float)
    labels = np.asarray([label_value(feature_by_sha[row["sha256"]]["metadata_label"]) for row in rows], dtype=int)
    return matrix, labels, names


def train_fold_features(
    train_rows: list[dict[str, str]], feature_by_sha: dict[str, dict[str, str]],
    base_names: list[str], mode: str,
):
    raw, labels, names = build_matrix(train_rows, feature_by_sha, base_names, mode)
    usable = []
    for index, name in enumerate(names):
        column = raw[:, index]
        if np.all(np.isfinite(column)) and np.unique(column).size > 1:
            usable.append(name)
    return usable


def standardize_pair(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    feature_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    split_path = args.splits.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv_rows(feature_path)
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    base_names = [
        name for name in schema.get("feature_fields", [])
        if name not in AUDIT_COLUMNS
    ]
    split_rows = read_split_rows(split_path)
    modes = (
        "base", "coverage", "coverage_interactions",
        "size_density_additive", "size_density_only",
    )
    result_rows = []
    summaries = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
        for row in train_rows + test_rows:
            if row["sha256"] not in feature_by_sha:
                raise SystemExit(f"missing feature row: {row['sha256']}")

        for mode in modes:
            selected_names = train_fold_features(train_rows, feature_by_sha, base_names, mode)
            x_train, y_train, _ = build_matrix(
                train_rows, feature_by_sha, base_names, mode, selected_names
            )
            x_test, y_test, _ = build_matrix(
                test_rows, feature_by_sha, base_names, mode, selected_names
            )
            x_train, x_test = standardize_pair(x_train, x_test)
            seed_metrics = []
            for seed in args.seeds:
                weights, bias = fit_logistic(
                    x_train, y_train, seed, iterations=1500,
                    learning_rate=0.05, l2=1e-3,
                )
                probability = 1.0 / (
                    1.0 + np.exp(-np.clip(x_test @ weights + bias, -40.0, 40.0))
                )
                metric = metrics(y_test, probability, threshold=args.threshold)
                seed_metrics.append(metric)
                for row, label, score in zip(test_rows, y_test, probability):
                    result_rows.append(
                        {
                            "split_name": split_name,
                            "feature_mode": mode,
                            "seed": seed,
                            "sha256": row["sha256"],
                            "label": int(label),
                            "probability": float(score),
                            "threshold": args.threshold,
                            "predicted_label": int(score >= args.threshold),
                        }
                    )
            summaries.append(
                {
                    "split_name": split_name,
                    "feature_mode": mode,
                    "train_count": len(train_rows),
                    "test_count": len(test_rows),
                    "feature_count_mean": len(selected_names),
                    "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_metrics])),
                    "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in seed_metrics])),
                    "precision_mean": float(np.mean([row["precision"] for row in seed_metrics])),
                    "recall_mean": float(np.mean([row["recall"] for row in seed_metrics])),
                    "f1_mean": float(np.mean([row["f1"] for row in seed_metrics])),
                    "auc_mean": float(np.mean([row["auc"] for row in seed_metrics])),
                    "test_rows_used_for_training_or_selection": False,
                }
            )

    result_path = output / "coverage_aware_predictions.csv"
    write_fields = [
        "split_name", "feature_mode", "seed", "sha256", "label",
        "probability", "threshold", "predicted_label",
    ]
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    summary_path = output / "coverage_aware_results.csv"
    summary_fields = list(summaries[0])
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_COVERAGE_AWARE_COMPACT_FEATURE_PILOT",
        "formal_training_enabled": False,
        "features": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "schema": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "feature_modes": list(modes),
        "interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
        "threshold": args.threshold,
        "seeds": list(args.seeds),
        "summaries": summaries,
        "outputs": {
            "results": str(summary_path),
            "predictions": str(result_path),
        },
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "new_features_are_pilot_not_formal_schema",
            "tif_per_apk_predictions_not_available",
        ],
        "notes": [
            "Derived features use only already extracted compact columns.",
            "Feature filtering and standardization are fit on each outer training partition only.",
            "No APK, raw graph, or test label is used during feature construction.",
            "This pilot must not be reported as the final formal result.",
        ],
    }
    audit_path = output / "coverage_aware_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
