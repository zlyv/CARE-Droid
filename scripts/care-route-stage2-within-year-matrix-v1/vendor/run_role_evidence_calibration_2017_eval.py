#!/usr/bin/env python3
"""Evaluate role-level capability/activation evidence calibration.

This module is deliberately different from endpoint/path branches.  It uses
the existing entry-view cache and represents each semantic role by compact
evidence states: global presence, entry reachability, internal/external
boundary exposure, and reliability-weighted support.  It does not preserve
method names or paths and never uses labels to construct the role evidence.
"""

from __future__ import annotations

import argparse
import csv
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
    ROLE_VOCABULARY_SIZE,
    attach_audit_metadata,
    entry_model_args,
    sha256_file,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from run_uncertainty_gated_dual_view_2017_eval import predict_logistic  # noqa: E402
from tasa_droid.p3.a3_mvp import compute_reliability  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_GRAPH_WEIGHT = 0.5
ROLE_EVIDENCE_WIDTH = 10
GLOBAL_EVIDENCE_WIDTH = 15
ROLE_EVIDENCE_DIMENSION = ROLE_VOCABULARY_SIZE * ROLE_EVIDENCE_WIDTH + GLOBAL_EVIDENCE_WIDTH
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
    raise RuntimeError("semantic role vocabulary mismatch")


def _role_ids(values) -> list[int]:
    return sorted({
        int(value) for value in (values or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    })


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def role_evidence_vector(
    payload: dict,
    compact_row: dict[str, str] | None = None,
    reliability_by_node_id: dict[str, dict[str, float]] | None = None,
    reliability_weights: dict[str, float] | None = None,
    reliability_prior: dict[str, float] | None = None,
    r_min: float = 0.10,
    r_max: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Build role-level global/activation evidence without labels.

    Global capability counts come from the already extracted compact feature
    row.  Entry activation counts come from the bounded entry-view cache.
    Keeping these sources separate is the central purpose of this module.
    """

    node_ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    roles = list(payload.get("node_role_ids") or [])
    edge_index = payload.get("edge_index") or [[], []]
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    edge_features = np.asarray(payload.get("edge_features") or [], dtype=np.float64)
    n_nodes = len(node_ids)
    if n_nodes == 0:
        return np.zeros(ROLE_EVIDENCE_DIMENSION, dtype=np.float64), {
            "node_count": 0, "role_count": 0, "entry_node_count": 0,
            "reachable_node_count": 0,
        }
    if base.shape != (n_nodes, 4) or len(roles) != n_nodes:
        raise ValueError("invalid cached node arrays")
    adjacency = [[] for _ in range(n_nodes)]
    for src, dst in zip(source, target):
        if 0 <= int(src) < n_nodes and 0 <= int(dst) < n_nodes:
            adjacency[int(src)].append(int(dst))
    entries = np.flatnonzero(base[:, 3] > 0.5).astype(int).tolist()
    reachable = set(entries)
    frontier = list(entries)
    for _ in range(4):
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    global_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    global_edge_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    reachable_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    internal_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    external_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    boundary_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    reliability_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    role_node_count = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)

    active_reliability = reliability_by_node_id or {}
    prior = reliability_prior or {"D": 0.0, "one_minus_F": 0.0}
    weights = reliability_weights or {name: 1.0 for name in ACTIVE_RELIABILITY_COMPONENTS}
    def row_value(name: str) -> float:
        if compact_row is None:
            return 0.0
        try:
            value = float(compact_row.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    for role, role_name in enumerate(ROLE_NAMES):
        global_support[role] = max(0.0, row_value(f"role_node__{role_name}"))
        global_edge_support[role] = max(0.0, row_value(f"role_edge__{role_name}"))

    for index, node_id in enumerate(node_ids):
        node_roles = _role_ids(roles[index])
        if not node_roles:
            continue
        is_reachable = index in reachable
        is_external = base[index, 0] > 0.5
        components = active_reliability.get(str(node_id), prior)
        reliability = compute_reliability(
            components, weights, r_min, r_max,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        for role in node_roles:
            global_support[role] += 1.0
            role_node_count[role] += 1.0
            if is_reachable:
                reachable_support[role] += 1.0
            if is_external:
                external_support[role] += 1.0
            else:
                internal_support[role] += 1.0
            if is_reachable and not is_external:
                boundary_support[role] += 1.0
            reliability_support[role] += reliability

    # Counts and ratios are kept in separate role slots so the classifier can
    # distinguish high support from high activation rather than multiplying
    # raw counts into a large sparse interaction table.
    role_parts = []
    for role in range(ROLE_VOCABULARY_SIZE):
        total = global_support[role]
        role_parts.extend([
            np.log1p(total),
            np.log1p(global_edge_support[role]),
            reachable_support[role] / max(total, 1.0),
            internal_support[role] / max(total, 1.0),
            external_support[role] / max(total, 1.0),
            boundary_support[role] / max(total, 1.0),
            np.log1p(reliability_support[role]),
            np.log1p(reachable_support[role]),
            np.log1p(external_support[role]),
            np.log1p(total - reachable_support[role]),
        ])

    total_nodes = float(n_nodes)
    global_parts = np.asarray([
        np.log1p(total_nodes),
        np.log1p(len(source)),
        np.log1p(len(entries)),
        np.log1p(len(reachable)),
        np.log1p(global_support.sum()),
        np.log1p(global_edge_support.sum()),
        np.log1p(reachable_support.sum()),
        reachable_support.sum() / max(global_support.sum(), 1.0),
        external_support.sum() / max(global_support.sum(), 1.0),
        boundary_support.sum() / max(global_support.sum(), 1.0),
        _entropy(global_support),
        _entropy(reachable_support),
        float(np.count_nonzero(global_support)),
        float(np.count_nonzero(reachable_support)),
        float(np.mean(base[:, 1] > 0.5)),
    ], dtype=np.float64)
    vector = np.concatenate([np.asarray(role_parts), global_parts])
    if vector.shape != (ROLE_EVIDENCE_DIMENSION,):
        raise ValueError("unexpected role evidence feature dimension")
    audit = {
        "node_count": n_nodes,
        "edge_count": int(len(source)),
        "role_count": int(np.count_nonzero(global_support)),
        "entry_node_count": len(entries),
        "reachable_node_count": len(reachable),
        "global_role_support": float(global_support.sum()),
        "global_role_edge_support": float(global_edge_support.sum()),
        "reachable_role_support": float(reachable_support.sum()),
        "reachable_role_ratio": float(
            reachable_support.sum() / max(global_support.sum(), 1.0)
        ),
        "external_role_ratio": float(
            external_support.sum() / max(global_support.sum(), 1.0)
        ),
    }
    return vector.astype(np.float64), audit


def evidence_matrix(rows, cache, feature_by_sha, evidence):
    vectors = []
    audits = []
    weights = {name: 1.0 for name in ACTIVE_RELIABILITY_COMPONENTS}
    for row in rows:
        vector, audit = role_evidence_vector(
            cache[row["sha256"]]["payload"],
            feature_by_sha[row["sha256"]],
            evidence.get("components_by_node_id", {}),
            weights,
            evidence.get("global_training_prior", {}),
        )
        audit["sha256"] = row["sha256"]
        vectors.append(vector)
        audits.append(audit)
    return np.asarray(vectors, dtype=np.float64), audits


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

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
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)
    selected = {
        row["sha256"] for groups in split_rows.values()
        for partition in ("train", "test") for row in groups[partition]
    }
    missing_cache = selected - set(cache)
    missing_features = selected - set(feature_by_sha)
    if missing_cache or missing_features:
        raise SystemExit(
            f"missing selected rows: cache={len(missing_cache)}, "
            f"features={len(missing_features)}"
        )
    observed_years = {str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected}
    if observed_years != {"2017"}:
        raise SystemExit(f"2017 development rows required; observed={sorted(observed_years)}")
    for groups in split_rows.values():
        for partition in ("train", "test"):
            for row in groups[partition]:
                row["metadata_label"] = feature_by_sha[row["sha256"]].get("metadata_label", "")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = (
        "compact_only",
        "compact_plus_entry",
        "compact_plus_role_evidence",
        "compact_plus_entry_plus_role_evidence",
    )
    split_results = {variant: [] for variant in variants}
    predictions = []
    audits = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(train_rows, cache, evidence, entry_args, True)
        entry_test, y_test = graph_matrix(test_rows, cache, evidence, entry_args, True)
        compact_train, compact_test, compact_count = compact_matrix(
            train_rows, test_rows, feature_by_sha, schema, feature_mode="base"
        )
        role_train, train_audit = evidence_matrix(
            train_rows, cache, feature_by_sha, evidence
        )
        role_test, test_audit = evidence_matrix(
            test_rows, cache, feature_by_sha, evidence
        )
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        role_train, role_test = standardize_pair(role_train, role_test)
        entry_train *= ENTRY_GRAPH_WEIGHT
        entry_test *= ENTRY_GRAPH_WEIGHT
        matrices = {
            "compact_only": (compact_train, compact_test),
            "compact_plus_entry": (
                np.concatenate([compact_train, entry_train], axis=1),
                np.concatenate([compact_test, entry_test], axis=1),
            ),
            "compact_plus_role_evidence": (
                np.concatenate([compact_train, role_train], axis=1),
                np.concatenate([compact_test, role_test], axis=1),
            ),
            "compact_plus_entry_plus_role_evidence": (
                np.concatenate([compact_train, entry_train, role_train], axis=1),
                np.concatenate([compact_test, entry_test, role_test], axis=1),
            ),
        }
        for audit in train_audit:
            audit.update({"split_name": split_name, "partition": "train"})
            audits.append(audit)
        for audit in test_audit:
            audit.update({"split_name": split_name, "partition": "test"})
            audits.append(audit)

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            for variant, (x_train, x_test) in matrices.items():
                probability = predict_logistic(
                    x_train, y_train, x_test, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                seed_metrics[variant].append(metrics(y_test, probability, threshold=THRESHOLD))
                for row, label, value in zip(test_rows, y_test, probability):
                    predictions.append({
                        "variant": variant, "split_name": split_name,
                        "sha256": row["sha256"], "seed": seed,
                        "label": int(label), "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })
        for variant in variants:
            values = seed_metrics[variant]
            split_results[variant].append({
                "variant": variant, "split_name": split_name,
                "train_count": len(train_rows), "test_count": len(test_rows),
                "compact_feature_count": int(compact_count),
                "role_evidence_feature_count": ROLE_EVIDENCE_DIMENSION,
                "entry_graph_weight": ENTRY_GRAPH_WEIGHT, "threshold": THRESHOLD,
                "accuracy_mean": float(np.mean([item["accuracy"] for item in values])),
                "balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in values])),
                "precision_mean": float(np.mean([item["precision"] for item in values])),
                "recall_mean": float(np.mean([item["recall"] for item in values])),
                "f1_mean": float(np.mean([item["f1"] for item in values])),
                "auc_mean": float(np.mean([item["auc"] for item in values])),
                "test_rows_used_for_training_or_selection": False,
            })
        print(f"completed {split_name}", flush=True)

    ranked = []
    for variant in variants:
        rows = split_results[variant]
        bas = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        ranked.append({
            "rank": 0, "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(bas.mean()),
            "minimum_split_balanced_accuracy": float(bas.min()),
            "std_balanced_accuracy": float(bas.std()),
            "stability_score": float(bas.mean() - bas.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    ranked.sort(key=lambda row: (-row["mean_balanced_accuracy"], -row["mean_auc"], row["variant"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "role_evidence_calibration_2017_summary.csv"
    per_split_path = output / "role_evidence_calibration_2017_per_split.csv"
    prediction_path = output / "role_evidence_calibration_2017_predictions.csv"
    audit_rows_path = output / "role_evidence_calibration_audit.csv"
    write_csv(summary_path, list(ranked[0]), ranked)
    write_csv(per_split_path, list(split_results[variants[0]][0]), [row for variant in variants for row in split_results[variant]])
    write_csv(prediction_path, list(predictions[0]), predictions)
    write_csv(audit_rows_path, list(audits[0]), audits)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P45_ROLE_LEVEL_CAPABILITY_ACTIVATION_EVIDENCE_CALIBRATION_2017",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "label_independent_role_evidence": True,
        "threshold": THRESHOLD,
        "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
        "role_evidence_feature_count": ROLE_EVIDENCE_DIMENSION,
        "role_evidence_width": ROLE_EVIDENCE_WIDTH,
        "global_evidence_width": GLOBAL_EVIDENCE_WIDTH,
        "seeds": list(args.seeds), "iterations": args.iterations,
        "learning_rate": args.learning_rate, "l2": args.l2,
        "cache": str(paths["cache"]), "cache_sha256": sha256_file(paths["cache"]),
        "features": str(paths["features"]), "features_sha256": sha256_file(paths["features"]),
        "splits": str(paths["splits"]), "splits_sha256": sha256_file(paths["splits"]),
        "variants": list(variants), "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path), "per_split": str(per_split_path),
            "predictions": str(prediction_path), "audit_rows": str(audit_rows_path),
        },
        "blocking_reasons": [
            "candidate_not_locked", "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Global role support comes from the extracted compact role_node/role_edge fields; entry activation support comes from the bounded entry cache.",
            "No method names, raw paths, labels, or test-derived feature statistics are used in role evidence construction.",
            "All standardization is fit separately on each chronological 2017 training partition.",
            "Classification threshold is fixed at 0.5.",
        ],
    }
    audit_path = output / "role_evidence_calibration_2017_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}', flush=True,
        )


if __name__ == "__main__":
    main()
