#!/usr/bin/env python3
"""Evaluate compact endpoint topology motifs on chronological 2017 folds.

The P25 endpoint cache already stores every sensitive endpoint and a bounded
caller/callee neighbourhood.  This script derives label-independent semantic
motifs from that cache: role-conditioned one-hop caller provenance, two-hop
caller provenance, one-hop callee provenance, and shared-caller tier
co-occurrence.  APKs and raw call graphs are never opened.

Only 2017 development rows are accepted.  The final threshold is fixed at
0.5, and the previously selected endpoint residual weight remains 0.25.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    endpoint_summary_vector,
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
ENDPOINT_RESIDUAL_WEIGHT = 0.25
PROVENANCE_COUNT = 3
TIER_COUNT = 5
ROLE_MOTIF_WIDTH = PROVENANCE_COUNT * 3
GLOBAL_MOTIF_WIDTH = TIER_COUNT * TIER_COUNT + 9
MOTIF_DIMENSION = ROLE_VOCABULARY_SIZE * ROLE_MOTIF_WIDTH + GLOBAL_MOTIF_WIDTH


def _payload_arrays(payload: dict):
    node_ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    if not node_ids:
        base = np.zeros((0, 15), dtype=np.float64)
    if base.ndim != 2 or base.shape != (len(node_ids), 15):
        raise ValueError("endpoint node_base_features must have shape [N, 15]")
    roles = list(payload.get("node_role_ids") or [])
    if len(roles) != len(node_ids):
        raise ValueError("endpoint role rows do not align with nodes")
    edge_index = payload.get("edge_index") or [[], []]
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if source.shape != target.shape:
        raise ValueError("endpoint edge index source/target mismatch")
    if len(source) and (
        source.min() < 0 or target.min() < 0
        or source.max() >= len(node_ids) or target.max() >= len(node_ids)
    ):
        raise ValueError("endpoint edge index is out of bounds")
    return node_ids, base, roles, source, target


def _node_reliability(node_ids: list, evidence: dict) -> np.ndarray:
    values = np.zeros(len(node_ids), dtype=np.float64)
    weights = {name: 1.0 for name in ACTIVE_RELIABILITY_COMPONENTS}
    for index, node_id in enumerate(node_ids):
        components = evidence["components_by_node_id"].get(
            str(node_id), evidence["global_training_prior"]
        )
        values[index] = compute_reliability(
            components,
            weights,
            0.10,
            0.95,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
    return np.maximum(values, 0.10)


def endpoint_motif_vector(
    payload: dict,
    evidence: dict,
    role_vocabulary_size: int = ROLE_VOCABULARY_SIZE,
) -> np.ndarray:
    """Return fixed-size role/provenance/topology motifs for one APK."""

    node_ids, base, roles, source, target = _payload_arrays(payload)
    reliability = _node_reliability(node_ids, evidence)
    endpoint_mask = base[:, 4] > 0.5
    provenance = (
        np.argmax(base[:, 5:8], axis=1)
        if len(base) else np.zeros(0, dtype=np.int64)
    )
    tier = (
        np.clip(np.rint(base[:, 2] * 4.0), 0, 4).astype(np.int64)
        if len(base) else np.zeros(0, dtype=np.int64)
    )
    role_rows = []
    for row in roles:
        role_rows.append([
            int(value) for value in row
            if 0 <= int(value) < role_vocabulary_size
        ])

    incoming_edges: dict[int, list[int]] = defaultdict(list)
    outgoing_edges: dict[int, list[int]] = defaultdict(list)
    for edge_index, (src, dst) in enumerate(zip(source, target)):
        outgoing_edges[int(src)].append(edge_index)
        incoming_edges[int(dst)].append(edge_index)

    # Per role: incoming one-hop provenance (3), incoming two-hop
    # provenance (3), and outgoing one-hop provenance (3).
    role_motifs = np.zeros(
        (role_vocabulary_size, ROLE_MOTIF_WIDTH), dtype=np.float64
    )
    direct_caller_targets: dict[int, set[int]] = defaultdict(set)
    endpoint_has_package_caller = 0
    endpoint_count = int(endpoint_mask.sum())

    for endpoint in np.flatnonzero(endpoint_mask):
        endpoint = int(endpoint)
        endpoint_roles = role_rows[endpoint]
        if not endpoint_roles:
            continue
        endpoint_weight = reliability[endpoint]
        has_package_caller = False
        for incoming_index in incoming_edges.get(endpoint, []):
            caller = int(source[incoming_index])
            caller_provenance = int(provenance[caller])
            if caller_provenance == 0:
                has_package_caller = True
            direct_caller_targets[caller].add(endpoint)
            for role in endpoint_roles:
                role_motifs[role, caller_provenance] += endpoint_weight

            # The cache retains two upstream hops. Count grandparents once
            # per endpoint/caller path; this represents local invocation
            # context, not an assertion of runtime execution.
            for second_index in incoming_edges.get(caller, []):
                grandparent = int(source[second_index])
                grandparent_provenance = int(provenance[grandparent])
                for role in endpoint_roles:
                    role_motifs[role, 3 + grandparent_provenance] += endpoint_weight

        endpoint_has_package_caller += int(has_package_caller)
        for outgoing_index in outgoing_edges.get(endpoint, []):
            callee = int(target[outgoing_index])
            callee_provenance = int(provenance[callee])
            for role in endpoint_roles:
                role_motifs[role, 6 + callee_provenance] += endpoint_weight

    # A caller invoking endpoints from several sensitivity tiers gives a
    # compact second-order behaviour motif without storing method identity.
    tier_pairs = np.zeros((TIER_COUNT, TIER_COUNT), dtype=np.float64)
    fanouts = []
    for endpoints in direct_caller_targets.values():
        endpoint_list = sorted(endpoints)
        fanouts.append(len(endpoint_list))
        tiers = sorted({int(tier[index]) for index in endpoint_list})
        for left in tiers:
            for right in tiers:
                tier_pairs[left, right] += 1.0

    incoming_contexts = sorted(direct_caller_targets)
    provenance_ratios = np.zeros(PROVENANCE_COUNT, dtype=np.float64)
    if incoming_contexts:
        caller_provenance = provenance[incoming_contexts]
        for value in range(PROVENANCE_COUNT):
            provenance_ratios[value] = float(
                np.mean(caller_provenance == value)
            )
    branching = [value for value in fanouts if value >= 2]
    globals_vector = np.concatenate([
        np.log1p(tier_pairs).reshape(-1),
        np.asarray([
            float(np.log1p(len(incoming_contexts))),
            float(np.log1p(len(branching))),
            float(np.log1p(max(fanouts, default=0))),
            float(np.mean(fanouts)) if fanouts else 0.0,
            float(len(branching) / len(fanouts)) if fanouts else 0.0,
            *provenance_ratios.tolist(),
            (
                float(endpoint_has_package_caller / endpoint_count)
                if endpoint_count else 0.0
            ),
        ], dtype=np.float64),
    ])
    result = np.concatenate([np.log1p(role_motifs).reshape(-1), globals_vector])
    if result.shape != (role_vocabulary_size * ROLE_MOTIF_WIDTH + GLOBAL_MOTIF_WIDTH,):
        raise ValueError("unexpected endpoint motif dimension")
    return result


def endpoint_motif_matrix(rows: list[dict], cache: dict, evidence: dict):
    values = []
    for row in rows:
        payload = cache[row["sha256"]]["payload"]
        values.append(np.concatenate([
            endpoint_summary_vector(payload, evidence),
            endpoint_motif_vector(payload, evidence),
        ]))
    return np.asarray(values, dtype=np.float64)


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
        "--motif-l2-values",
        nargs="+",
        type=float,
        default=[0.03],
        help="Motif-expert L2 values compared on 2017 folds only; entry and baseline endpoint experts retain --l2.",
    )
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.motif_l2_values):
        raise SystemExit("motif L2 values must be positive")

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
    years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    }
    if years != {"2017"}:
        raise SystemExit(f"P29 accepts only 2017 development rows; observed={sorted(years)}")
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
    control_variants = (
        "entry_control",
        "fixed_residual_blend_025_control",
    )
    motif_specs = []
    for motif_l2 in args.motif_l2_values:
        suffix = f"l2_{motif_l2:g}"
        motif_specs.append({
            "l2": float(motif_l2),
            "expert_variant": f"endpoint_motif_expert__{suffix}",
            "blend_variant": f"motif_residual_blend_025__{suffix}",
        })
    variants = control_variants + tuple(
        name
        for spec in motif_specs
        for name in (spec["expert_variant"], spec["blend_variant"])
    )
    split_results = {variant: [] for variant in variants}
    prediction_rows = []
    feature_dimensions = {}

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
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
        endpoint_train = np.asarray([
            endpoint_summary_vector(
                endpoint_cache[row["sha256"]]["payload"], endpoint_evidence
            ) for row in train_rows
        ])
        endpoint_validation = np.asarray([
            endpoint_summary_vector(
                endpoint_cache[row["sha256"]]["payload"], endpoint_evidence
            ) for row in validation_rows
        ])
        motif_train = endpoint_motif_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        motif_validation = endpoint_motif_matrix(
            validation_rows, endpoint_cache, endpoint_evidence
        )
        compact_train, compact_validation, _ = compact_matrix(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        entry_train, entry_validation = standardize_pair(entry_train, entry_validation)
        endpoint_train, endpoint_validation = standardize_pair(
            endpoint_train, endpoint_validation
        )
        motif_train, motif_validation = standardize_pair(
            motif_train, motif_validation
        )
        entry_train = np.concatenate([
            ENTRY_GRAPH_WEIGHT * entry_train, compact_train
        ], axis=1)
        entry_validation = np.concatenate([
            ENTRY_GRAPH_WEIGHT * entry_validation, compact_validation
        ], axis=1)
        endpoint_train = np.concatenate([endpoint_train, compact_train], axis=1)
        endpoint_validation = np.concatenate([
            endpoint_validation, compact_validation
        ], axis=1)
        motif_train = np.concatenate([motif_train, compact_train], axis=1)
        motif_validation = np.concatenate([
            motif_validation, compact_validation
        ], axis=1)
        feature_dimensions = {
            "entry_expert": int(entry_train.shape[1]),
            "endpoint_expert": int(endpoint_train.shape[1]),
            "endpoint_motif_expert": int(motif_train.shape[1]),
            "motif_only_addition": MOTIF_DIMENSION,
        }

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            entry_probability = predict_logistic(
                entry_train, y_train, entry_validation, seed,
                args.iterations, args.learning_rate, args.l2,
            )
            endpoint_probability = predict_logistic(
                endpoint_train, y_train, endpoint_validation, seed,
                args.iterations, args.learning_rate, args.l2,
            )
            scores = {
                "entry_control": entry_probability,
                "fixed_residual_blend_025_control": (
                    0.75 * entry_probability + 0.25 * endpoint_probability
                ),
            }
            for spec in motif_specs:
                motif_probability = predict_logistic(
                    motif_train, y_train, motif_validation, seed,
                    args.iterations, args.learning_rate, spec["l2"],
                )
                scores[spec["expert_variant"]] = motif_probability
                scores[spec["blend_variant"]] = (
                    0.75 * entry_probability + 0.25 * motif_probability
                )
            for variant, probability in scores.items():
                seed_metrics[variant].append(
                    metrics(y_validation, probability, threshold=THRESHOLD)
                )
                for row, label, value in zip(
                    validation_rows, y_validation, probability
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })
        for variant in variants:
            summary = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "threshold": THRESHOLD,
            }
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{name}_mean"] = float(np.mean([
                    row[name] for row in seed_metrics[variant]
                ]))
            split_results[variant].append(summary)
        print(f"completed {split_name}", flush=True)

    ranked = []
    for variant in variants:
        rows = split_results[variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        ranked.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
            "split_summaries": rows,
        })
    ranked.sort(key=lambda row: (
        -row["stability_score"], -row["mean_balanced_accuracy"], row["variant"]
    ))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "endpoint_motif_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, ranked)
    prediction_path = output / "endpoint_motif_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P29_ENDPOINT_ROLE_CONDITIONED_TOPOLOGY_MOTIF_2017_DEVELOPMENT",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "fixed_endpoint_residual_weight": ENDPOINT_RESIDUAL_WEIGHT,
        "residual_weight_tuning_performed": False,
        "baseline_expert_l2": args.l2,
        "motif_l2_values_compared_on_2017_only": args.motif_l2_values,
        "motif_dimension": MOTIF_DIMENSION,
        "motif_definition": {
            "per_role": [
                "one_hop_caller_provenance_3",
                "two_hop_caller_provenance_3",
                "one_hop_callee_provenance_3",
            ],
            "global": [
                "shared_caller_endpoint_tier_pairs_25",
                "caller_branching_provenance_and_package_coverage_9",
            ],
        },
        "feature_dimensions": feature_dimensions,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "splits": str(paths["splits"]),
        "splits_sha256": sha256_file(paths["splits"]),
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "motif_candidate_not_locked",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted.",
            "Motifs are derived from the existing bounded endpoint cache without opening APKs.",
            "Reliability and standardization are fitted on each training partition only.",
            "No 2018--2021 row is read or evaluated.",
        ],
    }
    audit_path = output / "endpoint_motif_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
