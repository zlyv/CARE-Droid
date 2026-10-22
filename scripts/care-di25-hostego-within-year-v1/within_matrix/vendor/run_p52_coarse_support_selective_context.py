#!/usr/bin/env python3
"""P52: coarse all-role support + selective detailed Entry context.

This experiment does NOT select roles again.

It consumes the role sets already frozen by P51:
  * error-guided Top-K
  * frequent Top-K
  * random Top-K controls

Representation:
  Compact
    + coarse Entry support for all semantic roles
    + detailed Entry context only for selected roles

Detailed context excludes role support itself and contains:
  * role external mean
  * role mapped mean
  * role tier mean
  * role entry-overlap mean
  * role reliability mean
  * entry-to-role reachability (4 dimensions)

Future labels are used only for final metrics.
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
    entry_model_args,
    sha256_file,
)
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
from run_entry_increment_ablation import (  # noqa: E402
    build_graphs_and_derived,
    entry_feature_names,
    entry_layout,
    load_role_names,
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
ROLE_REACH_WIDTH = 4


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attach_labels(rows, feature_by_sha):
    for row in rows:
        row["metadata_label"] = feature_by_sha[row["sha256"]]["metadata_label"]


def mean_prediction(
    x_train,
    y_train,
    x_test,
    seeds,
    iterations,
    learning_rate,
    l2,
):
    seed_probabilities = []

    for seed in seeds:
        probability = predict_logistic(
            x_train,
            y_train,
            x_test,
            seed,
            iterations,
            learning_rate,
            l2,
        )
        seed_probabilities.append(probability)

    return (
        np.mean(seed_probabilities, axis=0),
        seed_probabilities,
    )


def support_indices_for_roles(
    full_names: list[str],
    roles: list[str],
) -> np.ndarray:
    """Support-only columns for selected roles.

    For each role:
      global mean role presence
      global max role presence
      global std role presence
      role-stratified support
    """

    wanted = set()

    for role in roles:
        wanted.update({
            f"global__mean__node_role__{role}",
            f"global__max__node_role__{role}",
            f"global__std__node_role__{role}",
            f"role__{role}__support",
        })

    return np.asarray(
        [
            index
            for index, name in enumerate(full_names)
            if name in wanted
        ],
        dtype=np.int64,
    )


def detailed_indices_for_roles(
    full_names: list[str],
    role_names: list[str],
    selected_roles: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Detailed role context, explicitly excluding support.

    Full Entry part:
      external_mean
      mapped_mean
      tier_mean
      entry_overlap_mean
      reliability_mean

    Derived reachability part:
      reach_rate
      mean_min_hop
      mean_count_log
      internal_rate
    """

    wanted = set()

    for role in selected_roles:
        wanted.update({
            f"role__{role}__external_mean",
            f"role__{role}__mapped_mean",
            f"role__{role}__tier_mean",
            f"role__{role}__entry_overlap_mean",
            f"role__{role}__reliability_mean",
        })

    full_indices = np.asarray(
        [
            index
            for index, name in enumerate(full_names)
            if name in wanted
        ],
        dtype=np.int64,
    )

    reach_indices = []

    for role in selected_roles:
        role_index = role_names.index(role)
        reach_indices.extend(
            range(
                role_index * ROLE_REACH_WIDTH,
                (role_index + 1) * ROLE_REACH_WIDTH,
            )
        )

    return (
        full_indices,
        np.asarray(reach_indices, dtype=np.int64),
    )


def build_aux_block(
    full_matrix,
    reach_matrix,
    support_indices,
    detailed_full_indices=None,
    detailed_reach_indices=None,
):
    parts = [full_matrix[:, support_indices]]

    if detailed_full_indices is not None and len(detailed_full_indices):
        parts.append(full_matrix[:, detailed_full_indices])

    if detailed_reach_indices is not None and len(detailed_reach_indices):
        parts.append(reach_matrix[:, detailed_reach_indices])

    return np.concatenate(parts, axis=1)


def evaluate(
    split_rows,
    cache,
    feature_by_sha,
    schema,
    role_names,
    selections,
    definition,
    seeds,
    iterations,
    learning_rate,
    l2,
    entry_weight,
):
    entry_args = entry_model_args(definition)
    layout = entry_layout()

    full_names = entry_feature_names(role_names)

    all_support_idx = np.asarray(
        layout["sensitive_role_support"],
        dtype=np.int64,
    )

    error_roles = list(selections["error_guided"])
    frequent_roles = list(selections["frequent"])
    random_role_sets = [
        list(values) for values in selections["random"]
    ]

    error_support_idx = support_indices_for_roles(
        full_names,
        error_roles,
    )

    error_detail_full_idx, error_detail_reach_idx = (
        detailed_indices_for_roles(
            full_names,
            role_names,
            error_roles,
        )
    )

    frequent_detail_full_idx, frequent_detail_reach_idx = (
        detailed_indices_for_roles(
            full_names,
            role_names,
            frequent_roles,
        )
    )

    all_detail_full_idx, all_detail_reach_idx = (
        detailed_indices_for_roles(
            full_names,
            role_names,
            role_names,
        )
    )

    random_indices = []

    for roles in random_role_sets:
        random_indices.append(
            detailed_indices_for_roles(
                full_names,
                role_names,
                roles,
            )
        )

    result_rows = []
    prediction_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]

        if not train_rows or not test_rows:
            raise SystemExit(
                f"empty train/test partition in {split_name}"
            )

        attach_labels(train_rows, feature_by_sha)
        attach_labels(test_rows, feature_by_sha)

        train_shas = {
            row["sha256"] for row in train_rows
        }

        evidence = fit_training_fold_components(
            cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )

        full_train, y_train = graph_matrix(
            train_rows,
            cache,
            evidence,
            entry_args,
            True,
        )

        full_test, y_test = graph_matrix(
            test_rows,
            cache,
            evidence,
            entry_args,
            True,
        )

        compact_train, compact_test, _compact_names = (
            compact_matrix_named(
                train_rows,
                test_rows,
                feature_by_sha,
                schema,
            )
        )

        _path_train, reach_train = build_graphs_and_derived(
            train_rows,
            cache,
            evidence,
            definition,
        )

        _path_test, reach_test = build_graphs_and_derived(
            test_rows,
            cache,
            evidence,
            definition,
        )

        full_train, full_test = standardize_pair(
            full_train,
            full_test,
        )

        reach_train, reach_test = standardize_pair(
            reach_train,
            reach_test,
        )

        # --------------------------------------------------
        # Aux blocks
        # --------------------------------------------------

        all_support_train = build_aux_block(
            full_train,
            reach_train,
            all_support_idx,
        )

        all_support_test = build_aux_block(
            full_test,
            reach_test,
            all_support_idx,
        )

        error_guided_train = build_aux_block(
            full_train,
            reach_train,
            all_support_idx,
            error_detail_full_idx,
            error_detail_reach_idx,
        )

        error_guided_test = build_aux_block(
            full_test,
            reach_test,
            all_support_idx,
            error_detail_full_idx,
            error_detail_reach_idx,
        )

        frequent_train = build_aux_block(
            full_train,
            reach_train,
            all_support_idx,
            frequent_detail_full_idx,
            frequent_detail_reach_idx,
        )

        frequent_test = build_aux_block(
            full_test,
            reach_test,
            all_support_idx,
            frequent_detail_full_idx,
            frequent_detail_reach_idx,
        )

        all_detail_train = build_aux_block(
            full_train,
            reach_train,
            all_support_idx,
            all_detail_full_idx,
            all_detail_reach_idx,
        )

        all_detail_test = build_aux_block(
            full_test,
            reach_test,
            all_support_idx,
            all_detail_full_idx,
            all_detail_reach_idx,
        )

        # P51 reproduction:
        # selected-role support + selected-role detail + selected reach.
        p51_error_train = build_aux_block(
            full_train,
            reach_train,
            error_support_idx,
            error_detail_full_idx,
            error_detail_reach_idx,
        )

        p51_error_test = build_aux_block(
            full_test,
            reach_test,
            error_support_idx,
            error_detail_full_idx,
            error_detail_reach_idx,
        )

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),

            # Sanity reproduction of P50 full Entry w0.5.
            "compact_plus_full_entry_w0.5_repro": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * full_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * full_test,
                    ],
                    axis=1,
                ),
            ),

            # Sanity reproduction of P50 role-support variant:
            # P50 did not scale individual submodule variants.
            "compact_plus_all_role_support_w1_repro": (
                np.concatenate(
                    [
                        compact_train,
                        all_support_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        all_support_test,
                    ],
                    axis=1,
                ),
            ),

            # Locked 0.5 Entry evidence version used by P52.
            "compact_plus_all_role_support_w0.5": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * all_support_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * all_support_test,
                    ],
                    axis=1,
                ),
            ),

            # Sanity reproduction of P51 error-guided context.
            "compact_plus_p51_error_context_repro": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p51_error_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p51_error_test,
                    ],
                    axis=1,
                ),
            ),

            # -------------------------------
            # P52 core variants.
            # -------------------------------
            "compact_plus_all_support_error_guided_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * error_guided_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * error_guided_test,
                    ],
                    axis=1,
                ),
            ),

            "compact_plus_all_support_frequent_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * frequent_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * frequent_test,
                    ],
                    axis=1,
                ),
            ),

            "compact_plus_all_support_all_role_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * all_detail_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * all_detail_test,
                    ],
                    axis=1,
                ),
            ),
        }

        for random_index, (
            roles,
            (random_detail_full_idx, random_detail_reach_idx),
        ) in enumerate(
            zip(random_role_sets, random_indices)
        ):
            random_train = build_aux_block(
                full_train,
                reach_train,
                all_support_idx,
                random_detail_full_idx,
                random_detail_reach_idx,
            )

            random_test = build_aux_block(
                full_test,
                reach_test,
                all_support_idx,
                random_detail_full_idx,
                random_detail_reach_idx,
            )

            matrices[
                f"compact_plus_all_support_random_{random_index:02d}_detail"
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * random_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * random_test,
                    ],
                    axis=1,
                ),
            )

        probabilities = {}
        seed_metrics = {}

        for variant, (x_train, x_test) in matrices.items():
            mean_probability, probabilities_by_seed = mean_prediction(
                x_train,
                y_train,
                x_test,
                seeds,
                iterations,
                learning_rate,
                l2,
            )

            probabilities[variant] = mean_probability

            seed_metrics[variant] = [
                metrics(
                    y_test,
                    probability,
                    threshold=THRESHOLD,
                )
                for probability in probabilities_by_seed
            ]

        compact_probability = probabilities["compact_only"]
        compact_pred = (
            compact_probability >= THRESHOLD
        ).astype(np.int64)

        for variant in matrices:
            mean_probability = probabilities[variant]
            variant_pred = (
                mean_probability >= THRESHOLD
            ).astype(np.int64)

            rescue = int(
                np.sum(
                    (compact_pred != y_test)
                    & (variant_pred == y_test)
                )
            )

            harm = int(
                np.sum(
                    (compact_pred == y_test)
                    & (variant_pred != y_test)
                )
            )

            metric_rows = seed_metrics[variant]

            result_rows.append({
                "split_name": split_name,
                "variant": variant,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(
                    matrices[variant][0].shape[1]
                ),
                "entry_weight": (
                    1.0
                    if variant
                    == "compact_plus_all_role_support_w1_repro"
                    else entry_weight
                ),
                "accuracy_mean": float(
                    np.mean([
                        row["accuracy"]
                        for row in metric_rows
                    ])
                ),
                "balanced_accuracy_mean": float(
                    np.mean([
                        row["balanced_accuracy"]
                        for row in metric_rows
                    ])
                ),
                "f1_mean": float(
                    np.mean([
                        row["f1"]
                        for row in metric_rows
                    ])
                ),
                "auc_mean": float(
                    np.mean([
                        row["auc"]
                        for row in metric_rows
                    ])
                ),
                "rescue_vs_compact": rescue,
                "harm_vs_compact": harm,
                "net_rescue_vs_compact": rescue - harm,
            })

            for row, label, pc, pv in zip(
                test_rows,
                y_test,
                compact_probability,
                mean_probability,
            ):
                prediction_rows.append({
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "variant": variant,
                    "label": int(label),
                    "compact_probability": float(pc),
                    "variant_probability": float(pv),
                    "compact_predicted_label": int(
                        pc >= THRESHOLD
                    ),
                    "variant_predicted_label": int(
                        pv >= THRESHOLD
                    ),
                })

        print(
            f"completed {split_name}",
            flush=True,
        )

    return result_rows, prediction_rows


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        variant = row["variant"]

        if variant.startswith(
            "compact_plus_all_support_random_"
        ):
            key = "compact_plus_all_support_random_k_detail"
        else:
            key = variant

        groups[key].append(row)

    result = []

    for variant, selected in sorted(groups.items()):
        ba = np.asarray([
            row["balanced_accuracy_mean"]
            for row in selected
        ])

        result.append({
            "variant": variant,
            "observation_count": len(selected),
            "balanced_accuracy_mean": float(ba.mean()),
            "balanced_accuracy_std": float(ba.std()),
            "minimum_balanced_accuracy": float(ba.min()),
            "accuracy_mean": float(
                np.mean([
                    row["accuracy_mean"]
                    for row in selected
                ])
            ),
            "f1_mean": float(
                np.mean([
                    row["f1_mean"]
                    for row in selected
                ])
            ),
            "auc_mean": float(
                np.mean([
                    row["auc_mean"]
                    for row in selected
                ])
            ),
            "rescue_mean": float(
                np.mean([
                    row["rescue_vs_compact"]
                    for row in selected
                ])
            ),
            "harm_mean": float(
                np.mean([
                    row["harm_vs_compact"]
                    for row in selected
                ])
            ),
            "net_rescue_mean": float(
                np.mean([
                    row["net_rescue_vs_compact"]
                    for row in selected
                ])
            ),
        })

    result.sort(
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            row["variant"],
        )
    )

    for rank, row in enumerate(result, 1):
        row["rank"] = rank

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--entry-cache",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--eval-splits",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--selection-json",
        required=True,
        type=Path,
        help="Frozen selected_roles.json produced by P51.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--entry-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--l2",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(
            strict=True
        ),
        "features": args.features.expanduser().resolve(
            strict=True
        ),
        "schema": args.schema.expanduser().resolve(
            strict=True
        ),
        "mapping": args.mapping.expanduser().resolve(
            strict=True
        ),
        "eval_splits": args.eval_splits.expanduser().resolve(
            strict=True
        ),
        "selection_json": args.selection_json.expanduser().resolve(
            strict=True
        ),
    }

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(
        paths["entry_cache"]
    )

    feature_rows = read_csv_rows(
        paths["features"]
    )

    feature_by_sha = {
        row["sha256"].casefold(): row
        for row in feature_rows
    }

    schema = json.loads(
        paths["schema"].read_text(
            encoding="utf-8"
        )
    )

    role_names = load_role_names(
        paths["mapping"]
    )

    if len(role_names) != ROLE_VOCABULARY_SIZE:
        raise SystemExit(
            f"role count mismatch: "
            f"{len(role_names)} != {ROLE_VOCABULARY_SIZE}"
        )

    selections = json.loads(
        paths["selection_json"].read_text(
            encoding="utf-8"
        )
    )

    required_selection_keys = {
        "error_guided",
        "frequent",
        "random",
    }

    missing_selection_keys = (
        required_selection_keys - set(selections)
    )

    if missing_selection_keys:
        raise SystemExit(
            f"selection JSON missing keys: "
            f"{sorted(missing_selection_keys)}"
        )

    k = len(selections["error_guided"])

    if k <= 0:
        raise SystemExit(
            "error-guided role set is empty"
        )

    if len(selections["frequent"]) != k:
        raise SystemExit(
            "frequent role set has different K"
        )

    if any(
        len(values) != k
        for values in selections["random"]
    ):
        raise SystemExit(
            "a random role set has different K"
        )

    allowed_roles = set(role_names)

    for group_name in (
        "error_guided",
        "frequent",
    ):
        unknown = (
            set(selections[group_name])
            - allowed_roles
        )

        if unknown:
            raise SystemExit(
                f"{group_name} contains unknown roles: "
                f"{sorted(unknown)}"
            )

    for index, values in enumerate(
        selections["random"]
    ):
        unknown = (
            set(values)
            - allowed_roles
        )

        if unknown:
            raise SystemExit(
                f"random set {index} contains "
                f"unknown roles: {sorted(unknown)}"
            )

    split_rows = read_split_rows(
        paths["eval_splits"]
    )

    attach_audit_metadata(
        cache,
        feature_by_sha,
    )

    selected_shas = {
        row["sha256"]
        for split in split_rows.values()
        for partition in ("train", "test")
        for row in split[partition]
    }

    missing_cache = (
        selected_shas - set(cache)
    )

    missing_features = (
        selected_shas - set(feature_by_sha)
    )

    if missing_cache or missing_features:
        raise SystemExit(
            f"missing rows: "
            f"cache={len(missing_cache)}, "
            f"features={len(missing_features)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )

    result_rows, prediction_rows = evaluate(
        split_rows,
        cache,
        feature_by_sha,
        schema,
        role_names,
        selections,
        definition,
        args.seeds,
        args.iterations,
        args.learning_rate,
        args.l2,
        args.entry_weight,
    )

    summary_rows = summarize(
        result_rows
    )

    write_csv(
        output / "p52_per_split.csv",
        result_rows,
    )

    write_csv(
        output / "p52_summary.csv",
        summary_rows,
    )

    write_csv(
        output / "p52_predictions.csv",
        prediction_rows,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": (
            "P52_COARSE_SUPPORT_"
            "SELECTIVE_CONTEXT_REFINEMENT"
        ),
        "selection_source": str(
            paths["selection_json"]
        ),
        "selection_source_sha256": sha256_file(
            paths["selection_json"]
        ),
        "roles_reselected": False,
        "future_labels_used_for_selection": False,
        "top_k": k,
        "error_guided_roles": selections[
            "error_guided"
        ],
        "frequent_roles": selections[
            "frequent"
        ],
        "random_repeat_count": len(
            selections["random"]
        ),
        "classifier": {
            "type": "custom_l2_logistic",
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seeds": args.seeds,
            "threshold": THRESHOLD,
        },
        "entry_weight": args.entry_weight,
        "representation": {
            "coarse_all_role_support": True,
            "selected_detail_fields": [
                "external_mean",
                "mapped_mean",
                "tier_mean",
                "entry_overlap_mean",
                "reliability_mean",
                "entry_role_reach_rate",
                "entry_role_mean_min_hop",
                "entry_role_mean_count_log",
                "entry_role_internal_rate",
            ],
            "selected_detail_excludes_support": True,
        },
        "inputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "outputs": {
            "per_split": str(
                output / "p52_per_split.csv"
            ),
            "summary": str(
                output / "p52_summary.csv"
            ),
            "predictions": str(
                output / "p52_predictions.csv"
            ),
        },
        "sanity_expectations": {
            "compact_only_BA_approx": 0.8764,
            "full_entry_w0.5_BA_approx": 0.9022,
            "all_role_support_w1_BA_approx": 0.8946,
            "p51_error_context_BA_approx": 0.8909,
        },
    }

    (
        output / "p52_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\nP52 summary:",
        flush=True,
    )

    print(
        "rank variant BA std minBA AUC "
        "rescue harm net",
        flush=True,
    )

    for row in summary_rows:
        print(
            row["rank"],
            row["variant"],
            f'{row["balanced_accuracy_mean"]:.4f}',
            f'{row["balanced_accuracy_std"]:.4f}',
            f'{row["minimum_balanced_accuracy"]:.4f}',
            f'{row["auc_mean"]:.4f}',
            f'{row["rescue_mean"]:.2f}',
            f'{row["harm_mean"]:.2f}',
            f'{row["net_rescue_mean"]:.2f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
