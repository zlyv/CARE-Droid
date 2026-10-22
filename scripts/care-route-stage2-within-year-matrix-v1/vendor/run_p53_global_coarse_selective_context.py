#!/usr/bin/env python3
"""P53: global coarse Entry structure + all-role support + selective detail.

Frozen protocol:
  - semantic role sets come directly from P51 selected_roles.json
  - no role reselection
  - no future-year label is used for selection
  - K is frozen
  - classifier hyperparameters are frozen
  - Entry block weight is frozen

Core hypothesis:
  Coarse structural evidence can be retained globally, while richer
  role-specific context should be allocated selectively to semantic roles
  where Compact historically makes systematic errors.

Current cache supports role-level, not raw API-instance-level refinement.
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
    """Detailed context only; support is deliberately excluded."""

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


def concatenate_columns(*blocks):
    usable = [
        block
        for block in blocks
        if block is not None and block.shape[1] > 0
    ]

    if not usable:
        raise ValueError("no auxiliary feature blocks")

    return np.concatenate(usable, axis=1)


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

    global_idx = np.asarray(
        layout["entry_global_structure"],
        dtype=np.int64,
    )

    all_support_idx = np.asarray(
        layout["sensitive_role_support"],
        dtype=np.int64,
    )

    error_roles = list(selections["error_guided"])
    frequent_roles = list(selections["frequent"])
    random_role_sets = [
        list(values)
        for values in selections["random"]
    ]

    # Needed only to reproduce P52 exactly.
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

    random_detail_indices = [
        detailed_indices_for_roles(
            full_names,
            role_names,
            roles,
        )
        for roles in random_role_sets
    ]

    result_rows = []
    prediction_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]

        if not train_rows or not test_rows:
            raise SystemExit(
                f"empty train/test partition: {split_name}"
            )

        attach_labels(train_rows, feature_by_sha)
        attach_labels(test_rows, feature_by_sha)

        train_shas = {
            row["sha256"]
            for row in train_rows
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

        compact_train, compact_test, compact_names = (
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

        # Exactly training-fold standardization.
        full_train, full_test = standardize_pair(
            full_train,
            full_test,
        )

        reach_train, reach_test = standardize_pair(
            reach_train,
            reach_test,
        )

        # --------------------------------------------------
        # Basic Entry blocks
        # --------------------------------------------------

        global_train = full_train[:, global_idx]
        global_test = full_test[:, global_idx]

        support_train = full_train[:, all_support_idx]
        support_test = full_test[:, all_support_idx]

        error_detail_train = concatenate_columns(
            full_train[:, error_detail_full_idx],
            reach_train[:, error_detail_reach_idx],
        )

        error_detail_test = concatenate_columns(
            full_test[:, error_detail_full_idx],
            reach_test[:, error_detail_reach_idx],
        )

        frequent_detail_train = concatenate_columns(
            full_train[:, frequent_detail_full_idx],
            reach_train[:, frequent_detail_reach_idx],
        )

        frequent_detail_test = concatenate_columns(
            full_test[:, frequent_detail_full_idx],
            reach_test[:, frequent_detail_reach_idx],
        )

        all_detail_train = concatenate_columns(
            full_train[:, all_detail_full_idx],
            reach_train[:, all_detail_reach_idx],
        )

        all_detail_test = concatenate_columns(
            full_test[:, all_detail_full_idx],
            reach_test[:, all_detail_reach_idx],
        )

        # P52 exact reproduction:
        # all-role support + selected role detail.
        p52_error_aux_train = concatenate_columns(
            support_train,
            error_detail_train,
        )

        p52_error_aux_test = concatenate_columns(
            support_test,
            error_detail_test,
        )

        # P51 exact reproduction:
        # selected-role support + selected-role detail.
        p51_error_aux_train = concatenate_columns(
            full_train[:, error_support_idx],
            error_detail_train,
        )

        p51_error_aux_test = concatenate_columns(
            full_test[:, error_support_idx],
            error_detail_test,
        )

        # P53 coarse base:
        # global structure + all-role support.
        coarse_train = concatenate_columns(
            global_train,
            support_train,
        )

        coarse_test = concatenate_columns(
            global_test,
            support_test,
        )

        p53_error_aux_train = concatenate_columns(
            coarse_train,
            error_detail_train,
        )

        p53_error_aux_test = concatenate_columns(
            coarse_test,
            error_detail_test,
        )

        p53_frequent_aux_train = concatenate_columns(
            coarse_train,
            frequent_detail_train,
        )

        p53_frequent_aux_test = concatenate_columns(
            coarse_test,
            frequent_detail_test,
        )

        p53_all_detail_aux_train = concatenate_columns(
            coarse_train,
            all_detail_train,
        )

        p53_all_detail_aux_test = concatenate_columns(
            coarse_test,
            all_detail_test,
        )

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),

            # P50 sanity check.
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

            # P51 sanity check.
            "compact_plus_p51_error_context_repro": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p51_error_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p51_error_aux_test,
                    ],
                    axis=1,
                ),
            ),

            # P52 sanity check.
            "compact_plus_p52_error_guided_detail_repro": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p52_error_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p52_error_aux_test,
                    ],
                    axis=1,
                ),
            ),

            # P53 coarse baseline.
            "compact_plus_global_structure_all_support": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * coarse_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * coarse_test,
                    ],
                    axis=1,
                ),
            ),

            # P53 core.
            "compact_plus_global_support_error_guided_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p53_error_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p53_error_aux_test,
                    ],
                    axis=1,
                ),
            ),

            "compact_plus_global_support_frequent_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p53_frequent_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p53_frequent_aux_test,
                    ],
                    axis=1,
                ),
            ),

            "compact_plus_global_support_all_role_detail": (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * p53_all_detail_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * p53_all_detail_aux_test,
                    ],
                    axis=1,
                ),
            ),
        }

        for random_index, (
            random_roles,
            (
                random_detail_full_idx,
                random_detail_reach_idx,
            ),
        ) in enumerate(
            zip(
                random_role_sets,
                random_detail_indices,
            )
        ):
            random_detail_train = concatenate_columns(
                full_train[:, random_detail_full_idx],
                reach_train[:, random_detail_reach_idx],
            )

            random_detail_test = concatenate_columns(
                full_test[:, random_detail_full_idx],
                reach_test[:, random_detail_reach_idx],
            )

            random_aux_train = concatenate_columns(
                coarse_train,
                random_detail_train,
            )

            random_aux_test = concatenate_columns(
                coarse_test,
                random_detail_test,
            )

            matrices[
                f"compact_plus_global_support_random_{random_index:02d}_detail"
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * random_aux_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * random_aux_test,
                    ],
                    axis=1,
                ),
            )

        mean_probabilities = {}
        metric_by_variant = {}

        for variant, (x_train, x_test) in matrices.items():
            mean_probability, seed_probabilities = mean_prediction(
                x_train,
                y_train,
                x_test,
                seeds,
                iterations,
                learning_rate,
                l2,
            )

            mean_probabilities[variant] = mean_probability

            metric_by_variant[variant] = [
                metrics(
                    y_test,
                    probability,
                    threshold=THRESHOLD,
                )
                for probability in seed_probabilities
            ]

        compact_probability = mean_probabilities["compact_only"]
        compact_prediction = (
            compact_probability >= THRESHOLD
        ).astype(np.int64)

        for variant in matrices:
            mean_probability = mean_probabilities[variant]
            prediction = (
                mean_probability >= THRESHOLD
            ).astype(np.int64)

            rescue = int(
                np.sum(
                    (compact_prediction != y_test)
                    & (prediction == y_test)
                )
            )

            harm = int(
                np.sum(
                    (compact_prediction == y_test)
                    & (prediction != y_test)
                )
            )

            metric_rows = metric_by_variant[variant]

            result_rows.append({
                "split_name": split_name,
                "variant": variant,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "compact_feature_count": len(compact_names),
                "feature_count": int(
                    matrices[variant][0].shape[1]
                ),
                "aux_feature_count": int(
                    matrices[variant][0].shape[1]
                    - compact_train.shape[1]
                ),
                "entry_weight": entry_weight,
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


def summarize_each_variant(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["variant"]].append(row)

    result = []

    for variant, selected in sorted(groups.items()):
        ba = np.asarray([
            row["balanced_accuracy_mean"]
            for row in selected
        ])

        result.append({
            "variant": variant,
            "split_count": len(selected),
            "balanced_accuracy_mean": float(ba.mean()),
            "balanced_accuracy_std_across_splits": float(ba.std()),
            "minimum_balanced_accuracy": float(ba.min()),
            "accuracy_mean": float(np.mean([
                row["accuracy_mean"]
                for row in selected
            ])),
            "f1_mean": float(np.mean([
                row["f1_mean"]
                for row in selected
            ])),
            "auc_mean": float(np.mean([
                row["auc_mean"]
                for row in selected
            ])),
            "aux_feature_count_mean": float(np.mean([
                row["aux_feature_count"]
                for row in selected
            ])),
            "rescue_mean": float(np.mean([
                row["rescue_vs_compact"]
                for row in selected
            ])),
            "harm_mean": float(np.mean([
                row["harm_vs_compact"]
                for row in selected
            ])),
            "net_rescue_mean": float(np.mean([
                row["net_rescue_vs_compact"]
                for row in selected
            ])),
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


def build_random_null(variant_summary):
    random_rows = [
        row
        for row in variant_summary
        if row["variant"].startswith(
            "compact_plus_global_support_random_"
        )
    ]

    result = []

    for row in random_rows:
        token = row["variant"].split("_random_", 1)[1]
        random_id = token.split("_detail", 1)[0]

        result.append({
            "random_id": random_id,
            "balanced_accuracy_mean": row[
                "balanced_accuracy_mean"
            ],
            "minimum_balanced_accuracy": row[
                "minimum_balanced_accuracy"
            ],
            "auc_mean": row["auc_mean"],
            "rescue_mean": row["rescue_mean"],
            "harm_mean": row["harm_mean"],
            "net_rescue_mean": row[
                "net_rescue_mean"
            ],
        })

    result.sort(
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            row["random_id"],
        )
    )

    return result


def headline_summary(
    variant_summary,
    random_null,
):
    by_name = {
        row["variant"]: row
        for row in variant_summary
    }

    headline_names = [
        "compact_only",
        "compact_plus_full_entry_w0.5_repro",
        "compact_plus_p51_error_context_repro",
        "compact_plus_p52_error_guided_detail_repro",
        "compact_plus_global_structure_all_support",
        "compact_plus_global_support_frequent_detail",
        "compact_plus_global_support_error_guided_detail",
        "compact_plus_global_support_all_role_detail",
    ]

    rows = []

    for name in headline_names:
        if name in by_name:
            rows.append(dict(by_name[name]))

    if random_null:
        random_ba = np.asarray([
            row["balanced_accuracy_mean"]
            for row in random_null
        ])

        random_auc = np.asarray([
            row["auc_mean"]
            for row in random_null
        ])

        random_net = np.asarray([
            row["net_rescue_mean"]
            for row in random_null
        ])

        rows.append({
            "variant": (
                "compact_plus_global_support_random_k_detail"
            ),
            "split_count": len(random_null),
            "balanced_accuracy_mean": float(
                random_ba.mean()
            ),
            "balanced_accuracy_std_across_splits": float(
                random_ba.std()
            ),
            "minimum_balanced_accuracy": float(
                random_ba.min()
            ),
            "accuracy_mean": "",
            "f1_mean": "",
            "auc_mean": float(random_auc.mean()),
            "aux_feature_count_mean": "",
            "rescue_mean": "",
            "harm_mean": "",
            "net_rescue_mean": float(
                random_net.mean()
            ),
            "rank": "",
        })

    return rows


def random_comparison(
    variant_summary,
    random_null,
):
    by_name = {
        row["variant"]: row
        for row in variant_summary
    }

    target_name = (
        "compact_plus_global_support_"
        "error_guided_detail"
    )

    target = by_name.get(target_name)

    if target is None or not random_null:
        return {}

    target_ba = float(
        target["balanced_accuracy_mean"]
    )

    random_ba = np.asarray([
        row["balanced_accuracy_mean"]
        for row in random_null
    ])

    count_ge = int(
        np.sum(random_ba >= target_ba)
    )

    empirical_p = (
        1.0 + count_ge
    ) / (
        1.0 + len(random_ba)
    )

    return {
        "error_guided_ba": target_ba,
        "random_repeat_count": len(random_ba),
        "random_ba_mean": float(
            random_ba.mean()
        ),
        "random_ba_std": float(
            random_ba.std()
        ),
        "random_ba_min": float(
            random_ba.min()
        ),
        "random_ba_max": float(
            random_ba.max()
        ),
        "random_sets_equal_or_better": count_ge,
        "empirical_upper_tail_p_with_plus_one": float(
            empirical_p
        ),
        "note": (
            "Diagnostic only: with 10 random sets the "
            "empirical p-value has very coarse resolution."
        ),
    }


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
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            f"role vocabulary mismatch: "
            f"{len(role_names)} != "
            f"{ROLE_VOCABULARY_SIZE}"
        )

    selections = json.loads(
        paths["selection_json"].read_text(
            encoding="utf-8"
        )
    )

    required = {
        "error_guided",
        "frequent",
        "random",
    }

    missing_keys = required - set(selections)

    if missing_keys:
        raise SystemExit(
            f"selection JSON missing: "
            f"{sorted(missing_keys)}"
        )

    k = len(selections["error_guided"])

    if k <= 0:
        raise SystemExit(
            "empty error-guided selection"
        )

    if len(selections["frequent"]) != k:
        raise SystemExit(
            "frequent selection K mismatch"
        )

    if any(
        len(values) != k
        for values in selections["random"]
    ):
        raise SystemExit(
            "random selection K mismatch"
        )

    allowed = set(role_names)

    for name in (
        "error_guided",
        "frequent",
    ):
        unknown = (
            set(selections[name])
            - allowed
        )

        if unknown:
            raise SystemExit(
                f"{name} has unknown roles: "
                f"{sorted(unknown)}"
            )

    for index, values in enumerate(
        selections["random"]
    ):
        unknown = set(values) - allowed

        if unknown:
            raise SystemExit(
                f"random set {index} has "
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
        selected_shas
        - set(feature_by_sha)
    )

    if missing_cache or missing_features:
        raise SystemExit(
            f"missing selected rows: "
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

    variant_summary = summarize_each_variant(
        result_rows
    )

    random_null = build_random_null(
        variant_summary
    )

    headline = headline_summary(
        variant_summary,
        random_null,
    )

    random_test = random_comparison(
        variant_summary,
        random_null,
    )

    write_csv(
        output / "p53_per_split.csv",
        result_rows,
    )

    write_csv(
        output / "p53_variant_summary.csv",
        variant_summary,
    )

    write_csv(
        output / "p53_random_null.csv",
        random_null,
    )

    write_csv(
        output / "p53_headline_summary.csv",
        headline,
    )

    write_csv(
        output / "p53_predictions.csv",
        prediction_rows,
    )

    (
        output / "p53_random_comparison.json"
    ).write_text(
        json.dumps(
            random_test,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": (
            "P53_GLOBAL_COARSE_"
            "SELECTIVE_CONTEXT_REFINEMENT"
        ),
        "roles_reselected": False,
        "future_labels_used_for_selection": False,
        "selection_json": str(
            paths["selection_json"]
        ),
        "selection_json_sha256": sha256_file(
            paths["selection_json"]
        ),
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
            "global_entry_structure": True,
            "all_role_coarse_support": True,
            "selected_role_detail": [
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
        "sanity_expectations": {
            "compact_only_BA_approx": 0.8764,
            "full_entry_w0.5_BA_approx": 0.9022,
            "p51_error_context_BA_approx": 0.8909,
            "p52_error_guided_BA_approx": 0.8971,
        },
        "random_test": random_test,
        "inputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
    }

    (
        output / "p53_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nP53 headline:", flush=True)

    print(
        "variant BA minBA AUC aux_dim "
        "rescue harm net",
        flush=True,
    )

    for row in headline:
        print(
            row["variant"],
            (
                f'{row["balanced_accuracy_mean"]:.4f}'
                if isinstance(
                    row["balanced_accuracy_mean"],
                    (float, int),
                )
                else row["balanced_accuracy_mean"]
            ),
            (
                f'{row["minimum_balanced_accuracy"]:.4f}'
                if isinstance(
                    row["minimum_balanced_accuracy"],
                    (float, int),
                )
                else row["minimum_balanced_accuracy"]
            ),
            (
                f'{row["auc_mean"]:.4f}'
                if isinstance(
                    row["auc_mean"],
                    (float, int),
                )
                else row["auc_mean"]
            ),
            row.get(
                "aux_feature_count_mean",
                "",
            ),
            row.get("rescue_mean", ""),
            row.get("harm_mean", ""),
            row.get("net_rescue_mean", ""),
            flush=True,
        )

    print("\nRandom comparison:", flush=True)
    print(
        json.dumps(
            random_test,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
