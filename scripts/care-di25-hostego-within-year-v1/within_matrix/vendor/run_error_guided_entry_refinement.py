#!/usr/bin/env python3
"""Error-guided selective Entry refinement.

Protocol
--------
1. Use 2017 rolling validation only to identify semantic roles on which
   Compact has systematically elevated FN/FP rates.
2. Freeze Top-K roles.
3. Evaluate fixed-2017 -> future-year splits without using any future label
   for role selection.
4. Compare:
     Compact
     Compact + Full Entry
     Compact + error-guided role support
     Compact + error-guided role context
     Compact + frequent-K role context
     Compact + random-K role context

The current A3 cache does not contain raw API names/method signatures, so this
is deliberately role-level rather than API-instance-level refinement.
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


def label_value(value) -> int:
    value = str(value or "").strip().casefold()
    if value in {"benign", "0", "false"}:
        return 0
    if value in {"malware", "malicious", "1", "true"}:
        return 1
    raise ValueError(f"unknown label: {value!r}")


def average_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seeds: list[int],
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    values = []
    for seed in seeds:
        values.append(
            predict_logistic(
                x_train,
                y_train,
                x_test,
                seed,
                iterations,
                learning_rate,
                l2,
            )
        )
    return np.mean(values, axis=0), values


def attach_labels(rows: list[dict], feature_by_sha: dict[str, dict]) -> None:
    for row in rows:
        row["metadata_label"] = feature_by_sha[row["sha256"]]["metadata_label"]


def role_presence_matrix(
    rows: list[dict],
    feature_by_sha: dict[str, dict],
    role_names: list[str],
) -> np.ndarray:
    result = np.zeros((len(rows), len(role_names)), dtype=bool)

    for row_index, row in enumerate(rows):
        feature_row = feature_by_sha[row["sha256"]]
        for role_index, role in enumerate(role_names):
            field = f"role_node__{role}"
            try:
                result[row_index, role_index] = float(feature_row.get(field, 0.0)) > 0.0
            except (TypeError, ValueError):
                result[row_index, role_index] = False

    return result


def collect_error_guidance(
    split_rows: dict,
    feature_by_sha: dict[str, dict],
    schema: dict,
    role_names: list[str],
    seeds: list[int],
    iterations: int,
    learning_rate: float,
    l2: float,
    shrinkage: float,
) -> tuple[list[dict], list[dict]]:
    """Generate time-forward Compact errors using 2017 rolling splits."""

    fold_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]

        if not train_rows or not validation_rows:
            raise ValueError(f"empty rolling partition: {split_name}")

        attach_labels(train_rows, feature_by_sha)
        attach_labels(validation_rows, feature_by_sha)

        x_train, x_validation, _ = compact_matrix_named(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
        )

        y_train = np.asarray(
            [label_value(row["metadata_label"]) for row in train_rows],
            dtype=np.int64,
        )
        y_validation = np.asarray(
            [label_value(row["metadata_label"]) for row in validation_rows],
            dtype=np.int64,
        )

        probability, _ = average_predictions(
            x_train,
            y_train,
            x_validation,
            seeds,
            iterations,
            learning_rate,
            l2,
        )
        prediction = (probability >= THRESHOLD).astype(np.int64)

        positive = y_validation == 1
        negative = y_validation == 0

        global_fnr = float(
            np.mean(prediction[positive] == 0)
        ) if np.any(positive) else 0.0

        global_fpr = float(
            np.mean(prediction[negative] == 1)
        ) if np.any(negative) else 0.0

        presence = role_presence_matrix(
            validation_rows,
            feature_by_sha,
            role_names,
        )

        for role_index, role in enumerate(role_names):
            role_mask = presence[:, role_index]

            malicious_mask = role_mask & positive
            benign_mask = role_mask & negative

            malicious_support = int(malicious_mask.sum())
            benign_support = int(benign_mask.sum())
            total_support = malicious_support + benign_support

            fn_count = int(
                np.sum((prediction == 0) & malicious_mask)
            )
            fp_count = int(
                np.sum((prediction == 1) & benign_mask)
            )

            # Shrink sparse role-specific rates toward the fold-wide rate.
            fnr = (
                fn_count + shrinkage * global_fnr
            ) / (malicious_support + shrinkage)

            fpr = (
                fp_count + shrinkage * global_fpr
            ) / (benign_support + shrinkage)

            fold_rows.append({
                "split_name": split_name,
                "role": role,
                "validation_count": len(validation_rows),
                "global_fnr": global_fnr,
                "global_fpr": global_fpr,
                "malicious_support": malicious_support,
                "benign_support": benign_support,
                "total_support": total_support,
                "fn_count": fn_count,
                "fp_count": fp_count,
                "shrunk_fnr": float(fnr),
                "shrunk_fpr": float(fpr),
                "excess_fnr": float(fnr - global_fnr),
                "excess_fpr": float(fpr - global_fpr),
            })

    by_role = defaultdict(list)
    for row in fold_rows:
        by_role[row["role"]].append(row)

    ranking = []

    for role in role_names:
        rows = by_role[role]

        mal_support = sum(row["malicious_support"] for row in rows)
        ben_support = sum(row["benign_support"] for row in rows)
        total_support = mal_support + ben_support

        if mal_support:
            mean_excess_fn = float(
                np.average(
                    [row["excess_fnr"] for row in rows],
                    weights=[
                        max(1, row["malicious_support"])
                        for row in rows
                    ],
                )
            )
        else:
            mean_excess_fn = 0.0

        if ben_support:
            mean_excess_fp = float(
                np.average(
                    [row["excess_fpr"] for row in rows],
                    weights=[
                        max(1, row["benign_support"])
                        for row in rows
                    ],
                )
            )
        else:
            mean_excess_fp = 0.0

        if mean_excess_fn >= mean_excess_fp:
            direction = "FN"
            dominant_excess = mean_excess_fn
            consistency = float(np.mean([
                row["excess_fnr"] > 0.0 for row in rows
            ]))
        else:
            direction = "FP"
            dominant_excess = mean_excess_fp
            consistency = float(np.mean([
                row["excess_fpr"] > 0.0 for row in rows
            ]))

        support_factor = total_support / (total_support + 20.0)

        score = (
            max(0.0, dominant_excess)
            * consistency
            * support_factor
        )

        ranking.append({
            "role": role,
            "score": float(score),
            "dominant_error": direction,
            "mean_excess_fnr": mean_excess_fn,
            "mean_excess_fpr": mean_excess_fp,
            "fold_consistency": consistency,
            "malicious_support": mal_support,
            "benign_support": ben_support,
            "total_support": total_support,
            "fold_count": len(rows),
        })

    ranking.sort(
        key=lambda row: (
            -row["score"],
            -row["fold_consistency"],
            -row["total_support"],
            row["role"],
        )
    )

    for index, row in enumerate(ranking, 1):
        row["rank"] = index

    return fold_rows, ranking


def frequent_roles(
    split_rows: dict,
    feature_by_sha: dict[str, dict],
    role_names: list[str],
    top_k: int,
) -> tuple[list[str], list[dict]]:
    # Only historical rolling-training rows are used.
    train_shas = {
        row["sha256"]
        for split in split_rows.values()
        for row in split["train"]
    }

    counts = []

    for role in role_names:
        field = f"role_node__{role}"
        count = 0

        for sha in train_shas:
            try:
                if float(feature_by_sha[sha].get(field, 0.0)) > 0.0:
                    count += 1
            except (TypeError, ValueError):
                pass

        counts.append({
            "role": role,
            "document_frequency": count,
        })

    counts.sort(
        key=lambda row: (-row["document_frequency"], row["role"])
    )

    return [row["role"] for row in counts[:top_k]], counts


def selected_entry_indices(
    full_names: list[str],
    role_names: list[str],
    selected_roles: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support-only, role-context and reachability columns."""

    support_indices = []
    context_indices = []
    reach_indices = []

    for role in selected_roles:
        support_names = {
            f"global__mean__node_role__{role}",
            f"global__max__node_role__{role}",
            f"global__std__node_role__{role}",
            f"role__{role}__support",
        }

        role_context_prefix = f"role__{role}__"

        for index, name in enumerate(full_names):
            if name in support_names:
                support_indices.append(index)

            if (
                name in support_names
                or name.startswith(role_context_prefix)
            ):
                context_indices.append(index)

        role_index = role_names.index(role)
        reach_indices.extend(
            range(
                role_index * ROLE_REACH_WIDTH,
                (role_index + 1) * ROLE_REACH_WIDTH,
            )
        )

    return (
        np.asarray(sorted(set(support_indices)), dtype=np.int64),
        np.asarray(sorted(set(context_indices)), dtype=np.int64),
        np.asarray(sorted(set(reach_indices)), dtype=np.int64),
    )


def evaluate(
    eval_splits: dict,
    cache: dict,
    feature_by_sha: dict[str, dict],
    schema: dict,
    role_names: list[str],
    definition: PilotReliabilityDefinition,
    error_roles: list[str],
    frequent_role_set: list[str],
    random_role_sets: list[list[str]],
    seeds: list[int],
    iterations: int,
    learning_rate: float,
    l2: float,
    entry_weight: float,
) -> tuple[list[dict], list[dict]]:
    entry_args = entry_model_args(definition)
    full_names = entry_feature_names(role_names)

    variant_roles = {
        "error_guided": error_roles,
        "frequent": frequent_role_set,
    }

    for index, roles in enumerate(random_role_sets):
        variant_roles[f"random_{index:02d}"] = roles

    result_rows = []
    comparison_rows = []

    for split_name in sorted(eval_splits):
        train_rows = eval_splits[split_name]["train"]
        test_rows = eval_splits[split_name]["test"]

        if not train_rows or not test_rows:
            raise ValueError(f"empty evaluation partition: {split_name}")

        attach_labels(train_rows, feature_by_sha)
        attach_labels(test_rows, feature_by_sha)

        train_shas = {row["sha256"] for row in train_rows}

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

        compact_train, compact_test, _ = compact_matrix_named(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        _, reach_train = build_graphs_and_derived(
            train_rows,
            cache,
            evidence,
            definition,
        )
        _, reach_test = build_graphs_and_derived(
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

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),
            "compact_plus_full_entry": (
                np.concatenate(
                    [compact_train, entry_weight * full_train],
                    axis=1,
                ),
                np.concatenate(
                    [compact_test, entry_weight * full_test],
                    axis=1,
                ),
            ),
        }

        for selector_name, selected_roles in variant_roles.items():
            support_idx, context_idx, reach_idx = selected_entry_indices(
                full_names,
                role_names,
                selected_roles,
            )

            selected_support_train = full_train[:, support_idx]
            selected_support_test = full_test[:, support_idx]

            selected_context_train = np.concatenate(
                [
                    full_train[:, context_idx],
                    reach_train[:, reach_idx],
                ],
                axis=1,
            )
            selected_context_test = np.concatenate(
                [
                    full_test[:, context_idx],
                    reach_test[:, reach_idx],
                ],
                axis=1,
            )

            if selector_name == "error_guided":
                matrices["compact_plus_error_guided_support"] = (
                    np.concatenate(
                        [
                            compact_train,
                            entry_weight * selected_support_train,
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            compact_test,
                            entry_weight * selected_support_test,
                        ],
                        axis=1,
                    ),
                )

            matrices[
                f"compact_plus_{selector_name}_context"
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        entry_weight * selected_context_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        entry_weight * selected_context_test,
                    ],
                    axis=1,
                ),
            )

        probabilities = {}
        seed_metric_map = {}

        for variant, (x_train, x_test) in matrices.items():
            mean_probability, seed_probabilities = average_predictions(
                x_train,
                y_train,
                x_test,
                seeds,
                iterations,
                learning_rate,
                l2,
            )

            probabilities[variant] = mean_probability
            seed_metric_map[variant] = [
                metrics(
                    y_test,
                    probability,
                    threshold=THRESHOLD,
                )
                for probability in seed_probabilities
            ]

        compact_pred = (
            probabilities["compact_only"] >= THRESHOLD
        ).astype(np.int64)

        for variant, metric_rows in seed_metric_map.items():
            mean_probability = probabilities[variant]
            pred = (mean_probability >= THRESHOLD).astype(np.int64)

            rescue = int(np.sum(
                (compact_pred != y_test)
                & (pred == y_test)
            ))
            harm = int(np.sum(
                (compact_pred == y_test)
                & (pred != y_test)
            ))

            result = {
                "split_name": split_name,
                "variant": variant,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(matrices[variant][0].shape[1]),
                "entry_weight": entry_weight,
                "accuracy_mean": float(np.mean([
                    row["accuracy"] for row in metric_rows
                ])),
                "balanced_accuracy_mean": float(np.mean([
                    row["balanced_accuracy"] for row in metric_rows
                ])),
                "f1_mean": float(np.mean([
                    row["f1"] for row in metric_rows
                ])),
                "auc_mean": float(np.mean([
                    row["auc"] for row in metric_rows
                ])),
                "rescue_vs_compact": rescue,
                "harm_vs_compact": harm,
                "net_rescue_vs_compact": rescue - harm,
            }

            result_rows.append(result)

            for row, label, p_compact, p_variant in zip(
                test_rows,
                y_test,
                probabilities["compact_only"],
                mean_probability,
            ):
                comparison_rows.append({
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "variant": variant,
                    "label": int(label),
                    "compact_probability": float(p_compact),
                    "variant_probability": float(p_variant),
                    "compact_pred": int(p_compact >= THRESHOLD),
                    "variant_pred": int(p_variant >= THRESHOLD),
                })

        print(f"completed {split_name}", flush=True)

    return result_rows, comparison_rows


def summarize(result_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for row in result_rows:
        variant = row["variant"]

        if variant.startswith("compact_plus_random_"):
            group = "compact_plus_random_k_context"
        else:
            group = variant

        groups[group].append(row)

    summaries = []

    for variant, rows in sorted(groups.items()):
        ba = np.asarray(
            [row["balanced_accuracy_mean"] for row in rows],
            dtype=float,
        )

        summaries.append({
            "variant": variant,
            "observation_count": len(rows),
            "balanced_accuracy_mean": float(ba.mean()),
            "balanced_accuracy_std": float(ba.std()),
            "accuracy_mean": float(np.mean([
                row["accuracy_mean"] for row in rows
            ])),
            "f1_mean": float(np.mean([
                row["f1_mean"] for row in rows
            ])),
            "auc_mean": float(np.mean([
                row["auc_mean"] for row in rows
            ])),
            "rescue_mean": float(np.mean([
                row["rescue_vs_compact"] for row in rows
            ])),
            "harm_mean": float(np.mean([
                row["harm_vs_compact"] for row in rows
            ])),
            "net_rescue_mean": float(np.mean([
                row["net_rescue_vs_compact"] for row in rows
            ])),
        })

    summaries.sort(
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            row["variant"],
        )
    )

    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)

    parser.add_argument(
        "--rolling-splits",
        required=True,
        type=Path,
        help="2017-only time-forward development splits",
    )
    parser.add_argument(
        "--eval-splits",
        required=True,
        type=Path,
        help="locked future-year evaluation split manifest",
    )

    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260813)

    parser.add_argument("--shrinkage", type=float, default=8.0)

    parser.add_argument(
        "--entry-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--l2",
        required=True,
        type=float,
        help="Required intentionally: use the historically locked value.",
    )

    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "mapping": args.mapping.expanduser().resolve(strict=True),
        "rolling_splits": args.rolling_splits.expanduser().resolve(strict=True),
        "eval_splits": args.eval_splits.expanduser().resolve(strict=True),
    }

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(paths["entry_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {
        row["sha256"].casefold(): row
        for row in feature_rows
    }

    schema = json.loads(
        paths["schema"].read_text(encoding="utf-8")
    )

    role_names = load_role_names(paths["mapping"])

    if len(role_names) != ROLE_VOCABULARY_SIZE:
        raise SystemExit(
            f"role vocabulary mismatch: "
            f"{len(role_names)} != {ROLE_VOCABULARY_SIZE}"
        )

    rolling_splits = read_split_rows(paths["rolling_splits"])
    eval_splits = read_split_rows(paths["eval_splits"])

    attach_audit_metadata(cache, feature_by_sha)

    all_selected_shas = {
        row["sha256"]
        for source in (rolling_splits, eval_splits)
        for split in source.values()
        for partition in ("train", "test")
        for row in split[partition]
    }

    missing_cache = all_selected_shas - set(cache)
    missing_features = all_selected_shas - set(feature_by_sha)

    if missing_cache or missing_features:
        raise SystemExit(
            f"missing rows: cache={len(missing_cache)}, "
            f"features={len(missing_features)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )

    # ----------------------------------------------------------
    # Stage 1: 2017-only error guidance.
    # ----------------------------------------------------------
    fold_profile, ranking = collect_error_guidance(
        rolling_splits,
        feature_by_sha,
        schema,
        role_names,
        args.seeds,
        args.iterations,
        args.learning_rate,
        args.l2,
        args.shrinkage,
    )

    positive_ranking = [
        row for row in ranking if row["score"] > 0.0
    ]

    if len(positive_ranking) < args.top_k:
        raise SystemExit(
            f"only {len(positive_ranking)} roles have positive "
            f"error-guidance score; requested top-k={args.top_k}"
        )

    error_roles = [
        row["role"]
        for row in positive_ranking[:args.top_k]
    ]

    frequent_role_set, frequency_rows = frequent_roles(
        rolling_splits,
        feature_by_sha,
        role_names,
        args.top_k,
    )

    rng = np.random.RandomState(args.random_seed)

    random_role_sets = [
        sorted(
            rng.choice(
                role_names,
                size=args.top_k,
                replace=False,
            ).tolist()
        )
        for _ in range(args.random_repeats)
    ]

    selection = {
        "error_guided": error_roles,
        "frequent": frequent_role_set,
        "random": random_role_sets,
    }

    # ----------------------------------------------------------
    # Stage 2: locked future evaluation.
    # ----------------------------------------------------------
    result_rows, comparison_rows = evaluate(
        eval_splits,
        cache,
        feature_by_sha,
        schema,
        role_names,
        definition,
        error_roles,
        frequent_role_set,
        random_role_sets,
        args.seeds,
        args.iterations,
        args.learning_rate,
        args.l2,
        args.entry_weight,
    )

    summary_rows = summarize(result_rows)

    write_csv(
        output / "error_guidance_per_fold.csv",
        fold_profile,
    )
    write_csv(
        output / "error_guidance_role_ranking.csv",
        ranking,
    )
    write_csv(
        output / "role_frequency.csv",
        frequency_rows,
    )
    write_csv(
        output / "future_eval_per_split.csv",
        result_rows,
    )
    write_csv(
        output / "future_eval_summary.csv",
        summary_rows,
    )
    write_csv(
        output / "future_eval_predictions.csv",
        comparison_rows,
    )

    (
        output / "selected_roles.json"
    ).write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": "ERROR_GUIDED_SELECTIVE_ENTRY_REFINEMENT",
        "selection_data": "2017_ROLLING_ONLY",
        "future_labels_used_for_selection": False,
        "threshold": THRESHOLD,
        "top_k": args.top_k,
        "entry_weight": args.entry_weight,
        "l2": args.l2,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "seeds": args.seeds,
        "shrinkage": args.shrinkage,
        "selected_roles": selection,
        "inputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "outputs": {
            "error_profile": str(
                output / "error_guidance_per_fold.csv"
            ),
            "role_ranking": str(
                output / "error_guidance_role_ranking.csv"
            ),
            "selection": str(
                output / "selected_roles.json"
            ),
            "future_per_split": str(
                output / "future_eval_per_split.csv"
            ),
            "future_summary": str(
                output / "future_eval_summary.csv"
            ),
            "predictions": str(
                output / "future_eval_predictions.csv"
            ),
        },
        "limitations": [
            "Current cache supports role-level, not raw API-instance-level guidance.",
            "Role selection uses Compact role_node presence as the error anchor.",
            "Selected context uses current role-stratified Entry pooling plus role reachability.",
            "No 2018+ label participates in role selection.",
        ],
    }

    (
        output / "audit.json"
    ).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSelected error-guided roles:")
    for index, role in enumerate(error_roles, 1):
        matched = next(
            row for row in ranking if row["role"] == role
        )
        print(
            index,
            role,
            "score=",
            f'{matched["score"]:.6f}',
            "direction=",
            matched["dominant_error"],
            "support=",
            matched["total_support"],
        )

    print("\nFuture evaluation:")
    print(
        "variant BA_mean BA_std AUC rescue harm net",
        flush=True,
    )
    for row in summary_rows:
        print(
            row["variant"],
            f'{row["balanced_accuracy_mean"]:.4f}',
            f'{row["balanced_accuracy_std"]:.4f}',
            f'{row["auc_mean"]:.4f}',
            f'{row["rescue_mean"]:.2f}',
            f'{row["harm_mean"]:.2f}',
            f'{row["net_rescue_mean"]:.2f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
