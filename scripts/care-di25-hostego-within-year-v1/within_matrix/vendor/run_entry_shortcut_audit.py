#!/usr/bin/env python3
"""Audit whether the Entry view is mainly an entry-presence shortcut.

This audit consumes the existing bounded Entry cache and does not open APKs.
It separates the very small ``entry presence`` block from the remaining
role-stratified Entry representation, reports label prevalence conditional on
entry presence, and evaluates a train-only majority rule plus a logistic model
using only the presence block.  Test labels are used only for post-hoc audit
metrics.

The classification threshold is fixed at 0.5.  Standardization and model
fitting use outer-training rows only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    attach_audit_metadata,
    entry_model_args,
)
from run_entry_increment_ablation import (  # noqa: E402
    entry_layout,
    load_role_names,
)
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def label_of(row: dict) -> int:
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def binary_presence_features(full: np.ndarray, layout: dict[str, np.ndarray]) -> np.ndarray:
    """Return the exact three Entry-presence summary columns used by P50."""

    return np.asarray(full[:, layout["entry_presence"]], dtype=np.float64)


def raw_entry_indicator(full: np.ndarray, layout: dict[str, np.ndarray]) -> np.ndarray:
    """Recover the unstandardized entry indicator from the mean/max/std block.

    The max summary is one exactly when at least one entry node exists and zero
    otherwise.  This is the direct ``has entry`` shortcut being audited.
    """

    max_column = int(layout["entry_presence"][1])
    return (full[:, max_column] > 0.5).astype(np.int64)


def direct_entry_flag_free_features(
    full: np.ndarray, layout: dict[str, np.ndarray]
) -> np.ndarray:
    """Remove direct entry flags but retain the entry-conditioned subgraph.

    This is deliberately named *direct* flag-free: the current cache itself is
    built from an entry-guided bounded subgraph, so removing these columns does
    not remove all indirect entry information.
    """

    direct_flags = np.asarray(sorted(
        set(layout["entry_presence"].tolist())
        | set(layout["role_node_entry_overlap"].tolist())
    ), dtype=np.int64)
    keep = np.setdiff1d(np.arange(full.shape[1], dtype=np.int64), direct_flags)
    return np.asarray(full[:, keep], dtype=np.float64)


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prevalence_rows(
    split_name: str,
    partition: str,
    rows: list[dict],
    labels: np.ndarray,
    indicators: np.ndarray,
) -> list[dict]:
    result = []
    for indicator in (0, 1):
        mask = indicators == indicator
        n = int(mask.sum())
        positives = int(labels[mask].sum())
        result.append({
            "split_name": split_name,
            "partition": partition,
            "year": rows[0].get("metadata_year", "") if rows else "",
            "entry_present": indicator,
            "row_count": n,
            "malicious_count": positives,
            "malicious_rate": safe_rate(positives, n),
        })
    return result


def majority_rule_predictions(
    y_train: np.ndarray,
    train_indicator: np.ndarray,
    eval_indicator: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    assignments = {}
    global_majority = int(np.mean(y_train) >= 0.5)
    for indicator in (0, 1):
        mask = train_indicator == indicator
        assignments[indicator] = int(np.mean(y_train[mask]) >= 0.5) if np.any(mask) else global_majority
    return np.asarray([assignments.get(int(value), global_majority) for value in eval_indicator]), assignments


def auc_or_empty(result: dict) -> float:
    value = result.get("auc")
    return float(value) if value is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument(
        "--label-shuffle-seeds", nargs="+", type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds for the train-label permutation control.",
    )
    args = parser.parse_args()

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "mapping": args.mapping.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(paths["entry_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    # Validate that the supplied schema is readable and included in the audit.
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    role_names = load_role_names(paths["mapping"])
    if len(role_names) != 31:
        raise SystemExit(f"expected 31 semantic roles; observed={len(role_names)}")
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    layout = entry_layout()
    summary_rows = []
    prevalence = []
    rule_rows = []
    model_rows = []
    shuffle_rows = []
    year_samples: dict[tuple[str, str], tuple[int, int]] = {}

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        train_full, y_train = graph_matrix(train_rows, cache, evidence, entry_args, True)
        test_full, y_test = graph_matrix(test_rows, cache, evidence, entry_args, True)
        compact_train, compact_test, _ = compact_matrix_named(
            train_rows, test_rows, feature_by_sha, schema
        )
        train_presence = binary_presence_features(train_full, layout)
        test_presence = binary_presence_features(test_full, layout)
        train_indicator = raw_entry_indicator(train_full, layout)
        test_indicator = raw_entry_indicator(test_full, layout)
        train_indicator_features = train_indicator[:, None].astype(np.float64)
        test_indicator_features = test_indicator[:, None].astype(np.float64)
        train_flag_free = direct_entry_flag_free_features(train_full, layout)
        test_flag_free = direct_entry_flag_free_features(test_full, layout)
        train_presence, test_presence = standardize_pair(train_presence, test_presence)
        train_indicator_features, test_indicator_features = standardize_pair(
            train_indicator_features, test_indicator_features
        )
        train_flag_free, test_flag_free = standardize_pair(train_flag_free, test_flag_free)
        train_compact_flag_free = np.concatenate(
            [compact_train, 0.5 * train_flag_free], axis=1
        )
        test_compact_flag_free = np.concatenate(
            [compact_test, 0.5 * test_flag_free], axis=1
        )

        for partition, rows, labels, indicators in (
            ("train", train_rows, y_train, train_indicator),
            ("test", test_rows, y_test, test_indicator),
        ):
            prevalence.extend(prevalence_rows(split_name, partition, rows, labels, indicators))
            for row, label, indicator in zip(rows, labels, indicators):
                year = str(feature_by_sha[row["sha256"]].get("metadata_year", ""))
                year_samples[(year, row["sha256"])] = (int(label), int(indicator))

        rule_pred, assignments = majority_rule_predictions(y_train, train_indicator, test_indicator)
        rule_metric = metrics(y_test, rule_pred.astype(float), threshold=0.5)
        rule_rows.append({
            "split_name": split_name,
            "rule": "train_group_majority_by_entry_presence",
            "train_group_assignment_no_entry": assignments[0],
            "train_group_assignment_entry": assignments[1],
            **{f"{key}": value for key, value in rule_metric.items() if key in {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "tp", "tn", "fp", "fn"}},
        })

        for model_name, x_train, x_test in (
            ("entry_indicator_only_logistic", train_indicator_features, test_indicator_features),
            ("entry_presence_only_logistic", train_presence, test_presence),
            ("compact_plus_entry_without_direct_flags_logistic", train_compact_flag_free, test_compact_flag_free),
        ):
            diagnostic_metrics = []
            for seed in args.seeds:
                probability = predict_logistic(
                    x_train, y_train, x_test, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                diagnostic_metrics.append(metrics(y_test, probability, threshold=THRESHOLD))
            model_rows.append({
                "split_name": split_name,
                "model": model_name,
                "feature_count": int(x_train.shape[1]),
                "threshold": THRESHOLD,
                "accuracy_mean": float(np.mean([m["accuracy"] for m in diagnostic_metrics])),
                "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in diagnostic_metrics])),
                "precision_mean": float(np.mean([m["precision"] for m in diagnostic_metrics])),
                "recall_mean": float(np.mean([m["recall"] for m in diagnostic_metrics])),
                "f1_mean": float(np.mean([m["f1"] for m in diagnostic_metrics])),
                "auc_mean": float(np.mean([auc_or_empty(m) for m in diagnostic_metrics])),
            })

        shuffled_metrics = []
        for shuffle_seed in args.label_shuffle_seeds:
            rng = np.random.RandomState(shuffle_seed)
            shuffled_labels = np.asarray(y_train, dtype=np.int64).copy()
            rng.shuffle(shuffled_labels)
            probability = predict_logistic(
                train_presence, shuffled_labels, test_presence, shuffle_seed,
                args.iterations, args.learning_rate, args.l2,
            )
            shuffled_metrics.append(metrics(y_test, probability, threshold=THRESHOLD))
        shuffle_rows.append({
            "split_name": split_name,
            "model": "entry_presence_only_logistic_train_labels_shuffled",
            "shuffle_count": len(shuffled_metrics),
            "threshold": THRESHOLD,
            "accuracy_mean": float(np.mean([m["accuracy"] for m in shuffled_metrics])),
            "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in shuffled_metrics])),
            "auc_mean": float(np.mean([auc_or_empty(m) for m in shuffled_metrics])),
        })

        summary_rows.append({
            "split_name": split_name,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_entry_present_rate": float(train_indicator.mean()) if len(train_indicator) else 0.0,
            "test_entry_present_rate": float(test_indicator.mean()) if len(test_indicator) else 0.0,
            "train_malicious_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "test_malicious_rate": float(y_test.mean()) if len(y_test) else 0.0,
        })

    prevalence_rows_by_year = []
    grouped_year_samples = defaultdict(list)
    for (year, _sha256), values in year_samples.items():
        grouped_year_samples[year].append(values)
    for year, samples in sorted(grouped_year_samples.items()):
        labels = np.asarray([value[0] for value in samples], dtype=np.int64)
        indicators = np.asarray([value[1] for value in samples], dtype=np.int64)
        present = indicators == 1
        prevalence_rows_by_year.append({
            "year": year,
            "row_count": len(samples),
            "malicious_rate": float(labels.mean()) if len(labels) else 0.0,
            "entry_present_row_count": int(present.sum()),
            "entry_present_rate": float(indicators.mean()) if len(indicators) else 0.0,
            "entry_present_malicious_rate": float(labels[present].mean()) if np.any(present) else 0.0,
        })

    presence_models = [
        row for row in model_rows
        if row["model"] == "entry_presence_only_logistic"
    ]
    model_mean_ba = float(np.mean([row["balanced_accuracy_mean"] for row in presence_models])) if presence_models else 0.0
    model_mean_auc = float(np.mean([row["auc_mean"] for row in presence_models])) if presence_models else 0.0
    rule_mean_ba = float(np.mean([row["balanced_accuracy"] for row in rule_rows])) if rule_rows else 0.0
    shuffle_mean_ba = float(np.mean([row["balanced_accuracy_mean"] for row in shuffle_rows])) if shuffle_rows else 0.0
    write_csv(output / "entry_shortcut_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(output / "entry_shortcut_prevalence.csv", list(prevalence[0]), prevalence)
    write_csv(output / "entry_shortcut_prevalence_by_year.csv", list(prevalence_rows_by_year[0]), prevalence_rows_by_year)
    write_csv(output / "entry_shortcut_rule_metrics.csv", list(rule_rows[0]), rule_rows)
    write_csv(output / "entry_shortcut_model_metrics.csv", list(model_rows[0]), model_rows)
    write_csv(output / "entry_shortcut_label_shuffle_metrics.csv", list(shuffle_rows[0]), shuffle_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P50_ENTRY_PRESENCE_SHORTCUT_AUDIT",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "fixed_threshold": THRESHOLD,
        "classifier": "custom_l2_logistic_for_presence_only_diagnostic",
        "presence_definition": {
            "features": [
                "global_mean_node_is_entry",
                "global_max_node_is_entry",
                "global_std_node_is_entry",
            ],
            "direct_indicator": "global_max_node_is_entry > 0.5",
            "component_type_information_available": False,
        },
        "entry_feature_dimensions": {
            "entry_presence": int(len(layout["entry_presence"])),
            "full_entry": int(layout["full_width"][0]),
            "direct_entry_flags": int(
                len(layout["entry_presence"]) + len(layout["role_node_entry_overlap"])
            ),
            "entry_without_direct_flags": int(
                layout["full_width"][0]
                - len(layout["entry_presence"])
                - len(layout["role_node_entry_overlap"])
            ),
        },
        "aggregate": {
            "presence_only_logistic_mean_ba": model_mean_ba,
            "presence_only_logistic_mean_auc": model_mean_auc,
            "entry_presence_majority_rule_mean_ba": rule_mean_ba,
            "entry_presence_label_shuffle_mean_ba": shuffle_mean_ba,
            "direct_flag_free_model_is_not_entry_free": True,
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "outputs": {
            "summary": str(output / "entry_shortcut_summary.csv"),
            "prevalence": str(output / "entry_shortcut_prevalence.csv"),
            "prevalence_by_year": str(output / "entry_shortcut_prevalence_by_year.csv"),
            "rule_metrics": str(output / "entry_shortcut_rule_metrics.csv"),
            "model_metrics": str(output / "entry_shortcut_model_metrics.csv"),
            "label_shuffle_metrics": str(output / "entry_shortcut_label_shuffle_metrics.csv"),
        },
        "blocking_reasons": [
            "component_type_entry_information_not_retained",
            "test_label_prevalence_is_post_hoc_only",
        ],
        "notes": [
            "This audit does not claim that Entry presence is causal evidence.",
            "A high presence-only score indicates shortcut risk, not a valid final feature.",
            "The label-shuffle control should be near chance and is not a model result.",
            "The current cache does not retain Activity/Service/Receiver/Provider type.",
            "Removing direct entry columns cannot remove indirect entry conditioning from the cached subgraph.",
            "No APK is reopened and no feature extraction is performed.",
        ],
    }
    (output / "entry_shortcut_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
