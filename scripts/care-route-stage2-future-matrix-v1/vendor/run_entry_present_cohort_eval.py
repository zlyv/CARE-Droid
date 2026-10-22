#!/usr/bin/env python3
"""Evaluate Entry increment only among APKs with at least one bound entry node.

Restricting both training and evaluation to the entry-present cohort makes the
binary ``has entry`` indicator constant.  The experiment therefore tests
whether the remaining entry-guided role/structure representation adds value
beyond Compact after removing the strongest observed shortcut.

No APK is opened.  Threshold is fixed at 0.5; all fitting and scaling use the
filtered outer-training partition only.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    entry_model_args,
)
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
from run_entry_increment_ablation import entry_layout  # noqa: E402
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


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def payload_has_entry(cache_row: dict) -> bool:
    for values in cache_row.get("payload", {}).get("node_base_features", []) or []:
        if len(values) > 3 and float(values[3]) > 0.5:
            return True
    return False


def label_of_feature(row: dict) -> int:
    value = str(row.get("metadata_label", "")).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def direct_flag_free_indices(layout: dict[str, np.ndarray]) -> np.ndarray:
    direct = sorted(
        set(layout["entry_presence"].tolist())
        | set(layout["role_node_entry_overlap"].tolist())
    )
    return np.setdiff1d(
        np.arange(int(layout["full_width"][0]), dtype=np.int64),
        np.asarray(direct, dtype=np.int64),
    )


def summarize(per_split: list[dict]) -> list[dict]:
    output = []
    for variant in sorted({row["variant"] for row in per_split}):
        rows = [row for row in per_split if row["variant"] == variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        output.append({
            "variant": variant,
            "evaluated_split_count": len(rows),
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_split_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    output.sort(key=lambda row: (-row["mean_balanced_accuracy"], -row["mean_auc"]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--entry-weight", type=float, default=0.5)
    args = parser.parse_args()

    cache_path = args.entry_cache.expanduser().resolve(strict=True)
    feature_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    split_path = args.splits.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(cache_path)
    feature_rows = read_csv_rows(feature_path)
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split_rows = read_split_rows(split_path)
    attach_audit_metadata(cache, feature_by_sha)
    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    layout = entry_layout()
    keep_without_direct_flags = direct_flag_free_indices(layout)
    per_split = []
    cohort_rows = []

    for split_name in sorted(split_rows):
        original_train = split_rows[split_name]["train"]
        original_test = split_rows[split_name]["test"]
        train_rows = [row for row in original_train if payload_has_entry(cache[row["sha256"]])]
        test_rows = [row for row in original_test if payload_has_entry(cache[row["sha256"]])]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty entry-present cohort for {split_name}")
        train_labels = [label_of_feature(feature_by_sha[row["sha256"]]) for row in train_rows]
        test_labels = [label_of_feature(feature_by_sha[row["sha256"]]) for row in test_rows]
        if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
            raise SystemExit(f"entry-present cohort lacks both classes: {split_name}")

        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(train_rows, cache, evidence, entry_args, True)
        entry_test, y_test = graph_matrix(test_rows, cache, evidence, entry_args, True)
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows, test_rows, feature_by_sha, schema
        )
        flag_free_train = entry_train[:, keep_without_direct_flags]
        flag_free_test = entry_test[:, keep_without_direct_flags]
        variants = {
            "compact_only_entry_present": (compact_train, compact_test),
            "entry_only_entry_present": (
                args.entry_weight * entry_train,
                args.entry_weight * entry_test,
            ),
            "compact_plus_full_entry_entry_present": (
                np.concatenate([compact_train, args.entry_weight * entry_train], axis=1),
                np.concatenate([compact_test, args.entry_weight * entry_test], axis=1),
            ),
            "compact_plus_entry_without_direct_flags_entry_present": (
                np.concatenate([compact_train, args.entry_weight * flag_free_train], axis=1),
                np.concatenate([compact_test, args.entry_weight * flag_free_test], axis=1),
            ),
        }
        for variant, (x_train, x_test) in variants.items():
            values = []
            for seed in args.seeds:
                probability = predict_logistic(
                    x_train, y_train, x_test, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                values.append(metrics(y_test, probability, threshold=THRESHOLD))
            per_split.append({
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(x_train.shape[1]),
                "threshold": THRESHOLD,
                "accuracy_mean": float(np.mean([v["accuracy"] for v in values])),
                "balanced_accuracy_mean": float(np.mean([v["balanced_accuracy"] for v in values])),
                "precision_mean": float(np.mean([v["precision"] for v in values])),
                "recall_mean": float(np.mean([v["recall"] for v in values])),
                "f1_mean": float(np.mean([v["f1"] for v in values])),
                "auc_mean": float(np.mean([v["auc"] for v in values])),
            })
        cohort_rows.append({
            "split_name": split_name,
            "original_train_count": len(original_train),
            "entry_present_train_count": len(train_rows),
            "entry_present_train_benign": Counter(train_labels)[0],
            "entry_present_train_malicious": Counter(train_labels)[1],
            "original_test_count": len(original_test),
            "entry_present_test_count": len(test_rows),
            "entry_present_test_benign": Counter(test_labels)[0],
            "entry_present_test_malicious": Counter(test_labels)[1],
            "compact_feature_count": len(compact_names),
        })
        print(
            f"completed {split_name}: train={len(train_rows)} test={len(test_rows)}",
            flush=True,
        )

    summary = summarize(per_split)
    write_csv(output / "entry_present_cohort_summary.csv", list(summary[0]), summary)
    write_csv(output / "entry_present_cohort_per_split.csv", list(per_split[0]), per_split)
    write_csv(output / "entry_present_cohort_composition.csv", list(cohort_rows[0]), cohort_rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P52_ENTRY_PRESENT_MATCHED_COHORT_EVALUATION",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "cohort_rule": "global_max_node_is_entry_equals_one",
        "fixed_threshold": THRESHOLD,
        "entry_weight": args.entry_weight,
        "ranked_variants": summary,
        "outputs": {
            "summary": str(output / "entry_present_cohort_summary.csv"),
            "per_split": str(output / "entry_present_cohort_per_split.csv"),
            "composition": str(output / "entry_present_cohort_composition.csv"),
        },
        "blocking_reasons": [
            "retrospective_future_year_diagnostic_not_for_selection",
            "entry_present_cohort_is_label_imbalanced",
            "entry_cache_remains_entry_conditioned",
        ],
        "notes": [
            "The binary has-entry shortcut is constant within every evaluated cohort.",
            "Balanced accuracy is reported because the filtered cohorts are imbalanced.",
            "Removing direct flags does not remove indirect entry conditioning from cache construction.",
            "No APK is opened.",
        ],
    }
    (output / "entry_present_cohort_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(row["variant"], f'BA={row["mean_balanced_accuracy"]:.4f}', flush=True)


if __name__ == "__main__":
    main()
