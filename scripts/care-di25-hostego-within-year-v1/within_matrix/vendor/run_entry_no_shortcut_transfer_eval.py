#!/usr/bin/env python3
"""Cross-source Entry evaluation with explicit shortcut removal.

The ordinary Entry vector contains useful semantic/context features, but it
also contains fields that can expose whether an APK has an Entry-derived
subgraph at all.  This evaluator reports both the original and strict
shortcut-removed variants under the same fixed threshold and transfer splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shortcut_removed_indices(width: int) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Return fields that can reveal Entry availability directly.

    For the role-stratified layout, entry_layout() is:
      global structure + per-role support/context + direct entry overlap.
    The last two global fields are log node/edge counts.  They are removed
    because an Entry-conditioned graph can make them a proxy for coverage.
    """

    layout = entry_layout()
    expected_width = int(layout["full_width"][0])
    if width != expected_width:
        raise ValueError(f"unexpected Entry width {width}; expected {expected_width}")

    direct_presence = [int(value) for value in layout["entry_presence"]]
    direct_role_overlap = [
        int(value) for value in layout["role_node_entry_overlap"]
    ]
    global_width = int(layout["global_width"][0])
    graph_size = [global_width - 2, global_width - 1]
    removed = sorted(set(direct_presence + direct_role_overlap + graph_size))
    keep = np.asarray([index for index in range(width) if index not in set(removed)], dtype=np.int64)
    groups = {
        "entry_presence": direct_presence,
        "role_node_entry_overlap": direct_role_overlap,
        "entry_conditioned_graph_size": graph_size,
    }
    return keep, groups


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache).expanduser().resolve(strict=True)
    feature_path = Path(args.features).expanduser().resolve(strict=True)
    schema_path = Path(args.schema).expanduser().resolve(strict=True)
    split_path = Path(args.splits).expanduser().resolve(strict=True)
    cache = read_cache(cache_path)
    feature_rows = read_csv_rows(feature_path)
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split_rows = read_split_rows(split_path)
    attach_audit_metadata(cache, feature_by_sha)
    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    layout = entry_layout()
    expected_width = int(layout["full_width"][0])
    keep, removed_groups = shortcut_removed_indices(expected_width)
    per_split: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        full_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, include_reliability=True
        )
        full_test, y_test = graph_matrix(
            test_rows, cache, evidence, entry_args, include_reliability=True
        )
        if full_train.shape[1] != expected_width:
            raise SystemExit(
                f"unexpected Entry width {full_train.shape[1]}; expected {expected_width}"
            )
        full_train, full_test = standardize_pair(full_train, full_test)
        no_shortcut_train = full_train[:, keep]
        no_shortcut_test = full_test[:, keep]
        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows, test_rows, feature_by_sha, schema
        )
        variants = {
            "compact_only": (compact_train, compact_test),
            "entry_full": (args.graph_weight * full_train, args.graph_weight * full_test),
            "entry_no_shortcut": (
                args.graph_weight * no_shortcut_train,
                args.graph_weight * no_shortcut_test,
            ),
            "compact_plus_entry_full": (
                np.concatenate([compact_train, args.graph_weight * full_train], axis=1),
                np.concatenate([compact_test, args.graph_weight * full_test], axis=1),
            ),
            "compact_plus_entry_no_shortcut": (
                np.concatenate([compact_train, args.graph_weight * no_shortcut_train], axis=1),
                np.concatenate([compact_test, args.graph_weight * no_shortcut_test], axis=1),
            ),
        }
        for variant, (x_train, x_test) in variants.items():
            seed_metrics: list[dict[str, float]] = []
            for seed in args.seeds:
                probability = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed=seed,
                    iterations=1500,
                    learning_rate=0.05,
                    l2=args.l2,
                )
                result = metrics(y_test, probability, threshold=0.5)
                seed_metrics.append(result)
                for row, label, prob in zip(test_rows, y_test, probability):
                    prediction_rows.append(
                        {
                            "variant": variant,
                            "split_name": split_name,
                            "sha256": row["sha256"],
                            "seed": seed,
                            "label": int(label),
                            "probability": float(prob),
                            "threshold": 0.5,
                            "predicted_label": int(prob >= 0.5),
                        }
                    )
            row: dict[str, object] = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": x_train.shape[1],
                "threshold": 0.5,
                "graph_weight": args.graph_weight,
                "l2": args.l2,
                "test_rows_used_for_training_or_selection": False,
            }
            for metric_name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"):
                row[f"{metric_name}_mean"] = float(
                    np.mean([result[metric_name] for result in seed_metrics])
                )
            per_split.append(row)

    summary: list[dict[str, object]] = []
    for variant in variants:
        rows = [row for row in per_split if row["variant"] == variant]
        ba = np.asarray([float(row["balanced_accuracy_mean"]) for row in rows])
        summary.append(
            {
                "variant": variant,
                "evaluated_split_count": len(rows),
                "mean_accuracy": float(np.mean([float(row["accuracy_mean"]) for row in rows])),
                "mean_balanced_accuracy": float(np.mean(ba)),
                "minimum_split_balanced_accuracy": float(np.min(ba)),
                "std_balanced_accuracy": float(np.std(ba)),
                "stability_score": float(np.mean(ba) - np.std(ba)),
                "mean_auc": float(np.mean([float(row["auc_mean"]) for row in rows])),
            }
        )
    summary.sort(key=lambda row: float(row["mean_balanced_accuracy"]), reverse=True)
    for index, row in enumerate(summary, start=1):
        row["rank"] = index
    summary = [
        {key: row[key] for key in ("rank", *[key for key in row if key != "rank"])}
        for row in summary
    ]

    write_csv(output_dir / "entry_no_shortcut_summary.csv", summary)
    write_csv(output_dir / "entry_no_shortcut_per_split.csv", per_split)
    write_csv(output_dir / "entry_no_shortcut_predictions.csv", prediction_rows)

    audit = {
        "schema_version": "1.0.0",
        "phase": "P54_ENTRY_SHORTCUT_REMOVAL_TRANSFER_EVALUATION",
        "formal_training_enabled": False,
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "features": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "schema": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "splits": str(split_path.resolve()),
        "splits_sha256": sha256_file(split_path),
        "fixed_threshold": 0.5,
        "graph_weight": args.graph_weight,
        "l2": args.l2,
        "seeds": args.seeds,
        "entry_full_feature_count": expected_width,
        "entry_no_shortcut_feature_count": int(len(keep)),
        "removed_feature_groups": removed_groups,
        "shortcut_removal_scope": [
            "direct_entry_presence_mean_max_std",
            "role_node_entry_overlap",
            "entry_conditioned_log_node_count",
            "entry_conditioned_log_edge_count",
        ],
        "not_removed": [
            "semantic_role_distribution",
            "role_context_statistics",
            "edge_type_statistics",
            "reliability_statistics",
        ],
        "strict_test_feature_isolation": True,
        "outputs": {
            "summary": str((output_dir / "entry_no_shortcut_summary.csv").resolve()),
            "per_split": str((output_dir / "entry_no_shortcut_per_split.csv").resolve()),
            "predictions": str((output_dir / "entry_no_shortcut_predictions.csv").resolve()),
        },
        "blocking_reasons": [
            "underlying_entry_cache_is_entry_conditioned",
            "this_is_not_a_full_graph_entry_free_view",
            "formal_training_config_not_generated",
        ],
        "notes": [
            "The shortcut-free variant removes explicit Entry availability and size proxies.",
            "The cache still originates from an Entry-conditioned graph view, so this is a strict ablation, not a claim of full Entry independence.",
            "Labels are used only after feature construction for evaluation.",
        ],
    }
    (output_dir / "entry_no_shortcut_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
