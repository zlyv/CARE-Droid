#!/usr/bin/env python3
"""Evaluate a manifest-exposure module on chronological 2017 folds.

The core model remains the existing entry-graph plus Compact model.  This
script adds a label-free AndroidManifest exposure surface as a serial feature
module, rather than adding the endpoint-centred graph view.  Two additions
are compared:

* ``manifest_exposure_raw``: numeric exposure/protection fields extracted from
  the manifest;
* ``manifest_exposure_surface``: role-conditioned exposure features, e.g.
  global semantic-role support modulated by effective export and protection.

All constant filtering and standardization are fitted on each chronological
2017 training fold.  The threshold is fixed at 0.5 and no future-year row is
read by this development evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from run_graph_pool_diagnostic import (  # noqa: E402
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
AUDIT_COLUMNS = {
    "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
    "metadata_label", "package_name", "certificate_sha256", "version_code",
    "status", "manifest_parse_ok", "error_type", "error_message",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row.get(name, "0"))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def surface_features(
    row: dict[str, str],
    base_names: list[str],
    exposure_names: list[str],
) -> dict[str, float]:
    """Create role-conditioned manifest exposure features without labels."""

    exported = safe_float(row, "manifest_exposure__effective_exported_ratio")
    triggered = safe_float(row, "manifest_exposure__intent_filter_component_ratio")
    protected = safe_float(row, "manifest_exposure__protected_component_ratio")
    exported = float(np.clip(exported, 0.0, 1.0))
    triggered = float(np.clip(triggered, 0.0, 1.0))
    protected = float(np.clip(protected, 0.0, 1.0))
    unprotected = 1.0 - protected

    # The product represents externally exposed and weakly protected surface;
    # the average is retained as a smoother trigger prior.
    open_surface = exported * unprotected
    trigger_surface = triggered * unprotected
    result = {
        "surface__effective_exported_ratio": exported,
        "surface__intent_filter_component_ratio": triggered,
        "surface__protected_component_ratio": protected,
        "surface__open_exported_ratio": open_surface,
        "surface__open_trigger_ratio": trigger_surface,
        "surface__exported_trigger_product": exported * triggered,
        "surface__component_count_log": math.log1p(
            safe_float(row, "manifest_exposure__component_count")
        ),
        "surface__intent_filter_count_log": safe_float(
            row, "manifest_exposure__intent_filter_count_log"
        ),
        "surface__custom_permission_count_log": math.log1p(
            safe_float(row, "manifest_exposure__custom_permission_count")
        ),
    }

    # Retain a small raw manifest subset in the raw variant only.  This loop
    # keeps the set explicit and deterministic for schema auditing.
    del exposure_names

    for name in base_names:
        if name.startswith("role_node__"):
            role = name[len("role_node__"):]
            value = math.log1p(max(0.0, safe_float(row, name)))
            result[f"surface__role_node__{role}__open"] = value * open_surface
            result[f"surface__role_node__{role}__triggered"] = value * trigger_surface
            result[f"surface__role_node__{role}__protected"] = value * protected
        elif name.startswith("role_edge__"):
            role = name[len("role_edge__"):]
            value = math.log1p(max(0.0, safe_float(row, name)))
            result[f"surface__role_edge__{role}__open"] = value * open_surface
            result[f"surface__role_edge__{role}__triggered"] = value * trigger_surface
            result[f"surface__role_edge__{role}__protected"] = value * protected
    return result


def select_and_standardize_pair(
    fit_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    feature_by_sha: dict[str, dict[str, str]],
    names: list[str],
    derived: bool = False,
    base_names: list[str] | None = None,
    exposure_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    base_names = base_names or []
    exposure_names = exposure_names or []

    def values(row: dict[str, str]) -> dict[str, float]:
        source = feature_by_sha[row["sha256"]]
        if derived:
            return surface_features(source, base_names, exposure_names)
        return {name: safe_float(source, name) for name in names}

    fit_table = [values(row) for row in fit_rows]
    eval_table = [values(row) for row in eval_rows]
    selected = []
    for name in names:
        fit_values = np.asarray([table.get(name, 0.0) for table in fit_table])
        if np.all(np.isfinite(fit_values)) and np.unique(fit_values).size > 1:
            selected.append(name)
    x_fit = np.asarray(
        [[table.get(name, 0.0) for name in selected] for table in fit_table],
        dtype=np.float64,
    )
    x_eval = np.asarray(
        [[table.get(name, 0.0) for name in selected] for table in eval_table],
        dtype=np.float64,
    )
    if not selected:
        x_fit = np.zeros((len(fit_rows), 0), dtype=np.float64)
        x_eval = np.zeros((len(eval_rows), 0), dtype=np.float64)
    mean = x_fit.mean(axis=0) if x_fit.shape[1] else np.zeros(0)
    scale = x_fit.std(axis=0) if x_fit.shape[1] else np.zeros(0)
    scale[scale < 1e-12] = 1.0
    return (x_fit - mean) / scale, (x_eval - mean) / scale, len(selected)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--l2-values", nargs="+", type=float, default=[0.003, 0.01, 0.03])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.graph_weight <= 0.0 or any(value <= 0.0 for value in args.l2_values):
        raise SystemExit("graph weight and l2 values must be positive")
    if not args.seeds:
        raise SystemExit("at least one seed is required")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = read_cache(paths["entry_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)

    selected_shas = {
        row["sha256"]
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    }
    missing_cache = selected_shas - set(cache)
    missing_features = selected_shas - set(feature_by_sha)
    if missing_cache or missing_features:
        raise SystemExit(
            "selected rows missing from inputs: "
            f"cache={len(missing_cache)} features={len(missing_features)}"
        )
    selected_years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    }
    if selected_years != {"2017"}:
        raise SystemExit(
            "this evaluator is restricted to 2017 development rows; "
            f"observed years={sorted(selected_years)}"
        )

    all_fields = list(feature_rows[0]) if feature_rows else []
    schema_fields = list(schema.get("feature_fields", all_fields))
    base_names = sorted(
        name for name in schema_fields
        if name not in AUDIT_COLUMNS and not name.startswith("manifest_exposure__")
    )
    exposure_names = sorted(
        name for name in all_fields
        if name.startswith("manifest_exposure__") and name != "manifest_exposure__parse_ok"
    )
    if not exposure_names:
        raise SystemExit(
            "features do not contain manifest exposure fields; run the P23 "
            "manifest extraction first"
        )
    surface_names = sorted(
        surface_features(feature_rows[0], base_names, exposure_names)
    )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = [
        ("entry_compact_control", None),
        ("entry_compact_manifest_exposure", "raw"),
        ("entry_compact_exposure_surface", "surface"),
    ]
    split_summaries: dict[str, list[dict]] = {
        f"{name}__l2_{l2:g}": []
        for name, _ in variants for l2 in args.l2_values
    }

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        if not train_rows or not validation_rows:
            raise SystemExit(f"empty 2017 partition: {split_name}")
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, True
        )
        entry_validation, y_validation = graph_matrix(
            validation_rows, cache, evidence, entry_args, True
        )
        entry_train, entry_validation = standardize_pair(
            entry_train, entry_validation
        )
        compact_train, compact_validation, compact_count = select_and_standardize_pair(
            train_rows, validation_rows, feature_by_sha, base_names
        )
        exposure_train, exposure_validation, exposure_count = select_and_standardize_pair(
            train_rows, validation_rows, feature_by_sha, exposure_names
        )
        surface_train, surface_validation, surface_count = select_and_standardize_pair(
            train_rows, validation_rows, feature_by_sha, surface_names,
            derived=True, base_names=base_names, exposure_names=exposure_names,
        )

        matrices = {
            "entry_compact_control": (compact_train, compact_validation, 0),
            "entry_compact_manifest_exposure": (
                np.concatenate([compact_train, exposure_train], axis=1),
                np.concatenate([compact_validation, exposure_validation], axis=1),
                exposure_count,
            ),
            "entry_compact_exposure_surface": (
                np.concatenate([compact_train, surface_train], axis=1),
                np.concatenate([compact_validation, surface_validation], axis=1),
                surface_count,
            ),
        }
        for name, _ in variants:
            extra_train, extra_validation, extra_count = matrices[name]
            x_train = np.concatenate([args.graph_weight * entry_train, extra_train], axis=1)
            x_validation = np.concatenate(
                [args.graph_weight * entry_validation, extra_validation], axis=1
            )
            for l2 in args.l2_values:
                key = f"{name}__l2_{l2:g}"
                seed_metrics = []
                for seed in args.seeds:
                    weights, bias = fit_logistic(
                        x_train, y_train, seed,
                        iterations=args.iterations,
                        learning_rate=args.learning_rate,
                        l2=l2,
                    )
                    probability = 1.0 / (
                        1.0 + np.exp(-np.clip(x_validation @ weights + bias, -40.0, 40.0))
                    )
                    seed_metrics.append(metrics(y_validation, probability, THRESHOLD))
                summary = {
                    "split_name": split_name,
                    "variant": name,
                    "l2": l2,
                    "train_count": len(train_rows),
                    "validation_count": len(validation_rows),
                    "feature_count": int(x_train.shape[1]),
                    "compact_feature_count": int(compact_count),
                    "exposure_feature_count": int(extra_count),
                    "threshold": THRESHOLD,
                }
                for metric_name in (
                    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
                ):
                    summary[f"{metric_name}_mean"] = float(np.mean([
                        item[metric_name] for item in seed_metrics
                    ]))
                split_summaries[key].append(summary)
        print(f"completed {split_name}", flush=True)

    ranked = []
    for key, rows in split_summaries.items():
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        auc = np.asarray([row["auc_mean"] for row in rows])
        acc = np.asarray([row["accuracy_mean"] for row in rows])
        ranked.append({
            "variant": rows[0]["variant"],
            "l2": rows[0]["l2"],
            "mean_accuracy": float(acc.mean()),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(auc.mean()),
            "feature_counts": sorted({row["feature_count"] for row in rows}),
        })
    ranked.sort(key=lambda row: (
        -row["stability_score"], -row["mean_balanced_accuracy"], row["variant"], row["l2"]
    ))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "manifest_exposure_2017_summary.csv"
    write_csv(
        summary_path,
        [
            "rank", "variant", "l2", "feature_counts", "mean_accuracy",
            "mean_balanced_accuracy", "minimum_fold_balanced_accuracy",
            "std_balanced_accuracy", "stability_score", "mean_auc",
        ],
        ranked,
    )
    per_split_path = output / "manifest_exposure_2017_per_split.csv"
    write_csv(
        per_split_path,
        [
            "variant", "l2", "split_name", "train_count", "validation_count",
            "feature_count", "compact_feature_count", "exposure_feature_count",
            "threshold", "accuracy_mean", "balanced_accuracy_mean",
            "precision_mean", "recall_mean", "f1_mean", "auc_mean",
        ],
        [row for rows in split_summaries.values() for row in rows],
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P23_MANIFEST_EXPOSURE_2017_ROLLING_DEVELOPMENT_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(selected_years),
        "fixed_threshold": THRESHOLD,
        "graph_weight": args.graph_weight,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "module": {
            "raw_manifest_exposure_features": exposure_names,
            "surface_feature_count": len(surface_names),
            "label_free": True,
            "call_graph_view_added": False,
        },
        "classifier": {
            "type": "custom_l2_logistic",
            "l2_values": args.l2_values,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
        },
        "selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(per_split_path),
        },
        "blocking_reasons": [
            "future_years_not_evaluated_in_this_stage",
            "audited_first_seen_utc_not_joined",
            "formal_training_config_not_generated",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted.",
            "Manifest exposure fields are extracted without labels and without call graphs.",
            "All filtering and standardization use the corresponding training fold only.",
            "The classification threshold is fixed at 0.5.",
        ],
    }
    audit_path = output / "manifest_exposure_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("rank variant l2 mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"], row["l2"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
