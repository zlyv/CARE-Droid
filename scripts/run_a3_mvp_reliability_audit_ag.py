#!/usr/bin/env python3
"""Fit auditable pilot A3-MVP reliability components from the cache.

Only rows marked ``train`` in the supplied split manifest contribute to D/T/
one-minus-F.  Test rows are inspected only for an unseen-node audit and never
for component values.  This script intentionally produces a provisional
pilot artifact; it does not enable formal training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    _iter_node_observations,
    component_rows,
    fit_training_fold_components,
    read_cache,
    read_split_manifest,
)


def write_component_rows(path: Path, split_name: str, evidence: dict) -> int:
    count = 0
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        for row in component_rows(evidence):
            row = {"split_name": split_name, **row}
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def node_ids_for_rows(cache_rows: dict, shas: set[str]) -> set[str]:
    result = set()
    for sha256 in sorted(shas):
        row = cache_rows.get(sha256)
        if row is None:
            continue
        result.update(node_id for node_id, _mapped in _iter_node_observations(row))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-fold-only pilot A3-MVP reliability audit"
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha-prior", required=True, type=float)
    parser.add_argument("--kappa-support", required=True, type=float)
    parser.add_argument("--kappa-context", required=True, type=float)
    parser.add_argument("--r-min", required=True, type=float)
    parser.add_argument("--r-max", required=True, type=float)
    parser.add_argument("--split", action="append", dest="split_names")
    args = parser.parse_args()

    definition = PilotReliabilityDefinition(
        alpha_prior=args.alpha_prior,
        kappa_support=args.kappa_support,
        kappa_context=args.kappa_context,
        r_min=args.r_min,
        r_max=args.r_max,
    )
    definition.validate()
    cache_path = args.cache.expanduser().resolve()
    split_path = args.splits.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_rows = read_cache(cache_path)
    split_sets = read_split_manifest(split_path)
    requested = set(args.split_names or split_sets)
    unknown = requested - set(split_sets)
    if unknown:
        raise SystemExit(f"unknown split names: {sorted(unknown)}")

    component_path = output / "a3_mvp_reliability_components.jsonl.gz"
    if component_path.exists():
        component_path.unlink()
    summaries = []
    status_counts = Counter()
    for split_name in sorted(requested):
        partitions = split_sets[split_name]
        train_shas = partitions["train"]
        test_shas = partitions["test"]
        if not train_shas or not test_shas:
            raise SystemExit(f"empty train/test partition: {split_name}")
        missing_train = sorted(train_shas - set(cache_rows))
        if missing_train:
            raise SystemExit(
                f"cache missing {len(missing_train)} training rows for {split_name}"
            )
        evidence = fit_training_fold_components(cache_rows, train_shas, definition)
        test_nodes = node_ids_for_rows(cache_rows, test_shas)
        train_nodes = set(evidence["components_by_node_id"])
        component_row_count = write_component_rows(
            component_path, split_name, evidence
        )
        status_counts[evidence["definition_status"]] += 1
        summaries.append(
            {
                "split_name": split_name,
                "train_manifest_count": len(train_shas),
                "train_cache_count": evidence["training_row_count"],
                "test_manifest_count": len(test_shas),
                "component_node_count": evidence["node_count"],
                "component_row_count": component_row_count,
                "test_only_node_count": len(test_nodes - train_nodes),
                "test_rows_used_for_components": False,
                "training_years": evidence["training_years"],
            }
        )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_A3_MVP_TRAINING_FOLD_RELIABILITY_AUDIT",
        "formal_training_enabled": False,
        "cache": str(cache_path),
        "split_manifest": str(split_path),
        "cache_row_count": len(cache_rows),
        "definition_status": "PILOT_PROVISIONAL_NOT_FORMAL",
        "definition": {
            "D": "(distinct training APK support + alpha) / (support + alpha + kappa_support)",
            "T": "(distinct training-year support + alpha) / (training-year count + 2*alpha)",
            "one_minus_F": "(mapped-role occurrences + alpha) / (all node occurrences + alpha + kappa_context)",
            "alpha_prior": definition.alpha_prior,
            "kappa_support": definition.kappa_support,
            "kappa_context": definition.kappa_context,
            "r_min": definition.r_min,
            "r_max": definition.r_max,
        },
        "split_summaries": summaries,
        "status_counts": dict(status_counts),
        "outputs": {"components": str(component_path)},
        "blocking_reasons": [
            "pilot_reliability_definition_not_resolved_by_nested_validation",
            "formal_training_config_not_resolved",
        ],
        "notes": [
            "Only outer training rows contribute component values.",
            "Test rows are used only for an unseen-node audit count.",
            "Missing test-node components must use the explicit training prior during tensor construction.",
            "The pilot definitions must not be reported as final protocol values.",
        ],
    }
    (output / "a3_mvp_reliability_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
