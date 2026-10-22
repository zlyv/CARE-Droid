#!/usr/bin/env python3
"""Reproducible hard audit for the fixed 799-APK 2017 lineage and folds."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value):
    return str(value).strip().lower()


def safe_int(value):
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lineage = read(args.lineage); folds = read(args.folds); features = read(args.features)
    lineage_ids = [norm(row.get("sha256")) for row in lineage]
    fold_ids = [norm(row.get("sha256")) for row in folds]
    labels = [int(row["label"]) for row in folds]
    fold_values = [int(row["fold"]) for row in folds]
    years = [str(row.get("metadata_year", "")).strip() for row in lineage]
    metadata_labels = [norm(row.get("metadata_label")) for row in lineage]
    paths = [Path(row.get("absolute_path", "")) for row in lineage]
    feature_ids = [norm(row.get("sha256")) for row in features]
    totals = {
        "mapped_api_node_total": sum(safe_int(row.get("mapped_api_node_count")) for row in features),
        "external_node_total": sum(safe_int(row.get("external_node_count")) for row in features),
        "sensitive_edge_total": sum(safe_int(row.get("sensitive_edge_count")) for row in features),
    }
    checks = {
        "lineage_rows_799": len(lineage) == 799,
        "lineage_unique_799": len(set(lineage_ids)) == 799 and "" not in lineage_ids,
        "fold_rows_unique_799": len(folds) == len(set(fold_ids)) == 799 and "" not in fold_ids,
        "lineage_fold_ids_identical": set(lineage_ids) == set(fold_ids),
        "labels_400_399": Counter(labels) == Counter({0: 400, 1: 399}),
        "metadata_labels_400_399": Counter(metadata_labels) == Counter({"benign": 400, "malware": 399}),
        "fixed_fold_sizes": sorted(Counter(fold_values).values(), reverse=True) == [160, 160, 160, 160, 159],
        "fixed_fold_ids": set(fold_values) == {1, 2, 3, 4, 5},
        "source_year_only_2017": set(years) == {"2017"},
        "all_apk_paths_exist": all(path.is_file() for path in paths),
        "compact_features_exact_unique_799": len(features) == len(set(feature_ids)) == 799 and "" not in feature_ids,
        "compact_features_exact_lineage": set(feature_ids) == set(lineage_ids),
        "compact_features_all_ok": all(str(row.get("status", "")).strip() == "OK" for row in features),
        "canonical_mapped_api_total": totals["mapped_api_node_total"] == 35757,
        "canonical_external_node_total": totals["external_node_total"] == 5874100,
        "canonical_sensitive_edge_total": totals["sensitive_edge_total"] == 200961,
    }
    passed = all(checks.values())
    result = {"passed": passed, "checks": checks, "label_counts": dict(Counter(labels)),
              "fold_counts": dict(sorted(Counter(fold_values).items())), "years": sorted(set(years)),
              "canonical_recomputed_totals": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("source lineage gate failed; extraction and OOF are blocked")
    print("SOURCE_LINEAGE_2017_GATE_PASS")


if __name__ == "__main__":
    main()
