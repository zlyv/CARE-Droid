#!/usr/bin/env python3
"""Create deterministic leakage-blocked within-year cross-validation splits.

Each requested year is split independently.  Near-duplicate/blocking groups
remain intact, labels are used only to balance fold composition, and every APK
appears in exactly one test fold for its year.  This is a retrospective
same-year separability protocol, not a temporal generalization protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_label(value: str) -> str:
    value = str(value or "").strip().casefold()
    if value == "benign":
        return "benign"
    if value in {"malicious", "malware"}:
        return "malicious"
    raise ValueError(f"unsupported label: {value!r}")


def infer_column(rows: list[dict], explicit: str | None, candidates: tuple[str, ...]) -> str:
    if not rows:
        raise ValueError("input CSV is empty")
    if explicit:
        if explicit not in rows[0]:
            raise ValueError(f"column not found: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise ValueError(f"none of the candidate columns exist: {candidates}")


def allocate_group_folds(items: list[dict], fold_count: int) -> list[list[int]]:
    """Greedily balance fold size and positive count without splitting groups."""

    groups: dict[str, list[int]] = defaultdict(list)
    positives: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    for index, item in enumerate(items):
        group = item["blocking_group"]
        groups[group].append(index)
        positives[group] += item["label"] == "malicious"
        negatives[group] += item["label"] == "benign"
    actual = min(fold_count, len(groups))
    if actual < 2:
        raise ValueError("a year needs at least two blocking groups")
    folds = [[] for _ in range(actual)]
    sizes = [0] * actual
    positive_sizes = [0] * actual
    negative_sizes = [0] * actual
    target_size = len(items) / actual
    target_positive = sum(positives.values()) / actual
    target_negative = sum(negatives.values()) / actual
    assigned: set[str] = set()

    def assign(group: str, fold: int) -> None:
        indices = groups[group]
        folds[fold].extend(indices)
        sizes[fold] += len(indices)
        positive_sizes[fold] += positives[group]
        negative_sizes[fold] += negatives[group]
        assigned.add(group)

    # Guarantee that every test fold contains both classes before balancing.
    for counts, fold_counts in (
        (positives, positive_sizes),
        (negatives, negative_sizes),
    ):
        for fold in [index for index, value in enumerate(fold_counts) if value == 0]:
            candidates = [
                group for group in groups
                if group not in assigned and counts[group] > 0
            ]
            if not candidates:
                raise ValueError("not enough distinct blocking groups to put both classes in every fold")
            chosen = min(
                candidates,
                key=lambda group: (-counts[group], -len(groups[group]), group),
            )
            assign(chosen, fold)

    ordered = sorted(
        ((group, indices) for group, indices in groups.items() if group not in assigned),
        key=lambda pair: (-len(pair[1]), -max(positives[pair[0]], negatives[pair[0]]), pair[0]),
    )
    for group, indices in ordered:
        chosen = min(
            range(actual),
            key=lambda fold: (
                abs(positive_sizes[fold] + positives[group] - target_positive),
                abs(negative_sizes[fold] + negatives[group] - target_negative),
                abs(sizes[fold] + len(indices) - target_size),
                fold,
            ),
        )
        assign(group, chosen)

    for fold in folds:
        labels = {items[index]["label"] for index in fold}
        if labels != {"benign", "malicious"}:
            raise ValueError("a test fold contains only one class")
    return folds


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--blocking", required=True, type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019, 2020, 2021])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    feature_path = args.features.expanduser().resolve(strict=True)
    blocking_path = args.blocking.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    features = read_csv(feature_path)
    blocking = read_csv(blocking_path)
    group_column = infer_column(
        blocking,
        args.group_column,
        ("near_duplicate_block_id", "block_id", "blocking_group", "component_id"),
    )
    group_by_sha = {
        str(row.get("sha256", "")).strip().casefold(): str(row.get(group_column, "")).strip()
        for row in blocking
        if str(row.get("sha256", "")).strip()
    }

    years = sorted(set(args.years))
    by_year: dict[int, list[dict]] = defaultdict(list)
    missing_group_count = 0
    for row in features:
        sha = str(row.get("sha256", "")).strip().casefold()
        if not sha:
            raise SystemExit("feature row has an empty SHA256")
        try:
            year = int(str(row.get("metadata_year", "")).strip())
        except ValueError as exc:
            raise SystemExit(f"invalid year for {sha}") from exc
        if year not in years:
            continue
        try:
            label = normalize_label(row.get("metadata_label", ""))
        except ValueError as exc:
            raise SystemExit(f"{sha}: {exc}") from exc
        group = group_by_sha.get(sha, "")
        if not group:
            group = f"sha256:{sha}"
            missing_group_count += 1
        by_year[year].append({
            "sha256": sha,
            "year": year,
            "label": label,
            "blocking_group": group,
        })

    manifest_rows: list[dict] = []
    summaries = []
    for year in years:
        items = sorted(by_year.get(year, []), key=lambda row: row["sha256"])
        if not items:
            raise SystemExit(f"no feature rows for year {year}")
        folds = allocate_group_folds(items, args.folds)
        all_indices = set(range(len(items)))
        seen_test: set[int] = set()
        for fold_number, test_indices_list in enumerate(folds, start=1):
            test_indices = set(test_indices_list)
            train_indices = all_indices - test_indices
            seen_test.update(test_indices)
            train_groups = {items[index]["blocking_group"] for index in train_indices}
            test_groups = {items[index]["blocking_group"] for index in test_indices}
            if train_groups & test_groups:
                raise SystemExit(f"blocking-group overlap: year={year} fold={fold_number}")
            split_name = f"within_{year}_fold_{fold_number}_of_{len(folds)}"
            for partition, indices in (("train", train_indices), ("test", test_indices)):
                for index in sorted(indices):
                    item = items[index]
                    manifest_rows.append({
                        "split_name": split_name,
                        "partition": partition,
                        **item,
                        "exclusion_reason": "",
                    })
            summaries.append({
                "year": year,
                "fold": fold_number,
                "split_name": split_name,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "train_class_counts": dict(Counter(items[index]["label"] for index in train_indices)),
                "test_class_counts": dict(Counter(items[index]["label"] for index in test_indices)),
                "blocking_group_overlap_count": 0,
            })
        if seen_test != all_indices:
            raise SystemExit(f"not every row appears in one test fold for year {year}")

    manifest_path = output / "split_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["split_name", "partition", "sha256", "year", "label", "blocking_group", "exclusion_reason"],
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P41_GROUPED_WITHIN_YEAR_CROSS_VALIDATION_SPLITS",
        "formal_training_enabled": False,
        "protocol_status": "RETROSPECTIVE_WITHIN_YEAR_SEPARABILITY_ONLY",
        "years": years,
        "requested_folds": args.folds,
        "blocking_group_column": group_column,
        "missing_blocking_group_fallback_count": missing_group_count,
        "year_counts": {str(year): len(by_year[year]) for year in years},
        "fold_summaries": summaries,
        "inputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "blocking": {"path": str(blocking_path), "sha256": sha256_file(blocking_path)},
        },
        "outputs": {
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": sha256_file(manifest_path),
        },
        "notes": [
            "Labels are used only to balance fold composition.",
            "Blocking groups never cross train and test inside a fold.",
            "Every selected APK appears in exactly one test fold per year.",
            "This protocol measures same-year separability, not temporal transfer.",
        ],
    }
    audit_path = output / "within_year_split_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
