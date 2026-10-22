#!/usr/bin/env python3
"""Build and hard-audit the frozen 2018--2021 CARE future lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


YEARS = (2018, 2019, 2020, 2021)
EXPECTED_BY_YEAR = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED = sum(EXPECTED_BY_YEAR.values())


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def norm(value) -> str:
    return str(value or "").strip().lower()


def first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def remap_path(value: str, storage_root: Path) -> Path:
    """Resolve historical /home dataset paths against the mounted HDD copy."""
    text = str(value or "").strip()
    prefixes = (
        "/home/boyuanjie/datasets/",
        "/home/boyuanjie/projects/tasa-droid/datasets/",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return storage_root / "datasets" / text[len(prefix):]
    path = Path(text)
    if not path.is_absolute():
        return storage_root / "datasets" / path
    return path


def normalize_label(value: str) -> str:
    value = norm(value)
    if value in {"0", "0.0", "benign", "goodware", "good"}:
        return "benign"
    if value in {"1", "1.0", "malware", "malicious", "badware", "bad"}:
        return "malware"
    return ""


def truthy(value) -> bool:
    return norm(value) in {"1", "1.0", "true", "yes", "y"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sha256", "absolute_path", "source_id", "lineage_id",
        "metadata_year", "metadata_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--storage-root", type=Path, default=Path(
        "/mnt/HDD1/boyuanjie/tasa-droid-storage"
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    input_path = args.manifest if args.manifest else args.features
    if input_path is None or not input_path.is_file():
        raise SystemExit("an existing --manifest or --features input is required")
    _, feature_rows = read_csv(input_path)
    reference_path = None
    reference_rows = []
    reference_by_sha = {}
    if args.reference:
        if not args.reference.is_file():
            raise SystemExit("--reference does not exist: " + str(args.reference))
        reference_path = str(args.reference.resolve())
        _, reference_rows = read_csv(args.reference)
        for reference_row in reference_rows:
            reference_sha = norm(first(reference_row, ("sha256", "apk_id", "sample_id")))
            if reference_sha in reference_by_sha:
                raise SystemExit("duplicate SHA in frozen future reference: " + reference_sha)
            reference_by_sha[reference_sha] = reference_row
    status_by_sha = {}
    duplicate_status_ids = []
    if args.status:
        if not args.status.is_file():
            raise SystemExit("--status does not exist: " + str(args.status))
        _, status_rows = read_csv(args.status)
        for status_row in status_rows:
            status_sha = norm(first(status_row, ("sha256", "sample_id", "apk_id")))
            if not status_sha:
                continue
            if status_sha in status_by_sha:
                duplicate_status_ids.append(status_sha)
            else:
                status_by_sha[status_sha] = status_row
    grouped: dict[str, list[dict]] = defaultdict(list)
    malformed = []
    status_excluded = []
    status_missing = []
    for index, row in enumerate(feature_rows, start=2):
        year_text = first(row, ("metadata_year", "year"))
        try:
            year = int(float(year_text))
        except ValueError:
            continue
        if year not in YEARS:
            continue
        sha = norm(first(row, ("sha256", "apk_id", "sample_id")))
        if not sha:
            malformed.append(index); continue
        if reference_by_sha and sha not in reference_by_sha:
            continue
        if args.status:
            status_row = status_by_sha.get(sha)
            if status_row is None:
                status_missing.append(sha)
                continue
            extraction_status = norm(status_row.get("status"))
            eligible_text = status_row.get("temporal_eval_eligible", "")
            extraction_ok = extraction_status in {"ok", "success"}
            eligible = truthy(eligible_text) if "temporal_eval_eligible" in status_row else True
            if not extraction_ok or not eligible:
                status_excluded.append({
                    "sha256": sha,
                    "year": year,
                    "status": extraction_status,
                    "temporal_eval_eligible": str(eligible_text),
                    "error": str(status_row.get("error", "")),
                })
                continue
        grouped[sha].append(row)

    duplicate_detail = {}
    conflicts = []
    canonical = []
    for sha in sorted(grouped):
        rows = grouped[sha]
        if len(rows) > 1:
            duplicate_detail[sha] = len(rows)
        signatures = {
            (
                first(row, ("absolute_path", "apk_path", "path", "file_path")),
                first(row, ("metadata_year", "year")),
                normalize_label(first(row, ("metadata_label", "label"))),
                str(row.get("status", "")).strip(),
            )
            for row in rows
        }
        if len(signatures) != 1:
            conflicts.append(sha); continue
        row = rows[0]
        resolved_path = remap_path(
            first(row, ("absolute_path", "apk_path", "path", "file_path")),
            args.storage_root,
        )
        canonical.append({
            "sha256": sha,
            "absolute_path": str(resolved_path.resolve()),
            "source_id": first(row, ("source_id",)) or "future_2018_2021",
            "lineage_id": first(row, ("lineage_id",)) or sha,
            "metadata_year": str(int(float(first(row, ("metadata_year", "year"))))),
            "metadata_label": normalize_label(first(row, ("metadata_label", "label"))),
        })

    canonical.sort(key=lambda row: (int(row["metadata_year"]), row["sha256"]))
    ids = [row["sha256"] for row in canonical]
    year_counts = Counter(int(row["metadata_year"]) for row in canonical)
    label_counts = Counter(row["metadata_label"] for row in canonical)
    missing_paths = [row["sha256"] for row in canonical if not Path(row["absolute_path"]).is_file()]
    bad_labels = [row["sha256"] for row in canonical if row["metadata_label"] not in {"benign", "malware"}]

    reference_check = "not_requested"
    reference_metadata_conflicts = []
    if reference_by_sha:
        for row in canonical:
            reference_row = reference_by_sha[row["sha256"]]
            reference_year = int(float(first(reference_row, ("year", "metadata_year"))))
            reference_label = normalize_label(first(reference_row, ("label", "metadata_label")))
            if reference_year != int(row["metadata_year"]) or reference_label != row["metadata_label"]:
                reference_metadata_conflicts.append(row["sha256"])
        reference_year_counts = Counter(
            int(float(first(row, ("year", "metadata_year")))) for row in reference_rows
        )
        reference_check = (
            len(reference_by_sha) == EXPECTED
            and set(reference_by_sha) == set(ids)
            and dict(reference_year_counts) == EXPECTED_BY_YEAR
            and not reference_metadata_conflicts
        )

    checks = {
        "future_rows_exact_1989": len(canonical) == EXPECTED,
        "future_ids_unique_1989": len(set(ids)) == EXPECTED and "" not in ids,
        "future_years_exact": set(year_counts) == set(YEARS),
        "future_counts_by_year_exact": dict(year_counts) == EXPECTED_BY_YEAR,
        "duplicate_payload_conflicts_eq_0": len(conflicts) == 0,
        "malformed_future_ids_eq_0": len(malformed) == 0,
        "valid_binary_labels": len(bad_labels) == 0,
        "all_apk_paths_exist": len(missing_paths) == 0,
        "frozen_reference_required_and_matches": reference_check is True,
        "reference_metadata_conflicts_eq_0": len(reference_metadata_conflicts) == 0,
        "status_manifest_valid_when_present": args.status is None or args.status.is_file(),
        "status_ids_unique": len(duplicate_status_ids) == 0,
        "future_status_rows_missing_eq_0": len(status_missing) == 0,
        "future_status_exclusions_eq_0_for_reference_cohort": len(status_excluded) == 0,
    }
    passed = all(checks.values())
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "passed": passed,
        "checks": checks,
        "expected": EXPECTED,
        "year_counts": dict(sorted(year_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "lineage_input_file": str(input_path.resolve()),
        "lineage_input_sha256": digest(input_path),
        "reference_file": reference_path,
        "reference_check": reference_check,
        "reference_metadata_conflicts": reference_metadata_conflicts,
        "status_file": str(args.status.resolve()) if args.status else None,
        "status_sha256": digest(args.status) if args.status else None,
        "status_excluded_future_rows": status_excluded,
        "status_missing_future_ids": status_missing,
        "duplicate_status_ids": sorted(set(duplicate_status_ids)),
        "identical_duplicate_rows_collapsed": duplicate_detail,
        "conflicting_duplicate_ids": conflicts,
        "malformed_rows": malformed,
        "missing_apk_paths": missing_paths,
        "bad_label_ids": bad_labels,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("future lineage gate failed; extraction is blocked")
    write_csv(args.output, canonical)
    print("FUTURE_LINEAGE_2018_2021_GATE_PASS")


if __name__ == "__main__":
    main()
