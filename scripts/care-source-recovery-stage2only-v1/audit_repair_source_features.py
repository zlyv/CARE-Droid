#!/usr/bin/env python3
"""Audit and, only when safe, repair duplicate compact source-feature rows.

The repair is deliberately narrow: only byte-equivalent (after CSV parsing)
duplicate rows for an in-lineage SHA may be collapsed.  Missing samples,
outside-lineage samples, conflicting duplicate payloads, failures, or malformed
IDs are hard failures and are never hidden by a generic drop_duplicates call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def canonical_id(row: dict[str, str]) -> str:
    return str(row.get("sha256") or row.get("sample_id") or "").strip().lower()


def normalised_payload(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def safe_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_rewrite(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    backup = path.with_name(path.stem + ".pre_integrity_repair" + path.suffix)
    if backup.exists():
        raise RuntimeError(
            f"refusing to overwrite existing recovery backup: {backup}; "
            "inspect or rename it first"
        )
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copy2(path, backup)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return backup


def write_duplicate_rows(
    path: Path,
    fields: list[str],
    groups: dict[str, list[tuple[int, dict[str, str]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["canonical_apk_id", "source_csv_row", "duplicate_occurrence", "canonical_kept"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra + fields, extrasaction="ignore")
        writer.writeheader()
        for apk_id in sorted(groups):
            entries = groups[apk_id]
            if len(entries) <= 1:
                continue
            for occurrence, (row_number, row) in enumerate(entries, 1):
                output = dict(row)
                output.update({
                    "canonical_apk_id": apk_id,
                    "source_csv_row": row_number,
                    "duplicate_occurrence": occurrence,
                    "canonical_kept": int(occurrence == 1),
                })
                writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=799)
    parser.add_argument("--repair-identical-duplicates", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "source_lineage_integrity_audit.txt"
    duplicate_path = args.output / "source_duplicate_rows.csv"
    gate_path = args.output / "source_lineage_integrity_gate.json"

    lineage_fields, lineage_rows = read_csv(args.lineage)
    lineage_ids = [canonical_id(row) for row in lineage_rows]
    lineage_counts = Counter(lineage_ids)
    malformed_lineage = [index + 2 for index, value in enumerate(lineage_ids) if not value]
    lineage_duplicates = sorted(key for key, count in lineage_counts.items() if key and count > 1)
    lineage_set = {value for value in lineage_ids if value}

    feature_fields, feature_rows = read_csv(args.features)
    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    malformed_feature_rows = []
    for index, row in enumerate(feature_rows, 2):
        apk_id = canonical_id(row)
        if not apk_id:
            malformed_feature_rows.append(index)
        groups[apk_id].append((index, row))
    duplicate_groups = {key: values for key, values in groups.items() if key and len(values) > 1}
    write_duplicate_rows(duplicate_path, feature_fields, duplicate_groups)

    conflicting_duplicates = []
    for apk_id, entries in duplicate_groups.items():
        payloads = {normalised_payload(row, feature_fields) for _, row in entries}
        if len(payloads) != 1:
            conflicting_duplicates.append(apk_id)

    feature_set = {key for key in groups if key}
    missing = sorted(lineage_set - feature_set)
    outside = sorted(feature_set - lineage_set)
    failure_rows: list[dict[str, str]] = []
    if args.failures and args.failures.exists() and args.failures.stat().st_size:
        _, failure_rows = read_csv(args.failures)
    successful_failure_ids = sorted({canonical_id(row) for row in failure_rows if canonical_id(row)})

    preconditions = (
        len(lineage_rows) == args.expected
        and len(lineage_set) == args.expected
        and not malformed_lineage
        and not lineage_duplicates
        and not malformed_feature_rows
        and not missing
        and not outside
        and not conflicting_duplicates
        and not successful_failure_ids
    )
    backup = None
    repaired = False
    if duplicate_groups and args.repair_identical_duplicates and preconditions:
        canonical_rows = [groups[apk_id][0][1] for apk_id in lineage_ids]
        backup = atomic_rewrite(args.features, feature_fields, canonical_rows)
        repaired = True
        feature_fields, feature_rows = read_csv(args.features)
        groups = defaultdict(list)
        for index, row in enumerate(feature_rows, 2):
            groups[canonical_id(row)].append((index, row))

    final_ids = [canonical_id(row) for row in feature_rows]
    final_counts = Counter(final_ids)
    final_set = {value for value in final_ids if value}
    final_duplicates = sorted(key for key, count in final_counts.items() if key and count > 1)
    final_missing = sorted(lineage_set - final_set)
    final_outside = sorted(final_set - lineage_set)
    assertions = {
        "unique_feature_apks_eq_799": len(final_set) == args.expected,
        "eligible_apk_count_eq_799": len(lineage_set) == args.expected,
        "missing_lineage_apks_eq_0": len(final_missing) == 0,
        "outside_lineage_apks_eq_0": len(final_outside) == 0,
        "duplicate_apks_eq_0": len(final_duplicates) == 0,
        "feature_rows_eq_799": len(feature_rows) == args.expected,
        "all_status_ok": all(str(row.get("status", "")).strip() == "OK" for row in feature_rows),
        "failure_rows_eq_0": len(failure_rows) == 0,
    }
    passed = all(assertions.values())
    totals = {
        "mapped_api_node_total": sum(safe_int(row.get("mapped_api_node_count")) for row in feature_rows),
        "external_node_total": sum(safe_int(row.get("external_node_count")) for row in feature_rows),
        "sensitive_edge_total": sum(safe_int(row.get("sensitive_edge_count")) for row in feature_rows),
    }

    audit_json_path = args.features.parent / "feature_extraction_audit.json"
    if passed and audit_json_path.exists():
        try:
            audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit = {}
        audit.update({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_row_count": len(feature_rows),
            "failure_row_count": len(failure_rows),
            "status_counts": dict(Counter(row.get("status", "") for row in feature_rows + failure_rows)),
            **totals,
            "source_lineage_integrity_gate": str(gate_path.resolve()),
            "totals_recomputed_after_integrity_repair": True,
        })
        audit_json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gate = {
        "passed": passed,
        "expected": args.expected,
        "raw_feature_row_count_before_repair": sum(len(values) for values in groups.values())
        if not repaired else args.expected + sum(len(values) - 1 for values in duplicate_groups.values()),
        "unique_feature_apks_before_repair": len(feature_set),
        "duplicate_apk_count_before_repair": len(duplicate_groups),
        "duplicate_extra_row_count_before_repair": sum(len(v) - 1 for v in duplicate_groups.values()),
        "conflicting_duplicate_apks": conflicting_duplicates,
        "repair_performed": repaired,
        "backup": str(backup.resolve()) if backup else None,
        "assertions": assertions,
        "totals": totals,
        "artifact_hashes": {
            str(args.lineage.resolve()): sha256_file(args.lineage),
            str(args.features.resolve()): sha256_file(args.features),
        },
    }
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "CARE-Droid source lineage integrity audit", "",
        "Root cause:",
        "  Two extractor processes read the same stale completed-ID snapshot and appended overlapping pending APKs.",
        "  The old append-resume implementation had no exclusive writer lock.", "",
        f"lineage_rows: {len(lineage_rows)}",
        f"lineage_unique_apks: {len(lineage_set)}",
        f"raw_feature_rows_before_repair: {len(feature_rows) if not repaired else args.expected + sum(len(v) - 1 for v in duplicate_groups.values())}",
        f"unique_feature_apks_before_repair: {len(feature_set)}",
        f"duplicate_apks_before_repair: {len(duplicate_groups)}",
        f"duplicate_extra_rows_before_repair: {sum(len(v) - 1 for v in duplicate_groups.values())}",
        f"duplicate_counts: {dict(sorted((key, len(value)) for key, value in duplicate_groups.items()))}",
        f"conflicting_duplicate_apks: {conflicting_duplicates}",
        f"missing_lineage_apks_before_repair: {missing}",
        f"outside_lineage_apks_before_repair: {outside}",
        f"resume_state_note: completed IDs were represented as a set, but concurrent writers used stale snapshots; CSV row multiplicity therefore diverged from resume state.",
        f"repair_performed: {repaired}",
        f"backup: {str(backup) if backup else 'none'}", "",
        "Final hard assertions:",
    ]
    lines.extend(f"  {name}: {str(value).lower()}" for name, value in assertions.items())
    lines.extend([
        "", "Recomputed totals from the final canonical 799 rows:",
        f"  mapped_api_node_total: {totals['mapped_api_node_total']}",
        f"  external_node_total: {totals['external_node_total']}",
        f"  sensitive_edge_total: {totals['sensitive_edge_total']}",
        "", f"gate_passed: {str(passed).lower()}",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit("source lineage integrity gate failed; no downstream stage may start")


if __name__ == "__main__":
    main()
