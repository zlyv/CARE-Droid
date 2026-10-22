from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tasa_droid.config import ProjectConfig, resolve_data_path


HASH_CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class PendingApk:
    source_id: str
    source_purpose: str
    declared_pack_state: str
    apk_root: str
    absolute_path: str
    relative_path: str
    size_bytes: int
    modified_time_utc: str
    inferred_year: int | None
    inferred_year_origin: str
    inferred_label: str | None


@dataclass(frozen=True)
class InventoryRecord:
    source_id: str
    source_purpose: str
    declared_pack_state: str
    apk_root: str
    absolute_path: str
    relative_path: str
    size_bytes: int
    modified_time_utc: str
    sha256: str
    year: int | None
    year_origin: str
    label: str | None
    label_origin: str
    first_seen_utc: str | None
    historical_label_joined: bool
    package_name: str | None
    certificate_sha256: str | None
    cluster_id: str | None
    parse_status: str


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_year(relative_path: str, pattern: re.Pattern[str]) -> int | None:
    matches = sorted({int(value) for value in pattern.findall(relative_path)})
    return matches[0] if len(matches) == 1 else None


def _infer_year_with_root(
    relative_path: str,
    apk_root_name: str,
    pattern: re.Pattern[str],
) -> tuple[int | None, str]:
    relative_matches = sorted({int(value) for value in pattern.findall(relative_path)})
    if len(relative_matches) == 1:
        return relative_matches[0], "relative_path_regex"

    root_matches = sorted({int(value) for value in pattern.findall(apk_root_name)})
    if len(root_matches) == 1:
        return root_matches[0], "apk_root_name_regex"

    return None, "unresolved"


def _infer_label(relative_path: str, rules: dict[str, tuple[re.Pattern[str], ...]]) -> str | None:
    normalized = relative_path.replace("\\", "/").lower()
    matches = [
        label
        for label, patterns in rules.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]
    return matches[0] if len(matches) == 1 else None


def _discover(config: ProjectConfig) -> tuple[list[PendingApk], list[dict[str, str]]]:
    year_pattern = re.compile(config.scan.year_regex)
    label_patterns = {
        label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for label, patterns in config.scan.label_rules.items()
    }
    records: list[PendingApk] = []
    missing_roots: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for source in config.sources:
        if not source.enabled:
            continue
        for configured_root in source.apk_roots:
            apk_root = resolve_data_path(config, configured_root)
            if not apk_root.is_dir():
                missing_roots.append(
                    {
                        "source_id": source.source_id,
                        "configured_root": configured_root,
                        "resolved_root": str(apk_root),
                    }
                )
                continue
            for path in sorted(apk_root.rglob("*"), key=lambda value: value.as_posix()):
                if not path.is_file() or path.suffix.lower() not in config.scan.extensions:
                    continue
                resolved = str(path.resolve(strict=True))
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                relative = path.relative_to(apk_root).as_posix()
                stat = path.stat()
                inferred_year, inferred_year_origin = _infer_year_with_root(
                    relative,
                    apk_root.name,
                    year_pattern,
                )
                records.append(
                    PendingApk(
                        source_id=source.source_id,
                        source_purpose=source.purpose,
                        declared_pack_state=source.declared_pack_state,
                        apk_root=str(apk_root),
                        absolute_path=resolved,
                        relative_path=relative,
                        size_bytes=stat.st_size,
                        modified_time_utc=_utc_iso(stat.st_mtime),
                        inferred_year=inferred_year,
                        inferred_year_origin=inferred_year_origin,
                        inferred_label=_infer_label(relative, label_patterns),
                    )
                )
    records.sort(key=lambda value: (value.source_id, value.relative_path, value.absolute_path))
    return records, missing_roots


def _materialize_record(item: PendingApk, hash_mode: str) -> InventoryRecord:
    digest = _sha256(Path(item.absolute_path)) if hash_mode == "sha256" else ""
    return InventoryRecord(
        source_id=item.source_id,
        source_purpose=item.source_purpose,
        declared_pack_state=item.declared_pack_state,
        apk_root=item.apk_root,
        absolute_path=item.absolute_path,
        relative_path=item.relative_path,
        size_bytes=item.size_bytes,
        modified_time_utc=item.modified_time_utc,
        sha256=digest,
        year=item.inferred_year,
        year_origin=item.inferred_year_origin,
        label=item.inferred_label,
        label_origin="relative_path_regex" if item.inferred_label is not None else "unresolved",
        first_seen_utc=None,
        historical_label_joined=False,
        package_name=None,
        certificate_sha256=None,
        cluster_id=None,
        parse_status="NOT_PARSED",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return _sha256(path)


def _duplicate_rows(records: list[InventoryRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            groups[record.sha256].append(record)
    output: list[dict[str, Any]] = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        output.append(
            {
                "sha256": digest,
                "apk_count": len(members),
                "source_count": len({member.source_id for member in members}),
                "sources": json.dumps(sorted({member.source_id for member in members}), ensure_ascii=False),
                "paths": json.dumps([member.absolute_path for member in members], ensure_ascii=False),
            }
        )
    return output


def _year_label_counts(records: list[InventoryRecord]) -> list[dict[str, Any]]:
    counts = Counter(
        (str(record.year) if record.year is not None else "UNRESOLVED", record.label or "UNRESOLVED")
        for record in records
    )
    return [
        {"year": year, "label": label, "apk_count": count}
        for (year, label), count in sorted(counts.items())
    ]


def _planning_half_width(cluster_count: int | None) -> float | None:
    if not cluster_count:
        return None
    return 1.96 * math.sqrt(0.25 / cluster_count)


def _build_audit(
    config: ProjectConfig,
    records: list[InventoryRecord],
    missing_roots: list[dict[str, str]],
    duplicate_rows: list[dict[str, Any]],
    hash_mode: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    source_counts = Counter(record.source_id for record in records)
    unresolved_year = sum(record.year is None for record in records)
    unresolved_label = sum(record.label is None for record in records)
    cross_source_duplicate_groups = sum(row["source_count"] > 1 for row in duplicate_rows)

    blocking_reasons: list[str] = []
    if missing_roots:
        blocking_reasons.append("one_or_more_configured_apk_roots_are_missing")
    if not records:
        blocking_reasons.append("no_apk_files_discovered")
    if hash_mode != "sha256":
        blocking_reasons.append("sha256_inventory_not_generated")
    if unresolved_year:
        blocking_reasons.append("one_or_more_apks_have_unresolved_year")
    if unresolved_label:
        blocking_reasons.append("historical_labels_not_joined_or_unresolved")
    blocking_reasons.extend(
        [
            "audited_first_seen_utc_not_joined",
            "package_and_certificate_version_lines_not_built",
            "near_duplicate_clusters_not_built",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ]
    )

    source_metadata = {
        source.source_id: {
            "enabled": source.enabled,
            "purpose": source.purpose,
            "declared_pack_state": source.declared_pack_state,
            "pack_state_origin": "dataset_source_declaration_not_apk_measurement",
            "apk_count": source_counts.get(source.source_id, 0),
        }
        for source in config.sources
    }

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "phase": "P0_INITIAL_INVENTORY",
        "p0_status": "NOT_AUDITED",
        "formal_training_enabled": False,
        "confirmatory_eligible_years": [],
        "dataset_root": str(config.dataset_root),
        "config_path": str(config.path),
        "hash_mode": hash_mode,
        "inventory_sha256": inventory_sha256,
        "apk_count": len(records),
        "unresolved_year_count": unresolved_year,
        "unresolved_label_count": unresolved_label,
        "exact_duplicate_group_count": len(duplicate_rows),
        "cross_source_exact_duplicate_group_count": cross_source_duplicate_groups,
        "missing_apk_roots": missing_roots,
        "sources": source_metadata,
        "counts_by_year_and_label": _year_label_counts(records),
        "planning_half_width": {
            "value": _planning_half_width(None),
            "status": "NOT_COMPUTED_UNTIL_INDEPENDENT_CLUSTERS_EXIST",
        },
        "blocking_reasons": sorted(set(blocking_reasons)),
        "notes": [
            "declared_pack_state records dataset-level knowledge and is not an APK packer detection result",
            "relative-path year/label inference is provisional and must be replaced or verified by audited metadata",
            "exact SHA256 duplicates are not a substitute for package/certificate and near-duplicate clusters",
        ],
    }


def run_p0_scan(
    config: ProjectConfig,
    output_dir: str | Path,
    hash_mode: str = "sha256",
    workers: int = 4,
) -> dict[str, Any]:
    if hash_mode not in {"sha256", "none"}:
        raise ValueError("hash_mode 只允许 sha256 或 none")
    if workers < 1:
        raise ValueError("workers 必须 >= 1")

    output = Path(output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    pending, missing_roots = _discover(config)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda item: _materialize_record(item, hash_mode), pending))
    records.sort(key=lambda value: (value.source_id, value.relative_path, value.absolute_path))

    inventory_path = output / "inventory.csv"
    inventory_rows = [asdict(record) for record in records]
    inventory_fields = list(InventoryRecord.__dataclass_fields__.keys())
    _write_csv(inventory_path, inventory_rows, inventory_fields)
    inventory_sha256 = _file_sha256(inventory_path)

    unresolved_rows = [
        row
        for row in inventory_rows
        if row["year"] is None or row["label"] is None
    ]
    _write_csv(output / "unresolved.csv", unresolved_rows, inventory_fields)

    duplicates = _duplicate_rows(records)
    _write_csv(
        output / "exact_duplicates.csv",
        duplicates,
        ["sha256", "apk_count", "source_count", "sources", "paths"],
    )

    (output / "data_manifest.sha256").write_text(
        f"{inventory_sha256}  inventory.csv\n",
        encoding="ascii",
    )
    audit = _build_audit(
        config=config,
        records=records,
        missing_roots=missing_roots,
        duplicate_rows=duplicates,
        hash_mode=hash_mode,
        inventory_sha256=inventory_sha256,
    )
    (output / "p0_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit
