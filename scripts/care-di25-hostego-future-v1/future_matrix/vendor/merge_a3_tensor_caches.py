#!/usr/bin/env python3
"""Merge A3 JSONL-gzip caches with SHA256-level duplicate auditing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path.expanduser().resolve(strict=True), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}") from exc
            if row.get("status") != "OK":
                continue
            sha = str(row.get("sha256", "")).strip().lower()
            if len(sha) != 64:
                raise SystemExit(f"invalid cache SHA256 at {path}:{line_number}: {sha!r}")
            row["sha256"] = sha
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", required=True, type=Path)
    parser.add_argument("--new-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    old_path = args.old_cache.expanduser().resolve(strict=True)
    new_path = args.new_cache.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_cache = output / "a3_mvp_tensor_cache.jsonl.gz"

    old_rows = read_rows(old_path)
    new_rows = read_rows(new_path)
    selected: dict[str, dict] = {}
    source_by_sha: dict[str, str] = {}
    duplicate_rows = []
    status_counts = Counter()

    for source_name, rows in (("old", old_rows), ("new", new_rows)):
        for row in rows:
            sha = row["sha256"]
            status_counts[f"{source_name}_OK"] += 1
            if sha in selected:
                duplicate_rows.append(
                    {
                        "sha256": sha,
                        "first_source": source_by_sha[sha],
                        "duplicate_source": source_name,
                        "reason": "exact_sha256_duplicate_keep_first",
                    }
                )
                continue
            selected[sha] = row
            source_by_sha[sha] = source_name

    with gzip.open(output_cache, "wt", encoding="utf-8", newline="") as handle:
        for sha in sorted(selected):
            handle.write(json.dumps(selected[sha], ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    duplicate_path = output / "cache_duplicate_exclusions.csv"
    with duplicate_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sha256,first_source,duplicate_source,reason\n")
        for row in duplicate_rows:
            handle.write(
                f"{row['sha256']},{row['first_source']},{row['duplicate_source']},{row['reason']}\n"
            )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "MERGED_A3_TENSOR_CACHE",
        "old_cache": str(old_path),
        "old_cache_sha256": sha256_file(old_path),
        "new_cache": str(new_path),
        "new_cache_sha256": sha256_file(new_path),
        "old_ok_count": len(old_rows),
        "new_ok_count": len(new_rows),
        "merged_unique_sha256_count": len(selected),
        "exact_duplicate_count": len(duplicate_rows),
        "outputs": {
            "cache": str(output_cache),
            "duplicate_exclusions": str(duplicate_path),
        },
        "notes": [
            "Only status=OK cache rows are merged.",
            "Exact SHA256 duplicates keep the first row and are reported.",
            "Near-duplicate and package/certificate lineage relations are not resolved here.",
            "The cache contains bounded tensors, not raw graphs or DEX data.",
        ],
    }
    (output / "merged_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
