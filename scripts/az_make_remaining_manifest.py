#!/usr/bin/env python3
"""Build a Windows-only missing-APK CSV from the frozen AndroZoo 15k manifest.

Run once on the Linux server before Windows download. Reads original selected
samples and local completed APKs, writes a separate remaining_for_windows.csv;
never modifies the 15k manifest or deletes APKs. Optional --verify-existing
recomputes SHA256 instead of trusting expected file sizes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "datasets/android/AndroZoo/AZ_2017_2022_Random15K"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--verify-existing", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    source = root / "manifests" / "selected_all.csv"
    output = root / "manifests" / "remaining_for_windows.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not {"year", "label", "sha256", "apk_size"}.issubset(fields):
            raise ValueError("Missing manifest fields")
        rows = list(reader)
    if len(rows) != 15000 or len({r["sha256"] for r in rows}) != 15000:
        raise ValueError("Frozen selected_all.csv is not a unique 15k manifest")
    written = Counter()
    bytes_needed = Counter()
    missing = []
    for row in rows:
        year, sha = row["year"], row["sha256"]
        label = "benign" if row["label"] == "0" else "malware"
        path = root / "apks" / ("year" + year) / label / (sha + ".apk")
        expected_size = int(row.get("apk_size") or 0)
        is_present = path.is_file() and (expected_size <= 0 or path.stat().st_size == expected_size)
        if is_present and args.verify_existing:
            is_present = sha256_file(path) == sha
        if is_present:
            written[(year, "already_present")] += 1
        else:
            missing.append(row)
            written[(year, "remaining")] += 1
            bytes_needed[year] += expected_size
    temp = output.with_name(output.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, output)
    print(f"FROZEN_ROWS={len(rows)} REMAINING={len(missing)} SOURCE_UNCHANGED={source}")
    for year in range(2017, 2023):
        year = str(year)
        print(f"YEAR={year} EXISTING={written[year, 'already_present']} "
              f"REMAINING={written[year, 'remaining']} GB={bytes_needed[year]/1e9:.2f}")
    print(f"REMAINING_TOTAL_GB={sum(bytes_needed.values())/1e9:.2f}")
    print(f"OUTPUT={output}")
    print("WARNING: size-only existing-file check" if not args.verify_existing else "Existing APKs SHA256 verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
